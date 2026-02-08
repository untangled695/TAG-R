"""
TAG-R OOD Shootout Experiment

Tests whether topology-aware routing generalizes better than content-based routing
when faced with completely new semantic domains.

Setup:
- Train: Biological entities (animals, plants)
- Test: Artifacts (tools, vehicles, structures)

Hypothesis:
- MLP Router: Memorizes "dog -> Hyperbolic" but fails on "hammer" (unseen content)
- TAG-R Router: Recognizes tree structure in both domains (topology generalizes)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.dirname(__file__))

from dataset import WordNetHierarchyDataset
from models import HierarchyPredictor

try:
    from nltk.corpus import wordnet as wn
except ImportError:
    print("Installing NLTK...")
    import nltk
    nltk.download('wordnet')
    from nltk.corpus import wordnet as wn


# --- 1. OOD Dataset Implementation ---

class WordNetOODDataset(Dataset):
    """
    Splits WordNet into DISJOINT semantic domains for OOD testing.

    Train Domain: Biological entities (living_thing.n.01)
    Test Domain: Artifacts (artifact.n.01)

    Key: Both domains have hierarchical structure, but completely different content.
    """

    def __init__(self, split='train', max_samples=5000):
        self.samples = []
        self.synset_to_idx = {}

        # Domain roots (semantically disjoint)
        # Use specific subtrees that don't overlap
        bio_roots = {'animal.n.01', 'plant.n.02'}  # Living organisms
        artifact_roots = {'instrumentality.n.03', 'structure.n.01'}  # Human-made objects

        print(f"\nBuilding {split.upper()} dataset...")
        all_synsets = list(wn.all_synsets('n'))

        # Build domain-specific synsets (STRICT SEPARATION)
        bio_synsets = set()
        artifact_synsets = set()

        for s in all_synsets:
            paths = s.hypernym_paths()
            for path in paths:
                path_names = {node.name() for node in path}
                # Strictly classify as Bio XOR Artifact (not both)
                if not bio_roots.isdisjoint(path_names) and artifact_roots.isdisjoint(path_names):
                    bio_synsets.add(s.name())
                    break
                elif not artifact_roots.isdisjoint(path_names) and bio_roots.isdisjoint(path_names):
                    artifact_synsets.add(s.name())
                    break

        # Verify disjoint
        assert bio_synsets.isdisjoint(artifact_synsets), "Domains must be completely disjoint!"

        print(f"  Bio synsets: {len(bio_synsets)}")
        print(f"  Artifact synsets: {len(artifact_synsets)}")

        # Build global vocabulary (union of both domains)
        valid_synsets = bio_synsets | artifact_synsets
        sorted_synsets = sorted(list(valid_synsets))
        self.synset_to_idx = {name: i for i, name in enumerate(sorted_synsets)}
        self.idx_to_synset = {i: wn.synset(name) for name, i in self.synset_to_idx.items()}
        self.vocab_size = len(self.synset_to_idx)

        # Filter by domain for this split
        target_synsets = bio_synsets if split == 'train' else artifact_synsets
        target_roots = bio_roots if split == 'train' else artifact_roots

        # Only iterate over target domain synsets
        for s_name in target_synsets:
            s = wn.synset(s_name)
            paths = s.hypernym_paths()

            # Find path in target domain
            target_path = None
            for path in paths:
                path_names = {node.name() for node in path}
                if not target_roots.isdisjoint(path_names):
                    target_path = path
                    break

            if target_path and 2 <= len(target_path) <= 8:
                leaf_idx = self.synset_to_idx[s.name()]
                ancestors = []
                for p in target_path[:-1]:
                    if p.name() in self.synset_to_idx:
                        ancestors.append(self.synset_to_idx[p.name()])

                if ancestors:
                    self.samples.append({'leaf': leaf_idx, 'ancestors': ancestors})

        # Subsample
        if len(self.samples) > max_samples:
            np.random.seed(42)
            np.random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        print(f"  Domain: {target_roots}")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Shared vocab: {self.vocab_size}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_ood(batch):
    """Collate function for OOD dataset."""
    leaves = torch.tensor([b['leaf'] for b in batch], dtype=torch.long)
    max_anc = max(len(b['ancestors']) for b in batch)
    ancestors = torch.full((len(batch), max_anc), -1, dtype=torch.long)
    for i, b in enumerate(batch):
        ancestors[i, :len(b['ancestors'])] = torch.tensor(b['ancestors'])
    return leaves, ancestors


# --- 2. Router Implementations ---

class MLPRouter(nn.Module):
    """
    Baseline Router (CAT/MoS style).
    Routes based on content embedding only - no topology awareness.
    """

    def __init__(self, d_model, n_geometries=3, temperature=1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, n_geometries)
        )
        self.temperature = temperature

    def forward(self, H):
        """
        H: [batch, seq, d_model]

        Returns:
            weights: [batch, seq, n_geometries] (softmax probabilities)
            logits: [batch, seq, n_geometries] (raw scores)
        """
        logits = self.net(H)  # [batch, seq, n_geometries]
        weights = torch.softmax(logits / self.temperature, dim=-1)
        return weights, logits


class TopologyRouter(nn.Module):
    """
    TAG-R: Topology-Aware Geometric Router (Ours).
    Routes based on LOCAL GEOMETRIC STRUCTURE rather than content.

    Features:
    - Tree-ness: Variance of k-NN distances (trees have regular branching)
    - Density: Mean distance to neighbors (clusters vs sparse)
    - Outlier ratio: Max/mean distance (hierarchies have parent-child patterns)
    """

    def __init__(self, d_model, n_geometries=3, n_neighbors=8):
        super().__init__()

        # Content-based fallback (for when batch is small)
        self.content_router = nn.Linear(d_model, n_geometries)

        # Topology-based router (the innovation)
        self.topo_head = nn.Sequential(
            nn.Linear(3, 32),  # 3 topological features
            nn.ReLU(),
            nn.Linear(32, n_geometries)
        )

        # Learned mixing coefficient (starts at 0.5)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.n_neighbors = n_neighbors

    def compute_topology_features(self, H):
        """
        Compute local topology features from batch.

        Args:
            H: [batch, seq, d_model]

        Returns:
            features: [batch, seq, 3] (tree-ness, density, outlier_ratio)
        """
        batch, seq, d = H.shape

        # Normalize for cosine similarity
        H_norm = H / (H.norm(dim=-1, keepdim=True) + 1e-8)

        # For each sequence position, compute similarity to other batch elements
        # This simulates "retrieving similar concepts from memory"
        H_flat = H_norm.reshape(-1, d)  # [batch*seq, d]

        # Pairwise cosine similarity
        sim = H_flat @ H_flat.T  # [batch*seq, batch*seq]
        dists = 1 - sim  # Distance matrix

        # k-NN features
        k = min(self.n_neighbors, H_flat.size(0) - 1)
        if k < 2:
            # Not enough neighbors - return zeros
            return torch.zeros(batch, seq, 3, device=H.device)

        knn_dists, _ = torch.topk(dists, k=k+1, dim=-1, largest=False)
        knn_dists = knn_dists[:, 1:]  # Remove self (distance=0)

        # Feature 1: Tree-ness (variance of neighbor distances)
        tree_ness = knn_dists.std(dim=-1, keepdim=True)

        # Feature 2: Density (mean distance)
        density = knn_dists.mean(dim=-1, keepdim=True)

        # Feature 3: Outlier ratio (max/mean - hierarchies have parent far from siblings)
        outlier = knn_dists.max(dim=-1).values.unsqueeze(-1) / (density.squeeze(-1).unsqueeze(-1) + 1e-8)

        feats = torch.cat([tree_ness, density, outlier], dim=-1)  # [batch*seq, 3]
        feats = feats.reshape(batch, seq, 3)

        return feats

    def forward(self, H):
        """
        H: [batch, seq, d_model]

        Returns:
            weights: [batch, seq, n_geometries] (softmax probabilities)
            logits: [batch, seq, n_geometries] (raw scores)
        """

        # Content-based routing (baseline component)
        content_logits = self.content_router(H)

        # Topology-based routing (our innovation)
        topo_feats = self.compute_topology_features(H)
        topo_logits = self.topo_head(topo_feats)

        # Mix both signals (learned combination)
        mix = torch.sigmoid(self.alpha)
        logits = mix * content_logits + (1 - mix) * topo_logits
        weights = torch.softmax(logits, dim=-1)

        return weights, logits


# --- 3. Shootout Model Wrapper ---

class OODShootoutModel(nn.Module):
    """Model for OOD generalization testing."""

    def __init__(self, vocab_size, d_model, router_type='mlp', n_heads=4):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.router_type = router_type

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0, std=1e-3)

        # Import attention components
        from aga.models.aga_layer import AGALayer

        # Create AGA layer (uses CurvatureAwareRouter by default)
        self.aga = AGALayer(
            d_model=d_model,
            n_heads=n_heads,
            hyperbolic_curvature=-1.0,
        )

        # Replace router with selected type
        if router_type == 'mlp':
            self.aga.router = MLPRouter(d_model, n_geometries=3)
        elif router_type == 'tagr':
            self.aga.router = TopologyRouter(d_model, n_geometries=3)
        else:
            raise ValueError(f"Unknown router type: {router_type}")

        # Prediction head
        self.predictor = nn.Linear(d_model, vocab_size)

        # Store for diagnostics
        self.last_router_info = None

    def forward(self, x):
        """
        Args:
            x: [batch] leaf indices

        Returns:
            logits: [batch, vocab_size]
        """
        emb = self.embedding(x).unsqueeze(1)  # [batch, 1, d_model]

        # AGA forward (always returns tuple: output, router_info)
        out, router_info = self.aga(emb, return_router_info=False)

        # Prediction
        logits = self.predictor(out.squeeze(1))

        return logits


# --- 4. Training and Evaluation ---

def train_and_eval_ood(router_type, train_loader, test_loader, vocab_size, device='cpu'):
    """
    Train model on Bio domain, evaluate on Artifact domain.

    Returns:
        train_recall, test_recall (OOD performance)
    """
    print(f"\n{'='*60}")
    print(f"Training {router_type.upper()} Router")
    print(f"{'='*60}")

    # Create model
    model = OODShootoutModel(
        vocab_size=vocab_size,
        d_model=32,
        router_type=router_type,
    )
    model.to(device)

    # Split optimization (standard stability protocol)
    hyperbolic_params = []
    euclidean_params = []

    for name, param in model.named_parameters():
        if 'embedding' in name:
            hyperbolic_params.append(param)
        else:
            euclidean_params.append(param)

    opt_hyp = optim.SGD(hyperbolic_params, lr=1e-2)
    opt_euc = optim.AdamW(euclidean_params, lr=1e-3)

    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Split: {len(hyperbolic_params)} hyperbolic, {len(euclidean_params)} euclidean")

    # Train on Bio domain
    print(f"\nTraining on BIOLOGICAL domain...")
    model.train()

    for epoch in range(10):
        total_loss = 0
        num_batches = 0

        for leaves, ancestors in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            leaves = leaves.to(device)
            ancestors = ancestors.to(device)

            opt_hyp.zero_grad()
            opt_euc.zero_grad()

            # Forward
            logits = model(leaves)

            # Multi-label targets
            targets = torch.zeros_like(logits)
            for i in range(ancestors.shape[0]):
                valid = ancestors[i][ancestors[i] != -1]
                targets[i, valid] = 1.0

            # Loss
            loss = criterion(logits, targets)
            loss.backward()

            # Optimize
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_hyp.step()
            opt_euc.step()

            # Project embeddings
            with torch.no_grad():
                for param in hyperbolic_params:
                    if param.dim() == 2:
                        norm = torch.sqrt(torch.sum(param ** 2, dim=-1, keepdim=True) + 1e-10)
                        scaling = torch.where(norm > 0.95, 0.95 / norm, torch.ones_like(norm))
                        param.data.mul_(scaling)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        if epoch % 2 == 0:
            print(f"  Epoch {epoch+1}: Loss={avg_loss:.6f}")

    # Evaluate on both domains
    print(f"\nEvaluating...")

    def evaluate(loader, domain_name):
        model.eval()
        hits = 0
        total = 0

        with torch.no_grad():
            for leaves, ancestors in loader:
                leaves = leaves.to(device)
                ancestors = ancestors.to(device)

                logits = model(leaves)
                _, preds = torch.topk(logits, k=10, dim=-1)

                for i in range(len(leaves)):
                    valid = ancestors[i][ancestors[i] != -1].cpu().numpy()
                    if len(valid) == 0:
                        continue

                    pred_set = set(preds[i].cpu().numpy())
                    valid_set = set(valid)

                    if len(pred_set & valid_set) > 0:
                        hits += 1
                    total += 1

        recall = hits / total if total > 0 else 0.0
        print(f"  {domain_name}: Recall@10 = {recall:.4f}")
        return recall

    train_recall = evaluate(train_loader, "IN-DISTRIBUTION (Bio)")
    test_recall = evaluate(test_loader, "OUT-OF-DISTRIBUTION (Artifacts)")

    return train_recall, test_recall


# --- 5. Main Execution ---

if __name__ == "__main__":
    print("="*70)
    print("TAG-R OOD SHOOTOUT EXPERIMENT")
    print("="*70)
    print()
    print("Research Question: Does topology-aware routing generalize to")
    print("unseen semantic domains better than content-based routing?")
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()

    # Create datasets
    print("Building OOD datasets...")
    train_ds = WordNetOODDataset(split='train', max_samples=3000)
    test_ds = WordNetOODDataset(split='test', max_samples=1000)

    # Verify domains are disjoint
    train_leaves = {s['leaf'] for s in train_ds.samples}
    test_leaves = {s['leaf'] for s in test_ds.samples}
    overlap = train_leaves & test_leaves
    print(f"\n  Train/Test Overlap: {len(overlap)} concepts (should be ~0)")
    print()

    # Create loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_ood,
        drop_last=True,  # For batch stats
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_ood,
        drop_last=True,
    )

    # Vocab size must match
    vocab_size = train_ds.vocab_size

    # Run shootout
    results = {}

    print("="*70)
    print("EXPERIMENT 1: MLP Router (Baseline)")
    print("="*70)
    train_mlp, test_mlp = train_and_eval_ood('mlp', train_loader, test_loader, vocab_size, device)
    results['mlp'] = {'train': train_mlp, 'test': test_mlp}

    print("\n" + "="*70)
    print("EXPERIMENT 2: TAG-R Router (Ours)")
    print("="*70)
    train_tagr, test_tagr = train_and_eval_ood('tagr', train_loader, test_loader, vocab_size, device)
    results['tagr'] = {'train': train_tagr, 'test': test_tagr}

    # Report
    print("\n" + "="*70)
    print("SHOOTOUT RESULTS: OOD GENERALIZATION")
    print("="*70)
    print()
    print(f"{'Router':<15} {'In-Dist (Bio)':<20} {'OOD (Artifacts)':<20} {'Gap':<10}")
    print("-"*70)
    print(f"{'MLP (Baseline)':<15} {train_mlp:.4f}{'':<15} {test_mlp:.4f}{'':<15} {train_mlp - test_mlp:.4f}")
    print(f"{'TAG-R (Ours)':<15} {train_tagr:.4f}{'':<15} {test_tagr:.4f}{'':<15} {train_tagr - test_tagr:.4f}")
    print("="*70)
    print()

    # Calculate advantage
    ood_gap_mlp = train_mlp - test_mlp
    ood_gap_tagr = train_tagr - test_tagr
    advantage = ood_gap_mlp - ood_gap_tagr

    print(f"OOD Generalization Gap:")
    print(f"  MLP:  {ood_gap_mlp*100:.2f}pp drop on OOD")
    print(f"  TAG-R: {ood_gap_tagr*100:.2f}pp drop on OOD")
    print(f"  TAG-R Advantage: {advantage*100:.2f}pp better generalization")
    print()

    if advantage > 0.03:  # 3pp advantage
        print("✅ SUCCESS: TAG-R generalizes significantly better!")
        print("   Topology-aware routing is domain-invariant.")
    elif advantage > 0.01:
        print("✓ MODERATE: TAG-R shows some generalization advantage")
    else:
        print("⚠️ INCONCLUSIVE: No clear generalization advantage")
        print("   May need stronger topology features or different domains")

    print("="*70)
