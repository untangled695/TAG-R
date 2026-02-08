from setuptools import setup, find_packages

setup(
    name="adaptive-geometric-attention",
    version="0.1.0",
    description="Adaptive Geometric Attention with Dynamic Manifold Routing",
    author="Research Team",
    packages=find_packages(),
    python_requires=">=3.10,<3.14",
    install_requires=[
        "torch>=2.0.0,<2.1.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "geoopt==0.5.0",
        "geomstats>=2.6.0",
        "nltk>=3.8.1",
        "transformers>=4.30.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.13.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "pytest>=7.3.0",
        "pytest-cov>=4.1.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
