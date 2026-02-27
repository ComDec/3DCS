"""Core package for the 3DBench evaluation tooling."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("3dcs")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__", "benchmarks", "datasets", "embeddings"]
