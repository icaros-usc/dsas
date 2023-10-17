"""Internal subpackage with optimizers for use across emitters."""
from src.emitters.opt._cma_es import CMAEvolutionStrategy

__all__ = [
    "CMAEvolutionStrategy",
]
