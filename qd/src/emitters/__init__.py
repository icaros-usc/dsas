"""pyribs-compliant emitters."""
import gin

import ribs
from src.emitters.evolution_strategy_emitter import EvolutionStrategyEmitter
from src.emitters.uniform_random_emitter import UniformRandomEmitter

__all__ = [
    "GaussianEmitter",
    "IsoLineEmitter",
    "EvolutionStrategyEmitter",
    "GradientArborescenceEmitter",
    "UniformRandomEmitter",
]


@gin.configurable
class GaussianEmitter(ribs.emitters.GaussianEmitter):
    """gin-configurable version of pyribs GaussianEmitter."""


@gin.configurable
class IsoLineEmitter(ribs.emitters.IsoLineEmitter):
    """gin-configurable version of pyribs IsoLineEmitter."""


@gin.configurable
class GradientArborescenceEmitter(ribs.emitters.GradientArborescenceEmitter):
    """gin-configurable version of pyribs GradientArborescenceEmitter."""
