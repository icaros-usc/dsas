import gin
import numpy as np
from numpy.typing import ArrayLike

import ribs
from ribs.emitters import EmitterBase


@gin.configurable(denylist=["archive", "seed"])
class UniformRandomEmitter(EmitterBase):
    """Emitter that outputs uniform random solutions given bounds.

    Args:
        archive: Archive to store the solutions.
        solution_dim: Initial solution. Only used for solution_dim.
        bounds: Bounds of the solution space. Pass an array of shape (solution_dim, 2) to
            specify the bounds for each dim. If the bounds are disjoint, pass a 3D array
            of shape (num_bounds, solution_dim, 2).
        batch_size: Number of solutions to return in :meth:`ask`.
        seed: Random seed. (default None)
    """

    def __init__(
        self,
        archive: ribs.archives.ArchiveBase,
        solution_dim: int = gin.REQUIRED,
        bounds: ArrayLike = gin.REQUIRED,
        batch_size: int = gin.REQUIRED,
        seed: int = None,
    ):
        super().__init__(archive, solution_dim=solution_dim, bounds=None)
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size

        bounds = np.array(bounds)
        if len(bounds.shape) == 2:
            bounds = bounds[None, ...]

        # Calculate lower and upper bounds of shape (num_bounds, solution_dim).
        self._aug_lower_bounds = np.full(
            (bounds.shape[0], bounds.shape[1]), -np.inf, dtype=archive.dtype
        )
        self._aug_upper_bounds = np.full(
            (bounds.shape[0], bounds.shape[1]), np.inf, dtype=archive.dtype
        )
        for i, bi in enumerate(bounds):
            self._aug_lower_bounds[i], self._aug_upper_bounds[i] = self._process_bounds(
                bi, solution_dim, archive.dtype
            )

    def ask(self):
        # Calculate uniform random numbers of shape (num_bounds, batch_size, solution_dim)
        # within the corresponding bounds.
        all_rands = self.rng.uniform(
            low=self._aug_lower_bounds[:, None, :],
            high=self._aug_upper_bounds[:, None, :],
            size=(self._aug_lower_bounds.shape[0], self.batch_size, self.solution_dim),
        )

        # Randomly choose the bound from which the random number should be chosen from.
        bound_choices = self.rng.integers(
            self._aug_lower_bounds.shape[0], size=(self.batch_size, self.solution_dim)
        )

        return np.choose(bound_choices, all_rands)
