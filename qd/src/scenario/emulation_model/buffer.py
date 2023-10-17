"""Simple dataset for storing and sampling data for emulation model."""
from collections import namedtuple

import numpy as np
import torch

from src.device import DEVICE

Experience = namedtuple("Experience", ["input", "occ_grids", "output"])

# Used for batches of items, e.g. a batch of levels, a batch of objectives.
BatchExperience = namedtuple("BatchExperience", Experience._fields)


class Buffer:
    """Stores data samples for training the emulation model.

    Args:
        seed (int): Random seed to use (default None)
    """

    def __init__(
        self, seed: int = None,  # pylint: disable = unused-argument
    ):
        self.all_input = []
        self.all_occ_grids = []
        self.all_output = []

    def add(self, e: Experience):
        """Adds experience to the buffer."""
        self.all_input.append(e.input)
        self.all_occ_grids.append(e.occ_grids)
        self.all_output.append(e.output)

    def __len__(self):
        """Number of Experience in the buffer."""
        return len(self.all_input)

    def to_tensors(self):
        """Converts all buffer data to tensors."""
        # Convert to np.array due to this warning: Creating a tensor from a list
        # of numpy.ndarrays is extremely slow. Please consider converting the
        # list to a single numpy.ndarray with numpy.array() before converting to
        # a tensor.
        return BatchExperience(
            torch.as_tensor(np.array(self.all_input), device=DEVICE, dtype=torch.float),
            torch.as_tensor(
                np.array(self.all_occ_grids), device=DEVICE, dtype=torch.float
            ),
            torch.as_tensor(np.array(self.all_output), device=DEVICE, dtype=torch.float),
        )

    def to_dataloader(
        self,
        batch_size: int,
        torch_rng: torch.Generator,
        sample_size: int = None,
        sample_type: str = "recent",
    ):
        """Converts buffer data to a PyTorch dataloader.

        Args:
            batch_size: Batch size for the DataLoader.
            torch_rng: PyTorch random number generator for the DataLoader. We
                let the caller handle this since it is un-pickleable, so
                including it as a Buffer attribute would make Buffer
                un-pickleable.
            sample_size (int): Number of samples to return. Set to None if all
                available data should be returned. (default: None)
            sample_type (str): One of the following
                "random": Choose `n` uniformly random samples;
                "recent": Choose `n` most recent samples;
                "weighted": Choose samples s.t., on average, all samples are
                    seen the same number of times across training iterations.
                (default: "recent")
        """
        batch_experience = self.to_tensors()

        # No copy to make a TensorDataset - see:
        # https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#TensorDataset
        dataset = torch.utils.data.TensorDataset(*batch_experience)

        sample_size = sample_size or len(dataset)
        if sample_type == "random":
            idx = torch.randperm(len(dataset), generator=torch_rng)[:sample_size]
        elif sample_type == "recent":
            idx = range(len(dataset))[-sample_size:]
        else:
            raise NotImplementedError(f"'{sample_type}' sampling not implemented.")

        subset = torch.utils.data.Subset(dataset, idx)

        # See here for why we use Generator:
        # https://pytorch.org/docs/stable/notes/randomness.html#dataloader
        # We don't use seed_worker because we only run the DataLoader on the
        # main process here.
        return torch.utils.data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch_rng,
            drop_last=True,
        )
