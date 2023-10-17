"""Provides SurrogateModel."""
import logging
import pickle as pkl
from pathlib import Path

import cloudpickle
import gin
import numpy as np
import torch

from src.device import DEVICE
from src.scenario.emulation_model.buffer import Buffer, Experience
from src.scenario.emulation_model.networks import OccupancyNet, PredictNet
from src.scenario.emulation_model.sa_networks import SAOccupancyNet, SAPredictNet

logger = logging.getLogger(__name__)


@gin.configurable(denylist=["collab", "seed"])
class SurrogateModel:
    """Class for surrogate model.

    Args:
        use_occ_net (bool): True if occ_net should be used for prediction.
        occ_train_epochs (int): Number of times to iterate over the dataset when training
            the occupancy network.
        occ_optim_lr (float): Learning rate for occupancy network optimizer.
        pred_train_epochs (int): Number of times to iterate over the dataset when training
            the prediction network.
        pred_optim_lr (float): Learning rate for prediction network optimizer.
        train_batch_size (int): Batch size for each epoch of training.
        train_sample_size (int): Number of samples to choose for training. Set
            to None if all available data should be used. (default: None)
        train_sample_type (str): One of the following
            "random": Choose `n` uniformly random samples;
            "recent": Choose `n` most recent samples;
            "weighted": Choose samples s.t., on average, all samples are seen
                the same number of times across training iterations.
            (default: "recent")
        collab (bool): True for collab tasks. Passed in from Manager.
        seed (int): Master seed. Passed in from Manager.

    Usage:
        model = SurrogateModel(...)

        # Add inputs, objectives, measures to use for training.
        model.add(data)

        # Training hyperparameters should be passed in at initialization.
        model.train()

        # Ask for objectives and measures.
        model.predict(...)
    """

    def __init__(
        self,
        use_occ_net: bool = gin.REQUIRED,
        occ_train_epochs: int = None,
        occ_optim_lr: float = None,
        pred_train_epochs: int = gin.REQUIRED,
        pred_optim_lr: float = gin.REQUIRED,
        train_batch_size: int = gin.REQUIRED,
        train_sample_size: int = None,
        train_sample_type: str = "recent",
        collab: bool = True,
        seed: int = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.use_occ_net = use_occ_net

        if use_occ_net:
            if occ_train_epochs is None or occ_optim_lr is None:
                raise ValueError(
                    "occ_train_epochs and occ_optim_lr should be set if use_occ_net is "
                    "True."
                )

            occ_net = OccupancyNet() if collab else SAOccupancyNet()
            self.occ_net = occ_net.to(DEVICE)  # Args handled by gin.
            self.occ_opt = torch.optim.AdamW(self.occ_net.parameters(), lr=occ_optim_lr)
            self.occ_train_epochs = occ_train_epochs

        pred_net = PredictNet() if collab else SAPredictNet()
        self.pred_net = pred_net.to(DEVICE)  # Args handled by gin.
        self.pred_opt = torch.optim.AdamW(self.pred_net.parameters(), lr=pred_optim_lr)
        self.pred_train_epochs = pred_train_epochs

        self.dataset = Buffer(seed=seed)

        self.train_batch_size = train_batch_size
        self.train_sample_size = train_sample_size
        self.train_sample_type = train_sample_type

        self.torch_rng = torch.Generator("cpu")  # Required to be on CPU.
        self.torch_rng.manual_seed(seed)

    def add(self, e: Experience):
        """Adds experience to the buffer."""
        self.dataset.add(e)

    def train(self):
        """Trains occupancy network for self.occ_train_epochs epochs followed by
        prediction network for self.pred_train_epochs on the dataset."""
        if len(self.dataset) == 0:
            logger.warning("Skipping training as dataset is empty")
            return

        dataloader = self.dataset.to_dataloader(
            self.train_batch_size,
            self.torch_rng,
            self.train_sample_size,
            self.train_sample_type,
        )
        logger.info(f"Using {len(dataloader.dataset)} samples to train.")

        if self.use_occ_net:
            logger.info(f"Training occupancy network.")
            self.occ_net.train()
            occ_loss_func = torch.nn.KLDivLoss(reduction="batchmean")
            for e in range(1, self.occ_train_epochs + 1):
                epoch_loss = 0.0

                for inputs, occ_grids, outputs in dataloader:
                    occ_grid_pred = self.occ_net(inputs)
                    loss = occ_loss_func(occ_grid_pred, occ_grids)
                    epoch_loss += loss.item()

                    self.occ_opt.zero_grad()
                    loss.backward()
                    self.occ_opt.step()

                if e % 10 == 0 or e == self.occ_train_epochs:
                    logger.info(f"({e}/{self.occ_train_epochs}) Epoch Loss: {epoch_loss}")

        logger.info(f"Training prediction network.")
        if self.use_occ_net:
            self.occ_net.eval()
        self.pred_net.train()
        pred_loss_func = torch.nn.MSELoss()
        for e in range(1, self.pred_train_epochs + 1):
            epoch_loss = 0.0

            for inputs, occ_grids, outputs in dataloader:
                occ_grid_pred = None
                if self.use_occ_net:
                    with torch.no_grad():
                        occ_grid_pred = self.occ_net(inputs)
                output_pred = self.pred_net(inputs, occ_grid_pred)
                loss = pred_loss_func(output_pred, outputs)
                epoch_loss += loss.item()

                self.pred_opt.zero_grad()
                loss.backward()
                self.pred_opt.step()

            if e % 10 == 0 or e == self.pred_train_epochs:
                logger.info(f"({e}/{self.pred_train_epochs}) Epoch Loss: {epoch_loss}")

    def predict(self, inputs: np.ndarray):
        """Predicts objectives and measures for a batch of solutions.

        Args:
            inputs (np.ndarray): Batch of inputs (goals + obstacles) to predict.
        Returns:
            Batch of objectives and batch of measures.
        """
        # Handle no_grad here since we expect everything to be numpy arrays.
        with torch.no_grad():
            model_input_tensor = torch.as_tensor(inputs, device=DEVICE, dtype=torch.float)
            outputs = self.eval_predict_with_grad(model_input_tensor)
            outputs_np = outputs.cpu().detach().numpy()

            objs_np = outputs_np[:, 0]
            measures_np = outputs_np[:, 1:]

            return objs_np, measures_np

    def eval_predict_with_grad(self, model_input_tensor: torch.Tensor):
        """Predict the outputs without putting torch into no grad mode. The
        models are put into eval mode since we don't want batch norm to update
        its running mean and variance.

        Args:
            model_input_tensor: Input tensor with shape (batch_size, input_size)

        Returns:
            Output of the surrogate model prediction with shape (batch_size, 3)
            corresponding to the objective followed by the 2 measures.
        """
        if self.use_occ_net:
            self.occ_net.eval()
        self.pred_net.eval()

        grid_pred = None
        if self.use_occ_net:
            grid_pred = self.occ_net(model_input_tensor)
        return self.pred_net(model_input_tensor, grid_pred)

    def save(self, pickle_path: Path, pytorch_path: Path):
        """Saves data to a pickle file and a PyTorch file.

        The PyTorch file holds the network and the optimizer, and the pickle
        file holds the rng and the dataset. See here for more info:
        https://pytorch.org/tutorials/beginner/saving_loading_models.html#save
        """
        logger.info("Saving SurrogateModel pickle data")
        with pickle_path.open("wb") as file:
            cloudpickle.dump(
                {"rng": self.rng, "dataset": self.dataset}, file,
            )

        logger.info("Saving SurrogateModel PyTorch data")
        state_dict = {
            "pred_net": self.pred_net.state_dict(),
            "pred_opt": self.pred_opt.state_dict(),
            "torch_rng": self.torch_rng.get_state(),
        }

        if self.use_occ_net:
            state_dict["occ_net"] = self.occ_net.state_dict()
            state_dict["occ_opt"] = self.occ_opt.state_dict()

        torch.save(state_dict, pytorch_path)

    def load(self, pickle_path: Path, pytorch_path: Path):
        """Loads data from files saved by save()."""
        with open(pickle_path, "rb") as file:
            pickle_data = pkl.load(file)
            self.rng = pickle_data["rng"]
            self.dataset = pickle_data["dataset"]

        pytorch_data = torch.load(pytorch_path)
        if self.use_occ_net:
            self.occ_net.load_state_dict(pytorch_data["occ_net"])
            self.occ_opt.load_state_dict(pytorch_data["occ_opt"])
        self.pred_net.load_state_dict(pytorch_data["pred_net"])
        self.pred_opt.load_state_dict(pytorch_data["pred_opt"])
        self.torch_rng.set_state(pytorch_data["torch_rng"])
        return self
