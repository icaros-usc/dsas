"""Networks for surrogate model."""

import gin
import torch
import torch.nn.functional as F
from torch import nn


@gin.configurable
class OccupancyNet(nn.Module):
    """Class for network predicting occupancy.

    Args:
        num_inputs: Number of inputs to the network (num_goals * 2 + num_obstacles *3)
    """

    def __init__(self, num_inputs: int):
        super(OccupancyNet, self).__init__()
        self.num_inputs = num_inputs
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(self.num_inputs, 32, 4, 1, 0, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 4, 2, 1, bias=True),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(2),
        )

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: (batch_size, num_inputs) shape tensor.

        Returns:
            Predicted occupancy grids of shape (batch_size, 2, 32, 32).
        """
        x = x.view(-1, self.num_inputs, 1, 1)
        x = self.layers(x)
        x = F.log_softmax(x.view(-1, 2, 32 * 32), dim=2)
        return x.view(-1, 2, 32, 32)


@gin.configurable
class PredictNet(nn.Module):
    """Class for network predicting objective and measures.

    Args:
        num_inputs: Number of inputs to the network (num_goals * 2 + num_obstacles *3)
    """

    def __init__(self, num_inputs: int, occ_input: bool):
        super(PredictNet, self).__init__()
        self.num_inputs = num_inputs
        self.occ_input = occ_input

        self.preprocess = nn.Sequential(
            nn.Linear(self.num_inputs, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
        )

        if occ_input:
            self.direct_layers = nn.Sequential(
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
            )

            self.conv_layers = nn.Sequential(
                # 32 x 32 -> 16 x 16
                nn.Conv2d(2, 16, 4, 2, 1, bias=True),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(inplace=True),
                # 16 x 16 -> 8 x 8
                nn.Conv2d(16, 32, 4, 2, 1, bias=True),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(inplace=True),
                # 8 x 8 -> 4 x 4
                nn.Conv2d(32, 64, 4, 2, 1, bias=True),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(inplace=True),
                # 4 x 4 -> 1 x 1
                nn.Conv2d(64, 128, 4, 1, 0, bias=True),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(inplace=True),
            )

            self.head = nn.Linear(256, 3)
        else:
            self.direct_layers = nn.Sequential(
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
            )

            self.head = nn.Linear(128, 3)

    def forward(self, x, grid_pred=None):
        """Forward pass.

        Args:
            x: (batch_size, num_inputs) shape tensor.
            grid_pred: (batch_size, 2, 32, 32) shape tensor output by OccupancyNet.

        Returns:
            Predicted objective, measures of shape (batch_size, 3).
        """
        pre_features = self.preprocess(x)
        features = self.direct_layers(pre_features)

        if self.occ_input:
            if grid_pred is None:
                raise ValueError(
                    "grid_pred should be passed to pred_net's forward when occ_input is "
                    "set to True."
                )
            grid = grid_pred.view(-1, 2, 32, 32)
            conv_features = self.conv_layers(grid).view(-1, 128)

            features = torch.hstack([features, conv_features])
        outs = self.head(features)

        return outs
