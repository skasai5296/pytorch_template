import torch
from torch import nn

from utils import weight_init


class SampleModel(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        self.CONFIG = CONFIG
        self.network = nn.Sequential(
            nn.Linear(
                CONFIG.hyperparam.input_channel, CONFIG.hyperparam.output_channel
            ),
            nn.Sigmoid(),
        )

        # initialize weights
        self.network.apply(weight_init)

    def forward(self, hoge):
        """
        Args:
            torch.Tensor hoge:          (bs x input_channel)
        Returns:
            torch.Tensor out:           (bs x output_channel)
        """
        assert isinstance(hoge, torch.Tensor)
        out = self.network(hoge)
        return out

    def infer(self, hoge):
        out = self.network(hoge)
        return out
