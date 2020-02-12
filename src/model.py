import torch
from torch import nn

from utils import weight_init


class SampleModel(nn.Module):
    def __init__(self, CONFIG, criterion):
        super().__init__()
        self.criterion = criterion
        self.CONFIG = CONFIG
        self.linear = nn.Linear(self.CONFIG.input_channel, self.CONFIG.output_channel)

        # initialize weights
        self.linear.apply(weight_init)

    def forward(self, hoge, label):
        """
        Args:
            torch.Tensor hoge:          (bs x input_channel)
            torch.Tensor label:         (bs)
        Returns:
            torch.Tensor out:           (bs x output_channel)
        """
        assert isinstance(hoge, torch.Tensor)
        assert hoge.ndim() == 2
        assert hoge.size(1) == self.CONFIG.input_channel
        out = self.linear(hoge)
        assert out.size(1) == self.CONFIG.output_channel
        return out

    def infer(self, hoge):
        out = self.linear(hoge)
        return out
