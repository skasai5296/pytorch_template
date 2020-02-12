import torch
from torch import nn, optim


def get_optimizer(CONFIG, params):
    if CONFIG.optimizer.name == "Adam":
        optimizer = optim.Adam(
            params,
            lr=CONFIG.optimizer.lr,
            betas=(CONFIG.optimizer.beta1, CONFIG.optimizer.beta2),
        )
    return optimizer


class SampleLoss(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        self.criterion = []
        self.names = []
        self.weights = []
        if CONFIG.loss.weight.MSE >= 0.0:
            self.criterion.append(nn.MSELoss())
            self.names.append("MSELoss")
            self.weights.append(CONFIG.loss.weight.MSE)

    def forward(self, y, t):
        loss = 0
        losses = {}
        for criterion, name, weight in zip(self.criterion, self.names, self.weights):
            lossval = criterion(y, t)
            loss += lossval * weight
            losses[name] = lossval.item()
        return loss, losses
