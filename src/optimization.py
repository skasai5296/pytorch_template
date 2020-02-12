import torch
from torch import nn, optim


def get_optimizer(CONFIG, params):
    optimizer = getattr(optim, CONFIG.optimizer.name)(params, **CONFIG.optimizer)
    return optimizer


def get_criterion(CONFIG):
    criterion = {}
    criterion["MSELoss"] = nn.MSELoss()
    return criterion
