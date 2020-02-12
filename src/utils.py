import os
import time

import torch
import torch.nn as nn
import torch.nn.init as init


class ModelSaver:
    """Saves and Loads model and optimizer parameters"""

    def __init__(self, path, init_val=0):
        self.path = path
        self.best = init_val
        self.epoch = 1

    def load_ckpt(self, model, optimizer, device):
        if os.path.exists(self.path):
            print(f"loading model from {self.path}")
            ckpt = torch.load(self.path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            self.best = ckpt["bestscore"]
            self.epoch = ckpt["epoch"] + 1
            print(
                f"best score is set to {self.best}, restarting from epoch "
                + f"{self.epoch}"
            )
        else:
            print(f"{self.path} does not exist, not loading")

    def save_ckpt_if_best(self, model, optimizer, metric):
        if metric > self.best:
            print(
                f"score {metric} is better than previous best score of "
                + f"{self.best}, saving to {self.path}"
            )
            self.best = metric
            ckpt = {"optimizer": optimizer.state_dict()}
            if hasattr(model, "module"):
                ckpt["model"] = model.module.state_dict()
            else:
                ckpt["model"] = model.state_dict()
            ckpt["epoch"] = self.epoch
            ckpt["bestscore"] = self.best
            torch.save(ckpt, self.path)
        else:
            print(
                f"score {metric} is not better than previous best score of "
                + f"{self.best}, not saving"
            )
        self.epoch += 1


class Timer:
    """Computes and stores the time"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.begin = time.time()

    def __str__(self):
        return sec2str(time.time() - self.begin)


def sec2str(sec):
    if sec < 60:
        return "time elapsed: {:02d}s".format(int(sec))
    elif sec < 3600:
        min = int(sec / 60)
        sec = int(sec - min * 60)
        return "time elapsed: {:02d}m{:02d}s".format(min, sec)
    elif sec < 24 * 3600:
        min = int(sec / 60)
        hr = int(min / 60)
        sec = int(sec - min * 60)
        min = int(min - hr * 60)
        return "time elapsed: {:02d}h{:02d}m{:02d}s".format(hr, min, sec)
    elif sec < 365 * 24 * 3600:
        min = int(sec / 60)
        hr = int(min / 60)
        dy = int(hr / 24)
        sec = int(sec - min * 60)
        min = int(min - hr * 60)
        hr = int(hr - dy * 24)
        return "time elapsed: {:02d} days, {:02d}h{:02d}m{:02d}s".format(
            dy, hr, min, sec
        )


def count_parameters(model):
    """
    Usage:
        model = Model()
        print(f"number of trainable params: {count_parameters(model)}")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
