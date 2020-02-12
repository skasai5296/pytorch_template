import argparse
import os
import random
from pprint import pprint

import numpy as np
import torch
import wandb
import yaml
from addict import Dict
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import SampleDataset, get_collater
from evaluator import SampleEvaluator
from model import SampleModel
from optimization import SampleLoss, get_optimizer
from utils import ModelSaver, Timer


def train_epoch(loader, model, optimizer, criterion, device, CONFIG, epoch):
    train_timer = Timer()
    model.train()
    for it, data in enumerate(loader):
        hoge = data["hoge"]
        label = data["label"]

        hoge = hoge.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        out = model(hoge)
        loss, losses = criterion(out, label)

        if CONFIG.use_wandb:
            wandb.log(losses)
        lossstr = " | ".join([f"{name}:\t{val:7f}" for name, val in losses.items()])

        loss.backward()
        optimizer.step()

        if it % 10 == 9:
            print(f"train {train_timer} | iter {it+1} / {len(loader)} | {lossstr}")


def validate(loader, model, evaluator, device, CONFIG, epoch):
    valid_timer = Timer()
    valid_iters = (CONFIG.valid_size + loader.batch_size - 1) // loader.batch_size
    model.eval()
    hyp = []
    ans = []
    m = model.module if hasattr(model, "module") else model
    for it, data in enumerate(loader):
        hoge = data["hoge"]
        label = data["label"]

        hoge = hoge.to(device)
        label = label.to(device)

        with torch.no_grad():
            pred = m.infer(hoge)

        hyp.extend(pred)
        ans.extend(label)

        if it % 10 == 9:
            print(f"valid {valid_timer} | iter {it+1} / {valid_iters}")

        # only iterate for enough samples
        if it == valid_iters - 1:
            break
    hyp = hyp[: CONFIG.valid_size]
    ans = ans[: CONFIG.valid_size]

    metrics = {}
    print("\n\n---VALIDATION RESULTS---")
    metrics = evaluator.compute_metrics(hyp=hyp, ans=ans)
    print("\n".join([f"{name}:\t{val:7f}" for name, val in metrics.items()]))
    print("---VALIDATION RESULTS---\n\n")
    if CONFIG.use_wandb:
        wandb.init(config=CONFIG, project=CONFIG.project_name)

    return metrics


if __name__ == "__main__":

    global_timer = Timer()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="../cfg/sample.yml",
        help="path to configuration yml file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="denotes if to continue training, will use config",
    )
    opt = parser.parse_args()
    print(f"loading configuration from {opt.config}")
    CONFIG = Dict(yaml.safe_load(open(opt.config)))
    print("CONFIGURATIONS:")
    pprint(CONFIG)
    print("\n\n")

    CONFIG.gpu_ids = list(map(str, CONFIG.gpu_ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(CONFIG.gpu_ids)

    if CONFIG.random_seed is not None:
        random.seed(CONFIG.random_seed)
        np.random.seed(CONFIG.random_seed)
        torch.manual_seed(CONFIG.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # device
    if torch.cuda.is_available() and CONFIG.cuda:
        device = torch.device("cuda")
        print("using GPU numbers {}".format(CONFIG.gpu_ids))
    else:
        device = torch.device("cpu")
        print("using CPU")

    # prepare datasets
    print("loading datasets...")
    train_dataset = SampleDataset(CONFIG, "train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        collate_fn=get_collater("train"),
    )
    valid_dataset = SampleDataset(CONFIG, "val")
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        collate_fn=get_collater("val"),
    )

    # model, optimizer, criterion
    print("loading model and related components...")
    model = SampleModel(CONFIG)
    optimizer = get_optimizer(CONFIG, model.parameters())
    criterion = SampleLoss(CONFIG)
    model = model.to(device)

    # evaluator, saver
    evaluator = SampleEvaluator(CONFIG)
    # prepare output directory
    outdir = os.path.join(CONFIG.path.output, CONFIG.config_name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    model_path = os.path.join(outdir, "best_score.ckpt")
    saver = ModelSaver(model_path, init_val=-1e10)
    if opt.resume:
        saver.load_ckpt(model, optimizer, device)
    offset_epoch = saver.epoch
    if offset_epoch > CONFIG.max_epoch:
        raise RuntimeError(
            "trying to restart at epoch {} while max training is set to {} epochs".format(
                offset_epoch, CONFIG.max_epoch
            )
        )

    if torch.cuda.device_count() > 1 and CONFIG.dataparallel:
        model = nn.DataParallel(model)

    if CONFIG.use_wandb:
        wandb.watch(model)

    # training loop
    print("\n\n---BEGIN TRAINING---\n\n")
    for ep in range(offset_epoch - 1, CONFIG.max_epoch):
        print("global {} | begin training for epoch {}".format(global_timer, ep + 1))
        train_epoch(train_loader, model, optimizer, criterion, device, CONFIG, ep)
        print(
            "global {} | done with training for epoch {}, beginning validation".format(
                global_timer, ep + 1
            )
        )
        metrics = validate(valid_loader, model, evaluator, device, CONFIG, ep)
        if CONFIG.use_wandb:
            wandb.log(metrics)
        if CONFIG.valid_metric in metrics.keys():
            saver.save_ckpt_if_best(model, optimizer, metrics[CONFIG.valid_metric])
        print("global {} | end epoch {}".format(global_timer, ep + 1))
    print("\n\n---COMPLETED TRAINING---")
