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
    m = model.module if hasattr(model, "module") else model
    for it, data in enumerate(loader):
        hoge = data["hoge"]
        label = data["label"]

        hoge = hoge.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        out = model(hoge)
        loss, losses = criterion(out, label)

        if CONFIG.basic.use_wandb:
            wandb.log(losses)
        lossstr = " | ".join([f"{name}: {val:7f}" for name, val in losses.items()])

        loss.backward()
        if CONFIG.optimizer.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                m.parameters(), max_norm=CONFIG.optimizer.grad_clip
            )
        optimizer.step()

        if it % 10 == 9:
            print(f"train {train_timer} | iter {it+1} / {len(loader)} | {lossstr}")


def validate(loader, model, evaluator, device, CONFIG, epoch):
    valid_timer = Timer()
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
            print(f"valid {valid_timer} | iter {it+1} / {len(loader)}")

    metrics = {}
    print("\n\n---VALIDATION RESULTS---")
    metrics = evaluator.compute_metrics(hyp=hyp, ans=ans)
    print("\n".join([f"{name}:\t{val:7f}" for name, val in metrics.items()]))
    print("---VALIDATION RESULTS---\n\n")
    if CONFIG.basic.use_wandb:
        wandb.log(metrics)

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

    if CONFIG.basic.use_wandb:
        wandb.init(config=CONFIG, project=CONFIG.basic.project_name)

    CONFIG.basic.gpu_ids = list(map(str, CONFIG.basic.gpu_ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(CONFIG.basic.gpu_ids)

    if CONFIG.misc.random_seed != 0:
        random.seed(CONFIG.misc.random_seed)
        np.random.seed(CONFIG.misc.random_seed)
        torch.manual_seed(CONFIG.misc.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # device
    if torch.cuda.is_available() and CONFIG.basic.cuda:
        device = torch.device("cuda")
        print("using GPU numbers {}".format(CONFIG.basic.gpu_ids))
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
    outdir = os.path.join(CONFIG.basic.output_path, CONFIG.basic.config_name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    model_path = os.path.join(outdir, "best_score.ckpt")
    saver = ModelSaver(model_path, init_val=-1e10)
    if opt.resume:
        saver.load_ckpt(model, optimizer, device)
    offset_epoch = saver.epoch
    if offset_epoch > CONFIG.hyperparam.max_epoch:
        raise RuntimeError(
            "trying to restart at epoch {} while max training is set to {} epochs".format(
                offset_epoch, CONFIG.hyperparam.max_epoch
            )
        )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if CONFIG.basic.use_wandb:
        wandb.watch(model)

    # training loop
    print("\n\n---BEGIN TRAINING---\n\n")
    for ep in range(offset_epoch - 1, CONFIG.hyperparam.max_epoch):
        print("global {} | begin training for epoch {}".format(global_timer, ep + 1))
        train_epoch(train_loader, model, optimizer, criterion, device, CONFIG, ep)
        print(
            "global {} | done with training for epoch {}, beginning validation".format(
                global_timer, ep + 1
            )
        )
        metrics = validate(valid_loader, model, evaluator, device, CONFIG, ep)
        if CONFIG.basic.use_wandb:
            wandb.log(metrics)
        if CONFIG.scheduler.valid_metric in metrics.keys():
            saver.save_ckpt_if_best(
                model, optimizer, metrics[CONFIG.scheduler.valid_metric]
            )
        print("global {} | end epoch {}".format(global_timer, ep + 1))
    print("\n\n---COMPLETED TRAINING---")
