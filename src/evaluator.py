import torch


class SampleEvaluator:
    def __init__(self, CONFIG):
        self.CONFIG = CONFIG

    def compute_metrics(self, hyp, ans):
        metrics_dict = {}
        return metrics_dict
