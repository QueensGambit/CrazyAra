"""
@file: metrics_pytorch.py
Created on 13.06.22
@project: CrazyAra
@author: queensgambit

Metric definitions for Pytorch
"""
import torch


class Metric:
    def __init__(self):
        pass

    def reset(self) -> None:
        pass

    def update(self, preds: torch.Tensor, label: torch.Tensor) -> None:
        pass

    def compute(self) -> float:
        pass


class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        self.correct_cnt = 0
        self.total_cnt = 0

    def reset(self) -> None:
        self.correct_cnt = 0
        self.total_cnt = 0

    def update(self, preds: torch.Tensor, label: torch.Tensor) -> None:
        self.correct_cnt += float((preds == label.data).sum())
        self.total_cnt += preds.shape[1]

    def compute(self) -> float:
        return self.correct_cnt / self.total_cnt


class MSE(Metric):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()
        self.loss_value = 0

    def reset(self) -> None:
        self.loss_value = 0

    def update(self, preds: torch.Tensor, label: torch.Tensor) -> None:
        self.loss_value = self.loss(preds, label)

    def compute(self) -> float:
        return self.loss_value


class CrossEntropy(Metric):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()
        self.loss_value = 0

    def reset(self) -> None:
        self.loss_value = 0

    def update(self, preds: torch.Tensor, label: torch.Tensor) -> None:
        self.loss_value = self.loss(preds, label)

    def compute(self) -> float:
        return self.loss_value


class AccuracySign(Metric):
    def __init__(self):
        super().__init__()
        self.correct_cnt = 0
        self.denominator = 0

    def reset(self) -> None:
        self.correct_cnt = 0
        self.denominator = 0

    def update(self, preds: torch.Tensor, label: torch.Tensor) -> None:
        self.correct_cnt += float((preds.sign() == label.data.sign()).sum())
        self.denominator += label.shape[1] - (label == 0).sum()

    def compute(self) -> float:
        if self.denominator != 0:
            return self.correct_cnt / self.denominator
        return 0
