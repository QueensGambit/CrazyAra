"""
@file: metrics_pytorch.py
Created on 13.06.22
@project: CrazyAra
@author: queensgambit

Metric definitions for Pytorch
"""
import torch
from DeepCrazyhouse.src.training.trainer_agent_pytorch import SoftCrossEntropyLoss


class Metric:
    def __init__(self):
        pass

    def reset(self) -> None:
        pass

    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        pass

    def compute(self) -> float:
        pass


class Accuracy(Metric):
    def __init__(self, sparse_policy_label):
        super().__init__()
        self.correct_cnt = 0
        self.total_cnt = 0
        self.sparse_policy_label = sparse_policy_label

    def reset(self) -> None:
        self.correct_cnt = 0
        self.total_cnt = 0

    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        if self.sparse_policy_label:
            self.correct_cnt += float((preds == labels.data).sum())
        else:
            self.correct_cnt += float((preds == labels.argmax(dim=1)).sum())
        self.total_cnt += preds.shape[0]

    def compute(self) -> float:
        return self.correct_cnt / self.total_cnt


class MSE(Metric):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()
        self.loss_sum = 0
        self.nb_batches = 0

    def reset(self) -> None:
        self.loss_sum = 0
        self.nb_batches = 0

    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        self.loss_sum += self.loss(preds, labels)
        self.nb_batches += 1

    def compute(self) -> float:
        return self.loss_sum / self.nb_batches


class CrossEntropy(Metric):
    def __init__(self, sparse_policy_label):
        """
        :param: sparse_policy_label: Decides if the cross entropy loss has sparse labels
        """
        super().__init__()
        if sparse_policy_label:
            self.loss = torch.nn.CrossEntropyLoss()
        else:
            self.loss = SoftCrossEntropyLoss()
        self.loss_sum = 0
        self.nb_batches = 0
        self.sparse_policy_label = sparse_policy_label

    def reset(self) -> None:
        self.loss_sum = 0
        self.nb_batches = 0

    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        if self.sparse_policy_label:
            self.loss_sum += self.loss(preds, labels.long())
        else:
            self.loss_sum += self.loss(preds, labels)
        self.nb_batches += 1

    def compute(self) -> float:
        return self.loss_sum / self.nb_batches


class AccuracySign(Metric):
    def __init__(self):
        super().__init__()
        self.correct_cnt = 0
        self.denominator = 0

    def reset(self) -> None:
        self.correct_cnt = 0
        self.denominator = 0

    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        self.correct_cnt += float((preds.sign() == labels.data.sign()).sum())
        self.denominator += labels.shape[0] - (labels == 0).sum()

    def compute(self) -> float:
        if self.denominator != 0:
            return self.correct_cnt / self.denominator
        return 0
