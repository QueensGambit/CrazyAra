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


class BetaLoss(Metric):
    def __init__(self):
        super().__init__()
        self.loss_sum = 0
        self.nb_batches = 0

    def reset(self) -> None:
        self.loss_sum = 0
        self.nb_batches = 0

    def update(self, preds_mu: torch.Tensor, preds_beta: torch.Tensor, labels: torch.Tensor, nb_rollouts=800) -> None:
        self.loss_sum += value_loss_beta_uncertainty(preds_mu, preds_beta, labels, nb_rollouts)
        self.nb_batches += 1

    def compute(self) -> float:
        return self.loss_sum / self.nb_batches


def gamma_func(x):
    """Returns the gamma function output x: gamma(x) = (x-1)!"""
    return x.lgamma().exp()


def beta_func(x, y):
    """Returns the beta function output of x: beta(x) = (gamma(x)gamma(y))/gamma(x+y)"""
    return (gamma_func(x)*gamma_func(y))/gamma_func(x+y)


def value_loss_beta_uncertainty(mu, beta, value_target, nb_rollouts=800):
    """Computes the loss based on the beta distribution.
    :param mu: Value output (expected to be in [-1,+1]
    :param beta: Beta parameter of the beta function
    :param value_target: Value target to learn from in [-1,+1]
    :param nb_rollouts: Confidence of how accurate the value_target is. Based on the number of MCTS simulations.
    :return Returns the joint loss between the value loss and confidence
    """
    mu_transform = (mu + 1) / 2
    alpha = (beta * mu_transform) / (1 - mu_transform)
    value_target_transform = (value_target + 1) / 2
    nb_wins = value_target_transform * nb_rollouts
    nb_losses = nb_rollouts - nb_wins
    return (1/nb_rollouts * (beta_func(alpha, beta).log() - beta_func(alpha+nb_wins, beta+nb_losses).log())).mean()
