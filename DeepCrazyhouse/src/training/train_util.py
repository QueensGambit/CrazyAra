"""
@file: rl_training.py
Created on 23.06.22
@project: CrazyAra
@author: queensgambit

Utility methods for training with one of the training agents.
"""

import numpy as np
from DeepCrazyhouse.src.domain.variants.plane_policy_representation import FLAT_PLANE_IDX
from DeepCrazyhouse.src.domain.variants.constants import NB_LABELS_POLICY_MAP


def get_metrics(train_config):
    """
    Returns the metrics according to the used training framework.
    :param train_config: Training configuration object
    :return: Training metrics
    """
    if train_config.framework == 'mxnet':
        return _get_mxnet_metrics(train_config)
    if train_config.framework == 'gluon':
        return _get_gluon_metrics(train_config)
    if train_config.framework == 'pytorch':
        return _get_pytorch_metrics(train_config)


def _get_mxnet_metrics(train_config):
    try:
        import mxnet.metric as metric
    except ModuleNotFoundError:
        import mxnet.gluon.metric as metric
    from DeepCrazyhouse.src.training.trainer_agent_gluon import acc_sign
    metrics_mxnet = [
        metric.MSE(name='value_loss', output_names=['value_output'], label_names=['value_label']),
        metric.CrossEntropy(name='policy_loss', output_names=['policy_output'],
                            label_names=['policy_label']),
        metric.create(acc_sign, name='value_acc_sign', output_names=['value_output'],
                      label_names=['value_label']),
        metric.Accuracy(axis=1, name='policy_acc', output_names=['policy_output'],
                        label_names=['policy_label'])
    ]
    if train_config.use_wdl:
        metrics_mxnet.append(metric.CrossEntropy(name='wdl_loss',
                                                 output_names=['wdl_output'], label_names=['wdl_label']))
        metrics_mxnet.append(metric.Accuracy(axis=1, name='wdl_acc', output_names=['wdl_output'],
                                             label_names=['wdl_label']))
    if train_config.use_plys_to_end:
        metrics_mxnet.append(metric.MSE(name='plys_to_end_loss', output_names=['plys_to_end_output'],
                                        label_names=['plys_to_end_label']))
    return metrics_mxnet


def _get_gluon_metrics(train_config):
    try:
        import mxnet.metric as metric
    except ModuleNotFoundError:
        import mxnet.gluon.metric as metric
    from DeepCrazyhouse.src.training.trainer_agent_gluon import acc_sign, cross_entropy, acc_distribution
    metrics_gluon = {
        'value_loss': metric.MSE(name='value_loss', output_names=['value_output']),

        'value_acc_sign': metric.create(acc_sign, name='value_acc_sign', output_names=['value_output'],
                                        label_names=['value_label']),
    }
    if train_config.sparse_policy_label:
        # the default cross entropy only supports sparse labels
        metrics_gluon['policy_loss'] = metric.CrossEntropy(name='policy_loss', output_names=['policy_output'],
                                                           label_names=['policy_label']),
        metrics_gluon['policy_acc'] = metric.Accuracy(axis=1, name='policy_acc', output_names=['policy_output'],
                                                      label_names=['policy_label'])
    else:
        metrics_gluon['policy_loss'] = metric.create(cross_entropy, name='policy_loss', output_names=['policy_output'],
                                                     label_names=['policy_label'])
        metrics_gluon['policy_acc'] = metric.create(acc_distribution, name='policy_acc', output_names=['policy_output'],
                                                    label_names=['policy_label'])
    return metrics_gluon


def _get_pytorch_metrics(train_config):
    import DeepCrazyhouse.src.training.metrics_pytorch as pytorch_metrics
    metrics_pytorch = {
        'value_loss': pytorch_metrics.MSE(),
        'policy_loss': pytorch_metrics.CrossEntropy(train_config.sparse_policy_label),
        'value_acc_sign': pytorch_metrics.AccuracySign(),
        'policy_acc': pytorch_metrics.Accuracy(train_config.sparse_policy_label)
    }
    if train_config.use_wdl:
        metrics_pytorch['wdl_loss'] = pytorch_metrics.CrossEntropy(True)
        metrics_pytorch['wdl_acc'] = pytorch_metrics.Accuracy(True)
    if train_config.use_plys_to_end:
        metrics_pytorch['plys_to_end_loss'] = pytorch_metrics.MSE()

    return metrics_pytorch


def prepare_policy(y_policy, select_policy_from_plane, sparse_policy_label, is_policy_from_plane_data):
    """
    Modifies the layout of the policy vector in place according to the given definitions
    :param y_policy: Target policy vector
    :param select_policy_from_plane: If policy map representation shall be applied
    :param sparse_policy_label: True, if the labels are sparse (one-hot-encoded)
    :param is_policy_from_plane_data: True, if the policy representation is already in
     "select_policy_from_plane" representation
    :return: modified y_policy
    """
    if sparse_policy_label:
        y_policy = y_policy.argmax(axis=1)

        if select_policy_from_plane:
            y_policy[:] = FLAT_PLANE_IDX[y_policy]
    else:
        if select_policy_from_plane and not is_policy_from_plane_data:
            tmp = np.zeros((len(y_policy), NB_LABELS_POLICY_MAP), np.float32)
            tmp[:, FLAT_PLANE_IDX] = y_policy[:, :]
            y_policy = tmp
    return y_policy


def return_metrics_and_stop_training(k_steps, val_metric_values, k_steps_best, val_metric_values_best):
    return (k_steps,
            val_metric_values["value_loss"], val_metric_values["policy_loss"],
            val_metric_values["value_acc_sign"], val_metric_values["policy_acc"]), \
           (k_steps_best, val_metric_values_best)


def value_to_wdl_label(y_value):
    return y_value + 1


def prepare_plys_label(plys_to_end_label):
    return np.clip(plys_to_end_label, 0, 100) / 100
