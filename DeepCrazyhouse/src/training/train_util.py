"""
@file: rl_training.py
Created on 23.06.22
@project: CrazyAra
@author: queensgambit

Utility methods for training with one of the training agents.
"""
try:
    import mxnet.metric as metric
except ModuleNotFoundError:
    import mxnet.gluon.metric as metric
from DeepCrazyhouse.src.training.trainer_agent_gluon import acc_sign, cross_entropy, acc_distribution


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
