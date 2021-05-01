"""
@file: efficient_scaling.py
Created on 30.04.21
@project: CrazyAra
@author: queensgambit

Please describe what the content of this file is about
"""

from __future__ import print_function
import os
import sys
sys.path.insert(0,'../../../')
import glob
import logging
import numpy as np
import mxnet as mx
import pandas as pd
from multiprocessing import Process, Queue

try:
    import mxnet.metric as metric
except ModuleNotFoundError:
    import mxnet.gluon.metric as metrics

from DeepCrazyhouse.src.preprocessing.dataset_loader import load_pgn_dataset
from DeepCrazyhouse.src.domain.neural_net.architectures.rise_mobile_v3 import rise_mobile_v3_symbol
from DeepCrazyhouse.configs.main_config import main_config
from DeepCrazyhouse.configs.train_config import TrainConfig, TrainObjects
from DeepCrazyhouse.src.training.trainer_agent import acc_sign
from DeepCrazyhouse.src.training.trainer_agent_mxnet import TrainerAgentMXNET, get_context
from DeepCrazyhouse.src.training.lr_schedules.lr_schedules import *
from DeepCrazyhouse.src.domain.variants.plane_policy_representation import FLAT_PLANE_IDX
from DeepCrazyhouse.src.domain.variants.constants import NB_POLICY_MAP_CHANNELS

root = logging.getLogger()
root.setLevel(logging.INFO)

results_file = r'grid_search_results.csv'


tc = TrainConfig()
to = TrainObjects()

div_factor = 1
tc.use_mxnet_style = True
tc.val_loss_factor = 0.01
tc.policy_loss_factor = 0.99
tc.batch_steps = 100
tc.nb_training_epochs = 3
tc.q_value_ratio = 0
tc.device_id = 1
tc.sparse_policy_label = True
tc.batch_size = int(1024 / div_factor)
tc.batch_steps = 100 * div_factor  # this defines how often a new checkpoint will be saved and the metrics evaluated
tc.max_lr = 0.35 / div_factor
tc.min_lr = 0.2 / div_factor  # 0.00001

mode = main_config["mode"]
ctx = get_context(tc.context, tc.device_id)
# concatenated at the end of the final feature representation
use_extra_variant_input = False
cur_it = tc.k_steps_initial * tc.batch_steps  # iteration counter used for the momentum and learning rate schedule
# Fixing the random seed
mx.random.seed(tc.seed)

if not os.path.exists(tc.export_dir + "logs"):
    os.mkdir(tc.export_dir + "logs")
if not os.path.exists(tc.export_dir + "weights"):
    os.mkdir(tc.export_dir + "weights")

base_channels = 128
base_depth = 10

alphas = np.arange(1, 2.1, 0.1)

df = pd.DataFrame(columns=['alpha', 'beta', 'depth', 'channels', 'k_steps_best',
                           'val_loss', 'val_value_loss', 'val_policy_loss', 'val_policy_acc', 'val_value_acc_sign',
                           'main_config', 'train_config'])
new_row = {'main_config': str(main_config), 'train_config': tc}
df.to_csv(results_file)

print("main_config:", str(main_config))
print("train_config:", str(tc))


def run_training(alpha, queue):
    s_idcs_val, x_val, yv_val, yp_val, plys_to_end, pgn_datasets_val = load_pgn_dataset(dataset_type='val', part_id=0,
                                                                                        verbose=True,
                                                                                        normalize=tc.normalize)
    if tc.discount != 1:
        yv_val *= tc.discount ** plys_to_end

    if tc.select_policy_from_plane:
        val_iter = mx.io.NDArrayIter({'data': x_val}, {'value_label': yv_val,
                                                       'policy_label': np.array(FLAT_PLANE_IDX)[yp_val.argmax(axis=1)]},
                                     tc.batch_size)
    else:
        val_iter = mx.io.NDArrayIter({'data': x_val}, {'value_label': yv_val, 'policy_label': yp_val.argmax(axis=1)},
                                     tc.batch_size)

    tc.nb_parts = len(glob.glob(main_config['planes_train_dir'] + '**/*'))

    nb_it_per_epoch = (len(x_val) * tc.nb_parts) // tc.batch_size  # calculate how many iterations per epoch exist
    # one iteration is defined by passing 1 batch and doing backprop
    tc.total_it = int(nb_it_per_epoch * tc.nb_training_epochs)

    ### Define a Learning Rate schedule
    to.lr_schedule = OneCycleSchedule(start_lr=tc.max_lr / 8, max_lr=tc.max_lr, cycle_length=tc.total_it * .3,
                                      cooldown_length=tc.total_it * .6, finish_lr=tc.min_lr)
    to.lr_schedule = LinearWarmUp(to.lr_schedule, start_lr=tc.min_lr, length=tc.total_it / 30)

    ### Momentum schedule
    to.momentum_schedule = MomentumSchedule(to.lr_schedule, tc.min_lr, tc.max_lr, tc.min_momentum, tc.max_momentum)
    plot_schedule(to.momentum_schedule, iterations=tc.total_it, ylabel='Momentum')

    input_shape = x_val[0].shape

    beta = np.sqrt(2 / alpha)

    print("alpha:", alpha)
    print("beta:", beta)

    depth = int(round(base_depth * alpha))
    channels = int(round(base_channels * beta))

    kernels = [3] * depth
    se_types = [None] * len(kernels)
    channels_reduced = int(round(channels / 4))

    symbol = rise_mobile_v3_symbol(channels=channels, channels_reduced=channels_reduced, act_type='relu',
                                   channels_value_head=8, value_fc_size=256,
                                   channels_policy_head=NB_POLICY_MAP_CHANNELS,
                                   grad_scale_value=tc.val_loss_factor, grad_scale_policy=tc.policy_loss_factor,
                                   dropout_rate=tc.dropout_rate, select_policy_from_plane=True,
                                   kernels=kernels, se_types=se_types)

    # create a trainable module on compute context
    model = mx.mod.Module(symbol=symbol, context=ctx, label_names=['value_label', 'policy_label'])
    model.bind(for_training=True,
               data_shapes=[('data', (tc.batch_size, input_shape[0], input_shape[1], input_shape[2]))],
               label_shapes=val_iter.provide_label)
    model.init_params(mx.initializer.Xavier(rnd_type='uniform', factor_type='avg', magnitude=2.24))

    metrics_mxnet = [
        metric.MSE(name='value_loss', output_names=['value_output'], label_names=['value_label']),
        metric.CrossEntropy(name='policy_loss', output_names=['policy_output'],
                            label_names=['policy_label']),
        metric.create(acc_sign, name='value_acc_sign', output_names=['value_output'],
                      label_names=['value_label']),
        metric.Accuracy(axis=1, name='policy_acc', output_names=['policy_output'],
                        label_names=['policy_label'])
    ]

    to.metrics = metrics_mxnet
    train_agent = TrainerAgentMXNET(model, symbol, val_iter, tc, to, use_rtpt=True)
    print("model.score(val_iter, to.metrics:", model.score(val_iter, to.metrics))

    # Start the training process
    (k_steps_final, value_loss_final, policy_loss_final, value_acc_sign_final, val_p_acc_final), \
    (k_steps_best, val_metric_values_best) = train_agent.train(cur_it)

    new_row = {'alpha': alpha, 'beta': beta, 'depth': depth, 'channels': channels, 'k_steps_best': k_steps_best,
               'val_loss': val_metric_values_best['loss'], 'val_value_loss': val_metric_values_best['value_loss'],
               'val_policy_loss': val_metric_values_best['policy_loss'],
               'val_policy_acc': val_metric_values_best['policy_acc'],
               'val_value_acc': val_metric_values_best['value_acc_sign']}

    queue.put(new_row)
    print(new_row)


for alpha in alphas:
    queue = Queue()  # start a subprocess to be memory efficient

    process = Process(target=run_training, args=(alpha, queue))
    # export one batch of pgn games
    process.start()
    process.join()  # this blocks until the process terminates
    new_row = queue.get()

    print(">>> write to csv")
    df = df.append(new_row, ignore_index=True)
    df.to_csv(results_file)
