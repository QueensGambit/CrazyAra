"""
@file: rl_training.py
Created on 20.11.19
@project: CrazyAra
@author: queensgambit

Functionality for conducting a single NN update within the reinforcement learning loop
"""

import sys
import glob
import logging
import mxnet as mx

sys.path.append("../../../")
from DeepCrazyhouse.configs.main_config import main_config
from DeepCrazyhouse.configs.train_config import train_config
from DeepCrazyhouse.src.preprocessing.dataset_loader import load_pgn_dataset
from DeepCrazyhouse.src.training.trainer_agent import acc_sign, cross_entropy, acc_distribution
from DeepCrazyhouse.src.training.trainer_agent_mxnet import TrainerAgentMXNET, add_non_sparse_cross_entropy,\
    remove_no_sparse_cross_entropy, prepare_policy
from DeepCrazyhouse.src.training.lr_schedules.lr_schedules import MomentumSchedule, LinearWarmUp,\
    CosineAnnealingSchedule


def update_network(queue, nn_update_idx, k_steps_initial, max_lr, symbol_filename, params_filename, cwd):
    """
    Creates a new NN checkpoint in the model contender directory after training using the game files stored in the
     training directory
    :param queue: Queue object used to return items
    :param k_steps_initial: Initial amount of steps of the NN update
    :param nn_update_idx: Defines how many updates of the nn has already been done. This index should be incremented
    after every update.
    :param max_lr: Maximum learning rate used for the learning rate schedule
    :param symbol_filename: Architecture definition file
    :param params_filename: Weight file which will be loaded before training
    Updates the neural network with the newly acquired games from the replay memory
    :param cwd: Current working directory (must end with "/")
    :return: k_steps_final
    """

    # set the context on CPU, switch to GPU if there is one available (strongly recommended for training)
    ctx = mx.gpu(train_config["device_id"]) if train_config["context"] == "gpu" else mx.cpu()
    # set a specific seed value for reproducibility

    # Fixing the random seed
    mx.random.seed(train_config["seed"])

    nb_parts = len(glob.glob(main_config["planes_train_dir"] + '**/*.zip'))
    logging.info("number parts: %d" % nb_parts)

    if nb_parts <= 0:
        raise Exception('No .zip files for training available. Check the path in main_config["planes_train_dir"]:'
                        ' %s' % main_config["planes_train_dir"])

    _, x_val, y_val_value, y_val_policy, _, _ = load_pgn_dataset(dataset_type="val",
                                                                 part_id=0,
                                                                 normalize=train_config["normalize"],
                                                                 verbose=False,
                                                                 q_value_ratio=train_config["q_value_ratio"])

    y_val_policy = prepare_policy(y_val_policy, train_config["select_policy_from_plane"],
                                  train_config["sparse_policy_label"])

    symbol = mx.sym.load(symbol_filename)
    symbol = add_non_sparse_cross_entropy(symbol, train_config["val_loss_factor"],
                                          "value_tanh0_output", "flatten0_output")
    # "value_out_output", "policy_out_output")

    # calculate how many iterations per epoch exist
    nb_it_per_epoch = (len(x_val) * nb_parts) // train_config["batch_size"]
    # one iteration is defined by passing 1 batch and doing backprop
    total_it = int(nb_it_per_epoch * train_config["nb_epochs"])

    lr_schedule = CosineAnnealingSchedule(train_config["min_lr"], max_lr, total_it * .7)
    lr_schedule = LinearWarmUp(lr_schedule, start_lr=train_config["min_lr"], length=total_it * .25)
    momentum_schedule = MomentumSchedule(lr_schedule, train_config["min_lr"], max_lr,
                                         train_config["min_momentum"], train_config["max_momentum"])

    if train_config["select_policy_from_plane"]:
        val_iter = mx.io.NDArrayIter({'data': x_val}, {'value_label': y_val_value,
                                                       'policy_label': y_val_policy}, train_config["batch_size"])
    else:
        val_iter = mx.io.NDArrayIter({'data': x_val},
                                     {'value_label': y_val_value, 'policy_label': y_val_policy},
                                     train_config["batch_size"])

    # calculate how many iterations per epoch exist
    nb_it_per_epoch = (len(x_val) * nb_parts) // train_config["batch_size"]
    # one iteration is defined by passing 1 batch and doing backprop
    total_it = int(nb_it_per_epoch * train_config["nb_epochs"])

    input_shape = x_val[0].shape
    model = mx.mod.Module(symbol=symbol, context=ctx, label_names=['value_label', 'policy_label'])
    # mx.viz.print_summary(
    #     symbol,
    #     shape={'data': (1, input_shape[0], input_shape[1], input_shape[2])},
    # )
    model.bind(for_training=True,
               data_shapes=[('data', (train_config["batch_size"], input_shape[0], input_shape[1], input_shape[2]))],
               label_shapes=val_iter.provide_label)
    model.load_params(params_filename)

    metrics = [
        mx.metric.MSE(name='value_loss', output_names=['value_output'], label_names=['value_label']),
        mx.metric.create(acc_sign, name='value_acc_sign', output_names=['value_output'],
                         label_names=['value_label']),
        mx.metric.create(acc_distribution, name='policy_acc', output_names=['policy_output'],
                         label_names=['policy_label']),
        # mx.metric.Accuracy(axis=1, name='policy_acc', output_names=['policy_output'],
        #                    label_names=['policy_label'])
    ]

    if train_config["sparse_policy_label"]:
        # the default cross entropy only supports sparse lables
        metrics.append(mx.metric.CrossEntropy(name='policy_loss', output_names=['policy_output'],
                                              label_names=['policy_label']))
    else:
        metrics.append(mx.metric.create(cross_entropy, name='policy_loss', output_names=['policy_output'],
                                        label_names=['policy_label']))

    logging.info("Performance pre training")
    logging.info(model.score(val_iter, metrics))

    train_agent = TrainerAgentMXNET(model, symbol, val_iter, nb_parts, lr_schedule, momentum_schedule, total_it,
                                    train_config["optimizer_name"], wd=train_config["wd"],
                                    batch_steps=train_config["batch_steps"],
                                    k_steps_initial=k_steps_initial,
                                    cpu_count=train_config["cpu_count"],
                                    batch_size=train_config["batch_size"], normalize=train_config["normalize"],
                                    export_weights=train_config["export_weights"],
                                    export_grad_histograms=train_config["export_grad_histograms"],
                                    log_metrics_to_tensorboard=train_config["log_metrics_to_tensorboard"], ctx=ctx,
                                    metrics=metrics, use_spike_recovery=train_config["use_spike_recovery"],
                                    max_spikes=train_config["max_spikes"],
                                    spike_thresh=train_config["spike_thresh"],
                                    seed=train_config["seed"], val_loss_factor=train_config["val_loss_factor"],
                                    policy_loss_factor=train_config["policy_loss_factor"],
                                    select_policy_from_plane=train_config["select_policy_from_plane"],
                                    discount=train_config["discount"],
                                    sparse_policy_label=train_config["sparse_policy_label"],
                                    q_value_ratio=train_config["q_value_ratio"],
                                    cwd=cwd)
    # iteration counter used for the momentum and learning rate schedule
    cur_it = train_config["k_steps_initial"] * train_config["batch_steps"]
    (k_steps_final, val_value_loss_final, val_policy_loss_final, val_value_acc_sign_final,
     val_policy_acc_final), _ = train_agent.train(cur_it)

    symbol = remove_no_sparse_cross_entropy(symbol, train_config["val_loss_factor"],
                                            "value_tanh0_output", "flatten0_output")
    prefix = cwd + "model_contender/model-%.5f-%.5f-%.3f-%.3f" % (val_value_loss_final, val_policy_loss_final,
                                                                  val_value_acc_sign_final, val_policy_acc_final)

    symbol.save(prefix + "-symbol.json")
    model.save_params(prefix + "-" + "%04d.params" % nn_update_idx)

    logging.info("k_steps_final %d" % k_steps_final)
    queue.put(k_steps_final)
