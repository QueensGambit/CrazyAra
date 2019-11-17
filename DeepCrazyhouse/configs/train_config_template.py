"""
@file: train_config.py
Created on 01.11.19
@project: CrazyAra
@author: queensgambit

Training configuration file
"""

#  div factor is a constant which can be used to reduce the batch size and learning rate respectively
# use a value higher 1 if you encounter memory allocation errors
div_factor = 1

train_config = {
    # set the context on CPU, switch to GPU if there is one available (strongly recommended for training)
    "context": "gpu",
    "cpu_count": 8,
    "device_id": 0,
    # set a specific seed value for reproducibility
    "seed": 42,

    "export_weights": True,
    "log_metrics_to_tensorboard": True,
    "export_grad_histograms": True,
    # batch_steps = 1000 means for example that every 1000 batches the validation set gets processed
    "batch_steps": 1 * div_factor,  # this defines how often a new checkpoint will be saved and the metrics evaluated
    # k_steps_initial defines how many steps have been trained before
    # (k_steps_initial != 0 if you continue training from a checkpoint)
    "k_steps_initial": 0,  # 498
    # these are the weights to continue training with
    # symbol_file = 'model_init-symbol.json' #model-1.19246-0.603-symbol.json'
    # params_file = 'model_init-0000.params' #model-1.19246-0.603-0223.params'
    "symbol_file": 'model-1.19246-0.603-symbol.json',
    "params_file": 'model-1.19246-0.603-0223.params',

    "batch_size": int(1024 / div_factor),
    # 1024 # the batch_size needed to be reduced to 1024 in order to fit in the GPU 1080Ti
    # 4096 was originally used in the paper -> works slower for current GPU
    # 2048 was used in the paper Mastering the game of Go without human knowledge and fits in GPU memory
    # typically if you half the batch_size, you should double the lr
    #
    # # optimization parameters
    "optimizer_name": "nag",
    "max_lr": 0.1 / div_factor,  # 0.35 / div_factor
    "min_lr": 0.00001 / div_factor,  # 0.2 / div_factor  # 0.00001
    "max_momentum": 0.95,
    "min_momentum": 0.8,
    # loads a previous checkpoint if the loss increased significantly
    "use_spike_recovery": True,
    # stop training as soon as max_spikes has been reached
    "max_spikes": 20,
    # define spike threshold when the detection will be triggered
    "spike_thresh": 1.5,
    # weight decay
    "wd": 1e-4,
    # dropout_rate = 0  # 0.2
    # weight the value loss a lot lower than the policy loss in order to prevent overfitting
    "val_loss_factor": 1,  # 0.01
    "policy_loss_factor": 1,  # 0.99
    # ratio for mixing the value return with the corresponding q-value
    # for a ratio of 0 no q-value information will be used
    "q_value_ratio": 0.5,
    "discount": 1.0,

    "normalize": True,  # define whether to normalize input data to [0,1]
    "nb_epochs": 7,  # define how many epochs the network will be trained

    "select_policy_from_plane": True,  # Boolean if potential legal moves will be selected from final policy output
    # Boolean if the policy target is one-hot encoded (sparse=True) or a target distribution (sparse=False)
    "sparse_policy_label": False,
    # use_mxnet_style = True  # Decide between mxnet and gluon style for training
}
