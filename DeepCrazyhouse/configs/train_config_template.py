"""
@file: train_config.py
Created on 01.11.19
@project: CrazyAra
@author: queensgambit

Training configuration file
"""
from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Class which stores all training configuration"""

    # div factor is a constant which can be used to reduce the batch size and learning rate respectively
    # use a value higher 1 if you encounter memory allocation errors
    div_factor: int = 2

    # 1024 # the batch_size needed to be reduced to 1024 in order to fit in the GPU 1080Ti
    # 4096 was originally used in the paper -> works slower for current GPU
    # 2048 was used in the paper Mastering the game of Go without human knowledge and fits in GPU memory
    # typically if you half the batch_size you should double the lr
    batch_size: int = int(1024 / div_factor)

    # batch_steps = 1000 means for example that every 1000 batches the validation set gets processed
    batch_steps: int = 100 * div_factor  # this defines how often a new checkpoint will be saved and the metrics evaluated

    # set the context on CPU switch to GPU if there is one available (strongly recommended for training)
    context: str = "gpu"

    cpu_count: int = 4

    #  Current working directory (must end with "/"). If None it calls os.getcwd()
    cwd: str = None

    device_id: int = 0

    discount: float = 1.0

    dropout_rate: float = 0

    export_weights: bool = True

    export_grad_histograms: bool = True

    # Boolean if the policy data is also defined in select_policy_from_plane representation
    is_policy_from_plane_data: bool = False

    log_metrics_to_tensorboard: bool = True

    # k_steps_initial defines how many steps have been trained before
    # (k_steps_initial != 0 if you continue training from a checkpoint)
    k_steps_initial: int = 0  # 498
    # these are the weights to continue training with
    # symbol_file = 'model_init-symbol.json' # model-1.19246-0.603-symbol.json'
    # params_file = 'model_init-0000.params' # model-1.19246-0.603-0223.params'
    symbol_file: str = ''
    params_file: str = ''

    # # optimization parameters
    optimizer_name: str = "nag"
    max_lr: float = 0.1 / div_factor  # 0.35 / div_factor
    min_lr: float = 0.00001 / div_factor  # 0.2 / div_factor  # 0.00001
    max_momentum: float = 0.95
    min_momentum: float = 0.8
    # stop training as soon as max_spikes has been reached
    max_spikes: int = 20

    # name initials which are used to identify running training processes with rtpt
    # prefix for the process name in order to identify the process on a server
    name_initials: str = "JC"

    nb_parts: int = None

    normalize: bool = True  # define whether to normalize input data to [01]
    nb_epochs: str = 1  # define how many epochs the network will be trained

    policy_loss_factor: float = 1  # 0.99

    # ratio for mixing the value return with the corresponding q-value
    # for a ratio of 0 no q-value information will be used
    q_value_ratio: float = 0.15

    # set a specific seed value for reproducibility
    seed: int = 42
    select_policy_from_plane: bool = True  # Boolean if potential legal moves will be selected from final policy output

    # define spike threshold when the detection will be triggered
    spike_thresh: float = 1.5

    # Boolean if the policy target is one-hot encoded (sparse=True) or a target distribution (sparse=False)
    sparse_policy_label: bool = False

    # total of training iterations
    total_it: int = None

    use_mxnet_style = True  # Decide between mxnet and gluon style for training

    # loads a previous checkpoint if the loss increased significantly
    use_spike_recovery: bool = True
    # weight the value loss a lot lower than the policy loss in order to prevent overfitting
    val_loss_factor: float = 0.5  # 0.01

    # weight decay
    wd: float = 1e-4


@dataclass
class TrainObjects:
    """Defines training objects which must be set before the training"""
    lr_schedule = None  # learning rate schedule
    momentum_schedule = None
    metrics = None
    variant_metrics = None
