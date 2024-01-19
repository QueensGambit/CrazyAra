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
    """Class which stores all training configurations"""

    info_div_factor: str = "div factor is a constant which can be used to reduce the batch size and learning rate" \
                           " respectively use a value higher 1 if you encounter memory allocation errors"
    div_factor: int = 1

    info_batch_size: str = "batch size used during training. The batch-size may need to be reduced in order to fit on" \
                           " your GPU memory. 4096 was originally used in the paper, 2048 was used in the paper" \
                           " 'Mastering the game of Go without human knowledge'. Typically if you half the batch_size" \
                           " you should double the learning rate."
    batch_size: int = int(1024 / div_factor)

    info_batch_steps: str = "batch_steps = 1000 means for example that every 1000 batches the validation set is" \
                            " processed. It defines how often a new checkpoint will be saved and the metrics evaluated"
    batch_steps: int = 1000 * div_factor

    info_context: str = "context defines the computation device to use for training. Set the context to to 'gpu' if" \
                        " there is one available, otherwise you may train on 'cpu' instead."
    context: str = "gpu"

    info_cpu_count: str = "cpu_count defines the number of cpu cores to use for data processing while training."
    cpu_count: int = 4

    info_device_id: str = "device_id sets the GPU device to use for training."
    device_id: int = 0

    info_discount: str = "discount describes the discounting value to use for discounting the value target " \
                         "until reaching the final terminal value."
    discount: float = 1.0

    info_dropout_rate: str = "dropout_rate describes the drobout percentage as used in the neural network architecture."
    dropout_rate: float = 0

    info_export_dir: str = "export_dir sets the directory to write and read weights, log, onnx and other export logging" \
                           " files"
    export_dir: str = "./"

    info_export_weights: str = "export_weights is a boolean to decide if the neural network weights should be exported" \
                               "during training."
    export_weights: bool = True

    info_export_grad_histograms: str = "export_grad_histograms enables or disable the export of gradient diagrams " \
                                       "during training."
    export_grad_histograms: bool = True

    info_framework: str = "framework sets the deep learning framework to use. Currently only 'pytorch' is available." \
                     "mxnet and gluon have been deprecated."
    framework: str = 'pytorch'

    info_is_policy_from_plane_data: str = "is_policy_from_plane_data is a boolean to decide if the policy data is" \
                                          " already defined in select_policy_from_plane / plane representation."
    is_policy_from_plane_data: bool = False

    info_log_metrics_to_tensorboard: str = "log_metrics_to_tensorboard decides if the metrics should be exported with" \
                                           "tensorboard."
    log_metrics_to_tensorboard: bool = True

    info_model_type: str = "model_type defines the Model type that used during training (e.g. resnet, vit, risev2," \
                           " risev3, alphavile, alphavile-tiny, alphavile-small, alphavile-normal, alphavile-large," \
                           " NextViT)"
    model_type: str = "resnet"

    info_k_steps_initial: str = "k_steps_initial defines how many steps have been trained before (k_steps_initial != 0 if" \
                           " you continue training from a checkpoint)" \
                           " (TODO: Continuing training from a previous checkpoint is currently not available in" \
                           " pytorch training loop.)"
    k_steps_initial: int = 0

    info_symbol_file: str = "symbol_file is the neural network architecture file to continue training with (deprecated)" \
                            "(e.g. 'model_init-symbol.json', model-1.19246-0.603-symbol.json')"
    symbol_file: str = ''
    info_params_file: str = "params_file is the neural network weight file to continue training with (deprecated)" \
                            "(e.g. 'model_init-0000.params' # model-1.19246-0.603-0223.params')"
    params_file: str = ''

    info_optimizer_name: str = "optimizer_name is the optimizer that used in the training loop to update the weights." \
                               "(e.g. nag, sgd, adam, adamw)"
    optimizer_name: str = "nag"

    info_max_lr: str = "max_lr defines the maximum learning rate used for training."
    max_lr: float = 0.07 / div_factor
    info_min_lr: str = "min_lr defines the minimum learning rate used for training."
    min_lr: float = 0.00001 / div_factor

    if "adam" in optimizer_name:
        max_lr = 0.001001  # 1e-3
        min_lr = 0.001

    info_max_momentum: str = "max_momentum defines the maximum momentum factor used during training (only applicable to" \
                             "optimizers that are momentum based)"
    max_momentum: float = 0.95
    info_min_momentum: str = "min_momentum defines the minimum momentum factor used during training (only applicable to" \
                             "optimizers that are momentum based)"
    min_momentum: float = 0.8

    info_max_spikes: str = "max_spikes defines the maximum number of spikes. Training is stopped as soon as max_spikes" \
                           " has been reached."
    max_spikes: int = 20

    # name initials which are used to identify running training processes with rtpt
    # prefix for the process name in order to identify the process on a server
    info_name_initials: str = "name_initials sets the name initials which are used to identify running training" \
                              " processes with rtpt. It is used as a prefix for the process name in order to identify" \
                              " the process on a server."
    name_initials: str = "XX"

    info_nb_parts: str = "nb_parts sets the number of training zip files used for training. This value is normally " \
                         "dynamically set before training based on the number of .zip files available in the training " \
                         "directory."
    nb_parts: int = None

    info_normalize: str = "normalize decides if the training data should be normalized to the range of [0,1]."
    normalize: bool = True  # define whether to normalize input data to [01]

    info_nb_training_epochs: str = "nb_training_epochs defines how many epoch iterations the network will be trained."
    nb_training_epochs: int = 7

    info_plys_to_end_loss_factor: str = "plys_to_end_loss_factor defines the gradient scaling for the plys to end" \
                                        " output."
    plys_to_end_loss_factor: float = 0.002

    info_q_value_ratio: str = "q_value_ratio defines the ratio for mixing the value return with the corresponding " \
                              "q-value for a ratio of 0 no q-value information will be used."
    q_value_ratio: float = 0.0

    info_seed: str = "seed sets a specific seed value for reproducibility."
    seed: int = 42

    info_select_policy_from_plane: str = "select_policy_from_plan defines if potential legal moves will be selected" \
                                         " from final policy output in plane representation / convolution " \
                                         "representation rather than a flat representation."
    select_policy_from_plane: bool = True

    info_spike_thresh: str = "spike_thresh defines the spike threshold when the detection will be triggered. It is" \
                             " triggered when last_loss x spike_thresh < current_loss."
    spike_thresh: float = 1.5

    info_sparse_policy_label: str = "sparse_policy_label defines if the policy target is one-hot encoded (sparse=True)" \
                                    " or a target distribution (sparse=False)"
    sparse_policy_label: bool = True

    info_total_it: str = "total_it defines the total number of training iterations. Usually this value is determined by" \
                         "dynamically based on the number of zip files and the number of samples in the validation file."
    total_it: int = None

    info_use_custom_architecture: str = "use_custom_architecture decides if a custom network architecture should be " \
                                        "used, defined in the model_config.py file"
    use_custom_architecture: bool = False

    info_use_mlp_wdl_ply: str = "use_mlp_wdl_ply adds a small mlp to infer the value loss from wdl and plys_to_end" \
                                "_output"
    use_mlp_wdl_ply: bool = False
    info_use_plys_to_end: str = "use_plys_to_end enables training with the plys to end head."
    use_plys_to_end: bool = True
    info_use_wdl: str = "use_wdl enables training with a wdl head as intermediate target (mainly useful for" \
                        " environments with three outcomes WIN, DRAW, LOSS)"
    use_wdl: bool = True

    info_use_spike_recovery: str = "use_spike_recovery loads a previous checkpoint if the loss increased significantly."
    use_spike_recovery: bool = True
    info_val_loss_factor: str = "val_loss_factor weights the value loss a lot lower than the policy loss in order to" \
                                " prevent overfitting"
    val_loss_factor: float = 0.01
    info_policy_loss_factor: str = "policy_loss_factor defines the weighting factor for the policy loss."
    policy_loss_factor: float = 0.988 if use_plys_to_end else 0.99

    info_wdl_loss_factor: str = "wdl_loss_factor defines the weighting factor for the wdl-loss."
    wdl_loss_factor: float = 0.01

    info_wd: str = "wd defines the weight decay value for regularization as a measure to prevent overfitting."
    wd: float = 1e-4


def rl_train_config():
    tc = TrainConfig()

    tc.export_grad_histograms = True
    tc.div_factor = 2
    tc.batch_steps = 100 * tc.div_factor
    tc.batch_size = int(1024 / tc.div_factor)

    tc.max_lr = 0.1 / tc.div_factor
    tc.min_lr = 0.00001 / tc.div_factor

    tc.val_loss_factor = 0.499 if tc.use_plys_to_end else 0.5
    tc.policy_loss_factor = 0.499 if tc.use_plys_to_end else 0.5
    tc.plys_to_end_loss_factor = 0.002
    tc.wdl_loss_factor = 0.499 if tc.use_plys_to_end else 0.5

    tc.nb_training_epochs = 1  # define how many epochs the network will be trained
    tc.q_value_ratio = 0.15
    tc.sparse_policy_label = False

    return tc


@dataclass
class TrainObjects:
    """Defines training objects which must be set before the training"""
    lr_schedule = None  # learning rate schedule
    momentum_schedule = None
    metrics = None
    variant_metrics = None

