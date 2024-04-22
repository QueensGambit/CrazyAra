#!/usr/bin/env python
# coding: utf-8

# ## Training script for the CNN 
# 
# Loads in the converted plane representation of the pgn files, defines the network architecture and starts the training process. Checkpoints of the weights are saved if there's an improvement in the validation loss.
# The training performance metrics (e.g. losses, accuracies...) are exported to tensorboard and can be checked during training.
# * author: QueensGambit

# In[ ]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


#get_ipython().run_line_magic('reload_ext', 'autoreload')


# In[ ]:


from __future__ import print_function
import os
import sys
sys.path.insert(0,'../../../')
import glob
import chess
import shutil
import logging
import numpy as np
from pathlib import Path
from copy import deepcopy

from DeepCrazyhouse.src.training.train_util import get_metrics, prepare_policy, value_to_wdl_label, prepare_plys_label
from DeepCrazyhouse.src.domain.variants.input_representation import board_to_planes, planes_to_board
from DeepCrazyhouse.src.domain.variants.output_representation import policy_to_moves, policy_to_best_move, policy_to_move
from DeepCrazyhouse.src.preprocessing.dataset_loader import load_pgn_dataset
from DeepCrazyhouse.src.runtime.color_logger import enable_color_logging
from DeepCrazyhouse.configs.main_config import main_config
from DeepCrazyhouse.configs.train_config import TrainConfig, TrainObjects

from DeepCrazyhouse.src.training.lr_schedules.lr_schedules import *
from DeepCrazyhouse.src.domain.variants.plane_policy_representation import FLAT_PLANE_IDX
from DeepCrazyhouse.src.domain.variants.constants import NB_POLICY_MAP_CHANNELS, NB_LABELS, MODE_CHESS, MODE_CRAZYHOUSE
enable_color_logging()
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


tc = TrainConfig()
to = TrainObjects()
# Decide between 'pytorch', 'mxnet' and 'gluon' style for training
tc.framework: str = 'pytorch'


# ### pytorch imports

# In[ ]:


if tc.framework == 'pytorch':
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from torchsummary import summary
    from fvcore.nn import FlopCountAnalysis

    from DeepCrazyhouse.src.training.trainer_agent_pytorch import TrainerAgentPytorch, save_torch_state,\
    load_torch_state, export_to_onnx, get_context, get_data_loader, evaluate_metrics
    # architectures
    from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.rise_mobile_v3 import RiseV3, get_rise_v33_model_by_train_config
    from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.vision_transformer import VisionTransformer
    from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.vit_configs import get_b8_config
    from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.le_vit import LeViT
    from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.mobile_vit import MobileViT
    from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.trt_vit import TrtViT
    from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.next_vit_official import NextVit
    from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.a0_resnet import AlphaZeroResnet


# ### mxnet imports

# In[ ]:


if tc.framework == 'mxnet' or tc.framework == 'gluon':
    import mxnet as mx
    from mxnet import nd
    from mxnet import gluon
    try:
        import mxnet.metric as metric
    except ModuleNotFoundError:
        import mxnet.gluon.metric as metrics

    from DeepCrazyhouse.src.training.trainer_agent_gluon import TrainerAgentGluon, evaluate_metrics, acc_sign, get_data_loader
    from DeepCrazyhouse.src.training.trainer_agent_mxnet import TrainerAgentMXNET, get_context
    # architectures
    from DeepCrazyhouse.src.domain.neural_net.architectures.a0_resnet import AlphaZeroResnet
    from DeepCrazyhouse.src.domain.neural_net.architectures.mxnet_alpha_zero import alpha_zero_symbol
    from DeepCrazyhouse.src.domain.neural_net.architectures.rise_mobile_v2 import rise_mobile_v2_symbol
    from DeepCrazyhouse.src.domain.neural_net.architectures.rise_mobile_v3 import rise_mobile_v3_symbol
    from DeepCrazyhouse.src.domain.neural_net.architectures.preact_resnet_se import preact_resnet_se
    from DeepCrazyhouse.src.domain.neural_net.onnx.convert_to_onnx import convert_mxnet_model_to_onnx


# ## Settings

# In[ ]:


# set the context on CPU, switch to GPU if there is one available (strongly recommended for training)
tc.context = "gpu"
tc.device_id = 0

# set a specific seed value for reproducibility
tc.seed = 9 # 42

tc.export_weights = True
tc.log_metrics_to_tensorboard = True
tc.export_grad_histograms = False

phase_weights = {0: 1.0, 1: 1.0, 2: 1.0}  # specify the sample weight for each phase (will be normalized afterwards)
if "movecount" in main_config["phase_definition"]:
    assert len(phase_weights) == int(main_config["phase_definition"][-1])

# directory to write and read weights, logs, onnx and other export files
#tc.export_dir = "C:/workspace/Python/CrazyAra/data/train_phase2/"
tc.export_dir = f"/data/run_model_exports_movecount/movecount4_train_phase_0/"

tc.div_factor = 0.5  # div factor is a constant which can be used to reduce the batch size and learning rate respectively
# use a value greater 1 if you encounter memory allocation errors

# batch_steps = 1000 means for example that every 1000 batches the validation set gets processed
tc.batch_steps = 1000 * tc.div_factor # this defines how often a new checkpoint will be saved and the metrics evaluated
# k_steps_initial defines how many steps have been trained before
# (k_steps_initial != 0 if you continue training from a checkpoint)
tc.k_steps_initial = 0
# these are the weights to continue training with
tc.symbol_file = None # 'model-0.81901-0.713-symbol.json'
tc.tar_file = None # f"/data/run_model_exports/train_phase_None_0_25_0_25_1_0/best-model/model-1.25307-0.567-0529.tar" #'model-0.81901-0.713-0498.params'  # used to continue training from model params checkpoint

tc.batch_size = int(1024 / tc.div_factor) # 1024 # the batch_size needed to be reduced to 1024 in order to fit in the GPU 1080Ti
#4096 was originally used in the paper -> works slower for current GPU
# 2048 was used in the paper Mastering the game of Go without human knowledge and fits in GPU memory
#typically if you half the batch_size, you should double the lr

# optimization parameters
tc.optimizer_name = "nag"  # "adam" "adamw" # 
if tc.framework == 'pytorch':
    # strangely pytorch should use a different lr than mxnet
    tc.max_lr = 0.07 / tc.div_factor
else:
    tc.max_lr = 0.35 / tc.div_factor #0.01 # default lr for adam
tc.min_lr = 0.00001

if "adam" in tc.optimizer_name:
    tc.max_lr = 0.001001 #1e-3
    tc.min_lr = 0.001
    
tc.max_momentum = 0.95
tc.min_momentum = 0.8
# loads a previous checkpoint if the loss increased significanly
tc.use_spike_recovery = True
# stop training as soon as max_spikes has been reached
tc.max_spikes = 20
# define spike threshold when the detection will be triggered
tc.spike_thresh = 1.5
# weight decay
tc.wd = 1e-4
tc.dropout_rate = 0 #0.15

# enables training with a wdl head as intermediate target (mainly useful for environments with 3 outcomes)
tc.use_wdl = True
# enables training with ply to end head
tc.use_plys_to_end = True
# adds a small mlp to infer the value loss from wdl and plys_to_end_output
tc.use_mlp_wdl_ply = False

# weight the value loss a lot lower than the policy loss in order to prevent overfitting
tc.val_loss_factor = 0.01
tc.policy_loss_factor = 0.988 if tc.use_plys_to_end else 0.99
tc.plys_to_end_loss_factor = 0.002
tc.wdl_loss_factor = 0.01
tc.discount = 1.0

tc.normalize = True # define whether to normalize input data to [0,1]
tc.nb_training_epochs = 7 # define how many epochs the network will be trained
tc.select_policy_from_plane = True # Boolean if potential legal moves will be selected from final policy output
        
# additional custom validation set files which will be logged to tensorboard
to.variant_metrics = None # ["chess960", "koth", "three_check"]
# if use_extra_variant_input is true the current active variant is passed two each residual block and

# ratio for mixing the value return with the corresponding q-value
# for a ratio of 0 no q-value information will be used
tc.q_value_ratio = 0

# define if policy training target is one-hot encoded a distribution (e.g. mcts samples, knowledge distillation)
tc.sparse_policy_label = True
# define if the policy data is also defined in "select_policy_from_plane" representation
tc.is_policy_from_plane_data = False
tc.name_initials = "FH"


# In[ ]:


phase_weights_sum = sum(phase_weights.values())
to.phase_weights = {k: v/phase_weights_sum*len(phase_weights) for k, v in phase_weights.items()}  # normalize so that the average weight is 1.0 (assuming each phase occurs approximately equally often)
mode = main_config["mode"]
ctx = get_context(tc.context, tc.device_id)
# concatenated at the end of the final feature representation
use_extra_variant_input = False
cur_it = tc.k_steps_initial * tc.batch_steps # iteration counter used for the momentum and learning rate schedule


# In[ ]:


if tc.framework == 'mxnet' or tc.framework == 'gluon':
    # Fixing the random seed
    mx.random.seed(tc.seed)
    mx.__version__


# ### Create logs and weights directory

# In[ ]:


if not os.path.exists(tc.export_dir + "logs"):
    os.mkdir(tc.export_dir + "logs")
if not os.path.exists(tc.export_dir + "weights"):
    os.mkdir(tc.export_dir + "weights")


# ### Show the config files

# In[ ]:


print(main_config)


# In[ ]:


print(tc)


# In[ ]:


print(to)


# ### Load the dataset-files

# ### Validation Dataset (which is used during training)

# In[ ]:


pgn_dataset_arrays_dict = load_pgn_dataset(dataset_type='val', part_id=0,
                                           verbose=True, normalize=tc.normalize)
s_idcs_val = pgn_dataset_arrays_dict["start_indices"]
x_val = pgn_dataset_arrays_dict["x"]
yv_val = pgn_dataset_arrays_dict["y_value"]
yp_val = pgn_dataset_arrays_dict["y_policy"]
plys_to_end = pgn_dataset_arrays_dict["plys_to_end"]
pgn_datasets_val = pgn_dataset_arrays_dict["pgn_dataset"]
phase_vector = pgn_dataset_arrays_dict["phase_vector"]

if tc.discount != 1:
    yv_val *= tc.discount**plys_to_end
    
if tc.framework == 'mxnet':
    if tc.select_policy_from_plane:
        if tc.use_wdl and tc.use_wdl:
            val_iter = mx.io.NDArrayIter({'data': x_val},
                                         {'value_label': yv_val,
                                          'policy_label': np.array(FLAT_PLANE_IDX)[yp_val.argmax(axis=1)],
                                          'wdl_label': value_to_wdl_label(yv_val),
                                          'plys_to_end_label': prepare_plys_label(plys_to_end)},
                                          tc.batch_size)
        else:
            val_iter = mx.io.NDArrayIter({'data': x_val},
                                         {'value_label': yv_val,
                                          'policy_label': np.array(FLAT_PLANE_IDX)[yp_val.argmax(axis=1)]},
                                         tc.batch_size)
    else:
        val_iter = mx.io.NDArrayIter({'data': x_val}, {'value_label': yv_val, 'policy_label': yp_val.argmax(axis=1)}, tc.batch_size)
elif tc.framework == 'gluon' or tc.framework == 'pytorch':
    val_data = get_data_loader(x_val, yv_val, yp_val, plys_to_end, phase_vector, tc, shuffle=False)


# fill additional loaders that should be used for additional evaluations during training
if tc.framework == 'pytorch':
    additional_data_loaders = dict()
    for phase in [str(phase) for phase in to.phase_weights.keys()] + ["None"]:
        pgn_dataset_arrays_dict = load_pgn_dataset(dataset_type='test', part_id=0,
                                                   verbose=True, normalize=tc.normalize, phase=phase)
        s_idcs_val_tmp = pgn_dataset_arrays_dict["start_indices"]
        x_val_tmp = pgn_dataset_arrays_dict["x"]
        yv_val_tmp = pgn_dataset_arrays_dict["y_value"]
        yp_val_tmp = pgn_dataset_arrays_dict["y_policy"]
        plys_to_end_tmp = pgn_dataset_arrays_dict["plys_to_end"]
        pgn_datasets_val_tmp = pgn_dataset_arrays_dict["pgn_dataset"]
        phase_vector_tmp = pgn_dataset_arrays_dict["phase_vector"]

        if tc.discount != 1:
            yv_val_tmp *= tc.discount**plys_to_end_tmp

        data_loader = get_data_loader(x_val_tmp, yv_val_tmp, yp_val_tmp, plys_to_end_tmp, phase_vector_tmp, tc, shuffle=False)
        additional_data_loaders[f"Phase{phase}Test"] = data_loader

# In[ ]:


tc.nb_parts = len(glob.glob(main_config['planes_train_dir'] + '**/*'))


# In[ ]:


nb_it_per_epoch = (len(x_val) * tc.nb_parts) // tc.batch_size # calculate how many iterations per epoch exist
# one iteration is defined by passing 1 batch and doing backprop
tc.total_it = int(nb_it_per_epoch * tc.nb_training_epochs)
print(tc.total_it)


# ### Define a Learning Rate schedule

# In[ ]:


if "adam" in tc.optimizer_name:
    to.lr_schedule = ConstantSchedule(tc.min_lr)
else:
    to.lr_schedule = OneCycleSchedule(start_lr=tc.max_lr/8, max_lr=tc.max_lr, cycle_length=tc.total_it*.3, cooldown_length=tc.total_it*.6, finish_lr=tc.min_lr)
to.lr_schedule = LinearWarmUp(to.lr_schedule, start_lr=tc.min_lr, length=tc.total_it/30)

logging.getLogger().setLevel(logging.WARNING)
plot_schedule(to.lr_schedule, iterations=tc.total_it)
logging.getLogger().setLevel(logging.DEBUG)


# ### Momentum schedule

# In[ ]:


to.momentum_schedule = MomentumSchedule(to.lr_schedule, tc.min_lr, tc.max_lr, tc.min_momentum, tc.max_momentum)
plot_schedule(to.momentum_schedule, iterations=tc.total_it, ylabel='Momentum')


# ### Create the model

# In[ ]:


input_shape = x_val[0].shape
print(input_shape)


# In[ ]:


try:
    del net
except:
    pass


# ### Define the NN model / Load the pretrained model

# ### MXNet model definitions

# In[ ]:


symbol = None


# In[ ]:


#net = AlphaZeroResnet(n_labels=2272, channels=256, channels_value_head=8, channels_policy_head=81, num_res_blocks=19, value_fc_size=256, bn_mom=0.9, act_type='relu', select_policy_from_plane=select_policy_from_plane)


# In[ ]:


#net = alpha_zero_resnet(n_labels=2272, channels=256, channels_value_head=1, channels_policy_head=81, num_res_blocks=19, value_fc_size=256, bn_mom=0.9, act_type='relu')


# In[ ]:


#symbol = alpha_zero_symbol(num_filter=256, channels_value_head=4, channels_policy_head=81, workspace=1024, value_fc_size=256, num_res_blocks=19, bn_mom=0.9, act_type='relu',
#                            n_labels=2272, grad_scale_value=0.01, grad_scale_policy=0.99, select_policy_from_plane=select_policy_from_plane)


# In[ ]:


#bc_res_blocks = [3] * 13
#if tc.symbol_file is None:
#    symbol = rise_mobile_v2_symbol(channels=256, channels_operating_init=128, channel_expansion=64, channels_value_head=8,
#                      channels_policy_head=NB_POLICY_MAP_CHANNELS, value_fc_size=256, bc_res_blocks=bc_res_blocks, res_blocks=[], act_type='relu',
#                      n_labels=NB_LABELS, grad_scale_value=tc.val_loss_factor, grad_scale_policy=tc.policy_loss_factor, select_policy_from_plane=tc.select_policy_from_plane,
#                      use_se=True, dropout_rate=tc.dropout_rate, use_extra_variant_input=use_extra_variant_input)
#else:
#    symbol = mx.sym.load(tc.export_dir + "weights/" + tc.symbol_file)


# kernels = [3] * 15
# kernels[7] = 5
# kernels[11] = 5
# kernels[12] = 5
# kernels[13] = 5
# 
# se_types = [None] * len(kernels)
# se_types[5] = "eca_se"
# se_types[8] = "eca_se"
# se_types[12] = "eca_se"
# se_types[13] = "eca_se"
# se_types[14] = "eca_se"

# kernels = [3] * 7
# 
# se_types = [None] * len(kernels)
# se_types[5] = "eca_se"
# 

# symbol = rise_mobile_v3_symbol(channels=256, channels_operating_init=224, channel_expansion=32, act_type='relu',
#                                channels_value_head=8, value_fc_size=256,
#                                channels_policy_head=NB_POLICY_MAP_CHANNELS,
#                                grad_scale_value=tc.val_loss_factor,
#                                grad_scale_policy=tc.policy_loss_factor,
#                                grad_scale_wdl=tc.wdl_loss_factor,
#                                grad_scale_ply=tc.plys_to_end_loss_factor,
#                                dropout_rate=tc.dropout_rate, select_policy_from_plane=True,
#                                kernels=kernels, se_types=se_types, use_avg_features=False,
#                                use_wdl=tc.use_wdl, use_plys_to_end=tc.use_plys_to_end,
#                                use_mlp_wdl_ply=tc.use_mlp_wdl_ply
#                                )

# kernels = [3,3,3,3,3,3,5,5]
# 
# se_types = [
#     None, # 1
#     None, # 2
#     None,  # 3
#     "eca_se",  # 4
#     None, # 5
#     None,  # 6
#     None, # 7
#     "eca_se", # 8
# ] 
# 
# symbol = preact_resnet_se(channels=288, act_type='relu',
#                           channels_value_head=8, value_fc_size=256,
#                           channels_policy_head=NB_POLICY_MAP_CHANNELS,
#                           grad_scale_value=tc.val_loss_factor, grad_scale_policy=tc.policy_loss_factor, 
#                           dropout_rate=tc.dropout_rate, select_policy_from_plane=True,
#                           kernels=kernels, se_types=se_types, use_avg_features=True, use_raw_features=True)

# ### Pytorch model definitions

# In[ ]:


model = get_rise_v33_model_by_train_config(input_shape, tc)


# model = LeViT(
#     image_size = input_shape[1],
#     in_channels = input_shape[0],
#     channels_policy_head = NB_POLICY_MAP_CHANNELS,
#     stages = 1,             # number of stages
#     dim = (256,),  # dimensions at each stage
#     depth = 9,              # transformer of depth 4 at each stage
#     heads = (4,),      # heads at each stage
#     mlp_mult = 2,
#     dropout = 0.1,
#     select_policy_from_plane=tc.select_policy_from_plane,
#     use_wdl=tc.use_wdl, use_plys_to_end=tc.use_plys_to_end,
#     use_mlp_wdl_ply=tc.use_mlp_wdl_ply,    
# )

# model = MobileViT(
#     image_size = (input_shape[1], input_shape[2]),
#     in_channels = input_shape[0],
#     dims = [96, 120, 144],
#     channels = 256,
#     channels_policy_head = NB_POLICY_MAP_CHANNELS,
#     select_policy_from_plane=tc.select_policy_from_plane,
#     use_wdl=tc.use_wdl, use_plys_to_end=tc.use_plys_to_end,
#     use_mlp_wdl_ply=tc.use_mlp_wdl_ply,
# )

# model = TrtViT(
#     image_size = input_shape[1],
#     in_channels = input_shape[0],
#     channels_policy_head = NB_POLICY_MAP_CHANNELS,
#     channels=256,
#     select_policy_from_plane=tc.select_policy_from_plane,
#     use_wdl=tc.use_wdl, use_plys_to_end=tc.use_plys_to_end,
#     use_mlp_wdl_ply=tc.use_mlp_wdl_ply,  
#     )

# kernels = [3] * 12 #15
# kernels[10] = 5
# kernels[9] = 5
# kernels[5] = 5
# #kernels[13] = 5
# 
# use_transformers = [False] * len(kernels)
# use_transformers[11] = True
# #use_transformers[9] = True
# #use_transformers[4] = True
# 
# se_types = [None] * len(kernels)
# se_types[5] = "eca_se"
# se_types[11] = "eca_se"
# #se_types[12] = "eca_se"
# #se_types[13] = "eca_se"
# #se_types[14] = "eca_se"
# 
# class Args:
#     pass
# 
# args = Args()
# args.input_shape = input_shape
# args.channels_policy_head = NB_POLICY_MAP_CHANNELS
# args.n_labels = NB_LABELS
# args.select_policy_from_plane = tc.select_policy_from_plane
# args.use_wdl = tc.use_wdl
# args.use_plys_to_end = tc.use_plys_to_end
# args.use_mlp_wdl_ply = tc.use_mlp_wdl_ply
# 
# model = RiseV3(nb_input_channels=args.input_shape[0], board_height=args.input_shape[1], board_width=args.input_shape[2],
#                 channels=256, channels_operating_init=224, channel_expansion=32,
#                 channels_value_head=8, value_fc_size=256,
#                 channels_policy_head=args.channels_policy_head,
#                 dropout_rate=0, select_policy_from_plane=args.select_policy_from_plane,
#                 kernels=kernels, se_types=se_types, use_avg_features=False, n_labels=args.n_labels,
#                 use_wdl=args.use_wdl, use_plys_to_end=args.use_plys_to_end, use_mlp_wdl_ply=args.use_mlp_wdl_ply,
#                 use_transformers=use_transformers, path_dropout=0.07
#                )

# model = NextVit(
#     image_size = input_shape[1],
#     in_channels = input_shape[0],
#     channels_policy_head = NB_POLICY_MAP_CHANNELS,
#     stage3_repeat=1,
#     channels=256,
#     select_policy_from_plane=tc.select_policy_from_plane,
#     use_wdl=tc.use_wdl, use_plys_to_end=tc.use_plys_to_end,
#     use_mlp_wdl_ply=tc.use_mlp_wdl_ply,
#     use_transformer_heads=False, #True,
#     se_type=None, #'eca_se'
#     use_simple_transformer_blocks=False, #True
#     ) # -> 19 pool blocks

# model = AlphaZeroResnet(nb_input_channels=input_shape[0], board_height=input_shape[1], board_width=input_shape[2],
#                 channels=256, act_type='relu', num_res_blocks=19,
#                 channels_value_head=8, value_fc_size=256,
#                 channels_policy_head=NB_POLICY_MAP_CHANNELS,
#                 select_policy_from_plane=tc.select_policy_from_plane,
#                 n_labels=NB_LABELS,
#                 use_wdl=tc.use_wdl, use_plys_to_end=tc.use_plys_to_end, use_mlp_wdl_ply=tc.use_mlp_wdl_ply,
#                )

# model = VisionTransformer(get_b8_config(), img_size=8, in_channels=input_shape[0], num_classes=NB_POLICY_MAP_CHANNELS*64,
#                           use_wdl=tc.use_wdl, use_plys_to_end=tc.use_plys_to_end, use_mlp_wdl_ply=tc.use_mlp_wdl_ply,)

# ### Convert MXNet Symbol to Gluon Network

# In[ ]:


if tc.framework == 'gluon' and symbol is not None:
    inputs = mx.sym.var('data', dtype='float32')
    value_out = symbol.get_internals()[main_config['value_output']+'_output']
    policy_out = symbol.get_internals()[main_config['policy_output']+'_output']
    sym = mx.symbol.Group([value_out, policy_out])
    net = mx.gluon.SymbolBlock(sym, inputs)


# ## Network summary

# In[ ]:


if tc.framework == 'gluon':
    print(net)
elif tc.framework == 'pytorch':
    print(model)


# In[ ]:


if tc.framework != 'pytorch' and symbol is not None:
    display(mx.viz.plot_network(
        symbol,
        shape={'data':(1, input_shape[0], input_shape[1], input_shape[2])},
        node_attrs={"shape":"oval","fixedsize":"false"}
    ))
elif tc.framework == 'gluon':
    display(mx.viz.plot_network(
        net(mx.sym.var('data'))[1],
        shape={'data':(1, input_shape[0], input_shape[1], input_shape[2])},
        node_attrs={"shape":"oval","fixedsize":"false"}
    ))


# In[ ]:


if tc.framework == 'mxnet':
    mx.viz.print_summary(
        symbol,
        shape={'data':(1, input_shape[0], input_shape[1], input_shape[2])},
    )
elif tc.framework == 'gluon':
    mx.viz.print_summary(
    net(mx.sym.var('data'))[1], 
    shape={'data':(1, input_shape[0], input_shape[1], input_shape[2])},
    ) 
elif tc.framework == 'pytorch':
    print(summary(model, (input_shape[0], input_shape[1], input_shape[2]), device="cpu"))


# ## Analyze the Flops

# In[ ]:


if tc.framework == 'pytorch':
    dummy_input = torch.Tensor(np.expand_dims(x_val[0], axis=0)).to('cpu')
    flops = FlopCountAnalysis(model, dummy_input)
    print(flops.total())
    from fvcore.nn import flop_count_table
    print(flop_count_table(flops))


# ## Initialize the weights 
# (only needed if no pretrained weights are used)

# In[ ]:


if tc.framework == 'mxnet':
    # create a trainable module on compute context
    if tc.use_wdl and tc.use_plys_to_end:
        label_names=['value_label', 'policy_label', 'wdl_label', 'plys_to_end_label']
    else:
        label_names=['value_label', 'policy_label']
    
    model = mx.mod.Module(symbol=symbol, context=ctx, label_names=label_names)
    model.bind(for_training=True, data_shapes=[('data', (tc.batch_size, input_shape[0], input_shape[1], input_shape[2]))],
             label_shapes=val_iter.provide_label)
    model.init_params(mx.initializer.Xavier(rnd_type='uniform', factor_type='avg', magnitude=2.24))
    if tc.tar_file:
        model.load_params(tc.export_dir + "weights/" + tc.tar_file)
elif tc.framework == 'gluon':    
    # Initializing the parameters
    for param in net.collect_params('.*gamma|.*moving_mean|.*moving_var'):
        net.params[param].initialize(mx.initializer.Constant(1), ctx=ctx)
    for param in net.collect_params('.*beta|.*bias'):
        net.params[param].initialize(mx.initializer.Constant(0), ctx=ctx)
    for param in net.collect_params('.*weight'):
        net.params[param].initialize(mx.init.Xavier(rnd_type='uniform', factor_type='avg', magnitude=2.24), ctx=ctx)

    if tc.tar_file:
        net.collect_params().load(tc.export_dir + "weights/" + tc.tar_file, ctx)
    net.hybridize()
elif tc.framework == 'pytorch':
    def init_weights(m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, torch.nn.Linear):
                m.bias.data.fill_(0.01)
    #model.apply(init_weights)
    if tc.tar_file:
        print('load model params from file:', tc.tar_file)
        load_torch_state(model, torch.optim.SGD(model.parameters(), lr=tc.max_lr), tc.tar_file, tc.device_id)
    if torch.cuda.is_available():
        model.cuda(torch.device(f"cuda:{tc.device_id}"))


# ## Define the metrics to use

# In[ ]:


to.metrics = get_metrics(tc)


# ## Define a training agent

# In[ ]:


if tc.framework == 'mxnet':
    train_agent = TrainerAgentMXNET(model, symbol, val_iter, tc, to, use_rtpt=True)
elif tc.framework == 'gluon':
    train_agent = TrainerAgentGluon(net, val_data, tc, to, use_rtpt=True)
elif tc.framework == 'pytorch':
    train_agent = TrainerAgentPytorch(model, val_data, tc, to, use_rtpt=True,
                                      additional_loaders=additional_data_loaders)


# ## Performance Pre-Training

# In[ ]:


if tc.framework == 'mxnet':
    print(model.score(val_iter, to.metrics))


# ## Start the training process

# In[ ]:


(k_steps_final, value_loss_final, policy_loss_final, value_acc_sign_final, val_p_acc_final), \
    (k_steps_best, val_metric_values_best) = train_agent.train(cur_it)


# ## Export the last model state

# In[ ]:


prefix = tc.export_dir + "weights/model-%.5f-%.3f" % (policy_loss_final, val_p_acc_final)

if tc.framework == 'mxnet':
    # the export function saves both the architecture and the weights
    model.save_checkpoint(prefix, epoch=k_steps_final)
elif tc.framework == 'gluon':
    # the export function saves both the architecture and the weights
    net.export(prefix, epoch=k_steps_final)
    logging.info("Saved checkpoint to %s-%04d.params", prefix, k_steps_final)
elif tc.framework == 'pytorch':
    # the export function saves both the architecture and the weights
    save_torch_state(model, torch.optim.SGD(model.parameters(), lr=tc.max_lr), '%s-%04d.tar' % (prefix, k_steps_final))


# ## Print validation metrics for best model

# In[ ]:


print(val_metric_values_best)


# ## Copy best model to best-model directory

# In[ ]:


val_loss_best = val_metric_values_best["loss"]
val_p_acc_best = val_metric_values_best["policy_acc"]

model_name = "model-%.5f-%.3f" % (val_loss_best, val_p_acc_best)
model_prefix = tc.export_dir + "weights/" + model_name
model_arch_path = '%s-symbol.json' % model_prefix
model_params_path = '%s-%04d.params' % (model_prefix, k_steps_best)
model_tar_path = '%s-%04d.tar' % (model_prefix, k_steps_best)

if not os.path.exists(tc.export_dir + "best-model"):
    os.mkdir(tc.export_dir + "best-model")
    
best_model_prefix = tc.export_dir + "best-model/" + model_name
best_model_arch_path = '%s-symbol.json' % best_model_prefix
best_model_params_path = '%s-%04d.params' % (best_model_prefix, k_steps_best)
best_model_tar_path = '%s-%04d.tar' % (best_model_prefix, k_steps_best)

if tc.framework == 'mxnet' or tc.framework == 'gluon':
    shutil.copy(model_arch_path, best_model_arch_path)
    shutil.copy(model_params_path, best_model_params_path)
elif tc.framework == 'pytorch':
    shutil.copy(model_tar_path, best_model_tar_path)


# ## Load the best model once again

# In[ ]:


# delete the current net object form memory
if tc.framework == 'mxnet':
    del model
elif tc.framework == 'gluon':
    del net


# In[ ]:


print('load current best model:', model_params_path)

if tc.framework == 'mxnet' or tc.framework == 'gluon':
    symbol = mx.sym.load(model_arch_path)
    inputs = mx.sym.var('data', dtype='float32')
    value_out = symbol.get_internals()[main_config['value_output']+'_output']
    policy_out = symbol.get_internals()[main_config['policy_output']+'_output']
    if tc.use_wdl and tc.use_plys_to_end:
        auxiliary_out = symbol.get_internals()[main_config['auxiliary_output']+'_output']
        wdl_out = symbol.get_internals()[main_config['wdl_output']+'_output']
        ply_to_end_out = symbol.get_internals()[main_config['plys_to_end_output']+'_output']
        sym = mx.symbol.Group([value_out, policy_out, auxiliary_out, wdl_out, ply_to_end_out])
    else:
        sym = mx.symbol.Group([value_out, policy_out])
    net = mx.gluon.SymbolBlock(sym, inputs)
    net.collect_params().load(model_params_path, ctx)
elif tc.framework == 'pytorch':
    load_torch_state(model, torch.optim.SGD(model.parameters(), lr=tc.max_lr), model_tar_path, tc.device_id)


# In[ ]:


print('best val_loss: %.5f with v_policy_acc: %.5f at k_steps_best %d' % (val_loss_best, val_p_acc_best, k_steps_best))


# ## Convert to onnx

# In[ ]:


if tc.use_wdl and tc.use_plys_to_end:
    outputs = [main_config['value_output']+'_output', main_config['policy_output']+'_output',
               main_config['auxiliary_output']+'_output',
               main_config['wdl_output']+'_output', main_config['plys_to_end_output']+'_output']
else:
    outputs = [main_config['value_output']+'_output', main_config['policy_output']+'_output',]

if tc.framework == 'mxnet':
    convert_mxnet_model_to_onnx(best_model_arch_path, best_model_params_path, 
                                outputs, 
                                tuple(input_shape), tuple([1, 8, 16, 64]), True)
elif tc.framework == 'pytorch':
    model_prefix = "%s-%04d" % (model_name, k_steps_best)
    with torch.no_grad():
        ctx = get_context(tc.context, tc.device_id)
        dummy_input = torch.zeros(1, input_shape[0], input_shape[1], input_shape[2]).to(ctx)
        export_to_onnx(model, 1,
                       dummy_input,
                       Path(tc.export_dir) / Path("best-model"), model_prefix, tc.use_wdl and tc.use_plys_to_end,
                       True)


# In[ ]:


print("Saved json, weight & onnx files of the best model to %s" % (tc.export_dir + "best-model"))


# ## Show move predictions

# In[ ]:


idx = 0


# In[ ]:


if mode == MODE_CHESS:
    start_board = chess.Board()
elif mode == MODE_CRAZYHOUSE:
    start_board = chess.variant.CrazyhouseBoard()
else:
    start_board = planes_to_board(x_val[idx], normalized_input=tc.normalize, mode=mode)
board = start_board
print(chess.COLOR_NAMES[board.turn])
if board.uci_variant == "crazyhouse":
    print(board.pockets)
print(board)


# In[ ]:


def predict_single(net, x, select_policy_from_plane=False):
    
    out = [None, None]
    if tc.framework == 'mxnet' or tc.framework == 'gluon':
        pred = net(mx.nd.array(np.expand_dims(x, axis=0), ctx=ctx))
        out[0] = pred[0].asnumpy()
        out[1] = pred[1].softmax().asnumpy()
    elif tc.framework == 'pytorch':
        with torch.no_grad():
            pred = net(torch.Tensor(np.expand_dims(x, axis=0)).to(ctx))
            out[0] = pred[0].to(torch.device("cpu")).numpy()
            out[1] = pred[1].to(torch.device("cpu")).softmax(dim=1).numpy()
    if select_policy_from_plane:
        out[1] = out[1][:, FLAT_PLANE_IDX]
    
    return out


# In[ ]:


if tc.framework == 'pytorch':
    net = model
    net.eval()


# In[ ]:


x_start_pos = board_to_planes(board, normalize=tc.normalize, mode=mode)
pred = predict_single(net, x_start_pos, tc.select_policy_from_plane)
print(pred)


# In[ ]:


policy_to_best_move(board, yp_val[idx])


# In[ ]:


opts = 5
selected_moves, probs = policy_to_moves(board, pred[1][0])
print(selected_moves[:opts])


# In[ ]:


plt.barh(range(opts)[::-1], probs[:opts])
ax = plt.gca()
ax.set_yticks(range(opts)[::-1])
ax.set_yticklabels(selected_moves[:opts])


# In[ ]:


board = start_board
board.push_uci('e2e4')
board.push_uci('e7e5')
board.push_uci('f1c4')
board.push_uci('b8c6')
board.push_uci('d1h5')
x_scholar_atck = board_to_planes(board, normalize=tc.normalize, mode=mode)
print(board)


# In[ ]:


pred = predict_single(net, x_scholar_atck, tc.select_policy_from_plane)

selected_moves, probs = policy_to_moves(board, pred[1][0])
plt.barh(range(opts)[::-1], probs[:opts])
ax = plt.gca()
ax.set_yticks(range(opts)[::-1])
ax.set_yticklabels(selected_moves[:opts])


# In[ ]:


board.push(selected_moves[0])
print(board)


# ### Performance on test dataset
# 

# In[ ]:


pgn_dataset_arrays_dict = load_pgn_dataset(dataset_type='test', part_id=0,
                                           verbose=True, normalize=True)
s_idcs_test = pgn_dataset_arrays_dict["start_indices"]
x_test = pgn_dataset_arrays_dict["x"]
yv_test = pgn_dataset_arrays_dict["y_value"]
yp_test = pgn_dataset_arrays_dict["y_policy"]
yplys_test = pgn_dataset_arrays_dict["plys_to_end"]
pgn_datasets_test = pgn_dataset_arrays_dict["pgn_dataset"]
phase_vector_test = pgn_dataset_arrays_dict["phase_vector"]

test_data = get_data_loader(x_test, yv_test, yp_test, yplys_test, phase_vector_test, tc, shuffle=False)


# In[ ]:


if tc.framework == 'mxnet':
    metrics = metrics_gluon

print(evaluate_metrics(to.metrics, test_data, net, nb_batches=None, sparse_policy_label=tc.sparse_policy_label, ctx=ctx,
                 phase_weights=to.phase_weights, apply_select_policy_from_plane=tc.select_policy_from_plane,
                 use_wdl=tc.use_wdl, use_plys_to_end=tc.use_plys_to_end))


# ### Show result on mate-in-one problems

# In[ ]:


pgn_dataset_arrays_dict = load_pgn_dataset(dataset_type='mate_in_one', part_id=0,
                                           verbose=True, normalize=tc.normalize)

s_idcs_mate = pgn_dataset_arrays_dict["start_indices"]
x_mate = pgn_dataset_arrays_dict["x"]
yv_mate = pgn_dataset_arrays_dict["y_value"]
yp_mate = pgn_dataset_arrays_dict["y_policy"]
yplys_mate = pgn_dataset_arrays_dict["plys_to_end"]
pgn_dataset_mate = pgn_dataset_arrays_dict["pgn_dataset"]
phase_vector_mate = pgn_dataset_arrays_dict["phase_vector"]

yplys_mate = np.ones(len(yv_mate))
mate_data = get_data_loader(x_mate, yv_mate, yp_mate, yplys_mate, phase_vector_mate, tc, shuffle=False)


# ### Mate In One Performance

# In[ ]:


print(evaluate_metrics(to.metrics, mate_data, net, nb_batches=None, sparse_policy_label=tc.sparse_policy_label, ctx=ctx,
                 phase_weights=to.phase_weights, apply_select_policy_from_plane=tc.select_policy_from_plane,
                 use_wdl=tc.use_wdl, use_plys_to_end=tc.use_plys_to_end))


# ### Show some example mate problems

# In[ ]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# ### Evaluate Performance

# In[ ]:


def eval_pos(net, x_mate, yp_mate, verbose=False, select_policy_from_plane=False):
    
    board = planes_to_board(x_mate, normalized_input=tc.normalize, mode=mode)
    if verbose is True:
        print("{0}'s turn".format(chess.COLOR_NAMES[board.turn]))
        if board.uci_variant == "crazyhouse":
            print("black/white {0}".format(board.pockets))
    pred = predict_single(net, x_mate, select_policy_from_plane=select_policy_from_plane)
    
    true_move = policy_to_move(yp_mate, mirror_policy=board.turn==chess.BLACK)
    
    opts = 5
    pred_moves, probs = policy_to_moves(board, pred[1][0])
    pred_moves = pred_moves[:opts]
    
    legal_move_cnt = board.legal_moves.count()
    mate_move_cnt = str(board.legal_moves).count('#')
    
    is_mate_5_top = False
    
    for pred_move in pred_moves:
        board_5_top = deepcopy(board)
        board_5_top.push(pred_move)
        if board_5_top.is_checkmate() is True:
            is_mate_5_top = True
            break
    
    board.push(pred_moves[0])
    
    is_checkmate = False
    if board.is_checkmate() is True:
        is_checkmate = True
        
    filtered_pred = sorted(pred[1][0], reverse=True)
    
    if verbose is True:
        plt.barh(range(opts)[::-1], filtered_pred[:opts])
        ax = plt.gca()
        ax.set_yticks(range(opts)[::-1])
        ax.set_yticklabels(pred_moves)
        plt.title('True Move:' + str(true_move) +
                 '\nEval:' + str(pred[0][0]))
        plt.show()
    
    return pred, pred_moves, true_move, board, is_checkmate, is_mate_5_top, legal_move_cnt, mate_move_cnt


# In[ ]:


nb_pos = len(x_mate)
mates_found = []
mates_5_top_found = []
legal_mv_cnts = []
mate_mv_cnts = []

for i in range(nb_pos):
    pred, pred_moves, true_move, board, is_mate, is_mate_5_top, legal_mv_cnt, mate_mv_cnt= eval_pos(net, x_mate[i], yp_mate[i], select_policy_from_plane=tc.select_policy_from_plane)
    mates_found.append(is_mate)
    legal_mv_cnts.append(legal_mv_cnt)
    mate_mv_cnts.append(mate_mv_cnt)
    mates_5_top_found.append(is_mate_5_top)


# In[ ]:


print(np.array(mate_mv_cnts).mean())


# In[ ]:


print(np.array(legal_mv_cnts).mean())


# ### Random Guessing Baseline

# In[ ]:


print(np.array(mate_mv_cnts).mean() / np.array(legal_mv_cnts).mean())


# ### Prediciton Performance

# In[ ]:


print('mate_in_one_acc:', sum(mates_found) / nb_pos)


# In[ ]:


print(sum(mates_5_top_found) / nb_pos)


# In[ ]:


print(pgn_dataset_mate.tree())


# In[ ]:


metadata = np.array(pgn_dataset_mate['metadata'])
print(metadata[0, :])
print(metadata[1, :])


# In[ ]:


site_mate = metadata[1:, 1]


# In[ ]:


def clean_string(np_string):
    string = str(site_mate[i]).replace("b'", "")
    string = string.replace("'", "")
    string = string.replace('"', '')
    
    return string


# In[ ]:


import chess.svg
from IPython.display import SVG, HTML


# ## Show the result of the first 17 examples

# In[ ]:


for i in range(17):
    print(clean_string(site_mate[i]))
    pred, pred_moves, true_move, board, is_checkmate, is_mate_5_top, legal_move_cnt, mate_move_cnt = eval_pos(net, x_mate[i], yp_mate[i], verbose=True, select_policy_from_plane=tc.select_policy_from_plane)
    pred_move = pred_moves[0]
    pred_arrow = chess.svg.Arrow(pred_move.from_square, pred_move.to_square)
    SVG(data=chess.svg.board(board=board, arrows=[pred_arrow], size=400))


# ## Show examples where it failed

# In[ ]:


mate_missed = 0
for i in range(1000):
    pred, pred_moves, true_move, board, is_checkmate, is_mate_5_top, legal_move_cnt, mate_move_cnt = eval_pos(net, x_mate[i], yp_mate[i], verbose=False, select_policy_from_plane=tc.select_policy_from_plane)
    if is_mate_5_top is False:
        mate_missed += 1
        print(clean_string(site_mate[i]))
        pred, pred_moves, true_move, board, is_checkmate, is_mate_5_top, legal_move_cnt, mate_move_cnt = eval_pos(net, x_mate[i], yp_mate[i], verbose=True, select_policy_from_plane=tc.select_policy_from_plane)
        pred_move = pred_moves[0]
        pred_arrow = chess.svg.Arrow(pred_move.from_square, pred_move.to_square)
        SVG(data=chess.svg.board(board=board, arrows=[pred_arrow], size=400))
    if mate_missed == 15:
        break


# In[ ]:




