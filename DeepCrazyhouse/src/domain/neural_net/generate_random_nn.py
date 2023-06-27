"""
@file: generate_random_nn.py
Created on 27.05.21
@project: crazyara
@author: queensgambit

Command-line tool to generate a random initialized neural network and export ONNX weights.
"""
import os
import argparse
import mxnet as mx
import torch
from pathlib import Path
import logging
import numpy as np
import sys
sys.path.insert(0, '../../../../')
from DeepCrazyhouse.src.domain.neural_net.architectures.mxnet_alpha_zero import get_alpha_zero_symbol
from DeepCrazyhouse.src.domain.neural_net.architectures.rise_mobile_v3 import get_rise_v33_symbol
from DeepCrazyhouse.src.domain.neural_net.architectures.rise_mobile_v2 import get_rise_v2_symbol
from DeepCrazyhouse.src.domain.neural_net.onnx.convert_to_onnx import convert_mxnet_model_to_onnx
from DeepCrazyhouse.src.domain.variants.constants import NB_LABELS, NB_POLICY_MAP_CHANNELS, NB_CHANNELS_TOTAL,\
    BOARD_WIDTH, BOARD_HEIGHT
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.rise_mobile_v3 import get_rise_v33_model, get_rise_v2_model
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.a0_resnet import get_alpha_zero_model
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.alpha_vile import get_alpha_vile_model
from DeepCrazyhouse.src.runtime.color_logger import enable_color_logging
from DeepCrazyhouse.configs.train_config import TrainConfig
from DeepCrazyhouse.src.training.trainer_agent_pytorch import save_torch_state, export_to_onnx, get_context

enable_color_logging()


def parse_args(cmd_args: list):
    """
    Parses command-line argument and returns them as a dictionary object
    :param cmd_args: Command-line arguments (sys.argv[1:])
    :return: Parsed arguments as dictionary object
    """
    parser = argparse.ArgumentParser(description='Command-line tool to generate a random initialized neural network'
                                                 ' and export MXNet and ONNX weights.')
    parser.add_argument("--model-type", type=str, default="risev2",
                        help="available model types [alpha_zero, risev2, risev3.3, alpha_vile] (default: risev2)")
    parser.add_argument("--channels-policy-head", type=int, default=None,
                        help=" (default: None)")
    parser.add_argument("--n-labels", type=int, default=None,
                        help=" (default: None)")
    parser.add_argument("--select-policy-from-plane", default=False, action="store_true",
                        help="If true, the policy plane representation will be used (default: False)")
    parser.add_argument("--input-shape", type=int, nargs="*", default=None,
                        help="Input shape of the neural network. Arguments need to be passed as a list"
                             'e.g. pass "--input-shape 34 8 8" for the default crazyhouse input (default: None)')
    parser.add_argument("--val-loss-factor", type=float, default=0.01,
                        help="Value loss factor used during training. Only relevant if the MXNet symbol file will be"
                             " directly used for training. (default: 0.01)")
    parser.add_argument("--policy-loss-factor", type=float, default=0.99,
                        help="Policy loss factor used during training. Only relevant if the MXNet symbol will be"
                             "directly used for training. (default: 0.99)")
    parser.add_argument("--use-wdl", default=False, action="store_true",
                        help="If true, the WDL value loss will be used (only available in pytorch). (default: False)")
    parser.add_argument("--use-plys-to-end", default=False, action="store_true",
                        help="If true, the plys to end auxiliary loss will be used (only available in pytorch). (default: False)")
    parser.add_argument("--use-mlp-wdl-ply", default=False, action="store_true",
                        help="If true, a small mlp with value output for the wdl and ply head shall be used (only available in pytorch). (default: False)")
    parser.add_argument("--input-version", type=str, default="1.0",
                        help="Defines the version id for the input representation. (default: '1.0')")
    parser.add_argument("--context", type=str, default="gpu",
                        help="available context for saving the weights ['gpu', 'cpu'] (only relevant for pytorch) (default: 'gpu')")
    parser.add_argument("--device-id", type=int, default=0,
                        help="GPU device id used for exporting the weights (only relevant for pytorch) (default: 0)")
    parser.add_argument("--export-dir", type=str, default="./",
                        help="Directory where the model files will be exported.")
    args = parser.parse_args(cmd_args)

    if args.input_shape is None:
        args.input_shape = (NB_CHANNELS_TOTAL, BOARD_HEIGHT, BOARD_WIDTH)
        logging.info(f"Given 'input_shape' is 'None'. It was set to {args.input_shape}.")
    if not args.select_policy_from_plane and args.n_labels is None:
        args.n_labels = NB_LABELS
        logging.info(f"Given 'n_labels' is 'None'. It was set to {args.n_labels}.")
    if args.select_policy_from_plane and args.channels_policy_head is None:
        args.channels_policy_head = NB_POLICY_MAP_CHANNELS
        logging.info(f"Given 'channels_policy_head' is 'None'. It was set to {args.channels_policy_head}.")
    if not os.path.isdir(args.export_dir):
        raise Exception("The given directory %s does not exist." % args.model_dir)
    if args.export_dir[-1] != '/':
        args.export_dir += '/'
    if args.channels_policy_head is None:
        args.channels_policy_head = NB_POLICY_MAP_CHANNELS
        logging.info(f"Given 'channels_policy_head' is 'None'. It was set to {args.channels_policy_head}.")

    # convert list to tuple
    args.input_shape = tuple(args.input_shape)
    return args


def generate_random_nn_mnxet(args):
    """
    Generates a new neural network model with random parameter initialization and exports it to ONNX.
    """
    if args.model_type == "alpha_zero":
        symbol = get_alpha_zero_symbol(args)
    elif args.model_type == "risev2":
        symbol = get_rise_v2_symbol(args)
    elif args.model_type == "risev3.3":
        symbol = get_rise_v33_symbol(args)
    else:
        raise NotImplementedError

    x_dummy = np.zeros(shape=(1, args.input_shape[0], args.input_shape[1], args.input_shape[2]))
    y_value_dummy = np.zeros(shape=(1, 1))
    if args.select_policy_from_plane:
        y_policy_dummy = np.zeros(shape=(1, args.channels_policy_head + args.input_shape[1] * args.input_shape[2]))
    else:
        y_policy_dummy = np.zeros(shape=(1, args.n_labels))
    data_iter = mx.io.NDArrayIter({'data': x_dummy}, {'value_label': y_value_dummy,
                                                      'policy_label': y_policy_dummy.argmax(axis=1)}, 1)

    model = mx.mod.Module(symbol=symbol, context=mx.cpu(), label_names=['value_label', 'policy_label'])
    model.bind(for_training=True, data_shapes=[('data', x_dummy.shape)],
               label_shapes=data_iter.provide_label)
    model.init_params(mx.initializer.Xavier(rnd_type='uniform', factor_type='avg', magnitude=2.24))

    prefix = args.export_dir + args.model_type
    sym_file = prefix + "-symbol.json"
    params_file = prefix + "-" + "%04d.params" % 0

    # the export function saves both the architecture and the weights
    model.save_checkpoint(prefix, epoch=0)

    # if convert_to_onnx:
    convert_mxnet_model_to_onnx(sym_file, params_file, ["value_out_output", "policy_out_output"], args.input_shape,
                                [1, 8, 16], False)


def generate_random_nn_pytorch(args, train_config: TrainConfig):
    """
    Generates a new neural network model with random parameter initialization and exports it to ONNX.
    """
    if args.model_type == "alpha_zero":
        model = get_alpha_zero_model(args)
    elif args.model_type == "risev2":
        model = get_rise_v2_model(args)
    elif args.model_type == "risev3.3":
        model = get_rise_v33_model(args)
    elif args.model_type == "alpha_vil":
        model = get_alpha_vile_model(args)
    else:
        raise NotImplementedError

    if args.context == "gpu" and torch.cuda.is_available():
        model.cuda(torch.device(f"cuda:{args.device_id}"))

    prefix = args.export_dir + args.model_type
    k_steps_final = 0
    save_torch_state(model, torch.optim.SGD(model.parameters(), lr=train_config.max_lr), '%s-%04d.tar' % (prefix, k_steps_final))

    model_prefix = "%s-%04d" % (args.model_type, k_steps_final)
    with torch.no_grad():
        ctx = get_context(args.context, args.device_id)
        dummy_input = torch.zeros(1, args.input_shape[0], args.input_shape[1], args.input_shape[2]).to(ctx)
        export_to_onnx(model, 1,
                       dummy_input,
                       Path(args.export_dir), model_prefix, args.use_wdl and args.use_plys_to_end,
                       True, args.input_version)


def main():
    """
    Main function which is executed on start-up

    Exemplary calls:
     e.g. call for CrazyAra model
     python generate_random_nn.py --model-type risev2 --channels-policy-head 81 --input-shape 34 8 8 --select-policy-from-plane
     e.g. call for ClassicAra model
     python generate_random_nn.py --model-type risev3.3 --channels-policy-head 76 --input-shape 52 8 8 --select-policy-from-plane --input-version 3.0 --use-wdl --use-plys-to-end
     e.g. call for MultiAra model
     python generate_random_nn.py --model-type risev2 --channels-policy-head 84 --input-shape 63 8 8 --select-policy-from-plane
     e.g. call for AtariAra model (flat output)
     python3 generate_random_nn.py --model-type risev3.3 --n-labels 43 --channels-policy-head 1 --input-shape 3 160 210
    :return:
    """
    args = parse_args(sys.argv[1:])
    logging.basicConfig(level=logging.INFO)

    train_config = TrainConfig()

    if train_config.framework == 'mxnet' or train_config.framework == 'gluon':
        generate_random_nn_mnxet(args)
    if train_config.framework == 'pytorch':
        generate_random_nn_pytorch(args, train_config)


if __name__ == '__main__':
    main()
