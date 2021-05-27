"""
@file: generate_random_nn.py
Created on 27.05.21
@project: crazyara
@author: queensgambit

Command-line tool to generate a random initialized neural network and export ONNX weights.
"""
import sys
import argparse
import mxnet as mx
import logging
from DeepCrazyhouse.src.domain.neural_net.architectures.mxnet_alpha_zero import get_alpha_zero_symbol
from DeepCrazyhouse.src.domain.neural_net.architectures.rise_mobile_v3 import get_rise_v33_symbol
from DeepCrazyhouse.src.domain.neural_net.architectures.rise_mobile_v2 import get_rise_v2_symbol


def parse_args(cmd_args: list):
    """
    Parses command-line argument and returns them as a dictionary object
    :param cmd_args: Command-line arguments (sys.argv[1:])
    :return: Parsed arguments as dictionary object
    """
    parser = argparse.ArgumentParser(description='MXNet to ONN converter')

    parser.add_argument("--model-type", type=str, default="risev2",
                        help="model directory which contains the .param and .sym file (default: risev2")
    parser.add_argument("--channels_policy_head", type=int, default=None,
                        help=" (default: None)")
    parser.add_argument("--n-labels", type=int, default=None,
                        help=" (default: None)")
    parser.add_argument("--select-policy-from-plane", default=False, action="store_true",
                        help="If true, the ONNX model is validated after conversion (default: False)")
    parser.add_argument("--val-loss-factor", type=float, default=0.01,
                        help=" (default: 0.01)")
    parser.add_argument("--policy-loss-factor", type=float, default=0.99,
                        help=" (default: 0.99)")
    args = parser.parse_args(cmd_args)

    if args.input_shape is None:
        raise ValueError('Given input shape must not be "None"')

    # convert list to tuple
    args.input_shape = tuple(args.input_shape)
    return args


def generate_random_nn(args):
    if args.model_type == "alpha_zero":
        symbol = get_alpha_zero_symbol(args.channels_policy_head, args.n_labels, args.select_policy_from_plane,
                                       args.val_loss_factor, args.spolicy_loss_factor)
    elif args.model_type == "risev2":
        symbol = get_rise_v2_symbol(args.channels_policy_head, args.n_labels, args.select_policy_from_plane,
                                       args.val_loss_factor, args.spolicy_loss_factor)
    elif args.model_type == "risev3.3":
        symbol = get_rise_v33_symbol(args.channels_policy_head, args.n_labels, args.select_policy_from_plane,
                                       args.val_loss_factor, args.spolicy_loss_factor)
    else:
        raise NotImplementedError

    model = mx.mod.Module(symbol=symbol, context=mx.cpu(), label_names=['value_label', 'policy_label'])
    model.bind(for_training=True, data_shapes=[('data', (1, args.input_shape[0], args.input_shape[1], args.input_shape[2]))],
               label_shapes=val_iter.provide_label)
    model.init_params(mx.initializer.Xavier(rnd_type='uniform', factor_type='avg', magnitude=2.24))

    pass


def main():
    """
    Main function which is executed on start-up

    Exemplary call: e.g. TODO
    :return:
    """
    args = parse_args(sys.argv[1:])
    logging.basicConfig(level=logging.INFO)
    generate_random_nn(args)


if __name__ == '__main__':
    main()
