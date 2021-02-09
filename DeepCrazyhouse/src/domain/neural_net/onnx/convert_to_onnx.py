"""
@file: convert_to_onnx.py
Created on 07.02.20
@project: CrazyAra
@author: queensgambit

Utility script for cmd-line usage to convert a given MXNet model into the ONNX format.
The ONNX model can later be used to do inference with TensorRT or to load the model in a different DL-framework.

MXNet 1.5.1 only supports ONNX versions <= 1.3.0

Reference:
https://mxnet.apache.org/api/python/docs/tutorials/deploy/export/onnx.html
"""
import os
import sys
import argparse
import mxnet as mx
from glob import glob
import numpy as np
from mxnet.contrib import onnx as onnx_mxnet
import logging


def parse_args(cmd_args: list):
    """
    Parses command-line argument and returns them as a dictionary object
    :param cmd_args: Command-line arguments (sys.argv[1:])
    :return: Parsed arguments as dictionary object
    """
    parser = argparse.ArgumentParser(description='MXNet to ONN converter')

    parser.add_argument("--model-dir", type=str, default="./model",
                        help="model directory which contains the .param and .sym file")
    parser.add_argument("--onnx-file", type=str, default="model.onnx",
                        help="Output file after converting the model (default: model.onnx)")
    parser.add_argument("--input-shape", type=int, nargs="*", default=None,
                        help="Input shape of the neural network. Arguments need to be passed as a list"
                             'e.g. pass "--input-shape 34 8 8" for the default crazyhouse input (default: None)')
    parser.add_argument("--batch-sizes", type=int, nargs="*", default=None,
                        help="Batch sizes to export the neural network. Arguments need to be passed as a list"
                             'e.g. pass "--batch-sizes 1 8 16" to export separate models for the given batch-sizes'
                             ' in ONNX fromat (default: None)')
    parser.add_argument("--output-names", type=str, nargs="*", default=None,
                        help="Output layer names of the neural network. Arguments need to be passed as a list"
                             'e.g. pass "--output-names value_out_output policy_out_output" or '
                             '"--output-names value_tanh0_output flatten0_output" for the default chess networks '
                             '(default: None)')
    parser.add_argument("--validate", default=False, action="store_true",
                        help="If true, the ONNX model is validated after conversion (default: False)")

    args = parser.parse_args(cmd_args)

    if not os.path.isdir(args.model_dir):
        raise Exception("The given directory %s does not exist." % args.model_dir)
        
    args.sym_file = glob(args.model_dir + "/*.json")[0]
    args.params_file = glob(args.model_dir + "/*.params")[0]

    for file_path in [args.sym_file, args.params_file]:
        if not os.path.isfile(file_path):
            raise Exception("The given file path %s does not exist." % file_path)

    if args.input_shape is None:
        raise ValueError('Given input shape must not be "None"')

    # convert list to tuple
    args.input_shape = tuple(args.input_shape)
    return args


def convert_mxnet_model_to_gluon(sym_file: str, params_file: str, output_layer_names: list,
                                 input_data_name="data", input_dtype="float32", ctx=mx.cpu()):
    """
    Converts an exported MXNet model using the symbol-API into the MXNet-Gluon-format
    :param sym_file: Symbol file name of the network
    :param params_file: Parameter file name of the network
    :param output_layer_names: Names of the final network output layers (e.g. ["value_out_output", "policy_out_output"]
     or ["value_tanh0_output", "flatten0_output"])
    :param input_data_name: Data name for the input layer
    :param input_dtype: Data type for the neural network input
    :return: Gluon network object
    """
    symbol = mx.sym.load(sym_file)
    inputs = mx.sym.var(input_data_name, dtype=input_dtype)
    net_outputs = []
    for output_layer_name in output_layer_names:
        net_outputs.append(symbol.get_internals()[output_layer_name])
    sym = mx.symbol.Group(net_outputs)
    net = mx.gluon.SymbolBlock(sym, inputs)
    net.collect_params().load(params_file, ctx=ctx)
    return net


def convert_mxnet_model_to_onnx(sym_file, params_file, output_names, input_shape, batch_sizes, validate):
    """
    Converts the given model specified by symbol and param file to ONNX format.
    For parameters see: parse_args.
    """
    # convert the gluon mxnet model into onnx
    for batch_size in batch_sizes:
        onnx_file = params_file[:-7] + "-bsize-" + str(batch_size) + ".onnx"
        onnx_model_path = onnx_mxnet.export_model(sym_file, params_file,
                                                  [(batch_size, input_shape[0], input_shape[1], input_shape[2])],
                                                  np.float32, onnx_file)
        logging.info("Exported model to: %s" % onnx_model_path)

        if validate:
            from onnx import checker
            import onnx

            # load the ONNX-model
            model_proto = onnx.load_model(onnx_model_path)

            # check if the converted ONNX-protobuf is valid
            checker.check_graph(model_proto.graph)


def main():
    """
    Main function which is executed on start-up

    Exemplary call: e.g. crazyhouse model for releases <= 0.7.0
    python convert_to_onnx.py --model-dir ./model --input-shape 34 8 8 --batch-sizes 1 8 16\
    --onnx-file model-os-96-bsize-1.onnx --validate --output-names value_tanh0_output flatten0_output

    e.g. chess model for releases >= 0.8.0
    python3 convert_to_onnx.py --model-dir ./model --input-shape 39 8 8 --batch-sizes 1 8 16\
    --onnx-file model-bsize-1.onnx --validate --output-names value_out_output policy_out_output
    :return:
    """

    args = parse_args(sys.argv[1:])
    logging.basicConfig(level=logging.INFO)
    convert_mxnet_model_to_onnx(args.sym_file, args.params_file, args.output_names, args.input_shape, args.batch_sizes,
                                args.validate)


if __name__ == '__main__':
    main()
