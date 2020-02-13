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

    parser.add_argument("--sym-file", type=str, default="model-symbol.json",
                        help="symbol file of the network which describes the architecture (default: model-symbol.json")
    parser.add_argument('--params-file', type=str, default="model.params",
                        help='weight file of the network which holds the neural network parameters'
                             ' (default: model.params)')
    parser.add_argument("--onnx-file", type=str, default="model.onnx",
                        help="Output file after converting the model (default: model.onnx)")
    parser.add_argument("--input-shape", type=int, nargs="*", default=None,
                        help="Input shape of the neural network. Arguments need to be passed as a list"
                             'e.g. pass "--input-shape 1 34 8 8" for the default crazyhouse input (default: None)')
    parser.add_argument("--output-names", type=str, nargs="*", default=None,
                        help="Output layer names of the neural network. Arguments need to be passed as a list"
                             'e.g. pass "--output-names value_out_output policy_out_output" or '
                             '"--output-names value_tanh0_output flatten0_output" for the default chess networks '
                             '(default: None)')
    parser.add_argument("--validate", default=False, action="store_true",
                        help="If true, the ONNX model is validated after conversion (default: False)")

    args = parser.parse_args(cmd_args)

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


def main():
    """
    Main function which is executed on start-up

    Exemplary call:
    python convert_to_onnx.py --sym-file ./model-os-96-risev2/model/model-0.44052-1.08962-0.777-0.768-symbol.json\
     --params-file ./model-os-96-risev2/model/model-0.44052-1.08962-0.777-0.768-0096.params --input-shape 1 34 8 8\
      --onnx-file model-os-96-bsize-1.onnx --validate --output-names value_tanh0_output flatten0_output
    :return:
    """

    args = parse_args(sys.argv[1:])
    logging.basicConfig(level=logging.INFO)
    net = convert_mxnet_model_to_gluon(args.sym_file, args.params_file, args.output_names)

    net.export("model")
    # convert the gluon mxnet model into onnx
    onnx_model_path = onnx_mxnet.export_model("model-symbol.json", "model-0000.params", [args.input_shape], np.float32,
                                              args.onnx_file)
    logging.info("Exported model to: %s" % onnx_model_path)

    # delete temporary export files
    os.remove("model-0000.params")
    os.remove("model-symbol.json")

    if args.validate:
        from onnx import checker
        import onnx

        # load the ONNX-model
        model_proto = onnx.load_model(onnx_model_path)

        # check if the converted ONNX-protobuf is valid
        checker.check_graph(model_proto.graph)


if __name__ == '__main__':
    main()
