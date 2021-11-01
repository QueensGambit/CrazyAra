"""
@file: change_onnx.py
Created on 30.10.21
@project: CrazyAra
@author: queensgambit

Allows changing the batch size of an already existing ONNX-model.

Based on https://github.com/onnx/keras-onnx/issues/605
 https://github.com/sitting-duck/stuff/blob/32b77388c54c5ca5f5039365824b15483fc22fc2/ai/onnxruntime/edit.py
"""

import onnx
import os
import argparse
from glob import glob
import logging
import sys


def change_input_dim(model, init_bsize: int, target_bsize: int, dynamic: bool):
    """
    Changes the input and output dimension for a given ONNX model.
    Use some symbolic name "N" for dynamic==True and not used for any other dimension or an actual value "target_bsize".
    """
    sym_batch_dim = "N"

    inputs = model.graph.input
    outputs = model.graph.output
    for values in [inputs, outputs]:
        for value in values:
            # Checks omitted.This assumes that all inputs are tensors and have a shape with first dim.
            # Add checks as needed.
            dim1 = value.type.tensor_type.shape.dim[0]

            if dim1.dim_value == init_bsize:
                if dynamic:
                    dim1.dim_param = sym_batch_dim
                else:
                    dim1.dim_value = target_bsize


def parse_args(cmd_args: list):
    """
    Parses command-line argument and returns them as a dictionary object
    :param cmd_args: Command-line arguments (sys.argv[1:])
    :return: Parsed arguments as dictionary object
    """
    parser = argparse.ArgumentParser(description='ONNX Batch Size Modifier')

    parser.add_argument("--model-dir", type=str, default="./model",
                        help="Model directory which contains the .onnx file."
                             "It is assumed that the onnx-file has 'bsize-' in its file name.")
    parser.add_argument("--init-bsize", type=int, default=1,
                        help="Initial batch size which defines the model to load (default: 1),")
    parser.add_argument("--target-bsize", type=int, default=64,
                        help="Target batch size in which the model will be converted (default: 64).")
    parser.add_argument("--dynamic", default=False, action="store_true",
                        help="If true, the ONNX model will use a dynamic batch size and ignore target-bsize "
                             "(default: False).")

    args = parser.parse_args(cmd_args)

    if not os.path.isdir(args.model_dir):
        raise Exception("The given directory %s does not exist." % args.model_dir)

    args.onnx_file_path = glob(args.model_dir + f"/*bsize-{args.init_bsize}*.onnx")[0]

    if not os.path.isfile(args.onnx_file_path):
        raise Exception("The given file path %s does not exist." % args.onnx_file_path)

    if args.dynamic:
        args.onnx_export_file_path = args.onnx_file_path.replace(f"bsize-{args.init_bsize}", "")
    else:
        args.onnx_export_file_path = args.onnx_file_path.replace(f"bsize-{args.init_bsize}",
                                                                 f"bsize-{args.target_bsize}")

    return args


def main():
    """
    Main function which is executed on start-up

    Exemplary call: e.g. Convert batch-size 16 to batch-size 64
    python change_onnx_bsize.py --model-dir ./model --init-bsize 16 --target-bsize 64

    e.g. make batch-size dynamic
    python change_onnx_bsize.py --model-dir ./model --dynamic

    :return:
    """
    args = parse_args(sys.argv[1:])
    logging.basicConfig(level=logging.INFO)

    model = onnx.load(args.onnx_file_path)
    change_input_dim(model, args.init_bsize, args.target_bsize, args.dynamic)
    onnx.save(model, args.onnx_export_file_path)

    logging.info("Exported ONNX model to:", args.onnx_export_file_path)


if __name__ == '__main__':
    main()
