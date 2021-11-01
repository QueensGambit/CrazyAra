"""
@file: change_onnx.py
Created on 30.10.21
@project: CrazyAra
@author: queensgambit

Allows changing the batch size of an already existing onnx-model.

Based on https://github.com/onnx/keras-onnx/issues/605
 https://github.com/sitting-duck/stuff/blob/32b77388c54c5ca5f5039365824b15483fc22fc2/ai/onnxruntime/edit.py
"""

import onnx

init_bsize = 16
target_bsize = 32


def change_input_dim(model):
    # Use some symbolic name not used for any other dimension
    sym_batch_dim = "N"
    # or an actal value
    actual_batch_dim = target_bsize

    # The following code changes the first dimension of every input to be batch-dim
    # Modify as appropriate ... note that this requires all inputs to
    # have the same batch_dim
    inputs = model.graph.input
    for input in inputs:
        # Checks omitted.This assumes that all inputs are tensors and have a shape with first dim.
        # Add checks as needed.
        dim1 = input.type.tensor_type.shape.dim[0]
        # update dim to be a symbolic value
        # dim1.dim_param = sym_batch_dim
        # or update it to be an actual value:
        # print("input.type.tensor_type.shape:", input.type.tensor_type.shape)
        # print("dim1.dim_value", dim1.dim_value)

        print("dim1.dim_value", dim1.dim_value)
        if dim1.dim_value == init_bsize:
            dim1.dim_value = actual_batch_dim


    outputs = model.graph.output

    for output in outputs:
        dim1 = output.type.tensor_type.shape.dim[0]
        # update dim to be a symbolic value
        # dim1.dim_param = sym_batch_dim
        # or update it to be an actual value:
        print("dim1.dim_value:", dim1.dim_value)
        if dim1.dim_value == init_bsize:
            dim1.dim_value = actual_batch_dim


def apply(transform, infile, outfile):
    model = onnx.load(infile)
    transform(model)
    onnx.save(model, outfile)


onnx_dir = "/sample/dir/"
onnx_model = f"model-1.19414-0.585-0513-v3.0-bsize-{init_bsize}.onnx"
onnx_file_path = onnx_dir + onnx_model
onnx_export_file_path = onnx_dir + onnx_model.replace(f"bsize-{init_bsize}", f"bsize-{target_bsize}")

apply(change_input_dim, onnx_file_path, onnx_export_file_path)
print("exported to:", onnx_export_file_path)

# for key in node_map.keys():
#     print("key: " + key)
#     if "Clip" in key:
#         print("removing key: " + str(key))
#         graph.node.remove(node_map[key])

# onnx.save(model, "netout.onnx")
