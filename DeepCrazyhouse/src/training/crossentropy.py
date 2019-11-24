"""
@file: cross_entropy.py
Created on 16.11.19
@project: CrazyAra
@author: queensgambit

Definition of Cross-Entropy-Loss with non-sparse targets.
The default SoftmaxOutputLayer() only supports one-hot encoded labels.
The target definition distribution is meant to have values in [0,1].
This operator has no effect in the forward pass and is only meant to calculate the gradient.

Example usage:
    policy_out = mx.sym.SoftmaxOutput(data=data)

can be replaced by:
    policy_out = mx.sym.SoftmaxActivation(data=data)
    policy_out = mx.symbol.Custom(data=data, op_type='CrossEntropyLoss')

The operator doesn't require to rebuild MXNet, and can't be used within C++ because of this.
To load the model for inference in C++ you need to remove the CrossEntropyLayer after training.

Parts of this code is based on:
https://github.com/miraclewkf/multilabel-MXNet/blob/master/train_multilabel.py
"""

import mxnet as mx


class CrossEntropyLoss(mx.operator.CustomOp):
    """
    Output layer for the gradient cross-entropy loss with non-sparse targets:
    Loss is calculated by:
    L = - sum (y_true_i * log(y_pred_i))

    Derivative:
    d L / d y_pred_i = - (y_true_i / y_pred_i)
    """

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        y_pred = in_data[0]
        y_true = in_data[1]
        grad = -y_true / (y_pred + 1e-12)

        self.assign(in_grad[0], req[0], grad)


@mx.operator.register("CrossEntropyLoss")
class CrossEntropyProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(CrossEntropyProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def create_operator(self, ctx, shapes, dtypes):
        return CrossEntropyLoss()

    def infer_shape(self, in_shape):
        output_shape = in_shape[0]
        return in_shape, [output_shape], []
