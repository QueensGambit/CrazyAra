"""
@file: builder_util.py
Created on 25.09.18
@project: crazy_ara_refactor
@author: queensgambit

Please describe what the content of this file is about
"""

from mxnet.gluon.nn import Activation, AvgPool2D, MaxPool2D, PReLU, SELU, Swish, LeakyReLU


def get_act(act_type, **kwargs):
    if act_type in ["relu", "sigmoid", "softrelu", "softsign", "tanh"]:
        return Activation(act_type, **kwargs)
    elif act_type == "prelu":
        return PReLU(**kwargs)
    elif act_type == "selu":
        return SELU(**kwargs)
    elif act_type == "swish":
        return Swish(**kwargs)
    elif act_type == "lrelu":
        return LeakyReLU(alpha=0.2, **kwargs)
    else:
        raise NotImplementedError


def get_pool(pool_type, pool_size, strides, **kwargs):
    if pool_type == "maxpool":
        return MaxPool2D(pool_size=pool_size, strides=strides, **kwargs)
    elif pool_type == "avgpool":
        return AvgPool2D(pool_size=pool_size, strides=strides, **kwargs)
    else:
        raise NotImplementedError
