"""
@file: quantize_model.py
Created on 02.01.2021
@project: CrazyAra
@author: queensgambit

Script for creating INT8 quanitzation weights for the MXNet-CPU backend.

References:
https://cwiki.apache.org/confluence/display/MXNET/MXNet+Graph+Optimization+and+Quantization+based+on+subgraph+and+MKL-DNN
https://medium.com/apache-mxnet/model-quantization-for-production-level-neural-network-inference-f54462ebba05
"""

import sys
import glob
import mxnet as mx
from mxnet.contrib.quantization import *
sys.path.insert(0,'../../../')
from DeepCrazyhouse.configs.main_config import main_config
from DeepCrazyhouse.src.preprocessing.dataset_loader import load_pgn_dataset


def remove_labels(symbol, value_output_name, policy_output_name):
    """
    Removes the output labels for the mxnet symbol
    :param symbol: Mxnet symbol
    :param value_output_name: Layer name of the value output
    :param policy_output_name: Layer name of the policy output
    :return: symbol without labels
    """
    value_out = symbol.get_internals()[value_output_name]
    policy_out = symbol.get_internals()[policy_output_name]
    policy_out = mx.sym.SoftmaxActivation(data=policy_out, name='softmax')
    # group value_out and policy_out together
    return mx.symbol.Group([value_out, policy_out])


def save_params(fname, arg_params, aux_params, logger=None):
    """
    Saves the given parameters to a file
    :param fname: Filename
    :param arg_params: arg params / main parameters
    :param aux_params: auxiliary parameters
    :param logger: Logger object
    :return: None
    """
    if logger is not None:
        logger.info('Saving params into file at %s' % fname)
    save_dict = {('arg:%s' % k): v.as_in_context(cpu()) for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k): v.as_in_context(cpu()) for k, v in aux_params.items()})
    mx.nd.save(fname, save_dict)


def save_symbol(fname, sym, logger=None):
    """
    Saves the given symbol to a file
    :param fname: Filename
    :param sym: Symbol object
    :param logger: Logger object
    :return: None
    """
    if logger is not None:
        logger.info('Saving symbol into file at %s' % fname)
    sym.save(fname)


def main():
    # config
    batch_size = 32
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    ctx = mx.cpu(0)
    calib_mode = 'entropy'
    excluded_sym_names = ['stem_conv0']
    num_calib_batches = 128
    quantized_dtype = 'int8'


    symbol_path = glob.glob(main_config["model_architecture_dir"] + "*")[0]
    params_path = glob.glob(main_config["model_weights_dir"] + "*")[0]
    print("symbol_path:", symbol_path)
    print("params_path:", params_path)

    epoch = int(params_path[-11:-7])
    print(epoch)

    # load calibration dataset
    _, x_train, yv_train, yp_train, plys_to_end, _ = load_pgn_dataset(normalize=True)
    calib_data = mx.io.NDArrayIter({'data': x_train},
                                   {},
                                   batch_size,
                                   shuffle=True)

    # construct the model name based on the parameter file
    prefix = symbol_path.split("/")[-1].replace("-symbol.json", "")
    sym = mx.sym.load(symbol_path)
    sym = remove_labels(sym, main_config['value_output']+'_output', main_config['policy_output']+'_output')

    # https://github.com/apache/incubator-mxnet/issues/6951
    save_dict = mx.nd.load(params_path)
    arg_params = {}
    aux_params = {}
    for key, val in save_dict.items():
        param_type, name = key.split(":", 1)
        if param_type == "arg":
            arg_params[name] = val
        if param_type == "aux":
            aux_params[name] = val

    # quantize model
    sym = sym.get_backend_symbol('MKLDNN_QUANTIZE')
    label_names = []
    qsym, qarg_params, aux_params = quantize_model(sym=sym, arg_params=arg_params, aux_params=aux_params,
                                                    ctx=ctx, excluded_sym_names=excluded_sym_names,
                                                    excluded_op_names=excluded_sym_names,
                                                    calib_mode=calib_mode, calib_data=calib_data,
                                                    num_calib_examples=num_calib_batches * batch_size,
                                                    quantized_dtype=quantized_dtype,
                                                    quantize_mode='smart',
                                                    label_names=label_names,
                                                    logger=logger)

    sym_name = '%s-symbol.json' % (prefix + '-int8')
    save_symbol(sym_name, qsym, logger)
    param_name = '%s-%04d.params' % (prefix + '-int8', epoch)
    save_params(param_name, qarg_params, aux_params, logger)


if __name__ == '__main__':
    main()
