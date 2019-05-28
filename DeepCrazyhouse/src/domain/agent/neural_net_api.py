"""
@file: neural_net_api.py
Changed last on 16.01.19
@project: crazy_ara_cleaning
@author: queensgambit and matuiss2

Make the project NN easier to use

This file contains wrappers for NN handling
"""
import glob
import os
from multiprocessing import Queue
import mxnet as mx
import numpy as np
from DeepCrazyhouse.configs.main_config import main_config
from DeepCrazyhouse.src.domain.crazyhouse.constants import BOARD_HEIGHT, BOARD_WIDTH, NB_CHANNELS_FULL, NB_LABELS
from DeepCrazyhouse.src.domain.crazyhouse.plane_policy_representation import FLAT_PLANE_IDX


class NeuralNetAPI:
    """Groups every a lot of helpers to be used on NN handling"""

    def __init__(self, ctx="cpu", batch_size=1, select_policy_form_planes: bool = True,
                 model_architecture_dir="default", model_weights_dir="default"):
        """
        Constructor
        :param ctx: Context used for inference "cpu" or "gpu"
        :param batch_size: Batch size used for inference
        """
        self.batch_size = batch_size

        if not os.path.isdir(main_config["model_architecture_dir"]):
            raise Exception(
                "The given model_architecture_dir at: " + main_config["model_architecture_dir"] + " wasn't found."
            )
        if not os.path.isdir(main_config["model_weights_dir"]):
            raise Exception("The given model_weights_dir at: " + main_config["model_weights_dir"] + " wasn't found.")

        if model_architecture_dir == "default":
            self.symbol_path = glob.glob(main_config["model_architecture_dir"] + "*")[0]
        else:
            self.symbol_path = glob.glob(model_architecture_dir + "*")[0]

        if model_weights_dir == "default":
            self.params_path = glob.glob(main_config["model_weights_dir"] + "*")[0]
        else:
            self.params_path = glob.glob(model_weights_dir + "*")[0]
        # make sure the needed files have been found
        if self.symbol_path is None or ".json" not in self.symbol_path:
            raise Exception(
                "No symbol file (.json) was found in your given model_architecture_dir: "
                + main_config["model_architecture_dir"]
                + '. Please make sure that the path has a "/" at the end of the path.'
            )
        if self.params_path is None or ".params" not in self.params_path:
            raise Exception(
                "No params file (.params) was found in your given model_weights_dir: "
                + main_config["model_weights_dir"]
                + '. Please make sure that the path has a "/" at the end of the path.'
            )

        print("self.symbol_path:", self.symbol_path)
        print("self.params_path:", self.params_path)
        # construct the model name based on the parameter file
        self.model_name = self.params_path.split("/")[-1].replace(".params", "")
        sym = mx.sym.load(self.symbol_path)
        # https://github.com/apache/incubator-mxnet/issues/6951
        save_dict = mx.nd.load(self.params_path)
        arg_params = {}
        aux_params = {}
        for key, val in save_dict.items():
            param_type, name = key.split(":", 1)
            if param_type == "arg":
                arg_params[name] = val
            if param_type == "aux":
                aux_params[name] = val
        # set the context on CPU, switch to GPU if there is one available
        if ctx == "cpu":
            self.ctx = mx.cpu()
        elif ctx == "gpu":
            self.ctx = mx.gpu()
        else:
            raise Exception("Unavailable ctx mode given %s. You must either select 'cpu' or 'gpu'" % ctx)
        # define batch_size times executor objects which are used for inference
        # one executor object is used for the currently requested batch batch length
        # the requested batch length is variable and at maximum the given batch_size
        self.executors = []
        for i in range(batch_size):
            executor = sym.simple_bind(
                ctx=self.ctx,
                # add a new length for each size starting with 1
                data=(i + 1, NB_CHANNELS_FULL, BOARD_HEIGHT, BOARD_WIDTH),
                grad_req="null",
                force_rebind=True,
            )
            executor.copy_params_from(arg_params, aux_params)
            self.executors.append(executor)

        # check if the current net uses a select_policy_from_planes style
        output_dict = self.executors[0].output_dict
        for idx, key in enumerate(output_dict):
            print(key)
            # the policy output is always the 2nd one
            if idx == 1 and output_dict[key].shape[1] != NB_LABELS:
                self.select_policy_form_planes = select_policy_form_planes
            else:
                self.select_policy_form_planes = False

    def predict_single(self, x):
        """
        Gets the model prediction of a single input sample.
        This function supports the 'keras' and 'mxnet' as its model type.
        :param x: Plane representation of a single board state
        :return: [Value Prediction, Policy Prediction] as a list of numpy arrays
        """

        queue = Queue()  # start a subprocess
        self.predict_single_thread(queue, x)
        return queue.get()

    def predict_single_thread(self, queue, x):
        """
        Gets the model prediction of a single input sample.
        This function supports the 'keras' and 'mxnet' as its model type.
        :param x: Plane representation of a single board state
        :param queue: Stores the return values
        :return: [Value Prediction, Policy Prediction] as a list of numpy arrays
        """
        # choose the first executor object which support length 1
        pred = self.executors[0].forward(is_train=False, data=np.expand_dims(x, axis=0))
        if self.select_policy_form_planes:
            policy_preds = pred[1].asnumpy()
            policy_preds = policy_preds[:, FLAT_PLANE_IDX]
        else:
            policy_preds = pred[1].softmax().asnumpy()

        queue.put([pred[0].asnumpy()[0], policy_preds[0]])

    def get_batch_size(self):
        """Make the batch_size public access"""
        return self.batch_size

    def get_ctx(self):
        """Make the ctx public access"""
        return self.ctx

    def get_model_name(self):
        """Make the model_name public access"""
        return self.model_name
