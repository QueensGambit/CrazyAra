import glob
import os
from multiprocessing import Queue
import mxnet as mx
import numpy as np
from DeepCrazyhouse.configs.main_config import main_config
from DeepCrazyhouse.src.domain.crazyhouse.constants import BOARD_HEIGHT, BOARD_WIDTH, NB_CHANNELS_FULL


class NeuralNetAPI:
    def __init__(self, ctx="cpu", batch_size=1):

        self.batch_size = batch_size

        if os.path.isdir(main_config["model_architecture_dir"]) is False:
            raise Exception(
                "The given model_architecture_dir at: " + main_config["model_architecture_dir"] + " wasn't found."
            )
        if os.path.isdir(main_config["model_weights_dir"]) is False:
            raise Exception("The given model_weights_dir at: " + main_config["model_weights_dir"] + " wasn't found.")

        self.symbol_path = glob.glob(main_config["model_architecture_dir"] + "*")[0]
        self.params_path = glob.glob(main_config["model_weights_dir"] + "*")[0]

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
            tp, name = key.split(":", 1)
            if tp == "arg":
                arg_params[name] = val
            if tp == "aux":
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

    def predict_single(self, board_state):
        """
        Gets the model prediction of a single input sample.
        This function supports the 'keras' and 'mxnet' as its model type.
        :param board_state: Plane representation of a single board state
        :return: [Value Prediction, Policy Prediction] as a list of numpy arrays
        """

        # start a subprocess
        queue = Queue()
        self.predict_single_thread(queue, board_state)
        out = queue.get()

        return out

    def predict_single_thread(self, queue, board_state):
        """
        Gets the model prediction of a single input sample.
        This function supports the 'keras' and 'mxnet' as its model type.
        :param board_state: Plane representation of a single board state
        :param queue: Stores the return values
        :return: [Value Prediction, Policy Prediction] as a list of numpy arrays
        """
        out = [None, None]

        # choose the first executor object which support length 1
        pred = self.executors[0].forward(is_train=False, data=np.expand_dims(board_state, axis=0))

        out[0] = pred[0].asnumpy()[0]
        # when using a gluon model you still have to apply a softmax activation after the forward pass
        out[1] = pred[1].softmax().asnumpy()[0]

        queue.put(out)

    def get_batch_size(self):
        return self.batch_size

    def get_ctx(self):
        return self.ctx

    def get_model_name(self):
        return self.model_name
