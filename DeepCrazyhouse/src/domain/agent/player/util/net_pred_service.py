"""
@file: net_pred_service.py
Created on 13.10.18
@project: crazy_ara_refactor
@author: queensgambit

# TODO: description for the file
"""
from multiprocessing import connection
from threading import Thread
from time import time
import mxnet as mx
import numpy as np
from DeepCrazyhouse.src.domain.agent.neural_net_api import NeuralNetAPI
from DeepCrazyhouse.src.domain.variants.plane_policy_representation import FLAT_PLANE_IDX


class NetPredService:  # Too many instance attributes (9/7) - Too few public methods (1/2)
    """ Service which provides the network predictions to the workers"""

    def __init__(
        self,
        pipe_endings: [connection],
        net: NeuralNetAPI,
        batch_size,
        batch_state_planes: np.ndarray,
        batch_value_results: np.ndarray,
        batch_policy_results: np.ndarray,
    ):  # Too many arguments (7/5)
        """

        :param pipe_endings: List of pip endings which are for communicating with the thread workers.
        :param net: Neural Network API object which provides the reference for the neural network.
        :param batch_size: Constant batch_size used for inference.
        :param batch_state_planes: Shared numpy memory in which all threads set their state plane request for the
                                   prediction service. Each threads has it's own channel.
        :param batch_value_results: Shared numpy memory in which the value results of all threads are stored.
                                    Each threads has it's own channel.
        :param batch_policy_results: Shared numpy memory in which the policy results of all threads are stored.
                                    Each threads has it's own channel.
        """
        self.net = net
        self.my_pipe_endings = pipe_endings
        self.running = False
        self.time_start = None
        self.thread_inference = Thread(target=self._provide_inference, args=(pipe_endings,), daemon=True)
        self.batch_size = batch_size
        self.batch_state_planes = batch_state_planes
        self.batch_value_results = batch_value_results
        self.batch_policy_results = batch_policy_results

    def _provide_inference(self, pipe_endings):
        """
        Start an inference thread which listens for incoming requests for the NN
        :param pipe_endings: List of pipe endings
        :return:
        """
        send_batches = False

        while self.running:
            filled_pipes = connection.wait(pipe_endings)
            if filled_pipes:
                planes_ids = []
                if send_batches:
                    planes_batch = []
                    pipes_pred_output = []
                    for pipe in filled_pipes[: self.batch_size]:
                        while pipe.poll():
                            planes_batch.append(pipe.recv())
                            pipes_pred_output.append(pipe)
                    state_planes_mxnet = mx.nd.array(planes_batch, ctx=self.net.get_ctx())
                else:
                    pipes_pred_output = []
                    for pipe in filled_pipes[: self.batch_size]:
                        while pipe.poll():
                            planes_ids.append(pipe.recv())
                            pipes_pred_output.append(pipe)
                    state_planes_mxnet = mx.nd.array(self.batch_state_planes[planes_ids], ctx=self.net.get_ctx())
                pred = self.net.executors[len(state_planes_mxnet) - 1].forward(is_train=False, data=state_planes_mxnet)
                value_preds = pred[0].asnumpy()

                if self.net.select_policy_form_planes:
                    # when trained with mxnet symbol then softmax is already applied
                    policy_preds = pred[1].asnumpy()
                    policy_preds = policy_preds[:, FLAT_PLANE_IDX]
                else:
                    # for the policy prediction we still have to apply the softmax activation
                    #  because it's not done by the neural net if trained in gluon style
                    policy_preds = pred[1].softmax().asnumpy()

                # send the predictions back to the according workers
                for i, pipe in enumerate(pipes_pred_output):
                    if send_batches is True:
                        pipe.send([value_preds[i], policy_preds[i]])
                    else:
                        # get the according channel index for setting the result
                        channel_idx = planes_ids[i]
                        # set the value result
                        self.batch_value_results[channel_idx] = value_preds[i]
                        self.batch_policy_results[channel_idx] = policy_preds[i]
                        # give the thread the signal that the result has been set by sending back his channel_idx
                        pipe.send(channel_idx)

    def start(self):
        """ Start the thread inference and the time"""
        print("start inference thread...")
        self.running = True
        self.time_start = time()
        self.thread_inference.start()
        print("self.thread_inference.isAlive()", self.thread_inference.isAlive())
