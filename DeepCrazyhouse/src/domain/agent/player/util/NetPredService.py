"""
@file: NetPredService.py
Created on 13.10.18
@project: crazy_ara_refactor
@author: queensgambit

Please describe what the content of this file is about
"""

from DeepCrazyhouse.src.domain.agent.NeuralNetAPI import NeuralNetAPI
from multiprocessing import Barrier, Pipe, connection
import logging
from threading import Thread
import mxnet as mx
import numpy as np
from DeepCrazyhouse.src.domain.crazyhouse.output_representation import NB_LABELS, LABELS
from time import time
import cython


class NetPredService:

    def __init__(self, pipe_endings: [connection], net: NeuralNetAPI, batch_size, batch_state_planes: np.ndarray,
                 batch_value_results: np.ndarray, batch_policy_results: np.ndarray):
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
        #:param enable_timeout: Decides wether to enable a timout if a batch didn't occur under 1 second.
        """
        self.net = net
        self.my_pipe_endings = pipe_endings

        self.running = False
        self.thread_inference = Thread(target=self._provide_inference, args=(pipe_endings,), daemon=True)
        self.batch_size = batch_size

        self.batch_state_planes = batch_state_planes
        self.batch_value_results = batch_value_results
        self.batch_policy_results = batch_policy_results


    #@cython.boundscheck(False)
    #@cython.wraparound(False)
    def _provide_inference(self, pipe_endings):

        print('provide inference...')
        #use_random = True

        #cdef double[:, :, :, ::1] batch_state_planes_view = self.batch_state_planes
        #cdef double[::1] batch_value_results_view = self.batch_value_results
        #cdef double[:, ::1] batch_policy_results = self.batch_policy_results

        send_batches = False #True

        while self.running is True:

            filled_pipes = connection.wait(pipe_endings)

            if filled_pipes:

                if True or len(filled_pipes) >= self.batch_size: # 1

                        if send_batches is True:
                            planes_batch = []
                            pipes_pred_output = []

                            for pipe in filled_pipes[:self.batch_size]:
                                while pipe.poll():
                                    planes_batch.append(pipe.recv())
                                    pipes_pred_output.append(pipe)

                            # logging.debug('planes_batch length: %d %d' % (len(planes_batch), len(filled_pipes)))
                            state_planes_mxnet = mx.nd.array(planes_batch, ctx=self.net.get_ctx())
                        else:
                            planes_ids = []
                            pipes_pred_output = []

                            for pipe in filled_pipes[:self.batch_size]:
                                while pipe.poll():
                                    planes_ids.append(pipe.recv())
                                    pipes_pred_output.append(pipe)

                            #logging.debug('planes_batch length: %d %d' % (len(planes_batch), len(filled_pipes)))
                            state_planes_mxnet = mx.nd.array(self.batch_state_planes[planes_ids], ctx=self.net.get_ctx())


                        #print(len(state_planes_mxnet))
                        executor = self.net.executors[len(state_planes_mxnet)-1]
                        pred = executor.forward(is_train=False, data=state_planes_mxnet)
                        #pred = self.net.get_net()(state_planes_mxnet)
                        #print('pred: %.3f' % (time()-t_s)*1000)
                        #t_s = time()

                        value_preds = pred[0].asnumpy()

                        # renormalize to [0,1]
                        #value_preds += 1
                        #value_preds /= 2

                        # for the policy prediction we still have to apply the softmax activation
                        #  because it's not done by the neural net
                        #policy_preds = pred[1].softmax().asnumpy()
                        policy_preds = pred[1].softmax().asnumpy()

                        #if use_random is True:
                        #    value_preds = np.random.random(len(filled_pipes))
                        #    policy_preds = np.random.random((len(filled_pipes), NB_LABELS))

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

                        #print('send back res: %.3f' % (time()-t_s)*1000)

    def start(self):
        print('start inference thread...')
        self.running = True
        self.time_start = time()
        self.thread_inference.start()
        print('self.thread_inference.isAlive()', self.thread_inference.isAlive())
