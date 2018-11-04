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


class NetPredService:

    def __init__(self, pipe_endings: [connection], net: NeuralNetAPI, batch_size):
        self.net = net
        self.my_pipe_endings = pipe_endings

        self.running = False
        self.thread_inference = Thread(target=self._provide_inference, args=(pipe_endings,), daemon=True)
        self.batch_size = batch_size

    def _provide_inference(self, pipe_endings):

        print('provide inference...')
        use_random = False
        self.running = True

        while self.running is True:

            filled_pipes = connection.wait(pipe_endings)

            if filled_pipes and len(filled_pipes) >= self.batch_size:

                planes_batch = []
                pipes_pred_output = []

                for pipe in filled_pipes[:self.batch_size]:
                    while pipe.poll():
                        planes_batch.append(pipe.recv())
                        pipes_pred_output.append(pipe)

                #logging.debug('planes_batch length: %d %d' % (len(planes_batch), len(filled_pipes)))
                planes_batch = mx.nd.array(planes_batch, ctx=self.net.get_ctx())
                #pred = self.net.get_net()(planes_batch)
                pred = self.net.get_executor().forward(is_train=False, data=planes_batch)

                value_preds = pred[0].asnumpy()

                # for the policy prediction we still have to apply the softmax activation because it's not done by the neural net
                policy_preds = pred[1].softmax().asnumpy()

                if use_random is True:
                    value_preds = np.random.random(len(filled_pipes))
                    policy_preds = np.random.random((len(filled_pipes), NB_LABELS))

                # send the predictions back to the according workers
                for i, pipe in enumerate(pipes_pred_output):
                    pipe.send([value_preds[i], policy_preds[i]])

    def start(self):
        print('start inference thread...')
        self.running = True
        self.thread_inference.start()
        print('self.thread_inference.isAlive()', self.thread_inference.isAlive())

    def stop(self):

        logging.info('Send quit message to infernce thread...')
        self.pipe_controller_parent.send('quit')

