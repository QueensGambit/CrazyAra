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


class NetPredService:

    def __init__(self, pipe_endings: [connection], net: NeuralNetAPI):
        self.net = net
        self.my_pipe_endings = pipe_endings

        self.running = False
        self.thread_inference = Thread(target=self._provide_inference, args=(pipe_endings,), daemon=True)

    def _provide_inference(self, pipe_endings):

        print('provide inference...')

        self.running = True

        while self.running is True:

            filled_pipes = connection.wait(pipe_endings)

            if not filled_pipes:
                continue

            planes_batch = []
            pipes_pred_output = []

            for pipe in filled_pipes:
                while pipe.poll():
                    planes_batch.append(pipe.recv())
                    pipes_pred_output.append(pipe)

            #logging.debug('planes_batch length: %s' % len(planes_batch))
            planes_batch = mx.nd.array(planes_batch, ctx=self.net.get_ctx())
            pred = self.net.get_net()(planes_batch)

            value_preds = pred[0].asnumpy()
            # for the policy prediction we still have to apply the softmax activation because it's not done by the neural net
            policy_preds = pred[1].softmax().asnumpy()

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

