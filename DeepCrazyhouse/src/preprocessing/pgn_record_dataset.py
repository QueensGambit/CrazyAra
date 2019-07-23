"""
@file: PGNRecordDataset.py
Created on 25.09.18
@project: crazy_ara_refactor
@author: queensgambit

Utility class to load the rec dataset in the training loop of the CNN
"""
import os
import zlib
from mxnet.gluon.data.dataset import recordio, Dataset
import numpy as np
from DeepCrazyhouse.configs.main_config import main_config
from DeepCrazyhouse.src.domain.util import MATRIX_NORMALIZER


def __getitem__(self, idx):
    return self._record.read_idx(self._record.keys[idx])  # Access to a protected member


class PGNRecordDataset(Dataset):
    """ Everything related to loading the record dataset into the CNN"""

    def __init__(self, dataset_type, input_shape, normalize=True):
        """
        Constructor of the PGNRecordDataset class

        :param input_shape: Data shape of the plane representation
        :param normalize: If true the inputs will be normalized to [0., 1.]
        """

        # make sure that correct dataset_type has been selected
        # note that dataset type is stored in a folder with its time stamp
        dataset_types = {"train", "val", "test", "mate_in_one"}
        if dataset_type not in {"train", "val", "test", "mate_in_one"}:
            raise Exception(
                'Invalid dataset type "%s" given. It must be one of those: %s' % (dataset_type, dataset_types)
            )

        filename = main_config["rec_dir"] + dataset_type + ".rec"
        self.idx_file = os.path.splitext(filename)[0] + ".idx"
        self.filename = filename

        # super(PGNRecordDataset, self).__init__(filename)
        self._record = recordio.MXIndexedRecordIO(self.idx_file, self.filename, "r")
        self.input_shape = input_shape
        self.input_shape_flatten = input_shape[0] * input_shape[1] * input_shape[2]
        self.normalize = normalize

    def __getitem__(self, idx):
        """
        Overwrites the __getitem__ method from RecordFileDataset
        Each threads loads an individual data sample from the .rec file

        :param idx: String buffer index to load
        :return: x - plane representation
                y_value - value output (between -1, 1)
                y_policy - policy vector
        """
        item = self._record.read_idx(self._record.keys[idx])
        header, game = recordio.unpack(item)
        buf = zlib.decompress(game)
        x = np.frombuffer(buf, dtype=np.int16)[: self.input_shape_flatten].reshape(self.input_shape).astype(np.float32)

        if self.normalize:
            x *= MATRIX_NORMALIZER

        y_value = header[1][0]
        y_policy = header[1][1]
        return x, y_value, y_policy

    def __len__(self):
        return len(self._record.keys)

    def _get_data(self):
        raise NotImplementedError
