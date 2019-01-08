"""
@file: PGNRecordDataset.py
Created on 25.09.18
@project: crazy_ara_refactor
@author: queensgambit

Utility class to load the rec dataset in the training loop of the CNN
"""
import os
import zlib
import mxnet as mx
from mxnet.gluon.data.dataset import recordio, Dataset
import numpy as np
from DeepCrazyhouse.configs.main_config import main_config
from DeepCrazyhouse.src.domain.util import normalize_input_planes


def __getitem__(self, idx):
    return self._record.read_idx(self._record.keys[idx])


class PGNRecordDataset(Dataset):
    def __init__(self, dataset_type, input_shape, normalize=True):
        """
        Constructor of the PGNRecordDataset class

        :param filename: Name of the .rec file to load (the .rec file must be stored in main_config['rec_dir']
        :param input_shape: Data shape of the plane representation
        :param normalize: If true the inputs will be normalized to [0., 1.]
        """

        # make sure that correct dataset_type has been selected
        # note that dataset type is stored in a folder with its time stamp
        dataset_types = ["train", "val", "test", "mate_in_one"]
        if dataset_type not in ["train", "val", "test", "mate_in_one"]:
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
        Each threads loads an individual datasample from the .rec file

        :param idx: String buffer index to load
        :return: x - plane representation
                yv - value output (between -1, 1)
                yp - policy vector
        """
        item = self._record.read_idx(self._record.keys[idx])

        header, s = mx.recordio.unpack(item)
        buf = zlib.decompress(s)

        data = np.frombuffer(buf, dtype=np.int16)

        x = data[: self.input_shape_flatten]
        x = x.reshape(self.input_shape)
        x = x.astype(np.float32)

        if self.normalize is True:
            normalize_input_planes(x)

        yv = header[1][0]
        yp = header[1][1]

        return x, yv, yp

    def __len__(self):
        return len(self._record.keys)
