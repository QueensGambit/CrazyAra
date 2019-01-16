"""
@file: Planes2RecConverter
Created on 25.09.18
@project: deepcrazyhouse
@author: queensgambit

The Planes2RecConverter is used to transform the plane representation in a format which can be accessed
efficiently during training using multiprocessing

* This how you could read from a rec file (although it's recommended to use the PGNDataset class instead:

```python
import from mxnet.recordio import RecordIOHandle, MXIndexedRecordIO

record = mx.recordio.MXIndexedRecordIO('val.idx', 'val.rec', 'r')
for i in range(3):
    item = record.read_idx(i)
    header, s = mx.recordio.unpack(item)
    print(np.frombuffer(s, dtype=np.uint8))

record.close()
```
"""
import logging
import zlib
from glob import glob
from time import time
import mxnet as mx
from DeepCrazyhouse.configs.main_config import main_config
from DeepCrazyhouse.src.preprocessing.dataset_loader import load_pgn_dataset


class Planes2RecConverter:
    """ Transform the plane representation in a format that can be used easily during training"""
    def __init__(self, dataset_type="train"):

        # make sure that correct dataset_type has been selected
        # note that dataset type is stored in a folder with its time stamp
        if dataset_type == "train":
            self._import_dir = main_config["planes_train_dir"]
        elif dataset_type == "val":
            self._import_dir = main_config["planes_val_dir"]
        elif dataset_type == "test":
            self._import_dir = main_config["planes_test_dir"]
        elif dataset_type == "mate_in_one":
            self._import_dir = main_config["planes_mate_in_one_dir"]
        else:
            raise Exception(
                'Invalid dataset type "%s" given. It must be either "train", "val", "test" or "mate_in_one"'
                % dataset_type
            )

        self._dataset_type = dataset_type
        self._export_dir = main_config["rec_dir"] # all dataset types are export to a single .rec directory

    def convert_all_planes_to_rec(self):
        """
        Converts all part files from the via load_pgn_dataset() to a single .rec file

        :return:
        """

        # we must add '**/*' because we want to go into the time stamp directory
        plane_files = glob(self._import_dir + "**/*")
        # construct the export filepaths
        idx_filepath = "%s%s" % (self._export_dir, self._dataset_type + ".idx")
        rec_filepath = "%s%s" % (self._export_dir, self._dataset_type + ".rec")
        # create both an '.idx' and '.rec' file
        # the '.idx' file stores the indices to the string buffers
        # the '.rec' files stores the planes in a compressed binary string buffer format
        record = mx.recordio.MXIndexedRecordIO(idx_filepath, rec_filepath, "w")
        nb_parts = len(plane_files)
        idx = 0
        for part_id in range(nb_parts):

            t_s = time()
            logging.info("PART: %d", part_id)
            # load one chunk of the dataset from memory
            _, x, y_value, y_policy, _ = load_pgn_dataset(
                dataset_type=self._dataset_type,
                part_id=part_id,
                print_statistics=True,
                print_parameters=False,
                normalize=False,
            )

            # iterate over all board states aka. data samples in the file
            for position, value in enumerate(x):
                data = value.flatten()
                buf = zlib.compress(data.tobytes())
                # we only store the integer idx of the highest output
                header = mx.recordio.IRHeader(0, [y_value[position], y_policy[position].argmax()], idx, 0)
                packed_s = mx.recordio.pack(header, buf)
                record.write_idx(idx, packed_s)
                idx += 1

            logging.debug("elapsed time %.2fs", (time() - t_s)) # log the elapsed time for a single dataset part file

        record.close()# close the record file
        logging.debug("created %s successfully", idx_filepath)
        logging.debug("created %s successfully", rec_filepath)
