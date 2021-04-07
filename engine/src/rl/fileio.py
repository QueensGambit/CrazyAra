"""
@file: fileio.py
Created on 01.04.2021
@project: CrazyAra
@author: queensgambit, maxalexger

Contains the main class to handle files and directories during Reinforcement Learning.
Additionally a function to compress zarr datasets is provided.
"""
import os
import glob
import zarr
import time
import logging
import datetime
import numpy as np
from numcodecs import Blosc

from DeepCrazyhouse.configs.main_config import main_config
from engine.src.rl.rl_utils import create_dir, move_all_files


def compress_zarr_dataset(data, file_path, compression='lz4', clevel=5, start_idx=0, end_idx=0):
    """
    Loads in a zarr data set and exports it with a given compression type and level
    :param data: Zarr data set which will be compressed
    :param file_path: File name path where the data will be exported (e.g. "./export/data.zip")
    :param compression: Compression type
    :param clevel: Compression level
    :param start_idx: Starting index of data to be exported.
    :param end_idx: If end_idx != 0 the data set will be exported to the specified index,
    excluding the sample at end_idx (e.g. end_idx = len(x) will export it fully)
    :return: True if a NaN value was detected
    """
    compressor = Blosc(cname=compression, clevel=clevel, shuffle=Blosc.SHUFFLE)

    # open a dataset file and create arrays
    store = zarr.ZipStore(file_path, mode="w")
    zarr_file = zarr.group(store=store, overwrite=True)

    nan_detected = False
    for key in data.keys():
        if end_idx == 0:
            x = data[key]
        else:
            x = data[key][start_idx:end_idx]

        if np.isnan(x).any():
            nan_detected = True

        array_shape = list(x.shape)
        array_shape[0] = 128
        # export array
        zarr_file.create_dataset(
            name=key,
            data=x,
            shape=x.shape,
            dtype=type(x.flatten()[0]),
            chunks=array_shape,
            synchronizer=zarr.ThreadSynchronizer(),
            compression=compressor,
        )
    store.close()
    logging.info("dataset was exported to: %s", file_path)
    return nan_detected


class FileIO:
    """
    Class to facilitate creation of directories, reading of file
    names and moving of files during Reinforcement Learning.
    """
    def __init__(self, binary_dir: str, uci_variant: str):
        """
        Creates all necessary directories and sets all path variables.
        If no '*.param' file can be found in the 'binary-dir/model/' directory,
        we assume that every folder has another subdirectory named after the UCI-Variant.
        """
        self.binary_dir = binary_dir
        self.uci_variant = uci_variant

        # If there is no model in 'model/', we assume that the model and every
        # other path has an additional '<variant>' folder
        variant_suffix = f''
        if len(glob.glob(f'{binary_dir}model/*.params')) == 0:
            variant_suffix = f'{uci_variant}/'

        # Hard coded directory paths
        self.model_dir = binary_dir + "model/" + variant_suffix
        self.export_dir_gen_data = binary_dir + "export/new_data/" + variant_suffix
        self.train_dir = binary_dir + "export/train/" + variant_suffix
        self.val_dir = binary_dir + "export/val/" + variant_suffix
        self.weight_dir = binary_dir+"weights/" + variant_suffix
        self.train_dir_archive = binary_dir + "export/archive/train/" + variant_suffix
        self.val_dir_archive = binary_dir + "export/archive/val/" + variant_suffix
        self.model_contender_dir = binary_dir + "model_contender/" + variant_suffix
        self.model_dir_archive = binary_dir + "export/archive/model/" + variant_suffix

        self._create_directories()

        # Adjust paths in main_config
        main_config["planes_train_dir"] = binary_dir + "export/train/" + variant_suffix
        main_config["planes_val_dir"] = binary_dir + "export/val/" + variant_suffix
        assert os.path.isdir(main_config["planes_train_dir"]) is not False, \
            f'Please provide valid main_config["planes_train_dir"] directory'

    def _create_directories(self):
        """
        Creates directories in the binary folder which will be used during RL
        :return:
        """
        create_dir(self.binary_dir+"logs")
        create_dir(self.weight_dir)
        create_dir(self.export_dir_gen_data)
        create_dir(self.train_dir)
        create_dir(self.val_dir)
        create_dir(self.train_dir_archive)
        create_dir(self.val_dir_archive)
        create_dir(self.model_contender_dir)
        create_dir(self.model_dir_archive)

    def _include_data_from_replay_memory(self, nb_files: int, fraction_for_selection: float):
        """
        :param nb_files: Number of files to include from replay memory into training
        :param fraction_for_selection: Proportion for selecting files from the replay memory
        :return:
        """
        file_names = os.listdir(self.train_dir_archive)

        # invert ordering (most recent files are on top)
        file_names = file_names[::-1]

        if len(file_names) < nb_files:
            logging.info("Not enough replay memory available. Only current data will be used")
            return

        thresh_idx = max(int(len(file_names) * fraction_for_selection), nb_files)

        indices = np.arange(0, thresh_idx)
        np.random.shuffle(indices)

        # cap the index list
        indices = indices[:nb_files]

        # move selected files into train dir
        for index in list(indices):
            os.rename(self.train_dir_archive + file_names[index], self.train_dir + file_names[index])

    def _move_generated_data_to_train_val(self):
        """
        Moves the generated samples, games (pgn format) and the number how many games have been generated to the given
        training and validation directory
        :return:
        """
        file_names = os.listdir(self.export_dir_gen_data)

        # move the last file into the validation directory
        os.rename(self.export_dir_gen_data + file_names[-1], self.val_dir + file_names[-1])

        # move the rest into the training directory
        for file_name in file_names[:-1]:
            os.rename(self.export_dir_gen_data + file_name, self.train_dir + file_name)

    def _move_train_val_contender_into_archive(self):
        """
        Moves files from training, validation & model_contender dir into archive directory
        :return:
        """
        move_all_files(self.train_dir, self.train_dir_archive)
        move_all_files(self.val_dir, self.val_dir_archive)
        move_all_files(self.model_contender_dir, self.model_dir_archive)

    def _remove_files_in_weight_dir(self):
        """
        Removes all files in the weight directory.
        :return:
        """
        file_list = glob.glob(os.path.join(self.weight_dir, "model-*"))
        for file in file_list:
            os.remove(file)

    def compress_dataset(self, device_name: str):
        """
        Loads the uncompressed data file, selects all sample until the index specified in "startIdx.txt",
        compresses it and exports it.
        :param device_name: The currently active device name (context_device-id)
        :return:
        """
        data = zarr.load(self.binary_dir + "data_" + device_name + ".zarr")

        export_dir, time_stamp = self.create_export_dir(device_name)
        zarr_path = export_dir + time_stamp + ".zip"
        nan_detected = compress_zarr_dataset(data, zarr_path, start_idx=0)
        if nan_detected is True:
            logging.error("NaN value detected in file %s.zip" % time_stamp)
            new_export_dir = self.binary_dir + time_stamp
            os.rename(export_dir, new_export_dir)
            export_dir = new_export_dir
        self.move_game_data_to_export_dir(export_dir, device_name)

    def create_export_dir(self, device_name: str) -> (str, str):
        """
        Create a directory in the 'export_dir_gen_data' path,
        where the name consists of the current date, time and device ID.
        :param device_name: The currently active device name (context_device-id)
        :return: Path of the created directory; Time stamp used while creating
        """
        # include current timestamp in dataset export file
        time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H-%M-%S")
        time_stamp_dir = f'{self.export_dir_gen_data}{time_stamp}-{device_name}/'
        # create a directory of the current time_stamp
        if not os.path.exists(time_stamp_dir):
            os.makedirs(time_stamp_dir)

        return time_stamp_dir, time_stamp

    def get_current_model_arch_file(self) -> str:
        """
        Return the filename of the current active model architecture (.json) file
        """
        model_arch = glob.glob(self.model_dir + "/*.json")
        if len(model_arch) == 0:
            raise FileNotFoundError(f'No arch file found in {self.model_dir}')
        return model_arch[0]

    def get_current_model_weight_file(self) -> str:
        """
        Return the filename of the current active model weight (.params) file
        """
        model_params = glob.glob(self.model_dir + "/*.params")
        if len(model_params) == 0:
            raise FileNotFoundError(f'No model file found in {self.model_dir}')
        return model_params[0]

    def get_number_generated_files(self) -> int:
        """
        Returns the amount of file that have been generated since the last training run.
        :return:
        """
        return len(glob.glob(self.export_dir_gen_data + "**/*.zip"))

    def move_game_data_to_export_dir(self, export_dir: str, device_name: str):
        """
        Moves the generated games saved in .pgn format and the number how many games have been generated
        to the given export directory.
        :param export_dir: Export directory for the newly generated data
        :param device_name: The currently active device name (context_device-id)
        """
        file_names = ["games_" + device_name + ".pgn",
                      "gameIdx_" + device_name + ".txt"]
        for file_name in file_names:
            os.rename(self.binary_dir + file_name, export_dir + file_name)

    def prepare_data_for_training(self, rm_nb_files, rm_fraction_for_selection):
        """
        Move files from training, validation and model contender folder into archive.
        Moves newly generated files into training and validation directory.
        Remove files in weight directory. Include data from replay memory.
        :param rm_nb_files:
        :param rm_fraction_for_selection:
        """
        self._move_train_val_contender_into_archive()
        self._move_generated_data_to_train_val()
        # We don’t need them anymore; the last model from last training has already been saved
        self._remove_files_in_weight_dir()
        self._include_data_from_replay_memory(rm_nb_files, rm_fraction_for_selection)

    def replace_current_model_with_contender(self):
        """
        Moves the previous model into archive directory and the model-contender into the model directory
        :return:
        """
        move_all_files(self.model_dir, self.model_dir_archive)
        move_all_files(self.model_contender_dir, self.model_dir)
