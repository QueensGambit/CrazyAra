"""
@file: fileio.py
Created on 01.04.2021
@project: CrazyAra
@author: queensgambit, maxalexger

Contains the main class to handle files and directories during Reinforcement Learning.
Additionally, a function to compress zarr datasets is provided.
"""
import os
import shutil
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


def check_for_moe(model_dir: str):
    """
    Extracts the number of phases from the given model directory.
    Returns true if mixture of experts is used.
    The second return argument is the number of phases.
    :param model_dir: Model directory, where either the model directly is stored or the number of phase directories.
    :return: is_moe: bool, number_phases: int or None
    """
    number_phases = 0
    is_moe = False
    for entry in os.listdir(model_dir):
        if entry.startswith("phase") and "None" not in entry:
            number_phases += 1
            is_moe = True
    if not is_moe:
        number_phases = None
    return is_moe, number_phases


class FileIO:
    """
    Class to facilitate creation of directories, reading of file
    names and moving of files during Reinforcement Learning.
    """
    def __init__(self, orig_binary_name: str, binary_dir: str, uci_variant: str):
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
        self.model_dir = binary_dir + "model/" + orig_binary_name + "/" + variant_suffix
        self.export_dir_gen_data = binary_dir + "export/new_data/" + variant_suffix
        self.train_dir = binary_dir + "export/train/" + variant_suffix
        self.val_dir = binary_dir + "export/val/" + variant_suffix
        self.weight_dir = binary_dir + "weights/" + variant_suffix
        self.train_dir_archive = binary_dir + "export/archive/train/" + variant_suffix
        self.val_dir_archive = binary_dir + "export/archive/val/" + variant_suffix
        self.model_contender_dir = binary_dir + "model_contender/" + orig_binary_name + "/" + variant_suffix
        self.model_dir_archive = binary_dir + "export/archive/model/" + variant_suffix
        self.logs_dir_archive = binary_dir + "export/logs/" + variant_suffix
        self.logs_dir = binary_dir + "logs"

        self.timestamp_format = "%Y-%m-%d-%H-%M-%S"

        self.is_moe, self.number_phases = check_for_moe(self.model_dir)

        # Whether to use Staged learning v2.0 for MoE training,
        # i.e. first train on full data and then each phase separately
        self.use_moe_staged_learning = True if os.path.isdir(self.model_dir + "phaseNone") else False

        if self.is_moe:
            logging.info(f"Mixture of experts detected with {self.number_phases} phases.")
            logging.info(f"Use MoE staged learning is {self.use_moe_staged_learning}.")
        else:
            logging.info("No mixture of experts detected.")
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
        create_dir(self.logs_dir)
        create_dir(self.weight_dir)
        create_dir(self.export_dir_gen_data)
        create_dir(self.train_dir)
        create_dir(self.val_dir)
        create_dir(self.train_dir_archive)
        create_dir(self.val_dir_archive)
        create_dir(self.model_contender_dir)
        create_dir(self.model_dir_archive)
        create_dir(self.logs_dir_archive)

        if self.is_moe:
            for directory in [self.export_dir_gen_data, self.train_dir, self.val_dir, self.train_dir_archive,
                              self.val_dir_archive, self.model_contender_dir, self.model_dir_archive]:
                for phase_idx in range(self.number_phases):
                    create_dir(directory + f"phase{phase_idx}")
            if self.use_moe_staged_learning:
                create_dir(self.model_contender_dir + "phaseNone")
                create_dir(self.model_dir_archive + "phaseNone")

    def _include_data_from_replay_memory_wrapper(self, nb_files: int, fraction_for_selection: float):
        """
        Wrapper for _include_data_from_replay_memory() which handles MoE and non MoE cases.
        :param nb_files: Number of files to include from replay memory into training
        :param fraction_for_selection: Proportion for selecting files from the replay memory
        """

        if not self.is_moe:
            self._include_data_from_replay_memory(self.train_dir_archive, self.train_dir, nb_files,
                                                  fraction_for_selection)
        else:
            for phase_idx in range(self.number_phases):
                self._include_data_from_replay_memory(self.train_dir_archive + f"phase{phase_idx}/",
                                                      self.train_dir + f"phase{phase_idx}/", nb_files,
                                                      fraction_for_selection)

    def _include_data_from_replay_memory(self, from_dir: str, to_dir: str, nb_files: int, fraction_for_selection: float):
        """
        Moves data from the from_dir directory to the to_dir directory.
        :param from_dir: Usually train_dir_archive
        :param to_dir: Usually train_dir
        :param nb_files: Number of files to include from replay memory into training
        :param fraction_for_selection: Proportion for selecting files from the replay memory
        :return:
        """
        # get all file/folder names ignoring hidden files
        folder_names = [folder_name for folder_name in os.listdir(from_dir)
                        if not folder_name.startswith('.')]

        if len(folder_names) < nb_files:
            logging.info("Not enough replay memory available. Only current data will be used")
            return

        # sort files according to timestamp, ignoring the last device-id
        # invert ordering (most recent files are on top)
        folder_names.sort(key=lambda f: time.mktime(time.strptime(f.rsplit('-', 1)[0], f"{self.timestamp_format}")),
                          reverse=True)

        thresh_idx = max(int(len(folder_names) * fraction_for_selection + 0.5), nb_files)

        indices = np.arange(0, thresh_idx)
        np.random.shuffle(indices)

        # cap the index list
        indices = indices[:nb_files]

        # move selected files into train dir
        for index in list(indices):
            os.rename(from_dir + folder_names[index], to_dir + folder_names[index])

    def _move_generated_data_to_train_val(self):
        """
        Moves the generated samples, games (pgn format) and the number how many games have been generated to the given
        training and validation directory
        :return:
        """
        if not self.is_moe:
            file_names = os.listdir(self.export_dir_gen_data)

            # move the last file into the validation directory
            os.rename(self.export_dir_gen_data + file_names[-1], self.val_dir + file_names[-1])

            # move the rest into the training directory
            for file_name in file_names[:-1]:
                os.rename(self.export_dir_gen_data + file_name, self.train_dir + file_name)
        else:
            for phase_idx in range(self.number_phases):
                file_names = os.listdir(self.export_dir_gen_data + f"/phase{phase_idx}")

                # move the last file into the validation directory
                os.rename(self.export_dir_gen_data + f"/phase{phase_idx}/" + file_names[-1],
                          self.val_dir + f"/phase{phase_idx}/" + file_names[-1])

                # move the rest into the training directory
                for file_name in file_names[:-1]:
                    os.rename(self.export_dir_gen_data + f"/phase{phase_idx}/" + file_name,
                              self.train_dir + f"/phase{phase_idx}/" + file_name)

    def _move_train_val_data_into_archive(self):
        """
        Moves files from training, validation dir into archive directory
        :return:
        """
        self._move_all_files_wrapper(self.train_dir, self.train_dir_archive)
        self._move_all_files_wrapper(self.val_dir, self.val_dir_archive)

    def _remove_files_in_weight_dir(self):
        """
        Removes all files in the weight directory.
        :return:
        """
        if not self.is_moe:
            file_list = glob.glob(os.path.join(self.weight_dir, "model-*"))
            for file in file_list:
                os.remove(file)
        else:
            for phase_idx in range(self.number_phases):
                file_list = glob.glob(os.path.join(self.weight_dir, f"phase{phase_idx}/model-*"))
                for file in file_list:
                    os.remove(file)

    def _compress_single_dataset(self, phase: str, device_name: str):
        """
        Loads a single uncompressed data file, selects all samples, compresses it and exports it.
        :param phase: Phase to use, e.g. "phase0/", "phase1". Is empty string for no phase ("").
        :return: export_dir: str
        """
        data = zarr.load(self.binary_dir + phase + "data_" + device_name + ".zarr")

        export_dir, time_stamp = self.create_export_dir(phase, device_name)
        zarr_path = export_dir + time_stamp + ".zip"

        end_idx = self._retrieve_end_idx(data)

        nan_detected = compress_zarr_dataset(data, zarr_path, start_idx=0, end_idx=end_idx)
        if nan_detected is True:
            logging.error("NaN value detected in file %s.zip" % time_stamp)
            new_export_dir = self.binary_dir + time_stamp
            os.rename(export_dir, new_export_dir)
            export_dir = new_export_dir

        return export_dir

    def _retrieve_end_idx(self, data):
        """
        Checks the y_policy sum in the data for is_moe is False and
        returns the first occurrence of only 0s.
        An end_idx of 0 means the whole dataset will be used
        :param data: Zarr data object
        :return: end_idx
        """
        if self.is_moe is False:
            return 0

        sum_y_policy = data['y_policy'].sum(axis=1)
        potential_end_idx = sum_y_policy.argmin()
        if sum_y_policy[potential_end_idx] == 0:
            return potential_end_idx
        return 0

    def _merge_pgn_files(self, output_filename: str, input_prefix: str, num_files: int):
        """
        Merges the content of all small pgn files to a single file.
        :param output_filename: File in which all content will be exported
        :param input_prefix: Input prefix (e.g. "games_gpu0")
        :param num_files: Number of individual files
        :return:
        """
        with open(self.binary_dir + output_filename, 'wb') as outfile:
            for i in range(num_files):
                input_filename = self.binary_dir + f"{input_prefix}_{i}.pgn"

                if os.path.exists(input_filename):
                    with open(input_filename, 'rb') as infile:
                        shutil.copyfileobj(infile, outfile)
                else:
                    raise FileNotFoundError(f'Error: {input_filename} was not found')


    def _merge_game_idx_files(self, output_filename: str, input_prefix: str, num_files: int):
        """
        Reads numbers from multiple files, sums them up and exports the result.

        :param output_filename: File in which the sum will be exported to.
        :param input_prefix: Input prefix (e.g. "gameIdx_gpu0")
        :param num_files: Number of individual files
        :return:
        """
        total_count = 0

        for i in range(num_files):
            input_filename = self.binary_dir + f"{input_prefix}_{i}.txt"

            if os.path.exists(input_filename):
                try:
                    with open(input_filename, 'r') as infile:
                        content = infile.read().strip()
                        if content:
                            total_count += int(content)
                except ValueError:
                    raise ValueError(f"The content of {input_filename} is not a valid number.")
            else:
                raise FileNotFoundError(f'Error: {input_filename} was not found')

        # write final result to file
        with open(self.binary_dir + output_filename, 'w') as outfile:
            outfile.write(str(total_count))

    def _remove_individual_files(self, file_prefix: str, suffix, num_files):
        """
        Removes the individual files.
        :param file_prefix: File prefix (e.g. "games_gpu0")
        :param suffix: file type
        :param num_files: Number of files
        :return:
        """
        for i in range(num_files):
            filename = self.binary_dir + f"{file_prefix}_{i}.{suffix}"
            os.remove(filename)

    def _merge_and_compress_zarr_datasets(self, device_name, input_prefix: str, num_files: int, compression='lz4', clevel=5):
        """
        Combines multiple uncompressed Zarr directories into a single compressed ZipStore.

        :param device_name: Device name (e.g. "gpu0")
        :param input_prefix: The prefix before the index (e.g., "data_gpu0")
        :param num_files: Number of input files to process
        :param compression: Compression algorithm for Blosc
        :param clevel: Compression level (1-9)
        :return: export_dir
        """
        # 1. Gather all existing input paths
        input_paths = [self.binary_dir + f"{input_prefix}_{i}.zarr" for i in range(num_files)]
        sources = []

        for path in input_paths:
            if os.path.exists(path):
                sources.append(zarr.open(path, mode='r'))
            else:
                logging.warning(f"Source file not found: {path}")

        if not sources:
            logging.error("No source files found to merge.")
            return False

        export_dir, time_stamp = self.create_export_dir("", device_name)
        zarr_path = export_dir + time_stamp + ".zip"

        # 2. Setup compressor and destination store
        compressor = Blosc(cname=compression, clevel=clevel, shuffle=Blosc.SHUFFLE)
        store = zarr.ZipStore(zarr_path, mode="w")
        target_group = zarr.group(store=store)

        # Get keys from the first source (e.g., 'input_planes', 'value_targets', etc.)
        keys = list(sources[0].keys())
        nan_detected = False

        for key in keys:
            logging.debug(f"Merging key: {key}...")

            # Calculate the total number of samples (sum of first dimension)
            total_rows = sum(src[key].shape[0] for src in sources)

            # Determine target shape and chunking
            # Chunks are set to 128 for the first dimension, as per your requirements
            original_shape = list(sources[0][key].shape)
            target_shape = original_shape.copy()
            target_shape[0] = total_rows

            chunk_shape = original_shape.copy()
            chunk_shape[0] = 128

            # 3. Create the empty dataset in the target file
            target_ds = target_group.create_dataset(
                name=key,
                shape=tuple(target_shape),
                chunks=tuple(chunk_shape),
                dtype=sources[0][key].dtype,
                compression=compressor,
                synchronizer=zarr.ThreadSynchronizer()
            )

            # 4. Copy data from sources into the target slice-by-slice
            current_offset = 0
            for src in sources:
                data_chunk = src[key][:]  # Load source data into memory

                # Check for data integrity
                if np.isnan(data_chunk).any():
                    nan_detected = True
                    logging.error(f"NaN detected in {key} of a source file!")

                num_rows = data_chunk.shape[0]
                # Write to the specific pre-calculated slice in the destination
                target_ds[current_offset: current_offset + num_rows] = data_chunk
                current_offset += num_rows

        if nan_detected is True:
            logging.error("NaN value detected in file %s.zip" % time_stamp)
            new_export_dir = self.binary_dir + time_stamp
            os.rename(export_dir, new_export_dir)
            export_dir = new_export_dir

        store.close()
        logging.info(f"Successfully exported compressed dataset to: {zarr_path}")
        return export_dir

    def combine_dataset_and_files(self, device_name: str, number_parallel_games: int):
        """
        Combines the dataset as well as pgn and txt-files into a single file each.
        :param device_name: The currently active device name (context_device-id)
        :param number_parallel_games: How many files have been generated
        :return:
        """

        self._merge_pgn_files(
            output_filename=f"games_{device_name}.pgn",
            input_prefix=f"games_{device_name}",
            num_files=number_parallel_games
        )

        self._merge_game_idx_files(
            output_filename=f"gameIdx_{device_name}.txt",
            input_prefix=f"gameIdx_{device_name}",
            num_files=number_parallel_games
        )

        export_dir = self._merge_and_compress_zarr_datasets(device_name=device_name,
                                                            input_prefix=f"data_{device_name}",
                                                            num_files=number_parallel_games)

        self._remove_individual_files(f"games_{device_name}", "pgn", number_parallel_games)
        self._remove_individual_files(f"gameIdx_{device_name}", "txt", number_parallel_games)

        self.move_game_data_to_export_dir(export_dir, device_name)

    def compress_dataset(self, device_name: str):
        """
        Calls _compress_single_dataset() for each phase or a single time for no phases.
        Also moves the game data to export directory.
        :param device_name: The currently active device name (context_device-id)
        :return:
        """
        if self.is_moe:
            for phase_idx in range(self.number_phases):
                export_dir = self._compress_single_dataset(f"phase{phase_idx}/", device_name)
                if phase_idx == 0:
                    self.move_game_data_to_export_dir(export_dir, device_name)
        else:
            export_dir = self._compress_single_dataset("", device_name)
            self.move_game_data_to_export_dir(export_dir, device_name)

    def create_export_dir(self, phase: str, device_name: str) -> (str, str):
        """
        Create a directory in the 'export_dir_gen_data' path,
        where the name consists of the current date, time and device ID.
        :param phase: Phase to use, e.g. "phase0/", "phase1". Is empty string for no phase ("").
        :param device_name: The currently active device name (context_device-id)
        :return: Path of the created directory; Time stamp used while creating
        """
        # include current timestamp in dataset export file
        time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime(self.timestamp_format)
        time_stamp_dir = f'{self.export_dir_gen_data}{phase}{time_stamp}-{device_name}/'
        # create a directory of the current time_stamp
        if not os.path.exists(time_stamp_dir):
            os.makedirs(time_stamp_dir)

        return time_stamp_dir, time_stamp

    def get_current_model_tar_file(self, phase=None, base_dir=None) -> str:
        """
        :param phase: Phase to use. Should be "" if no MoE is used and otherwise e.g. "phase2".
        :param base_dir: Should be self.model_dir in the normal case
        For None default "phase0" or "" will be used.
        Return the filename of the current active model weight (.tar) file for pytorch
        """
        if phase is None:
            if self.is_moe:
                phase = "phase0"
            else:
                phase = ""
        if base_dir is None:
            base_dir = self.model_dir
        model_params = glob.glob(base_dir + phase + "/*.tar")
        if len(model_params) == 0:
            raise FileNotFoundError(f'No model file found in {self.model_dir}')
        return model_params[0]

    def get_number_generated_files(self) -> int:
        """
        Returns the amount of file that have been generated since the last training run.
        :return: nb_training_files: int
        """
        if self.is_moe:
            phase = "phase0/"
        else:
            phase = ""
        return len(glob.glob(self.export_dir_gen_data + phase + "**/*.zip"))

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

    def move_training_logs(self, nn_update_index):
        """
        Rename logs and move it from /logs to /export/logs/
        """
        time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime(self.timestamp_format)
        dir_name = f'logs-{self.uci_variant}-update{nn_update_index}-{time_stamp}'
        os.rename(self.logs_dir, os.path.join(self.logs_dir_archive, dir_name))
        create_dir(self.logs_dir)

    def prepare_data_for_training(self, rm_nb_files: int, rm_fraction_for_selection: float, did_contender_win: bool):
        """
        Move files from training, validation and model contender folder into archive.
        Moves newly generated files into training and validation directory.
        Remove files in weight directory. Include data from replay memory.
        :param rm_nb_files: Number of files of the replay memory to include
        :param rm_fraction_for_selection: Proportion for selecting files from the replay memory
        :param did_contender_win: Defines if the last contender won vs the generator
        """
        if did_contender_win:
            self._move_train_val_data_into_archive()
        # move last contender into archive
        self._move_all_files_wrapper(self.model_contender_dir, self.model_dir_archive)

        self._move_generated_data_to_train_val()
        # We don’t need them anymore; the last model from last training has already been saved
        self._remove_files_in_weight_dir()
        self._include_data_from_replay_memory_wrapper(rm_nb_files, rm_fraction_for_selection)

    def remove_intermediate_weight_files(self):
        """
        Deletes all files (excluding folders) inside the weight directory
        """
        # Replace _weight_dir with self.weight_dir, if the trainer can save weights dynamically
        _weight_dir = self.binary_dir + 'weights/'
        files = glob.glob(_weight_dir + 'model-*')
        for f in files:
            os.remove(f)

    def _move_all_files_wrapper(self, from_dir, to_dir):
        """
        Wrapper function for move_all_files(from_dir, to_dir).
        In case of self.is_moe, all phases directories are moved as well.
        :param from_dir: Origin directory where the files are located
        :param to_dir: Destination directory where all files (including subdirectories directories) will be moved
        :return:
        """
        if not self.is_moe:
            move_all_files(from_dir, to_dir)
        else:
            for phase_idx in range(self.number_phases):
                move_all_files(from_dir + f"phase{phase_idx}/", to_dir + f"phase{phase_idx}/")

            if self.use_moe_staged_learning:
                from_dir_final = from_dir + "phaseNone/"
                to_dir_final = to_dir + "phaseNone/"
                if os.path.isdir(from_dir_final) and os.path.isdir(to_dir_final):
                    move_all_files(from_dir_final, to_dir_final)

    def replace_current_model_with_contender(self):
        """
        Moves the previous model into archive directory and the model-contender into the model directory
        """
        self._move_all_files_wrapper(self.model_dir, self.model_dir_archive)
        self._move_all_files_wrapper(self.model_contender_dir, self.model_dir)
