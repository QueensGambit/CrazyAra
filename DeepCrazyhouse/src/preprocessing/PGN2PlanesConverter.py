"""
@file: PGN2PlanesConverter.py
Created on 09.06.18
@project: DeepCrazyhouse
@author: queensgambit

Loads the pgn-file and calls the pgn_converter functions to create a plane represenation.
Multiprocessing is used for loading, computing and saving.
"""

import chess.pgn
import matplotlib.pyplot as plt
from time import time
import io
from multiprocessing import Process, Pool, Queue
from copy import deepcopy
from DeepCrazyhouse.src.preprocessing.pgn_converter_util import get_planes_from_pgn
from DeepCrazyhouse.src.domain.util import get_dic_sorted_by_key
from DeepCrazyhouse.configs.main_config_sample import main_config
import math
import numpy as np

import logging
import os
import zarr
from numcodecs import Blosc
import gc
import psutil
import datetime
import re


class PGN2PlanesConverter(object):
    """
    Class which enables the conversion from pgn-text files to a plane representation which can be used for Neural
    Networks. The representation will be exported in parallel, using compression to a dataset file (e.g. .zip)
    """

    def __init__(
        self,
        limit_nb_games_to_analyze=4096,
        nb_games_per_file=1000,
        max_nb_files=2,
        min_elo_both=1700,
        termination_conditions=None,
        compression="lz4",
        clevel=5,
        log_lvl=logging.DEBUG,
        dataset_type="train",
    ):
        """
        Set the member variables and loads the config file

        :param limit_nb_games_to_analyze: Limits the maximum number of games to analyze of the file.
                (if 0 process all games of the file)
        :param nb_games_per_file: Number of selected games which are exported in one exported part
        :param max_nb_files: Maximum number of hdf5-files to create
                             (if 0 process all batches of the file)
        :param min_elo_both: only selects games in which either black or white has at least this elo rating
        :param termination_conditions: only select games in which one of the given termination conditions is hold
        :param compression: Compression type for compressing the image planes
                            Available options are:
                            -> blosc.list_compressors()
                            ['blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib', 'zstd']
        :param clevel: compression level (more info at: https://zarr.readthedocs.io/en/stable/tutorial.html#compressors)
                            # see http://www.h5py.org/lzf/ for performance comparision
                            # Tutorial about compression:
         https://dziganto.github.io/out-of-core%20computation/HDF5-Or-How-I-Learned-To-Love-Data-Compression-And-Partial-Input-Output/
        :param log_lvl: Sets the log_lvl for the log messages
            root = logging.getLogger()
            root.setLevel(logging.INFO) # root.setLevel(logging.DEBUG)...
        :param dataset_type: either ['train', 'val', 'test', 'mate_in_one']
        :return:
        """
        if termination_conditions is None:
            termination_conditions = ["Normal"]

        self._limit_nb_games = limit_nb_games_to_analyze
        self._batch_size = nb_games_per_file
        self._max_nb_files = max_nb_files
        self._min_elo_both = min_elo_both
        self._termination_conditions = termination_conditions
        self._compression = compression
        self._clevel = clevel
        self._log_lvl = log_lvl

        root = logging.getLogger()
        root.setLevel(log_lvl)

        plt.style.use("seaborn-paper")

        logging.info("SCRIPT START")

        logging.info("reading in config.json file...")

        # show the contents of the config file
        logging.debug("%s", main_config)

        if dataset_type == "train":
            self._import_dir = main_config["pgn_train_dir"]
            self._export_dir = main_config["planes_train_dir"]
            self._mate_in_one = False
        elif dataset_type == "val":
            self._import_dir = main_config["pgn_val_dir"]
            self._export_dir = main_config["planes_val_dir"]
            self._mate_in_one = False
        elif dataset_type == "test":
            self._import_dir = main_config["pgn_test_dir"]
            self._export_dir = main_config["planes_test_dir"]
            self._mate_in_one = False
        elif dataset_type == "mate_in_one":
            self._import_dir = main_config["pgn_mate_in_one_dir"]
            self._export_dir = main_config["planes_mate_in_one_dir"]
            self._mate_in_one = True
        else:
            raise Exception(
                'Invalid dataset type "%s" given. It must be either "train", "val", "test" or "mate_in_one"'
                % dataset_type
            )

        # initialize the png_name to the first pgn file in the import directory
        self._pgn_name = os.listdir(self._import_dir)[0]

        # include current timestamp in dataset export file
        timestmp = datetime.datetime.fromtimestamp(time()).strftime("%Y-%m-%d-%H-%M-%S")
        self._timestmp_dir = self._export_dir + timestmp + "/"

        # https://stackoverflow.com/questions/15455048/releasing-memory-in-python
        self._proc = psutil.Process(os.getpid())
        gc.collect()
        self._mem0 = self._proc.memory_info().rss

    def filter_pgn(self):
        """
        Wrapper for _filter_pgn_thread()
        :return:
        """

        logging.info("loading pgn file into memory...")
        # load the pgn file
        pgn = open(self._import_dir + self._pgn_name, "r")

        # start a subprocess to be memory efficient
        q = Queue()
        p = Process(target=self._filter_pgn_thread, args=(q, pgn))

        # filter the according pgns
        logging.debug("start subprocess")
        p.start()

        # receive the return arguments
        all_pgn_sel = q.get()
        nb_games_sel = q.get()
        batch_white_won = q.get()
        batch_black_won = q.get()
        batch_draw = q.get()

        p.join()  # this blocks until the process terminates
        logging.debug("subprocess finished")

        return all_pgn_sel, nb_games_sel, batch_white_won, batch_black_won, batch_draw

    def _filter_pgn_thread(self, queue, pgn):
        """
        Selects the pgn which fullfill the given conditions
        :param queue: Stores the result/return variables
        :param pgn: PGN file
        :return: Queue filled with the following iterms:
        - all_pgn_sel: List of the selected pgn files
        - nb_games_sel: Number of games which have been selected
        - batch_white_won: list of number of games which have been won by the white player in this batch
        - batch_black_won: list of number of games which have been won by the black player in this batch
        - batch_draw: list of number of games which have been drawn in this batch
        """
        # read the pgn content into a string
        content = pgn.read()
        nb_games = content.count("Crazy")

        logging.debug("nb_games: %d", nb_games)

        # split the content for each single game
        all_games = content.split("[Event")

        for i in range(len(all_games)):
            all_games[i] = "[Event" + all_games[i]

        pgns = []
        del all_games[0]

        if self._limit_nb_games != 0:
            games = all_games[: self._limit_nb_games]
        else:
            games = all_games

        for game in games:

            # only add game with at least one move played
            if "1. " in game:
                # for mate in one make sure the game ended in a checkmate
                if self._mate_in_one is False:
                    pgns.append(io.StringIO(game))
                else:
                    # look for '#' as soon as the move history begins
                    mv_hist_start = game.find("1. ", 0, len(game))
                    # only searching for '#' lead to the problem that one event was called "Crazyhouse Revolution #1"
                    # also some games are annotated and contain # in the evaluation
                    # \S Matches any non-whitespace character; this is equivalent to the class [^ \t\n\r\f\v].
                    m = re.search(r"\S#", game[mv_hist_start:])
                    if m is not None:
                        pgns.append(io.StringIO(game))

        logging.info("select games based on given conditions...")
        # only select games according to the conditions

        all_pgn_sel = []

        nb_white_won = 0
        nb_black_won = 0
        nb_draws = 0

        batch_white_won = []
        batch_black_won = []
        batch_draw = []

        for game_pgn in pgns:
            # we need to create a deep copy, otherwise the end of the file is reached for later
            game_pgn_copy = deepcopy(game_pgn)
            for offset, headers in chess.pgn.scan_headers(game_pgn_copy):
                for term_cond in self._termination_conditions:
                    if term_cond in headers["Termination"]:
                        if (
                            int(headers["WhiteElo"]) >= self._min_elo_both
                            and int(headers["BlackElo"]) >= self._min_elo_both
                        ):
                            if headers["Result"] == "1-0":
                                nb_white_won += 1
                            elif headers["Result"] == "0-1":
                                nb_black_won += 1
                            elif headers["Result"] == "1/2-1/2":
                                nb_draws += 1
                            else:
                                raise Exception("Illegal Game Result: ", headers["Result"])

                            all_pgn_sel.append(game_pgn)

                            if len(all_pgn_sel) % self._batch_size == 0:
                                # save the stats of 1 batch part
                                batch_white_won.append(nb_white_won)
                                batch_black_won.append(nb_black_won)
                                batch_draw.append(nb_draws)

                                nb_white_won = 0
                                nb_black_won = 0
                                nb_draws = 0

        # add the remaining stats to the last batch
        if nb_white_won > 0 or nb_black_won > 0 or nb_draws > 0:
            # save the stats of 1 batch part
            batch_white_won.append(nb_white_won)
            batch_black_won.append(nb_black_won)
            batch_draw.append(nb_draws)

        nb_games_sel = len(all_pgn_sel)

        logging.debug(
            "Number of selected games: %d [1-0: %d, 0-1: %d, 1/2-1/2: %d]",
            nb_games_sel,
            sum(batch_white_won),
            sum(batch_black_won),
            sum(batch_draw),
        )

        # add the results to the queue
        queue.put(all_pgn_sel)
        queue.put(nb_games_sel)
        queue.put(batch_white_won)
        queue.put(batch_black_won)
        queue.put(batch_draw)

    def filter_all_pgns(self):
        """
        Filters out all games based on the given conditions in the constructor and returns all games in
        :return: lst_all_pgn_sel: List of selected games in String-IO format
                 lst_nb_games_sel: Number of selected games for each pgn file
                 lst_batch_white_won: Number of white wins in each pgn file
                 lst_black_won: Number of black wins in each pgn file
                 lst_draw_won: Number of draws in each pgn file
        """

        total_games_exported = 0

        lst_all_pgn_sel = []
        lst_nb_games_sel = []
        lst_batch_white_won = []
        lst_batch_black_won = []
        lst_batch_draw = []

        pgns = os.listdir(self._import_dir)
        for pgn_name in pgns:
            self._pgn_name = pgn_name
            all_pgn_sel, nb_games_sel, batch_white_won, batch_black_won, batch_draw = self.filter_pgn()
            lst_all_pgn_sel.append(all_pgn_sel)
            lst_nb_games_sel.append(nb_games_sel)
            lst_batch_white_won.append(batch_white_won)
            lst_batch_black_won.append(batch_black_won)
            lst_batch_draw.append(batch_draw)

        return lst_all_pgn_sel, lst_nb_games_sel, lst_batch_white_won, lst_batch_black_won, lst_batch_draw

    def convert_all_pgns_to_planes(self):
        """
        Master function which calls convert_pgn_to_planes() for all available pgns in the import directory
        :return: total_games_exported - Total number of games which have been exported
        """
        total_games_exported = 0

        pgns = os.listdir(self._import_dir)
        for pgn_name in pgns:
            cur_games_exported = self.convert_pgn_to_planes(pgn_name)
            total_games_exported += cur_games_exported
            logging.info("Total Games Exported: %d", total_games_exported)
        return total_games_exported

    def convert_pgn_to_planes(self, pgn_name):
        """
        Wrapper class for convert_pgn_to_planes_thread()
        :param pgn_name: pgn file to load
        :return:
        """
        logging.info("PGN-Name: %s", pgn_name)
        q = Queue()
        p = Process(target=self._convert_pgn_to_planes_thread, args=(pgn_name, q))
        # export one batch of pgn games
        p.start()
        games_exported = q.get()  # receive the return argument from the Queue()
        p.join()  # this blocks until the process terminates

        return games_exported

    # @profile
    def _convert_pgn_to_planes_thread(self, pgn_name, queue):
        """
        This function should be called in a subprocess for memory reason.
        Loads the game of a given pgn file, filters the game based on the given conditions and
        export them to plane representation.
        The games are exported in hdf5 format in the directory of config['dataset_export'].
        If the directory config['dataset_export'] doesn't exist it will be created automatically.
        :param pgn_name: pgn file to load
        :param queue: Queue which stores the return argument: The number of games exported for this pgn
        """
        # set the current pgn_name of which the games will be converted
        self._pgn_name = pgn_name

        all_pgn_sel, nb_games_sel, batch_white_won, batch_black_won, batch_draw = self.filter_pgn()
        # if max_nb_files was set to 0, look at all games of the file, filter them and convert them
        if self._max_nb_files == 0:
            max_nb_files = math.floor(nb_games_sel / self._batch_size)
        else:
            max_nb_files = min(self._max_nb_files, int(math.floor(nb_games_sel / self._batch_size)))

        # make sure that one package is built if no full batch could be created fully
        if max_nb_files == 0 and nb_games_sel > 0:
            max_nb_files = 1

        game_idx_end = None
        for cur_part in range(max_nb_files):
            logging.info(
                "PART: %d [1-0: %d, 0-1: %d, 1/2-1/2: %d]",
                cur_part,
                batch_white_won[cur_part],
                batch_black_won[cur_part],
                batch_draw[cur_part],
            )

            # select only a subset of games to analyze
            game_idx_start = cur_part * self._batch_size
            if cur_part < max_nb_files - 1 or (cur_part + 2) * self._batch_size < len(all_pgn_sel):
                # if you limit the number of files you don't take the remaining parts
                game_idx_end = (cur_part + 1) * self._batch_size
            # if you're at the last part include the remaining games resulting in a slightly bigger package
            else:
                # this could lead to store all games in one package, that's why the 'or' condition is included
                game_idx_end = len(all_pgn_sel)

            pgn_sel = all_pgn_sel[game_idx_start:game_idx_end]

            nb_white_wins = batch_white_won[cur_part]
            nb_black_wins = batch_black_won[cur_part]
            nb_draws = batch_draw[cur_part]

            logging.debug("processing games: [%d, %d]", game_idx_start, game_idx_end)

            # We create a new subprocess for each function call to ensure
            #  that not too much memory will get allocated over time
            # https://stackoverflow.com/questions/2046603/is-it-possible-to-run-function-in-a-subprocess-without-threading-or-writing-a-se
            logging.debug("start subprocess")
            p = Process(
                target=self.export_pgn_batch,
                args=(cur_part, game_idx_start, game_idx_end, pgn_sel, nb_white_wins, nb_black_wins, nb_draws),
            )
            # export one batch of pgn games
            p.start()
            p.join()  # this blocks until the process terminates
            logging.debug("subprocess finished")

            # https://stackoverflow.com/questions/15455048/releasing-memory-in-python
            mem_cur = self._proc.memory_info().rss

            # pd = lambda x2, x1: 100.0 * (mem_cur - self._mem0) / self._mem0
            logging.debug("memory usage: %0.2f mb", mem_cur / (1024 * 1024))

        logging.info("SCRIPT END")

        # store the return argument in the queue
        queue.put(game_idx_end)

    def export_pgn_batch(self, cur_part, game_idx_start, game_idx_end, pgn_sel, nb_white_wins, nb_black_wins, nb_draws):
        """
        Exports one part of the pgn-files of the current games selected.
        After the export of one part the memory can be freed of the local variables.
        If the function has been ran successfully a new dataset-partfile was created in the dataset export directory
        For loading and exporting multiprocessing is used

        :param cur_part: Current part (integer value which start at 0).
        :param game_idx_start: Starting game index of the selected game for this part
        :param game_idx_end: End game index of the current part
        :param pgn_sel: Selected PGN data which will be used for the export
        :param nb_white_wins: Number of games which white won in the current part
        :param nb_black_wins: Number of games which black won in the current part
        :param nb_draws: Number of draws in the current part
        :return:
        """

        # create a param input list which will concatenate the pgn with it's corresponding game index
        params_inp = []
        for i, pgn in enumerate(pgn_sel):
            game_idx = game_idx_start + i

            params_inp.append((pgn, game_idx, self._mate_in_one))

        logging.info("starting conversion to planes...")
        t_s = time()
        p = Pool()

        x_dic = {}
        y_value_dic = {}
        y_policy_dic = {}

        metadata_dic = {}

        if not os.path.exists(self._export_dir):
            os.makedirs(self._export_dir)
            logging.info("the dataset_export directory was created at: %s", self._export_dir)

        # create a directory of the current timestmp
        if not os.path.exists(self._timestmp_dir):
            os.makedirs(self._timestmp_dir)

        # http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html
        zarr_path = self._timestmp_dir + self._pgn_name.replace(".pgn", "_" + str(cur_part) + ".zip")

        # open a dataset file and create arrays
        store = zarr.ZipStore(zarr_path, mode="w")

        zarr_file = zarr.group(store=store, overwrite=True)

        # the games occur in random order due to multiprocessing
        # in order to keep structure we store the result in a dictionary first
        for metadata, game_idx, x, y_value, y_policy in p.map(get_planes_from_pgn, params_inp):
            metadata_dic[game_idx] = metadata
            x_dic[game_idx] = x
            y_value_dic[game_idx] = y_value
            y_policy_dic[game_idx] = y_policy

        p.close()
        p.join()
        t_e = time() - t_s
        logging.debug("elapsed time: %fs", t_e)
        t_mean = t_e / self._batch_size
        logging.debug("mean time for 1 game: %f ms", t_mean * 1000)
        # logging.debug('approx time for whole file (nb_games: %d): %fs', self._nb_games, t_mean * self._nb_games)

        # now we can convert the dictionary to a list
        metadata = get_dic_sorted_by_key(metadata_dic)
        x = get_dic_sorted_by_key(x_dic)
        y_value = get_dic_sorted_by_key(y_value_dic)
        y_policy = get_dic_sorted_by_key(y_policy_dic)

        # create a list which describes where each game starts
        start_indices = np.zeros(len(x))
        for i, x_cur in enumerate(x[:-1]):
            start_indices[i + 1] = start_indices[i] + len(x_cur)

        # next we stack the list into a numpy-array
        metadata = np.concatenate(metadata, axis=0)
        x = np.concatenate(x, axis=0)
        y_value = np.concatenate(y_value, axis=0)
        y_policy = np.concatenate(y_policy, axis=0)

        logging.debug("metadata.shape %s", metadata.shape)
        logging.debug("x.shape %s", x.shape)
        logging.debug("y_value.shape %s", y_value.shape)
        logging.debug("y_policy.shape %s", y_policy.shape)

        # Save the dataset to a file
        logging.info("saving the dataset to a file...")

        # define the compressor object
        compressor = Blosc(cname=self._compression, clevel=self._clevel, shuffle=Blosc.SHUFFLE)

        # export the metadata
        zarr_file.create_dataset(
            name="metadata",
            data=metadata,
            shape=metadata.shape,
            dtype=metadata.dtype,
            synchronizer=zarr.ThreadSynchronizer(),
            compression=compressor,
        )

        # export the images
        zarr_file.create_dataset(
            name="x",
            data=x,
            shape=x.shape,
            dtype=np.int16,
            chunks=(128, x.shape[1], x.shape[2], x.shape[3]),
            synchronizer=zarr.ThreadSynchronizer(),
            compression=compressor,
        )

        # create the label arrays and copy the labels data in them
        zarr_file.create_dataset(
            name="y_value", shape=y_value.shape, dtype=np.int16, data=y_value, synchronizer=zarr.ThreadSynchronizer()
        )
        zarr_file.create_dataset(
            name="y_policy",
            shape=y_policy.shape,
            dtype=np.int16,
            data=y_policy,
            chunks=(128, y_policy.shape[1]),
            synchronizer=zarr.ThreadSynchronizer(),
            compression=compressor,
        )

        zarr_file.create_dataset(
            name="start_indices",
            shape=start_indices.shape,
            dtype=np.int32,
            data=start_indices,
            synchronizer=zarr.ThreadSynchronizer(),
            compression=compressor,
        )

        # export the parameter settings and statistics of the file
        zarr_file.create_group("/parameters")

        zarr_file.create_dataset(
            name="/parameters/pgn_name",
            shape=(1,),
            dtype="S" + str(len(self._pgn_name) + 1),
            data=[self._pgn_name.encode("ascii", "ignore")],
            compression=compressor,
        )

        zarr_file.create_dataset(
            name="/parameters/limit_nb_games",
            data=[self._limit_nb_games],
            shape=(1,),
            dtype=np.int16,
            compression=compressor,
        )
        zarr_file.create_dataset(
            name="/parameters/batch_size", shape=(1,), dtype=np.int16, data=[self._batch_size], compression=compressor
        )
        zarr_file.create_dataset(
            name="/parameters/max_nb_files",
            shape=(1,),
            dtype=np.int16,
            data=[self._max_nb_files],
            compression=compressor,
        )
        zarr_file.create_dataset(
            name="/parameters/min_elo_both",
            shape=(1,),
            dtype=np.int16,
            data=[self._min_elo_both],
            compression=compressor,
        )
        if self._compression is not None:
            zarr_file.create_dataset(
                "/parameters/compression",
                shape=(1,),
                dtype="S" + str(len(self._compression) + 1),
                data=[self._compression.encode("ascii", "ignore")],
                compression=compressor,
            )
        # https://stackoverflow.com/questions/23220513/storing-a-list-of-strings-to-a-hdf5-dataset-from-python
        ascii_list = [n.encode("ascii", "ignore") for n in self._termination_conditions]
        max_length = max(len(s) for s in self._termination_conditions)
        zarr_file.create_dataset(
            "/parameters/termination_conditions",
            shape=(1, 1),
            dtype="S" + str(max_length),
            data=ascii_list,
            compression=compressor,
        )

        zarr_file.create_group("/statistics")
        zarr_file.create_dataset(
            "/statistics/number_selected_games", shape=(1,), dtype=np.int16, data=[len(pgn_sel)], compression=compressor
        )
        zarr_file.create_dataset(
            "/statistics/game_idx_start", shape=(1,), dtype=np.int16, data=[game_idx_start], compression=compressor
        )
        zarr_file.create_dataset(
            "/statistics/game_idx_end", shape=(1,), dtype=np.int16, data=[game_idx_end], compression=compressor
        )
        zarr_file.create_dataset(
            "/statistics/white_wins", shape=(1,), dtype=np.int16, data=[nb_white_wins], compression=compressor
        )
        zarr_file.create_dataset(
            "/statistics/black_wins", shape=(1,), dtype=np.int16, data=[nb_black_wins], compression=compressor
        )
        zarr_file.create_dataset(
            "/statistics/draws", shape=(1,), dtype=np.int16, data=[nb_draws], compression=compressor
        )

        store.close()

        logging.debug("dataset was exported to: %s", zarr_path)

        return True


def export_pgn_to_datasetfile():
    PGN2PlanesConverter(
        limit_nb_games_to_analyze=10000,
        nb_games_per_file=1000,
        max_nb_files=3,
        min_elo_both=1700,
        termination_conditions=["Normal"],
        log_lvl=logging.DEBUG,
        compression="lz4",
        clevel=5,
        dataset_type="train",
    ).convert_pgn_to_planes(pgn_name="lichess_db_crazyhouse_rated_2016-01.pgn")


def export_mate_in_one_scenarios():
    PGN2PlanesConverter(
        limit_nb_games_to_analyze=10024,
        nb_games_per_file=1000,
        max_nb_files=1,
        min_elo_both=1700,
        termination_conditions=["Normal"],
        log_lvl=logging.DEBUG,
        compression="lz4",
        clevel=5,
        dataset_type="mate_in_one",
    ).convert_pgn_to_planes(pgn_name="lichess_db_crazyhouse_rated_2018-04.pgn")


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # export_mate_in_one_scenarios()
    export_pgn_to_datasetfile()
