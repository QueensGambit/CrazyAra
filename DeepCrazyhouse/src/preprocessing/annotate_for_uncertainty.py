"""
@file: annotate_for_uncertainty.py
Created on 15.03.23
@project: DeepCrazyhouse
@author: Martin Ruzicka and QueensGambit

Appends eval_single and eval_search to the crazyhouse data set.
"""
import subprocess
from time import time
from numcodecs import Blosc

import sys
sys.path.insert(0,'../../../')
import logging

import numpy as np
import glob
import zarr
from DeepCrazyhouse.src.domain.variants.crazyhouse.input_representation import planes_to_board

from DeepCrazyhouse.configs.main_config import main_config
from DeepCrazyhouse.src.preprocessing.dataset_loader import load_pgn_dataset_file



def put(command, engine):
    engine.stdin.write(command)


def get(engine):
    # using the 'isready' command (engine has to answer 'readyok')
    # to indicate current last line of stdout
    put('isready\n', engine)
    # print('\nengine:')
    while True:
        text = engine.stdout.readline().strip()
        if text == 'readyok':
            break


def get_eval_init(fen):
    # using the 'isready' command (engine has to answer 'readyok')
    # to indicate current last line of stdout
    get(engine_init)
    put('position fen ' + fen, engine_init)
    get(engine_init)
    put('go nodes 1', engine_init)
    put('isready\n', engine_init)
    while True:
        text = engine_init.stdout.readline().strip()
        if text == 'readyok':
            break
        if text.__contains__('value'):
            # print('\t' + text)
            txt = text.split(' ')
            idx = txt.index('value')
            result = fen, txt[idx + 1]
    return result


def get_eval(fen, num_nodes, engine):
    # using the 'isready' command (engine has to answer 'readyok')
    # to indicate current last line of stdout
    # print("first get")
    if engine is None:
            return None
    get(engine)
    put('position fen ' + fen + '\n', engine)
    get(engine)
    put('go nodes ' + num_nodes + '\n', engine)
    put('isready\n', engine)
    while True:
        text = engine.stdout.readline().strip()
        if text == 'readyok':
            break
        if text.__contains__('value'):
            txt = text.split(' ')
            idx = txt.index('value')
            result = txt[idx + 1]
    return result


def analyze_fen(game):
    eval_init = np.empty(len(game))  # used to store the eval of the net, i.e. CrazyAra without search
    eval_search = np.empty(len(game))  # used to store the eval after search with NUM_PLAYOUTS playouts
    NUM_PLAYOUTS = '800'  # the number of playouts in CrazyAra that we use to get the "ground-truth"
    # Iterate through all moves (except the last one) and play them on a board.
    # you don't want to push the last move on the board because you had no evaluation to learn from in this case
    # The moves get pushed at the end of the for-loop and is only used in the next loop.
    # Therefor we can iterate over 'all' moves
    i = 0
    for move in game:
        fen = move.fen()
        if engine_search is not None:
            result_search = get_eval(fen, NUM_PLAYOUTS, engine_search)
            eval_search[i] = float(result_search)
        else:
            result_search = None
        result_init = get_eval(fen, '1', engine_init)
        eval_init[i] = float(result_init)
        i += 1
    return eval_search, eval_init


def zarr_test(filepath, results_search, results_init):
    zarr_filepath = filepath
    store = zarr.ZipStore(zarr_filepath, mode="a")
    zarr_file = zarr.group(store=store, overwrite=False)
    compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)

    eval_init_np = results_init
    eval_search_np = results_search

    zarr_file.create_dataset(
        name="eval_single",   # eval_init
        data=eval_init_np,
        shape=eval_init_np.shape,
        dtype=eval_init_np.dtype,
        chunks=(eval_init_np.shape[0]),
        synchronizer=zarr.ThreadSynchronizer(),
        compression=compressor,
    )
    #if eval_search_np is not None:
    #zarr_file.create_dataset(
    #    name="eval_search",
    #    data=eval_search_np,
    #    shape=eval_search_np.shape,
    #    dtype=eval_search_np.dtype,
    #    chunks=(eval_search_np.shape[0]),
    #    synchronizer=zarr.ThreadSynchronizer(),
    #    compression=compressor,
    #)
    store.close()


if __name__ == "__main__":


    engine_dir = "/home/queensgambit/Desktop/CrazyAra-builds/build-engine-Desktop-RelWithDebInfo/"
    engine_file_path = engine_dir + "CrazyAra"
    model_dir = engine_dir + "model/CrazyAra/crazyhouse"

    engine_init = subprocess.Popen(
        engine_file_path,
        universal_newlines=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        bufsize=1,
    )
    #engine_search = subprocess.Popen(
    #    engine_file_path,
    #    universal_newlines=True,
    #    stdin=subprocess.PIPE,
    #    stdout=subprocess.PIPE,
    #    bufsize=1,
    #)
    engine_search = None

    ROOT = logging.getLogger()
    ROOT.setLevel(logging.INFO)
    put('setoption name Use_Raw_Network value true', engine_init)
    put('\n', engine_init)
    put(f'setoption name Model_Directory value {model_dir}', engine_init)
    put('\n', engine_init)
    put('setoption name MCTS_Solver value false', engine_init)
    put('\n', engine_init)
    get(engine_init)
    print("Engine ready!")

    dataset_types = ["train", "val", "test", "mate_in_one"]

    for dataset_type in dataset_types:
        if dataset_type == "train":
            zarr_filepaths = glob.glob(main_config["planes_train_dir"] + "**/*.zip")
        elif dataset_type == "val":
            zarr_filepaths = glob.glob(main_config["planes_val_dir"] + "**/*.zip")
        elif dataset_type == "test":
            zarr_filepaths = glob.glob(main_config["planes_test_dir"] + "**/*.zip")
        elif dataset_type == "mate_in_one":
            zarr_filepaths = glob.glob(main_config["planes_mate_in_one_dir"] + "**/*.zip")

        t_start = time()

        for filepath in zarr_filepaths:
            i = 0
            game = []
            j = 1
            results_search = np.array([])
            results_init = np.array([])
            start_indices, planes, _, _, _, _, _, _ = load_pgn_dataset_file(filepath, True, False, 0)
            for plane in planes:
                game.append(planes_to_board(planes=plane))
                i += 1
                if start_indices[j] == i or i == len(planes):
                    print(j)
                    eval_search, eval_init = analyze_fen(game)
                    results_search = np.append(results_search, eval_search)
                    results_init = np.append(results_init, eval_init)
                    if j != len(start_indices) - 1:
                        j += 1
                    game = []
                    #if j == 5:
                    #   break
            zarr_test(filepath, results_search, results_init)
        print(f"done in {time()-t_start} seconds")
