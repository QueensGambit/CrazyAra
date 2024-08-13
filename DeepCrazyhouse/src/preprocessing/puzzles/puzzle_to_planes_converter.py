"""
@file: puzzle_to_planes_converter.py
Created on 05.04.24
@project: DeepCrazyhouse
@author: queensgambit

Format

Puzzles are formatted as standard CSV. The fields are as follows:

PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags

Sample

00sHx,q3k1nr/1pp1nQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 b k - 0 17,e8d7 a2e6 d7d8 f7f8,1760,80,83,72,mate mateIn2 middlegame short,https://lichess.org/yyznGmXs/black#34,Italian_Game Italian_Game_Classical_Variation
00sJ9,r3r1k1/p4ppp/2p2n2/1p6/3P1qb1/2NQR3/PPB2PP1/R1B3K1 w - - 5 18,e3g3 e8e1 g1h2 e1c1 a1c1 f4h6 h2g1 h6c1,2671,105,87,325,advantage attraction fork middlegame sacrifice veryLong,https://lichess.org/gyFeQsOE#35,French_Defense French_Defense_Exchange_Variation
00sJb,Q1b2r1k/p2np2p/5bp1/q7/5P2/4B3/PPP3PP/2KR1B1R w - - 1 17,d1d7 a5e1 d7d1 e1e3 c1b1 e3b6,2235,76,97,64,advantage fork long,https://lichess.org/kiuvTFoE#33,Sicilian_Defense Sicilian_Defense_Dragon_Variation
00sO1,1k1r4/pp3pp1/2p1p3/4b3/P3n1P1/8/KPP2PN1/3rBR1R b - - 2 31,b8c7 e1a5 b7b6 f1d1,998,85,94,293,advantage discoveredAttack master middlegame short,https://lichess.org/vsfFkG0s/black#62,

Source: https://database.lichess.org/#puzzles

We can use all moves except the first move as our training samples.
"""
from pathlib import Path

import sys

sys.path.insert(0, '../../../../')
import pandas as pd
import chess
import chess.engine
import logging
from time import time
from multiprocessing import Pool
import zarr
import numpy as np
from numcodecs import Blosc
import argparse
import glob
import datetime
import os

from DeepCrazyhouse.src.domain.util import get_dic_sorted_by_key
from DeepCrazyhouse.src.preprocessing.pgn_converter_util import get_planes_from_move_sequence
from DeepCrazyhouse.src.preprocessing.pgn_to_planes_converter import export_main_data


def get_eval(board: chess.Board, engine: chess.engine):
    """
    Evaluates the given board position with the given engine and returns -1, 0, +1 respective to Losing, Drawn or Winning
    :param board: Board position
    :param engine: Chess engine object
    """
    if not board.is_game_over():
        result = engine.analyse(board, chess.engine.Limit(time=0.1))
        print(result)

        if result['score'].is_mate():
            return -1
        elif result['score'].relative.score() > 100:
            return 1
        elif result['score'].relative.score() < 100:
            return -1
        else:
            return 0
    elif board.is_checkmate():
        return -1
    else:
        return 0


def sort_concat_data(data_dic: dict):
    """Sorts the dictionary object based on the index and returns the concatenated version.
    :param data_dic: Data dictionary object
    return: np.array
    """
    data = get_dic_sorted_by_key(data_dic)
    return np.concatenate(data, axis=0)


def process_chunk(chunk_id: int, chunksize: int, df_chunk: pd.DataFrame, export_dir: Path, processes: int):
    """
    Processes a data frame chunk by exporting all chess puzzle positions in this chunk.
    :param chunk_id: Unique id of the data chunk
    :param chunksize: Size of each chunk
    :param df_chunk: Data frame chunk
    :param export_dir: Export directory where the .zip files will be stored
    :param processes: Number of processes
    return: None
    """

    # engine = chess.engine.SimpleEngine.popen_uci(r"stockfish")

    logging.info("starting conversion to planes...")
    pool = Pool(processes=processes)
    x_dic = {}
    y_value_dic = {}
    y_policy_dic = {}
    plys_to_end_dic = {}
    phase_vector_dic = {}

    params_inp = _prepare_parameter_inputs(chunk_id, chunksize, df_chunk)

    # use pool.starmap here and parallelize the export
    for puzzle_idx, (x, y_value, y_policy, plys_to_end, phase_vector) in enumerate(pool.starmap(
            get_planes_from_move_sequence, params_inp)):
        # metadata_dic[puzzle_idx] = metadata
        x_dic[puzzle_idx] = x
        y_value_dic[puzzle_idx] = y_value
        y_policy_dic[puzzle_idx] = y_policy
        plys_to_end_dic[puzzle_idx] = plys_to_end
        phase_vector_dic[puzzle_idx] = phase_vector
    pool.close()
    pool.join()

    _export_data(chunk_id, export_dir, phase_vector_dic, plys_to_end_dic, x_dic, y_policy_dic, y_value_dic)

    # engine.quit()


def _prepare_parameter_inputs(chunk_id, chunksize, df_chunk):
    params_inp = []
    for puzzle_idx in range(chunk_id * chunksize, chunk_id * chunksize + len(df_chunk)):
        board = chess.Board(fen=df_chunk["FEN"][puzzle_idx])
        moves = df_chunk["Moves"][puzzle_idx]

        for move in moves.split(" "):
            board.push_uci(move)

        # skip evaluation with Stockfish
        # eval = -get_eval(board, engine)
        eval = 1

        board = chess.Board(fen=df_chunk["FEN"][puzzle_idx])
        board_2 = chess.Board(fen=df_chunk["FEN"][puzzle_idx])
        all_moves = []
        moves_uci = moves.split(" ")
        for idx, move in enumerate(moves_uci):
            board_2.push_uci(move)
            if idx == 0:
                board.push_uci(move)
            else:
                all_moves.append(board_2.move_stack[-1])

        params_inp.append((board, eval, all_moves, False))
    return params_inp


def _export_data(chunk_id, export_dir, phase_vector_dic, plys_to_end_dic, x_dic, y_policy_dic, y_value_dic):
    # open a dataset file and create arrays
    zarr_path = export_dir / f"puzzles_{chunk_id}.zip"
    store = zarr.ZipStore(str(zarr_path), mode="w")
    zarr_file = zarr.group(store=store, overwrite=True)
    # metadata = sort_concat_data(metadata_dic)
    x = sort_concat_data(x_dic)
    y_value = sort_concat_data(y_value_dic)
    y_policy = sort_concat_data(y_policy_dic)
    plys_to_end = sort_concat_data(plys_to_end_dic)
    phase_vector = sort_concat_data(phase_vector_dic)
    start_indices = np.zeros(len(x))  # create a list which describes where each game starts
    # define the compressor object
    compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
    export_main_data(zarr_file, compressor, start_indices, x, y_value, y_policy, plys_to_end, phase_vector)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script allows converting a puzzle csv file in the lichess-format'
                                                 'into plane representation. ')

    parser.add_argument('--puzzle-csv-dir', type=str, default='./', help='Directory where the puzzle csv file is stored.')
    parser.add_argument('--export-dir', type=str, default='./', help='Directory where the .zip files will be exported to.')
    parser.add_argument('--processes', type=int, default='4', help='Number of parallel processes.')

    args = parser.parse_args()

    # check if directories exist
    puzzle_csv_dir = Path(args.puzzle_csv_dir)
    export_dir = Path(args.export_dir)

    if not puzzle_csv_dir.is_dir():
        raise Exception("The given puzzle-csv-dir is not a valid directory.")
    if not export_dir.is_dir():
        raise Exception("The given export-dir is not a valid directory.")

    puzzle_file_path = glob.glob(args.puzzle_csv_dir + "*.csv")
    if len(puzzle_file_path) == 0:
        raise Exception("The given puzzle-csv-dir does not contain a csv file.")
    puzzle_file_path = puzzle_file_path[0]

    # include current timestamp in dataset export file
    timestmp = datetime.datetime.fromtimestamp(time()).strftime("%Y-%m-%d-%H-%M-%S")
    timestmp_dir = export_dir / timestmp

    # create a directory of the current timestamp
    if not timestmp_dir.is_dir():
        os.makedirs(timestmp_dir)
    export_dir = timestmp_dir

    # https://stackoverflow.com/questions/25962114/how-do-i-read-a-large-csv-file-with-pandas#25962187
    chunksize = 10 ** 4
    with pd.read_csv(puzzle_file_path, chunksize=chunksize) as reader:
        for chunk_id, df_chunk in enumerate(reader):
            print('chunk:', df_chunk)
            process_chunk(chunk_id, chunksize, df_chunk, export_dir, args.processes)
