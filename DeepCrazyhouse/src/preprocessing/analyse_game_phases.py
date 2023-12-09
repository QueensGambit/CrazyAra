"""
@file: analyse_game_phases
Created on 07.07.2023
@project: CrazyAra
@author: Felix

Used to analyse the distribution of game outcomes, y_values, moves and move counts per phase
"""

import os
import sys
sys.path.insert(0,'../../../')
import glob
import chess.pgn
import shutil
import logging
import io
import numpy as np
from pathlib import Path
from DeepCrazyhouse.configs.main_config import main_config
from DeepCrazyhouse.src.preprocessing.game_phase_detector import get_game_phase
from tqdm import tqdm
from copy import deepcopy
import multiprocessing as mp
import json


def get_phase_and_move_info_from_game(pgn):
    game = chess.pgn.read_game(pgn)
    y_values = []
    phase_move_counter = {"0": 0, "1": 0, "2": 0}
    phase_y_values = {"0": list(), "1": list(), "2": list()}
    phase_moves = {"0": list(), "1": list(), "2": list()}
    board = game.board()  # get the initial board state
    # update the y value accordingly
    if board.turn == chess.WHITE:
        y_init = 1
    else:
        y_init = -1
    if game.headers["Result"] == "0-1":
        y_init *= -1
    elif game.headers["Result"] == "1/2-1/2":
        y_init = 0

    all_moves = []  # Extract all moves first and save them into a list
    for move in game.main_line():
        all_moves.append(move)

    # Iterate through all moves (except the last one) and play them on a board.
    # you don't want to push the last move on the board because you had no movement policy to learn from in this case
    # The moves get pushed at the end of the for-loop and is only used in the next loop.
    # Therefore we can iterate over 'all' moves
    for plys, move in enumerate(all_moves):
        # we insert the move i (and not i+1), because the start is the empty board position
        next_move = all_moves[plys]

        curr_phase = get_game_phase(board)[4]
        phase_move_counter[str(curr_phase)] += 1
        phase_y_values[str(curr_phase)].append(y_init)
        phase_moves[str(curr_phase)].append(next_move.uci())

        y_values.append(y_init)

        y_init *= -1  # flip the y_init value after each move
        board.push(move)  # push the next move on the board

    return [[move.uci() for move in all_moves], y_values, len(all_moves), phase_move_counter, phase_y_values, phase_moves]


def do_parallel_processing(pgns, processes):
    pool = mp.Pool(processes)

    results = list(tqdm(pool.imap(get_phase_and_move_info_from_game, pgns), total=len(pgns)))

    pool.close()

    return results


if __name__ == "__main__":

    #with open("data.json", 'r') as f:
    #    results, all_outcomes_flattened, all_elos_flattened = json.load(f)
    import_dir = main_config["pgn_train_dir"]
    pgn_filenames = os.listdir(import_dir)
    len(pgn_filenames)
    all_pgns_all_files = list()
    all_game_elos_all_files = list()
    all_game_outcomes_all_files = list()
    num_games_all_files = list()

    for file_idx, pgn_name in enumerate(tqdm(pgn_filenames)):
        logging.info(pgn_name)
        pgn = open(import_dir + pgn_name, "r")
        content = pgn.read()
        nb_games = content.count("[Result")
        logging.info("total games of file: %d", nb_games)
        all_games = content.split("[Event ")  # split the content for each single game

        for idx, _ in enumerate(all_games):
            all_games[idx] = "[Event " + all_games[idx]

        pgns = []
        del all_games[0]

        games = all_games

        for game in games:
            # only add game with at least _min_number_moves played
            if game.find(f"{5:d}. ") != -1:
                game_start_char = game.find("1. ")
                if game_start_char != -1:
                    if game[:game_start_char].find('Variant "Chess960"'):
                        # 2019-09-28: fix for chess960 because in the default position lichess denotes FEN as "?"
                        game = game.replace('[FEN "?"]',
                                            '[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]')
                    pgns.append(io.StringIO(game))

        all_pgn_sel = []
        all_game_outcomes = []
        all_game_elos = []  # list of tuples with (white Elo, black Elo)

        for game_pgn in pgns:
            # we need to create a deep copy, otherwise the end of the file is reached for later
            game_pgn_copy = deepcopy(game_pgn)
            for _, headers in chess.pgn.scan_headers(game_pgn_copy):
                try:
                    white_elo = headers["WhiteElo"]
                except KeyError:
                    white_elo = "?"
                try:
                    black_elo = headers["BlackElo"]
                except KeyError:
                    black_elo = "?"

                if headers["Result"] != "*":

                    if (white_elo != "?" and
                                 black_elo != "?" and
                                 int(white_elo) >= 1950
                                 and int(black_elo) >= 1950):

                        if headers["Result"] == "1-0":
                            all_game_outcomes.append(1)
                        elif headers["Result"] == "0-1":
                            all_game_outcomes.append(-1)
                        elif headers["Result"] == "1/2-1/2":
                            all_game_outcomes.append(0)
                        else:
                            raise Exception("Illegal Game Result: ", headers["Result"])

                        all_pgn_sel.append(game_pgn)
                        all_game_elos.append((white_elo, black_elo))

        all_pgns_all_files.append(all_pgn_sel)
        all_game_outcomes_all_files.append(all_game_outcomes)
        all_game_elos_all_files.append(all_game_elos)
        num_games_all_files.append(len(all_pgn_sel))
        logging.info("selected games of file: %d", len(all_pgn_sel))

    all_outcomes_flattened = [outcome for file_outcomes in all_game_outcomes_all_files for outcome in file_outcomes]
    all_pgns_flattened = [pgn for file_pgns in all_pgns_all_files for pgn in file_pgns]
    all_elos_flattened = [elos for file_elos in all_game_elos_all_files for elos in file_elos]

    num_draws_total = all_outcomes_flattened.count(0)
    num_white_wins_total = all_outcomes_flattened.count(1)
    num_black_wins_total = all_outcomes_flattened.count(-1)

    print(num_draws_total, num_white_wins_total, num_black_wins_total, len(all_pgns_flattened))

    all_pgns_flattened_copy = deepcopy(all_pgns_flattened)
    print("starting parallel processing")
    processes = mp.cpu_count()
    results = do_parallel_processing(all_pgns_flattened_copy, processes)

    with open("analyse_game_phase_data_train_960.json", 'w') as f:
        json.dump((results, all_outcomes_flattened, all_elos_flattened), f)

    print("done")
