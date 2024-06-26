"""
@file: game_phase_detector.py
Created on 08.06.2023
@project: CrazyAra
@author: HelpstoneX

Analyses a given board state defined by a python-chess object and outputs the game phase according to a given definition
"""


import chess
import chess.pgn
import numpy as np
import matplotlib.pyplot as plt
import io
from DeepCrazyhouse.configs.main_config import main_config
import os
import re


def get_majors_and_minors_count(board):
    """
    Returns the number of major and minor pieces (not including king) currently present on the board (either color)

    :param board:  python-chess board object
    :return: pieces_left - integer representing how many pieces are left
    """
    pieces_left = bin(board.queens | board.rooks | board.knights | board.bishops).count("1")
    return pieces_left


def is_backrank_sparse(board, max_pieces_allowed=3):
    """
    Determines whether the backrank of either player is sparse
    where sparseness is defined by the amount of pieces on the first (for white) or last (for black) rank

    :param board:  python-chess board object
    :param max_pieces_allowed: integer representing the maximum pieces (including the king) allowed on the backrank
                               for it to be considered sparse
    :return: backrank_sparseness - boolean representing whether either backrank is currently sparse
    """
    white_backrank_sparse = bin(board.occupied_co[chess.WHITE] & chess.BB_RANK_1).count("1") <= max_pieces_allowed
    black_backrank_sparse = bin(board.occupied_co[chess.BLACK] & chess.BB_RANK_8).count("1") <= max_pieces_allowed
    return white_backrank_sparse or black_backrank_sparse


def score(num_white_pieces_in_region, num_black_pieces_in_region, rank):
    """
    Calculates the mixedness contribution of a particular 2x2 square/region

    :param num_white_pieces_in_region: integer representing the amount of white pieces in the current 2x2 region
    :param num_black_pieces_in_region: integer representing the amount of black pieces in the current 2x2 region
    :param rank: rank of the current 2x2 region
    :return: mixedness_score - integer representing the mixedness score of the current 2x2 square
    """
    score_map = {
        (0, 0): 0,
        (1, 0): 1 + (8 - rank),
        (2, 0): 2 + (rank - 2) if rank > 2 else 0,
        (3, 0): 3 + (rank - 1) if rank > 1 else 0,
        (4, 0): 3 + (rank - 1) if rank > 1 else 0,
        (0, 1): 1 + rank,
        (1, 1): 5 + abs(3 - rank),
        (2, 1): 4 + rank,
        (3, 1): 5 + rank,
        (0, 2): 2 + (6 - rank) if rank < 6 else 0,
        (1, 2): 4 + (6 - rank),
        (2, 2): 7,
        (0, 3): 3 + (7 - rank) if rank < 7 else 0,
        (1, 3): 5 + (6 - rank),
        (0, 4): 3 + (7 - rank) if rank < 7 else 0
    }
    return score_map.get((num_white_pieces_in_region, num_black_pieces_in_region), 0)


def get_mixedness(board):
    """
    Calculates the mixedness of a position based on the lichess definition of mixedness,
    which is roughly speaking the amount of intertwining of black and white pieces in all 2x2 squares of the board
    more info: https://github.com/lichess-org/scalachess/blob/master/src/main/scala/Divider.scala

    :param board: python-chess board object
    :return: mixedness_score - integer representing the current mixedness score of the position
                               (according to the lichess definition)
    """
    mix = 0

    for rank_idx in range(7):  # use ranks 1 to 7 (indices 0 to 6)
        for file_idx in range(7):  # use files A to G (indices 0 to 6)
            num_white_pieces_in_region = 0
            num_black_pieces_in_region = 0
            for dx in [0, 1]:
                for dy in [0, 1]:
                    square = chess.square(file_idx+dx, rank_idx+dy)
                    if board.piece_at(square):
                        if board.piece_at(square).color == chess.WHITE:
                            num_white_pieces_in_region += 1
                        else:
                            num_black_pieces_in_region += 1
            mix += score(num_white_pieces_in_region, num_black_pieces_in_region, rank_idx + 1)

    return mix


def get_game_phase(board, phase_definition="lichess", average_movecount_per_game=42.85):
    """
    Determines the game phase based on the current board state and the given phase definition type

    :param board: python-chess board object
    :param phase_definition: determines, which phase definition type to use,
                             either "lichess"
                             or "movecountX" where X describes the amount of phases
                             (separated by equidistant move count buckets)
    :param average_movecount_per_game: specifies the average movecount per game
                                       (used to determine phase borders when using phases by movecount)
    :return: str - str representation of the phase (for lichess definition) or empty str
             num_majors_and_minors - the amount of major and minor pieces left (for lichess phase EDA purposes)
             backrank_sparse - whether the backrank of either player is sparse (for lichess phase EDA purposes)
             mixedness_score - current mixedness score of the position (for lichess phase EDA purposes)
             phase - integer from 0 to num_phases-1 representing the phase the current position belongs to
    """

    if phase_definition == "lichess":
        # returns the game phase based on the lichess definition implemented in:
        # https://github.com/lichess-org/scalachess/blob/master/src/main/scala/Divider.scala

        num_majors_and_minors = get_majors_and_minors_count(board)
        backrank_sparse = is_backrank_sparse(board)
        mixedness_score = get_mixedness(board)

        if num_majors_and_minors <= 6:
            return "endgame", num_majors_and_minors, backrank_sparse, mixedness_score, 2
        elif num_majors_and_minors <= 10 or backrank_sparse or (mixedness_score > 150):
            return "midgame", num_majors_and_minors, backrank_sparse, mixedness_score, 1
        else:
            return "opening", num_majors_and_minors, backrank_sparse, mixedness_score, 0

    # matches "movecount" directly followed by a number
    pattern_match_result = re.match(r"\bmovecount(\d+)", phase_definition)

    if pattern_match_result:  # if it is a valid match
        # use number at the end of the string to determine the number of phases to be used
        num_phases = int(pattern_match_result.group(1))
        phase_length = round(average_movecount_per_game/num_phases)

        # board.fullmove_number describes the move number of the next move that happens in the game,
        # e.g., after 8 half moves board.fullmove_number is 5
        # so we use board.fullmove_number -1 to get the current full moves played
        moves_completed = board.fullmove_number - 1
        phase = int(moves_completed/phase_length)  # determine phase by rounding down to the next integer
        phase = min(phase, num_phases-1)  # ensure that all higher results are attributed to the last phase
        return "", 0, 0, 0, phase

    else:
        return "Phase definition not supported or wrongly formatted. Should be 'movecountX' or 'lichess'"


if __name__ == "__main__":
    print(get_game_phase(chess.Board("q6k/P1P5/3p2Q1/5p1p/3N4/3b3P/5KP1/R3R3 w - - 1 36"), "movecount4"))
    print("done")
