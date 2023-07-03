"""
@file: input_representation.py
Created on 03.07.23
@project: CrazyAra
@author: queensgambit

Input representation 2.0 for crazyhouse based on the input representation 1.0 of chess.
"""

import chess
from DeepCrazyhouse.src.domain.variants.classical_chess.input_representation import board_to_planes as\
    board_to_planes_chess_v1


def board_to_planes(board: chess.Board, board_occ, normalize=True, last_moves=None):
    """
    Returns the plane representation 2.0 of a given board state for crazyhouse.
    This representation is based on the chess representation 1.0 and adds missing additional information for crazyhouse.

    ## Crazyhouse

    Feature | Planes

    Chess inputs planes features as in chess v1.0 | 39

    ---
    39 planes

    * * *

    P1 prisoner count | 5 (pieces are ordered: PAWN, KNIGHT, BISHOP, ROOK, QUEEN) (excluding the KING)

    P2 prisoner count | 5 (pieces are ordered: PAWN, KNIGHT, BISHOP, ROOK, QUEEN) (excluding the KING)

    P1 Promoted Pawns Mask | 1 (binary map indicating the pieces which have been promoted)

    P2 Promoted Pawns Mask | 1 (binary map indicating the pieces which have been promoted)

    ---
    12 planes

    The total number of planes is calculated as follows:
    # --------------
    39 + 12
    Total: 51 planes

    """

    planes = board_to_planes_chess_v1(board, board_occ, normalize, last_moves)
    # TODO: Add crazyhouse info
    return planes
