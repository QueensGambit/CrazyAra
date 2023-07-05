"""
@file: input_representation.py
Created on 03.07.23
@project: CrazyAra
@author: queensgambit

Input representation 3.0 for crazyhouse based on the input representation 3.0 of chess.
"""
import chess
import DeepCrazyhouse.src.domain.variants.classical_chess.v3.input_representation as chess_v3
from DeepCrazyhouse.src.domain.variants.constants import MODE_CRAZYHOUSE
from DeepCrazyhouse.src.domain.variants.input_representation import _set_crazyhouse_info


def board_to_planes(board: chess.Board, board_occ, normalize=True, last_moves=None):
    """
    Returns the plane representation 3.0 of a given board state for crazyhouse.
    This representation is based on the chess representation 3.0 and adds missing additional information for crazyhouse.

    ## Crazyhouse

    Feature | Planes

    Chess inputs planes features as in chess v3.0 | 52

    ---
    52 planes

    * * *

    P1 prisoner count | 5 (pieces are ordered: PAWN, KNIGHT, BISHOP, ROOK, QUEEN) (excluding the KING)

    P2 prisoner count | 5 (pieces are ordered: PAWN, KNIGHT, BISHOP, ROOK, QUEEN) (excluding the KING)

    P1 Promoted Pawns Mask | 1 (binary map indicating the pieces which have been promoted)

    P2 Promoted Pawns Mask | 1 (binary map indicating the pieces which have been promoted)

    ---
    12 planes

    The total number of planes is calculated as follows:
    # --------------
    52 + 12
    Total: 64 planes

    """
    planes = chess_v3.board_to_planes(board, board_occ, normalize, last_moves)
    _set_crazyhouse_info(board, MODE_CRAZYHOUSE, planes, normalize)
    return planes
