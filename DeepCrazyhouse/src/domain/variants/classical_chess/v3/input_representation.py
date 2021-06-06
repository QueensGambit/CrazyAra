"""
@file: input_representation.py
Created on 05.06.21
@project: CrazyAra
@author: queensgambit

Input representation 3.0 - No legal move information, but with history again and 50 move rule information.
"""
import chess
from DeepCrazyhouse.src.domain.variants.constants import BOARD_WIDTH, BOARD_HEIGHT, NB_CHANNELS_TOTAL,\
    NB_LAST_MOVES, NB_CHANNELS_PER_HISTORY_ITEM
from DeepCrazyhouse.src.domain.util import opposite_colored_bishops, get_row_col, np, checkerboard, checkers
from DeepCrazyhouse.src.domain.variants.classical_chess.v2.input_representation import set_pieces, set_castling_rights,\
    set_ep_square

NORMALIZE_MOBILITY = 64
NORMALIZE_PIECE_NUMBER = 8
NORMALIZE_50_MOVE_RULE = 50
# These constant describe the starting channel for the corresponding info
CHANNEL_PIECES = 0
CHANNEL_REPETITION = 12
CHANNEL_EN_PASSANT = 14
CHANNEL_CASTLING = 15
CHANNEL_NO_PROGRESS = 19
CHANNEL_LAST_MOVES = 20
CHANNEL_IS_960 = 36
CHANNEL_PIECE_MASK = 37
CHANNEL_CHECKERBOARD = 39
CHANNEL_MATERIAL_DIFF = 40
CHANNEL_OPP_BISHOPS = 45
CHANNEL_CHECKERS = 46
CHANNEL_MATERIAL_COUNT = 47


def board_to_planes(board: chess.Board, board_occ, normalize=True, last_moves=None):
    """
    Gets the plane representation of a given board state.

    ## Chess:

    Feature | Planes

    --- | ---

    P1 piece | 6 (pieces are ordered: PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)

    P2 piece | 6 (pieces are ordered: PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)

    Repetitions | 2 (two planes (full zeros/ones) indicating how often the board positions has occurred)

    En-passant square | 1 (Binary map indicating the square where en-passant capture is possible)

    ---
    15 planes

    * * *

    P1 castling | 2 (One if castling is possible, else zero)

    P2 castling | 2 (One if castling is possible, else zero)

    No-progress count | 1 (Setting the no progress counter as integer values, (described by uci halfmoves format)

    ---
    5 planes

    * * *

    Last 8 moves | 16 (indicated by origin and destination square, the most recent move is described by first 2 planes)

    ---
    16 planes

    * * *

    is960 = | 1 (boolean, 1 when active)

    ---
    1 plane

    ---

    P1 pieces | 1 | A grouped mask of all WHITE pieces |
    P2 pieces | 1 | A grouped mask of all BLACK pieces |
    Checkerboard | 1 | A chess board pattern |
    P1 Material Diff | 5 | (pieces are ordered: PAWN, KNIGHT, BISHOP, ROOK, QUEEN), normalized with 8, + means positive, - means negative |
    Opposite Color Bishops | 1 | Indicates if they are only two bishops and the bishops are opposite color |
    Checkers | 1 | Indicates all pieces giving check |
    P1 Material Count | 5 | (pieces are ordered: PAWN, KNIGHT, BISHOP, ROOK, QUEEN), normalized with 8 |
    ---
    15 planes

    The total number of planes is calculated as follows:
    # --------------
    15 + 5 + 16 + 1 + 15
    Total: 52 planes

    :param board: Board handle (Python-chess object)
    :param board_occ: Number of board occurences
    :param normalize: True if the inputs shall be normalized to the range [0.-1.]
    :param last_moves: List of last last moves. The most recent move is the first entry.
    :return: planes - the plane representation of the current board state
    """

    # return the plane representation of the given board
    # return variants.board_to_planes(board, board_occ, normalize, mode=MODE_CHESS)
    planes = np.zeros((NB_CHANNELS_TOTAL, BOARD_HEIGHT, BOARD_WIDTH))

    # channel will be incremented by 1 at first plane
    channel = 0
    me = board.turn
    you = not board.turn
    colors = [me, you]

    # mirror all bitboard entries for the black player
    mirror = board.turn == chess.BLACK

    assert channel == CHANNEL_PIECES
    # Fill in the piece positions
    # Channel: 0 - 11
    # Iterate over both color starting with WHITE
    for color in colors:
        # the PIECE_TYPE is an integer list in python-chess
        for piece_type in chess.PIECE_TYPES:
            # iterate over the piece mask and receive every position square of it
            for pos in board.pieces(piece_type, color):
                row, col = get_row_col(pos, mirror=mirror)
                # set the bit at the right position
                planes[channel, row, col] = 1
            channel += 1

    assert channel == CHANNEL_REPETITION
    # Channel: 12 - 13
    # set how often the position has already occurred in the game (default 0 times)
    # this is used to check for claiming the 3 fold repetition rule
    if board_occ >= 1:
        planes[channel, :, :] = 1
        if board_occ >= 2:
            planes[channel + 1, :, :] = 1
    channel += 2

    # Channel: 14
    # En Passant Square
    assert channel == CHANNEL_EN_PASSANT
    if board.ep_square and board.has_legal_en_passant(): # is not None:
        row, col = get_row_col(board.ep_square, mirror=mirror)
        planes[channel, row, col] = 1
    channel += 1

    # Channel: 15 - 18
    assert channel == CHANNEL_CASTLING
    for color in colors:
        # check for King Side Castling
        if board.has_kingside_castling_rights(color):
            planes[channel, :, :] = 1
        channel += 1
        # check for Queen Side Castling
        if board.has_queenside_castling_rights(color):
            planes[channel, :, :] = 1
        channel += 1

    # Channel: 19
    # (IV.4) No Progress Count
    # define a no 'progress' counter
    # it gets incremented by 1 each move
    # however, whenever a piece gets dropped, a piece is captured or a pawn is moved, it is reset to 0
    # halfmove_clock is an official metric in fen notation
    #  -> see: https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
    # check how often the position has already occurred in the game
    assert channel == CHANNEL_NO_PROGRESS
    planes[channel, :, :] = board.halfmove_clock
    channel += 1

    # Channel: 20 - 35
    assert channel == CHANNEL_LAST_MOVES
    # Last 8 moves
    if last_moves:
        assert(len(last_moves) == NB_LAST_MOVES)
        for move in last_moves:
            if move:
                from_row, from_col = get_row_col(move.from_square, mirror=mirror)
                to_row, to_col = get_row_col(move.to_square, mirror=mirror)
                planes[channel, from_row, from_col] = 1
                channel += 1
                planes[channel, to_row, to_col] = 1
                channel += 1
            else:
                channel += 2
    else:
        channel += NB_LAST_MOVES * NB_CHANNELS_PER_HISTORY_ITEM

    # Channel: 36
    # Chess960
    assert channel == CHANNEL_IS_960
    if board.chess960:
        planes[channel + 1, :, :] = 1
    channel += 1

    # Channel: 37 - 38
    # All white pieces and black pieces in a single map
    assert channel == CHANNEL_PIECE_MASK
    for color in colors:
        # the PIECE_TYPE is an integer list in python-chess
        for piece_type in chess.PIECE_TYPES:
            # iterate over the piece mask and receive every position square of it
            for pos in board.pieces(piece_type, color):
                row, col = get_row_col(pos, mirror=mirror)
                # set the bit at the right position
                planes[channel, row, col] = 1
        channel += 1

    # Channel: 39
    # Checkerboard
    assert(channel == CHANNEL_CHECKERBOARD)
    planes[channel, :, :] = checkerboard()
    channel += 1

    # Channel: 40 - 44
    # Relative material difference (negative if less pieces than opponent and positive if more)
    # iterate over all pieces except the king
    assert channel == CHANNEL_MATERIAL_DIFF
    for piece_type in chess.PIECE_TYPES[:-1]:
        material_count = len(board.pieces(piece_type, me)) - len(board.pieces(piece_type, you))
        planes[channel, :, :] = material_count / NORMALIZE_PIECE_NUMBER if normalize else material_count
        channel += 1

    # Channel: 45
    # Opposite color bishops
    assert (channel == CHANNEL_OPP_BISHOPS)
    if opposite_colored_bishops(board):
        planes[channel, :, :] = 1
    channel += 1

    # Channel: 46
    # Checkers
    assert channel == CHANNEL_CHECKERS
    board_checkers = checkers(board)
    if board_checkers:
        # iterate over the piece mask and receive every position square of it
        for pos in chess.SquareSet(board_checkers):
            row, col = get_row_col(pos, mirror=mirror)
            # set the bit at the right position
            planes[channel, row, col] = 1
    channel += 1

    # Channel: 47 - 51
    # Material
    assert channel == CHANNEL_MATERIAL_COUNT
    for piece_type in chess.PIECE_TYPES[:-1]:
        material_count = len(board.pieces(piece_type, me))
        planes[channel, :, :] = material_count / NORMALIZE_PIECE_NUMBER if normalize else material_count
        channel += 1

    assert channel == NB_CHANNELS_TOTAL

    return planes


def planes_to_board(planes):
    """
    Converts a board in plane representation to the python chess board representation
    see get_planes_of_board() for input encoding description
    ! Board is always returned with WHITE to move and move number and no progress counter = 0 !

    :param planes: Input plane representation
    :return: python chess board object
    """
    is960 = planes[CHANNEL_IS_960, 0, 0] == 1
    board = chess.Board(chess960=is960)
    # clear the full board (the pieces will be set later)
    board.clear()

    set_pieces(board, planes)

    # (I.5) En Passant Square
    # mark the square where an en-passant capture is possible
    set_ep_square(board, CHANNEL_EN_PASSANT, planes)

    # (II.2) Castling Rights
    set_castling_rights(board, CHANNEL_CASTLING, planes, is960)

    return board


def normalize_input_planes(planes):
    """
    Normalizes input planes to range [0,1]. Works in place / meaning the input parameter x is manipulated
    :param planes: Input planes representation
    :return: The normalized planes
    """
    channel = CHANNEL_MATERIAL_DIFF
    for _ in chess.PIECE_TYPES[:-1]:
        planes[channel, :, :] /= NORMALIZE_PIECE_NUMBER
        channel += 1
    planes[CHANNEL_NO_PROGRESS, :, :] /= NORMALIZE_50_MOVE_RULE
    channel = CHANNEL_MATERIAL_COUNT
    for _ in chess.PIECE_TYPES[:-1]:
        planes[channel, :, :] /= NORMALIZE_PIECE_NUMBER
        channel += 1

    return planes
