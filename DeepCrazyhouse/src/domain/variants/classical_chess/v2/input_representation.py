"""
@file: input_representation.py
Created on 26.05.21
@project: CrazyAra
@author: queensgambit

Input representation for chess v2.1.
This presentation avoids potential overfitting and bias, e.g. no color information, no move counter, no progress counter
and adds features which are hard for the CNN to extract, e.g. material info, number legal moves, checkerboard,
opposite color bishops.
"""
import chess
from DeepCrazyhouse.src.domain.variants.constants import BOARD_WIDTH, BOARD_HEIGHT, NB_CHANNELS_TOTAL, PIECES
from DeepCrazyhouse.src.domain.util import opposite_colored_bishops, get_row_col, np, checkerboard,\
    get_board_position_index

NORMALIZE_NB_LEGAL_MOVES = 100
NORMALIZE_PIECE_NUMBER = 8
# These constant describe the starting channel for the corresponding info
CHANNEL_PIECES = 0
CHANNEL_EN_PASSANT = 12
CHANNEL_CASTLING = 13
CHANNEL_LAST_MOVES = 17
CHANNEL_IS_960 = 33
CHANNEL_PIECE_MASK = 34
CHANNEL_CHECKERBOARD = 36
CHANNEL_MATERIAL = 37
CHANNEL_NB_LEGAL_MOVES = 42
CHANNEL_OPP_BISHOPS = 43


def board_to_planes(board: chess.Board, normalize=True, last_moves=None):
    """
    Gets the plane representation of a given board state.

    ## Chess:

    Feature | Planes

    --- | ---

    P1 piece | 6 (pieces are ordered: PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)

    P2 piece | 6 (pieces are ordered: PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)

    En-passant square | 1 (Binary map indicating the square where en-passant capture is possible)

    ---
    13 planes

    * * *

    P1 castling | 2 (One if castling is possible, else zero)

    P2 castling | 2 (One if castling is possible, else zero)

    ---
    4 planes

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
    Chessboard | 1 | A chess board pattern |
    P1 Material Diff | 5 | (pieces are ordered: PAWN, KNIGHT, BISHOP, ROOK, QUEEN), normalized with 8, + means positive, - means negative |
    P1 Number legal moves | 1 | Gives the number of legal moves for the active player |
    Opposite Color Bishops | 1 | Indicates if they are only two bishops and the bishops are opposite color |

    ---
    10 planes

    The total number of planes is calculated as follows:
    # --------------
    13 + 4 + 16 + 1 + 10
    Total: 44 planes

    :param board: Board handle (Python-chess object)
    :param board_occ: Sets how often the board state has occurred before (by default 0)
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

    assert (channel == CHANNEL_PIECES)
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

    # Channel: 12
    # En Passant Square
    assert(channel == CHANNEL_EN_PASSANT)
    if board.ep_square is not None:
        row, col = get_row_col(board.ep_square, mirror=mirror)
        planes[channel, row, col] = 1
    channel += 1

    # Channel: 13 - 16
    assert (channel == CHANNEL_CASTLING)
    for color in colors:
        # check for King Side Castling
        if board.has_kingside_castling_rights(color):
            planes[channel, :, :] = 1
        channel += 1
        # check for Queen Side Castling
        if board.has_queenside_castling_rights(color):
            planes[channel, :, :] = 1
        channel += 1

    # Channel: 17 - 32
    assert(channel == CHANNEL_LAST_MOVES)
    # Last 8 moves
    if last_moves:
        assert(len(last_moves) == 8)
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
        channel += 16

    # Channel: 33
    # Chess960
    assert (channel == CHANNEL_IS_960)
    if board.chess960:
        planes[channel + 1, :, :] = 1
    channel += 1

    # Channel: 34 - 35
    # All white pieces and black pieces in a single map
    assert(channel == CHANNEL_PIECE_MASK)
    for color in colors:
        # the PIECE_TYPE is an integer list in python-chess
        for piece_type in chess.PIECE_TYPES:
            # iterate over the piece mask and receive every position square of it
            for pos in board.pieces(piece_type, color):
                row, col = get_row_col(pos, mirror=mirror)
                # set the bit at the right position
                planes[channel, row, col] = 1
        channel += 1

    # Channel: 36
    # Checkerboard
    assert(channel == CHANNEL_CHECKERBOARD)
    planes[channel, :, :] = checkerboard()
    channel += 1

    # Channel: 37 - 41
    # Relative material difference (negative if less pieces than opponent and positive if more)
    # iterate over all pieces except the king
    assert(channel == CHANNEL_MATERIAL)
    for piece_type in chess.PIECE_TYPES[:-1]:
        matt_diff = len(board.pieces(piece_type, me)) - len(board.pieces(piece_type, you))
        planes[channel, :, :] = matt_diff / NORMALIZE_PIECE_NUMBER if normalize else matt_diff
        channel += 1

    # Channel: 42
    # Number legal moves
    assert (channel == CHANNEL_NB_LEGAL_MOVES)
    planes[channel, :, :] = len(list(board.legal_moves)) / NORMALIZE_NB_LEGAL_MOVES if normalize else len(list(board.legal_moves))
    channel += 1

    # Channel: 43
    # Opposite color bishops
    assert (channel == CHANNEL_OPP_BISHOPS)
    if opposite_colored_bishops(board):
        planes[channel, :, :] = 1

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

    # iterate over all piece types
    for idx, piece in enumerate(PIECES):
        # iterate over all fields and set the current piece type
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                # check if there's a piece at the current position
                if planes[idx, row, col] == 1:
                    # check if the piece was promoted
                    promoted = False
                    board.set_piece_at(
                        square=get_board_position_index(row, col),
                        piece=chess.Piece.from_symbol(piece),
                        promoted=promoted,
                    )

    # (I.5) En Passant Square
    # mark the square where an en-passant capture is possible
    channel = CHANNEL_EN_PASSANT
    ep_square = np.argmax(planes[channel])
    if ep_square != 0:
        # if no entry 'one' exists, index 0 will be returned
        board.ep_square = ep_square

    # (II.2) Castling Rights
    channel = CHANNEL_CASTLING
    set_castling_rights(board, channel, planes, is960)

    return board


def set_castling_rights(board, channel, planes, is960):
    # reset the castling_rights for initialization
    # set to 0, previously called chess.BB_VOID for chess version of 0.23.X and chess.BB_EMPTY for versions > 0.27.X
    board.castling_rights = 0
    # WHITE
    # check for King Side Castling
    # White can castle with the h1 rook
    # add castling option by modifying the castling fen
    castling_fen = ""
    # check for King Side Castling
    if planes[channel, 0, 0] == 1:
        if is960:
            castling_fen += "K"
        else:
            board.castling_rights |= chess.BB_H1
    # check for Queen Side Castling
    if planes[channel + 1, 0, 0] == 1:
        if is960:
            castling_fen += "Q"
        else:
            board.castling_rights |= chess.BB_A1
    # BLACK
    # check for King Side Castling
    if planes[channel + 2, 0, 0] == 1:
        if is960:
            castling_fen += "k"
        else:
            board.castling_rights |= chess.BB_H8
    # check for Queen Side Castling
    if planes[channel + 3, 0, 0] == 1:
        if is960:
            castling_fen += "q"
        else:
            board.castling_rights |= chess.BB_A8
    # configure the castling rights
    if castling_fen:
        board.set_castling_fen(castling_fen)


def normalize_input_planes(planes):
    """
    Normalizes input planes to range [0,1]. Works in place / meaning the input parameter x is manipulated
    :param planes: Input planes representation
    :return: The normalized planes
    """
    planes[CHANNEL_NB_LEGAL_MOVES, :, :] /= NORMALIZE_NB_LEGAL_MOVES
    channel = CHANNEL_MATERIAL - 1
    for _ in chess.PIECE_TYPES[:-1]:
        planes[++channel, :, :] /= CHANNEL_MATERIAL
    return planes
