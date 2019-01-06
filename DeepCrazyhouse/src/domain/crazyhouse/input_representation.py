"""
@file: input_representation.py
Created on 20.06.18
@project: jc39qevo-deep-learning-project
@author: queensgambit

Input representation for the Crazyhouse board state which is passed to the neural network
"""

from chess.variant import CrazyhouseBoard
from DeepCrazyhouse.src.domain.util import *
from DeepCrazyhouse.src.domain.crazyhouse.constants import *


def board_to_planes(board, board_occ=0, normalize=True):
    """
    Gets the plane representation of a given board state.
    (Now history of past board positions is used.)

    ## Crazyhouse:

    Feature | Planes

    --- | ---

    P1 piece | 6 (pieces are ordered: PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)

    P2 piece | 6 (pieces are ordered: PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)

    Repetitions | 2 (two planes (full zeros/ones) indicating how often the board positions has occurred)

    P1 prisoner count | 5 (pieces are ordered: PAWN, KNIGHT, BISHOP, ROOK, QUEEN) (excluding the KING)

    P2 prisoner count | 5 (pieces are ordered: PAWN, KNIGHT, BISHOP, ROOK, QUEEN) (excluding the KING)

    P1 Promoted Pawns Mask | 1 (binary map indicating the pieces which have been promoted)

    P2 Promoted Pawns Mask | 1 (binary map indicating the pieces which have been promoted)

    En-passant square | 1 (Binary map indicating the square where en-passant capture is possible)

    ---
    27 planes

    * * *

    Colour | 1 (all zeros for black and all ones for white)

    Total move count | 1 (integer value setting the move count (uci notation))

    P1 castling | 2 (One if castling is possible, else zero)

    P2 castling | 2 (One if castling is possible, else zero)

    No-progress count | 1 (Setting the no progress counter as integer values, (described by uci halfmoves format)

    # --------------
    7 planes

    The history list of the past 7 board states have been removed
    The total number of planes is calculated as follows:

    27 + 7
    Total: 34 planes

    :param board: Board handle (Python-chess object)
    :param board_occ: Sets how often the board state has occurred before (by default 0)
    :param normalize: True if the inputs shall be normalized to the range [0.-1.]
    :return: planes - the plane representation of the current board state
    """

    # TODO: Remove board.mirror() for black by addressing the according color channel

    # (I) Define the Input Representation for one position
    planes_pos = np.zeros((NB_CHANNELS_POS, BOARD_HEIGHT, BOARD_WIDTH))
    planes_const = np.zeros((NB_CHANNELS_CONST, BOARD_HEIGHT, BOARD_WIDTH))

    # save whose turn it is
    board_turn = chess.WHITE

    # check who's player turn it is and flip the board if it's black turn
    if board.turn == chess.BLACK:
        board_turn = chess.BLACK
        board = board.mirror()

    # Fill in the piece positions

    # Iterate over both color starting with WHITE
    for z, color in enumerate(chess.COLORS):
        # the PIECE_TYPE is an integer list in python-chess
        for piece_type in chess.PIECE_TYPES:
            # define the channel by the piecetype (the input representation uses the same ordering as python-chess)
            # we add an offset for the black pieces
            # note that we subtract 1 because in python chess the PAWN has index 1 and not 0
            channel = (piece_type - 1) + z * len(chess.PIECE_TYPES)
            # iterate over the piece mask and receive every position square of it
            for pos in board.pieces(piece_type, color):
                row, col = get_row_col(pos)
                # set the bit at the right position
                planes_pos[channel, row, col] = 1

    # (II) Fill in the Repetition Data
    # a game to test out if everything is working correctly is: https://lichess.org/jkItXBWy#73
    ch = CHANNEL_MAPPING_POS["repetitions"]

    # set how often the position has already occurred in the game (default 0 times)
    # this is used to check for claiming the 3 fold repetition rule
    if board_occ >= 1:
        planes_pos[ch, :, :] = 1
        if board_occ >= 2:
            planes_pos[ch + 1, :, :] = 1

    # Fill in the Prisoners / Pocket Pieces

    # iterate over all pieces except the king
    for p_type in chess.PIECE_TYPES[:-1]:
        # p_type -1 because p_type starts with 1
        ch = CHANNEL_MAPPING_POS["prisoners"] + p_type - 1

        planes_pos[ch, :, :] = board.pockets[chess.WHITE].count(p_type)
        # the prison for black begins 5 channels later
        planes_pos[ch + 5, :, :] = board.pockets[chess.BLACK].count(p_type)

    # (III) Fill in the promoted pieces
    # iterate over all promoted pieces according to the mask and set the according bit
    ch = CHANNEL_MAPPING_POS["promo"]
    for pos in chess.SquareSet(board.promoted):
        row, col = get_row_col(pos)

        if board.piece_at(pos).color == chess.WHITE:
            planes_pos[ch, row, col] = 1
        else:
            planes_pos[ch + 1, row, col] = 1

    # (III.2) En Passant Square
    # mark the square where an en-passant capture is possible
    ch = CHANNEL_MAPPING_POS["ep_square"]
    if board.ep_square is not None:
        row, col = get_row_col(board.ep_square)
        planes_pos[ch, row, col] = 1

    # (IV) Constant Value Inputs
    # (IV.1) Color
    if board_turn == chess.WHITE:
        planes_const[CHANNEL_MAPPING_CONST["color"], :, :] = 1
    # otherwise the mat will remain zero

    # (IV.2) Total Move Count
    planes_const[CHANNEL_MAPPING_CONST["total_mv_cnt"], :, :] = board.fullmove_number
    # alternatively, you could use the half-moves-counter: len(board.move_stack)

    # (IV.3) Castling Rights
    ch = CHANNEL_MAPPING_CONST["castling"]

    # WHITE
    # check for King Side Castling
    if bool(board.castling_rights & chess.BB_H1) is True:
        # White can castle with the h1 rook
        planes_const[ch, :, :] = 1
    # check for Queen Side Castling
    if bool(board.castling_rights & chess.BB_A1) is True:
        planes_const[ch + 1, :, :] = 1

    # BLACK
    # check for King Side Castling
    if bool(board.castling_rights & chess.BB_H8) is True:
        # White can castle with the h1 rook
        planes_const[ch + 2, :, :] = 1
    # check for Queen Side Castling
    if bool(board.castling_rights & chess.BB_A8) is True:
        planes_const[ch + 3, :, :] = 1

    # (IV.4) No Progress Count
    # define a no 'progress' counter
    # it gets incremented by 1 each move
    # however, whenever a piece gets dropped, a piece is captured or a pawn is moved, it is reset to 0
    # halfmove_clock is an official metric in fen notation
    #  -> see: https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
    no_progress_cnt = board.halfmove_clock

    # check how often the position has already occurred in the game
    planes_const[CHANNEL_MAPPING_CONST["no_progress_cnt"], :, :] = no_progress_cnt

    # (V) Merge the Matrix-Stack
    planes = np.concatenate((planes_pos, planes_const), axis=0)

    # revert the board if the players turn was black
    # ! DO NOT DELETE OR UNCOMMENT THIS BLOCK BECAUSE THE PARAMETER board IS CHANGED IN PLACE !
    if board_turn == chess.BLACK:
        board = board.mirror()

    if normalize is True:
        planes *= MATRIX_NORMALIZER
        # planes = normalize_input_planes(planes)

    # return the plane representation of the given board
    return planes


def planes_to_board(planes, normalized_input=False):
    """
    Converts a board in plane representation to the python chess board representation
    see get_planes_of_board() for input encoding description

    :param planes: Input plane representation
    :param normalized_input: True if the input has been normalized to range[0., 1.]
    :return: python chess board object
    """

    # setup new initial board
    board = CrazyhouseBoard()
    board.clear_board()

    # extract the maps for the board position
    mat_pos = planes[:NB_CHANNELS_POS]
    # extract the last maps which for the constant values
    mat_const = planes[-NB_CHANNELS_CONST:]
    # iterate over all piece types
    for idx, piece in enumerate(PIECES):
        # iterate over all fields and set the current piece type
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                # check if there's a piece at the current position
                if mat_pos[idx, row, col] == 1:
                    # check if the piece was promoted
                    promoted = False
                    ch = CHANNEL_MAPPING_POS["promo"]
                    if mat_pos[ch, row, col] == 1 or mat_pos[ch + 1, row, col] == 1:
                        promoted = True

                    board.set_piece_at(
                        square=get_board_position_index(row, col),
                        piece=chess.Piece.from_symbol(piece),
                        promoted=promoted,
                    )

    # (I) Fill in the Repetition Data
    # check how often the position has already occurred in the game
    # TODO: Find a way to set this on the board state
    # -> apparently this isn't possible because it's also not available in the board uci representation

    # ch = channel_mapping['repetitions']

    # Fill in the Prisoners / Pocket Pieces

    # iterate over all pieces except the king
    for p_type in chess.PIECE_TYPES[:-1]:
        # p_type -1 because p_type starts with 1
        ch = CHANNEL_MAPPING_POS["prisoners"] + p_type - 1

        # the full board is filled with the same value
        # it's sufficient to take only the first value
        nb_prisoners = mat_pos[ch, 0, 0]

        # add prisoners for the current player
        # the whole board is set with the same entry, we can just take the first one
        if normalized_input is True:
            nb_prisoners *= MAX_NB_PRISONERS
            nb_prisoners = int(round(nb_prisoners))

        for i in range(nb_prisoners):
            board.pockets[chess.WHITE].add(p_type)

        # add prisoners for the opponent
        nb_prisoners = mat_pos[ch + 5, 0, 0]
        if normalized_input is True:
            nb_prisoners *= MAX_NB_PRISONERS
            nb_prisoners = int(round(nb_prisoners))

        for i in range(nb_prisoners):
            board.pockets[chess.BLACK].add(p_type)

    # (I.5) En Passant Square
    # mark the square where an en-passant capture is possible
    ch = CHANNEL_MAPPING_POS["ep_square"]
    ep_square = np.argmax(mat_pos[ch])
    if ep_square != 0:
        # if no entry 'one' exists, index 0 will be returned
        board.ep_square = ep_square

    # (II) Constant Value Inputs
    # (II.1) Total Move Count
    ch = CHANNEL_MAPPING_CONST["total_mv_cnt"]
    total_mv_cnt = mat_const[ch, 0, 0]

    if normalized_input is True:
        total_mv_cnt *= MAX_NB_MOVES
        total_mv_cnt = int(round(total_mv_cnt))

    board.fullmove_number = total_mv_cnt

    # (II.2) Castling Rights
    ch = CHANNEL_MAPPING_CONST["castling"]

    # reset the castling_rights for initialization
    board.castling_rights = chess.BB_VOID

    # WHITE
    # check for King Side Castling
    # White can castle with the h1 rook

    # add castling option by applying logical-OR operation
    if mat_const[ch, 0, 0] == 1:
        board.castling_rights |= chess.BB_H1
    # check for Queen Side Castling
    if mat_const[ch + 1, 0, 0] == 1:
        board.castling_rights |= chess.BB_A1

    # BLACK
    # check for King Side Castling
    if mat_const[ch + 2, 0, 0] == 1:
        board.castling_rights |= chess.BB_H8
    # check for Queen Side Castling
    if mat_const[ch + 3, 0, 0] == 1:
        board.castling_rights |= chess.BB_A8

    # (II.3) No Progress Count
    ch = CHANNEL_MAPPING_CONST["no_progress_cnt"]
    no_progress_cnt = mat_const[ch, 0, 0]
    if normalized_input is True:
        no_progress_cnt *= MAX_NB_NO_PROGRESS
        no_progress_cnt = int(round(no_progress_cnt))

    board.halfmove_clock = no_progress_cnt

    # (II.4) Color
    ch = CHANNEL_MAPPING_CONST["color"]

    if mat_const[ch, 0, 0] == 1:
        board.board_turn = chess.WHITE
    else:
        board = board.mirror()
        board.board_turn = chess.BLACK

    return board
