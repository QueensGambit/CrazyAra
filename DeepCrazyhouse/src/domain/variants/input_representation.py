"""
@file: input_representation
Created on 26.09.19
@project: CrazyAra
@author: queensgambit

Input representation for all available lichess chess variants board states (including  crazyhouse)
which is passed to the neural network
"""

from chess.variant import CrazyhouseBoard
import DeepCrazyhouse.src.domain.variants.classical_chess.v2.input_representation as chess_v2
import DeepCrazyhouse.src.domain.variants.classical_chess.v3.input_representation as chess_v3
from DeepCrazyhouse.src.domain.variants.constants import (
    MODE,
    NB_CHANNELS_TOTAL,
    MODES,
    VERSION,
    MODE_CRAZYHOUSE,
    MODE_LICHESS,
    MODE_CHESS,
    MODE_XIANGQI,
    POCKETS_SIZE_PIECE_TYPE,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    CHANNEL_MAPPING_CONST,
    CHANNEL_MAPPING_POS,
    CHANNEL_MAPPING_VARIANTS,
    MAX_NB_MOVES,
    MAX_NB_NO_PROGRESS,
    MAX_NB_PRISONERS,
    NB_CHANNELS_CONST,
    NB_CHANNELS_POS,
    NB_CHANNELS_VARIANTS,
    NB_LAST_MOVES,
    NB_CHANNELS_HISTORY,
    PIECES,
    chess,
    VARIANT_MAPPING_BOARDS)
from DeepCrazyhouse.src.domain.util import get_board_position_index, get_row_col, np
from DeepCrazyhouse.configs.main_config import main_config


def _fill_position_planes(planes_pos, board, board_occ=0, mode=MODE_CRAZYHOUSE):

    # Fill in the piece positions

    # Iterate over both color starting with WHITE
    for idx, color in enumerate(chess.COLORS):
        # the PIECE_TYPE is an integer list in python-chess
        for piece_type in chess.PIECE_TYPES:
            # define the channel by the piece_type (the input representation uses the same ordering as python-chess)
            # we add an offset for the black pieces
            # note that we subtract 1 because in python chess the PAWN has index 1 and not 0
            channel = (piece_type - 1) + idx * len(chess.PIECE_TYPES)
            # iterate over the piece mask and receive every position square of it
            for pos in board.pieces(piece_type, color):
                row, col = get_row_col(pos)
                # set the bit at the right position
                planes_pos[channel, row, col] = 1

    # (II) Fill in the Repetition Data
    # a game to test out if everything is working correctly is: https://lichess.org/jkItXBWy#73
    channel = CHANNEL_MAPPING_POS["repetitions"]

    # set how often the position has already occurred in the game (default 0 times)
    # this is used to check for claiming the 3 fold repetition rule
    if board_occ >= 1:
        planes_pos[channel, :, :] = 1
        if board_occ >= 2:
            planes_pos[channel + 1, :, :] = 1

    # Fill in the Prisoners / Pocket Pieces
    if board.uci_variant == "crazyhouse":
        # iterate over all pieces except the king
        for p_type in chess.PIECE_TYPES[:-1]:
            # p_type -1 because p_type starts with 1
            channel = CHANNEL_MAPPING_POS["prisoners"] + p_type - 1

            planes_pos[channel, :, :] = board.pockets[chess.WHITE].count(p_type)
            # the prison for black begins 5 channels later
            planes_pos[channel + 5, :, :] = board.pockets[chess.BLACK].count(p_type)

    # (III) Fill in the promoted pieces
    # iterate over all promoted pieces according to the mask and set the according bit
    if mode == MODE_CRAZYHOUSE or mode == MODE_LICHESS:
        channel = CHANNEL_MAPPING_POS["promo"]
        for pos in chess.SquareSet(board.promoted):
            row, col = get_row_col(pos)

            if board.piece_at(pos).color == chess.WHITE:
                planes_pos[channel, row, col] = 1
            else:
                planes_pos[channel + 1, row, col] = 1

    # (III.2) En Passant Square
    # mark the square where an en-passant capture is possible
    channel = CHANNEL_MAPPING_POS["ep_square"]
    if board.ep_square is not None:
        row, col = get_row_col(board.ep_square)
        planes_pos[channel, row, col] = 1

    return planes_pos


def _fill_constant_planes(planes_const, board, board_turn):

    # (IV) Constant Value Inputs
    # (IV.1) Color
    if board_turn == chess.WHITE:
        planes_const[CHANNEL_MAPPING_CONST["color"], :, :] = 1
    # otherwise the mat will remain zero

    # (IV.2) Total Move Count
    planes_const[CHANNEL_MAPPING_CONST["total_mv_cnt"], :, :] = board.fullmove_number
    # alternatively, you could use the half-moves-counter: len(board.move_stack)

    # (IV.3) Castling Rights
    # has_kingside_castling_rights() and has_queenside_castling_rights() support chess960, too
    channel = CHANNEL_MAPPING_CONST["castling"]

    # WHITE
    # check for King Side Castling
    if board.has_kingside_castling_rights(chess.WHITE):
        planes_const[channel, :, :] = 1
    # check for Queen Side Castling
    if board.has_queenside_castling_rights(chess.WHITE):
        planes_const[channel + 1, :, :] = 1

    # BLACK
    # check for King Side Castling
    if board.has_kingside_castling_rights(chess.BLACK):
        planes_const[channel + 2, :, :] = 1
    # check for Queen Side Castling
    if board.has_queenside_castling_rights(chess.BLACK):
        planes_const[channel + 3, :, :] = 1

    # (IV.4) No Progress Count
    # define a no 'progress' counter
    # it gets incremented by 1 each move
    # however, whenever a piece gets dropped, a piece is captured or a pawn is moved, it is reset to 0
    # halfmove_clock is an official metric in fen notation
    #  -> see: https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
    no_progress_cnt = board.halfmove_clock

    # check how often the position has already occurred in the game
    planes_const[CHANNEL_MAPPING_CONST["no_progress_cnt"], :, :] = no_progress_cnt

    # set the remaining checks (only needed for "3check")
    if board.uci_variant == "3check":
        channel = CHANNEL_MAPPING_CONST["remaining_checks"]
        if board.remaining_checks[chess.WHITE] <= 2:
            planes_const[channel, :, :] = 1
            if board.remaining_checks[chess.WHITE] == 1:
                planes_const[channel+1, :, :] = 1
        if board.remaining_checks[chess.BLACK] <= 2:
            planes_const[channel+2, :, :] = 1
            if board.remaining_checks[chess.BLACK] == 1:
                planes_const[channel+3, :, :] = 1

    return planes_const


def _fill_variants_plane(board, planes_variants):
    # (V) Variants specification

    # set the is960 boolean flag when active
    if board.chess960 is True:
        planes_variants[CHANNEL_MAPPING_VARIANTS["is960"], :, :] = 1

    # set the current active variant as a one-hot encoded entry
    for variant_name in VARIANT_MAPPING_BOARDS:  # exclude "is960"
        if variant_name == board.uci_variant:
            planes_variants[CHANNEL_MAPPING_VARIANTS[variant_name], :, :] = 1
            break


def flip_board(board, mode,) -> bool:
    """
    Decides whether to flip the board based on the side to move.
    If the board is the racing variant, flipping the board is disabled
    :param board: Board object
    ;param mode: Active config mode
    :return: bool
    """
    if mode == MODE_LICHESS and board.uci_variant == "racingkings":
        return False
    return board.turn == chess.BLACK


def board_to_planes(board, board_occ=0, normalize=True, mode=MODE_CRAZYHOUSE, last_moves=None):
    """
    Gets the plane representation of a given board state.
    (No history of past board positions is used.)

    :param board: Board handle (Python-chess object)
    :param board_occ: Sets how often the board state has occurred before (by default 0)
    :param normalize: True if the inputs shall be normalized to the range [0.-1.]
    :param mode: 0 - MODE_CRAZYHOUSE: Crazyhouse only specification.
                 (Visit variants.crazyhouse.input_representation for detailed documentation)
                 1 - MODE_LICHESS: Specification for all supported variants on lichess.org
                 (Visit variants.lichess.input_representation for detailed documentation)
                 2 - MODE_CHESS: Specification for chess only with chess960 support
                 (Visit variants.chess.input_representation for detailed documentation)
    :param last_moves: List of last moves. It is assumed that the most recent move is the first entry !
    :return: planes - the plane representation of the current board state
    """

    # TODO: Remove board.mirror() for black by addressing the according color channel

    if mode == MODE_CHESS and VERSION == 2:
        return chess_v2.board_to_planes(board, normalize, last_moves)
    if mode == MODE_CHESS and VERSION == 3:
        return chess_v3.board_to_planes(board, board_occ, normalize, last_moves)

    # (I) Define the Input Representation for one position
    planes_pos = np.zeros((NB_CHANNELS_POS, BOARD_HEIGHT, BOARD_WIDTH))
    planes_const = np.zeros((NB_CHANNELS_CONST, BOARD_HEIGHT, BOARD_WIDTH))

    if mode == MODE_CRAZYHOUSE:
        # crazyhouse only doesn't contain remaining checks
        planes_variants = False
    elif mode == MODE_LICHESS:
        planes_variants = np.zeros((NB_CHANNELS_VARIANTS, BOARD_HEIGHT, BOARD_WIDTH))
    else:  # mode = MODE_CHESS
        # chess doesn't contain pocket pieces and remaining checks
        planes_variants = np.zeros((NB_CHANNELS_VARIANTS, BOARD_HEIGHT, BOARD_WIDTH))

    # check who's player turn it is and flip the board if it's black turn (except for racing kings)
    mirror_board = flip_board(board, mode)

    if mirror_board:
        board = board.mirror()

    _fill_position_planes(planes_pos, board, board_occ, mode)
    _fill_constant_planes(planes_const, board, board.turn)
    if mode == MODE_LICHESS:
        _fill_variants_plane(board, planes_variants)
    elif mode == MODE_CHESS:
        if board.chess960 is True:
            planes_variants[:, :, :] = 1

    # create the move planes
    planes_moves = np.zeros((NB_CHANNELS_HISTORY, BOARD_HEIGHT, BOARD_WIDTH))
    if last_moves:
         for i, move in enumerate(last_moves):
             if move:
                 from_row, from_col = get_row_col(move.from_square, mirror=mirror_board)
                 to_row, to_col = get_row_col(move.to_square, mirror=mirror_board)
                 planes_moves[i*2, from_row, from_col] = 1
                 planes_moves[i*2+1, to_row, to_col] = 1

    # (VI) Merge the Matrix-Stack
    if mode == MODE_CRAZYHOUSE:
        planes = np.concatenate((planes_pos, planes_const), axis=0)
    else:  # mode = MODE_LICHESS | mode == MODE_CHESS
        planes = np.concatenate((planes_pos, planes_const, planes_variants, planes_moves), axis=0)

    # revert the board if the players turn was black
    # ! DO NOT DELETE OR UNCOMMENT THIS BLOCK BECAUSE THE PARAMETER board IS CHANGED IN PLACE !
    if mirror_board:
        board = board.mirror()

    if normalize is True:
        planes *= MATRIX_NORMALIZER
        # planes = normalize_input_planes(planes)

    # return the plane representation of the given board
    return planes


def planes_to_board(planes, normalized_input=False, mode=MODE_CRAZYHOUSE):
    """
    Converts a board in plane representation to the python chess board representation
    see get_planes_of_board() for input encoding description

    :param planes: Input plane representation
    :param normalized_input: True if the input has been normalized to range[0., 1.]
    :param mode: 0 - MODE_CRAZYHOUSE: Crazyhouse only specification.
                 (Visit variants.crazyhouse.input_representation for detailed documentation)
                 1 - MODE_LICHESS: Specification for all supported variants on lichess.org
                 (Visit variants.lichess.input_representation for detailed documentation)
                 2 - MODE_CHESS: Specification for chess only with chess960 support
                 (Visit variants.chess.input_representation for detailed documentation)
    :return: python chess board object
    """
    if mode not in MODES:
        raise ValueError(f"Given {mode} is not {MODES}.")

    if mode == MODE_CHESS and VERSION == 2:
        return chess_v2.planes_to_board(planes)
    if mode == MODE_CHESS and VERSION == 3:
        return chess_v3.planes_to_board(planes)

    # extract the maps for the board position
    planes_pos = planes[:NB_CHANNELS_POS]
    # extract the last maps which for the constant values
    end_board_idx = NB_CHANNELS_POS+NB_CHANNELS_CONST
    planes_const = planes[NB_CHANNELS_POS:end_board_idx]

    if mode == MODE_LICHESS:
        # extract the variants definition section
        planes_variants = planes[end_board_idx:end_board_idx+NB_CHANNELS_VARIANTS]

        # setup new initial board
        is960 = planes_variants[CHANNEL_MAPPING_VARIANTS["is960"], 0, 0] == 1

        # iterate through all available variants
        board = None
        for variant in VARIANT_MAPPING_BOARDS:  # exclude "is960"
            if planes_variants[CHANNEL_MAPPING_VARIANTS[variant], 0, 0] == 1:
                board = VARIANT_MAPPING_BOARDS[variant](chess960=is960)
                break

        if board is None:
            raise Exception("No chess variant was recognized in your given input planes")
    elif mode == MODE_CRAZYHOUSE:
        board = CrazyhouseBoard()
    else:  # mode = MODE_CHESS:
        # extract the variants definition section
        plane_960 = planes[-1:]
        is960 = plane_960[CHANNEL_MAPPING_VARIANTS["is960"], 0, 0] == 1
        board = chess.Board(chess960=is960)

    # clear the full board (the pieces will be set later)
    board.clear()

    # iterate over all piece types
    for idx, piece in enumerate(PIECES):
        # iterate over all fields and set the current piece type
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                # check if there's a piece at the current position
                if planes_pos[idx, row, col] == 1:
                    # check if the piece was promoted
                    promoted = False
                    if mode == MODE_CRAZYHOUSE or mode == MODE_LICHESS:
                        # promoted pieces are not defined in the chess plane representation
                        channel = CHANNEL_MAPPING_POS["promo"]
                        if planes_pos[channel, row, col] == 1 or planes_pos[channel + 1, row, col] == 1:
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
    if mode == MODE_CRAZYHOUSE or board.uci_variant == "crazyhouse":
        # iterate over all pieces except the king
        for p_type in chess.PIECE_TYPES[:-1]:
            # p_type -1 because p_type starts with 1
            channel = CHANNEL_MAPPING_POS["prisoners"] + p_type - 1

            # the full board is filled with the same value
            # it's sufficient to take only the first value
            nb_prisoners = planes_pos[channel, 0, 0]

            # add prisoners for the current player
            # the whole board is set with the same entry, we can just take the first one
            if normalized_input is True:
                nb_prisoners *= MAX_NB_PRISONERS
                nb_prisoners = int(round(nb_prisoners))

            for _ in range(nb_prisoners):
                board.pockets[chess.WHITE].add(p_type)

            # add prisoners for the opponent
            nb_prisoners = planes_pos[channel + 5, 0, 0]
            if normalized_input is True:
                nb_prisoners *= MAX_NB_PRISONERS
                nb_prisoners = int(round(nb_prisoners))

            for _ in range(nb_prisoners):
                board.pockets[chess.BLACK].add(p_type)

    # (I.5) En Passant Square
    # mark the square where an en-passant capture is possible
    channel = CHANNEL_MAPPING_POS["ep_square"]
    ep_square = np.argmax(planes_pos[channel])
    if ep_square != 0:
        # if no entry 'one' exists, index 0 will be returned
        board.ep_square = ep_square

    # (II) Constant Value Inputs
    # (II.1) Total Move Count
    channel = CHANNEL_MAPPING_CONST["total_mv_cnt"]
    total_mv_cnt = planes_const[channel, 0, 0]

    if normalized_input is True:
        total_mv_cnt *= MAX_NB_MOVES
        total_mv_cnt = int(round(total_mv_cnt))

    board.fullmove_number = total_mv_cnt

    # (II.2) Castling Rights
    channel = CHANNEL_MAPPING_CONST["castling"]

    # reset the castling_rights for initialization
    # set to 0, previously called chess.BB_VOID for chess version of 0.23.X and chess.BB_EMPTY for versions > 0.27.X
    board.castling_rights = 0

    # WHITE
    # check for King Side Castling
    # White can castle with the h1 rook

    # add castling option by modifying the castling fen
    castling_fen = ""
    # check for King Side Castling
    if planes_const[channel, 0, 0] == 1:
        castling_fen += "K"
        board.castling_rights |= chess.BB_H1
    # check for Queen Side Castling
    if planes_const[channel + 1, 0, 0] == 1:
        castling_fen += "Q"
        board.castling_rights |= chess.BB_A1

    # BLACK
    # check for King Side Castling
    if planes_const[channel + 2, 0, 0] == 1:
        castling_fen += "k"
        board.castling_rights |= chess.BB_H8
    # check for Queen Side Castling
    if planes_const[channel + 3, 0, 0] == 1:
        castling_fen += "q"
        board.castling_rights |= chess.BB_A8

    # configure the castling rights
    if castling_fen:
        board.set_castling_fen(castling_fen)

    # (II.3) No Progress Count
    channel = CHANNEL_MAPPING_CONST["no_progress_cnt"]
    no_progress_cnt = planes_const[channel, 0, 0]
    if normalized_input is True:
        no_progress_cnt *= MAX_NB_NO_PROGRESS
        no_progress_cnt = int(round(no_progress_cnt))

    board.halfmove_clock = no_progress_cnt

    # set the number of remaining checks (only needed for 3check) and might be mirrored later
    if mode == MODE_LICHESS:
        channel = CHANNEL_MAPPING_CONST["remaining_checks"]
        if planes_const[channel, 0, 0] == 1:
            board.remaining_checks[chess.WHITE] -= 1
        if planes_const[channel+1, 0, 0] == 1:
            board.remaining_checks[chess.WHITE] -= 1
        if planes_const[channel+2, 0, 0] == 1:
            board.remaining_checks[chess.BLACK] -= 1
        if planes_const[channel+3, 0, 0] == 1:
            board.remaining_checks[chess.BLACK] -= 1

    # (II.4) Color
    channel = CHANNEL_MAPPING_CONST["color"]

    if planes_const[channel, 0, 0] == 1:
        board.board_turn = chess.WHITE
    else:
        board = board.mirror()
        board.board_turn = chess.BLACK

    return board


def normalize_input_planes(x):
    """
    Normalizes input planes to range [0,1]. Works in place / meaning the input parameter x is manipulated
    :param x: Input planes representation
    :return: The normalized planes
    """

    # convert the input planes to float32 assuming that the datatype is int
    if x.dtype != np.float32:
        x = x.astype(np.float32)

    if MODE == MODE_CHESS and VERSION == 2:
        return chess_v2.normalize_input_planes(x)
    if MODE == MODE_CHESS and VERSION == 3:
        return chess_v3.normalize_input_planes(x)

    mat_pos = x[:NB_CHANNELS_POS, :, :]
    mat_const = x[NB_CHANNELS_POS:, :, :]

    # iterate over all pieces except the king, (because the king can't be in a pocket)
    if MODE == MODE_CRAZYHOUSE or MODE == MODE_LICHESS:
        for p_type in chess.PIECE_TYPES[:-1]:
            # p_type -1 because p_type starts with 1
            channel = CHANNEL_MAPPING_POS["prisoners"] + p_type - 1
            mat_pos[channel, :, :] /= MAX_NB_PRISONERS
            # the prison for black begins 5 channels later
            mat_pos[channel + POCKETS_SIZE_PIECE_TYPE, :, :] /= MAX_NB_PRISONERS
    # xiangqi has 7 piece types (king/general is excluded as prisoner)
    elif MODE == MODE_XIANGQI:
        for p_type in range(6):
            channel = CHANNEL_MAPPING_POS["prisoners"] + p_type
            mat_pos[channel, :, :] /= MAX_NB_PRISONERS
            # the prison for opponent begins 6 channels later
            mat_pos[channel + POCKETS_SIZE_PIECE_TYPE, :, :] /= MAX_NB_PRISONERS

    # Total Move Count
    # 500 was set as the max number of total moves
    mat_const[CHANNEL_MAPPING_CONST["total_mv_cnt"], :, :] /= MAX_NB_MOVES
    # No progress count
    # after 40 moves of no progress the 40 moves rule for draw applies
    if MODE != MODE_XIANGQI:
        mat_const[CHANNEL_MAPPING_CONST["no_progress_cnt"], :, :] /= MAX_NB_NO_PROGRESS

    return x


def get_planes_statistics(board: chess.Board, normalize: bool, last_moves_uci: list, board_occ=0):
    """
    Returns a dictionary for statistics of the plane which can be used for Unit-Testing.
    e.g get_planes_statistics(board, False, last_moves=[chess.Move.from_uci("d7d5")])
    :param board: Chess board object
    :param normalize: Decides if the planes should be normalized
    :param last_moves_uci: Last moves in UCI notation. Chronoligically ordered, meaning first move is first entry and
    most recent move is last entry.
    :param board_occ: Gives information on how often this position has occured already.
    """
    last_moves = []
    for uci_move in last_moves_uci[::-1]:
        last_moves.append(chess.Move.from_uci(uci_move))
    if len(last_moves) < NB_LAST_MOVES:
        for _ in range(NB_LAST_MOVES-len(last_moves)):
            last_moves.append(None)
    last_moves = last_moves[:NB_LAST_MOVES]

    planes = board_to_planes(board, board_occ=board_occ, normalize=normalize, mode=main_config['mode'], last_moves=last_moves)

    planes = planes.flatten()
    stats = {}
    stats['sum'] = planes.sum()
    stats['argMax'] = planes.argmax()
    stats['maxNum'] = planes.max()
    stats['key'] = 0
    for i in range(len(planes)):
        stats['key'] += i * planes[i]
    return stats

# use a constant matrix for normalization to allow broad cast operations
# in policy version 2, the king promotion moves were added to support antichess, this deprecates older nets
MATRIX_NORMALIZER = normalize_input_planes(np.ones((NB_CHANNELS_TOTAL, BOARD_HEIGHT, BOARD_WIDTH)))
