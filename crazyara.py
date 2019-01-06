"""
@file: crazyara.py
Created on 18.10.18
@project: crazy_ara_refactor
@author: queensgambit

Universal chess interface (CUI) communication protocol interface for the engine.
The protocol was published by Stefan-Meyer Kahlen (ShredderChess) and can be accessed at:
http://wbec-ridderkerk.nl/html/UCIProtocol.html
"""

from __future__ import print_function
import sys
import chess.pgn

import traceback
import collections
import numpy as np

# import the Colorer to have a nicer logging printout
from DeepCrazyhouse.src.runtime.ColorLogger import enable_color_logging

enable_color_logging()

# Constants
MIN_SEARCH_TIME_MS = 100
MAX_SEARCH_TIME_MS = 10e10
INC_FACTOR = 7
INC_DIV = 8
MIN_MOVES_LEFT = 10
MAX_BAD_POS_VALUE = -0.10  # When pos eval [-1.0 to 1.0] is equal or worst than this then extend time
MOVES_LEFT_INCREMENT = 10  # Used to reduce the movetime in the opening

# this is the assumed "maximum" blitz game length for calculating a constant movetime
# after 80% of this game length a new time management starts which is based on movetime left
BLITZ_GAME_LENGTH = 50
# use less time in the opening defined by "max_move_num_to_reduce_movetime" by using a portion of the constant move time
MV_TIME_OPENING_PORTION = 0.7
# this variable is intended to increase the variance in the moves played by using a small different amount of time each
# move
RANDOM_MV_TIME_PORTION = 0.1

# enable this variable if you want to see debug messages in certain environments, like the lichess.org api
ENABLE_LICHESS_DEBUG_MSG = False

client = {"name": "CrazyAra", "version": "0.3.1", "authors": "Johannes Czech, Moritz Willig, Alena Beyer"}


INTRO_PART1 = (
    ""
    "                                  _                                           \n"
    "                   _..           /   ._   _.  _        /\   ._   _.           \n"
    "                 .' _ `\         \_  |   (_|  /_  \/  /--\  |   (_|           \n"
    "                /  /e)-,\                         /                           \n"
    "               /  |  ,_ |                    __    __    __    __             \n"
    "              /   '-(-.)/          bw     8 /__////__////__////__////         \n"
    "            .'--.   \  `                 7 ////__////__////__////__/          \n"
    "           /    `\   |                  6 /__////__////__////__////           \n"
    "         /`       |  / /`\.-.          5 ////__////__////__////__/            \n"
    "       .'        ;  /  \_/__/         4 /__////__////__////__////             \n"
    "     .'`-'_     /_.'))).-` \         3 ////__////__////__////__/              \n"
    "    / -'_.'---;`'-))).-'`\_/        2 /__////__////__////__////               \n"
)
INTRO_PART2 = (
    ""
    "   (__.'/   /` .'`                 1 ////__////__////__////__/                \n"
    "    (_.'/ /` /`                       a  b  c  d  e  f  g  h                  \n"
    "      _|.' /`                                                                 \n"
    "jgs.-` __.'|  Developers: Johannes Czech, Moritz Willig, Alena Beyer          \n"
    "    .-'||  |  Source-Code: QueensGambit/CrazyAra (GPLv3-License)              \n"
    "       \_`/   Inspiration: A0-paper by Silver, Hubert, Schrittwieser et al.  \n"
    "              ASCII-Art: Joan G. Stark, Chappell, Burton                      \n"
)

log_file_path = "CrazyAra-log.txt"
score_file_path = "score-log.txt"

try:
    log_file = open(log_file_path, "w")
except:
    log_file = None
    # print out the error message
    traceback_text = traceback.format_exc()
    print("info string An error occured while trying to open the log_file %s" % log_file_path)
    print(traceback_text)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def print_if_debug(string):
    if ENABLE_LICHESS_DEBUG_MSG is True:
        eprint("[debug] " + string)


def log_print(text: str):
    print(text)
    print_if_debug(text)
    if log_file:
        log_file.write("< %s\n" % text)
        log_file.flush()


def write_score_to_file(score: str):
    # score_file = open(score_file_path, 'w')

    with open(score_file_path, "w") as f:
        f.seek(0)
        f.write(score)
        f.truncate()


def log(text: str):
    if log_file:
        log_file.write("> %s\n" % text)
        log_file.flush()


eprint(INTRO_PART1, end="")
eprint(INTRO_PART2, end="")

# GLOBAL VARIABLES
mcts_agent = None
rawnet_agent = None
gamestate = None
setup_done = False
bestmove_value = None
constant_move_time = None
engine_played_move = 0
score = None

# SETTINGS
s = {
    "UCI_Variant": "crazyhouse",
    # set the context in which the neural networks calculation will be done
    # choose 'gpu' using the settings if there is one available
    "context": "cpu",
    "use_raw_network": False,
    "threads": 16,
    "batch_size": 8,
    "neural_net_services": 2,
    "playouts_empty_pockets": 8192,
    "playouts_filled_pockets": 8192,
    "centi_cpuct": 250,
    "centi_dirichlet_epsilon": 25,
    "centi_dirichlet_alpha": 20,
    "max_search_depth": 40,
    "centi_temperature": 7,
    "temperature_moves": 0,
    "opening_guard_moves": 7,
    "centi_clip_quantil": 0,
    "virtual_loss": 3,
    "centi_q_value_weight": 70,
    "threshold_time_for_raw_net_ms": 100,
    "move_overhead_ms": 300,
    "moves_left": 40,
    "extend_time_on_bad_position": True,
    "max_move_num_to_reduce_movetime": 4,
    "check_mate_in_one": False,
    "use_pruning": True,
    "use_oscillating_cpuct": False,
    "use_time_management": True,
    "verbose": False,
}


def setup_network():
    """
    Load the libraries and the weights of the neural network
    :return:
    """

    global gamestate
    global setup_done
    global rawnet_agent
    global mcts_agent
    global s
    global engine_played_move

    if setup_done is False:
        from DeepCrazyhouse.src.domain.crazyhouse.GameState import GameState
        from DeepCrazyhouse.src.domain.agent.NeuralNetAPI import NeuralNetAPI
        from DeepCrazyhouse.src.domain.agent.player.RawNetAgent import RawNetAgent
        from DeepCrazyhouse.src.domain.agent.player.MCTSAgent import MCTSAgent

        # check for valid parameter setup and do auto-corrections if possible
        param_validity_check()

        nets = []
        for i in range(s["neural_net_services"]):
            nets.append(NeuralNetAPI(ctx=s["context"], batch_size=s["batch_size"]))

        rawnet_agent = RawNetAgent(
            nets[0], temperature=s["centi_temperature"] / 100, temperature_moves=s["temperature_moves"]
        )

        mcts_agent = MCTSAgent(
            nets,
            cpuct=s["centi_cpuct"] / 100,
            playouts_empty_pockets=s["playouts_empty_pockets"],
            playouts_filled_pockets=s["playouts_filled_pockets"],
            max_search_depth=s["max_search_depth"],
            dirichlet_alpha=s["centi_dirichlet_alpha"] / 100,
            q_value_weight=s["centi_q_value_weight"] / 100,
            dirichlet_epsilon=s["centi_dirichlet_epsilon"] / 100,
            virtual_loss=s["virtual_loss"],
            threads=s["threads"],
            temperature=s["centi_temperature"] / 100,
            temperature_moves=s["temperature_moves"],
            verbose=s["verbose"],
            min_movetime=MIN_SEARCH_TIME_MS,
            batch_size=s["batch_size"],
            check_mate_in_one=s["check_mate_in_one"],
            use_pruning=s["use_pruning"],
            use_oscillating_cpuct=s["use_oscillating_cpuct"],
            use_time_management=s["use_time_management"],
            opening_guard_moves=s["opening_guard_moves"],
        )

        gamestate = GameState()

        setup_done = True


def validity_with_threads(optname: str):
    """
    Checks for consistency with the number of threads with the given parameter
    :param optname: Option name
    :return:
    """

    if s[optname] > s["threads"]:
        log_print(
            "info string The given batch_size %d is higher than the number of threads %d. "
            "The maximum legal batch_size is the same as the number of threads (here: %d) "
            % (s[optname], s["threads"], s["threads"])
        )
        s[optname] = s["threads"]
        log_print("info string The batch_size was reduced to %d" % s[optname])

    if s["threads"] % s[optname] != 0:
        log_print(
            "info string You requested an illegal combination of threads %d and batch_size %d."
            " The batch_size must be a divisor of the number of threads" % (s["threads"], s[optname])
        )
        divisor = s["threads"] // s[optname]
        s[optname] = s["threads"] // divisor
        log_print("info string The batch_size was changed to %d" % s[optname])


def param_validity_check():
    """
    Handles some possible issues when giving an illegal batch_size and number of threads combination.
    :return:
    """

    validity_with_threads("batch_size")
    validity_with_threads("neural_net_services")


def perform_action(cmd_list):
    """
    Computes the 'best move' according to the engine and the given settings.
    After the search is done it will print out ' bestmove e2e4' for example on std-out.
    :return: 
    """

    global gamestate
    global mcts_agent
    global rawnet_agent
    global bestmove_value
    global engine_played_move
    global constant_move_time
    global score

    movetime_ms = MIN_SEARCH_TIME_MS
    tc_type = None

    if len(cmd_list) >= 5:
        if cmd_list[1] == "wtime" and cmd_list[3] == "btime":

            wtime = int(cmd_list[2])
            btime = int(cmd_list[4])

            my_inc, winc, binc = 0, 0, 0
            if "winc" in cmd_list:
                winc = int(cmd_list[6])
            if "binc" in cmd_list:
                binc = int(cmd_list[8])

            if gamestate.is_white_to_move():
                my_time = wtime
                my_inc = winc
            else:
                my_time = btime
                my_inc = binc

            if constant_move_time is None:
                constant_move_time = (my_time + BLITZ_GAME_LENGTH * my_inc) / BLITZ_GAME_LENGTH

            # TC with period (traditional) like 40/60 or 40 moves in 60 sec repeating
            if "movestogo" in cmd_list:
                tc_type = "traditional"
                if "winc" in cmd_list and "binc" in cmd_list:
                    moves_left = int(cmd_list[10])
                else:
                    moves_left = int(cmd_list[6])
                # If we are close to the period limit, save extra time to avoid time forfeit
                if moves_left <= 3:
                    moves_left += 1
            else:
                tc_type = "blitz"
                moves_left = s["moves_left"]

            moves_left = adjust_moves_left(moves_left, tc_type, bestmove_value)
            if tc_type == "blitz" and engine_played_move < BLITZ_GAME_LENGTH * 0.8:
                movetime_ms = (
                    constant_move_time + (np.random.rand() - 0.5) * RANDOM_MV_TIME_PORTION * constant_move_time
                )

                if engine_played_move < s["max_move_num_to_reduce_movetime"]:
                    # avoid spending too much time in the opening
                    movetime_ms *= MV_TIME_OPENING_PORTION
            else:
                movetime_ms = max(
                    my_time / moves_left + INC_FACTOR * my_inc // INC_DIV - s["move_overhead_ms"], MIN_SEARCH_TIME_MS
                )

    # movetime in UCI protocol, go movetime x, search exactly x mseconds
    # UCI protocol: http://wbec-ridderkerk.nl/html/UCIProtocol.html
    elif len(cmd_list) == 3 and cmd_list[1] == "movetime":
        movetime_ms = max(int(cmd_list[2]) - s["move_overhead_ms"], MIN_SEARCH_TIME_MS)

    mcts_agent.update_movetime(movetime_ms)
    log_print("info string Time for this move is %dms" % movetime_ms)
    log_print("info string Requested pos: %s" % gamestate)

    # assign search depth
    try:
        # we try to extract the search depth from the cmd list
        depth_idx = cmd_list.index("depth") + 1
        mcts_agent.set_max_search_depth(int(cmd_list[depth_idx]))
        # increase the movetime to maximum to make sure to reach the given depth
        movetime_ms = MAX_SEARCH_TIME_MS
        mcts_agent.update_movetime(movetime_ms)
    except ValueError:
        # the given command wasn't found in the command list
        pass

    # disable noise for short move times
    if movetime_ms < 1000:
        mcts_agent.dirichlet_epsilon = 0.1
    elif movetime_ms < 7000:
        # reduce noise for very short move times
        mcts_agent.dirichlet_epsilon = 0.2

    if s["use_raw_network"] or movetime_ms <= s["threshold_time_for_raw_net_ms"]:
        log_print("info string Using raw network for fast mode...")
        value, selected_move, confidence, _, cp, depth, nodes, time_elapsed_s, nps, pv = rawnet_agent.perform_action(
            gamestate
        )
    else:
        value, selected_move, confidence, _, cp, depth, nodes, time_elapsed_s, nps, pv = mcts_agent.perform_action(
            gamestate
        )

    score = "score cp %d depth %d nodes %d time %d nps %d pv %s" % (cp, depth, nodes, time_elapsed_s, nps, pv)
    if ENABLE_LICHESS_DEBUG_MSG:
        try:
            write_score_to_file(score)
        except Exception:
            pass
    # print out the search information
    log_print("info %s" % score)

    # Save the bestmove value [-1.0 to 1.0] to modify the next movetime
    bestmove_value = float(value)
    engine_played_move += 1

    # apply CrazyAra's selected move the global gamestate
    if gamestate.get_pythonchess_board().is_legal(selected_move):
        # apply the last move CrazyAra played
        _apply_move(selected_move)
    else:
        raise Exception("all_ok is false! - crazyara_last_move")

    log_print("bestmove %s" % selected_move.uci())


def setup_gamestate(cmd_list):
    """
    Prepare the gamestate according to the user's wishes.

    :param cmd_list: Input-command lists arguments
    :return:
    """

    global gamestate
    global mcts_agent

    position_type = cmd_list[1]

    if "moves" in cmd_list:
        # position startpos moves e2e4 g8f6
        if position_type == "startpos":
            mv_list = cmd_list[3:]
        else:
            # position fen rn2N2k/pp5p/3pp1pN/3p4/3q1P2/3P1p2/PP3PPP/RN3RK1/Qrbbpbb b - - 3 27 moves d4f2 f1f2
            mv_list = cmd_list[9:]

        # try to apply opponent last move to the board state

        if len(mv_list) > 0:
            # the move the opponent just played is the last move in the list
            opponent_last_move = chess.Move.from_uci(mv_list[-1])
            if gamestate.get_pythonchess_board().is_legal(opponent_last_move):
                # apply the last move the opponent played
                _apply_move(opponent_last_move)
                mv_compatible = True
            else:
                log_print("info string  all_ok is false! - opponent_last_move %s" % opponent_last_move)
                mv_compatible = False
        else:
            mv_compatible = False

        if not mv_compatible:
            log_print("info string The given last two moves couldn't be applied to the previous board-state.")
            log_print("info string Rebuilding the game from scratch...")

            # create a new game state from scratch
            if position_type == "startpos":
                new_game()
            else:
                fen = " ".join(cmd_list[2:8])
                gamestate.set_fen(fen)

            for move in mv_list:
                _apply_move(chess.Move.from_uci(move))
        else:
            log_print("info string Move Compatible")
    else:
        if position_type == "fen":
            fen = " ".join(cmd_list[2:8])
            gamestate.set_fen(fen)
            mcts_agent.update_tranposition_table((gamestate.get_transposition_key(),))
            # log_print("info string Added %s - count %d" % (gamestate.get_board_fen(),
            #                                               mcts_agent.transposition_table[gamestate.get_transposition_key()]))


def _apply_move(selected_move: chess.Move):
    """
    Applies the given move on the gamestate and updates the transposition table of the environment
    :param selected_move: Move in python chess format
    :return:
    """
    global gamestate
    global mcts_agent

    gamestate.apply_move(selected_move)
    mcts_agent.update_tranposition_table((gamestate.get_transposition_key(),))
    # log_print("info string Added %s - count %d" % (gamestate.get_board_fen(),
    #                                               mcts_agent.transposition_table[
    #                                                   gamestate.get_transposition_key()]))


def new_game():
    global gamestate
    global mcts_agent

    log_print("info string >> New Game")
    gamestate.new_game()
    mcts_agent.transposition_table = collections.Counter()
    mcts_agent.time_buffer_ms = 0
    mcts_agent.dirichlet_epsilon = s["centi_dirichlet_epsilon"] / 100


def set_options(cmd_list):
    """
    Updates the internal options as requested by the use via the uci-protocoll
    An example call could be: "setoption name nb_threads value 1"
    :param cmd_list: List of received of commands
    :return:
    """
    # SETTINGS
    global s

    # make sure there exists enough items in the given command list like "setoption name nb_threads value 1"
    if len(cmd_list) >= 5:
        if cmd_list[1] != "name" or cmd_list[3] != "value":
            log_print("info string The given setoption command wasn't understood")
            log_print('info string An example call could be: "setoption name threads value 4"')
        else:
            option_name = cmd_list[2]

            if option_name not in s:
                log_print("info string The given option %s wasn't found in the settings list" % option_name)
            else:

                if option_name in [
                    "UCI_Variant",
                    "context",
                    "use_raw_network",
                    "extend_time_on_bad_position",
                    "verbose",
                    "check_mate_in_one",
                    "use_pruning",
                    "use_oscillating_cpuct",
                    "use_time_management",
                ]:

                    value = cmd_list[4]
                else:
                    value = int(cmd_list[4])

                if option_name == "use_raw_network":
                    s["use_raw_network"] = True if value == "true" else False
                elif option_name == "extend_time_on_bad_position":
                    s["extend_time_on_bad_position"] = True if value == "true" else False
                elif option_name == "verbose":
                    s["verbose"] = True if value == "true" else False
                elif option_name == "check_mate_in_one":
                    s["check_mate_in_one"] = True if value == "true" else False
                elif option_name == "use_pruning":
                    s["use_pruning"] = True if value == "true" else False
                elif option_name == "use_oscillating_cpuct":
                    s["use_oscillating_cpuct"] = True if value == "true" else False
                elif option_name == "use_time_management":
                    s["use_time_management"] = True if value == "true" else False
                else:
                    # by default all options are treated as integers
                    s[option_name] = value

                    # Guard threads limits
                    if option_name == "threads":
                        s[option_name] = min(4096, max(1, s[option_name]))

                log_print("info string Updated option %s to %s" % (option_name, value))


def adjust_moves_left(moves_left, tc_type, prev_bm_value):
    """
    We can reduce the movetime early in the opening as the NN may be able to handle it well.
    Or when the position is bad we can increase the movetime especially if there are enough time left.
    To increase/decrease the movetime, we decrease/increase the moves_left.
    movetime = time_left/moves_left
    :param moves_left: Moves left for the next period for traditional or look ahead moves for blitz
    :param tc_type: Can be blitz (60+1) or traditional (40/60)
    :param prev_bm_value: The value of the previous bestmove. value is in the range [-1 to 1]
    :return: moves_left
    """
    global engine_played_move

    # Don't spend too much time in the opening, we increase the moves_left
    # so that the movetime is reduced. engine_played_move is the actual moves
    # made by the engine excluding the book moves input from a GUI.
    if engine_played_move < s["max_move_num_to_reduce_movetime"]:
        moves_left += MOVES_LEFT_INCREMENT

    # Increase movetime by reducing the moves left if our prev bestmove value is below 0.0
    elif s["extend_time_on_bad_position"] and prev_bm_value is not None and prev_bm_value <= MAX_BAD_POS_VALUE:
        if tc_type == "blitz":
            # The more the bad position is, the more that we extend the search time
            moves_left -= abs(prev_bm_value) * s["moves_left"]
            moves_left = max(moves_left, MIN_MOVES_LEFT)
        # Else if TC is traditional, we extend with more time if we have more time left
        elif moves_left > 4:
            moves_left = moves_left - moves_left // 8

    return moves_left


def uci_reply():
    log_print("id name %s %s" % (client["name"], client["version"]))
    log_print("id author %s" % client["authors"])
    # tell the GUI all possible options
    log_print("option name UCI_Variant type combo default crazyhouse var crazyhouse")
    log_print("option name context type combo default %s var cpu var gpu" % s["context"])
    log_print("option name use_raw_network type check default %s" % ("false" if not s["use_raw_network"] else "true"))
    log_print("option name threads type spin default %d min 1 max 4096" % s["threads"])
    log_print("option name batch_size type spin default %d min 1 max 4096" % s["batch_size"])
    log_print("option name neural_net_services type spin default %d min 1 max 10" % s["neural_net_services"])
    log_print("option name playouts_empty_pockets type spin default %d min 56 max 8192" % s["playouts_empty_pockets"])
    log_print("option name playouts_filled_pockets type spin default %d min 56 max 8192" % s["playouts_filled_pockets"])
    log_print("option name centi_cpuct type spin default %d min 1 max 500" % s["centi_cpuct"])
    log_print("option name centi_dirichlet_epsilon type spin default %d min 0 max 100" % s["centi_dirichlet_epsilon"])
    log_print("option name centi_dirichlet_alpha type spin default %d min 0 max 100" % s["centi_dirichlet_alpha"])
    log_print("option name max_search_depth type spin default %d min 1 max 100" % s["max_search_depth"])
    log_print("option name centi_temperature type spin default %d min 0 max 100" % s["centi_temperature"])
    log_print("option name temperature_moves type spin default %d min 0 max 99999" % s["temperature_moves"])
    log_print("option name opening_guard_moves type spin default %d min 0 max 99999" % s["opening_guard_moves"])
    log_print("option name centi_clip_quantil type spin default 0 min 0 max 100")
    log_print("option name virtual_loss type spin default 3 min 0 max 10")
    log_print("option name centi_q_value_weight type spin default %d min 0 max 100" % s["centi_q_value_weight"])
    log_print(
        "option name threshold_time_for_raw_net_ms type spin default %d min 1 max 300000"
        % s["threshold_time_for_raw_net_ms"]
    )
    log_print("option name move_overhead_ms type spin default %d min 0 max 60000" % s["move_overhead_ms"])
    log_print("option name moves_left type spin default %d min 10 max 320" % s["moves_left"])
    log_print(
        "option name extend_time_on_bad_position type check default %s"
        % ("false" if not s["extend_time_on_bad_position"] else "true")
    )
    log_print(
        "option name max_move_num_to_reduce_movetime type spin default %d min 0 max 120"
        % s["max_move_num_to_reduce_movetime"]
    )
    log_print(
        "option name check_mate_in_one type check default %s" % ("false" if not s["check_mate_in_one"] else "true")
    )
    log_print("option name use_pruning type check default %s" % ("false" if not s["use_pruning"] else "true"))
    log_print(
        "option name use_oscillating_cpuct type check default %s"
        % ("false" if not s["use_oscillating_cpuct"] else "true")
    )
    log_print(
        "option name use_time_management type check default %s" % ("false" if not s["use_time_management"] else "true")
    )
    log_print("option name verbose type check default %s" % ("false" if not s["verbose"] else "true"))

    # verify that all options have been sent
    log_print("uciok")


# main waiting loop for processing command line inputs
def main():
    global bestmove_value
    global engine_played_move
    global log_file

    while True:
        line = input()
        print_if_debug("waiting ...")
        print_if_debug(line)

        # wait for an std-in input command
        if line:
            # split the line to a list which makes parsing easier
            cmd_list = line.rstrip().split(" ")
            # extract the first command from the list for evaluation
            main_cmd = cmd_list[0]

            # write the given command to the log-file
            log(line)

            try:
                if main_cmd == "uci":
                    uci_reply()
                elif main_cmd == "isready":
                    setup_network()
                    log_print("readyok")
                elif main_cmd == "ucinewgame":
                    bestmove_value = None
                    engine_played_move = 0
                    new_game()
                elif main_cmd == "position":
                    setup_gamestate(cmd_list)
                elif main_cmd == "setoption":
                    set_options(cmd_list)
                elif main_cmd == "go":
                    perform_action(cmd_list)
                elif main_cmd == "quit" or main_cmd == "exit":
                    if log_file:
                        log_file.close()
                    return 0
                else:
                    # give the user a message that the command was ignored
                    print("info string Unknown command: %s" % line)
            except:
                # log the error message to the log-file and exit the script
                traceback_text = traceback.format_exc()
                log_print(traceback_text)
                return -1


if __name__ == "__main__":
    main()
