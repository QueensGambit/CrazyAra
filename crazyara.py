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


# Constants
MIN_SEARCH_TIME_MS = 100
INC_FACTOR = 7
INC_DIV = 8
MIN_MOVES_LEFT = 10
MAX_BAD_POS_VALUE = -0.10  # When pos eval [-1.0 to 1.0] is equal or worst than this then extend time
MOVES_LEFT_INCREMENT = 10  # Used to reduce the movetime in the opening

client = {
    'name': 'CrazyAra',
    'version': '0.2.0',
    'authors': 'Johannes Czech, Moritz Willig, Alena Beyer et al.'
}


INTRO_PART1 = "" \
        "                                  _                                           \n" \
        "                   _..           /   ._   _.  _        /\   ._   _.           \n" \
        "                 .' _ `\         \_  |   (_|  /_  \/  /--\  |   (_|           \n" \
        "                /  /e)-,\                         /                           \n" \
        "               /  |  ,_ |                    __    __    __    __             \n" \
        "              /   '-(-.)/          bw     8 /__////__////__////__////         \n" \
        "            .'--.   \  `                 7 ////__////__////__////__/          \n" \
        "           /    `\   |                  6 /__////__////__////__////           \n" \
        "         /`       |  / /`\.-.          5 ////__////__////__////__/            \n" \
        "       .'        ;  /  \_/__/         4 /__////__////__////__////             \n" \
        "     .'`-'_     /_.'))).-` \         3 ////__////__////__////__/              \n" \
        "    / -'_.'---;`'-))).-'`\_/        2 /__////__////__////__////               \n" \

INTRO_PART2 = "" \
            "   (__.'/   /` .'`                 1 ////__////__////__////__/                \n" \
            "    (_.'/ /` /`                       a  b  c  d  e  f  g  h                  \n" \
            "      _|.' /`                                                                 \n" \
            "jgs.-` __.'|  Developers: Johannes Czech, Moritz Willig, Alena Beyer et al.   \n" \
            "    .-'||  |  Source-Code: QueensGambit/CrazyAra (GPLv3-License)              \n" \
            "       \_`/   Inspiration: A0-paper by Silver, Hubert, Schrittwieser et al.  \n" \
            "              ASCII-Art: Joan G. Stark, Chappell, Burton                      \n"

log_file_path = "CrazyAra-log.txt"

try:
    log_file = open(log_file_path, 'w')
except:
    log_file = None
    # print out the error message
    traceback_text = traceback.format_exc()
    print("info string An error occured while trying to open the log_file %s" % log_file_path)
    print(traceback_text)


def log_print(text: str):
    print(text)
    if log_file:
        log_file.write("< %s\n" % text)
        log_file.flush()


def log(text: str):
    if log_file:
        log_file.write("> %s\n" % text)
        log_file.flush()


print(INTRO_PART1, end="")
print(INTRO_PART2, end="")

# GLOBAL VARIABLES
mcts_agent = None
rawnet_agent = None
gamestate = None
setup_done = False
bestmove_value = None
engine_played_move = 0

# SETTINGS
s = {
    'UCI_Variant': 'crazyhouse',
    # set the context in which the neural networks calculation will be done
    # choose 'gpu' using the settings if there is one available
    "context": 'cpu',
    "use_raw_network": False,
    "threads": 16,
    "batch_size": 8,
    "playouts_empty_pockets": 8192,
    "playouts_filled_pockets": 8192,
    "centi_cpuct": 300,
    "centi_dirichlet_epsilon": 10,
    "centi_dirichlet_alpha": 20,
    "max_search_depth": 40,
    "centi_temperature": 0,
    "centi_clip_quantil": 0,
    "virtual_loss": 3,
    "centi_q_value_weight": 65,
    "threshold_time_for_raw_net_ms": 100,
    "move_overhead_ms": 300,
    "moves_left": 40,
    "extend_time_on_bad_position": True,
    "max_move_num_to_reduce_movetime": 0,
    "check_mate_in_one": False,
    "verbose": False
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

        net = NeuralNetAPI(ctx=s['context'], batch_size=s['batch_size'])
        rawnet_agent = RawNetAgent(net, temperature=s['centi_temperature'], clip_quantil=s['centi_clip_quantil'])

        mcts_agent = MCTSAgent(net, cpuct=s['centi_cpuct'] / 100, playouts_empty_pockets=s['playouts_empty_pockets'],
                               playouts_filled_pockets=s['playouts_filled_pockets'], max_search_depth=s['max_search_depth'],
                               dirichlet_alpha=s['centi_dirichlet_alpha'] / 100, q_value_weight=s['centi_q_value_weight'] / 100,
                               dirichlet_epsilon=s['centi_dirichlet_epsilon'] / 100, virtual_loss=s['virtual_loss'],
                               threads=s['threads'], temperature=s['centi_temperature'] / 100, verbose=s['verbose'],
                               clip_quantil=s['centi_clip_quantil'] / 100, min_movetime=MIN_SEARCH_TIME_MS,
                               batch_size=s['batch_size'], check_mate_in_one=s['check_mate_in_one'])

        gamestate = GameState()

        setup_done = True


def param_validity_check():
    """
    Handles some possible issues when giving an illegal batch_size and number of threads combination.
    :return:
    """
    if s['batch_size'] > s['threads']:
        log_print('info string The given batch_size %d is higher than the number of threads %d. '
              'The maximum legal batch_size is the same as the number of threads (here: %d) '
              % (s['batch_size'], s['threads'], s['threads']))
        s['batch_size'] = s['threads']
        log_print('info string The batch_size was reduced to %d' % s['batch_size'])

    if s['threads'] % s['batch_size'] != 0:
        log_print('info string You requested an illegal combination of threads %d and batch_size %d.'
              ' The batch_size must be a divisor of the number of threads' % (s['threads'], s['batch_size']))
        divisor = s['threads'] // s['batch_size']
        s['batch_size'] = s['threads'] // divisor
        log_print('info string The batch_size was changed to %d' % s['batch_size'])


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

    movetime_ms = MIN_SEARCH_TIME_MS
    tc_type = None

    if len(cmd_list) >= 5:
        if cmd_list[1] == 'wtime' and cmd_list[3] == 'btime':

            wtime = int(cmd_list[2])
            btime = int(cmd_list[4])

            my_inc, winc, binc = 0, 0, 0
            if 'winc' in cmd_list:
                winc = int(cmd_list[6])
            if 'binc' in cmd_list:
                binc = int(cmd_list[8])

            if gamestate.is_white_to_move():
                my_time = wtime
                my_inc = winc
            else:
                my_time = btime
                my_inc = binc

            # TC with period (traditional) like 40/60 or 40 moves in 60 sec repeating
            if 'movestogo' in cmd_list:
                tc_type = 'traditional'
                if 'winc' in cmd_list and 'binc' in cmd_list:
                    moves_left = int(cmd_list[10])
                else:
                    moves_left = int(cmd_list[6])
                # If we are close to the period limit, save extra time to avoid time forfeit
                if moves_left <= 3:
                    moves_left += 1
            else:
                tc_type = 'blitz'
                moves_left = s['moves_left']

            moves_left = adjust_moves_left(moves_left, tc_type, bestmove_value)
            movetime_ms = max(my_time/moves_left + INC_FACTOR*my_inc//INC_DIV - s['move_overhead_ms'], MIN_SEARCH_TIME_MS)

    # movetime in UCI protocol, go movetime x, search exactly x mseconds
    # UCI protocol: http://wbec-ridderkerk.nl/html/UCIProtocol.html
    elif len(cmd_list) == 3 and cmd_list[1] == 'movetime':
        movetime_ms = max(int(cmd_list[2]) - s['move_overhead_ms'], MIN_SEARCH_TIME_MS)

    mcts_agent.update_movetime(movetime_ms)
    log_print('info string Time for this move is %dms' % movetime_ms)

    if s['use_raw_network'] or movetime_ms <= s['threshold_time_for_raw_net_ms']:
        log_print('info string Using raw network for fast mode...')
        value, selected_move, confidence, _ = rawnet_agent.perform_action(gamestate)
    else:
        value, selected_move, confidence, _ = mcts_agent.perform_action(gamestate)

    # Save the bestmove value [-1.0 to 1.0] to modify the next movetime
    bestmove_value = float(value)
    engine_played_move += 1

    log_print('bestmove %s' % selected_move.uci())


def setup_gamestate(cmd_list):
    """
    Prepare the gamestate according to the user's wishes.

    :param cmd_list: Input-command lists arguments
    :return:
    """
    #artificial_max_game_len = 30

    position_type = cmd_list[1]
    if position_type == "startpos":
        gamestate.new_game()
    else:
        fen = " ".join(cmd_list[2:8])
        gamestate.set_fen(fen)

    if 'moves' in cmd_list:
        if position_type == 'startpos':
            mv_list = cmd_list[3:]
        else:
            # position fen rn2N2k/pp5p/3pp1pN/3p4/3q1P2/3P1p2/PP3PPP/RN3RK1/Qrbbpbb b - - 3 27 moves d4f2 f1f2
            mv_list = cmd_list[9:]
        for move in mv_list:
            gamestate.apply_move(chess.Move.from_uci(move))

        #if len(mv_list)//2 > artificial_max_game_len:
        #    log_print('info string Setting fullmove_number to %d' % artificial_max_game_len)
        #    gamestate.get_pythonchess_board().fullmove_number = artificial_max_game_len


def set_options(cmd_list):
    """
    Updates the internal options as requested by the use via the uci-protocoll
    An example call could be: "setoption name nb_threads value 1"
    :param cmd_list: List of received of commands
    :return:
    """
    # SETTINGS
    global s

    if cmd_list[1] == 'name' and cmd_list[3] == 'value':
        option_name = cmd_list[2]

        if option_name not in s:
            raise Exception("The given option %s wasn't found in the settings list" % option_name)

        if option_name in ['UCI_Variant', 'context', 'use_raw_network',
                           'extend_time_on_bad_position', 'verbose', 'check_mate_in_one']:

            value = cmd_list[4]
        else:
            value = int(cmd_list[4])

        if option_name == 'use_raw_network':
            s['use_raw_network'] = True if value == 'true' else False
        elif option_name == 'extend_time_on_bad_position':
            s['extend_time_on_bad_position'] = True if value == 'true' else False
        elif option_name == 'verbose':
            s['verbose'] = True if value == 'true' else False
        elif option_name == 'check_mate_in_one':
            s['check_mate_in_one'] = True if value == 'true' else False
        else:
            # by default all options are treated as integers
            s[option_name] = value

            # Guard threads limits
            if option_name == 'threads':
                s[option_name] = min(4096, max(1, s[option_name]))

        log_print('info string Updated option %s to %s' % (option_name, value))


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
    if engine_played_move < s['max_move_num_to_reduce_movetime']:
        moves_left += MOVES_LEFT_INCREMENT

    # Increase movetime by reducing the moves left if our prev bestmove value is below 0.0
    elif s['extend_time_on_bad_position'] and prev_bm_value is not None and\
                                prev_bm_value <= MAX_BAD_POS_VALUE:
        if tc_type == 'blitz':
            # The more the bad position is, the more that we extend the search time
            moves_left -= abs(prev_bm_value) * s['moves_left']
            moves_left = max(moves_left, MIN_MOVES_LEFT)
        # Else if TC is traditional, we extend with more time if we have more time left
        elif moves_left > 4:
            moves_left = moves_left - moves_left//8

    return moves_left


def uci_reply():
    log_print('id name %s %s' % (client['name'], client['version']))
    log_print('id author %s' % client['authors'])
    # tell the GUI all possible options
    log_print('option name UCI_Variant type combo default crazyhouse var crazyhouse')
    log_print('option name context type combo default cpu var cpu var gpu')
    log_print('option name use_raw_network type check default %s' %\
              ('false' if not s['use_raw_network'] else 'true'))
    log_print('option name threads type spin default %d min 1 max 4096' % s['threads'])
    log_print('option name batch_size type spin default %d min 1 max 4096' % s['batch_size'])    
    log_print('option name playouts_empty_pockets type spin default %d min 56 max 8192' %\
              s['playouts_empty_pockets'])
    log_print('option name playouts_filled_pockets type spin default %d min 56 max 8192' %\
              s['playouts_filled_pockets'])
    log_print('option name centi_cpuct type spin default 100 min 1 max 500')
    log_print('option name centi_dirichlet_epsilon type spin default 10 min 0 max 100')
    log_print('option name centi_dirichlet_alpha type spin default 20 min 0 max 100')
    log_print('option name max_search_depth type spin default 40 min 1 max 100')
    log_print('option name centi_temperature type spin default 0 min 0 max 100')
    log_print('option name centi_clip_quantil type spin default 0 min 0 max 100')
    log_print('option name virtual_loss type spin default 3 min 0 max 10')
    log_print('option name centi_q_value_weight type spin default %d min 0 max 100' % s['centi_q_value_weight'])
    log_print('option name threshold_time_for_raw_net_ms type spin default %d min 1 max 300000' %\
              s['threshold_time_for_raw_net_ms'])
    log_print('option name move_overhead_ms type spin default %d min 0 max 60000' % s['move_overhead_ms'])
    log_print('option name moves_left type spin default %d min 10 max 320' % s['moves_left'])
    log_print('option name extend_time_on_bad_position type check default %s' %\
              ('false' if not s['extend_time_on_bad_position'] else 'true'))
    log_print('option name max_move_num_to_reduce_movetime type spin default %d min 0 max 120' %\
              s['max_move_num_to_reduce_movetime'])
    log_print('option name check_mate_in_one type check default %s' %\
              ('false' if not s['check_mate_in_one'] else 'true'))
    log_print('option name verbose type check default %s' %\
              ('false' if not s['verbose'] else 'true'))

    # verify that all options have been sent
    log_print('uciok')


# main waiting loop for processing command line inputs
def main():
    global bestmove_value
    global engine_played_move
    global log_file

    while True:
        line = input()

        # wait for an std-in input command
        if line:
            # split the line to a list which makes parsing easier
            cmd_list = line.rstrip().split(' ')
            # extract the first command from the list for evaluation
            main_cmd = cmd_list[0]

            # write the given command to the log-file
            log(line)

            try:
                if main_cmd == 'uci':
                    uci_reply()
                elif main_cmd == 'isready':
                    setup_network()
                    log_print('readyok')
                elif main_cmd == 'ucinewgame':
                    bestmove_value = None
                    engine_played_move = 0
                elif main_cmd == "position":
                    setup_gamestate(cmd_list)
                elif main_cmd == "setoption":
                    set_options(cmd_list)
                elif main_cmd == 'go':
                    perform_action(cmd_list)
                elif main_cmd == 'quit' or main_cmd == 'exit':
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
