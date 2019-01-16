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
import collections
import sys
import traceback
import chess.pgn
import numpy as np
from DeepCrazyhouse.src.runtime.ColorLogger import enable_color_logging


class CrazyAra:
    def __init__(self):
        enable_color_logging()
        # Constants
        self.min_search_time = 100
        self.max_search_time = 10e10
        self.inc_factor = 7
        self.inc_div = 8
        self.min_moves_left = self.moves_left_increment = 10  # Used to reduce the movetime in the opening
        self.max_bad_pos_value = -0.10  # When pos eval [-1.0 to 1.0] is equal or worst than this then extend time
        # this is the assumed "maximum" blitz game length for calculating a constant movetime
        # after 80% of this game length a new time management starts which is based on movetime left
        self.blitz_game_length = 50
        # use less time in the opening defined by "max_move_num_to_reduce_movetime" by using the constant move time
        self.mv_time_opening_portion = 0.7
        # this variable is intended to increase variance in the moves played by using a different amount of time each
        # move
        self.random_mv_time_portion = 0.1
        # enable this variable if you want to see debug messages in certain environments, like the lichess.org api
        self.enable_lichess_debug_msg = self.setup_done = False
        self.client = {"name": "CrazyAra", "version": "0.3.1", "authors": "Johannes Czech, Moritz Willig, Alena Beyer"}
        self.MCTS_agent = self.rawnet_agent = self.gamestate = self.bestmove_value = self.move_time = self.score = None
        self.engine_played_move = 0
        self.log_file_path = "CrazyAra-log.txt"
        self.score_file_path = "score-log.txt"
        self.settings = {
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
        try:
            self.log_file = open(self.log_file_path, "w")
        except IOError:
            self.log_file = None
            # print out the error message
            print("info string An error occurred while trying to open the self.log_file %s" % self.log_file_path)
            traceback.print_exc()

        self.intro = """
                                              _                                           
                               _..           /   ._   _.  _        /\   ._   _.           
                             .' _ `\         \_  |   (_|  /_  \/  /--\  |   (_|           
                            /  /e)-,\                         /                           
                           /  |  ,_ |                    __    __    __    __             
                          /   '-(-.)/          bw     8 /__////__////__////__////         
                        .'--.   \  `                 7 ////__////__////__////__/          
                       /    `\   |                  6 /__////__////__////__////           
                     /`       |  / /`\.-.          5 ////__////__////__////__/            
                   .'        ;  /  \_/__/         4 /__////__////__////__////             
                 .'`-'_     /_.'))).-` \         3 ////__////__////__////__/              
                / -'_.'---;`'-))).-'`\_/        2 /__////__////__////__////        
               (__.'/   /` .'`                 1 ////__////__////__////__/                
                (_.'/ /` /`                       a  b  c  d  e  f  g  h                  
                  _|.' /`                                                                 
            jgs.-` __.'|  Developers: Johannes Czech, Moritz Willig, Alena Beyer          
                .-'||  |  Source-Code: QueensGambit/CrazyAra (GPLv3-License)              
                   \_`/   Inspiration: A0-paper by Silver, Hubert, Schrittwieser et al.  
                          ASCII-Art: Joan G. Stark, Chappell, Burton                      """

    def eprint(*args, **kwargs):
        print(*args, file=sys.stderr, **kwargs)

    def print_if_debug(self, string):
        if self.enable_lichess_debug_msg is True:
            self.eprint("[debug] " + string)

    def log_print(self, text: str):
        print(text)
        self.print_if_debug(text)
        if self.log_file:
            self.log_file.write("< %s\n" % text)
            self.log_file.flush()

    def write_score_to_file(self, score: str):
        # score_file = open(score_file_path, 'w')

        with open(self.score_file_path, "w") as selected_file:
            selected_file.seek(0)
            selected_file.write(score)
            selected_file.truncate()

    def log(self, text: str):
        if self.log_file:
            self.log_file.write("> %s\n" % text)
            self.log_file.flush()

    def setup_network(self):
        """
        Load the libraries and the weights of the neural network
        :return:
        """
        if self.setup_done is False:
            from DeepCrazyhouse.src.domain.crazyhouse.game_state import GameState
            from DeepCrazyhouse.src.domain.agent.neural_net_api import NeuralNetAPI
            from DeepCrazyhouse.src.domain.agent.player.raw_net_agent import RawNetAgent
            from DeepCrazyhouse.src.domain.agent.player.MCTSAgent import MCTSAgent

            # check for valid parameter setup and do auto-corrections if possible
            self.param_validity_check()

            nets = []
            for _ in range(self.settings["neural_net_services"]):
                nets.append(NeuralNetAPI(ctx=self.settings["context"], batch_size=self.settings["batch_size"]))

            self.rawnet_agent = RawNetAgent(
                nets[0],
                temperature=self.settings["centi_temperature"] / 100,
                temperature_moves=self.settings["temperature_moves"],
            )

            self.MCTS_agent = MCTSAgent(
                nets,
                cpuct=self.settings["centi_cpuct"] / 100,
                playouts_empty_pockets=self.settings["playouts_empty_pockets"],
                playouts_filled_pockets=self.settings["playouts_filled_pockets"],
                max_search_depth=self.settings["max_search_depth"],
                dirichlet_alpha=self.settings["centi_dirichlet_alpha"] / 100,
                q_value_weight=self.settings["centi_q_value_weight"] / 100,
                dirichlet_epsilon=self.settings["centi_dirichlet_epsilon"] / 100,
                virtual_loss=self.settings["virtual_loss"],
                threads=self.settings["threads"],
                temperature=self.settings["centi_temperature"] / 100,
                temperature_moves=self.settings["temperature_moves"],
                verbose=self.settings["verbose"],
                min_movetime=self.min_search_time,
                batch_size=self.settings["batch_size"],
                check_mate_in_one=self.settings["check_mate_in_one"],
                use_pruning=self.settings["use_pruning"],
                use_oscillating_cpuct=self.settings["use_oscillating_cpuct"],
                use_time_management=self.settings["use_time_management"],
                opening_guard_moves=self.settings["opening_guard_moves"],
            )

            self.gamestate = GameState()

            self.setup_done = True

    def validity_with_threads(self, optname: str):
        """
        Checks for consistency with the number of threads with the given parameter
        :param optname: Option name
        :return:
        """

        if self.settings[optname] > self.settings["threads"]:
            self.log_print(
                "info string The given batch_size %d is higher than the number of threads %d. "
                "The maximum legal batch_size is the same as the number of threads (here: %d) "
                % (self.settings[optname], self.settings["threads"], self.settings["threads"])
            )
            self.settings[optname] = self.settings["threads"]
            self.log_print("info string The batch_size was reduced to %d" % self.settings[optname])

        if self.settings["threads"] % self.settings[optname] != 0:
            self.log_print(
                "info string You requested an illegal combination of threads %d and batch_size %d."
                " The batch_size must be a divisor of the number of threads"
                % (self.settings["threads"], self.settings[optname])
            )
            divisor = self.settings["threads"] // self.settings[optname]
            self.settings[optname] = self.settings["threads"] // divisor
            self.log_print("info string The batch_size was changed to %d" % self.settings[optname])

    def param_validity_check(self):
        """
        Handles some possible issues when giving an illegal batch_size and number of threads combination.
        :return:
        """

        self.validity_with_threads("batch_size")
        self.validity_with_threads("neural_net_services")

    def perform_action(self, cmd_list):
        """
        Computes the 'best move' according to the engine and the given settings.
        After the search is done it will print out ' bestmove e2e4' for example on std-out.
        :return:
        """

        movetime_ms = self.min_search_time

        if len(cmd_list) >= 5:
            if cmd_list[1] == "wtime" and cmd_list[3] == "btime":

                wtime = int(cmd_list[2])
                btime = int(cmd_list[4])

                my_inc, winc, binc = 0, 0, 0
                if "winc" in cmd_list:
                    winc = int(cmd_list[6])
                if "binc" in cmd_list:
                    binc = int(cmd_list[8])

                if self.gamestate.is_white_to_move():
                    my_time = wtime
                    my_inc = winc
                else:
                    my_time = btime
                    my_inc = binc

                if self.move_time is None:
                    self.move_time = (my_time + self.blitz_game_length * my_inc) / self.blitz_game_length

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
                    moves_left = self.settings["moves_left"]

                moves_left = self.adjust_moves_left(moves_left, tc_type, self.bestmove_value)
                if tc_type == "blitz" and self.engine_played_move < self.blitz_game_length * 0.8:
                    movetime_ms = (
                        self.move_time + (np.random.rand() - 0.5) * self.random_mv_time_portion * self.move_time
                    )

                    if self.engine_played_move < self.settings["max_move_num_to_reduce_movetime"]:
                        # avoid spending too much time in the opening
                        movetime_ms *= self.mv_time_opening_portion
                else:
                    movetime_ms = max(
                        my_time / moves_left
                        + self.inc_factor * my_inc // self.inc_div
                        - self.settings["move_overhead_ms"],
                        self.min_search_time,
                    )

        # movetime in UCI protocol, go movetime x, search exactly x mseconds
        # UCI protocol: http://wbec-ridderkerk.nl/html/UCIProtocol.html
        elif len(cmd_list) == 3 and cmd_list[1] == "movetime":
            movetime_ms = max(int(cmd_list[2]) - self.settings["move_overhead_ms"], self.min_search_time)

        self.MCTS_agent.update_movetime(movetime_ms)
        self.log_print("info string Time for this move is %dms" % movetime_ms)
        self.log_print("info string Requested pos: %s" % self.gamestate)

        # assign search depth
        try:
            # we try to extract the search depth from the cmd list
            depth_idx = cmd_list.index("depth") + 1
            self.MCTS_agent.set_max_search_depth(int(cmd_list[depth_idx]))
            # increase the movetime to maximum to make sure to reach the given depth
            movetime_ms = self.max_search_time
            self.MCTS_agent.update_movetime(movetime_ms)
        except ValueError:
            # the given command wasn't found in the command list
            pass

        # disable noise for short move times
        if movetime_ms < 1000:
            self.MCTS_agent.dirichlet_epsilon = 0.1
        elif movetime_ms < 7000:
            # reduce noise for very short move times
            self.MCTS_agent.dirichlet_epsilon = 0.2

        if self.settings["use_raw_network"] or movetime_ms <= self.settings["threshold_time_for_raw_net_ms"]:
            self.log_print("info string Using raw network for fast mode...")
            value, selected_move, _, _, centipawn, depth, nodes, time_elapsed_s, nps, pv = self.rawnet_agent.perform_action(
                self.gamestate
            )
        else:
            value, selected_move, _, _, centipawn, depth, nodes, time_elapsed_s, nps, pv = self.MCTS_agent.perform_action(
                self.gamestate
            )

        self.score = "score centipawn %d depth %d nodes %d time %d nps %d pv %s" % (
            centipawn,
            depth,
            nodes,
            time_elapsed_s,
            nps,
            pv,
        )
        if self.enable_lichess_debug_msg:
            try:
                self.write_score_to_file(self.score)
            except IOError:
                traceback.print_exc()
        # print out the search information
        self.log_print("info %s" % self.score)

        # Save the bestmove value [-1.0 to 1.0] to modify the next movetime
        self.bestmove_value = float(value)
        self.engine_played_move += 1

        # apply CrazyAra's selected move the global gamestate
        if self.gamestate.get_pythonchess_board().is_legal(selected_move):
            # apply the last move CrazyAra played
            self._apply_move(selected_move)
        else:
            raise Exception("all_ok is false! - crazyara_last_move")

        self.log_print("bestmove %s" % selected_move.uci())

    def setup_gamestate(self, cmd_list):
        """
        Prepare the gamestate according to the user's wishes.

        :param cmd_list: Input-command lists arguments
        :return:
        """
        position_type = cmd_list[1]

        if "moves" in cmd_list:
            # position startpos moves e2e4 g8f6
            if position_type == "startpos":
                mv_list = cmd_list[3:]
            else:
                # position fen rn2N2k/pp5p/3pp1pN/3p4/3q1P2/3P1p2/PP3PPP/RN3RK1/Qrbbpbb b - - 3 27 moves d4f2 f1f2
                mv_list = cmd_list[9:]

            # try to apply opponent last move to the board state

            if mv_list:
                # the move the opponent just played is the last move in the list
                opponent_last_move = chess.Move.from_uci(mv_list[-1])
                if self.gamestate.get_pythonchess_board().is_legal(opponent_last_move):
                    # apply the last move the opponent played
                    self._apply_move(opponent_last_move)
                    mv_compatible = True
                else:
                    self.log_print("info string  all_ok is false! - opponent_last_move %s" % opponent_last_move)
                    mv_compatible = False
            else:
                mv_compatible = False

            if not mv_compatible:
                self.log_print("info string The given last two moves couldn't be applied to the previous board-state.")
                self.log_print("info string Rebuilding the game from scratch...")

                # create a new game state from scratch
                if position_type == "startpos":
                    self.new_game()
                else:
                    fen = " ".join(cmd_list[2:8])
                    self.gamestate.set_fen(fen)

                for move in mv_list:
                    self._apply_move(chess.Move.from_uci(move))
            else:
                self.log_print("info string Move Compatible")
        else:
            if position_type == "fen":
                fen = " ".join(cmd_list[2:8])
                self.gamestate.set_fen(fen)
                self.MCTS_agent.update_transposition_table((self.gamestate.get_transposition_key(),))
                # log_print("info string Added %s - count %d" % (gamestate.get_board_fen(),
                #                                    mcts_agent.transposition_table[gamestate.get_transposition_key()]))

    def _apply_move(self, selected_move: chess.Move):
        """
        Applies the given move on the gamestate and updates the transposition table of the environment
        :param selected_move: Move in python chess format
        :return:
        """

        self.gamestate.apply_move(selected_move)
        self.MCTS_agent.update_transposition_table((self.gamestate.get_transposition_key(),))
        # log_print("info string Added %s - count %d" % (gamestate.get_board_fen(),
        #                                               mcts_agent.transposition_table[
        #                                                   gamestate.get_transposition_key()]))

    def new_game(self):

        self.log_print("info string >> New Game")
        self.gamestate.new_game()
        self.MCTS_agent.transposition_table = collections.Counter()
        self.MCTS_agent.time_buffer_ms = 0
        self.MCTS_agent.dirichlet_epsilon = self.settings["centi_dirichlet_epsilon"] / 100

    def set_options(self, cmd_list):
        """
        Updates the internal options as requested by the use via the uci-protocol
        An example call could be: "setoption name nb_threads value 1"
        :param cmd_list: List of received of commands
        :return:
        """
        # make sure there exists enough items in the given command list like "setoption name nb_threads value 1"
        if len(cmd_list) >= 5:
            if cmd_list[1] != "name" or cmd_list[3] != "value":
                self.log_print("info string The given setoption command wasn't understood")
                self.log_print('info string An example call could be: "setoption name threads value 4"')
            else:
                option_name = cmd_list[2]

                if option_name not in self.settings:
                    self.log_print("info string The given option %s wasn't found in the settings list" % option_name)
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

                    for option in cmd_list:
                        self.settings[option] = True

                    self.log_print("info string Updated option %s to %s" % (option_name, value))

    def adjust_moves_left(self, moves_left, tc_type, prev_bm_value):
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

        # Don't spend too much time in the opening, we increase the moves_left
        # so that the movetime is reduced. engine_played_move is the actual moves
        # made by the engine excluding the book moves input from a GUI.
        if self.engine_played_move < self.settings["max_move_num_to_reduce_movetime"]:
            moves_left += self.moves_left_increment

        # Increase movetime by reducing the moves left if our prev bestmove value is below 0.0
        elif (
            self.settings["extend_time_on_bad_position"]
            and prev_bm_value is not None
            and prev_bm_value <= self.max_bad_pos_value
        ):
            if tc_type == "blitz":
                # The more the bad position is, the more that we extend the search time
                moves_left -= abs(prev_bm_value) * self.settings["moves_left"]
                moves_left = max(moves_left, self.min_moves_left)
            # Else if TC is traditional, we extend with more time if we have more time left
            elif moves_left > 4:
                moves_left = moves_left - moves_left // 8

        return moves_left

    def uci_reply(self):
        self.log_print("id name %s %s" % (self.client["name"], self.client["version"]))
        self.log_print("id author %s" % self.client["authors"])
        # tell the GUI all possible options
        self.log_print("option name UCI_Variant type combo default crazyhouse var crazyhouse")
        self.log_print("option name context type combo default %s var cpu var gpu" % self.settings["context"])
        self.log_print(
            "option name use_raw_network type check default %s"
            % ("false" if not self.settings["use_raw_network"] else "true")
        )
        self.log_print("option name threads type spin default %d min 1 max 4096" % self.settings["threads"])
        self.log_print("option name batch_size type spin default %d min 1 max 4096" % self.settings["batch_size"])
        self.log_print(
            "option name neural_net_services type spin default %d min 1 max 10" % self.settings["neural_net_services"]
        )
        self.log_print(
            "option name playouts_empty_pockets type spin default %d min 56 max 8192"
            % self.settings["playouts_empty_pockets"]
        )
        self.log_print(
            "option name playouts_filled_pockets type spin default %d min 56 max 8192"
            % self.settings["playouts_filled_pockets"]
        )
        self.log_print("option name centi_cpuct type spin default %d min 1 max 500" % self.settings["centi_cpuct"])
        self.log_print(
            "option name centi_dirichlet_epsilon type spin default %d min 0 max 100"
            % self.settings["centi_dirichlet_epsilon"]
        )
        self.log_print(
            "option name centi_dirichlet_alpha type spin default %d min 0 max 100"
            % self.settings["centi_dirichlet_alpha"]
        )
        self.log_print(
            "option name max_search_depth type spin default %d min 1 max 100" % self.settings["max_search_depth"]
        )
        self.log_print(
            "option name centi_temperature type spin default %d min 0 max 100" % self.settings["centi_temperature"]
        )
        self.log_print(
            "option name temperature_moves type spin default %d min 0 max 99999" % self.settings["temperature_moves"]
        )
        self.log_print(
            "option name opening_guard_moves type spin default %d min 0 max 99999"
            % self.settings["opening_guard_moves"]
        )
        self.log_print("option name centi_clip_quantil type spin default 0 min 0 max 100")
        self.log_print("option name virtual_loss type spin default 3 min 0 max 10")
        self.log_print(
            "option name centi_q_value_weight type spin default %d min 0 max 100"
            % self.settings["centi_q_value_weight"]
        )
        self.log_print(
            "option name threshold_time_for_raw_net_ms type spin default %d min 1 max 300000"
            % self.settings["threshold_time_for_raw_net_ms"]
        )
        self.log_print(
            "option name move_overhead_ms type spin default %d min 0 max 60000" % self.settings["move_overhead_ms"]
        )
        self.log_print("option name moves_left type spin default %d min 10 max 320" % self.settings["moves_left"])
        self.log_print(
            "option name extend_time_on_bad_position type check default %s"
            % ("false" if not self.settings["extend_time_on_bad_position"] else "true")
        )
        self.log_print(
            "option name max_move_num_to_reduce_movetime type spin default %d min 0 max 120"
            % self.settings["max_move_num_to_reduce_movetime"]
        )
        self.log_print(
            "option name check_mate_in_one type check default %s"
            % ("false" if not self.settings["check_mate_in_one"] else "true")
        )
        self.log_print(
            "option name use_pruning type check default %s" % ("false" if not self.settings["use_pruning"] else "true")
        )
        self.log_print(
            "option name use_oscillating_cpuct type check default %s"
            % ("false" if not self.settings["use_oscillating_cpuct"] else "true")
        )
        self.log_print(
            "option name use_time_management type check default %s"
            % ("false" if not self.settings["use_time_management"] else "true")
        )
        self.log_print(
            "option name verbose type check default %s" % ("false" if not self.settings["verbose"] else "true")
        )

        # verify that all options have been sent
        self.log_print("uciok")

    # main waiting loop for processing command line inputs

    def main(self):
        self.eprint(self.intro)
        while True:
            line = input()
            self.print_if_debug("waiting ...")
            self.print_if_debug(line)

            # wait for an std-in input command
            if line:
                # split the line to a list which makes parsing easier
                cmd_list = line.rstrip().split(" ")
                # extract the first command from the list for evaluation
                main_cmd = cmd_list[0]

                # write the given command to the log-file
                self.log(line)

                try:
                    if main_cmd == "uci":
                        self.uci_reply()
                    elif main_cmd == "isready":
                        self.setup_network()
                        self.log_print("readyok")
                    elif main_cmd == "ucinewgame":
                        self.bestmove_value = None
                        self.engine_played_move = 0
                        self.new_game()
                    elif main_cmd == "position":
                        self.setup_gamestate(cmd_list)
                    elif main_cmd == "setoption":
                        self.set_options(cmd_list)
                    elif main_cmd == "go":
                        self.perform_action(cmd_list)
                    elif main_cmd in ("quit", "exit"):
                        if self.log_file:
                            self.log_file.close()

                    else:
                        # give the user a message that the command was ignored
                        print("info string Unknown command: %s" % line)
                except IOError:
                    # log the error message to the log-file and exit the script
                    traceback.print_exc()


if __name__ == "__main__":
    CrazyAra.main(CrazyAra())
