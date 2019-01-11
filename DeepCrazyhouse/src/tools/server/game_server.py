import json
import logging
import chess
from flask import Flask, request, send_from_directory
from DeepCrazyhouse.src.domain.agent.NeuralNetAPI import NeuralNetAPI
from DeepCrazyhouse.src.domain.agent.player.MCTSAgent import MCTSAgent
from DeepCrazyhouse.src.domain.agent.player.RawNetAgent import RawNetAgent
from DeepCrazyhouse.src.domain.crazyhouse.GameState import GameState


file_lookup = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7}
rank_lookup = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7}


def get_square_index_from_name(name):
    if name is None:
        return None

    if len(name) != 2:
        return None

    col = file_lookup[name[0]] if name[0] in file_lookup else None
    row = rank_lookup[name[1]] if name[1] in rank_lookup else None
    if col is None or row is None:
        return None

    return chess.square(col, row)


batch_size = 8
nb_playouts = 16
cpuct = 1
dirichlet_epsilon = 0.25
nb_workers = 64


class ChessServer:
    def __init__(self, name):
        self.app = Flask(name)

        self.app.add_url_rule("/api/state", "api/state", self._wrap_endpoint(ChessServer.serve_state))
        self.app.add_url_rule("/api/new", "api/new", self._wrap_endpoint(ChessServer.serve_new_game))
        self.app.add_url_rule("/api/move", "api/move", self._wrap_endpoint(ChessServer.serve_move))
        self.app.add_url_rule("/", "serve_client_r", self._wrap_endpoint(ChessServer.serve_client))
        self.app.add_url_rule("/<path:path>", "serve_client", self._wrap_endpoint(ChessServer.serve_client))

        self._gamestate = GameState()

        net = NeuralNetAPI()

        # Loading network

        player_agents = {
            "raw_net": RawNetAgent(net),
            "mcts": MCTSAgent(
                net, virtual_loss=3, threads=batch_size, cpuct=cpuct, dirichlet_epsilon=dirichlet_epsilon
            ),
        }

        # Setting up agent
        self.agent = player_agents["raw_net"]
        # self.agent = player_agents["mcts"]

    def _wrap_endpoint(self, func):
        def wrapper(kwargs):
            return func(self, **kwargs)

        return lambda **kwargs: wrapper(kwargs)

    def run(self):
        self.app.run()

    @staticmethod
    def serve_client(path=None):
        if path is None:
            path = "index.html"
        return send_from_directory("./client", path)

    def serve_state(self):
        return self.serialize_game_state()

    def serve_new_game(self):
        logging.debug("staring new game()")
        self.perform_new_game()
        return self.serialize_game_state()

    def serve_move(self):

        # read move data
        drop_piece = request.args.get("drop")
        from_square = request.args.get("from")
        to_square = request.args.get("to")
        promotion_piece = request.args.get("promotion")
        from_square_idx = get_square_index_from_name(from_square)
        to_square_idx = get_square_index_from_name(to_square)
        if (from_square_idx is None and drop_piece is None) or to_square_idx is None:
            return self.serialize_game_state("board name is invalid")

        promotion = None
        drop = None

        if drop_piece is not None:
            from_square_idx = to_square_idx

            if not drop_piece in chess.PIECE_SYMBOLS:
                return self.serialize_game_state("drop piece name is invalid")
            drop = chess.PIECE_SYMBOLS.index(drop_piece)

        if promotion_piece is not None:
            if not promotion_piece in chess.PIECE_SYMBOLS:
                return self.serialize_game_state("promotion piece name is invalid")
            promotion = chess.PIECE_SYMBOLS.index(promotion_piece)

        move = chess.Move(from_square_idx, to_square_idx, promotion, drop)

        # perform move
        try:
            self.perform_move(move)
        except ValueError as e:
            logging.error("ValueError %s", e)
            return self.serialize_game_state(e.args[0])

        # calculate agent response
        if not self.perform_agent_move():
            return self.serialize_game_state("Black has no more moves to play", True)

        return self.serialize_game_state()

    def perform_new_game(self):
        self._gamestate = GameState()

    def perform_move(self, move):
        logging.debug("perform_move(): %s", move)
        # check if move is valid
        if move not in list(self._gamestate.board.legal_moves):
            raise ValueError("The given move %s is invalid for the current position" % move)
        self._gamestate.apply_move(move)
        if self._gamestate.is_won():
            logging.debug("Checkmate")
            return False
        return None

    def perform_agent_move(self):

        if self._gamestate.is_won():
            logging.debug("Checkmate")
            return False

        value, move, _, _ = self.agent.perform_action(self._gamestate)

        if self._gamestate.is_white_to_move() is False:
            value = -value

        logging.debug("Value %.4f", value)

        if move is None:
            logging.error("None move proposed!")
            return False

        self.perform_move(move)
        return True

    def serialize_game_state(self, message=None, finished=None):
        if message is None:
            message = ""

        board_str = "" + self._gamestate.board.__str__()
        pocket_str = "" + self._gamestate.board.pockets[1].__str__() + "|" + self._gamestate.board.pockets[0].__str__()
        state = {"board": board_str, "pocket": pocket_str, "message": message}
        if finished is not None:
            state["finished"] = finished
        return json.dumps(state)


print("Setting up server")
server = ChessServer("DeepCrazyHouse")

print("RUN")
server.run()
print("SHUTDOWN")
