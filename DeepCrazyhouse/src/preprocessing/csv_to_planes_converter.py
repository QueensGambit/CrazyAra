from xiangqi_board.xiangqi_board import XiangqiBoard
from DeepCrazyhouse.src.domain.variants.constants import LABELS_XIANGQI
from ucci_util import xiangqi_board_move_to_ucci, mirror_ucci
import time
import re
import zarr
import math
import logging
import numpy as np
import pandas as pd
from numcodecs import Blosc


class CSV2PlanesConverter:
    def __init__(self,
                 path_csv,
                 min_elo=None,
                 min_number_moves=None,
                 num_games_per_file=1000,
                 clevel=5,
                 compression="lz4"):
        log = logging.getLogger()
        log.setLevel(logging.DEBUG)

        logging.info("Reading csv into pandas dataframe")
        self._df = pd.read_csv(path_csv, delimiter=';')
        logging.info("Reading csv finished")

        self._min_elo = min_elo
        self._min_number_moves = min_number_moves

        if min_elo is not None or min_number_moves is not None:
            self._filter_csv()

        self.xiangqi_board = XiangqiBoard()

        self._num_games_per_file = num_games_per_file
        self._clevel = clevel
        self._compression = compression

    def _filter_csv(self):
        logging.info("Filter csv")

        if self._min_elo is not None:
            self._df = self._df[(self._df.red_elo >= self._min_elo) & (self._df.black_elo >= self._min_elo)]
        if self._min_number_moves is not None:
            self._df = self._df[self._df.num_moves >= self._min_number_moves]

        logging.info("Filter csv finished")

    def get_plain_piece_planes(self):
        plain_board = np.zeros((10, 9), dtype=np.int)

        king_plane, advisor_plane, elephant_plane, horse_plane, \
        rook_plane, cannon_plane, pawn_plane = (plain_board.copy() for i in range(7))

        red_piece_planes = np.array((king_plane,
                                     advisor_plane,
                                     elephant_plane,
                                     horse_plane,
                                     rook_plane,
                                     cannon_plane,
                                     pawn_plane))

        black_piece_planes = np.array((king_plane,
                                       advisor_plane,
                                       elephant_plane,
                                       horse_plane,
                                       rook_plane,
                                       cannon_plane,
                                       pawn_plane))
        return red_piece_planes, black_piece_planes

    def get_plain_pocket_count_planes(self):
        plain_board = np.zeros((10, 9), dtype=np.int)

        advisor_pocket_plane, elephant_pocket_plane, horse_pocket_plane, rook_pocket_plane, \
        cannon_pocket_plane, pawn_pocket_plane = (plain_board.copy() for i in range(6))

        red_pocket_planes = np.array((advisor_pocket_plane,
                                      elephant_pocket_plane,
                                      horse_pocket_plane,
                                      rook_pocket_plane,
                                      cannon_pocket_plane,
                                      pawn_pocket_plane))

        black_pocket_planes = np.array((advisor_pocket_plane,
                                        elephant_pocket_plane,
                                        horse_pocket_plane,
                                        rook_pocket_plane,
                                        cannon_pocket_plane,
                                        pawn_pocket_plane))
        return red_pocket_planes, black_pocket_planes

    def get_pocket_count_planes(self):
        red_pocket_planes, black_pocket_planes = self.get_plain_pocket_count_planes()

        initial_pieces_on_board = {'k': 1, 'a': 2, 'e': 2, 'h': 2, 'r': 2, 'c': 2, 'p': 5,
                                   'K': 1, 'A': 2, 'E': 2, 'H': 2, 'R': 2, 'C': 2, 'P': 5}
        current_pieces_on_board = self.xiangqi_board.get_num_figures()
        for wxf_identifier in current_pieces_on_board.keys():
            if wxf_identifier == 'A':
                red_pocket_planes[0].fill(initial_pieces_on_board['A'] - current_pieces_on_board['A'])
            elif wxf_identifier == 'a':
                black_pocket_planes[0].fill(initial_pieces_on_board['a'] - current_pieces_on_board['a'])
            elif wxf_identifier == 'E':
                red_pocket_planes[1].fill(initial_pieces_on_board['E'] - current_pieces_on_board['E'])
            elif wxf_identifier == 'e':
                black_pocket_planes[1].fill(initial_pieces_on_board['e'] - current_pieces_on_board['e'])
            elif wxf_identifier == 'H':
                red_pocket_planes[2].fill(initial_pieces_on_board['H'] - current_pieces_on_board['H'])
            elif wxf_identifier == 'h':
                black_pocket_planes[2].fill(initial_pieces_on_board['h'] - current_pieces_on_board['h'])
            elif wxf_identifier == 'R':
                red_pocket_planes[3].fill(initial_pieces_on_board['R'] - current_pieces_on_board['R'])
            elif wxf_identifier == 'r':
                black_pocket_planes[3].fill(initial_pieces_on_board['r'] - current_pieces_on_board['r'])
            elif wxf_identifier == 'C':
                red_pocket_planes[4].fill(initial_pieces_on_board['C'] - current_pieces_on_board['C'])
            elif wxf_identifier == 'c':
                black_pocket_planes[4].fill(initial_pieces_on_board['c'] - current_pieces_on_board['c'])
            elif wxf_identifier == 'P':
                red_pocket_planes[5].fill(initial_pieces_on_board['P'] - current_pieces_on_board['P'])
            elif wxf_identifier == 'p':
                black_pocket_planes[5].fill(initial_pieces_on_board['p'] - current_pieces_on_board['p'])

        return red_pocket_planes, black_pocket_planes

    def board_to_planes(self, red_move, pocket_count=True, flip=True):
        red_player_planes, black_player_planes = self.get_plain_piece_planes()

        for row in range(len(self.xiangqi_board.board)):
            for col in range(len(self.xiangqi_board.board[0])):
                piece = self.xiangqi_board.board[row][col]
                if piece != 0:
                    if piece.wxf_identifier == 'K':
                        red_player_planes[0][row, col] = 1
                    elif piece.wxf_identifier == 'k':
                        black_player_planes[0][row, col] = 1

                    elif piece.wxf_identifier == 'A':
                        red_player_planes[1][row, col] = 1
                    elif piece.wxf_identifier == 'a':
                        black_player_planes[1][row, col] = 1

                    elif piece.wxf_identifier == 'E':
                        red_player_planes[2][row, col] = 1
                    elif piece.wxf_identifier == 'e':
                        black_player_planes[2][row, col] = 1

                    elif piece.wxf_identifier == 'H':
                        red_player_planes[3][row, col] = 1
                    elif piece.wxf_identifier == 'h':
                        black_player_planes[3][row, col] = 1

                    elif piece.wxf_identifier == 'R':
                        red_player_planes[4][row, col] = 1
                    elif piece.wxf_identifier == 'r':
                        black_player_planes[4][row, col] = 1

                    elif piece.wxf_identifier == 'C':
                        red_player_planes[5][row, col] = 1
                    elif piece.wxf_identifier == 'c':
                        black_player_planes[5][row, col] = 1

                    elif piece.wxf_identifier == 'P':
                        red_player_planes[6][row, col] = 1
                    elif piece.wxf_identifier == 'p':
                        black_player_planes[6][row, col] = 1

        if flip and not red_move:
            for i in range(7):
                red_player_planes[i] = np.flip(red_player_planes[i], 0)
                black_player_planes[i] = np.flip(black_player_planes[i], 0)

        if red_move:
            planes = np.vstack((red_player_planes, black_player_planes))
        else:
            planes = np.vstack((black_player_planes, red_player_planes))

        if pocket_count:
            red_pocket_planes, black_pocket_planes = self.get_pocket_count_planes()
            if red_move:
                pocket_planes = np.vstack((red_pocket_planes, black_pocket_planes))
            else:
                pocket_planes = np.vstack((black_pocket_planes, red_pocket_planes))
            planes = np.vstack((planes, pocket_planes))

        return planes

    def game_to_planes(self, result, movelist, flip=True):
        red_move = movelist[0][0].isupper()

        total_move_count = 0
        total_move_count_fen = 0

        board_planes = self.board_to_planes(red_move, pocket_count=True, flip=flip)
        color_plane = np.ones((10, 9), dtype=np.int) if red_move else np.zeros((10, 9), dtype=np.int)
        total_move_count_plane = np.full((10, 9), total_move_count_fen, dtype=np.int)

        X_planes = [np.vstack((board_planes, np.array((color_plane, total_move_count_plane))))]
        y_value = []
        y_policy = []

        for i in range(len(movelist)):
            if red_move:
                if result == "1-0":
                    y_value.append(1)
                elif result == "0-1":
                    y_value.append(-1)
                else:
                    y_value.append(0)
            else:
                if result == "1-0":
                    y_value.append(-1)
                elif result == "0-1":
                    y_value.append(1)
                else:
                    y_value.append(0)

            # Play move
            coordinate_change = self.xiangqi_board.parse_single_move(movelist[i], red_move)[0]

            # Build policy
            old_pos = coordinate_change[0]
            new_pos = coordinate_change[1]
            ucci = xiangqi_board_move_to_ucci(old_pos, new_pos)
            if flip and not red_move:
                ucci = mirror_ucci(ucci)

            # Index of ucci in LABELS_XIANGQI constant
            index = np.where(np.asarray(LABELS_XIANGQI) == ucci)[0][0]

            plain_policy_vector = np.zeros((len(LABELS_XIANGQI)))
            plain_policy_vector[index] = 1
            y_policy.append(plain_policy_vector)

            total_move_count += 1
            if total_move_count % 2 == 0:
                total_move_count_fen += 1
            red_move = not red_move

            board_planes = self.board_to_planes(red_move, pocket_count=True, flip=flip)
            color_plane = np.ones((10, 9), dtype=np.int) if red_move else np.zeros((10, 9), dtype=np.int)
            total_move_count_plane = np.full((10, 9), total_move_count_fen, dtype=np.int)

            if i != len(movelist) - 1:
                X_planes.append(np.vstack((board_planes, np.array((color_plane, total_move_count_plane)))))

        # Reset board
        self.xiangqi_board = XiangqiBoard()

        return np.asarray(X_planes), np.asarray(y_value), np.asarray(y_policy)

    def export_batches(self, export_path):
        start_time = time.time()

        batch_size = self._num_games_per_file
        num_batches = math.ceil(len(self._df) / batch_size)

        regex = re.compile(r'^\d+\.$')
        batch_start = 0
        for b in range(num_batches):
            logging.info("Creating batch {}/{}".format(str(b + 1), num_batches))

            batch = {}
            for g in range(batch_start, batch_start + batch_size):
                if g >= len(self._df):
                    break

                movelist = self._df.iloc[g].moves
                movelist = [move for move in movelist.split(' ') if not regex.match(move)]
                result = self._df.iloc[g].result
                x, y_value, y_policy = self.game_to_planes(result, movelist)

                # Game playing data
                if 'x' not in batch.keys():
                    batch['x'] = x
                    batch['start_indices'] = np.asarray([0])
                else:
                    batch['start_indices'] = np.concatenate((batch['start_indices'],
                                                             np.asarray([len(batch['x'])])))
                    batch['x'] = np.concatenate((batch['x'], x))

                if 'y_value' not in batch.keys():
                    batch['y_value'] = y_value
                else:
                    batch['y_value'] = np.concatenate((batch['y_value'], y_value))

                if 'y_policy' not in batch.keys():
                    batch['y_policy'] = y_policy
                else:
                    batch['y_policy'] = np.concatenate((batch['y_policy'], y_policy))

                # Statistics
                if 'elo_red' not in batch.keys():
                    batch['elo_red'] = np.asarray([int(self._df.iloc[g].red_elo)])
                else:
                    current_elo_red = int(self._df.iloc[g].red_elo)
                    batch['elo_red'] = np.concatenate((batch['elo_red'], np.asarray([current_elo_red])))

                if 'elo_black' not in batch.keys():
                    batch['elo_black'] = np.asarray([int(self._df.iloc[g].black_elo)])
                else:
                    current_elo_black = int(self._df.iloc[g].black_elo)
                    batch['elo_black'] = np.concatenate((batch['elo_black'], np.asarray([current_elo_black])))

                if 'num_moves' not in batch.keys():
                    batch['num_moves'] = np.asarray([int(self._df.iloc[g].num_moves)])
                else:
                    current_num_moves = int(self._df.iloc[g].num_moves)
                    batch['num_moves'] = np.concatenate((batch['num_moves'], np.asarray([current_num_moves])))

                # Metadata
                if 'player_red' not in batch.keys():
                    batch['player_red'] = [self._df.iloc[g].red]
                else:
                    batch['player_red'].append(self._df.iloc[g].red)

                if 'player_black' not in batch.keys():
                    batch['player_black'] = [self._df.iloc[g].black]
                else:
                    batch['player_black'].append(self._df.iloc[g].black)

                if 'result' not in batch.keys():
                    batch['result'] = [result]
                else:
                    batch['result'].append(result)

                if 'event' not in batch.keys():
                    batch['event'] = [self._df.iloc[g].event]
                else:
                    batch['event'].append(self._df.iloc[g].event)

            logging.info("Exporting batch (time: {:.3f}m)".format((time.time() - start_time) / 60))
            self.export_batch(batch, export_path, "batch_" + str(b))

            batch_start += batch_size
        return True

    def export_batch(self, batch, export_path, filename):
        start_time = time.time()
        store = zarr.ZipStore(export_path + filename + ".zip", mode="w")
        zarr_file = zarr.group(store=store, overwrite=True)
        compressor = Blosc(cname=self._compression, clevel=self._clevel, shuffle=Blosc.SHUFFLE)

        x = batch['x']
        start_indices = batch['start_indices']
        y_value = batch['y_value']
        y_policy = batch['y_policy']

        elo_red = batch['elo_red']
        elo_black = batch['elo_black']
        num_moves = batch['num_moves']

        # Discard missing entries from average elo
        indices_red_elo_not_zero = np.where(elo_red > 0)[0]
        indices_black_elo_not_zero = np.where(elo_black > 0)[0]

        avg_elo_red = int(elo_red[indices_red_elo_not_zero].sum() / len(indices_red_elo_not_zero))
        avg_elo_black = int(elo_black[indices_black_elo_not_zero].sum() / len(indices_black_elo_not_zero))
        avg_elo = int((avg_elo_red + avg_elo_black) / 2)

        # metadata
        player_red = np.asarray(batch['player_red'], dtype='<U20')
        player_black = np.asarray(batch['player_black'], dtype='<U20')
        result = np.asarray(batch['result'], dtype='<U7')

        num_red_wins = len(np.where(result == '1-0')[0])
        num_black_wins = len(np.where(result == '0-1')[0])
        num_draws = len(np.where(result == '0.5-0.5')[0])

        zarr_file.create_dataset(name="x",
                                 data=x,
                                 shape=x.shape,
                                 dtype=np.int16,
                                 compression=compressor)

        zarr_file.create_dataset(name="start_indices",
                                 data=start_indices,
                                 shape=start_indices.shape,
                                 dtype=np.int32,
                                 compression=compressor)

        zarr_file.create_dataset(name="y_value",
                                 data=y_value,
                                 shape=y_value.shape,
                                 dtype=np.int16,
                                 compression=compressor)

        zarr_file.create_dataset(name="y_policy",
                                 data=y_policy,
                                 shape=y_policy.shape,
                                 dtype=np.int16,
                                 compression=compressor)

        zarr_file.create_group("/metadata")
        zarr_file.create_dataset(name="/metadata/player_red",
                                 data=player_red,
                                 shape=player_red.shape,
                                 dtype=player_red.dtype,
                                 compression=compressor)

        zarr_file.create_dataset(name="/metadata/player_black",
                                 data=player_black,
                                 shape=player_black.shape,
                                 dtype=player_black.dtype,
                                 compression=compressor)

        zarr_file.create_dataset(name="/metadata/result",
                                 data=result,
                                 shape=result.shape,
                                 dtype=result.dtype,
                                 compression=compressor)

        zarr_file.create_group("/statistics")
        zarr_file.create_dataset(name="/statistics/elo_red",
                                 data=elo_red,
                                 shape=elo_red.shape,
                                 dtype=np.int16,
                                 compression=compressor)

        zarr_file.create_dataset(name="/statistics/elo_black",
                                 data=elo_black,
                                 shape=elo_black.shape,
                                 dtype=np.int16,
                                 compression=compressor)

        zarr_file.create_dataset(name="/statistics/num_moves",
                                 data=num_moves,
                                 shape=num_moves.shape,
                                 dtype=np.int16,
                                 compression=compressor)

        zarr_file.create_dataset(name="/statistics/avg_elo_red",
                                 data=[avg_elo_red],
                                 shape=(1,),
                                 dtype=np.int16,
                                 compression=compressor)

        zarr_file.create_dataset(name="/statistics/avg_elo_black",
                                 data=[avg_elo_black],
                                 shape=(1,),
                                 dtype=np.int16,
                                 compression=compressor)

        zarr_file.create_dataset(name="/statistics/avg_elo",
                                 data=[avg_elo],
                                 shape=(1,),
                                 dtype=np.int16,
                                 compression=compressor)

        zarr_file.create_dataset(name="/statistics/num_red_wins",
                                 data=[num_red_wins],
                                 shape=(1,),
                                 dtype=np.int16,
                                 compression=compressor)

        zarr_file.create_dataset(name="/statistics/num_black_wins",
                                 data=[num_black_wins],
                                 shape=(1,),
                                 dtype=np.int16,
                                 compression=compressor)

        zarr_file.create_dataset(name="/statistics/num_draws",
                                 data=[num_draws],
                                 shape=(1,),
                                 dtype=np.int16,
                                 compression=compressor)

        if self._min_elo is not None and self._min_number_moves is not None:
            zarr_file.create_group("/parameters")
            zarr_file.create_dataset(name="/parameters/min_elo",
                                     data=[self._min_elo],
                                     shape=(1,),
                                     dtype=np.int16,
                                     compression=compressor)

            zarr_file.create_dataset(name="/parameters/min_moves",
                                     data=[self._min_number_moves],
                                     shape=(1,),
                                     dtype=np.int16,
                                     compression=compressor)

        store.close()
        logging.info("Export finished (time: {:.1f}s)".format(time.time() - start_time))
        return True

    def display_csv(self):
        print(self._df)
