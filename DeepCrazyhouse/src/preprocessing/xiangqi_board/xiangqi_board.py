import re
from enum import Enum


class XiangqiBoard:
    """
    Represents a Xiangqi board.
    The board is initialized from the view of the red player (on the bottom).
    Empty fields and fields without figures of the corresponding player are represented as zeros.
    """

    def __init__(self):
        self.figure_mapping = {'king': 1, 'advisor': 2, 'elephant': 3,
                               'horse': 4, 'rook': 5, 'cannon': 6, 'pawn': 7}

        self.board = self.init_board()
        self.figures = self.get_num_figures()

    def get_positions(self, wxf_identifier, col=None):
        pos = []

        if col is None:
            for row in range(len(self.board)):
                for col in range(len(self.board[0])):
                    if self.board[row][col] != 0:
                        if self.board[row][col].wxf_identifier == wxf_identifier:
                            pos.append((row, col))
        else:
            for row in range(len(self.board)):
                if self.board[row][col] != 0:
                    if self.board[row][col].wxf_identifier == wxf_identifier:
                        pos.append((row, col))
        return pos

    def get_position_upper(self, wxf_identifier, col=None):
        pos = self.get_positions(wxf_identifier)

        if col is None:
            cols = [p[1] for p in pos]
            shared_col = list(set([c for c in cols if (cols.count(c) > 1)]))[0]
        else:
            shared_col = col

        upper_pos = None
        for p in pos:
            if p[1] == shared_col:
                if upper_pos is None:
                    upper_pos = p
                else:
                    if wxf_identifier.isupper():
                        if p[0] < upper_pos[0]:
                            upper_pos = p
                    else:
                        if p[0] > upper_pos[0]:
                            upper_pos = p
        return upper_pos

    def get_position_lower(self, wxf_identifier, col=None):
        pos = self.get_positions(wxf_identifier)

        if col is None:
            cols = [p[1] for p in pos]
            shared_col = list(set([c for c in cols if (cols.count(c) > 1)]))[0]
        else:
            shared_col = col

        lower_pos = None
        for p in pos:
            if p[1] == shared_col:
                if lower_pos is None:
                    lower_pos = p
                else:
                    if wxf_identifier.isupper():
                        if p[0] > lower_pos[0]:
                            lower_pos = p
                    else:
                        if p[0] < lower_pos[0]:
                            lower_pos = p
        return lower_pos

    def get_position_middle(self, wxf_identifier, col=None):
        pos = self.get_positions(wxf_identifier)

        if col is None:
            cols = [p[1] for p in pos]
            shared_col = list(set([c for c in cols if (cols.count(c) > 1)]))[0]
        else:
            shared_col = col

        rows = [p[0] for p in pos if p[1] == shared_col]
        rows.sort()
        mid_row = rows[len(rows) // 2]

        return (mid_row, shared_col)

    def get_position_consider_tandem(self, move):
        wxf_identifier = move[0]
        if move[1] == '+':
            old_row, old_col = self.get_position_upper(wxf_identifier)
        elif move[1] == '-':
            old_row, old_col = self.get_position_lower(wxf_identifier)
        else:
            if wxf_identifier.isupper():
                old_col = 9 - int(move[1])
            else:
                old_col = int(move[1]) - 1
            old_row = self.get_positions(wxf_identifier, col=old_col)[0][0]

        return (old_row, old_col)

    def get_positions_sorted_by_row(self, wxf_identifier, col=None):
        pos = self.get_positions(wxf_identifier)

        if col is None:
            cols = [p[1] for p in pos]
            shared_col = list(set([c for c in cols if (cols.count(c) > 1)]))[0]
        else:
            shared_col = col

        pos_shared_col = [p for p in pos if p[1] == shared_col]
        return sorted(pos_shared_col, key=lambda tup: tup[0])

    def move_king(self, move):
        wxf_identifier = move[0]
        old_row, old_col = self.get_positions(wxf_identifier)[0]

        piece = self.board[old_row][old_col]
        self.board[old_row][old_col] = 0

        direction = move[2]
        if wxf_identifier.isupper():
            if direction == '.':
                new_col = 9 - int(move[3])
                new_row = old_row
            elif direction == '+':
                new_col = old_col
                new_row = old_row - 1
            elif direction == '-':
                new_col = old_col
                new_row = old_row + 1
        else:
            if direction == '.':
                new_col = int(move[3]) - 1
                new_row = old_row
            elif direction == '+':
                new_col = old_col
                new_row = old_row + 1
            elif direction == '-':
                new_col = old_col
                new_row = old_row - 1

        self.board[new_row][new_col] = piece
        return [(old_row, old_col), (new_row, new_col)]

    def move_advisor(self, move):
        wxf_identifier = move[0]

        old_row, old_col = self.get_position_consider_tandem(move)
        piece = self.board[old_row][old_col]
        self.board[old_row][old_col] = 0

        direction = move[2]
        if wxf_identifier.isupper():
            new_col = 9 - int(move[3])
            if direction == '+':
                new_row = old_row - 1
            elif direction == '-':
                new_row = old_row + 1
        else:
            new_col = int(move[3]) - 1
            if direction == '+':
                new_row = old_row + 1
            elif direction == '-':
                new_row = old_row - 1

        self.board[new_row][new_col] = piece
        return [(old_row, old_col), (new_row, new_col)]

    def move_elephant(self, move):
        wxf_identifier = move[0]

        old_row, old_col = self.get_position_consider_tandem(move)
        piece = self.board[old_row][old_col]
        self.board[old_row][old_col] = 0

        direction = move[2]
        if wxf_identifier.isupper():
            new_col = 9 - int(move[3])
            if direction == '+':
                new_row = old_row - 2
            elif direction == '-':
                new_row = old_row + 2
        else:
            new_col = int(move[3]) - 1
            if direction == '+':
                new_row = old_row + 2
            elif direction == '-':
                new_row = old_row - 2

        self.board[new_row][new_col] = piece
        return [(old_row, old_col), (new_row, new_col)]

    def move_horse(self, move):
        wxf_identifier = move[0]

        old_row, old_col = self.get_position_consider_tandem(move)
        piece = self.board[old_row][old_col]
        self.board[old_row][old_col] = 0

        direction = move[2]
        if wxf_identifier.isupper():
            new_col = 9 - int(move[3])
            if direction == '+':
                new_row = old_row - 2 if abs(old_col - new_col) == 1 else old_row - 1
            elif direction == '-':
                new_row = old_row + 2 if abs(old_col - new_col) == 1 else old_row + 1
        else:
            new_col = int(move[3]) - 1
            if direction == '+':
                new_row = old_row + 2 if abs(old_col - new_col) == 1 else old_row + 1
            elif direction == '-':
                new_row = old_row - 2 if abs(old_col - new_col) == 1 else old_row - 1

        self.board[new_row][new_col] = piece
        return [(old_row, old_col), (new_row, new_col)]

    def move_chariot_or_cannon(self, move):
        # Chariots and cannons share possible movements
        wxf_identifier = move[0]

        old_row, old_col = self.get_position_consider_tandem(move)
        piece = self.board[old_row][old_col]
        self.board[old_row][old_col] = 0

        direction = move[2]
        if wxf_identifier.isupper():
            if direction == '.':
                new_col = 9 - int(move[3])
                new_row = old_row
            elif direction == '+':
                new_col = old_col
                new_row = old_row - int(move[3])
            elif direction == '-':
                new_col = old_col
                new_row = old_row + int(move[3])
        else:
            if direction == '.':
                new_col = int(move[3]) - 1
                new_row = old_row
            elif direction == '+':
                new_col = old_col
                new_row = old_row + int(move[3])
            elif direction == '-':
                new_col = old_col
                new_row = old_row - int(move[3])

        self.board[new_row][new_col] = piece
        return [(old_row, old_col), (new_row, new_col)]

    def move_pawn(self, move, red_move):
        # There are special cases for pawns as we can have
        # 1, 2, 3, 4, 5 pawns in a column, as well as
        # tandem pawns in two columns
        one_to_nine_str = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

        direction = move[2]

        # The given identifier might not be the figure but
        # the column in which the figure currently is positioned
        wxf_figure = 'P' if red_move else 'p'
        pos = self.get_positions(wxf_figure)

        # Find the current position
        cols = [p[1] for p in pos]
        shared_cols = list(set([c for c in cols if (cols.count(c) > 1)]))
        # Check whether we want to move a pawn from a tandem position
        if move[1] in one_to_nine_str:
            col_on_board = (9 - int(move[1])) if red_move else (int(move[1]) - 1)
            if col_on_board not in shared_cols:
                shared_cols = []

        if len(shared_cols) == 0:
            col = (9 - int(move[1])) if red_move else (int(move[1]) - 1)
            old_row, old_col = self.get_positions(wxf_figure, col=col)[0]
        elif len(shared_cols) == 1:
            shared_col = shared_cols[0]
            # how many pawns in the same column
            rows = [p[0] for p in pos if p[1] == shared_col]
            if len(rows) == 1:
                old_row, old_col = (rows[0], shared_col)
            elif len(rows) == 2:
                if move[1] == '+':
                    old_row, old_col = self.get_position_upper(wxf_figure, col=shared_col)
                elif move[1] == '-':
                    old_row, old_col = self.get_position_lower(wxf_figure, col=shared_col)
            elif len(rows) == 3:
                if move[1] in one_to_nine_str:
                    old_row, old_col = self.get_position_middle(wxf_figure, col=shared_col)
                elif move[1] == '+':
                    old_row, old_col = self.get_position_upper(wxf_figure, col=shared_col)
                elif move[1] == '-':
                    old_row, old_col = self.get_position_lower(wxf_figure, col=shared_col)
            elif len(rows) == 4:
                if move[0] == '+':
                    old_row, old_col = self.get_position_upper(wxf_figure, col=shared_col)
                elif move[0] == '-':
                    old_row, old_col = self.get_position_lower(wxf_figure, col=shared_col)
                elif move[0] == wxf_figure:
                    possible_pos = self.get_positions_sorted_by_row(wxf_figure, col=shared_col)
                    if move[1] == '+':
                        old_row, old_col = possible_pos[-2]
                    elif move[1] == '-':
                        old_row, old_col = possible_pos[1]
            elif len(rows) == 5:
                if move[0] == '+' and move[1] == '+':
                    old_row, old_col = self.get_position_upper(wxf_figure, col=shared_col)
                elif move[0] == '-' and move[1] == '-':
                    old_row, old_col = self.get_position_lower(wxf_figure, col=shared_col)
                elif move[0] == wxf_figure:
                    if move[1] == '+':
                        old_row, old_col = self.get_positions_sorted_by_row(wxf_figure, col=shared_col)[-2]
                    elif move[1] == '-':
                        old_row, old_col = self.get_positions_sorted_by_row(wxf_figure, col=shared_col)[1]
                    elif move[1] in one_to_nine_str:
                        old_row, old_col = self.get_position_middle(wxf_figure, col=shared_col)
        else:
            # the current column of the figure
            wxf_identifier = (9 - int(move[0])) if red_move else (int(move[0]) - 1)
            if move[1] in one_to_nine_str:
                old_row, old_col = self.get_position_middle(wxf_figure, col=wxf_identifier)
            elif move[1] == '+':
                old_row, old_col = self.get_position_upper(wxf_figure, col=wxf_identifier)
            elif move[1] == '-':
                old_row, old_col = self.get_position_lower(wxf_figure, col=wxf_identifier)

        piece = self.board[old_row][old_col]
        self.board[old_row][old_col] = 0

        if wxf_figure.isupper():
            if direction == '.':
                new_col = 9 - int(move[3])
                new_row = old_row
            elif direction == '+':
                new_col = old_col
                new_row = old_row - 1
        else:
            if direction == '.':
                new_col = int(move[3]) - 1
                new_row = old_row
            elif direction == '+':
                new_col = old_col
                new_row = old_row + 1

        self.board[new_row][new_col] = piece
        return [(old_row, old_col), (new_row, new_col)]

    def parse_movelist(self, movelist, display_moves=False):
        # In the case that 2 different columns are shared by at least 2 pawns each
        one_to_nine_str = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

        if display_moves:
            self.display_board()

        coordinate_changes = []

        regex = re.compile(r'^\d+\.$')
        movelist = [move for move in movelist.split(' ') if not regex.match(move)]

        red_move = True if movelist[0][0].isupper() else False
        for move in movelist:
            if display_moves:
                print("Move: ", move)

            wxf_identifier = move[0]

            if wxf_identifier in ['K', 'k']:
                coordinate_change = self.move_king(move)
            elif wxf_identifier in ['A', 'a']:
                coordinate_change = self.move_advisor(move)
            elif wxf_identifier in ['E', 'e']:
                coordinate_change = self.move_elephant(move)
            elif wxf_identifier in ['H', 'h']:
                coordinate_change = self.move_horse(move)
            elif wxf_identifier in ['C', 'c', 'R', 'r']:
                coordinate_change = self.move_chariot_or_cannon(move)
            elif wxf_identifier in ['P', 'p', '+', '-'] or wxf_identifier in one_to_nine_str:
                coordinate_change = self.move_pawn(move, red_move)

            coordinate_changes.append(coordinate_change)

            red_move = not red_move

            if display_moves:
                self.display_board()

        return coordinate_changes

    def parse_single_move(self, move, red_move, display_move=False):
        # In the case that 2 different columns are shared by at least 2 pawns each
        one_to_nine_str = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

        if display_move:
            self.display_board()

        coordinate_changes = []

        if display_move:
            print("Move: ", move)

        wxf_identifier = move[0]

        if wxf_identifier in ['K', 'k']:
            coordinate_change = self.move_king(move)
        elif wxf_identifier in ['A', 'a']:
            coordinate_change = self.move_advisor(move)
        elif wxf_identifier in ['E', 'e']:
            coordinate_change = self.move_elephant(move)
        elif wxf_identifier in ['H', 'h']:
            coordinate_change = self.move_horse(move)
        elif wxf_identifier in ['C', 'c', 'R', 'r']:
            coordinate_change = self.move_chariot_or_cannon(move)
        elif wxf_identifier in ['P', 'p', '+', '-'] or wxf_identifier in one_to_nine_str:
            coordinate_change = self.move_pawn(move, red_move)

        coordinate_changes.append(coordinate_change)

        if display_move:
            self.display_board()

        return coordinate_changes

    def init_board(self):
        board = [[0] * 9 for _ in range(10)]

        board[0][0] = Rook(Color.BLACK)
        board[0][-1] = Rook(Color.BLACK)
        board[-1][0] = Rook(Color.RED)
        board[-1][-1] = Rook(Color.RED)

        board[0][1] = Horse(Color.BLACK)
        board[0][-2] = Horse(Color.BLACK)
        board[-1][1] = Horse(Color.RED)
        board[-1][-2] = Horse(Color.RED)

        board[0][2] = Elephant(Color.BLACK)
        board[0][-3] = Elephant(Color.BLACK)
        board[-1][2] = Elephant(Color.RED)
        board[-1][-3] = Elephant(Color.RED)

        board[0][3] = Advisor(Color.BLACK)
        board[0][-4] = Advisor(Color.BLACK)
        board[-1][3] = Advisor(Color.RED)
        board[-1][-4] = Advisor(Color.RED)

        board[0][4] = King(Color.BLACK)
        board[-1][4] = King(Color.RED)

        board[2][1] = Cannon(Color.BLACK)
        board[2][-2] = Cannon(Color.BLACK)
        board[-3][1] = Cannon(Color.RED)
        board[-3][-2] = Cannon(Color.RED)

        board[3][0] = Pawn(Color.BLACK)
        board[3][2] = Pawn(Color.BLACK)
        board[3][4] = Pawn(Color.BLACK)
        board[3][6] = Pawn(Color.BLACK)
        board[3][8] = Pawn(Color.BLACK)
        board[-4][0] = Pawn(Color.RED)
        board[-4][2] = Pawn(Color.RED)
        board[-4][4] = Pawn(Color.RED)
        board[-4][6] = Pawn(Color.RED)
        board[-4][8] = Pawn(Color.RED)

        return board

    def get_num_figures(self):
        figures = {'k': 0, 'a': 0, 'e': 0, 'h': 0, 'r': 0, 'c': 0, 'p': 0,
                   'K': 0, 'A': 0, 'E': 0, 'H': 0, 'R': 0, 'C': 0, 'P': 0}

        for row in range(len(self.board)):
            for col in range(len(self.board[0])):
                if self.board[row][col] != 0:
                    wxf_identifier = self.board[row][col].wxf_identifier
                    figures[wxf_identifier] += 1

        return figures

    def display_board(self):
        board = [[0] * 9 for _ in range(10)]
        for row in range(len(self.board)):
            for col in range(len(self.board[0])):
                if self.board[row][col] != 0:
                    piece = self.board[row][col]
                    board[row][col] = str(piece.wxf_identifier)
                else:
                    board[row][col] = '0'
        print(board)

    def get_bitboard(self):
        board = [[0] * 9 for _ in range(10)]
        for row in range(len(self.board)):
            for col in range(len(self.board[0])):
                if self.board[row][col] != 0:
                    board[row][col] = 1
                else:
                    board[row][col] = 0
        return board


class Color(Enum):
    RED = 1
    BLACK = 2


class King:
    def __init__(self, color):
        self.color = color
        self.wxf_identifier = 'K' if color == color.RED else 'k'


class Advisor:
    def __init__(self, color):
        self.color = color
        self.wxf_identifier = 'A' if color == color.RED else 'a'


class Elephant:
    def __init__(self, color):
        self.color = color
        self.wxf_identifier = 'E' if color == color.RED else 'e'


class Rook:
    def __init__(self, color):
        self.color = color
        self.wxf_identifier = 'R' if color == color.RED else 'r'


class Cannon:
    def __init__(self, color):
        self.color = color
        self.wxf_identifier = 'C' if color == color.RED else 'c'


class Horse:
    def __init__(self, color):
        self.color = color
        self.wxf_identifier = 'H' if color == color.RED else 'h'


class Pawn:
    def __init__(self, color):
        self.color = color
        self.wxf_identifier = 'P' if color == color.RED else 'p'
