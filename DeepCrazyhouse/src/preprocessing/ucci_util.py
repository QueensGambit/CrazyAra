from DeepCrazyhouse.src.domain.variants.constants import LABELS_XIANGQI


ROW_NUMBERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
COL_LETTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']


def xiangqi_board_move_to_ucci(old_pos, new_pos):
    """
    Converts a position (index) on a Xiangqi board represented
    as an numpy array of shape (9, 10) to the corresponding UCCI label.
    E.g., xiangqi_board_move_to_ucci((0, 0), (1, 0)) returns "a9a8"

    Args:
        old_pos (int, int): Tuple of integers representing the old position on the board.
        new_pos (int, int): Tuple of integers representing the new position on the board.

    Returns:
        ucci (string): UCCI string.
    """
    # Key corresponds to position on the board given by the arguments
    row_mapping = {0: '9', 1: '8', 2: '7', 3: '6', 4: '5', 5: '4', 6: '3',
                   7: '2', 8: '1', 9: '0'}
    col_mapping = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g',
                   7: 'h', 8: 'i'}
    ucci = col_mapping[old_pos[1]] + row_mapping[old_pos[0]] + \
           col_mapping[new_pos[1]] + row_mapping[new_pos[0]]

    return ucci


def mirror_ucci(ucci):
    """
    Mirrors the given UCCI label.
    Used when the whole board is mirrored horizontally.

    Args:
        ucci (string): Move in UCCI notation.

    Returns:
        ucci_mirrored (string): Mirrored move in UCCI notation.
    """
    ucci_flipped = ucci[0] + str(9 - int(ucci[1]))
    ucci_flipped += ucci[2] + str(9 - int(ucci[3]))
    return ucci_flipped


def generate_ucci_labels():
    """
    Generates UCCI labels of all legal moves.

    Returns:
        ucci_labels: List of strings representing all legal moves in UCCI notation.
    """
    ucci_labels = []
    for old_row in range(10):
        for old_col in range(9):
            destinations = [(old_row, new_col) for new_col in range(9)] + \
                           [(new_row, old_col) for new_row in range(10)]
            # horse moves
            destinations += [(old_row + row_change, old_col + col_change)
                             for (row_change, col_change) in
                             [(-2, -1), (-1, -2), (1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1)]]
            # elephant moves
            destinations += [(old_row + row_change, old_col + col_change)
                             for (row_change, col_change) in
                             [(2, -2), (2, 2), (-2, -2), (-2, 2)]
                             if (old_row, old_col) in
                             [(2, 0), (7, 0), (0, 2), (9, 2), (2, 4),
                              (7, 4), (0, 6), (9, 6), (2, 8), (7, 8)]] + \
                            [(old_row + row_change, old_col + col_change)
                             for (row_change, col_change) in
                             [(-2, -2), (-2, 2)]
                             if (old_row, old_col) in
                             [(4, 2), (4, 6)]] + \
                            [(old_row + row_change, old_col + col_change)
                             for (row_change, col_change) in
                             [(2, -2), (2, 2)]
                             if (old_row, old_col) in
                             [(5, 6), (5, 2)]]
            # advisor diagonal moves from mid palace
            destinations += [(old_row + row_change, old_col + col_change)
                             for (row_change, col_change) in
                             [(-1, -1), (1, -1), (1, 1), (-1, 1)]
                             if (old_row, old_col) in
                             [(1, 4), (8, 4)]]

            for (new_row, new_col) in destinations:
                if (old_row, old_col) != (new_row, new_col) \
                        and new_row in range(10) \
                        and new_col in range(9):
                    move = COL_LETTERS[old_col] + ROW_NUMBERS[old_row] + \
                           COL_LETTERS[new_col] + ROW_NUMBERS[new_row]
                    ucci_labels.append(move)

    # advisor moves to mid palace
    ucci_labels.append("d0e1")
    ucci_labels.append("f0e1")
    ucci_labels.append("d2e1")
    ucci_labels.append("f2e1")
    ucci_labels.append("d9e8")
    ucci_labels.append("f9e8")
    ucci_labels.append("d7e8")
    ucci_labels.append("f7e8")
    return ucci_labels


def get_target_index_of_mirrored_move(mirrored_ucci):
    """
    Returns the index of the original move, i.e. the move before it
    was mirrored, in the UCCI_LABELS constant.

    Args:
        mirrored_ucci: Move in UCCI notation that is mirrored.

    Returns:
        Index of the "unmirrored" move in the LABELS_XIANGQI constant.
    """
    original_move = mirror_ucci(mirrored_ucci)
    return LABELS_XIANGQI.index(original_move)


def write_ucci_labels_to_file(path):
    """
    Writes all legal moves in UCCI notation comma and new line separated 
    to a textfile at given path.

    Args:
        path (string): Path to textfile.
    """
    ucci_labels = generate_ucci_labels()
    with open(path, 'w') as file:
        for ucci_label in ucci_labels:
            # file.write("'%s',\n" % ucci_label)
            file.write("%s\n" % ucci_label)


def write_mirrored_ucci_indices_to_file(path):
    """
    Writes mirrored_moves and corresponding indices to a file
    in python dict notation.
    E.g., d9d3: 1964

    Args:
        path (string): Path to textfile.
    """
    with open(path, 'w') as file:
        for ucci in LABELS_XIANGQI:
            mirrored_ucci = mirror_ucci(ucci)
            index = get_target_index_of_mirrored_move(mirrored_ucci)
            file.write("'{}': {},\n".format(mirrored_ucci, index))