from enpass.sudoku import (
    # Cell,
    # CellPosition,
    # NumberPuzzle,
    # RectangularNumberPuzzle,
    # Region,
    # SquareNumberPuzzle,
    Sudoku,
)

# def blank_windoku():
#     sudoku = Sudoku(given_digits={}, puzzle_size=4)
#     sudoku.add_region_from_positions("Window", (2, 2), (3, 2), (2, 3), (3, 3))
#     return sudoku
#


def test_hidden_single():
    sudoku = Sudoku(
        puzzle_size=4,
        given_digits={
            (1, 1): 1,
            (2, 1): 2,
            (1, 3): 2,
            (2, 4): 4,
        },
    )
    sudoku.add_region_from_positions("Window", (2, 2), (3, 2), (2, 3), (3, 3))
    sudoku.bruteforce_solve()
    assert not sudoku.is_unsolved()


def test_naked_pair():
    sudoku = Sudoku(
        puzzle_size=6,
        given_digits={
            (2, 1): 1,
            (4, 1): 5,
            (6, 1): 6,
            (6, 2): 1,
            (5, 3): 2,
            (6, 3): 3,
            (2, 5): 5,
            (1, 6): 4,
            (3, 6): 3,
        },
    )
    sudoku.bruteforce_solve()
    assert not sudoku.is_unsolved()


def test_is_unique():
    sudoku_no_guessing_only_singles = Sudoku(
        puzzle_size=4,
        given_digits={
            (1, 1): 1,
            (2, 1): 2,
            (1, 3): 2,
            (2, 4): 4,
        },
    )
    sudoku_no_guessing_only_singles.add_region_from_positions("Window", (2, 2), (3, 2), (2, 3), (3, 3))
    assert sudoku_no_guessing_only_singles.is_unique()

    unique_sudoku_with_non_singles = Sudoku(
        puzzle_size=6,
        given_digits={
            (2, 1): 1,
            (4, 1): 5,
            (6, 1): 6,
            (6, 2): 1,
            (5, 3): 2,
            (6, 3): 3,
            (2, 5): 5,
            (1, 6): 4,
            (3, 6): 3,
        },
    )
    unique_sudoku_with_non_singles.bruteforce_solve()
    assert unique_sudoku_with_non_singles.is_unique()

    non_unique_sudoku = Sudoku(
        puzzle_size=4,
        given_digits={
            (1, 1): 1,
            (2, 1): 2,
            (1, 3): 2,
            # (2, 4): 4,
        },
    )
    non_unique_sudoku.add_region_from_positions("Window", (2, 2), (3, 2), (2, 3), (3, 3))
    assert not non_unique_sudoku.is_unique()

    impossible_sudoku = Sudoku(
        puzzle_size=4,
        given_digits={
            (1, 1): 1,
            (2, 1): 2,
            (1, 3): 2,
            (2, 4): 2,
        },
    )
    impossible_sudoku.add_region_from_positions("Window", (2, 2), (3, 2), (2, 3), (3, 3))
    assert not impossible_sudoku.is_unique()
