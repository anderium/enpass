import pytest

from enpass.sudoku import (
    # Cell,
    CellPosition,
    # NumberPuzzle,
    # RectangularNumberPuzzle,
    # Region,
    # SquareNumberPuzzle,
    Sudoku,
)


def blank_windoku():
    sudoku = Sudoku(given_digits={}, puzzle_size=4)
    sudoku.add_region_from_positions("Window", (2, 2), (3, 2), (2, 3), (3, 3))
    return sudoku


def test_hidden_single():
    sudoku = Sudoku(
        puzzle_size=4,
        given_digits={
            CellPosition(1, 1): 1,
            CellPosition(2, 1): 2,
            CellPosition(1, 3): 2,
            CellPosition(2, 4): 4,
        },
    )
    sudoku.add_region_from_positions("Window", (2, 2), (3, 2), (2, 3), (3, 3))
    sudoku.solve_logically()
    assert not sudoku.is_unsolved()


@pytest.mark.skip("NotImplemented")
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
    sudoku.solve_logically()
    assert not sudoku.is_unsolved()
