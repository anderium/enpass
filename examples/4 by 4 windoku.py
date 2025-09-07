import os
import sys

# Fix for running examples by running python3 examples/<file>.py
sys.path.append(os.getcwd())  # noqa: PTH109

from variant_sudoku.constraints import LTGT
from variant_sudoku.sudoku import CellPosition, Sudoku


def blank_windoku():
    sudoku = Sudoku(given_digits={}, puzzle_size=4)
    sudoku.add_region_from_positions("Window", (2, 2), (3, 2), (2, 3), (3, 3))
    return sudoku


def windoku_with_digits():
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
    return sudoku


def windoku_patched():
    sudoku = blank_windoku()
    sudoku.add_constraints(
        LTGT(smaller=(2, 2), larger=(2, 1)),
        LTGT(smaller=(3, 2), larger=(3, 1)),
        LTGT(smaller=(3, 2), larger=(4, 2)),
        LTGT(smaller=(2, 3), larger=(1, 3)),
        LTGT(smaller=(2, 3), larger=(2, 4)),
    )
    return sudoku


if __name__ == "__main__":
    sudoku = windoku_with_digits()
    # boxes = sudoku.visualise_boxes_as_string()
    # print(boxes)
    sudoku.solve_logically()
    print(sudoku)
