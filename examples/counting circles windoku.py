import os
import sys
from collections import Counter
from itertools import combinations
from typing import Any, Callable, Generator

# Fix for running examples by running python3 examples/<file>.py
sys.path.append(os.getcwd())  # noqa: PTH109

from enpass.constraints import AnyPosition, Constraint, to_CellPositions
from enpass.sudoku import Cell, Sudoku


def blank_windoku():
    sudoku = Sudoku(given_digits={}, puzzle_size=4)
    sudoku.add_region_from_positions("Window", (2, 2), (3, 2), (2, 3), (3, 3))
    return sudoku


class CountingCircles(Constraint):
    constraint_name = "Counting Circles"

    def __init__(self, *, cells: tuple[AnyPosition, ...], **kwargs: Any):
        cells_positions = to_CellPositions(cells)
        super().__init__(affected_cells=[*cells_positions], **kwargs)

    def update_cell_candidates(self, cells: list[Cell]) -> list[Cell]:
        return []

    # This is efficient, but it looks like there are things missing.
    # Still, that is not the case because cells cannot be empty in solved sudokus.
    def check(self, cells: list[Cell]) -> bool:
        how_often_do_digits_occur = Counter(cell.digit for cell in cells if cell.digit is not None)
        if sum(how_often_do_digits_occur.keys()) > len(cells):
            return False
        return all(how_often <= digit for digit, how_often in how_often_do_digits_occur.items())


type Pos = tuple[int, int]


def iterate_six_max_two_per_box() -> Generator[tuple[Pos, Pos, Pos, Pos, Pos, Pos], None, None]:
    top_left = ((1, 1), (2, 1), (1, 2), (2, 2))
    top_right = ((3, 1), (4, 1), (3, 2), (4, 2))
    bottom_left = ((1, 3), (2, 3), (1, 4), (2, 4))
    bottom_right = ((3, 3), (4, 3), (3, 4), (4, 4))
    # None in top left box
    for pos1, pos2 in combinations(top_right, 2):
        for pos3, pos4 in combinations(bottom_left, 2):
            for pos5, pos6 in combinations(bottom_right, 2):
                yield pos1, pos2, pos3, pos4, pos5, pos6
    # One in top left box, one in top right box
    for pos1 in top_left:
        for pos2 in top_right:
            for pos3, pos4 in combinations(bottom_left, 2):
                for pos5, pos6 in combinations(bottom_right, 2):
                    yield pos1, pos2, pos3, pos4, pos5, pos6
    # One in top left box, one in bottom right box
    for pos1 in top_left:
        for pos2, pos3 in combinations(top_right, 2):
            for pos4 in bottom_right:
                for pos5, pos6 in combinations(bottom_left, 2):
                    yield pos1, pos2, pos3, pos4, pos5, pos6


# Oh, obviously this fails, there's no way to distinguish between 14 and 23 outside the circles.
def iterate_five_max_two_per_box() -> Generator[tuple[Pos, Pos, Pos, Pos, Pos], None, None]:
    top_left = ((1, 1), (2, 1), (1, 2), (2, 2))
    top_right = ((3, 1), (4, 1), (3, 2), (4, 2))
    bottom_left = ((1, 3), (2, 3), (1, 4), (2, 4))
    bottom_right = ((3, 3), (4, 3), (3, 4), (4, 4))
    # None in top left box, one in top right
    # for pos_tr in combinations(top_right, 1):
    for pos_tr in top_right:
        for pos_bl in combinations(bottom_left, 2):
            for pos_br in combinations(bottom_right, 2):
                yield (pos_tr, *pos_bl, *pos_br)
    # None in top left box, one in bottom right box
    for pos_tr in combinations(top_right, 2):
        for pos_bl in combinations(bottom_left, 2):
            for pos_br in combinations(bottom_right, 1):
                yield (*pos_tr, *pos_bl, *pos_br)  # pyright: ignore[reportReturnType]
    # Two in top left box
    for pos_tl in combinations(top_left, 2):
        for pos_tr in combinations(top_right, 1):
            for pos_bl in combinations(bottom_left, 1):
                for pos_br in combinations(bottom_right, 1):
                    yield (*pos_tl, *pos_tr, *pos_bl, *pos_br)  # pyright: ignore[reportReturnType]


def iterate_seven_max_two_per_box() -> Generator[tuple[Pos, Pos, Pos, Pos, Pos, Pos, Pos], None, None]:
    top_left = ((1, 1), (2, 1), (1, 2), (2, 2))
    top_right = ((3, 1), (4, 1), (3, 2), (4, 2))
    bottom_left = ((1, 3), (2, 3), (1, 4), (2, 4))
    bottom_right = ((3, 3), (4, 3), (3, 4), (4, 4))
    # One in top left box, one in top right box
    for pos1 in top_left:
        for pos2, pos3 in combinations(top_right, 2):
            for pos4, pos5 in combinations(bottom_left, 2):
                for pos6, pos7 in combinations(bottom_right, 2):
                    yield pos1, pos2, pos3, pos4, pos5, pos6, pos7


def iterate_eight_max_two_per_box() -> Generator[tuple[Pos, Pos, Pos, Pos, Pos, Pos, Pos, Pos], None, None]:
    top_left = ((1, 1), (2, 1), (1, 2), (2, 2))
    top_right = ((3, 1), (4, 1), (3, 2), (4, 2))
    bottom_left = ((1, 3), (2, 3), (1, 4), (2, 4))
    bottom_right = ((3, 3), (4, 3), (3, 4), (4, 4))
    # One in top left box, one in top right box
    for pos0, pos1 in combinations(top_left, 2):
        for pos2, pos3 in combinations(top_right, 2):
            for pos4, pos5 in combinations(bottom_left, 2):
                for pos6, pos7 in combinations(bottom_right, 2):
                    yield pos0, pos1, pos2, pos3, pos4, pos5, pos6, pos7


def iterate_max_two_per_box_row_column[T: tuple[Pos, ...]](
    iterator: Callable[[], Generator[T, None, None]] = iterate_six_max_two_per_box,
) -> Generator[T, None, None]:
    middle_box = {(2, 2), (3, 2), (2, 3), (3, 3)}
    for all_positions in iterator():
        # Skip the ones that have more than two in rows and columns.
        xs = Counter(x for x, _y in all_positions)
        if xs.most_common(1)[0][1] > 2:
            continue
        ys = Counter(y for _x, y in all_positions)
        if ys.most_common(1)[0][1] > 2:
            continue
        # Extra region should also just have two at most
        if len(middle_box - set(all_positions)) < 2:
            continue
        yield all_positions


def iterate_max_two_per_box_row_column_all_used[T: tuple[Pos, ...]](
    iterator: Callable[[], Generator[T, None, None]] = iterate_max_two_per_box_row_column,
) -> Generator[T, None, None]:
    for all_positions in iterator():
        # Skip the ones that have more than two in rows and columns.
        xs = {x for x, _y in all_positions}
        if xs != {1, 2, 3, 4}:
            continue
        ys = {y for _x, y in all_positions}
        if ys != {1, 2, 3, 4}:
            continue
        yield all_positions


def visualise_circles(sudoku: Sudoku) -> str:
    grid = [[" "] * sudoku.puzzle_width for _ in range(sudoku.puzzle_height)]
    for cell in sudoku.constraints[0].affected_cells:
        grid[cell.y - 1][cell.x - 1] = "○"
    # Oops, ruff makes that code unreadble. It's adding a boundary around the grid
    return (
        f"┌{'─' * sudoku.puzzle_width}┐\n│{'│\n│'.join(''.join(row) for row in grid)}│\n└{'─' * sudoku.puzzle_width}┘"
    )


def main():
    # for positions in iterate_max_two_per_box_row_column_all_used():
    # for positions in iterate_max_two_per_box_row_column():
    # for positions in iterate_max_two_per_box_row_column(iterate_seven_max_two_per_box):
    for positions in iterate_max_two_per_box_row_column(iterate_eight_max_two_per_box):
        sudoku = blank_windoku()
        sudoku.add_constraints(constraint := CountingCircles(cells=positions))  # pyright: ignore [reportUnusedVariable]
        if sudoku.is_unique():
            print("UNIQUE")
            print(positions)
            circles = visualise_circles(sudoku)
            sudoku_repeat = blank_windoku()
            sudoku_repeat.add_constraints(constraint)
            sudoku_repeat.bruteforce_solve()
            digits = sudoku_repeat.visualise_digits_as_string()
            for circle_line, digits_line in zip(circles.split("\n"), digits.split("\n")):
                print(circle_line, digits_line, sep="\t\t")
            # print("impossible?", sudoku_repeat.is_impossible())
            # print("counting circles valid?", sudoku_repeat.check_constraint(constraint))
            print("------")


if __name__ == "__main__":
    main()
