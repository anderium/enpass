(
    """"""
    """
Notes for the future:

- Disjoint boxes are actually just additional regions on the box indices.  You can add that constraint with regions of
    [(1, 1), (1, 4), (1, 7), (4, 1), ..., (7, 7)], [(2, 1), ..., (8, 7)], etc.
- Clues outside the grid that are unknown are supported as cells with non-default candidates that do not belong to
    any region.
- Current implementation does not support Schroedinger cells
- Not sure how deconstructed sudoku would work. Empty cells aren't supported either.
- Actually, things like doubler cells and “unique on clues” are even more difficult?


Idea:
- “random” selection for which candidate to try: array of all candidates (max len normal sudoku = 243)
    index using [3_659_314_638_915_016_499 % array_length], which is (∏ p < 243) + 1 mod 2**63
"""
)

import itertools
import operator
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from operator import index
from os import devnull
from typing import Any, Callable, Generator, Iterator, NamedTuple, SupportsIndex

# To avoid circularity you should use it as constraints.Constraint. (Although we're supposed to be avoiding circularity there too.)
import enpass.constraints as constraints

range_from_0 = range


def range_from_1(stop_inclusive: SupportsIndex, step: SupportsIndex | None = None, /) -> range:
    if step is None:
        return range_from_0(1, index(stop_inclusive) + 1)
    return range_from_0(1, index(stop_inclusive) + 1, step)


def range():
    raise AssertionError("Specifically use range_from_[0|1] since you want to count from 1 for sudoku!")


@contextmanager
def suppress_stdout():
    """A context manager that redirects stdout to devnull."""
    # Copied from SO cause I'm lazy: https://stackoverflow.com/a/52442331
    with open(devnull, "w") as fnull, redirect_stdout(fnull) as out:
        yield out


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull."""
    # Copied from SO cause I'm lazy: https://stackoverflow.com/a/52442331
    with open(devnull, "w") as fnull, redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
        yield (err, out)


class CellPosition(NamedTuple):
    x: int
    y: int


# TODO: Doubler Cells constraint. Perhaps this can be done by adding a list of applied modifiers and candidate modifiers to cells?
@dataclass
class Cell:
    x: int = field(init=False)
    y: int = field(init=False)
    position: CellPosition
    candidates: set[int]
    """The digits that are still possible for this cell."""
    given: bool = field(default=False, init=False)
    _known_and_processed: bool = field(default=False, init=False)
    _bruteforce_stack: list[tuple[bool, set[int]]] = field(default_factory=list[tuple[bool, set[int]]], init=False)
    """Stack that contains the set of digits that this cell considers at the point that Cell.save_to_stack is called."""

    def __post_init__(self):
        self.x, self.y = self.position

    def __str__(self) -> str:
        return f"Cell(x={self.x}, y={self.y}: {self.candidates})"

    @property
    def known(self) -> bool:
        return self._known_and_processed

    @property
    def impossible(self) -> bool:
        return len(self.candidates) == 0

    @property
    def digit(self) -> int | None:
        if self.known:
            return next(iter(self.candidates))
        return None

    # TODO: In the future doublers, negators, squarers may influence this.
    @property
    def value(self) -> int | None:
        digit = self.digit
        if digit is None:
            return None
        return digit

    # TODO: In the future doublers, negators, squarers may influence this.
    @property
    def minimum(self) -> int:
        return min(self.candidates)

    # TODO: In the future doublers, negators, squarers may influence this.
    @property
    def maximum(self) -> int:
        return max(self.candidates)

    def save_to_stack(self) -> None:
        # TODO: If candidates are not ints (read: immutable) anymore, don't shallow copy.
        self._bruteforce_stack.append((self._known_and_processed, self.candidates.copy()))

    def restore_from_stack(self):
        # Do not set to self.candidates, since the set might be referenced somewhere…
        known, candidates = self._bruteforce_stack.pop()
        self._known_and_processed = known
        self.candidates.update(candidates)

    def set_digit(self, digit: int) -> None:
        self._known_and_processed = True
        self.candidates.intersection_update({digit})

    # TODO: Forcing this might be enough to adapt to doubler cells. Perhaps make candidates private as self._candidates
    def _discard_with_op[T](self, operator: Callable[[int, T], bool], value: T) -> bool:
        candidates_to_remove = set[int]()
        for candidate in self.candidates:
            if operator(candidate, value):
                candidates_to_remove.add(candidate)
        self.candidates.difference_update(candidates_to_remove)
        return len(candidates_to_remove) > 0

    def discard_less_than(self, value: int) -> bool:
        return self._discard_with_op(operator.lt, value)

    def discard_less_than_equal(self, value: int) -> bool:
        return self._discard_with_op(operator.le, value)

    def discard_greater_than(self, value: int) -> bool:
        return self._discard_with_op(operator.gt, value)

    def discard_greater_than_equal(self, value: int) -> bool:
        return self._discard_with_op(operator.ge, value)

    def discard_values(self, values: set[int]) -> bool:
        return self._discard_with_op((lambda candidate, values: candidate in values), values)

    def discard_other_values(self, values: set[int]) -> bool:
        return self._discard_with_op((lambda candidate, values: candidate not in values), values)

    # TODO: Should cells know about the sudoku, or how can I reasonably do this?
    # TODO: Should this be a property of the cells, or of the entire sudoku?
    #       And if it's the latter, how do I handle this with the constraints that use these properties?
    @property
    def orthogonally_adjacent_neighbours(self) -> list["Cell"]:
        raise NotImplementedError

    @property
    def king_move_adjacent_neighbours(self) -> list["Cell"]:
        raise NotImplementedError

    @property
    def seen_cells(self) -> list["Cell"]:
        # Neighbours, and everything in the same region
        raise NotImplementedError


class Region:
    def __init__(self, label: str, cells: list[Cell]):
        self.label = label
        self.cells: list[Cell] = cells
        self.cell_positions: set[CellPosition] = {cell.position for cell in cells}
        # TODO: Should regions even care about these links? Or just the puzzle?
        # TODO: This is something that I added for logical solves, but implementing that is probably far away
        self.strong_links: list[Any] = []  # TODO: Typing
        self.weak_links: list[Any] = []  # TODO: Typing

    def __repr__(self) -> str:
        return f"Region({self.label}, cells = { {tuple(pos) for pos in self.cell_positions}!r}, strong_links = {self.strong_links!r}, weak_links = {self.weak_links!r})"

    def __len__(self) -> int:
        return len(self.cells)

    def __iter__(self) -> Iterator[Cell]:
        return iter(self.cells)

    @property
    def unknown_cells(self) -> Generator[Cell, None, None]:
        return (cell for cell in self.cells if not cell.known)


class NumberPuzzle:
    def __init__(self, *, given_digits: dict[CellPosition, int], **kwargs: Any):
        super().__init__(**kwargs)
        self.cell_positions: set[CellPosition] = set()
        self.cells: list[Cell] = []
        self.valid_numbers: set[int] = set()

        self.regions: list[Region] = []
        # TODO: Partial Regions
        self.partial_regions: list[set[CellPosition]] = []
        self.constraints: list["constraints.Constraint"] = []

        self.position_to_cell: dict[CellPosition, Cell] = {}
        self.cell_position_to_regions = defaultdict[CellPosition, list[int]](list)
        self.cell_position_to_constraints = defaultdict[CellPosition, list["constraints.Constraint"]](list)

        self.given_digits = given_digits
        self._bruteforce_stack: list[tuple[Cell, int]] = []

    def __repr__(self):
        parts = [
            f"valid_numbers = {self.valid_numbers!r}",
            f"cells = {self.cells!r}",
            # f"cells = {self.cell_positions!r}",
            f"regions = {self.regions!r}",
            f"partial_regions = {self.partial_regions!r}",
        ]
        return f"{self.__class__.__name__}(\n\t{',\n\t'.join(parts)}\n)"

    @property
    def unknown_cells(self):
        return (cell for cell in self.cells if not cell.known)

    def positions_to_cells(self, positions: list[CellPosition]):
        return (self.position_to_cell[position] for position in positions)

    def add_cells(self, *cells: Cell):
        """Add cell to the list of all cells, and mark its given digit appropriately if it exists."""
        for cell in cells:
            if (given_digit := self.given_digits.get(cell.position)) is not None:
                cell.set_digit(given_digit)
                cell.given = True
            self.position_to_cell[cell.position] = cell
        self.cells.extend(cells)

    def add_regions(self, *regions: Region):
        """Add region to the list of all regions, and update the cell_position_to_regions appropriately."""
        for region_index, region in enumerate(regions, len(self.regions)):
            for cell in region:
                self.cell_position_to_regions[cell.position].append(region_index)
        self.regions.extend(regions)

    def add_region_from_positions(self, region_name: str, *cell_positions: CellPosition | tuple[int, int]):
        """Add region by only specifying the positions that should be in it."""
        cell_pos_set = {CellPosition(*position) for position in cell_positions}
        cells = [cell for cell in self.cells if cell.position in cell_pos_set]
        region = Region(region_name, cells)
        self.add_regions(region)

    # def validate_regions(self) -> bool:
    #     """Return whether all regions contain only cells from inside the number puzzle."""
    #     return all(cell.position in self.cell_positions for region in self.regions for cell in region) and all(
    #         cell in self.cell_positions for region in self.partial_regions for cell in region
    #     )

    def add_constraints(self, *constraints: "constraints.Constraint"):
        self.constraints.extend(constraints)
        for constraint in constraints:
            for cell_position in constraint.affected_cells:
                self.cell_position_to_constraints[cell_position].append(constraint)

    # These two are not properties because they are not efficient implementations!
    def is_unsolved(self) -> bool:
        return not all(cell.known for cell in self.cells)

    def is_impossible(self, cells_subset: set[CellPosition] | None = None) -> bool:
        if cells_subset is None:
            cells_subset = {cell.position for cell in self.cells}
            cells = self.cells
        else:
            cells = self.positions_to_cells(list(cells_subset))
        return any(cell.impossible for cell in cells) or not self.verify_constraints(cells_subset)

    def is_solved_double_check(self) -> bool:
        for region in self.regions:
            if {cell.digit for cell in region} != self.valid_numbers:
                return False
        for constraint in self.constraints:
            cells = list(self.positions_to_cells(constraint.affected_cells))
            if not constraint.check(cells):
                return False
        return True

    def set_cell(self, cell: Cell, digit: int):
        # TODO: Should it propagate to constraints? That is slow for bruteforce and might be unclear for logical solves.
        affected_positions: set[CellPosition] = {cell.position}
        cell.set_digit(digit)
        affected_regions = self.cell_position_to_regions[cell.position]
        for region_index in affected_regions:
            region = self.regions[region_index]
            for affected_cell in region:
                if affected_cell.position == cell.position:
                    continue
                affected_cell.candidates.discard(digit)
                affected_positions.add(affected_cell.position)
        return affected_positions
        # TODO: Update constraint candidates
        affected_constraints: set["constraints.Constraint"] = set()
        for position in affected_positions:
            affected_constraints.update(self.cell_position_to_constraints[position])
        affected_cells: list[Cell] = []
        for affected_constraint in affected_constraints:
            provided_cells = list(self.positions_to_cells(affected_constraint.affected_cells))
            cells_with_changed_candidates = affected_constraint.update_cell_candidates(provided_cells)
            affected_cells.extend(cells_with_changed_candidates)
        return affected_cells

    def verify_constraints(self, changed_position: set[CellPosition] | None = None) -> bool:
        if changed_position is None:
            constraints_to_check = self.constraints
        elif len(changed_position) == 0:
            return True
        else:
            constraints_to_check = set["constraints.Constraint"].union(
                *(set(self.cell_position_to_constraints[position]) for position in changed_position)
            )
        return all(self.check_constraint(constraint) for constraint in constraints_to_check)

    def initial_candidates(self):
        # Initial candidates from given digits
        for position, digit in self.given_digits.items():
            self.set_cell(self.position_to_cell[position], digit)
        # TODO: Initial constraint candidates
        # Actually, nevermind cause then what? It's also using the reductions before it.

    # "Last digit" is a naked single so included in that logic.
    def solve_all_singles(self) -> set[CellPosition]:
        any_singles = True
        changed_positions = set[CellPosition]()
        while any_singles:
            any_singles = False
            # Naked singles
            for cell in self.unknown_cells:
                if len(cell.candidates) == 1:
                    updated_candidates = self.set_cell(cell, next(iter(cell.candidates)))
                    any_singles = True
                    changed_positions.update(updated_candidates)
            # Hidden singles
            for region in self.regions:
                digits_to_cells = defaultdict[int, list[Cell]](list)
                for cell in region.unknown_cells:
                    for candidate in cell.candidates:
                        digits_to_cells[candidate].append(cell)
                for digit, cells in digits_to_cells.items():
                    if len(cells) == 1:
                        updated_candidates = self.set_cell(cells[0], digit)
                        any_singles = True
                        changed_positions.update(updated_candidates)
        return changed_positions

    def check_constraint(self, constraint: "constraints.Constraint") -> bool:
        cells = list(self.positions_to_cells(constraint.affected_cells))
        return constraint.check(cells)

    def update_constraint(self, constraint: "constraints.Constraint") -> bool:
        cells = list(self.positions_to_cells(constraint.affected_cells))
        changed = constraint.update_cell_candidates(cells)
        return len(changed) > 0

    # Might have to change this largely to match the way sudoku.coach does it. For now it's fine.
    def solve_logically(self):
        self.initial_candidates()
        while self.is_unsolved():
            changed_with_singles = self.solve_all_singles()
            constraints_to_check = set["constraints.Constraint"].union(
                *(set(self.cell_position_to_constraints[position]) for position in changed_with_singles)
            )
            changed_with_constraints = False
            for constraint in constraints_to_check:
                changed_with_constraints |= self.update_constraint(constraint)
            # Cannot solve logically (yet)
            if not changed_with_constraints:
                break

    def save_to_stack(self, cell_that_gets_set: Cell, digit_it_gets_set_to: int):
        self._bruteforce_stack.append((cell_that_gets_set, digit_it_gets_set_to))
        for cell in self.cells:
            cell.save_to_stack()
        self.set_cell(cell_that_gets_set, digit_it_gets_set_to)
        for constraint in self.constraints:
            constraint.save_to_stack()

    def restore_from_stack(self) -> tuple[Cell, int]:
        for cell in self.cells:
            cell.restore_from_stack()
        for constraint in self.constraints:
            constraint.restore_from_stack()
        return self._bruteforce_stack.pop()

    def bruteforce_solve(self):
        self.initial_candidates()
        while self.is_unsolved():
            changed_positions = self.solve_all_singles()
            # Only try to brute force if solving singles didn't reduce anything
            if len(changed_positions) == 0:
                next_unknown_cell = next(self.unknown_cells)
                attempted_digit = next(iter(next_unknown_cell.candidates))
                self.save_to_stack(next_unknown_cell, attempted_digit)
                changed_positions = self.set_cell(next_unknown_cell, attempted_digit)
            if self.is_impossible(changed_positions):
                if len(self._bruteforce_stack) == 0:
                    print("Bruteforce says no solutions!")
                    break
                attempted_cell, attempted_digit = self.restore_from_stack()
                # Remove and not discard, because it MUST be there or something has gone wrong
                attempted_cell.candidates.remove(attempted_digit)

    def is_unique(self) -> bool:
        with suppress_stdout():
            self.bruteforce_solve()
            # Solved without guessing must be the only possible solution, if it's even valid
            if len(self._bruteforce_stack) == 0:
                return not self.is_impossible()
            # Remove last guess, if it still solves there are multiple solutions
            attempted_cell, attempted_digit = self.restore_from_stack()
            attempted_cell.candidates.remove(attempted_digit)
            self.bruteforce_solve()
        return self.is_impossible()

    # # Needed kind of for resetting the brute force
    # def reset_to_initial_state(self): ...


class RectangularNumberPuzzle(NumberPuzzle):
    def __init__(self, *, puzzle_width: int, puzzle_height: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.puzzle_width = puzzle_width
        self.puzzle_height = puzzle_height
        valid_numbers = set(range_from_1(max(puzzle_width, puzzle_height)))
        self.valid_numbers.update(valid_numbers)
        # All cells
        cells = [
            Cell(CellPosition(x, y), valid_numbers.copy())
            for y, x in itertools.product(range_from_1(puzzle_height), range_from_1(puzzle_width))
        ]
        self.add_cells(*cells)
        self.cell_positions.update(cell.position for cell in cells)
        # Rows
        all_row_cells: list[list[Cell]] = [[] for _ in range_from_1(puzzle_height)]
        for cell in cells:
            all_row_cells[cell.y - 1].append(cell)
        self.add_regions(*(Region(f"Row {y}", row_cells) for y, row_cells in enumerate(all_row_cells, 1)))
        # Columns
        all_column_cells: list[list[Cell]] = [[] for _ in range_from_1(puzzle_width)]
        for cell in cells:
            all_column_cells[cell.x - 1].append(cell)
        self.add_regions(*(Region(f"Column {x}", column_cells) for x, column_cells in enumerate(all_column_cells, 1)))


class SquareNumberPuzzle(RectangularNumberPuzzle):
    def __init__(self, *, puzzle_size: int, **kwargs: Any):
        super().__init__(puzzle_width=puzzle_size, puzzle_height=puzzle_size, **kwargs)


# Automatic regions supported up to grid size 25x25, should be enough…
class Sudoku(SquareNumberPuzzle):
    def __init__(self, *, puzzle_size: int, **kwargs: Any):
        super().__init__(puzzle_size=puzzle_size, **kwargs)
        # Predefined box size
        if puzzle_size == 4:
            box_width = box_height = 2
        elif puzzle_size == 9:
            box_width = box_height = 3
        elif puzzle_size == 16:
            box_width = box_height = 4
        elif puzzle_size == 25:
            box_width = box_height = 5
        elif puzzle_size in (2, 3, 5, 7, 11, 13, 17, 19, 23):
            raise ValueError("No default regions for prime size sudokus.")
        elif puzzle_size in (6, 8, 10, 14, 22):
            box_height = 2
            box_width = puzzle_size // 2
        elif puzzle_size in (12, 15, 18, 21):
            box_height = 3
            box_width = puzzle_size // 3
        elif puzzle_size in (20, 24):
            box_height = 4
            box_width = puzzle_size // 4
        else:
            raise NotImplementedError("Automatic regions defined only for grid sizes 4 to 25, excluding primes.")
        # Calculate cell positions for each box
        boxes = [
            {
                CellPosition(x + dx, y + dy)
                for dy, dx in itertools.product(range_from_0(box_height), range_from_0(box_width))
            }
            for y, x in itertools.product(range_from_1(puzzle_size, box_height), range_from_1(puzzle_size, box_width))
        ]
        # Cell to boxes mapping
        all_box_cells: list[list[Cell]] = [[] for _ in boxes]
        for cell in self.cells:
            for box_index, box in enumerate(boxes):
                if cell.position in box:
                    all_box_cells[box_index].append(cell)
                    break
        self.boxes = boxes
        self.add_regions(*(Region(f"Box {x}", box_cells) for x, box_cells in enumerate(all_box_cells, 1)))

    def visualise_boxes_as_string(self) -> str:
        # i'll only do a boundary for now
        # ┌┬┐├┼┤└┴┘─│
        grid = [[" "] * self.puzzle_width for _ in range_from_0(self.puzzle_height)]
        for box_index, box in enumerate(self.boxes, 1):
            str_box_index = str(box_index)
            for cell_position in box:
                grid[cell_position.y - 1][cell_position.x - 1] = str_box_index
        # Oops, ruff makes that code unreadble. It's adding a boundary around the grid
        return (
            f"┌{'─' * self.puzzle_width}┐\n│{'│\n│'.join(''.join(row) for row in grid)}│\n└{'─' * self.puzzle_width}┘"
        )

    def visualise_digits_as_string(self) -> str:
        # i'll only do a boundary for now
        # ┌┬┐├┼┤└┴┘─│
        grid = [[" "] * self.puzzle_width for _ in range_from_0(self.puzzle_height)]
        for cell in self.cells:
            grid[cell.position.y - 1][cell.position.x - 1] = str(cell.value) if cell.value is not None else "◻"
        # Oops, ruff makes that code unreadble. It's adding a boundary around the grid
        return (
            f"┌{'─' * self.puzzle_width}┐\n│{'│\n│'.join(''.join(row) for row in grid)}│\n└{'─' * self.puzzle_width}┘"
        )
