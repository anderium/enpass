# import pytest

from variant_sudoku.sudoku import (
    Cell,
    CellPosition,
    NumberPuzzle,
    RectangularNumberPuzzle,
    Region,
    SquareNumberPuzzle,
    Sudoku,
)


def test_NumberPuzzle_add_cells():
    puzzle = NumberPuzzle(given_digits={CellPosition(2, 1): 2})
    puzzle.add_cells(
        Cell(CellPosition(1, 1), {1, 2}),
        cell_with_given_digit := Cell(CellPosition(2, 1), {2, 3}),
        Cell(CellPosition(1, 2), {3, 4}),
    )
    assert len(puzzle.cells) == 3
    assert cell_with_given_digit.given
    assert cell_with_given_digit.candidates == {2}


def test_NumberPuzzle_unknown_cells():
    puzzle = NumberPuzzle(given_digits={CellPosition(2, 1): 2})
    puzzle.add_cells(
        Cell(CellPosition(1, 1), {1, 2}),
        cell_with_given_digit := Cell(CellPosition(2, 1), {2, 3}),
        Cell(CellPosition(1, 2), {3, 4}),
    )
    unknown = list(puzzle.unknown_cells)
    assert cell_with_given_digit not in unknown
    assert len(unknown) == 2


def test_NumberPuzzle_add_regions():
    puzzle = NumberPuzzle(given_digits={})
    region0 = Region("test region index 0", [Cell(CellPosition(1, 1), {1, 2}), Cell(CellPosition(2, 1), {1, 2})])
    region1 = Region("test region index 1", [Cell(CellPosition(1, 1), {1, 2}), Cell(CellPosition(1, 2), {1, 2})])
    puzzle.add_regions(region0, region1)
    assert puzzle.cell_position_to_regions[CellPosition(1, 1)] == [0, 1]
    assert puzzle.cell_position_to_regions[CellPosition(1, 2)] == [1]


def test_NumberPuzzle_add_region_from_positions():
    puzzle = NumberPuzzle(given_digits={})
    puzzle.add_cells(
        Cell(CellPosition(1, 1), {1, 2}),
        two_one := Cell(CellPosition(2, 1), {2, 3}),
        one_two := Cell(CellPosition(1, 2), {3, 4}),
    )
    puzzle.add_region_from_positions("tuples", (1, 1), (2, 1))
    puzzle.add_region_from_positions("CellPositions", CellPosition(1, 1), CellPosition(1, 2))
    assert two_one in puzzle.regions[0]
    assert len(puzzle.regions[0]) == 2
    assert one_two in puzzle.regions[1]
    assert len(puzzle.regions[0]) == 2


def test_NumberPuzzle_save_to_and_restore_from_stack():
    puzzle = NumberPuzzle(given_digits={})
    puzzle.add_cells(
        one_one := Cell(CellPosition(1, 1), {1, 2, 3, 4}),
        two_one := Cell(CellPosition(2, 1), {1, 2, 3, 4}),
        one_two := Cell(CellPosition(1, 2), {1, 2, 3, 4}),
        two_two := Cell(CellPosition(2, 2), {1, 2, 3, 4}),
    )
    puzzle.add_region_from_positions("Row 1", (1, 1), (2, 1))
    puzzle.add_region_from_positions("Row 2", (1, 2), (2, 2))
    puzzle.add_region_from_positions("Col 1", (1, 1), (1, 2))
    puzzle.add_region_from_positions("Col 2", (2, 1), (2, 2))

    # Single save restore
    puzzle.set_cell(one_one, 1)
    assert 1 not in two_one.candidates
    assert one_one.known

    puzzle.save_to_stack(two_one, 2)
    assert one_one.known
    assert two_one.known
    assert two_one.value == 2
    assert 2 not in two_two.candidates

    assert (two_one, 2) == puzzle.restore_from_stack()
    assert one_one.known
    assert not two_one.known
    assert two_one.candidates == {2, 3, 4}
    assert 2 in two_two.candidates

    # Multiple save restore
    puzzle.save_to_stack(two_one, 2)
    puzzle.save_to_stack(two_two, 3)
    assert two_one.known
    assert two_one.value == 2
    assert two_two.known
    assert two_two.value == 3
    assert 3 not in one_two.candidates

    assert (two_two, 3) == puzzle.restore_from_stack()
    assert two_one.known
    assert two_one.value == 2
    assert not two_two.known
    assert 2 not in two_two.candidates
    assert 3 in two_two.candidates
    assert 3 in one_two.candidates

    assert (two_one, 2) == puzzle.restore_from_stack()
    assert not two_one.known
    assert two_one.candidates == {2, 3, 4}
    assert 2 in two_two.candidates


def test_RectangularNumberPuzzle():
    puzzle = RectangularNumberPuzzle(given_digits={}, puzzle_width=5, puzzle_height=3)
    # fmt: off
    assert puzzle.cell_positions == {
        CellPosition(1, 1), CellPosition(2, 1), CellPosition(3, 1), CellPosition(4, 1), CellPosition(5, 1),
        CellPosition(1, 2), CellPosition(2, 2), CellPosition(3, 2), CellPosition(4, 2), CellPosition(5, 2),
        CellPosition(1, 3), CellPosition(2, 3), CellPosition(3, 3), CellPosition(4, 3), CellPosition(5, 3),
    }
    # TODO: Make it not depend on the order of the regions?
    assert [region.cell_positions for region in puzzle.regions] == [
        # rows
        {CellPosition(1, 1), CellPosition(2, 1), CellPosition(3, 1), CellPosition(4, 1), CellPosition(5, 1)},
        {CellPosition(1, 2), CellPosition(2, 2), CellPosition(3, 2), CellPosition(4, 2), CellPosition(5, 2)},
        {CellPosition(1, 3), CellPosition(2, 3), CellPosition(3, 3), CellPosition(4, 3), CellPosition(5, 3)},
        # columns
        {CellPosition(1, 1), CellPosition(1, 2), CellPosition(1, 3)},
        {CellPosition(2, 1), CellPosition(2, 2), CellPosition(2, 3)},
        {CellPosition(3, 1), CellPosition(3, 2), CellPosition(3, 3)},
        {CellPosition(4, 1), CellPosition(4, 2), CellPosition(4, 3)},
        {CellPosition(5, 1), CellPosition(5, 2), CellPosition(5, 3)},
    ]
    # fmt: on


def test_SquareNumberPuzzle():
    puzzle = SquareNumberPuzzle(given_digits={}, puzzle_size=4)
    # fmt: off
    assert puzzle.cell_positions == {
        CellPosition(1, 1), CellPosition(2, 1), CellPosition(3, 1), CellPosition(4, 1),
        CellPosition(1, 2), CellPosition(2, 2), CellPosition(3, 2), CellPosition(4, 2),
        CellPosition(1, 3), CellPosition(2, 3), CellPosition(3, 3), CellPosition(4, 3),
        CellPosition(1, 4), CellPosition(2, 4), CellPosition(3, 4), CellPosition(4, 4),
    }
    # TODO: Make it not depend on the order of the regions?
    assert [region.cell_positions for region in puzzle.regions] == [
        # rwos
        {CellPosition(1, 1), CellPosition(2, 1), CellPosition(3, 1), CellPosition(4, 1)},
        {CellPosition(1, 2), CellPosition(2, 2), CellPosition(3, 2), CellPosition(4, 2)},
        {CellPosition(1, 3), CellPosition(2, 3), CellPosition(3, 3), CellPosition(4, 3)},
        {CellPosition(1, 4), CellPosition(2, 4), CellPosition(3, 4), CellPosition(4, 4)},
        # columns
        {CellPosition(1, 1), CellPosition(1, 2), CellPosition(1, 3), CellPosition(1, 4)},
        {CellPosition(2, 1), CellPosition(2, 2), CellPosition(2, 3), CellPosition(2, 4)},
        {CellPosition(3, 1), CellPosition(3, 2), CellPosition(3, 3), CellPosition(3, 4)},
        {CellPosition(4, 1), CellPosition(4, 2), CellPosition(4, 3), CellPosition(4, 4)},
    ]
    # fmt: on
    # Digits
    assert puzzle.valid_numbers == {1, 2, 3, 4}


def test_Sudoku():
    puzzle = Sudoku(given_digits={}, puzzle_size=4)
    # fmt: off
    assert puzzle.cell_positions == {
        CellPosition(1, 1), CellPosition(2, 1), CellPosition(3, 1), CellPosition(4, 1),
        CellPosition(1, 2), CellPosition(2, 2), CellPosition(3, 2), CellPosition(4, 2),
        CellPosition(1, 3), CellPosition(2, 3), CellPosition(3, 3), CellPosition(4, 3),
        CellPosition(1, 4), CellPosition(2, 4), CellPosition(3, 4), CellPosition(4, 4),
    }
    # TODO: Make it not depend on the order of the regions?
    assert [region.cell_positions for region in puzzle.regions] == [
        # rows
        {CellPosition(1, 1), CellPosition(2, 1), CellPosition(3, 1), CellPosition(4, 1)},
        {CellPosition(1, 2), CellPosition(2, 2), CellPosition(3, 2), CellPosition(4, 2)},
        {CellPosition(1, 3), CellPosition(2, 3), CellPosition(3, 3), CellPosition(4, 3)},
        {CellPosition(1, 4), CellPosition(2, 4), CellPosition(3, 4), CellPosition(4, 4)},
        # columns
        {CellPosition(1, 1), CellPosition(1, 2), CellPosition(1, 3), CellPosition(1, 4)},
        {CellPosition(2, 1), CellPosition(2, 2), CellPosition(2, 3), CellPosition(2, 4)},
        {CellPosition(3, 1), CellPosition(3, 2), CellPosition(3, 3), CellPosition(3, 4)},
        {CellPosition(4, 1), CellPosition(4, 2), CellPosition(4, 3), CellPosition(4, 4)},
        # boxes
        {CellPosition(1, 1), CellPosition(1, 2), CellPosition(2, 1), CellPosition(2, 2)},
        {CellPosition(3, 1), CellPosition(3, 2), CellPosition(4, 1), CellPosition(4, 2)},
        {CellPosition(1, 3), CellPosition(1, 4), CellPosition(2, 3), CellPosition(2, 4)},
        {CellPosition(3, 3), CellPosition(3, 4), CellPosition(4, 3), CellPosition(4, 4)},
    ]
    # fmt: on
    # Digits
    assert puzzle.valid_numbers == {1, 2, 3, 4}
    assert len(puzzle.cell_position_to_regions[CellPosition(1, 1)]) == 3


# def test_validate_regions(): ...
