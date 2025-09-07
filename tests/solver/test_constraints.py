import pytest

from enpass.constraints import LocalMaximum, LocalMinimum, RegionSumLine
from enpass.sudoku import CellPosition, Sudoku


@pytest.mark.xfail(raises=NotImplementedError)
def test_LocalExtemum_affected_cells():
    sudoku = Sudoku(puzzle_size=4, given_digits={})
    one_one = sudoku.cells[0]
    constraint = LocalMinimum(cells=[one_one])
    assert constraint.neighbours == {CellPosition(2, 1), CellPosition(1, 2)}
    three_two = sudoku.cells[6]
    three_three = sudoku.cells[10]
    constraint2 = LocalMaximum(cells=[three_two, three_three])
    assert constraint2.neighbours == {
        CellPosition(3, 1),
        CellPosition(3, 4),
        CellPosition(2, 2),
        CellPosition(4, 2),
        CellPosition(2, 3),
        CellPosition(4, 3),
    }


@pytest.mark.xfail(raises=NotImplementedError)
def test_RegionSumLine_from_positions():
    sudoku = Sudoku(puzzle_size=4, given_digits={})
    regions = [region for region in sudoku.regions if region.label.startswith("Box ")]
    # TODO: For jigsaw puzzles this should also work, but I have to be careful with the concave corner crossings.
    # regions = [region for region in sudoku.regions if region.label.startswith("Jigsaw ")]
    # TODO: What about when some cells aren't in the regions, e.g. because they're outside skyscraper clues or because
    #       the regions are extra regions that don't fill the grid.
    constraint = RegionSumLine.from_positions(positions=[(1, 1), (2, 2), (3, 3), (4, 4)], regions=regions)
    assert constraint.segment_positions == [[(1, 1), (2, 2)], [(3, 3), (4, 4)]]
