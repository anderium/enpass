from typing import Callable

import pytest

from enpass.constraints import (
    LTGT,
    Arrow,
    CalculatorCage,
    ConsecutiveLine,
    CountingCircles,
    DoubleArrow,
    EquidifferenceLine,
    Even,
    HighDigits_4by4,
    KillerCage,
    KropkiDifference,
    KropkiRatio,
    LocalMaximum,
    LocalMinimum,
    LowDigits_4by4,
    Odd,
    ParityLine,
    PillArrow,
    Quadruple,
    RegionSumLine,
    RenbanLine,
    RomanSum,
    SkyLine,
    Thermometer,
    TugOfWarLine,
    ZipperLine,
)
from enpass.sudoku import Cell, CellPosition, Sudoku


def blank_windoku(given_digits: dict[CellPosition, int] = {}):  # noqa: B006
    sudoku = Sudoku(given_digits=given_digits, puzzle_size=4)
    sudoku.add_region_from_positions("Window", (2, 2), (3, 2), (2, 3), (3, 3))
    return sudoku


def test_LTGT_windoku_patched():
    sudoku = blank_windoku()
    sudoku.add_constraints(
        LTGT(smaller=(2, 2), larger=(2, 1)),
        LTGT(smaller=(3, 2), larger=(3, 1)),
        LTGT(smaller=(3, 2), larger=(4, 2)),
        LTGT(smaller=(2, 3), larger=(1, 3)),
        LTGT(smaller=(2, 3), larger=(2, 4)),
    )
    sudoku.bruteforce_solve()
    assert not sudoku.is_unsolved()
    print(sudoku.visualise_digits_as_string())
    assert sudoku.is_solved_double_check()
    solution_string = "2431132431424213"
    solution_values = [*map(int, solution_string)]
    # assert all(cell.digit == expected_digit for cell, expected_digit in zip(sudoku.cells, solution_values))
    for cell, expected_digit in zip(sudoku.cells, solution_values):
        assert cell.digit == expected_digit


def test_DoubleArrow_windoku_listening_to_music_while_programming():
    sudoku = blank_windoku()
    sudoku.add_constraints(
        DoubleArrow(heads=((1, 2), (3, 2)), line=[(1, 1), (2, 1), (3, 1)]),
        DoubleArrow(heads=((2, 4), (3, 4)), line=[(1, 4), (1, 3), (2, 3), (3, 3)]),
    )
    sudoku.bruteforce_solve()
    assert not sudoku.is_unsolved()
    print(sudoku.visualise_digits_as_string())
    solution_string = "4123324123141432"
    solution_values = [*map(int, solution_string)]
    # assert all(cell.digit == expected_digit for cell, expected_digit in zip(sudoku.cells, solution_values))
    for cell, expected_digit in zip(sudoku.cells, solution_values):
        assert cell.digit == expected_digit


def test_Arrow_windoku_crossbow():
    sudoku = blank_windoku()
    sudoku.add_constraints(
        Arrow(head=(1, 1), line=[(2, 2), (3, 3), (4, 4)]),
        Arrow(head=(4, 2), line=[(3, 3), (2, 4)]),
    )
    sudoku.bruteforce_solve()
    assert not sudoku.is_unsolved()
    print(sudoku.visualise_digits_as_string())
    solution_string = "4312213414233241"
    solution_values = [*map(int, solution_string)]
    # assert all(cell.digit == expected_digit for cell, expected_digit in zip(sudoku.cells, solution_values))
    for cell, expected_digit in zip(sudoku.cells, solution_values):
        assert cell.digit == expected_digit


def test_Thermo_windoku_i_have_a_question():
    sudoku = blank_windoku()
    sudoku.add_constraints(
        Thermometer(line=[(2, 4), (1, 3), (2, 2)]),
        Thermometer(line=[(4, 2), (3, 3)]),
    )
    sudoku.bruteforce_solve()
    assert not sudoku.is_unsolved()
    print(sudoku.visualise_digits_as_string())
    solution_string = "1342243131244213"
    solution_values = [*map(int, solution_string)]
    # assert all(cell.digit == expected_digit for cell, expected_digit in zip(sudoku.cells, solution_values))
    for cell, expected_digit in zip(sudoku.cells, solution_values):
        assert cell.digit == expected_digit


def test_RomanSum_windoku_roman_window():
    sudoku = blank_windoku()
    sudoku.add_constraints(
        RomanSum(cell_one=(1, 1), cell_two=(1, 2), sum=3),
        RomanSum(cell_one=(2, 2), cell_two=(2, 3), sum=4),
        RomanSum(cell_one=(3, 1), cell_two=(3, 2), sum=5),
        RomanSum(cell_one=(2, 4), cell_two=(3, 4), sum=3),
    )
    sudoku.bruteforce_solve()
    assert not sudoku.is_unsolved()
    print(sudoku.visualise_digits_as_string())
    solution_string = "2431132431424213"
    solution_values = [*map(int, solution_string)]
    # assert all(cell.digit == expected_digit for cell, expected_digit in zip(sudoku.cells, solution_values))
    for cell, expected_digit in zip(sudoku.cells, solution_values):
        assert cell.digit == expected_digit


def test_Kropki_windoku_and_two_and_three():
    sudoku = blank_windoku()
    sudoku.add_constraints(
        KropkiDifference(cell_one=(2, 2), cell_two=(3, 2)),
        KropkiDifference(cell_one=(3, 3), cell_two=(4, 3)),
        KropkiDifference(cell_one=(1, 3), cell_two=(1, 4)),
        KropkiRatio(cell_one=(2, 2), cell_two=(2, 3)),
        KropkiRatio(cell_one=(3, 3), cell_two=(3, 4)),
    )
    sudoku.bruteforce_solve()
    print(sudoku.visualise_digits_as_string())
    assert not sudoku.is_unsolved()
    solution_string = "4312123421433421"
    solution_values = [*map(int, solution_string)]
    # assert all(cell.digit == expected_digit for cell, expected_digit in zip(sudoku.cells, solution_values))
    for cell, expected_digit in zip(sudoku.cells, solution_values):
        assert cell.digit == expected_digit


@pytest.mark.xfail(raises=NotImplementedError)
def test_LocalExtrema_windoku_mount_and_valley():
    sudoku = blank_windoku()
    get_cell: Callable[[int, int], Cell] = lambda x, y: sudoku.position_to_cell[CellPosition(x, y)]  # noqa: E731
    sudoku.add_constraints(
        LocalMinimum(cells=[get_cell(2, 1), get_cell(2, 2)]),
        LocalMaximum(cells=[get_cell(2, 3), get_cell(3, 3)]),
    )
    sudoku.bruteforce_solve()
    print(sudoku.visualise_digits_as_string())
    assert not sudoku.is_unsolved()
    solution_string = "4312123421433421"
    solution_values = [*map(int, solution_string)]
    # assert all(cell.digit == expected_digit for cell, expected_digit in zip(sudoku.cells, solution_values))
    for cell, expected_digit in zip(sudoku.cells, solution_values):
        assert cell.digit == expected_digit


def test_KillerCage_windoku_gridlock():
    sudoku = blank_windoku()
    sudoku.add_constraints(
        KillerCage(cells=[(1, 2), (1, 3), (1, 4)], sum=6),
        KillerCage(cells=[(3, 1), (3, 2)], sum=6),
        KillerCage(cells=[(2, 3), (3, 3), (4, 3)], sum=7),
    )
    sudoku.bruteforce_solve()
    print(sudoku.visualise_digits_as_string())
    assert not sudoku.is_unsolved()
    solution_string = "4123234132141432"
    solution_values = [*map(int, solution_string)]
    # assert all(cell.digit == expected_digit for cell, expected_digit in zip(sudoku.cells, solution_values))
    for get_cell, expected_digit in zip(sudoku.cells, solution_values):
        assert get_cell.digit == expected_digit


def test_EvenOddHighLow_windoku_even_less_high_odds():
    sudoku = blank_windoku()
    sudoku.add_constraints(
        Odd(cell=(1, 1)),
        Odd(cell=(1, 4)),
        Even(cell=(2, 1)),
        Even(cell=(2, 4)),
        HighDigits_4by4(cell=(3, 1)),
        LowDigits_4by4(cell=(4, 1)),
    )
    sudoku.bruteforce_solve()
    print(sudoku.visualise_digits_as_string())
    assert not sudoku.is_unsolved()
    solution_string = "1432234141233214"
    solution_values = [*map(int, solution_string)]
    # assert all(cell.digit == expected_digit for cell, expected_digit in zip(sudoku.cells, solution_values))
    for get_cell, expected_digit in zip(sudoku.cells, solution_values):
        assert get_cell.digit == expected_digit


def test_CountingCircles_windoku_just_count():
    sudoku = blank_windoku()
    sudoku.add_constraints(
        CountingCircles(positions=((3, 1), (4, 1), (1, 3), (2, 4), (4, 3), (3, 4))),
    )
    sudoku.bruteforce_solve()
    print(sudoku.visualise_digits_as_string())
    assert not sudoku.is_unsolved()
    solution_string = "2413132431424231"
    solution_values = [*map(int, solution_string)]
    # assert all(cell.digit == expected_digit for cell, expected_digit in zip(sudoku.cells, solution_values))
    for get_cell, expected_digit in zip(sudoku.cells, solution_values):
        assert get_cell.digit == expected_digit


def test_RenbanLine_windoku_one_digit():
    sudoku = blank_windoku(given_digits={CellPosition(4, 1): 1})
    sudoku.add_constraints(
        RenbanLine(positions=[(1, 2), (2, 1), (3, 1)]),
        RenbanLine(positions=[(2, 3), (3, 3)]),
    )
    sudoku.bruteforce_solve()
    print(sudoku.visualise_digits_as_string())
    assert not sudoku.is_unsolved()
    solution_string = "4231134231242413"
    solution_values = [*map(int, solution_string)]
    # assert all(cell.digit == expected_digit for cell, expected_digit in zip(sudoku.cells, solution_values))
    for get_cell, expected_digit in zip(sudoku.cells, solution_values):
        assert get_cell.digit == expected_digit


def test_ZipperLine_windoku_closing_the_zipper():
    sudoku = blank_windoku()
    sudoku.add_constraints(
        ZipperLine(positions=[(1, 1), (2, 2), (2, 3), (3, 4), (4, 3)]),
        RenbanLine(positions=[(1, 4), (2, 3)]),
    )
    sudoku.bruteforce_solve()
    print(sudoku.visualise_digits_as_string())
    assert not sudoku.is_unsolved()
    solution_string = "1342423124133124"
    solution_values = [*map(int, solution_string)]
    # assert all(cell.digit == expected_digit for cell, expected_digit in zip(sudoku.cells, solution_values))
    for get_cell, expected_digit in zip(sudoku.cells, solution_values):
        assert get_cell.digit == expected_digit


def test_Quadruple_windoku_keyring_no_fog():
    sudoku = blank_windoku()
    sudoku.add_constraints(
        Quadruple(top_left=(2, 1), digits=[1, 1, 2, 2]),
        Quadruple(top_left=(3, 2), digits=[2, 2, 4, 4]),
    )
    sudoku.bruteforce_solve()
    print(sudoku.visualise_digits_as_string())
    assert not sudoku.is_unsolved()
    solution_string = "4213312413422431"
    solution_values = [*map(int, solution_string)]
    # assert all(cell.digit == expected_digit for cell, expected_digit in zip(sudoku.cells, solution_values))
    for get_cell, expected_digit in zip(sudoku.cells, solution_values):
        assert get_cell.digit == expected_digit


def test_ConsecutiveLine_windoku_the_right_one():
    sudoku = blank_windoku(given_digits={CellPosition(4, 4): 1})
    sudoku.add_constraints(
        ConsecutiveLine(positions=[(1, 1), (2, 2), (1, 3)]),
        ConsecutiveLine(positions=[(4, 2), (3, 3)]),
    )
    sudoku.bruteforce_solve()
    print(sudoku.visualise_digits_as_string())
    # print(sudoku.constraints[0].check([sudoku.cells[0], sudoku.cells[5], sudoku.cells[8]]))
    assert not sudoku.is_unsolved()
    solution_string = "3124421313422431"
    solution_values = [*map(int, solution_string)]
    # assert all(cell.digit == expected_digit for cell, expected_digit in zip(sudoku.cells, solution_values))
    for get_cell, expected_digit in zip(sudoku.cells, solution_values):
        assert get_cell.digit == expected_digit


def test_RegionSumLine_windoku_umbrella():
    sudoku = blank_windoku()
    sudoku.add_constraints(
        RegionSumLine(segments=[[(1, 3)], [(1, 2), (2, 1)], [(3, 1), (4, 2)]]),
        ConsecutiveLine(positions=[(3, 2), (3, 3), (3, 4)]),
    )
    sudoku.bruteforce_solve()
    print(sudoku.visualise_digits_as_string())
    assert not sudoku.is_unsolved()
    solution_string = "2314142341323241"
    solution_values = [*map(int, solution_string)]
    # assert all(cell.digit == expected_digit for cell, expected_digit in zip(sudoku.cells, solution_values))
    for get_cell, expected_digit in zip(sudoku.cells, solution_values):
        assert get_cell.digit == expected_digit


def test_ParityLine_windoku_umbrella():
    sudoku = blank_windoku()
    sudoku.add_constraints(
        ParityLine(positions=[(1, 3), (2, 4), (3, 4), (4, 4), (4, 3)]),
        KropkiRatio(cell_one=(1, 1), cell_two=(1, 2)),
        KropkiRatio(cell_one=(3, 2), cell_two=(3, 3)),
    )
    sudoku.bruteforce_solve()
    print(sudoku.visualise_digits_as_string())
    assert not sudoku.is_unsolved()
    solution_string = "2143431234211234"
    solution_values = [*map(int, solution_string)]
    # assert all(cell.digit == expected_digit for cell, expected_digit in zip(sudoku.cells, solution_values))
    for get_cell, expected_digit in zip(sudoku.cells, solution_values):
        assert get_cell.digit == expected_digit


def test_CalculatorCage_windoku_windows_calculator():
    sudoku = blank_windoku()
    sudoku.add_constraints(
        CalculatorCage(cell_one=(1, 1), cell_two=(1, 2), result=2),
        CalculatorCage(cell_one=(3, 1), cell_two=(4, 1), result=6),
        CalculatorCage(cell_one=(1, 4), cell_two=(2, 4), result=6),
        CalculatorCage(cell_one=(4, 3), cell_two=(4, 4), result=4),
    )
    sudoku.bruteforce_solve()
    print(sudoku.visualise_digits_as_string())
    assert not sudoku.is_unsolved()
    solution_string = "4132231414233241"
    solution_values = [*map(int, solution_string)]
    # assert all(cell.digit == expected_digit for cell, expected_digit in zip(sudoku.cells, solution_values))
    for get_cell, expected_digit in zip(sudoku.cells, solution_values):
        assert get_cell.digit == expected_digit


def test_PillArrow_windoku_Phistomefels_ring():
    sudoku = Sudoku(given_digits={}, puzzle_size=4)
    sudoku.add_region_from_positions("Window", (1, 1), (4, 1), (1, 4), (4, 4))
    sudoku.add_constraints(
        PillArrow(head=((1, 2), (2, 2)), line=[(3, 2), (2, 3), (1, 4), (2, 4), (3, 4), (4, 4)]),
        KropkiDifference(cell_one=(1, 3), cell_two=(2, 3)),
    )
    sudoku.bruteforce_solve()
    print(sudoku.visualise_digits_as_string())
    assert not sudoku.is_unsolved()
    solution_string = "4231132421433412"
    solution_values = [*map(int, solution_string)]
    # assert all(cell.digit == expected_digit for cell, expected_digit in zip(sudoku.cells, solution_values))
    for get_cell, expected_digit in zip(sudoku.cells, solution_values):
        assert get_cell.digit == expected_digit


def test_EquidifferenceLine_windoku_a_different_one():
    sudoku = Sudoku(given_digits={CellPosition(4, 1): 1}, puzzle_size=4)  # type: ignore
    sudoku.add_region_from_positions("Window", (1, 1), (4, 1), (1, 4), (4, 4))
    sudoku.add_constraints(
        EquidifferenceLine(positions=[(1, 2), (2, 1), (3, 1), (4, 2)]),
        EquidifferenceLine(positions=[(2, 2), (2, 3), (2, 4)]),
        EquidifferenceLine(positions=[(3, 3), (4, 3), (4, 4)]),
    )
    sudoku.bruteforce_solve()
    print(sudoku.visualise_digits_as_string())
    assert not sudoku.is_unsolved()
    solution_string = "3421213412434312"
    solution_values = [*map(int, solution_string)]
    # assert all(cell.digit == expected_digit for cell, expected_digit in zip(sudoku.cells, solution_values))
    for get_cell, expected_digit in zip(sudoku.cells, solution_values):
        assert get_cell.digit == expected_digit


def test_TugOfWarLine_windoku_a_different_one():
    sudoku = Sudoku(given_digits={CellPosition(4, 1): 1}, puzzle_size=4)  # type: ignore
    sudoku.add_region_from_positions("Window", (1, 1), (4, 1), (1, 4), (4, 4))
    sudoku.add_constraints(
        TugOfWarLine(positions=[(3, 1), (3, 2), (4, 2), (4, 3), (4, 4), (3, 3), (2, 2), (2, 3), (2, 4), (1, 4)])
    )
    sudoku.bruteforce_solve()
    print(sudoku.visualise_digits_as_string())
    assert not sudoku.is_unsolved()
    solution_string = "3241142341322314"
    solution_values = [*map(int, solution_string)]
    # assert all(cell.digit == expected_digit for cell, expected_digit in zip(sudoku.cells, solution_values))
    for get_cell, expected_digit in zip(sudoku.cells, solution_values):
        assert get_cell.digit == expected_digit


def test_SkyLine_windoku_simple_skyline():
    sudoku = Sudoku(given_digits={CellPosition(4, 1): 1}, puzzle_size=4)  # type: ignore
    sudoku.add_region_from_positions("Window", (1, 1), (4, 1), (1, 4), (4, 4))
    sudoku.add_constraints(
        SkyLine(head=(1, 1), line=[(2, 1), (3, 1), (4, 1)]),
        SkyLine(head=(1, 1), line=[(1, 2), (1, 3), (1, 4)]),
        SkyLine(head=(3, 4), line=[(2, 3), (3, 2)]),
        SkyLine(head=(3, 4), line=[(3, 3), (4, 2)]),
    )
    sudoku.bruteforce_solve()
    # assert sudoku.is_unique()
    print(sudoku.visualise_digits_as_string())
    # return
    assert not sudoku.is_unsolved()
    solution_string = ""
    solution_values = [*map(int, solution_string)]
    # assert all(cell.digit == expected_digit for cell, expected_digit in zip(sudoku.cells, solution_values))
    for get_cell, expected_digit in zip(sudoku.cells, solution_values):
        assert get_cell.digit == expected_digit
