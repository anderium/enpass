import itertools
import operator
from abc import ABCMeta, abstractmethod
from collections import Counter
from math import prod
from typing import Any, ClassVar, Generator, Iterable, Literal, Self, overload, override

import variant_sudoku.sudoku as sudoku

type CellPosition = "sudoku.CellPosition"
type AnyPosition = CellPosition | tuple[int, int]


# Overloads so typing hints are nicer
@overload
def to_CellPositions(single_elem_tuple: tuple[AnyPosition], /) -> tuple[CellPosition]: ...
@overload
def to_CellPositions(two_elem_tuple: tuple[AnyPosition, AnyPosition], /) -> tuple[CellPosition, CellPosition]: ...
@overload
def to_CellPositions(
    three_elem_tuple: tuple[AnyPosition, AnyPosition, AnyPosition], /
) -> tuple[CellPosition, CellPosition, CellPosition]: ...
@overload
def to_CellPositions(
    four_elem_tuple: tuple[AnyPosition, AnyPosition, AnyPosition, AnyPosition], /
) -> tuple[CellPosition, CellPosition, CellPosition, CellPosition]: ...
@overload
def to_CellPositions(
    five_elem_tuple: tuple[AnyPosition, AnyPosition, AnyPosition, AnyPosition, AnyPosition], /
) -> tuple[CellPosition, CellPosition, CellPosition, CellPosition, CellPosition]: ...
# @overload
# def to_CellPositions(any_length_tuple: tuple[AnyPosition, ...], /) -> tuple[CellPosition, ...]: ...
@overload
def to_CellPositions(list: Iterable[AnyPosition], /) -> tuple[CellPosition, ...]: ...
# Implementation
def to_CellPositions(tuple_or_list: Iterable[AnyPosition], /) -> tuple[CellPosition, ...]:
    return tuple(sudoku.CellPosition(*position) for position in tuple_or_list)


class Constraint(metaclass=ABCMeta):
    constraint_name: ClassVar[str]
    persistent_attributes: ClassVar[tuple[str, ...]] = ()
    """IMMUTABLE attributes that track information to make the check go quicker,
    but which should be reset after the bruteforce guess is reset."""

    def __init__(self, *, affected_cells: list[CellPosition], **kwargs: Any):
        self.affected_cells = affected_cells
        self._bruteforce_stack: list[Any] = []

    @abstractmethod
    def update_cell_candidates(self, cells: list["sudoku.Cell"]) -> list["sudoku.Cell"]:
        """Update the cell.candidates for each cell.

        Parameters
        ----------
        cells : list[sudoku.Cell]
            The cell instances for the affected cells. The order of the
            list should match the order of `self.affected_cells`.

        Returns
        -------
        list[sudoku.Cell]
            Returns all cells which had their candidates updated.
        """
        # TODO: On the other hand, letting the solver code handle checking if cells have been updated might be fine?
        ...

    @abstractmethod
    def check(self, cells: list["sudoku.Cell"]) -> bool:
        """Return whether the cells in their current form can satisfy this constraint.

        If not all cells are not known, return whether the constraint
        can be satisfied looking naively at the candidates.

        Parameters
        ----------
        cells : list[sudoku.Cell]
            The cell instances for the affected cells. The order of the
            list should match the order of `self.affected_cells`.

        Returns
        -------
        bool
            Do the values of the cell provide a way to satisfy the constraint?
        """
        ...

    def save_to_stack(self):
        backup = tuple(getattr(self, attr) for attr in self.persistent_attributes)
        self._bruteforce_stack.append(backup)

    def restore_from_stack(self):
        backup = self._bruteforce_stack.pop()
        for attr, backup_value in zip(self.persistent_attributes, backup):
            setattr(self, attr, backup_value)

    def __hash__(self) -> int:
        return hash(id(self))


class LTGT(Constraint):
    constraint_name = "Less/greater than"
    persistent_attributes = ("smaller_minimum", "larger_maximum")

    def __init__(
        self,
        *,
        smaller: AnyPosition,
        larger: AnyPosition,
        **kwargs: Any,
    ):
        smaller, larger = to_CellPositions((smaller, larger))
        super().__init__(affected_cells=[smaller, larger], **kwargs)
        self.smaller = smaller
        self.larger = larger
        self.smaller_minimum: int | None = None
        self.larger_maximum: int | None = None

    @override
    def update_cell_candidates(self, cells: list["sudoku.Cell"]) -> list["sudoku.Cell"]:
        smaller_cell, larger_cell = cells

        larger_cell_maximum = max(larger_cell.candidates)
        changed_smaller = False
        candidates_to_remove = set[int]()
        for candidate in smaller_cell.candidates:
            if candidate >= larger_cell_maximum:
                # smaller_cell.candidates.remove(candidate)
                candidates_to_remove.add(candidate)
                changed_smaller = True
        smaller_cell.candidates.difference_update(candidates_to_remove)

        smaller_cell_minimum = min(smaller_cell.candidates)
        changed_larger = False
        candidates_to_remove = set[int]()
        for candidate in larger_cell.candidates:
            if candidate <= smaller_cell_minimum:
                # larger_cell.candidates.remove(candidate)
                candidates_to_remove.add(candidate)
                changed_larger = True
        larger_cell.candidates.difference_update(candidates_to_remove)

        changed_cells: list[sudoku.Cell] = []
        if changed_smaller:
            changed_cells.append(smaller_cell)
        if changed_larger:
            changed_cells.append(larger_cell)
        return changed_cells

    @override
    def check(self, cells: list["sudoku.Cell"]) -> bool:
        smaller, larger = cells
        return smaller.minimum < larger.maximum


class DoubleArrow(Constraint):
    constraint_name = "Double Arrow"

    def __init__(
        self,
        *,
        heads: tuple[AnyPosition, AnyPosition],
        line: list[AnyPosition],
        **kwargs: Any,
    ):
        heads_tuple: tuple[CellPosition, CellPosition] = to_CellPositions(heads)
        line_tuple: tuple[CellPosition, ...] = to_CellPositions(line)
        super().__init__(affected_cells=[*heads_tuple, *line_tuple], **kwargs)
        self.heads = heads_tuple
        self.line = line_tuple

    @override
    def update_cell_candidates(self, cells: list["sudoku.Cell"]) -> list["sudoku.Cell"]:
        head_cell_1, head_cell_2, *line = cells

        # TODO: Cells that see each other obviously increase the minimum and maximum.
        # This is a common operation, so I should add it to the Cell class anyway?
        # Then cells should track the other cells they see?

        minimum_head_sum = head_cell_1.minimum + head_cell_2.minimum
        maximum_head_sum = head_cell_1.maximum + head_cell_2.maximum

        minimum_line_sum = sum(cell.minimum for cell in line)
        maximum_line_sum = sum(cell.maximum for cell in line)

        _ = minimum_head_sum + maximum_head_sum + minimum_line_sum + maximum_line_sum

        return []

        # smaller_cell_minimum = min(smaller_cell.candidates)
        # changed_larger = False
        # for candidate in larger_cell.candidates:
        #     if candidate <= smaller_cell_minimum:
        #         larger_cell.candidates.remove(candidate)
        #         changed_larger = True
        #
        # changed_cells: list[sudoku.Cell] = []
        # if changed_smaller:
        #     changed_cells.append(smaller_cell)
        # if changed_larger:
        #     changed_cells.append(larger_cell)
        # return changed_cells

    def check(self, cells: list["sudoku.Cell"]) -> bool:
        head_cell_1, head_cell_2, *line = cells
        head_minimum = head_cell_1.minimum + head_cell_2.minimum
        head_maximum = head_cell_1.maximum + head_cell_2.maximum
        line_minimum = sum(cell.minimum for cell in line)
        line_maximum = sum(cell.maximum for cell in line)
        # True:  [()] . ([]) . ([)] . [(])
        # False: []() . ()[]
        return head_minimum <= line_minimum <= head_maximum or line_minimum <= head_minimum <= line_maximum


class Arrow(Constraint):
    constraint_name = "Arrow"

    def __init__(
        self,
        *,
        head: AnyPosition,
        line: list[AnyPosition],
        **kwargs: Any,
    ):
        (head_position,) = to_CellPositions((head,))
        line_tuple: tuple[CellPosition, ...] = to_CellPositions(line)
        super().__init__(affected_cells=[head_position, *line_tuple], **kwargs)
        self.head = head_position
        self.line = line_tuple

    @override
    def update_cell_candidates(self, cells: list["sudoku.Cell"]) -> list["sudoku.Cell"]:
        head, *line = cells

        # TODO: Cells that see each other obviously increase the minimum and maximum.
        # This is a common operation, so I should add it to the Cell class anyway?
        # Then cells should track the other cells they see?

        minimum_head = head.minimum
        maximum_head = head.maximum

        minimum_line_sum = sum(cell.minimum for cell in line)
        maximum_line_sum = sum(cell.maximum for cell in line)

        _ = minimum_head + maximum_head + minimum_line_sum + maximum_line_sum

        return []

    def check(self, cells: list["sudoku.Cell"]) -> bool:
        head, *line = cells
        line_minimum = sum(cell.minimum for cell in line)
        line_maximum = sum(cell.maximum for cell in line)
        # True:  [()] . ([]) . ([)] . [(])
        # False: []() . ()[]
        return head.minimum <= line_minimum <= head.maximum or line_minimum <= head.minimum <= line_maximum


class Thermometer(Constraint):
    constraint_name = "Thermometer"

    def __init__(
        self,
        *,
        line: list[AnyPosition],
        **kwargs: Any,
    ):
        line_tuple: tuple[CellPosition, ...] = to_CellPositions(line)
        super().__init__(affected_cells=[*line_tuple], **kwargs)
        self.line = line_tuple

    @override
    def update_cell_candidates(self, cells: list["sudoku.Cell"]) -> list["sudoku.Cell"]:
        changed: set[CellPosition] = set()

        for cell_low, cell_high in itertools.pairwise(cells):
            if cell_high.discard_less_than_equal(cell_low.minimum):
                changed.add(cell_high.position)

        for cell_high, cell_low in itertools.pairwise(reversed(cells)):
            if cell_low.discard_greater_than_equal(cell_high.maximum):
                changed.add(cell_low.position)

        return [cell for cell in cells if cell.position in changed]

    def check(self, cells: list["sudoku.Cell"]) -> bool:
        return all(cell_low.minimum < cell_high.maximum for cell_low, cell_high in itertools.pairwise(cells))


class RomanSum(Constraint):
    constraint_name = "Roman Sum"

    def __init__(
        self,
        *,
        cell_one: AnyPosition,
        cell_two: AnyPosition,
        sum: int,
        **kwargs: Any,
    ):
        cell_one, cell_two = to_CellPositions((cell_one, cell_two))
        super().__init__(affected_cells=[cell_one, cell_two], **kwargs)
        # self.cells = cell_one, cell_two
        self.sum = sum

    @override
    def update_cell_candidates(self, cells: list["sudoku.Cell"]) -> list["sudoku.Cell"]:
        changed: list[sudoku.Cell] = []

        cell_one, cell_two = cells

        cell_two_options = {self.sum - candidate for candidate in cell_one.candidates}
        cell_one_options = {self.sum - candidate for candidate in cell_two.candidates}

        cell_one_prev_size = len(cell_one.candidates)
        cell_one.candidates.intersection_update(cell_one_options)
        if len(cell_one.candidates) < cell_one_prev_size:
            changed.append(cell_one)

        cell_two_prev_size = len(cell_two.candidates)
        cell_two.candidates.intersection_update(cell_two_options)
        if len(cell_two.candidates) < cell_two_prev_size:
            changed.append(cell_two)

        return changed

    def check(self, cells: list["sudoku.Cell"]) -> bool:
        cell_one, cell_two = cells
        return cell_one.minimum + cell_two.minimum <= self.sum <= cell_one.maximum + cell_two.maximum


class KropkiDifference(Constraint):
    constraint_name = "Kropki Difference"

    def __init__(
        self,
        *,
        cell_one: AnyPosition,
        cell_two: AnyPosition,
        difference: int = 1,
        **kwargs: Any,
    ):
        cell_one, cell_two = to_CellPositions((cell_one, cell_two))
        super().__init__(affected_cells=[cell_one, cell_two], **kwargs)
        # self.cells = cell_one, cell_two
        self.difference = difference

    @override
    def update_cell_candidates(self, cells: list["sudoku.Cell"]) -> list["sudoku.Cell"]:
        changed: list[sudoku.Cell] = []

        cell_one, cell_two = cells

        cell_two_options = set[int].union(
            *({candidate + self.difference, candidate - self.difference} for candidate in cell_one.candidates)
        )
        cell_one_options = set[int].union(
            *({candidate + self.difference, candidate - self.difference} for candidate in cell_two.candidates)
        )

        cell_one_prev_size = len(cell_one.candidates)
        cell_one.candidates.intersection_update(cell_one_options)
        if len(cell_one.candidates) < cell_one_prev_size:
            changed.append(cell_one)

        cell_two_prev_size = len(cell_two.candidates)
        cell_two.candidates.intersection_update(cell_two_options)
        if len(cell_two.candidates) < cell_two_prev_size:
            changed.append(cell_two)

        return changed

    def check(self, cells: list["sudoku.Cell"]) -> bool:
        cell_one, cell_two = cells
        cell_two_options = set[int].union(
            *({candidate + self.difference, candidate - self.difference} for candidate in cell_one.candidates)
        )
        possibilities = cell_two.candidates & cell_two_options
        return len(possibilities) > 0


class KropkiRatio(Constraint):
    constraint_name = "Kropki Ratio"

    def __init__(
        self,
        *,
        cell_one: AnyPosition,
        cell_two: AnyPosition,
        ratio: int = 2,
        **kwargs: Any,
    ):
        cell_one, cell_two = to_CellPositions((cell_one, cell_two))
        super().__init__(affected_cells=[cell_one, cell_two], **kwargs)
        # self.cells = cell_one, cell_two
        self.ratio = ratio

    # TODO: Support things that make non-integer values possible e.g. 3/2*4 = 6
    def double_and_half_if_whole(self, candidate: int):
        options: set[int] = {candidate * self.ratio}
        if candidate / self.ratio == candidate // self.ratio:
            options.add(candidate // self.ratio)
        return options

    @override
    def update_cell_candidates(self, cells: list["sudoku.Cell"]) -> list["sudoku.Cell"]:
        changed: list[sudoku.Cell] = []

        cell_one, cell_two = cells

        cell_two_options = set[int].union(
            *(self.double_and_half_if_whole(candidate) for candidate in cell_one.candidates)
        )
        cell_one_options = set[int].union(
            *(self.double_and_half_if_whole(candidate) for candidate in cell_two.candidates)
        )

        cell_one_prev_size = len(cell_one.candidates)
        cell_one.candidates.intersection_update(cell_one_options)
        if len(cell_one.candidates) < cell_one_prev_size:
            changed.append(cell_one)

        cell_two_prev_size = len(cell_two.candidates)
        cell_two.candidates.intersection_update(cell_two_options)
        if len(cell_two.candidates) < cell_two_prev_size:
            changed.append(cell_two)

        return changed

    def check(self, cells: list["sudoku.Cell"]) -> bool:
        cell_one, cell_two = cells
        cell_two_options = set[int].union(
            *(self.double_and_half_if_whole(candidate) for candidate in cell_one.candidates)
        )
        possibilities = cell_two.candidates & cell_two_options
        return len(possibilities) > 0


class _LocalExtremum(Constraint):
    constraint_name = "Local Extremum"

    def __init__(
        self,
        *,
        cells: list["sudoku.Cell"],
        **kwargs: Any,
    ):
        cells_tuple = [cell.position for cell in cells]
        neighbours: set[CellPosition] = set[CellPosition].union(
            *({neighbour.position for neighbour in cell.orthogonally_adjacent_neighbours} for cell in cells)
        )
        super().__init__(affected_cells=[*cells_tuple, *neighbours], **kwargs)
        self.cells = set(cells_tuple)
        self.neighbours = neighbours

    @staticmethod
    @abstractmethod
    def comparison(neighbour: "sudoku.Cell", cell: "sudoku.Cell") -> bool: ...

    @override
    def update_cell_candidates(self, cells: list["sudoku.Cell"]) -> list["sudoku.Cell"]:
        changed: list[sudoku.Cell] = []

        return changed

    def check(self, cells: list["sudoku.Cell"]) -> bool:
        cells, _neighbours = cells[: len(self.cells)], cells[len(self.cells) :]
        for cell in cells:
            for neighbour in cell.orthogonally_adjacent_neighbours:
                if neighbour.position not in self.cells and self.comparison(neighbour, cell):
                    return False
        return True


class LocalMinimum(_LocalExtremum):
    constraint_name = "Local Minimum"

    @staticmethod
    @override
    def comparison(neighbour: "sudoku.Cell", cell: "sudoku.Cell") -> bool:
        return neighbour.minimum < cell.maximum


class LocalMaximum(_LocalExtremum):
    constraint_name = "Local Maximum"

    @staticmethod
    @override
    def comparison(neighbour: "sudoku.Cell", cell: "sudoku.Cell") -> bool:
        return neighbour.maximum > cell.minimum


class KillerCage(Constraint):
    constraint_name = "Killer Cage"

    def __init__(
        self,
        *,
        cells: list[AnyPosition],
        sum: int,
        **kwargs: Any,
    ):
        cells_tuple = to_CellPositions(cells)
        super().__init__(affected_cells=[*cells_tuple], **kwargs)
        # self.cells = cell_one, cell_two
        self.sum = sum

    @override
    def update_cell_candidates(self, cells: list["sudoku.Cell"]) -> list["sudoku.Cell"]:
        changed: list[sudoku.Cell] = []

        return changed

    def check(self, cells: list["sudoku.Cell"]) -> bool:
        digits = set[int]()
        for cell in cells:
            # Not using known because there may be doubler cells in the future
            if cell.digit is None:
                break
            if cell.digit in digits:
                return False
            digits.add(cell.digit)
        return sum(cell.minimum for cell in cells) <= self.sum <= sum(cell.maximum for cell in cells)


class LimitedValues(Constraint):
    constraint_name = "Limited Values"
    persistent_attributes = ("_candidates_eliminated",)

    def __init__(
        self,
        *,
        cell: AnyPosition,
        allowed_values: set[int],
        **kwargs: Any,
    ):
        (cell_pos,) = to_CellPositions((cell,))
        super().__init__(affected_cells=[cell_pos], **kwargs)
        self.cell = cell
        self.allowed_values = allowed_values
        self._candidates_eliminated = False

    @override
    def update_cell_candidates(self, cells: list["sudoku.Cell"]) -> list["sudoku.Cell"]:
        if self._candidates_eliminated:
            return []
        self._candidates_eliminated = True
        (cell,) = cells
        if cell.discard_other_values(self.allowed_values):
            return [cell]
        return []

    def check(self, cells: list["sudoku.Cell"]) -> bool:
        return len(cells[0].candidates & self.allowed_values) > 0


class _InfiniteLimitedValues(Constraint):
    constraint_name = "Limited Values"
    persistent_attributes = ("_candidates_eliminated",)

    def __init__(
        self,
        *,
        cell: AnyPosition,
        **kwargs: Any,
    ):
        (cell_pos,) = to_CellPositions((cell,))
        super().__init__(affected_cells=[cell_pos], **kwargs)
        self.cell = cell
        self._candidates_eliminated = False

    @staticmethod
    @abstractmethod
    def allowed(value: int) -> bool: ...

    @override
    def update_cell_candidates(self, cells: list["sudoku.Cell"]) -> list["sudoku.Cell"]:
        if self._candidates_eliminated:
            return []
        self._candidates_eliminated = True
        (cell,) = cells
        disallowed_values = {candidate for candidate in cell.candidates if not self.allowed(candidate)}
        if cell.discard_values(disallowed_values):
            return [cell]
        return []

    def check(self, cells: list["sudoku.Cell"]) -> bool:
        return any(self.allowed(candidate) for candidate in cells[0].candidates)


class Even(_InfiniteLimitedValues):
    constraint_name = "Even"

    @staticmethod
    @override
    def allowed(value: int) -> bool:
        return value % 2 == 0


class Odd(_InfiniteLimitedValues):
    constraint_name = "Odd"

    @staticmethod
    @override
    def allowed(value: int) -> bool:
        return value % 2 == 1


class LowDigits_4by4(LimitedValues):
    constraint_name = "Low Digits in 4×4"  # noqa: RUF001

    def __init__(self, *, cell: AnyPosition, **kwargs: Any):
        super().__init__(cell=cell, allowed_values={1, 2}, **kwargs)


class HighDigits_4by4(LimitedValues):
    constraint_name = "Low Digits in 4×4"  # noqa: RUF001

    def __init__(self, *, cell: AnyPosition, **kwargs: Any):
        super().__init__(cell=cell, allowed_values={3, 4}, **kwargs)


class CountingCircles(Constraint):
    constraint_name = "Counting Circles"

    def __init__(self, *, positions: tuple[AnyPosition, ...], **kwargs: Any):
        cells_positions = to_CellPositions(positions)
        super().__init__(affected_cells=[*cells_positions], **kwargs)

    def update_cell_candidates(self, cells: list["sudoku.Cell"]) -> list["sudoku.Cell"]:
        return []

    # This is efficient, but it looks like there are things missing.
    # Still, that is not the case because cells cannot be empty in solved sudokus. (Yet…)
    def check(self, cells: list["sudoku.Cell"]) -> bool:
        how_often_do_digits_occur = Counter(cell.digit for cell in cells if cell.digit is not None)
        if sum(how_often_do_digits_occur.keys()) > len(cells):
            return False
        return all(how_often <= digit for digit, how_often in how_often_do_digits_occur.items())


class RenbanLine(Constraint):
    constraint_name = "Renban Line"

    def __init__(self, *, positions: list[AnyPosition], **kwargs: Any):
        cells_positions = to_CellPositions(positions)
        super().__init__(affected_cells=[*cells_positions], **kwargs)
        self.expected_difference = len(positions) - 1

    def update_cell_candidates(self, cells: list["sudoku.Cell"]) -> list["sudoku.Cell"]:
        return []

    # How can I make this more efficient?
    # I guess it additionally qualifies as a partial region?
    def check(self, cells: list["sudoku.Cell"]) -> bool:
        seen = set[int]()
        for cell in cells:
            if cell.value is None:
                continue
            if cell.value in seen:
                return False
            seen.add(cell.value)
        return len(seen) == 0 or max(seen) - min(seen) <= self.expected_difference


class ZipperLine(Constraint):
    constraint_name = "Zipper Line"

    def __init__(self, *, positions: list[AnyPosition], **kwargs: Any):
        cells_positions = to_CellPositions(positions)
        super().__init__(affected_cells=[*cells_positions], **kwargs)
        div, mod = divmod(len(positions), 2)
        has_middle = mod == 1
        self.middle = (True, div) if has_middle else (False, (div - 1, div))

    def update_cell_candidates(self, cells: list["sudoku.Cell"]) -> list["sudoku.Cell"]:
        return []

    # How can I make this more efficient?
    def check(self, cells: list["sudoku.Cell"]) -> bool:
        match self.middle:
            case (True, middle_index):
                sum_minimum = cells[middle_index].minimum
                sum_maximum = cells[middle_index].maximum
                middle_stop_forwards = middle_index
                middle_stop_backwards = middle_index
            case (False, (index1, index2)):
                sum_minimum = cells[index1].minimum + cells[index2].minimum
                sum_maximum = cells[index1].maximum + cells[index2].maximum
                middle_stop_forwards = index1
                middle_stop_backwards = index2
        for cell, opposite in zip(cells[:middle_stop_forwards], cells[-1:middle_stop_backwards:-1]):
            this_minimum = cell.minimum + opposite.minimum
            this_maximum = cell.maximum + opposite.maximum
            sum_minimum = max(sum_minimum, this_minimum)
            sum_maximum = min(sum_maximum, this_maximum)
            if sum_maximum < sum_minimum:
                return False
        return True


class Quadruple(Constraint):
    constraint_name = "Quadruple Circle"

    def __init__(self, *, top_left: AnyPosition, digits: list[int], **kwargs: Any):
        (top_left_position,) = to_CellPositions((top_left,))
        top_right = sudoku.CellPosition(top_left_position.x + 1, top_left_position.y)
        bottom_left = sudoku.CellPosition(top_left_position.x, top_left_position.y + 1)
        bottom_right = sudoku.CellPosition(top_left_position.x + 1, top_left_position.y + 1)
        super().__init__(affected_cells=[top_left_position, top_right, bottom_left, bottom_right], **kwargs)
        self.digits = digits
        self.target_counter = Counter(digits)

    def update_cell_candidates(self, cells: list["sudoku.Cell"]) -> list["sudoku.Cell"]:
        return []

    def check(self, cells: list["sudoku.Cell"]) -> bool:
        total_candidate_counter = Counter(itertools.chain(*(cell.candidates for cell in cells)))
        # return (self.target_counter - total_candidate_counter).most_common(1)[0][1] <= 0
        # idk, apparently this isn't the same as the `subtract` update?
        return len(self.target_counter - total_candidate_counter) == 0


class ConsecutiveLine(Constraint):
    constraint_name = "Consecutive Line"

    def __init__(self, *, positions: list[AnyPosition], **kwargs: Any):
        cells_positions = to_CellPositions(positions)
        super().__init__(affected_cells=[*cells_positions], **kwargs)
        self.expected_difference = len(positions) - 1

    def update_cell_candidates(self, cells: list["sudoku.Cell"]) -> list["sudoku.Cell"]:
        return []

    # How can I make this more efficient?
    # I guess it additionally qualifies as a partial region?
    def check(self, cells: list["sudoku.Cell"]) -> bool:
        valid_ascending_start: int | Literal[False] | None = None
        valid_descending_start: int | Literal[False] | None = None
        for offset, cell in enumerate(cells):
            if cell.value is None:
                continue

            value_of_ascending_start = cell.value - offset
            if valid_ascending_start is None:
                valid_ascending_start = value_of_ascending_start
            # Don't do this, it bugs and doesn't prevent descending also failing
            # elif valid_ascending_start is False:
            #     continue
            elif valid_ascending_start != value_of_ascending_start:
                valid_ascending_start = False

            value_of_descending_start = cell.value + offset
            if valid_descending_start is None:
                valid_descending_start = value_of_descending_start
                valid_ascending_start = value_of_ascending_start
            # elif valid_descending_start is False:
            #     continue
            elif valid_descending_start != value_of_descending_start:
                valid_descending_start = False
        return valid_ascending_start is not False or valid_descending_start is not False


class RegionSumLine(Constraint):
    constraint_name = "Region Sum Line"

    def __init__(self, *, segments: list[list[AnyPosition]], **kwargs: Any):
        segment_positions = [to_CellPositions(segment) for segment in segments]
        super().__init__(affected_cells=list(itertools.chain(*segment_positions)), **kwargs)
        self.segment_positions = segment_positions

    @classmethod
    def from_positions(cls, *, positions: list[AnyPosition], regions: list["sudoku.Region"]) -> Self:
        segments = [positions]
        return cls(segments=segments)

    def cells_to_segments(self, cells: list["sudoku.Cell"]) -> Generator[list["sudoku.Cell"], None, None]:
        cell_iterator = iter(cells)
        for segment in self.segment_positions:
            yield [next(cell_iterator) for _ in range(len(segment))]

    def update_cell_candidates(self, cells: list["sudoku.Cell"]) -> list["sudoku.Cell"]:
        return []

    def check(self, cells: list["sudoku.Cell"]) -> bool:
        sum_minimum = None
        sum_maximum = None
        for segment in self.cells_to_segments(cells):
            segment_minimum = sum(cell.minimum for cell in segment)
            segment_maximum = sum(cell.maximum for cell in segment)
            if sum_minimum is None:
                sum_minimum = segment_minimum
            sum_minimum = max(sum_minimum, segment_minimum)
            if sum_maximum is None:
                sum_maximum = segment_maximum
            sum_maximum = min(sum_maximum, segment_maximum)
            if sum_maximum < sum_minimum:
                return False
        return True


class ParityLine(Constraint):
    constraint_name = "Parity Line"

    def __init__(self, *, positions: list[AnyPosition], **kwargs: Any):
        cell_positions = to_CellPositions(positions)
        super().__init__(affected_cells=[*cell_positions], **kwargs)

    def update_cell_candidates(self, cells: list["sudoku.Cell"]) -> list["sudoku.Cell"]:
        return []

    def check(self, cells: list["sudoku.Cell"]) -> bool:
        parity_matches_cell_index = None
        for cell, is_even_index in zip(cells, itertools.cycle((True, False))):
            if cell.value is None:
                continue
            parity_even = cell.value % 2 == 0
            this_parity_matches_index = parity_even == is_even_index
            if parity_matches_cell_index is None:
                parity_matches_cell_index = this_parity_matches_index
            elif parity_matches_cell_index != this_parity_matches_index:
                return False
        return True


class CalculatorCage(Constraint):
    constraint_name = "Calculator Cage"

    def __init__(
        self,
        *,
        cell_one: AnyPosition,
        cell_two: AnyPosition,
        result: int,
        known_ordered: bool = False,
        **kwargs: Any,
    ):
        cell_one, cell_two = to_CellPositions((cell_one, cell_two))
        super().__init__(affected_cells=[cell_one, cell_two], **kwargs)
        self.result = result
        self.known_ordered = known_ordered

    def update_cell_candidates(self, cells: list["sudoku.Cell"]) -> list["sudoku.Cell"]:
        return []

    def check(self, cells: list["sudoku.Cell"]) -> bool:
        cell_one, cell_two = cells
        # I don't know how to check this efficiently when not all digits are known
        if cell_one.value is None:
            return True
        if cell_two.value is None:
            return True
        operators = [operator.add, operator.sub, operator.mul, operator.truediv]
        if not self.known_ordered:
            operators.append(lambda a, b: b - a)
            operators.append(lambda a, b: b / a)
        return any(op(cell_one.value, cell_two.value) == self.result for op in operators)


# Every arrow is also a pill arrow, but the logic depends on the size of the pill, so i guess this will just be length 2.
class PillArrow(Constraint):
    constraint_name = "Pill Arrow"

    def __init__(
        self,
        *,
        head: tuple[AnyPosition, AnyPosition],
        line: list[AnyPosition],
        **kwargs: Any,
    ):
        head_positions = to_CellPositions(head)
        arrow_positions = to_CellPositions(line)
        super().__init__(affected_cells=[*head_positions, *arrow_positions], **kwargs)
        self.head = head
        self.arrow_positions = arrow_positions

    def update_cell_candidates(self, cells: list["sudoku.Cell"]) -> list["sudoku.Cell"]:
        return []

    def check(self, cells: list["sudoku.Cell"]) -> bool:
        head_cell_one, head_cell_two, *line = cells
        if head_cell_one.value is None:
            return True
        if head_cell_two.value is None:
            return True
        string_of_sum = f"{head_cell_one.value}{head_cell_two.value}"
        target_sum = int(string_of_sum)
        # TODO: This is a function that many constraints use. Though a lot of them use the
        # check that allows partially unfinished sums.
        return has_sum(target_sum, line)


class EquidifferenceLine(Constraint):
    constraint_name = "Equi-difference Line"

    def __init__(
        self,
        *,
        positions: list[AnyPosition],
        **kwargs: Any,
    ):
        positions_tuple = to_CellPositions(positions)
        super().__init__(affected_cells=[*positions_tuple], **kwargs)
        self.positions = positions_tuple

    def update_cell_candidates(self, cells: list["sudoku.Cell"]) -> list["sudoku.Cell"]:
        return []

    def check(self, cells: list["sudoku.Cell"]) -> bool:
        return True


class TugOfWarLine(Constraint):
    constraint_name = "Tug-of-War Line"

    def __init__(
        self,
        *,
        positions: list[AnyPosition],
        **kwargs: Any,
    ):
        positions_tuple = to_CellPositions(positions)
        super().__init__(affected_cells=[*positions_tuple], **kwargs)
        self.positions = positions_tuple

    def update_cell_candidates(self, cells: list["sudoku.Cell"]) -> list["sudoku.Cell"]:
        return []

    def check(self, cells: list["sudoku.Cell"]) -> bool:
        increases_at_even_index = None
        for (cell1, cell2), is_even_index in zip(itertools.pairwise(cells), itertools.cycle((True, False))):
            if cell1.value is None:
                continue
            if cell2.value is None:
                continue
            if cell1.value == cell2.value:
                return False
            increases = cell1.value < cell2.value
            this_increase_matches_index = increases == is_even_index
            if increases_at_even_index is None:
                increases_at_even_index = this_increase_matches_index
            elif increases_at_even_index != this_increase_matches_index:
                return False
        return True


class RegionProductLine(Constraint):
    constraint_name = "Region Product Line"

    def __init__(self, *, segments: list[list[AnyPosition]], **kwargs: Any):
        segment_positions = [to_CellPositions(segment) for segment in segments]
        super().__init__(affected_cells=list(itertools.chain(*segment_positions)), **kwargs)
        self.segment_positions = segment_positions

    @classmethod
    def from_positions(cls, *, positions: list[AnyPosition], regions: list["sudoku.Region"]) -> Self:
        segments = [positions]
        return cls(segments=segments)

    def cells_to_segments(self, cells: list["sudoku.Cell"]) -> Generator[list["sudoku.Cell"], None, None]:
        cell_iterator = iter(cells)
        for segment in self.segment_positions:
            yield [next(cell_iterator) for _ in range(len(segment))]

    def update_cell_candidates(self, cells: list["sudoku.Cell"]) -> list["sudoku.Cell"]:
        return []

    def check(self, cells: list["sudoku.Cell"]) -> bool:
        sum_minimum = None
        sum_maximum = None
        for segment in self.cells_to_segments(cells):
            segment_minimum = prod(cell.minimum for cell in segment)
            segment_maximum = prod(cell.maximum for cell in segment)
            if sum_minimum is None:
                sum_minimum = segment_minimum
            sum_minimum = max(sum_minimum, segment_minimum)
            if sum_maximum is None:
                sum_maximum = segment_maximum
            sum_maximum = min(sum_maximum, segment_maximum)
            if sum_maximum < sum_minimum:
                return False
        return True


# TODO: Name this better
class AnticonsecutiveLine(Constraint):
    """Digits on a red-magenta (cherry?) anticonsecutive line must come from
    a set of consecutive digits, but these digits may not touch.
    """

    constraint_name = "Anticonsecutive Line"

    def __init__(self, *, segments: list[list[AnyPosition]], **kwargs: Any):
        segment_positions = [to_CellPositions(segment) for segment in segments]
        super().__init__(affected_cells=list(itertools.chain(*segment_positions)), **kwargs)
        self.segment_positions = segment_positions

    def update_cell_candidates(self, cells: list["sudoku.Cell"]) -> list["sudoku.Cell"]:
        return []

    def check(self, cells: list["sudoku.Cell"]) -> bool:
        return True


class SkyLine(Constraint):
    constraint_name = "Anticonsecutive Line"

    def __init__(self, *, head: AnyPosition, line: list[AnyPosition], **kwargs: Any):
        (head_position,) = to_CellPositions((head,))
        line_tuple = to_CellPositions(line)
        super().__init__(affected_cells=[head_position, *line_tuple], **kwargs)
        self.head = head_position
        self.line = line_tuple

    def update_cell_candidates(self, cells: list["sudoku.Cell"]) -> list["sudoku.Cell"]:
        # head, line = cells
        return []

    def check(self, cells: list["sudoku.Cell"]) -> bool:
        head, *line = cells
        if (needs_to_see := head.value) is None:
            return True
        seen_at_least = 0
        seen_at_most = 0
        highest_seen = None
        for cell in line:
            if cell.value is None:
                seen_at_most += 1
            elif highest_seen is None or cell.value > highest_seen:
                highest_seen = cell.value
                seen_at_least += 1
                seen_at_most += 1
        return seen_at_least <= needs_to_see <= seen_at_most
