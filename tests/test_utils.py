import io
import sys

import pytest

from topo_metrics.utils import (
    hushed,
    min_count,
    to_tuple,
    uniform_repr,
)


def test_hushed():
    """ Test that hushed context manager suppresses stdout and stderr. """

    captured_output = io.StringIO()
    sys.stdout = captured_output  # Redirect stdout to capture print output

    with hushed():
        print("This should not be seen")
        print("Nor this", file=sys.stderr)

    sys.stdout = sys.__stdout__  # Restore stdout

    assert captured_output.getvalue() == ""  # Nothing should be captured


@pytest.mark.parametrize(
    "input_list, expected_output",
    [
        ([1, 2, [3, 4, [5, 6]], 7], (1, 2, (3, 4, (5, 6)), 7)),
        ("string", "string"),  # Non-list input should return as is
        (123, 123),  # Non-list integer should return as is
        ([[[1], 2], 3], (((1,), 2), 3)),  # Nested lists
        ([], ()),  # Empty list should return empty tuple
    ]
)
def test_to_tuple(input_list, expected_output):
    """ Test conversion of lists to tuples, including nested structures. """

    assert to_tuple(input_list) == expected_output


def test_uniform_repr_single_line():
    """ Test uniform_repr for a short single-line output. """

    result = uniform_repr("TestObject", 1, 2, key1="value1", key2="value2")
    expected = 'TestObject(1, 2, key1="value1", key2="value2")'
    assert result == expected


def test_uniform_repr_multiline():
    """ Test uniform_repr for multi-line formatting when exceeding max width."""

    result = uniform_repr(
        "TestObject",
        "a" * 30,
        "b" * 30,
        key1="value1",
        key2="value2",
        max_width=50,
    )
    assert "\n" in result  # Multiline representation should contain newlines
    assert "TestObject(\n" in result # Should start with object name and newline


@pytest.mark.parametrize(
    "input_list, expected_output",
    [
        ([3, 1, 2, 1, 4, 1], (1, 3)),  # Minimum value is 1, occurs 3 times
        ([0, 0, 0], (0, 3)),  # All elements are the minimum
        ([5], (5, 1)),  # Single-element list
        ([-2, -2, -1, 0], (-2, 2)),  # Negative numbers
    ]
)
def test_min_count(input_list, expected_output):
    """ Test min_count function with various lists. """

    assert min_count(input_list) == expected_output


def test_min_count_empty():
    """ Test that min_count raises an error on an empty list. """

    with pytest.raises(ValueError, match="The list X cannot be empty"):
        min_count([])

