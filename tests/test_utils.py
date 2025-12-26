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
    """Test that hushed context manager suppresses stdout and stderr."""

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
    ],
)
def test_to_tuple(input_list, expected_output):
    """Test conversion of lists to tuples, including nested structures."""

    assert to_tuple(input_list) == expected_output


def test_uniform_repr_single_line():
    """Test uniform_repr for a short single-line output."""

    result = uniform_repr("TestObject", 1, 2, key1="value1", key2="value2")
    expected = 'TestObject(1, 2, key1="value1", key2="value2")'
    assert result == expected


def test_uniform_repr_multiline():
    """Test uniform_repr for multi-line formatting when exceeding max width."""

    result = uniform_repr(
        "TestObject",
        "a" * 30,
        "b" * 30,
        key1="value1",
        key2="value2",
        max_width=50,
    )
    assert "\n" in result  # Multiline representation should contain newlines
    assert (
        "TestObject(\n" in result
    )  # Should start with object name and newline


@pytest.mark.parametrize(
    "input_list, expected_output",
    [
        ([3, 1, 2, 1, 4, 1], (1, 3)),  # Minimum value is 1, occurs 3 times
        ([0, 0, 0], (0, 3)),  # All elements are the minimum
        ([5], (5, 1)),  # Single-element list
        ([-2, -2, -1, 0], (-2, 2)),  # Negative numbers
    ],
)
def test_min_count(input_list, expected_output):
    """Test min_count function with various lists."""

    assert min_count(input_list) == expected_output


def test_min_count_empty():
    """Test that min_count raises an error on an empty list."""

    with pytest.raises(ValueError, match="The list X cannot be empty"):
        min_count([])


def test_uniform_repr_no_stringify():
    """Test uniform_repr with stringify=False."""

    result = uniform_repr("TestObject", "value", key="test", stringify=False)
    # Without stringify, strings should not be quoted
    assert result == "TestObject(value, key=test)"


def test_uniform_repr_custom_indent():
    """Test uniform_repr with custom indent size."""

    result = uniform_repr(
        "TestObject",
        "a" * 30,
        "b" * 30,
        max_width=50,
        indent_size=4,
    )
    # Check that indentation is correct (4 spaces)
    lines = result.split("\n")
    if len(lines) > 1:
        assert lines[1].startswith("    ")


def test_uniform_repr_newline_in_value():
    """Test uniform_repr with newline characters in values."""

    result = uniform_repr(
        "TestObject",
        "line1\nline2",
        max_width=10,
    )
    # Should trigger multiline mode due to newline
    assert "\n" in result
    assert "TestObject(\n" in result


def test_uniform_repr_only_positional():
    """Test uniform_repr with only positional arguments."""

    result = uniform_repr("TestObject", 1, 2, 3)
    assert result == "TestObject(1, 2, 3)"


def test_uniform_repr_only_keyword():
    """Test uniform_repr with only keyword arguments."""

    result = uniform_repr("TestObject", key1="val1", key2="val2")
    assert result == 'TestObject(key1="val1", key2="val2")'


def test_uniform_repr_integer_values():
    """Test uniform_repr with integer values (no stringify)."""

    result = uniform_repr("TestObject", 1, 2, key=3)
    assert result == "TestObject(1, 2, key=3)"


def test_to_tuple_generator():
    """Test to_tuple with a generator."""

    gen = (x for x in [1, 2, 3])
    result = to_tuple(gen)
    assert result == (1, 2, 3)


def test_to_tuple_dict():
    """Test to_tuple with a dict (iterable but not list)."""

    # Dict iteration gives keys
    result = to_tuple({"a": 1, "b": 2})
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_to_tuple_mixed_types():
    """Test to_tuple with mixed nested types."""

    input_data = [1, "string", [2, 3], {"a": 1}]
    result = to_tuple(input_data)
    assert isinstance(result, tuple)
    assert result[0] == 1
    assert result[1] == "string"
    assert result[2] == (2, 3)
