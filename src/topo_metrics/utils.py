from __future__ import annotations

import logging
import os
import sys
import warnings
from typing import Any, Iterable


class hushed:
    """
    A context manager to suppress stdout and stderr.
    
    Notes
    -----
    - I have made this as aggressive as possible to suppress all warnings and
    logging messages. This is because the function `instantiate_julia`brings in 
    a lot of noise that irritates me.
    """
    
    def __enter__(self):
    
        # Save original file descriptors
        self._stdout_fd = sys.stdout.fileno()
        self._stderr_fd = sys.stderr.fileno()
        self._saved_stdout = os.dup(self._stdout_fd)
        self._saved_stderr = os.dup(self._stderr_fd)

        # Open /dev/null and redirect stdout & stderr
        self._devnull = open(os.devnull, 'w')
        os.dup2(self._devnull.fileno(), self._stdout_fd)
        os.dup2(self._devnull.fileno(), self._stderr_fd)

        # Suppress warnings
        warnings.simplefilter("ignore")

        # Disable logging
        logging.disable(logging.CRITICAL)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
    
        # Restore stdout and stderr
        os.dup2(self._saved_stdout, self._stdout_fd)
        os.dup2(self._saved_stderr, self._stderr_fd)
        os.close(self._saved_stdout)
        os.close(self._saved_stderr)

        # Close /dev/null
        self._devnull.close()

        # Restore warnings and logging
        warnings.resetwarnings()
        logging.disable(logging.NOTSET)


def to_tuple(not_tuple: list | Any) -> tuple:
    """
    Recursively converts a list (or nested lists) to a tuple. 

    This function is designed to handle lists that may contain nested lists, 
    and it will recursively convert all levels of lists into tuples.

    Parameters
    ----------
    not_tuple
        The input to be converted. If the input is a list, it will be 
        recursively converted to a tuple. Otherwise, it will be returned as-is.

    Returns
    -------
    A tuple equivalent of the input list, or the original element if it is not a
    list.

    Example
    -------
    >>> totuple([1, 2, [3, 4, [5, 6]], 7])
    (1, 2, (3, 4, (5, 6)), 7)

    >>> totuple('string')
    'string'
    """

    if isinstance(not_tuple, str):
        return not_tuple
    
    if isinstance(not_tuple, Iterable):
        return tuple(to_tuple(i) for i in not_tuple)
    
    return not_tuple


def uniform_repr(
    object_name: str,
    *positional_args: Any,
    max_width: int = 60,
    stringify: bool = True,
    indent_size: int = 2,
    **keyword_args: Any,
) -> str:
    """
    Generates a uniform string representation of an object, supporting both
    positional and keyword arguments.
    """

    def format_value(value: Any) -> str:
        """
        Converts a value to a string, optionally wrapping strings in quotes.
        """
        if isinstance(value, str) and stringify:
            return f'"{value}"'
        return str(value)

    # Format positional and keyword arguments
    components = [format_value(arg) for arg in positional_args]
    components += [
        f"{key}={format_value(value)}" 
        for key, value in keyword_args.items()
    ]

    # Construct a single-line representation
    single_line_repr = f"{object_name}({', '.join(components)})"
    if len(single_line_repr) < max_width and "\n" not in single_line_repr:
        return single_line_repr

    # If exceeding max width, format as a multi-line representation.
    def indent(text: str) -> str:
        """Indents text with a specified number of spaces."""
        indentation = " " * indent_size
        return "\n".join(f"{indentation}{line}" for line in text.split("\n"))

    # Build multi-line representation
    multi_line_repr = f"{object_name}(\n"
    multi_line_repr += ",\n".join(indent(component) for component in components)
    multi_line_repr += "\n)"

    return multi_line_repr


def min_count(x: list[int]) -> tuple[int, int]:
    """
    Count the number of times the minimum value appears in the list `X`.
    """
    x = list(x)

    if not x:
        raise ValueError("The list X cannot be empty")
        
    min_value = min(x)
    return min_value, x.count(min_value)