topo_metrics.utils
==================

.. py:module:: topo_metrics.utils


Classes
-------

.. autoapisummary::

   topo_metrics.utils.hushed


Functions
---------

.. autoapisummary::

   topo_metrics.utils.to_tuple
   topo_metrics.utils.uniform_repr
   topo_metrics.utils.min_count


Module Contents
---------------

.. py:class:: hushed

   A context manager to suppress stdout and stderr.

   .. rubric:: Notes

   - I have made this as aggressive as possible to suppress all warnings and
   logging messages. This is because the function `instantiate_julia`brings in
   a lot of noise that irritates me.


.. py:function:: to_tuple(not_tuple: list | Any) -> Any

   Recursively converts a list (or nested lists) to a tuple.

   This function is designed to handle lists that may contain nested lists,
   and it will recursively convert all levels of lists into tuples.

   :param not_tuple: The input to be converted. If the input is a list, it will be
                     recursively converted to a tuple. Otherwise, it will be returned as-is.

   :returns: * *A tuple equivalent of the input list, or the original element if it is not a*
             * *list.*

   .. rubric:: Example

   >>> totuple([1, 2, [3, 4, [5, 6]], 7])
   (1, 2, (3, 4, (5, 6)), 7)

   >>> totuple('string')
   'string'


.. py:function:: uniform_repr(object_name: str, *positional_args: Any, max_width: int = 60, stringify: bool = True, indent_size: int = 2, **keyword_args: Any) -> str

   Generates a uniform string representation of an object, supporting both
   positional and keyword arguments.


.. py:function:: min_count(x: list[int]) -> tuple[int, int]

   Count the number of times the minimum value appears in the list `X`.


