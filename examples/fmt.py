def header(example_name: str) -> None:
    """ Print a header for the example. """

    print("\n" + "-" * len(example_name))
    print(example_name)
    print("-" * len(example_name), end="\n\n")


def footer() -> None:
    """ Print a footer for the example. """

    print("\n" + "-" * 15)
    print("End of analysis")
    print("-" * 15, end="\n\n")