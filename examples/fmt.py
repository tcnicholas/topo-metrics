from pathlib import Path


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


def options() -> list[str]:
    """ Auto-find all .cgd files in the data directory. """

    data_dir = Path(__file__).parent / "data"
    return [f.stem for f in data_dir.glob("*.cgd")]