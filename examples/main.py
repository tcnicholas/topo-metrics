import argparse  # noqa: I001
from pathlib import Path

from topo_metrics import Topology

from fmt import footer, header, options  # noqa: I001


def main(struct_name: str):
    """Quick analysis of the alpha-cristobalite (dia) structure."""

    header(f"{struct_name} structure analysis")

    # load the alpha-cristobalite structure from a .cgd file.
    structure_file = Path(__file__).parent / f"data/{struct_name}.cgd"
    topology = Topology.from_cgd(structure_file)
    print("This network contains:\n\n", topology, end="\n\n")

    # check topology if not too large (i.e., don't apply to amorphous networks).
    if len(topology.nodes) < 400:
        topology_code = topology.get_topological_genome()
        print("It has topology code:\n\n", topology_code.strip(), end="\n\n")

    # check rings.
    rings = topology.get_rings(depth=6)
    print("This the summary of the rings analysis:\n\n", rings, end="\n\n")

    # print the ring sizes.
    print("The ring sizes are:\n")
    print("Ring Size | Count")
    print("----------|------")
    for size, count in rings.ring_size_count:
        if count > 0:
            print(f"{size:^9} | {count:^5}")

    footer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "struct_name",
        help="The name of the structure to analyse.",
        choices=options(),
    )
    args = parser.parse_args()
    main(args.struct_name)
