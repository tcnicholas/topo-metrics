from pathlib import Path

from topo_metrics import Topology

from .fmt import footer, header


def main():
    """ Quick analysis of the alpha-cristobalite (dia) structure. """

    header("Alpha-cristobalite structure analysis")

    # load the alpha-cristobalite structure from a .cgd file.
    structure_file = Path(__file__).parent / "data/alpha-cristobalite.cgd"
    topology = Topology.from_cgd(structure_file)
    print("This network contains:\n\n", topology, end="\n\n")

    # check topology.
    topology_code = topology.get_topological_genome()
    print("It has topology code:\n\n", topology_code.strip(), end="\n\n")

    # check rings.
    rings = topology.get_rings(depth=6)
    print(
        "This the summary of the rings analysis:\n\n", 
        rings, 
        end="\n\n"
    )

    # print the ring sizes.
    print("The ring sizes are:\n")
    print(f"{'Size':>5} | {'Count':>5}")
    print("-" * 13)
    for size, count in rings.ring_size_count:
        print(f"{size:>4}  | {count:>3} ")

    footer()

if __name__ == "__main__":
    main()