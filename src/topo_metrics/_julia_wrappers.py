from topo_metrics.paths import RingStatistics as RS


def run_rings(edges, depth):
    return RS.run_rings(edges, depth)


def run_strong_rings(edges, depth):
    return RS.run_strong_rings(edges, depth)


def get_topological_genome(nodes, edges, cell_lengths, cell_angles):
    return RS.get_topological_genome(nodes.T, edges, cell_lengths, cell_angles)
