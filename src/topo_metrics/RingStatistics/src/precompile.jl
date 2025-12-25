using PrecompileTools
using .RingStatistics


@setup_workload begin

    # ---------------------------------------------------------- #
    # We precompile the main routines using a sodalite structure #
    # ---------------------------------------------------------- #

    # atomic positions.
    positions = [
        0.0  0.75  0.5;
        0.0  0.5   0.75;
        0.75 0.5   0.0;
        0.5  0.75  0.0;
        0.0  0.25  0.5;
        0.0  0.5   0.25;
        0.25 0.5   0.0;
        0.5  0.25  0.0;
        0.5  0.0   0.25;
        0.25 0.0   0.5;
        0.5  0.0   0.75;
        0.75 0.0   0.5
    ]'

    # neighbour list.
    edges_list = [
        6  7   0  0   0;
        4  11  0  1  -1;
        2  7   0  0   1;
        1  2   0  0   0;
        5  10  0  0   0;
        8  11  0  0  -1;
        9  10  0  0   0;
        11 12  0  0   0;
        5  6   0  0   0;
        3  4   0  0   0;
        5  12 -1  0   0;
        2  5   0  0   0;
        10 11  0  0   0;
        1  10  0  1   0;
        3  6   1  0   0;
        1  6   0  0   0;
        3  8   0  0   0;
        9  12  0  0   0;
        7  8   0  0   0;
        4  7   0  0   0;
        2  3  -1  0   1;
        1  12 -1  1   0;
        8  9   0  0   0;
        4  9   0  1   0
    ]

    # unit cell.
    cell_lengths = (8.965, 8.965, 8.965)
    cell_angles = (90.0, 90.0, 90.0)

    @compile_workload begin
        
        RingStatistics.run_rings(edges_list)
        RingStatistics.get_all_rings(edges_list)
        RingStatistics.run_strong_rings(edges_list)
        RingStatistics.run_coordination_sequences(edges_list, 20)
        RingStatistics.get_topological_genome(
            positions, 
            edges_list, 
            cell_lengths, 
            cell_angles
        )

    end

end