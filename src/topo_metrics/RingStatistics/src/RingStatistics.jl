module RingStatistics

    using Graphs, PeriodicGraphs, PeriodicGraphEmbeddings, CrystalNets
    using LinearAlgebra, StaticArrays, Base.Threads

    include("Knots.jl")
    using .Knots
    export Knots

    export greet
    export run_bfgs
    export get_topology
    export run_rings, run_strong_rings
    export run_coordination_sequences
    export run_bond_distance_rdf

    """
        import_check()
    """
    function import_check()
        println("RingStatistics is accessible!")
    end

    """
        run_bfgs(edges, v, depth)

    nv(g::PeriodicGraph3D)

    Calculate the number of nodes in the given periodic graph.
    """
    function nv(g::PeriodicGraph3D)
        length(g.nlist)
    end

    """
        ne(g::PeriodicGraph3D)

    Calculate the number of edges in the given periodic graph.
    """
    function ne(g::PeriodicGraph3D)
        g.ne[]
    end

    """
        graph_from_edges(edges)

    Construct a periodic graph from a given set of edges.
    """
    function graph_from_edges(edges)
        pedges = [
            PeriodicEdge3D(edge[1], edge[2], (edge[3],edge[4],edge[5])) 
            for edge in eachrow(edges)
        ]
        PeriodicGraph3D(pedges)
    end

    """
    count_rings(graph, depth)

    Count the number of rings up to a specified depth in the given graph.
    If a ring larger than `depth` is found, a warning is issued (once per ring 
    size) and the ring is ignored. Returns a two-dimensional array where the 
    first column is the ring size and the second column is the ring count.
    """
    function count_rings(graph, depth)
        rcount = zeros(Int, 2*depth+3)
        warned_sizes = Set()
        for r in sort!(first(graph))
            ring_size = length(r)
            if ring_size > 2*depth+3
                if ring_size ∉ warned_sizes
                    @warn "Ring size $ring_size exceeds the specified depth $depth. This ring will be ignored."
                    push!(warned_sizes, ring_size)
                end
                continue
            end
            rcount[ring_size] += 1
        end
        return hcat(1:length(rcount), rcount)
    end

    """
        gather_ring_nodes(graph)

    Gather the ring nodes for each node in the graph. Returns two lists:
    1. The number of rings each node is a part of.
    2. The rings associated with each node.
    """
    function gather_ring_nodes(graph, depth)
        ras = RingAttributions(graph, false, depth)
        num_rings_per_node = length.(ras.attrs)
        rings_per_node = [collect_rings(node) for node in ras]
        num_rings_per_node, rings_per_node
    end

    """
    gather_ring_nodes(graph)

    Gather the ring nodes for each node in the graph. Returns two lists:
    1. The number of rings each node is a part of.
    2. The rings associated with each node.
    """
    function gather_strong_ring_nodes(graph, depth)
        ras = RingAttributions(graph, true, depth)
        num_rings_per_node = length.(ras.attrs)
        rings_per_node = [collect_rings(node) for node in ras]
        num_rings_per_node, rings_per_node
    end

    """
        collect_rings(node)

    Helper function to collect rings for a given node.
    """
    function collect_rings(node)
        [collect_ring(ring) for ring in node]
    end

    """
        collect_ring(ring)

    Helper function to collect vertices for a given ring.
    """
    function collect_ring(ring)
        [(vertex.v, vertex.ofs) for vertex in ring]
    end

    """
        run_rings(edges)

    Main function to run the ring statistics computation. Takes in the edges,
    constructs a graph, and computes ring statistics and ring nodes.
    """
    function run_rings(edges, depth=12)
        g = graph_from_edges(edges)
        rcount = count_rings(rings(g, depth), depth)
        nrnodes, rnodes = gather_ring_nodes(g, depth)
        rcount, nrnodes, rnodes
    end

    function canonical_ring_key(seq)
        n = length(seq)
        norm(rot) = begin
            b = rot[1][2]
            Tuple((v, (ofs[1]-b[1], ofs[2]-b[2], ofs[3]-b[3])) for (v, ofs) in rot)
        end
        best = nothing
        for k in 0:(n-1)
            rot = [seq[mod1(i+k, n)] for i in 1:n]
            for cand in (rot, reverse(rot))
                key = norm(cand)
                if best === nothing || key < best
                    best = key
                end
            end
        end
        return best
    end


    """
    get_all_rings(edges, depth)

    Retrieve all rings up to a specified depth in the given graph. Returns a 
    list of rings, where each ring is represented as a list of tuples containing 
    the vertex index and offset vector.
    """
    function get_all_rings(edges, depth=12)
        g = graph_from_edges(edges)
        ras = RingAttributions(g, false, depth)
        seen = Set{Any}()
        rings_list = Vector{Vector{Tuple{Int,Tuple{Int,Int,Int}}}}()
        for node in ras
            for ring in node
                ring_size = length(ring)
                ring_size > 2*depth + 3 && continue
                seq = collect_ring(ring)
                key = canonical_ring_key(seq)
                if !(key in seen)
                    push!(seen, key)
                    push!(rings_list, collect(key))
                end
            end
        end
        return rings_list
    end

    """
        run_strong_rings(edges)

    Main function to run the ring statistics computation. Takes in the edges,
    constructs a graph, and computes ring statistics and ring nodes.
    """
    function run_strong_rings(edges, depth=12)
        g = graph_from_edges(edges)
        rcount = count_rings(strong_rings(g, depth), depth)
        nrnodes, rnodes = gather_strong_ring_nodes(g, depth)
        rcount, nrnodes, rnodes
    end

    function run_coordination_sequences(edges, dmax)
        g = graph_from_edges(edges)
        num_vertices = nv(g)
        coord_sequences = []
        for v in 1:num_vertices
            seq = coordination_sequence(g, v, dmax)
            push!(coord_sequences, seq)
        end
        return coord_sequences
    end

    function run_bfs(edges, v::Int, depth::Int)
        g = graph_from_edges(edges)
        return node_indices_and_offsets(g, v, depth)
    end

    """
        node_indices_and_offsets(g::PeriodicGraph, v::Integer, dmax)

    For a given vertex `v` in a periodic graph `g`, compute the node indices
    and offsets for nodes at each distance up to `dmax`.

    Returns a vector where each entry corresponds to distance `d`, and contains
    a list of tuples `(node_index, offset_vector)` for nodes at that distance.
    """
    function node_indices_and_offsets(g::PeriodicGraph, v::Integer, dmax)
        Q = Graphs._neighborhood(g, v, dmax, weights(g), outneighbors)
        popfirst!(Q)
        result = [Vector{Tuple{Int, Tuple{Int, Int, Int}}}() for _ in 1:dmax]
        for (node, distance) in Q
            if distance <= dmax
                offset_tuple = Tuple(node.ofs)
                push!(result[distance], (node.v, offset_tuple))
            end
        end
        return result
    end

    """ 
    get_topological_genome(nodes, edges, cell)

    Construct a periodic graph embedding from a given set of nodes, edges, and 
    cell, and then determine the topology.
    """
    function get_topological_genome(
        nodes,         # N×3 array
        edges,         # M×5 array
        cell_lengths,  # length-3 vector (a, b, c)
        cell_angles    # length-3 vector (α, β, γ)
    )

        CrystalNets.toggle_export(false); 
        CrystalNets.toggle_warning(false);
        
        cell_lengths = collect(cell_lengths)
        cell_angles = collect(cell_angles)
    
        @assert size(nodes,1) == 3 "Nodes must have 3 coordinates per point."
        @assert size(edges,2) == 5 "Each edge should be a row with 5 entries."
        @assert length(cell_lengths) == 3 "Lengths must be a 3-element vector."
        @assert length(cell_angles) == 3 "Angles must be a 3-element vector."
    
        graph = graph_from_edges(edges)
        cell = Cell(1, cell_lengths, cell_angles)
        embedding = PeriodicGraphEmbedding3D(graph, nodes, cell)
        topology, _ = only(topological_genome(embedding.g))

        return string(topology)

    end

    @inline function _as_3xN(nodes)
        X = Array{Float64}(nodes)
        if size(X,1) == 3
            return X
        elseif size(X,2) == 3
            return permutedims(X)  # N×3 -> 3×N
        else
            error("nodes must be 3×N or N×3; got size(nodes) = $(size(X))")
        end
    end

    @inline function _shell_volumes(dr::Float64, nbins::Int)
        vols = Vector{Float64}(undef, nbins)
        @inbounds for k in 1:nbins
            r1 = (k-1) * dr
            r2 = k * dr
            vols[k] = (4.0/3.0) * pi * (r2^3 - r1^3)
        end
        return vols
    end

    @inline function _pair_index(α::Int, β::Int, ntypes::Int)
        if α > β
            α, β = β, α
        end
        m = α - 1
        base = m * (ntypes + 1) - (m * (m + 1)) ÷ 2
        return base + (β - α + 1)
    end

    function _type_mapping(node_types, N::Int)
        if node_types === nothing
            types = ["X"]
            type_id = ones(Int, N)
            Ntype = [N]
            return types, type_id, Ntype
        end

        @assert length(node_types) == N "node_types must have length N (n_nodes)."

        types = String[]
        type_id = Vector{Int}(undef, N)
        map = Dict{String,Int}()

        for i in 1:N
            t = String(node_types[i])
            id = get!(map, t) do
                push!(types, t)
                length(types)
            end
            type_id[i] = id
        end

        ntypes = length(types)
        Ntype = zeros(Int, ntypes)
        @inbounds for id in type_id
            Ntype[id] += 1
        end

        return types, type_id, Ntype
    end

    function run_bond_distance_rdf(
        nodes,
        edges,
        cell::Cell;
        dmax::Int = 6,
        rmax::Float64 = 10.0,
        dr::Float64 = 0.02,
        normalise::Bool = true,
        node_types = nothing,
        partial::Bool = false,
    )
        g = graph_from_edges(edges)

        X = _as_3xN(nodes)
        N = size(X, 2)
        N < 2 && error("Need at least 2 nodes.")

        # Cell matrix (triclinic-safe): columns are lattice vectors in Cartesian
        A = SMatrix{3,3,Float64}(cell.mat)
        V = abs(det(A))
        V == 0.0 && error("Cell volume is zero/degenerate (det(cell.mat)=0).")
        ρ = N / V

        nbins = Int(floor(rmax / dr))
        nbins < 1 && error("rmax/dr gives nbins < 1. Increase rmax or decrease dr.")
        rcenters = ((0:nbins-1) .+ 0.5) .* dr
        ΔV = _shell_volumes(dr, nbins)

        # Types (optional)
        types, type_id, Ntype = _type_mapping(node_types, N)
        ntypes = length(types)
        ρtype = [Ntype[t] / V for t in 1:ntypes]

        nt = nthreads()

        if !partial
            # counts[D, k]
            counts_tls = [zeros(Int, dmax, nbins) for _ in 1:nt]

            @threads for i in 1:N
                counts = counts_tls[threadid()]
                ri = @SVector [X[1,i], X[2,i], X[3,i]]

                Q = Graphs._neighborhood(g, i, dmax, weights(g), outneighbors)
                popfirst!(Q)

                @inbounds for (pv, D) in Q
                    (D < 1 || D > dmax) && continue
                    j = pv.v
                    ofs = pv.ofs

                    rj = @SVector [X[1,j], X[2,j], X[3,j]]
                    rj_img = rj + A * SVector{3,Float64}(Tuple(ofs))

                    dist = norm(rj_img - ri)
                    if dist < rmax
                        k = Int(floor(dist / dr)) + 1
                        (1 <= k <= nbins) && (counts[D, k] += 1)
                    end
                end
            end

            counts = zeros(Int, dmax, nbins)
            for t in 1:nt
                counts .+= counts_tls[t]
            end

            gD = Array{Float64}(undef, dmax, nbins)
            if normalise
                @inbounds for k in 1:nbins
                    denom = N * ρ * ΔV[k]
                    invden = denom == 0 ? NaN : 1.0 / denom
                    for D in 1:dmax
                        gD[D, k] = counts[D, k] * invden
                    end
                end
            else
                @inbounds for D in 1:dmax, k in 1:nbins
                    gD[D, k] = counts[D, k]
                end
            end

            gtot = vec(sum(gD, dims=1))
            return rcenters, gD, gtot

        else
            # Tidy pair list: (α<=β)
            npairs = ntypes * (ntypes + 1) ÷ 2
            pairs = Vector{Tuple{String,String}}(undef, npairs)
            p = 0
            @inbounds for α in 1:ntypes
                for β in α:ntypes
                    p += 1
                    pairs[p] = (types[α], types[β])
                end
            end

            # counts[pair, D, k] with unordered pair indexing
            counts_tls = [zeros(Int, npairs, dmax, nbins) for _ in 1:nt]

            @threads for i in 1:N
                counts = counts_tls[threadid()]
                ti = type_id[i]
                ri = @SVector [X[1,i], X[2,i], X[3,i]]

                Q = Graphs._neighborhood(g, i, dmax, weights(g), outneighbors)
                popfirst!(Q)

                @inbounds for (pv, D) in Q
                    (D < 1 || D > dmax) && continue
                    j = pv.v
                    tj = type_id[j]
                    pair = _pair_index(ti, tj, ntypes)

                    ofs = pv.ofs
                    rj = @SVector [X[1,j], X[2,j], X[3,j]]
                    rj_img = rj + A * SVector{3,Float64}(Tuple(ofs))

                    dist = norm(rj_img - ri)
                    if dist < rmax
                        k = Int(floor(dist / dr)) + 1
                        if 1 <= k <= nbins
                            counts[pair, D, k] += 1
                        end
                    end
                end
            end

            counts = zeros(Int, npairs, dmax, nbins)
            for t in 1:nt
                counts .+= counts_tls[t]
            end

            gD_pairs = Array{Float64}(undef, npairs, dmax, nbins)

            if normalise
                # For α≠β we aggregated BOTH directions into one unordered pair,
                # so divide by 2 to return the symmetrized (undirected) partial RDF.
                p = 0
                @inbounds for α in 1:ntypes
                    Nα = Ntype[α]
                    for β in α:ntypes
                        p += 1
                        ρβ = ρtype[β]
                        factor = (α == β) ? 1.0 : 2.0
                        for k in 1:nbins
                            denom = factor * Nα * ρβ * ΔV[k]
                            invden = denom == 0 ? NaN : 1.0 / denom
                            for D in 1:dmax
                                gD_pairs[p, D, k] = counts[p, D, k] * invden
                            end
                        end
                    end
                end
            else
                @inbounds for p in 1:npairs, D in 1:dmax, k in 1:nbins
                    gD_pairs[p, D, k] = counts[p, D, k]
                end
            end

            gtot_pairs = dropdims(sum(gD_pairs, dims=2), dims=2)  # (npairs, nbins)
            return rcenters, pairs, gD_pairs, gtot_pairs
        end
    end

    function run_bond_distance_rdf(
        nodes,
        edges,
        cell_lengths,
        cell_angles;
        dmax::Int = 6,
        rmax::Float64 = 10.0,
        dr::Float64 = 0.02,
        normalise::Bool = true,
        node_types = nothing,
        partial::Bool = false,
    )
        cell = Cell(1, collect(cell_lengths), collect(cell_angles))
        return run_bond_distance_rdf(nodes, edges, cell;
            dmax=dmax, rmax=rmax, dr=dr, normalise=normalise,
            node_types=node_types, partial=partial
        )
    end

    include("precompile.jl")

end # module RingStatistics
