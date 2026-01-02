# Knots.jl
# - Gauss linking number (method 1a) with PBC image scanning
#
# Rings are N×3 Float64 matrices of unwrapped Cartesian coordinates.
#
# Exports:
#   drop_duplicate_endpoint, segments_closed
#   linking_number_1a
#   linking_number_pbc_1a
#   all_pairs_linking_pbc_1a

module Knots

using LinearAlgebra
using Base.Threads

export drop_duplicate_endpoint
export segments_closed
export linking_number_1a
export linking_number_pbc_1a
export all_pairs_linking_pbc_1a

# -----------------------------
# Utilities
# -----------------------------

@inline function drop_duplicate_endpoint(P::AbstractMatrix{<:Real}; atol=1e-12)
    n = size(P, 1)
    if n >= 2
        # compare first and last row
        d2 = 0.0
        @inbounds for k in 1:3
            δ = float(P[1,k]) - float(P[n,k])
            d2 += δ*δ
        end
        if d2 <= atol^2
            return Matrix{Float64}(P[1:n-1, :])
        end
    end
    return Matrix{Float64}(P)
end

@inline function centroid(P::AbstractMatrix{<:Real})
    c = zeros(Float64, 3)
    n = size(P,1)
    @inbounds for i in 1:n
        c[1] += float(P[i,1]); c[2] += float(P[i,2]); c[3] += float(P[i,3])
    end
    c ./= n
    return c
end

@inline function segments_closed(P::AbstractMatrix{<:Real})
    # returns (A,B) as m×3 Float64 matrices, with segments A[i,:] -> B[i,:]
    Q = drop_duplicate_endpoint(P)
    n = size(Q,1)
    if n < 2
        return zeros(Float64, 0, 3), zeros(Float64, 0, 3)
    end
    A = Matrix{Float64}(Q)
    B = similar(A)
    @inbounds for i in 1:n-1
        B[i,1] = A[i+1,1]; B[i,2] = A[i+1,2]; B[i,3] = A[i+1,3]
    end
    B[n,1] = A[1,1]; B[n,2] = A[1,2]; B[n,3] = A[1,3]
    return A, B
end

@inline function median_segment_length(P::AbstractMatrix{<:Real})
    A,B = segments_closed(P)
    m = size(A,1)
    if m == 0
        return 0.0
    end
    lens = Vector{Float64}(undef, m)
    @inbounds for i in 1:m
        dx = B[i,1]-A[i,1]; dy = B[i,2]-A[i,2]; dz = B[i,3]-A[i,3]
        lens[i] = sqrt(dx*dx+dy*dy+dz*dz)
    end
    sort!(lens)
    return lens[(m+1)>>>1]
end

@inline function auto_disjoint_tol(A::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real};
                                  rel::Float64=1e-3, abs_::Float64=1e-8)
    m1 = median_segment_length(A)
    m2 = median_segment_length(B)
    m = (m1>0 && m2>0) ? min(m1,m2) : max(m1,m2)
    return max(abs_, rel*max(1.0, m))
end

# -----------------------------
# Scalar math kernels
# -----------------------------

@inline clamp01(x::Float64) = x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x)
@inline clampm11(x::Float64) = x < -1.0 ? -1.0 : (x > 1.0 ? 1.0 : x)

@inline function unit_cross(ax,ay,az, bx,by,bz, eps::Float64)
    cx = ay*bz - az*by
    cy = az*bx - ax*bz
    cz = ax*by - ay*bx
    n2 = cx*cx + cy*cy + cz*cz
    if n2 < eps*eps
        return 0.0,0.0,0.0,false
    end
    invn = inv(sqrt(n2))
    return cx*invn, cy*invn, cz*invn, true
end

@inline function dot3(ax,ay,az, bx,by,bz)
    return ax*bx + ay*by + az*bz
end

@inline function cross3(ax,ay,az, bx,by,bz)
    return (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx)
end

# Gauss pair contribution: method 1a, signed V* (NOT divided by 4π)
@inline function gauss_pair_1a(
    p1x,p1y,p1z, p2x,p2y,p2z,
    p3x,p3y,p3z, p4x,p4y,p4z,
    eps::Float64
)::Float64
    r12x = p2x-p1x; r12y = p2y-p1y; r12z = p2z-p1z
    r34x = p4x-p3x; r34y = p4y-p3y; r34z = p4z-p3z

    r13x = p3x-p1x; r13y = p3y-p1y; r13z = p3z-p1z
    r14x = p4x-p1x; r14y = p4y-p1y; r14z = p4z-p1z
    r23x = p3x-p2x; r23y = p3y-p2y; r23z = p3z-p2z
    r24x = p4x-p2x; r24y = p4y-p2y; r24z = p4z-p2z

    n1x,n1y,n1z,ok1 = unit_cross(r13x,r13y,r13z, r14x,r14y,r14z, eps);  ok1 || return 0.0
    n2x,n2y,n2z,ok2 = unit_cross(r14x,r14y,r14z, r24x,r24y,r24z, eps);  ok2 || return 0.0
    n3x,n3y,n3z,ok3 = unit_cross(r24x,r24y,r24z, r23x,r23y,r23z, eps);  ok3 || return 0.0
    n4x,n4y,n4z,ok4 = unit_cross(r23x,r23y,r23z, r13x,r13y,r13z, eps);  ok4 || return 0.0

    t1 = clampm11(dot3(n1x,n1y,n1z, n2x,n2y,n2z))
    t2 = clampm11(dot3(n2x,n2y,n2z, n3x,n3y,n3z))
    t3 = clampm11(dot3(n3x,n3y,n3z, n4x,n4y,n4z))
    t4 = clampm11(dot3(n4x,n4y,n4z, n1x,n1y,n1z))

    Vstar = asin(t1) + asin(t2) + asin(t3) + asin(t4)

    cx,cy,cz = cross3(r34x,r34y,r34z, r12x,r12y,r12z)
    triple = dot3(cx,cy,cz, r13x,r13y,r13z)
    if triple > 0.0
        return Vstar
    elseif triple < 0.0
        return -Vstar
    else
        return 0.0
    end
end

# Segment-segment distance squared (Ericson-style), robust clamping
@inline function segseg_dist2(
    p1x,p1y,p1z, p2x,p2y,p2z,
    q1x,q1y,q1z, q2x,q2y,q2z
)::Float64
    ux = p2x-p1x; uy = p2y-p1y; uz = p2z-p1z
    vx = q2x-q1x; vy = q2y-q1y; vz = q2z-q1z
    wx = p1x-q1x; wy = p1y-q1y; wz = p1z-q1z

    a = ux*ux + uy*uy + uz*uz
    b = ux*vx + uy*vy + uz*vz
    c = vx*vx + vy*vy + vz*vz
    d = ux*wx + uy*wy + uz*wz
    e = vx*wx + vy*wy + vz*wz
    D = a*c - b*b
    eps = 1e-15

    sN = 0.0; sD = D
    tN = 0.0; tD = D

    if D < eps
        sN = 0.0; sD = 1.0
        tN = e;   tD = c
    else
        sN = (b*e - c*d)
        tN = (a*e - b*d)
        if sN < 0.0
            sN = 0.0
            tN = e
            tD = c
        elseif sN > sD
            sN = sD
            tN = e + b
            tD = c
        end
    end

    if tN < 0.0
        tN = 0.0
        if -d < 0.0
            sN = 0.0
        elseif -d > a
            sN = sD
        else
            sN = -d
            sD = a
        end
    elseif tN > tD
        tN = tD
        if (-d + b) < 0.0
            sN = 0.0
        elseif (-d + b) > a
            sN = sD
        else
            sN = (-d + b)
            sD = a
        end
    end

    sc = abs(sN) < eps ? 0.0 : (sN / sD)
    tc = abs(tN) < eps ? 0.0 : (tN / tD)

    dx = wx + sc*ux - tc*vx
    dy = wy + sc*uy - tc*vy
    dz = wz + sc*uz - tc*vz
    return dx*dx + dy*dy + dz*dz
end

# Min point-point distance squared between A and shifted B (cheap ranking)
function min_point_dist2_shift(A::Matrix{Float64}, B::Matrix{Float64}, sx,sy,sz)
    best = Inf
    na = size(A,1); nb = size(B,1)
    @inbounds for i in 1:na
        ax = A[i,1]; ay = A[i,2]; az = A[i,3]
        for j in 1:nb
            bx = B[j,1]-sx; by = B[j,2]-sy; bz = B[j,3]-sz
            dx = ax-bx; dy = ay-by; dz = az-bz
            d2 = dx*dx + dy*dy + dz*dz
            if d2 < best
                best = d2
            end
        end
    end
    return best
end

# Min segment-segment distance squared between segments (A1->B1) and shifted (A2->B2)
function min_seg_dist2_shift(A1::Matrix{Float64}, B1::Matrix{Float64},
                            A2::Matrix{Float64}, B2::Matrix{Float64},
                            sx,sy,sz;
                            early_exit2::Float64=0.0)
    best = Inf
    m1 = size(A1,1); m2 = size(A2,1)
    @inbounds for i in 1:m1
        p1x = A1[i,1]; p1y = A1[i,2]; p1z = A1[i,3]
        p2x = B1[i,1]; p2y = B1[i,2]; p2z = B1[i,3]
        for j in 1:m2
            q1x = A2[j,1]-sx; q1y = A2[j,2]-sy; q1z = A2[j,3]-sz
            q2x = B2[j,1]-sx; q2y = B2[j,2]-sy; q2z = B2[j,3]-sz
            d2 = segseg_dist2(p1x,p1y,p1z, p2x,p2y,p2z, q1x,q1y,q1z, q2x,q2y,q2z)
            if d2 < best
                best = d2
                if best <= early_exit2
                    return best
                end
            end
        end
    end
    return best
end

# Sum of method-1a Vstar contributions between two rings, with ring2 shifted by -shift
function link_sum_1a_shift(A1::Matrix{Float64}, B1::Matrix{Float64},
                           A2::Matrix{Float64}, B2::Matrix{Float64},
                           sx,sy,sz, eps::Float64)
    sumV = 0.0
    m1 = size(A1,1); m2 = size(A2,1)
    @inbounds for i in 1:m1
        p1x = A1[i,1]; p1y = A1[i,2]; p1z = A1[i,3]
        p2x = B1[i,1]; p2y = B1[i,2]; p2z = B1[i,3]
        for j in 1:m2
            p3x = A2[j,1]-sx; p3y = A2[j,2]-sy; p3z = A2[j,3]-sz
            p4x = B2[j,1]-sx; p4y = B2[j,2]-sy; p4z = B2[j,3]-sz
            sumV += gauss_pair_1a(p1x,p1y,p1z, p2x,p2y,p2z, p3x,p3y,p3z, p4x,p4y,p4z, eps)
        end
    end
    return sumV
end

# -----------------------------
# Non-PBC linking number (1a)
# -----------------------------

function linking_number_1a(ringA::AbstractMatrix{<:Real}, ringB::AbstractMatrix{<:Real};
                           eps::Float64=1e-12,
                           disjoint_tol::Union{Nothing,Float64}=nothing,
                           disjoint_rel::Float64=1e-3)
    A = drop_duplicate_endpoint(ringA)
    B = drop_duplicate_endpoint(ringB)
    A1,B1 = segments_closed(A)
    A2,B2 = segments_closed(B)
    if size(A1,1) < 2 || size(A2,1) < 2
        return 0.0
    end

    dt = isnothing(disjoint_tol) ? auto_disjoint_tol(A,B; rel=disjoint_rel) : disjoint_tol
    d2 = min_seg_dist2_shift(A1,B1,A2,B2, 0.0,0.0,0.0; early_exit2=dt^2)
    if d2 < dt^2
        return NaN
    end

    sumV = link_sum_1a_shift(A1,B1,A2,B2, 0.0,0.0,0.0, eps)
    return sumV / (4.0*pi)
end

# -----------------------------
# PBC helpers
# -----------------------------

function pbc_base_integer_shift(cA::Vector{Float64}, cB::Vector{Float64},
                                cell::Matrix{Float64},
                                pbc::NTuple{3,Bool})
    inv_cell = inv(cell)
    δ = cB .- cA
    # matches Python: df = (cB-cA) @ inv_cell (row-vector)
    df = inv_cell' * δ
    n0 = zeros(Int, 3)
    @inbounds for k in 1:3
        n0[k] = pbc[k] ? round(Int, df[k]) : 0
    end
    return n0
end

# -----------------------------
# Linking number with PBC scan (1a)
# -----------------------------

"""
    linking_number_pbc_1a(ringA, ringB, cell; ...)

Returns (lk::Float64, shift::NTuple{3,Int}) where shift is integer (nx,ny,nz)
such that Bimg = B - (cell' * n).
"""
function linking_number_pbc_1a(ringA::AbstractMatrix{<:Real}, ringB::AbstractMatrix{<:Real},
                               cell_in::AbstractMatrix{<:Real};
                               pbc::NTuple{3,Bool}=(true,true,true),
                               n_images::Int=1,
                               eps::Float64=1e-12,
                               check_top_k::Union{Nothing,Int}=nothing,
                               disjoint_tol::Union{Nothing,Float64}=nothing,
                               disjoint_rel::Float64=1e-3,
                               integer_tol::Float64=1e-6)

    cell = Matrix{Float64}(cell_in)
    size(cell) == (3,3) || throw(ArgumentError("cell must be 3×3"))

    A = drop_duplicate_endpoint(ringA)
    B = drop_duplicate_endpoint(ringB)

    A1,B1 = segments_closed(A)
    A2,B2 = segments_closed(B)
    if size(A1,1) < 2 || size(A2,1) < 2
        return 0.0, (0,0,0)
    end

    dt = isnothing(disjoint_tol) ? auto_disjoint_tol(A,B; rel=disjoint_rel) : disjoint_tol
    dt2 = dt*dt

    # centroid min-image base shift
    n0 = pbc_base_integer_shift(centroid(A), centroid(B), cell, pbc)

    rng = -n_images:n_images
    # candidates store: (rank_d2, nx,ny,nz, sx,sy,sz)
    cand = Vector{NTuple{7,Float64}}()
    sizehint!(cand, (2n_images+1)^3)

    for i in rng, j in rng, k in rng
        nx = n0[1] + i
        ny = n0[2] + j
        nz = n0[3] + k
        if !pbc[1]; nx = 0; end
        if !pbc[2]; ny = 0; end
        if !pbc[3]; nz = 0; end

        nvec = [nx,ny,nz]
        shift = cell' * nvec  # 3-vector
        sx,sy,sz = shift[1], shift[2], shift[3]
        d2 = min_point_dist2_shift(A, B, sx,sy,sz)
        push!(cand, (d2, float(nx), float(ny), float(nz), sx,sy,sz))
    end

    sort!(cand, by = x -> x[1])
    if !isnothing(check_top_k)
        k = max(1, Int(check_top_k))
        if length(cand) > k
            resize!(cand, k)
        end
    end

    best_lk = NaN
    best_n  = (0,0,0)
    best_key = (-Inf, Inf, Inf)  # (abs(lk_int) maximize, dist2 minimize, resid minimize)

    for c in cand
        dist2 = c[1]
        nx = Int(round(c[2])); ny = Int(round(c[3])); nz = Int(round(c[4]))
        sx,sy,sz = c[5], c[6], c[7]

        # disjointness filter (segment-segment). Early exit at dt2.
        d2seg = min_seg_dist2_shift(A1,B1,A2,B2, sx,sy,sz; early_exit2=dt2)
        if d2seg < dt2
            continue
        end

        sumV = link_sum_1a_shift(A1,B1,A2,B2, sx,sy,sz, eps)
        lk = sumV / (4.0*pi)

        lk_int = round(Int, lk)
        resid  = abs(lk - lk_int)

        # If it’s not close to integer, keep it but de-prioritize strongly
        # (still lets you debug if something goes weird).
        int_weight = resid <= integer_tol ? abs(lk_int) : 0

        key = (float(int_weight), dist2, resid)

        # lexicographic: maximize int_weight, minimize dist2, minimize resid
        if (key[1] > best_key[1]) ||
           (key[1] == best_key[1] && key[2] < best_key[2]) ||
           (key[1] == best_key[1] && key[2] == best_key[2] && key[3] < best_key[3])
            best_key = key
            best_lk  = lk
            best_n   = (nx,ny,nz)
        end
    end

    return best_lk, best_n
end

# -----------------------------
# All-pairs driver (unique pairs) + threading
# -----------------------------

# Unrank 1-based pair index k into (i,j) with 1 ≤ i < j ≤ n
function unrank_pair(k::Int, n::Int)
    # Solve i from cumulative C(i)=i*(2n-i-1)/2 >= k
    # i = ceil(((2n-1) - sqrt((2n-1)^2 - 8k))/2)
    a = 2n - 1
    disc = float(a*a - 8k)
    i = Int(ceil((a - sqrt(disc))/2))
    prev = (i-1)*(2n - i) ÷ 2
    j = i + (k - prev)
    return i, j
end

"""
    all_pairs_linking_pbc_1a(rings, cell; kwargs...)

Computes linking_number_pbc_1a for all unique pairs i<j.

Returns:
  - lks::Vector{Float64} (length nC2)
  - shifts::Vector{NTuple{3,Int}}
  - I::Vector{Int}, J::Vector{Int}  (pair indices, same length)

You can filter linked pairs afterward: abs(round(lk)) >= 1 (with your own tol).
"""
function all_pairs_linking_pbc_1a(rings::Vector{<:AbstractMatrix{<:Real}}, cell;
                                 pbc::NTuple{3,Bool}=(true,true,true),
                                 n_images::Int=1,
                                 eps::Float64=1e-12,
                                 check_top_k::Union{Nothing,Int}=nothing,
                                 disjoint_tol::Union{Nothing,Float64}=nothing,
                                 disjoint_rel::Float64=1e-3,
                                 integer_tol::Float64=1e-6)

    n = length(rings)
    npairs = n*(n-1) ÷ 2
    lks = Vector{Float64}(undef, npairs)
    shifts = Vector{NTuple{3,Int}}(undef, npairs)
    I = Vector{Int}(undef, npairs)
    J = Vector{Int}(undef, npairs)

    Threads.@threads for k in 1:npairs
        i,j = unrank_pair(k, n)
        lk, sh = linking_number_pbc_1a(rings[i], rings[j], cell;
                                       pbc=pbc, n_images=n_images, eps=eps,
                                       check_top_k=check_top_k,
                                       disjoint_tol=disjoint_tol,
                                       disjoint_rel=disjoint_rel,
                                       integer_tol=integer_tol)
        lks[k] = lk
        shifts[k] = sh
        I[k] = i
        J[k] = j
    end

    return lks, shifts, I, J
end

end # module
