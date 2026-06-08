"""
    Metric{R}

A symmetric bilinear form g : V × V → R on a VectorSpace, stored as an n×n
matrix over the ring R.

Algebraically this defines the quadratic form Q(u) = g(u,u) that drives Clifford
normalization: the relation eᵢeᵢ = Q(eᵢ)·1 in Cl(V,Q) replaces eᵢ² = 0 in Λ(V).

# Signature conventions

For signature (p,q,r) — p positive, q negative, r null directions — use the
convenience constructor `signature_metric(space, R, p, q, r)`.  The diagonal
entries are ordered [+1,…,+1, −1,…,−1, 0,…,0].

For a general symmetric bilinear form (off-diagonal entries present) use the
matrix constructor directly.  Off-diagonal entries g_{ij} affect the cross-term
in the Clifford relation eᵢeⱼ + eⱼeᵢ = 2g(eᵢ,eⱼ)·1.
"""
struct Metric{R}
    space :: VectorSpace
    g     :: Matrix{R}

    function Metric{R}(space::VectorSpace, g::AbstractMatrix{R}) where R
        n = space.n
        size(g) == (n, n) || throw(ArgumentError(
            "Metric matrix size $(size(g)) does not match space dimension $n"))
        for i in 1:n, j in i+1:n
            g[i,j] == g[j,i] || throw(ArgumentError(
                "Metric matrix is not symmetric: g[$i,$j]=$(g[i,j]) ≠ g[$j,$i]=$(g[j,i])"))
        end
        new(space, Matrix{R}(g))
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Convenience constructors

"""
    diagonal_metric(space, R, diag) → Metric{R}

Build a diagonal metric from a vector of diagonal entries.  `diag[i] = g(eᵢ,eᵢ)`.
"""
function diagonal_metric(space::VectorSpace, ::Type{R}, diag::Vector{R}) where R
    n = space.n
    length(diag) == n || throw(ArgumentError(
        "diagonal has length $(length(diag)) but space has dimension $n"))
    g = fill(zero(R), n, n)
    for i in 1:n
        g[i,i] = diag[i]
    end
    Metric{R}(space, g)
end

"""
    signature_metric(space, R, p, q, r) → Metric{R}

Diagonal metric with p positive (+1), q negative (−1), and r null (0) directions.
Requires p + q + r == space.n.
"""
function signature_metric(space::VectorSpace, ::Type{R}, p::Int, q::Int, r::Int) where R
    p >= 0 && q >= 0 && r >= 0 || throw(ArgumentError("p, q, r must be non-negative"))
    p + q + r == space.n || throw(ArgumentError(
        "p+q+r = $(p+q+r) must equal space dimension $(space.n)"))
    diag = vcat(
        fill(one(R),       p),
        fill(-one(R),      q),
        fill(zero(R),      r),
    )
    diagonal_metric(space, R, diag)
end

"""
    zero_metric(space, R) → Metric{R}

The trivial metric g ≡ 0.  A Clifford algebra with the zero metric is isomorphic
to the Exterior algebra.
"""
zero_metric(space::VectorSpace, ::Type{R}) where R =
    signature_metric(space, R, 0, 0, space.n)

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation

"""
    bilinear_form(metric, i, j) → R

Return g(eᵢ, eⱼ) = metric.g[i, j].
"""
bilinear_form(metric::Metric{R}, i::Int, j::Int) where R = metric.g[i, j]

"""
    quadratic_form(metric, i) → R

Return Q(eᵢ) = g(eᵢ, eᵢ) = metric.g[i, i].
"""
quadratic_form(metric::Metric{R}, i::Int) where R = metric.g[i, i]

# ─────────────────────────────────────────────────────────────────────────────
# Equality and hashing

Base.:(==)(a::Metric{R}, b::Metric{R}) where R =
    a.space == b.space && a.g == b.g

Base.hash(m::Metric{R}, h::UInt) where R =
    hash(m.g, hash(m.space, h))

# ─────────────────────────────────────────────────────────────────────────────
export Metric,
       diagonal_metric, signature_metric, zero_metric,
       bilinear_form, quadratic_form
