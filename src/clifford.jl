"""
    CliffordTensor{R}

An element of the Clifford algebra Cl(V, g) associated with a Metric{R}.

The Clifford relation is

    eᵢeⱼ + eⱼeᵢ = 2g(eᵢ,eⱼ)·1    for all i, j

which in the diagonal (signature) case reduces to eᵢ² = Q(eᵢ) = g(eᵢ,eᵢ) and
eᵢeⱼ = −eⱼeᵢ for i ≠ j.

As a vector space, Cl(V,g) ≅ Λ(V): both have dimension 2ⁿ and the same basis
of strictly-increasing multi-indices.  The products differ.

## Relation to Phase 2 AlgebraTensor

`CliffordTensor{R}` is a standalone type rather than `AlgebraTensor{A,R}` because
the metric is an _instance_ (not a type parameter), and Julia cannot use arbitrary
structs as type parameters.  A unified interface is planned for Phase 4.

## Storage

`terms :: Dict{Vector{Int}, R}` maps strictly-increasing multi-indices to nonzero
coefficients.  Grade-k elements have length-k keys; the scalar uses `Int[]`.
"""
struct CliffordTensor{R}
    metric :: Metric{R}
    terms  :: Dict{Vector{Int}, R}

    function CliffordTensor{R}(
            metric :: Metric{R},
            raw    :: Dict{Vector{Int}, R}
        ) where R
        terms = Dict{Vector{Int}, R}()
        n     = metric.space.n
        for (idx, coef) in raw
            iszero(coef) && continue
            for i in idx
                1 <= i <= n || throw(ArgumentError(
                    "basis index $i out of range 1:$n"))
            end
            _clifford_normalize!(terms, metric.g, idx, coef)
        end
        new(metric, terms)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Core normalization
#
# Rewrites any multi-index into a sum of canonical (strictly-increasing) ones by
# repeatedly applying:
#
#   Equal adjacent pair   eₐeₐ → g(eₐ,eₐ)·(rest)           (contract)
#   Inverted adjacent pair eₐeᵦ (a > b) → 2g(eₐ,eᵦ)·(rest) − eᵦeₐ·(rest)
#
# Both steps strictly reduce the rewriting measure (lexicographic order of
# (length, first inversion position)), so termination is guaranteed.
#
# For a diagonal metric g(eᵢ,eⱼ) = 0 for i≠j, so the cross-term vanishes and
# the algorithm is isomorphic to Exterior normalization except that equal-index
# pairs contract to g(eᵢ,eᵢ) instead of 0.

function _clifford_normalize!(
    result :: Dict{Vector{Int}, R},
    g      :: Matrix{R},
    idx    :: Vector{Int},
    coef   :: R
) where R
    iszero(coef) && return
    k = length(idx)

    for p in 1:k-1
        a, b = idx[p], idx[p+1]

        if a == b
            # Contract: eₐ² = g(eₐ,eₐ)
            g_val   = g[a, a]
            new_idx = vcat(idx[1:p-1], idx[p+2:end])
            _clifford_normalize!(result, g, new_idx, coef * g_val)
            return

        elseif a > b
            # Anticommute: eₐeᵦ = −eᵦeₐ + 2g(eₐ,eᵦ)·(rest)

            swapped      = copy(idx)
            swapped[p]   = b
            swapped[p+1] = a
            _clifford_normalize!(result, g, swapped, -coef)

            g_val = g[a, b]   # symmetric: g[a,b] == g[b,a]
            if !iszero(g_val)
                new_idx = vcat(idx[1:p-1], idx[p+2:end])
                # 2·coef without integer literal conversion
                _clifford_normalize!(result, g, new_idx, (coef + coef) * g_val)
            end
            return
        end
        # a < b: position p is canonical, continue to p+1
    end

    # Fully canonical (strictly increasing)
    prev = get(result, idx, zero(R))
    val  = prev + coef
    if iszero(val)
        delete!(result, idx)
    else
        result[idx] = val
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Element constructors

"""
    clifford_basis_element(metric, idx) → CliffordTensor{R}

Basis element for multi-index `idx` (need not be canonical; will be normalized).
"""
function clifford_basis_element(metric::Metric{R}, idx::Vector{Int}) where R
    CliffordTensor{R}(metric, Dict{Vector{Int}, R}(idx => one(R)))
end

clifford_basis_element(metric::Metric{R}, ::Type{R}, idx::Vector{Int}) where R =
    clifford_basis_element(metric, idx)

"""
    clifford_basis_vector(metric, i) → CliffordTensor{R}

Grade-1 basis element eᵢ in Cl(V,g).
"""
clifford_basis_vector(metric::Metric{R}, i::Int) where R =
    clifford_basis_element(metric, [i])

clifford_basis_vector(metric::Metric{R}, ::Type{R}, i::Int) where R =
    clifford_basis_vector(metric, i)

"""
    clifford_scalar(metric, c) → CliffordTensor{R}

Grade-0 scalar c·1.
"""
function clifford_scalar(metric::Metric{R}, c::R) where R
    iszero(c) && return CliffordTensor{R}(metric, Dict{Vector{Int}, R}())
    CliffordTensor{R}(metric, Dict{Vector{Int}, R}(Int[] => c))
end

clifford_scalar(metric::Metric{R}, ::Type{R}, c::R) where R = clifford_scalar(metric, c)

"""
    clifford_zero(metric) → CliffordTensor{R}

Additive identity in Cl(V,g).
"""
clifford_zero(metric::Metric{R}) where R =
    CliffordTensor{R}(metric, Dict{Vector{Int}, R}())

clifford_zero(metric::Metric{R}, ::Type{R}) where R = clifford_zero(metric)

"""
    clifford_one(metric) → CliffordTensor{R}

Multiplicative identity (scalar 1) in Cl(V,g).
"""
clifford_one(metric::Metric{R}) where R = clifford_scalar(metric, one(R))

clifford_one(metric::Metric{R}, ::Type{R}) where R = clifford_one(metric)

# ─────────────────────────────────────────────────────────────────────────────
# Predicates and equality

Base.iszero(t::CliffordTensor) = isempty(t.terms)

function Base.:(==)(a::CliffordTensor{R}, b::CliffordTensor{R}) where R
    a.metric == b.metric && a.terms == b.terms
end

Base.hash(t::CliffordTensor{R}, h::UInt) where R =
    hash(t.terms, hash(t.metric, h))

function _check_compatible(a::CliffordTensor{R}, b::CliffordTensor{R}) where R
    a.metric == b.metric || throw(ArgumentError(
        "Cannot operate on CliffordTensors with different metrics"))
end

# ─────────────────────────────────────────────────────────────────────────────
# Arithmetic

function Base.:+(a::CliffordTensor{R}, b::CliffordTensor{R}) where R
    _check_compatible(a, b)
    terms = copy(a.terms)
    for (idx, coef) in b.terms
        prev = get(terms, idx, zero(R))
        val  = prev + coef
        if iszero(val)
            delete!(terms, idx)
        else
            terms[idx] = val
        end
    end
    # Addition preserves canonical form: just wrap without re-normalizing.
    # Use the inner constructor (which would re-normalize — harmless but wasteful).
    # Build directly:
    CliffordTensor{R}(a.metric, terms)
end

function Base.:-(t::CliffordTensor{R}) where R
    terms = Dict{Vector{Int}, R}()
    for (idx, coef) in t.terms
        terms[idx] = -coef
    end
    CliffordTensor{R}(t.metric, terms)
end

Base.:-(a::CliffordTensor{R}, b::CliffordTensor{R}) where R = a + (-b)

function Base.:*(c::R, t::CliffordTensor{R}) where R
    iszero(c) && return clifford_zero(t.metric)
    terms = Dict{Vector{Int}, R}()
    for (idx, coef) in t.terms
        terms[idx] = c * coef
    end
    CliffordTensor{R}(t.metric, terms)
end

Base.:*(t::CliffordTensor{R}, c::R) where R = c * t
Base.:*(n::Integer, t::CliffordTensor{R}) where R = R(n) * t
Base.:*(t::CliffordTensor{R}, n::Integer) where R = R(n) * t

# ─────────────────────────────────────────────────────────────────────────────
# Geometric product

"""
    a * b → CliffordTensor{R}

The Clifford geometric product.  Concatenates multi-indices (T(V) product) then
applies `_clifford_normalize!` to impose the Clifford relations from the shared
metric.
"""
function Base.:*(a::CliffordTensor{R}, b::CliffordTensor{R}) where R
    _check_compatible(a, b)
    result = Dict{Vector{Int}, R}()
    for (ai, ac) in a.terms, (bi, bc) in b.terms
        _clifford_normalize!(result, a.metric.g, vcat(ai, bi), ac * bc)
    end
    CliffordTensor{R}(a.metric, result)
end

# ─────────────────────────────────────────────────────────────────────────────
# Grade and graded components

"""
    grade(t::CliffordTensor) → Int

Grade of a homogeneous element.  Throws for zero or inhomogeneous elements.
"""
function grade(t::CliffordTensor{R}) where R
    isempty(t.terms) && throw(ArgumentError(
        "grade is undefined for the zero element"))
    gs = unique([length(idx) for idx in keys(t.terms)])
    length(gs) == 1 || throw(ArgumentError(
        "grade is undefined for an inhomogeneous element (grades present: $gs)"))
    gs[1]
end

"""
    grades(t::CliffordTensor) → Vector{Int}

All grades present in t, sorted.
"""
grades(t::CliffordTensor{R}) where R =
    sort(unique([length(idx) for idx in keys(t.terms)]))

"""
    homogeneous_component(t, k) → CliffordTensor{R}

Grade-k part of t.
"""
function homogeneous_component(t::CliffordTensor{R}, k::Int) where R
    terms = Dict{Vector{Int}, R}(
        idx => coef for (idx, coef) in t.terms if length(idx) == k
    )
    CliffordTensor{R}(t.metric, terms)
end

# ─────────────────────────────────────────────────────────────────────────────
# Homogeneous basis

"""
    clifford_homogeneous_basis(metric, R, k) → Vector{CliffordTensor{R}}

Grade-k canonical basis for Cl(V,g).  Uses strictly-increasing multi-indices of
length k — the same vector-space basis as Λ(V).
"""
function clifford_homogeneous_basis(metric::Metric{R}, ::Type{R}, k::Int) where R
    idxs = all_ext_grade_k_indices(metric.space, k)   # reuse Exterior enumeration
    [clifford_basis_element(metric, idx) for idx in idxs]
end

# ─────────────────────────────────────────────────────────────────────────────
export CliffordTensor,
       clifford_basis_element, clifford_basis_vector,
       clifford_scalar, clifford_zero, clifford_one,
       clifford_homogeneous_basis,
       _clifford_normalize!
