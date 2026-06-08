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
struct CliffordTensor{R} <: AbstractTensorElement{R}
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
# AbstractTensorElement hooks
#
# iszero, ==, hash, grade, grades, homogeneous_component come from the generic
# methods in abstract_tensor.jl.  A Clifford element's identity is pinned by its
# *metric* (not merely its space): two elements with equal terms but different
# metrics are different elements, so _eq_key returns the metric.

_eq_key(t::CliffordTensor)    = t.metric
base_space(t::CliffordTensor) = t.metric.space
_rebuild(t::CliffordTensor{R}, terms::Dict{Vector{Int}, R}) where R =
    CliffordTensor{R}(t.metric, terms)

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
# Display
#
# Basis blades print like their exterior counterparts (e₁∧e₂): a canonical
# Clifford basis blade of strictly-increasing distinct indices *is* the wedge
# of those basis vectors.  We reuse the ±1 coefficient shortcuts and `isequal`
# (rather than `==`) so the formatting is safe over a symbolic ring.

function _clifford_idx_str(space::VectorSpace, idx::Vector{Int})
    isempty(idx) && return "𝟏"
    join(string.(space.labels[idx]), "∧")
end

function Base.show(io::IO, t::CliffordTensor{R}) where R
    if isempty(t.terms)
        print(io, "0")
        return
    end
    sorted = sort(collect(t.terms); by = kv -> (length(kv[1]), kv[1]))
    parts  = String[]
    for (idx, c) in sorted
        s = _clifford_idx_str(t.metric.space, idx)
        if isequal(c, one(R))
            push!(parts, s)
        elseif isequal(c, -one(R))
            push!(parts, "-$s")
        else
            push!(parts, "($c)⋅$s")
        end
    end
    print(io, join(parts, " + "))
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
