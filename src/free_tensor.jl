# ── Free tensor algebra T(V) ─────────────────────────────────────────────────
#
# MATHEMATICAL CONTRACT
#
# T(V) = ⊕_{k≥0} V^{⊗k}  =  R ⊕ V ⊕ (V⊗V) ⊕ (V⊗V⊗V) ⊕ …
#
# Basis of grade-k piece: all ordered length-k index sequences with repetition
# allowed — there are n^k of them.  Index sequences are vectors in ℤ≥1^k.
#
# Algebra product: concatenation of index sequences.
#
#   (e_{i₁}⊗…⊗e_{iₖ}) · (e_{j₁}⊗…⊗e_{jₘ})  =  e_{i₁}⊗…⊗e_{iₖ}⊗e_{j₁}⊗…⊗e_{jₘ}
#
# This product is:
#   - associative
#   - non-commutative  (e₁⊗e₂ ≠ e₂⊗e₁ in general)
#   - imposes NO relations  (e_i⊗e_i ≠ 0, unlike Exterior/Clifford)
#
# REPRESENTATION
#
# An element is stored as a sparse Dict{Vector{Int}, R}:
#   multi-index (ordered, repetition allowed) → coefficient in R
#
# The empty multi-index Int[] is the grade-0 / scalar component.
# Zero coefficients are pruned; the zero element has an empty dict.
#
# WHY NOT A BITMASK?
#
# GASmith uses a uint8_t blade bitmask as its core index.  A bitmask encodes
# a *subset* of basis vectors — it cannot represent e_i⊗e_i (same index
# twice) and implicitly imposes antisymmetry (the set {i,j} = {j,i}).  That
# is exactly the exterior algebra's index structure, not T(V)'s.  The general
# representation must live underneath the bitmask; the bitmask is a valid
# optimization for the Exterior/Clifford leaf (Phase 2–3) only.

# ─────────────────────────────────────────────────────────────────────────────

"""
    FreeTensor{R}

An element of the free tensor algebra T(V) over scalar ring `R`.

Stored as a sparse `Dict{Vector{Int}, R}` mapping ordered multi-indices
(with repetition) to coefficients.  The empty multi-index `Int[]` represents
the grade-0 / scalar component.

## Constructors
- [`basis_element(V, R, idx)`](@ref)   — pure tensor e_{idx[1]} ⊗ ⋯ ⊗ e_{idx[k]}
- [`basis_vector(V, R, i)`](@ref)      — grade-1 basis vector eᵢ
- [`scalar_element(V, c)`](@ref)       — grade-0 element c·𝟏
- `zero(FreeTensor{R}, V)`             — additive identity
- `one(FreeTensor{R}, V)`              — multiplicative identity (= scalar 1)

## Operations
- `a + b`, `a - b`, `-a`              — linear combination
- `a * b`  (or `tensor_product(a,b)`) — concatenation product
- `c * a`, `a * c`                    — scalar multiplication
- `==`                                — exact equality (requires exact R)

## Grading
- [`grade(t)`](@ref)                  — grade of a homogeneous element
- [`grades(t)`](@ref)                 — set of all grades present
- [`homogeneous_component(t, k)`](@ref) — grade-k projection

## Basis enumeration
- [`all_grade_k_indices(V, k)`](@ref) — all grade-k multi-indices (n^k of them)
- [`homogeneous_basis(V, R, k)`](@ref) — corresponding FreeTensor elements
"""
struct FreeTensor{R}
    space :: VectorSpace
    terms :: Dict{Vector{Int}, R}   # multi-index → coefficient

    function FreeTensor{R}(space::VectorSpace,
                           terms::Dict{Vector{Int}, R}) where R
        n = space.n
        for (idx, _) in terms
            all(1 <= j <= n for j in idx) ||
                throw(ArgumentError(
                    "Multi-index $idx has entries outside {1,…,$n}"))
        end
        # Canonical form: no zero coefficients
        pruned = Dict{Vector{Int}, R}(
            idx => c for (idx, c) in terms if !iszero(c))
        new(space, pruned)
    end
end

# ── Constructors ──────────────────────────────────────────────────────────────

"""
    basis_element(space, R, idx) -> FreeTensor{R}

The pure tensor `e_{idx[1]} ⊗ ⋯ ⊗ e_{idx[k]}` with coefficient `one(R)`.
`idx` is an ordered multi-index with entries in `{1,…,space.n}`; repetitions
are allowed (and meaningful — this is T(V), not Exterior).

# Example
```julia
V = VectorSpace(3)
t = basis_element(V, Rational{BigInt}, [1, 1, 2])  # e₁⊗e₁⊗e₂
```
"""
function basis_element(space::VectorSpace, ::Type{R},
                       idx::AbstractVector{Int}) where R
    FreeTensor{R}(space, Dict{Vector{Int}, R}(Vector{Int}(idx) => one(R)))
end

"""
    basis_vector(space, R, i) -> FreeTensor{R}

The grade-1 basis vector `eᵢ`.  Equivalent to `basis_element(space, R, [i])`.
"""
function basis_vector(space::VectorSpace, ::Type{R}, i::Int) where R
    1 <= i <= space.n ||
        throw(ArgumentError(
            "Basis index $i out of range for $(space)"))
    basis_element(space, R, [i])
end

"""
    scalar_element(space, c::R) -> FreeTensor{R}

The grade-0 element `c·𝟏` (the scalar `c` embedded as the identity in T(V)).
"""
function scalar_element(space::VectorSpace, c::R) where R
    iszero(c) ?
        FreeTensor{R}(space, Dict{Vector{Int}, R}()) :
        FreeTensor{R}(space, Dict{Vector{Int}, R}(Int[] => c))
end

Base.zero(::Type{FreeTensor{R}}, space::VectorSpace) where R =
    FreeTensor{R}(space, Dict{Vector{Int}, R}())

Base.one(::Type{FreeTensor{R}}, space::VectorSpace) where R =
    scalar_element(space, one(R))

# ── Predicates ────────────────────────────────────────────────────────────────

Base.iszero(t::FreeTensor) = isempty(t.terms)

Base.:(==)(a::FreeTensor{R}, b::FreeTensor{R}) where R =
    a.space == b.space && a.terms == b.terms

Base.hash(t::FreeTensor, h::UInt) = hash(t.terms, hash(t.space, h))

# ── Grading ───────────────────────────────────────────────────────────────────

"""
    grade(t::FreeTensor) -> Int

The grade of a *homogeneous* element — the common length of all its
multi-indices.

Throws `ArgumentError` if `t` is the zero element (grade undefined for zero)
or if `t` is not homogeneous.  Use [`grades(t)`](@ref) to inspect an arbitrary
element, and [`homogeneous_component`](@ref) to extract a specific grade.
"""
function grade(t::FreeTensor{R}) where R
    isempty(t.terms) &&
        throw(ArgumentError("grade is undefined for the zero element"))
    gs = unique([length(idx) for idx in keys(t.terms)])
    length(gs) == 1 ||
        throw(ArgumentError(
            "Element is not homogeneous; grades present: $(sort(gs))"))
    gs[1]
end

"""
    grades(t::FreeTensor) -> Vector{Int}

Sorted list of all grades present in `t`.  Returns `Int[]` for the zero element.
"""
grades(t::FreeTensor{R}) where R =
    sort(unique([length(idx) for idx in keys(t.terms)]))

"""
    homogeneous_component(t::FreeTensor{R}, k::Int) -> FreeTensor{R}

Extract the grade-`k` part of `t`.  Returns the zero element if `t` has no
grade-`k` terms.
"""
function homogeneous_component(t::FreeTensor{R}, k::Int) where R
    terms = Dict{Vector{Int}, R}(
        idx => c for (idx, c) in t.terms if length(idx) == k)
    FreeTensor{R}(t.space, terms)
end

# ── Arithmetic ────────────────────────────────────────────────────────────────

function _check_same_space(a::FreeTensor, b::FreeTensor, op::Symbol)
    a.space == b.space ||
        throw(ArgumentError(
            "Cannot $op FreeTensors over different spaces:\n  $(a.space)\n  $(b.space)"))
end

function Base.:+(a::FreeTensor{R}, b::FreeTensor{R}) where R
    _check_same_space(a, b, :add)
    terms = copy(a.terms)
    for (idx, c) in b.terms
        prev   = get(terms, idx, zero(R))
        newval = prev + c
        if iszero(newval)
            delete!(terms, idx)
        else
            terms[idx] = newval
        end
    end
    # Construct directly to skip re-validation (indices already valid)
    FreeTensor{R}(a.space, terms)
end

function Base.:-(a::FreeTensor{R}, b::FreeTensor{R}) where R
    a + (-b)
end

function Base.:-(t::FreeTensor{R}) where R
    terms = Dict{Vector{Int}, R}(idx => -c for (idx, c) in t.terms)
    FreeTensor{R}(t.space, terms)
end

# Scalar × tensor and tensor × scalar
function Base.:*(c::R, t::FreeTensor{R}) where R
    iszero(c) && return zero(FreeTensor{R}, t.space)
    terms = Dict{Vector{Int}, R}(idx => c * v for (idx, v) in t.terms)
    FreeTensor{R}(t.space, terms)
end

function Base.:*(t::FreeTensor{R}, c::R) where R
    iszero(c) && return zero(FreeTensor{R}, t.space)
    terms = Dict{Vector{Int}, R}(idx => v * c for (idx, v) in t.terms)
    FreeTensor{R}(t.space, terms)
end

# Allow integer literals: 2 * e[1] without explicit R conversion
Base.:*(n::Integer, t::FreeTensor{R}) where R = R(n) * t
Base.:*(t::FreeTensor{R}, n::Integer) where R = t * R(n)

"""
    a * b  ≡  tensor_product(a, b)

The algebra product of T(V): concatenation of multi-indices.

    (e_{I}) · (e_{J})  =  e_{I ∥ J}

where I ∥ J is the concatenation of the ordered sequences I and J.

This product is associative and non-commutative.  It imposes *no* relations:
`eᵢ⊗eᵢ ≠ 0` and `eᵢ⊗eⱼ ≠ ±eⱼ⊗eᵢ` in general.  Those relations enter only
when we impose a quotient (Phase 2+).
"""
function Base.:*(a::FreeTensor{R}, b::FreeTensor{R}) where R
    _check_same_space(a, b, :multiply)
    isempty(a.terms) && return zero(FreeTensor{R}, a.space)
    isempty(b.terms) && return zero(FreeTensor{R}, a.space)
    terms = Dict{Vector{Int}, R}()
    for (ai, ac) in a.terms, (bi, bc) in b.terms
        key  = vcat(ai, bi)      # ← concatenation: the heart of T(V)
        prev = get(terms, key, zero(R))
        val  = prev + ac * bc
        if iszero(val)
            delete!(terms, key)
        else
            terms[key] = val
        end
    end
    FreeTensor{R}(a.space, terms)
end

"""
    tensor_product(a, b) -> FreeTensor

Alias for `a * b` (the concatenation product in T(V)).
"""
tensor_product(a::FreeTensor{R}, b::FreeTensor{R}) where R = a * b

# Unicode alias  a ⊗ b
⊗(a::FreeTensor{R}, b::FreeTensor{R}) where R = a * b

# ── Basis enumeration ─────────────────────────────────────────────────────────

"""
    all_grade_k_indices(space, k) -> Vector{Vector{Int}}

All ordered multi-indices of length `k` with entries in `{1,…,n}`, repetitions
allowed.  Returns `n^k` indices — the complete grade-`k` basis of T(V).

The `k = 0` case returns `[Int[]]` (one index: the empty sequence for the
scalar / grade-0 piece), consistent with `n^0 = 1` for all n ≥ 0.
"""
function all_grade_k_indices(space::VectorSpace, k::Int)
    k >= 0 || throw(ArgumentError("Grade k must be ≥ 0, got $k"))
    k == 0 && return [Int[]]          # scalar: one grade-0 index for any n
    n = space.n
    n == 0 && return Vector{Vector{Int}}()   # 0^k = 0 for k ≥ 1
    result = Vector{Vector{Int}}()
    _enumerate!(result, Int[], n, k)
    result
end

function _enumerate!(result::Vector{Vector{Int}},
                     current::Vector{Int}, n::Int, k::Int)
    if length(current) == k
        push!(result, copy(current))
        return
    end
    for i in 1:n
        push!(current, i)
        _enumerate!(result, current, n, k)
        pop!(current)
    end
end

"""
    grade_dimension(space, k) -> Int

Dimension of the grade-`k` piece of T(V): `n^k` where `n = space.n`.
"""
grade_dimension(space::VectorSpace, k::Int) = space.n ^ k

"""
    homogeneous_basis(space, R, k) -> Vector{FreeTensor{R}}

All grade-`k` basis elements of T(V) as `FreeTensor{R}` objects.
There are `n^k` of them.  Their multi-indices are in the order returned by
[`all_grade_k_indices`](@ref).
"""
function homogeneous_basis(space::VectorSpace, ::Type{R}, k::Int) where R
    [basis_element(space, R, idx) for idx in all_grade_k_indices(space, k)]
end

# ── Display ───────────────────────────────────────────────────────────────────

# Build a human-readable string for a single multi-index, e.g. "e1⊗e2⊗e1"
function _idx_str(space::VectorSpace, idx::Vector{Int})
    isempty(idx) && return "𝟏"
    join(string.(space.labels[idx]), "⊗")
end

function Base.show(io::IO, t::FreeTensor{R}) where R
    if isempty(t.terms)
        print(io, "0")
        return
    end
    sorted = sort(collect(t.terms); by = kv -> (length(kv[1]), kv[1]))
    parts  = String[]
    for (idx, c) in sorted
        s = _idx_str(t.space, idx)
        if c == one(R)
            push!(parts, s)
        elseif c == -one(R)
            push!(parts, "-$s")
        else
            push!(parts, "($c)⋅$s")
        end
    end
    print(io, join(parts, " + "))
end

# ── Exports ───────────────────────────────────────────────────────────────────

export FreeTensor,
       basis_element, basis_vector, scalar_element,
       homogeneous_basis, homogeneous_component,
       grade, grades,
       tensor_product, ⊗,
       all_grade_k_indices, grade_dimension
