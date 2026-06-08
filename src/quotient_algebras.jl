# ── Quotient algebras: Symmetric and Exterior ─────────────────────────────────
#
# MATHEMATICAL SETUP
#
# Each named algebra is T(V)/I for a specific two-sided ideal I.  We realize
# this in code as a *normalization rule*: a function that maps any multi-index
# (the output of concatenation) to a canonical representative plus a scalar.
# The product in every algebra is then identical in structure:
#
#   a ·_A b  =  Σ_{I,J}  c_I · d_J · normalize_A( I ∥ J )
#
# where I ∥ J is index concatenation (the raw T(V) product) and normalize_A
# reduces the result to the canonical form of algebra A.
#
# | Algebra    | Ideal ⟨generators⟩  | Canonical form          | Grade-k dim  |
# |------------|---------------------|-------------------------|--------------|
# | Symmetric  | x⊗y − y⊗x          | non-decreasing index    | C(n+k−1, k)  |
# | Exterior   | x⊗x                 | strictly increasing idx | C(n, k)      |
# |            |                     | + sign; zero on repeat  |              |
#
# Clifford (Phase 3) uses ideal ⟨x⊗x − Q(x)·1⟩; its normalize is the same
# as Exterior but with metric contractions replacing the zero-on-repeat rule.
# The mandatory test: Clifford with Q=0 must degenerate to Exterior exactly.
#
# WHY DISPATCH ON THE ALGEBRA TYPE?
# This is the "one product, many normalizers" invariant from the spec.
# Julia's multiple dispatch is perfect here: `normalize(alg, idx, coef)`
# dispatches on `alg` at compile time with zero runtime branching.  Adding a
# new algebra (say, Clifford) is just a new method of `normalize` — no
# existing code changes.

# ── Algebra tag types ─────────────────────────────────────────────────────────
#
# Singleton structs (no fields).  They exist purely for dispatch.
# Constructing A() for a singleton is free; Julia may optimize it away entirely.

"""Tag for the free tensor algebra T(V). `normalize` is the identity."""
struct FreeAlgebra end

"""
Tag for the symmetric algebra Sym(V) = T(V)/⟨x⊗y − y⊗x⟩.

Canonical multi-index form: non-decreasing sequence (multiset representation).
As an algebra, Sym(V) is exactly the graded polynomial ring R[e₁,…,eₙ].
Grade-k dimension: `C(n+k−1, k)`.
"""
struct SymmetricAlgebra end

"""
Tag for the exterior (Grassmann) algebra Λ(V) = T(V)/⟨x⊗x⟩.

Canonical multi-index form: strictly increasing sequence (set representation)
with a sign absorbed into the coefficient.  Any multi-index with a repeated
entry maps to zero (implementing `eᵢ∧eᵢ = 0` in a characteristic-free way).

Grade-k dimension: `C(n, k)`.  Total dimension: `2ⁿ`.

`eᵢ∧eⱼ = −eⱼ∧eᵢ` follows automatically from the normalization.
"""
struct ExteriorAlgebra end

# ── Normalization functions ───────────────────────────────────────────────────
#
# normalize(alg, idx, coef) → (canonical_idx, canonical_coef)
#
# Contracts:
#   - idx and coef on input represent a single pure-tensor term from T(V)
#   - canonical_idx on output is the reduced index in algebra `alg`
#   - canonical_coef absorbs any sign from reordering
#   - iszero(canonical_coef) means the term vanishes in this algebra
#   - these are internal; users interact through `*` and `∧`

normalize(::FreeAlgebra, idx::Vector{Int}, coef::R) where R = (idx, coef)

"""Symmetric normalization: sort the multi-index non-decreasingly, no sign."""
function normalize(::SymmetricAlgebra, idx::Vector{Int}, coef::R) where R
    length(idx) <= 1 && return (idx, coef)    # scalar or grade-1: already canonical
    (sort(idx), coef)
end

"""
Exterior normalization: sort via bubble sort tracking sign; zero on any repeat.

The ideal ⟨x⊗x⟩ is the characteristic-free generator (valid over any ring,
including characteristic 2), producing `eᵢ∧eᵢ = 0` and `eᵢ∧eⱼ = −eⱼ∧eᵢ`.
"""
function normalize(::ExteriorAlgebra, idx::Vector{Int}, coef::R) where R
    length(idx) <= 1 && return (idx, coef)    # scalar or grade-1: already canonical
    n = length(idx)

    # Check for repeated entries → term vanishes
    # O(n²) for small grade; fine for typical usage (grade rarely exceeds ~10)
    for i in 1:n, j in i+1:n
        if idx[i] == idx[j]
            return (Int[], zero(R))
        end
    end

    # Bubble sort with sign tracking.
    # Each adjacent swap corresponds to applying the relation eᵢ∧eⱼ = −eⱼ∧eᵢ.
    # sign_flip: true means an odd number of swaps → multiply coef by -1.
    idx       = copy(idx)
    sign_flip = false
    for i in 1:n-1
        for j in 1:n-i
            if idx[j] > idx[j+1]
                idx[j], idx[j+1] = idx[j+1], idx[j]
                sign_flip = !sign_flip
            end
        end
    end
    (idx, sign_flip ? -coef : coef)
end

# ── AlgebraTensor{A,R} ────────────────────────────────────────────────────────

"""
    AlgebraTensor{A, R}

An element of the quotient algebra T(V)/I, where the algebra tag `A`
determines the ideal I and the normalization rule.

## Type parameter A
- `SymmetricAlgebra` — Sym(V), non-decreasing indices
- `ExteriorAlgebra`  — Λ(V), strictly increasing indices + sign
- `CliffordAlgebra{Metric}` (Phase 3)

## Storage
A sparse `Dict{Vector{Int}, R}`, identical to `FreeTensor{R}`.  The difference:
every stored multi-index is in the canonical form of algebra `A`.  Canonicalization
happens in the constructor and on every product — never afterward.

## Construction
- `alg_basis_vector(V, R, A, i)` — grade-1 basis vector eᵢ in algebra A
- `alg_basis_element(V, R, A, idx)` — arbitrary pure tensor (normalized on input)
- `alg_scalar(V, c, A)`  — grade-0 element c·𝟏
- `sym_basis_vector(V, R, i)`, `ext_basis_vector(V, R, i)` — convenience

## Operations
- `a + b`, `a - b`, `-a`, `c * a`, `a * c`  — linear algebra
- `a * b`                                    — algebra product (normalized)
- `a ∧ b`                                    — wedge product (Exterior only)
"""
struct AlgebraTensor{A, R}
    space :: VectorSpace
    terms :: Dict{Vector{Int}, R}

    function AlgebraTensor{A,R}(space::VectorSpace,
                                terms::Dict{Vector{Int}, R}) where {A,R}
        n = space.n
        result = Dict{Vector{Int}, R}()
        for (idx, coef) in terms
            iszero(coef) && continue
            all(1 <= j <= n for j in idx) ||
                throw(ArgumentError(
                    "Multi-index $idx has entries outside {1,…,$n}"))
            (nidx, ncoef) = normalize(A(), idx, coef)
            iszero(ncoef) && continue
            prev = get(result, nidx, zero(R))
            val  = prev + ncoef
            if iszero(val)
                delete!(result, nidx)
            else
                result[nidx] = val
            end
        end
        new(space, result)
    end
end

# ── Constructors ──────────────────────────────────────────────────────────────

"""
    alg_basis_element(space, R, A, idx) -> AlgebraTensor{A,R}

Pure tensor `e_{idx[1]} ⊗ ⋯ ⊗ e_{idx[k]}` normalized into algebra `A`.
The constructor applies `normalize(A(), idx, one(R))`, so passing an
out-of-order index is valid and will be sorted (and sign-adjusted for Exterior).
"""
function alg_basis_element(space::VectorSpace, ::Type{R}, ::Type{A},
                            idx::AbstractVector{Int}) where {R,A}
    AlgebraTensor{A,R}(space, Dict{Vector{Int}, R}(Vector{Int}(idx) => one(R)))
end

"""
    alg_basis_vector(space, R, A, i) -> AlgebraTensor{A,R}

Grade-1 basis vector `eᵢ` in algebra `A`.
"""
function alg_basis_vector(space::VectorSpace, ::Type{R}, ::Type{A},
                           i::Int) where {R,A}
    1 <= i <= space.n ||
        throw(ArgumentError("Basis index $i out of range for $(space)"))
    AlgebraTensor{A,R}(space, Dict{Vector{Int}, R}([i] => one(R)))
end

"""
    alg_scalar(space, c::R, A) -> AlgebraTensor{A,R}

Grade-0 element `c·𝟏` in algebra `A`.
"""
function alg_scalar(space::VectorSpace, c::R, ::Type{A}) where {R,A}
    iszero(c) ?
        AlgebraTensor{A,R}(space, Dict{Vector{Int}, R}()) :
        AlgebraTensor{A,R}(space, Dict{Vector{Int}, R}(Int[] => c))
end

# Convenience wrappers for the two Phase-2 algebras
sym_basis_vector(V::VectorSpace, ::Type{R}, i::Int) where R =
    alg_basis_vector(V, R, SymmetricAlgebra, i)

ext_basis_vector(V::VectorSpace, ::Type{R}, i::Int) where R =
    alg_basis_vector(V, R, ExteriorAlgebra, i)

sym_scalar(V::VectorSpace, c::R) where R = alg_scalar(V, c, SymmetricAlgebra)
ext_scalar(V::VectorSpace, c::R) where R = alg_scalar(V, c, ExteriorAlgebra)

Base.zero(::Type{AlgebraTensor{A,R}}, space::VectorSpace) where {A,R} =
    AlgebraTensor{A,R}(space, Dict{Vector{Int}, R}())

Base.one(::Type{AlgebraTensor{A,R}}, space::VectorSpace) where {A,R} =
    alg_scalar(space, one(R), A)

# ── Predicates ────────────────────────────────────────────────────────────────

Base.iszero(t::AlgebraTensor)                               = isempty(t.terms)
Base.:(==)(a::AlgebraTensor{A,R}, b::AlgebraTensor{A,R}) where {A,R} =
    a.space == b.space && a.terms == b.terms
Base.hash(t::AlgebraTensor, h::UInt)                        =
    hash(t.terms, hash(t.space, h))

# ── Grading ───────────────────────────────────────────────────────────────────
# Grade = multi-index length (same semantic as FreeTensor).
# In Sym this is polynomial degree; in Ext this is the exterior power.

function grade(t::AlgebraTensor{A,R}) where {A,R}
    isempty(t.terms) &&
        throw(ArgumentError("grade is undefined for the zero element"))
    gs = unique([length(idx) for idx in keys(t.terms)])
    length(gs) == 1 ||
        throw(ArgumentError(
            "Element is not homogeneous; grades present: $(sort(gs))"))
    gs[1]
end

grades(t::AlgebraTensor{A,R}) where {A,R} =
    sort(unique([length(idx) for idx in keys(t.terms)]))

function homogeneous_component(t::AlgebraTensor{A,R}, k::Int) where {A,R}
    terms = Dict{Vector{Int}, R}(
        idx => c for (idx, c) in t.terms if length(idx) == k)
    AlgebraTensor{A,R}(t.space, terms)
end

# ── Arithmetic ────────────────────────────────────────────────────────────────

function _check_alg_space(a::AlgebraTensor, b::AlgebraTensor, op::Symbol)
    a.space == b.space ||
        throw(ArgumentError(
            "Cannot $op AlgebraTensors over different spaces"))
end

function Base.:+(a::AlgebraTensor{A,R}, b::AlgebraTensor{A,R}) where {A,R}
    _check_alg_space(a, b, :add)
    terms = copy(a.terms)
    for (idx, c) in b.terms
        prev = get(terms, idx, zero(R))
        val  = prev + c
        if iszero(val)
            delete!(terms, idx)
        else
            terms[idx] = val
        end
    end
    AlgebraTensor{A,R}(a.space, terms)   # constructor re-normalizes (no-op on canonical terms)
end

Base.:-(a::AlgebraTensor{A,R}, b::AlgebraTensor{A,R}) where {A,R} = a + (-b)

Base.:-(t::AlgebraTensor{A,R}) where {A,R} =
    AlgebraTensor{A,R}(t.space,
        Dict{Vector{Int}, R}(idx => -c for (idx, c) in t.terms))

function Base.:*(c::R, t::AlgebraTensor{A,R}) where {A,R}
    iszero(c) && return zero(AlgebraTensor{A,R}, t.space)
    AlgebraTensor{A,R}(t.space,
        Dict{Vector{Int}, R}(idx => c * v for (idx, v) in t.terms))
end

function Base.:*(t::AlgebraTensor{A,R}, c::R) where {A,R}
    iszero(c) && return zero(AlgebraTensor{A,R}, t.space)
    AlgebraTensor{A,R}(t.space,
        Dict{Vector{Int}, R}(idx => v * c for (idx, v) in t.terms))
end

Base.:*(n::Integer, t::AlgebraTensor{A,R}) where {A,R} = R(n) * t
Base.:*(t::AlgebraTensor{A,R}, n::Integer) where {A,R} = t * R(n)

"""
    a * b  — algebra product in T(V)/I

For each pair of terms (I, cI) from `a` and (J, cJ) from `b`:
  1. Concatenate the indices: K = I ∥ J  (the T(V) product)
  2. Apply `normalize_A(K, cI·cJ)` to reduce to canonical form
  3. Accumulate into the result, combining like terms

This is literally `normalize_A ∘ concatenate`, the quotient map applied
term-by-term.  The algebra type A is fixed at compile time via dispatch.
"""
function Base.:*(a::AlgebraTensor{A,R}, b::AlgebraTensor{A,R}) where {A,R}
    _check_alg_space(a, b, :multiply)
    isempty(a.terms) && return zero(AlgebraTensor{A,R}, a.space)
    isempty(b.terms) && return zero(AlgebraTensor{A,R}, a.space)
    terms = Dict{Vector{Int}, R}()
    for (ai, ac) in a.terms, (bi, bc) in b.terms
        raw_idx  = vcat(ai, bi)        # step 1: concatenation (T(V) product)
        raw_coef = ac * bc
        iszero(raw_coef) && continue
        (nidx, ncoef) = normalize(A(), raw_idx, raw_coef)   # step 2: quotient
        iszero(ncoef)  && continue
        prev = get(terms, nidx, zero(R))
        val  = prev + ncoef
        if iszero(val)
            delete!(terms, nidx)
        else
            terms[nidx] = val
        end
    end
    AlgebraTensor{A,R}(a.space, terms)
end

"""
    a ∧ b  — wedge (exterior) product

Notation alias for `a * b` when both operands are in `ExteriorAlgebra`.
Lets you write `e1 ∧ e2 ∧ e3` naturally.  Associativity is inherited from `*`.
"""
∧(a::AlgebraTensor{ExteriorAlgebra,R},
  b::AlgebraTensor{ExteriorAlgebra,R}) where R = a * b

# ── Basis enumeration ─────────────────────────────────────────────────────────

"""
    all_sym_grade_k_indices(space, k) -> Vector{Vector{Int}}

All non-decreasing multi-indices of length `k` from `{1,…,n}`.
These are the grade-`k` basis indices of Sym(V); count is `C(n+k−1, k)`.
"""
function all_sym_grade_k_indices(space::VectorSpace, k::Int)
    k >= 0 || throw(ArgumentError("Grade k must be ≥ 0, got $k"))
    k == 0 && return [Int[]]
    n = space.n
    n == 0 && return Vector{Vector{Int}}()
    result = Vector{Vector{Int}}()
    _sym_enum!(result, Int[], n, k, 1)
    result
end

function _sym_enum!(result, current, n, k, min_val)
    if length(current) == k
        push!(result, copy(current))
        return
    end
    for i in min_val:n
        push!(current, i)
        _sym_enum!(result, current, n, k, i)   # allow repeats: next ≥ i
        pop!(current)
    end
end

"""
    all_ext_grade_k_indices(space, k) -> Vector{Vector{Int}}

All strictly increasing multi-indices of length `k` from `{1,…,n}`.
These are the grade-`k` basis indices of Λ(V); count is `C(n, k)`.
"""
function all_ext_grade_k_indices(space::VectorSpace, k::Int)
    k >= 0 || throw(ArgumentError("Grade k must be ≥ 0, got $k"))
    k == 0 && return [Int[]]
    n = space.n
    (n == 0 || k > n) && return Vector{Vector{Int}}()
    result = Vector{Vector{Int}}()
    _ext_enum!(result, Int[], n, k, 1)
    result
end

function _ext_enum!(result, current, n, k, min_val)
    if length(current) == k
        push!(result, copy(current))
        return
    end
    for i in min_val:n
        push!(current, i)
        _ext_enum!(result, current, n, k, i + 1)  # strictly increasing: next > i
        pop!(current)
    end
end

"""
    sym_homogeneous_basis(space, R, k) -> Vector{AlgebraTensor{SymmetricAlgebra,R}}

All grade-`k` basis elements of Sym(V).  Count: `C(n+k−1, k)`.
"""
function sym_homogeneous_basis(space::VectorSpace, ::Type{R}, k::Int) where R
    [alg_basis_element(space, R, SymmetricAlgebra, idx)
     for idx in all_sym_grade_k_indices(space, k)]
end

"""
    ext_homogeneous_basis(space, R, k) -> Vector{AlgebraTensor{ExteriorAlgebra,R}}

All grade-`k` basis elements of Λ(V).  Count: `C(n, k)`.
"""
function ext_homogeneous_basis(space::VectorSpace, ::Type{R}, k::Int) where R
    [alg_basis_element(space, R, ExteriorAlgebra, idx)
     for idx in all_ext_grade_k_indices(space, k)]
end

"""
    sym_grade_dim(n, k) -> Int

Dimension of the grade-`k` piece of Sym(V): `binomial(n+k−1, k)`.
Equals the number of degree-`k` monomials in `n` variables.
"""
sym_grade_dim(n::Int, k::Int) =
    k == 0 ? 1 : (n == 0 ? 0 : binomial(n + k - 1, k))

"""
    ext_grade_dim(n, k) -> Int

Dimension of the grade-`k` piece of Λ(V): `binomial(n, k)`.  Zero for `k > n`.
"""
ext_grade_dim(n::Int, k::Int) = k > n ? 0 : binomial(n, k)

# ── Display ───────────────────────────────────────────────────────────────────

_alg_sep(::SymmetricAlgebra) = "·"
_alg_sep(::ExteriorAlgebra)  = "∧"

function _alg_idx_str(space::VectorSpace, alg, idx::Vector{Int})
    isempty(idx) && return "𝟏"
    join(string.(space.labels[idx]), _alg_sep(alg))
end

function Base.show(io::IO, t::AlgebraTensor{A,R}) where {A,R}
    if isempty(t.terms)
        print(io, "0")
        return
    end
    alg    = A()
    sorted = sort(collect(t.terms); by = kv -> (length(kv[1]), kv[1]))
    parts  = String[]
    for (idx, c) in sorted
        s = _alg_idx_str(t.space, alg, idx)
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
# Note: grade, grades, homogeneous_component are already exported by free_tensor.jl;
# adding new methods here for AlgebraTensor requires no additional export.

export FreeAlgebra, SymmetricAlgebra, ExteriorAlgebra
export AlgebraTensor
export alg_basis_element, alg_basis_vector, alg_scalar
export sym_basis_vector, sym_scalar, sym_homogeneous_basis
export ext_basis_vector, ext_scalar, ext_homogeneous_basis
export all_sym_grade_k_indices, all_ext_grade_k_indices
export sym_grade_dim, ext_grade_dim
export ∧
