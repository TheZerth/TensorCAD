# ── Phase 6: Tensor-calculus core ────────────────────────────────────────────
#
# MATHEMATICAL SETUP
#
# A VectorSpace V has a dual V* with cobasis {e¹,…,eⁿ} satisfying the natural
# pairing ⟨eⁱ, eⱼ⟩ = δⁱⱼ.  A tensor of type (p,q) lives in
#
#     V^{⊗p} ⊗ (V*)^{⊗q}
#
# with p contravariant (upper) slots and q covariant (lower) slots, in a fixed
# slot order.  We realize this by generalizing the multi-index: each slot carries
# both an index in 1:n AND a variance tag, `Up` (contravariant) or `Down`
# (covariant).  The free tensor algebra T(V) is exactly the all-`Up` special
# case, and MixedTensor reduces to it on all-`Up` keys (see `as_free_tensor`).
#
# REPRESENTATION
#
# A slot is `(i, Up())` or `(i, Down())`; a key is a `Vector` of slots, stored
# as `MixedIndex`.  The empty key is the grade-0 / scalar component.  Storage is
# the same sparse `Dict{MixedIndex, R}` as the other element types, canonicalized
# (zero-pruned, index-range-checked) in the constructor.
#
# DUAL SPACE
#
# V* is not a separate struct: a covector is simply a MixedTensor whose single
# slot is `Down`.  `mixed_basis_covector(V, R, i)` builds the cobasis covector
# eⁱ.  The pairing ⟨eⁱ, eⱼ⟩ = δⁱⱼ is exactly what `contract` computes.

# ── Variance tags ─────────────────────────────────────────────────────────────
#
# Singleton types, present purely for dispatch and to label each slot.

"""
    Variance

Supertype of the two slot variances [`Up`](@ref) (contravariant) and
[`Down`](@ref) (covariant).
"""
abstract type Variance end

"""`Up` — a contravariant (upper) slot; its index labels a basis vector `eᵢ ∈ V`."""
struct Up   <: Variance end

"""`Down` — a covariant (lower) slot; its index labels a cobasis covector `eⁱ ∈ V*`."""
struct Down <: Variance end

# Slot = (index, variance); a key is a vector of slots.
const Slot       = Tuple{Int, Variance}
const MixedIndex = Vector{Slot}

# ── MixedTensor ───────────────────────────────────────────────────────────────

"""
    MixedTensor{R} <: AbstractTensorElement{R}

A mixed-variance tensor over scalar ring `R`: an element of
`V^{⊗p} ⊗ (V*)^{⊗q}` for some slot pattern.

Stored as a sparse `Dict{MixedIndex, R}` mapping per-slot `(index, variance)`
sequences to coefficients; the empty key is the scalar component.

This is a *sibling* of [`FreeTensor`](@ref), not a replacement: `FreeTensor`
keeps its lean all-contravariant `Vector{Int}` representation and all of its
behaviour.  An all-`Up` `MixedTensor` reproduces `T(V)` semantics exactly —
see [`MixedTensor(::FreeTensor)`](@ref) and [`as_free_tensor`](@ref).

## Constructors
- [`mixed_basis_element(V, R, slots)`](@ref) — a pure tensor from explicit slots
- [`mixed_basis_vector(V, R, i)`](@ref)      — the vector `eᵢ`        (type (1,0))
- [`mixed_basis_covector(V, R, i)`](@ref)    — the covector `eⁱ`      (type (0,1))
- [`mixed_scalar(V, c)`](@ref), `mixed_zero`, `mixed_one`
- [`identity_tensor(V, R)`](@ref)            — the (1,1) identity `Σᵢ eᵢ⊗eⁱ`

## Operations
- `a + b`, `a - b`, `-a`, `c*a`, `a*c`       — linear structure
- `a ⊗ b` (or `a * b`, `tensor_product`)     — tensor product, slots concatenated
- [`contract(t, a, b)`](@ref), [`trace`](@ref) — metric-free contraction
- [`lower(t, slot, g)`](@ref), [`raise(t, slot, g)`](@ref) — musical isomorphisms

## Type bookkeeping
- [`tensor_type(t)`](@ref) → `(p, q)`, [`variance_pattern(t)`](@ref)
"""
struct MixedTensor{R} <: AbstractTensorElement{R}
    space :: VectorSpace
    terms :: Dict{MixedIndex, R}

    function MixedTensor{R}(space::VectorSpace, terms::Dict{MixedIndex, R}) where R
        n      = space.n
        pruned = Dict{MixedIndex, R}()
        for (slots, c) in terms
            iszero(c) && continue
            for (i, _) in slots
                1 <= i <= n || throw(ArgumentError(
                    "slot index $i out of range 1:$n"))
            end
            pruned[slots] = c
        end
        new(space, pruned)
    end
end

# AbstractTensorElement hooks (grade, grades, ==, hash, iszero,
# homogeneous_component come from abstract_tensor.jl).
_eq_key(t::MixedTensor)    = t.space
base_space(t::MixedTensor) = t.space
_rebuild(t::MixedTensor{R}, terms::Dict{MixedIndex, R}) where R =
    MixedTensor{R}(t.space, terms)

# ── Accumulation helper (generic over key type) ──────────────────────────────

# Add `coef` into `result[key]`, deleting the key if the running sum hits zero.
function _acc!(result::Dict{K, R}, key::K, coef::R) where {K, R}
    iszero(coef) && return
    prev = get(result, key, zero(R))
    val  = prev + coef
    if iszero(val)
        delete!(result, key)
    else
        result[key] = val
    end
end

# ── Constructors ──────────────────────────────────────────────────────────────

"""
    mixed_basis_element(space, R, slots) -> MixedTensor{R}

The pure tensor with coefficient `one(R)` and the given `slots`, each a
`(index, variance)` pair (`(i, Up())` or `(i, Down())`).

# Example
```julia
V = VectorSpace(3)
# e₂ ⊗ e³  (a type-(1,1) tensor)
t = mixed_basis_element(V, Rational{BigInt}, [(2, Up()), (3, Down())])
```
"""
function mixed_basis_element(space::VectorSpace, ::Type{R},
                             slots::AbstractVector{<:Tuple{Int, Variance}}) where R
    MixedTensor{R}(space, Dict{MixedIndex, R}(MixedIndex(slots) => one(R)))
end

"""
    mixed_basis_vector(space, R, i) -> MixedTensor{R}

The contravariant basis vector `eᵢ ∈ V`, a type-(1,0) tensor with one `Up` slot.
"""
mixed_basis_vector(space::VectorSpace, ::Type{R}, i::Int) where R =
    mixed_basis_element(space, R, Slot[(_check_index(space, i), Up())])

"""
    mixed_basis_covector(space, R, i) -> MixedTensor{R}

The covariant cobasis covector `eⁱ ∈ V*`, a type-(0,1) tensor with one `Down`
slot.  Satisfies the natural pairing `⟨eⁱ, eⱼ⟩ = δⁱⱼ` via [`contract`](@ref).
"""
mixed_basis_covector(space::VectorSpace, ::Type{R}, i::Int) where R =
    mixed_basis_element(space, R, Slot[(_check_index(space, i), Down())])

function _check_index(space::VectorSpace, i::Int)
    1 <= i <= space.n ||
        throw(ArgumentError("Basis index $i out of range for $(space)"))
    i
end

"""
    mixed_scalar(space, c::R) -> MixedTensor{R}

The grade-0 element `c·𝟏` (type (0,0)).
"""
mixed_scalar(space::VectorSpace, c::R) where R =
    iszero(c) ? MixedTensor{R}(space, Dict{MixedIndex, R}()) :
                MixedTensor{R}(space, Dict{MixedIndex, R}(MixedIndex() => c))

mixed_zero(space::VectorSpace, ::Type{R}) where R =
    MixedTensor{R}(space, Dict{MixedIndex, R}())

mixed_one(space::VectorSpace, ::Type{R}) where R = mixed_scalar(space, one(R))

Base.zero(::Type{MixedTensor{R}}, space::VectorSpace) where R = mixed_zero(space, R)
Base.one(::Type{MixedTensor{R}}, space::VectorSpace) where R  = mixed_one(space, R)

"""
    identity_tensor(space, R) -> MixedTensor{R}

The identity / Kronecker-delta tensor `δ = Σᵢ eᵢ ⊗ eⁱ`, of type (1,1).
Contracting its two slots returns the scalar `n = dim V` (see [`trace`](@ref)).
"""
function identity_tensor(space::VectorSpace, ::Type{R}) where R
    terms = Dict{MixedIndex, R}()
    for i in 1:space.n
        terms[Slot[(i, Up()), (i, Down())]] = one(R)
    end
    MixedTensor{R}(space, terms)
end

# ── Conversions to/from FreeTensor (the all-Up bridge) ───────────────────────

"""
    MixedTensor(t::FreeTensor{R}) -> MixedTensor{R}

Embed a free tensor as an all-contravariant (all-`Up`) `MixedTensor`.  Every
`T(V)` operation on `t` corresponds to the matching `MixedTensor` operation on
its image, which is what makes the mixed layer a faithful generalization of
T(V).
"""
function MixedTensor(t::FreeTensor{R}) where R
    terms = Dict{MixedIndex, R}()
    for (idx, c) in t.terms
        terms[MixedIndex(Slot[(i, Up()) for i in idx])] = c
    end
    MixedTensor{R}(t.space, terms)
end

"""
    as_free_tensor(t::MixedTensor{R}) -> FreeTensor{R}

Project an all-contravariant (all-`Up`) `MixedTensor` back to a `FreeTensor`.
Throws `ArgumentError` if any slot is covariant (`Down`), since `T(V)` has no
covariant slots.
"""
function as_free_tensor(t::MixedTensor{R}) where R
    terms = Dict{Vector{Int}, R}()
    for (slots, c) in t.terms
        all(v === Up() for (_, v) in slots) || throw(ArgumentError(
            "as_free_tensor requires an all-contravariant (all-Up) tensor; " *
            "found a covariant slot in $slots"))
        terms[Int[i for (i, _) in slots]] = c
    end
    FreeTensor{R}(t.space, terms)
end

# ── Type bookkeeping ──────────────────────────────────────────────────────────

"""
    variance_pattern(t::MixedTensor) -> Vector{Variance}

The common per-slot variance sequence of `t` (e.g. `[Up(), Down()]`).

Throws `ArgumentError` if `t` is the zero tensor (pattern undefined) or if its
terms do not all share one variance pattern — a genuine type-(p,q) tensor has a
single, well-defined slot structure.
"""
function variance_pattern(t::MixedTensor)
    isempty(t.terms) &&
        throw(ArgumentError("variance pattern is undefined for the zero tensor"))
    pats = unique(Variance[v for (_, v) in slots] for slots in keys(t.terms))
    length(pats) == 1 || throw(ArgumentError(
        "tensor mixes variance patterns ($(collect(pats))); it is not a " *
        "homogeneous type-(p,q) tensor"))
    first(pats)
end

"""
    tensor_type(t::MixedTensor) -> Tuple{Int,Int}

The valence `(p, q)` of `t`: `p` contravariant (`Up`) and `q` covariant (`Down`)
slots.  Requires a homogeneous variance pattern (see [`variance_pattern`](@ref)).
"""
function tensor_type(t::MixedTensor)
    pat = variance_pattern(t)
    p = count(v -> v === Up(),   pat)
    q = count(v -> v === Down(), pat)
    (p, q)
end

# ── Arithmetic ────────────────────────────────────────────────────────────────

function _check_mixed_space(a::MixedTensor, b::MixedTensor, op::Symbol)
    a.space == b.space || throw(ArgumentError(
        "Cannot $op MixedTensors over different spaces:\n  $(a.space)\n  $(b.space)"))
end

function Base.:+(a::MixedTensor{R}, b::MixedTensor{R}) where R
    _check_mixed_space(a, b, :add)
    terms = copy(a.terms)
    for (slots, c) in b.terms
        _acc!(terms, slots, c)
    end
    MixedTensor{R}(a.space, terms)
end

Base.:-(a::MixedTensor{R}, b::MixedTensor{R}) where R = a + (-b)

Base.:-(t::MixedTensor{R}) where R =
    MixedTensor{R}(t.space, Dict{MixedIndex, R}(s => -c for (s, c) in t.terms))

function Base.:*(c::R, t::MixedTensor{R}) where R
    iszero(c) && return mixed_zero(t.space, R)
    MixedTensor{R}(t.space, Dict{MixedIndex, R}(s => c * v for (s, v) in t.terms))
end
Base.:*(t::MixedTensor{R}, c::R) where R = c * t
Base.:*(n::Integer, t::MixedTensor{R}) where R = R(n) * t
Base.:*(t::MixedTensor{R}, n::Integer) where R = R(n) * t

"""
    a ⊗ b  ≡  a * b  ≡  tensor_product(a, b)

The tensor product of two `MixedTensor`s: slot sequences are concatenated and
coefficients multiplied.  A type-(p,q) and a type-(p′,q′) tensor produce a
type-(p+p′, q+q′) tensor (slots kept in order — variance is carried along).
"""
function Base.:*(a::MixedTensor{R}, b::MixedTensor{R}) where R
    _check_mixed_space(a, b, :multiply)
    (isempty(a.terms) || isempty(b.terms)) && return mixed_zero(a.space, R)
    terms = Dict{MixedIndex, R}()
    for (ai, ac) in a.terms, (bi, bc) in b.terms
        _acc!(terms, vcat(ai, bi), ac * bc)
    end
    MixedTensor{R}(a.space, terms)
end

tensor_product(a::MixedTensor{R}, b::MixedTensor{R}) where R = a * b
⊗(a::MixedTensor{R}, b::MixedTensor{R}) where R              = a * b

# ── Contraction (metric-free) ─────────────────────────────────────────────────

"""
    contract(t::MixedTensor, a::Int, b::Int) -> MixedTensor

Contract upper slot `a` against lower slot `b` via the natural pairing
`⟨eⁱ, eⱼ⟩ = δⁱⱼ`: identify the two indices, sum over `1:n`, and delete both
slots.  Maps a type-(p,q) tensor to type-(p−1, q−1).  **No metric is required.**

`a` must be a contravariant (`Up`) slot and `b` a covariant (`Down`) slot;
otherwise an `ArgumentError` is thrown.  In the sparse representation the
δ-sum is automatic: a stored basis term survives iff its index at slot `a`
equals its index at slot `b`, and the surviving term is that term with both
slots removed.

# Example
```julia
V = VectorSpace(3)
trace_of_identity = contract(identity_tensor(V, Rational{BigInt}), 1, 2)
# == scalar 3
```
"""
function contract(t::MixedTensor{R}, a::Int, b::Int) where R
    a == b && throw(ArgumentError(
        "cannot contract a slot with itself (a = b = $a)"))
    pat = variance_pattern(t)        # also rejects the zero / mixed-type tensor
    k   = length(pat)
    (1 <= a <= k && 1 <= b <= k) || throw(ArgumentError(
        "slot positions ($a, $b) out of range 1:$k"))
    pat[a] === Up() || throw(ArgumentError(
        "contract requires slot $a to be contravariant (Up); it is $(pat[a])"))
    pat[b] === Down() || throw(ArgumentError(
        "contract requires slot $b to be covariant (Down); it is $(pat[b])"))

    result = Dict{MixedIndex, R}()
    for (slots, c) in t.terms
        (slots[a][1] == slots[b][1]) || continue   # δ: only matching indices survive
        newslots = MixedIndex(Slot[slots[m] for m in eachindex(slots)
                                   if m != a && m != b])
        _acc!(result, newslots, c)
    end
    MixedTensor{R}(t.space, result)
end

"""
    trace(t::MixedTensor) -> MixedTensor

Full trace of a type-(1,1) tensor: `contract(t, 1, 2)`, yielding a grade-0
scalar.  Throws `ArgumentError` if `t` is not of type (1,1).
"""
function trace(t::MixedTensor{R}) where R
    tensor_type(t) == (1, 1) || throw(ArgumentError(
        "trace is defined for type-(1,1) tensors; got type $(tensor_type(t))"))
    contract(t, 1, 2)
end

# ── Exact matrix inverse (zero non-stdlib deps) ──────────────────────────────
#
# Raising indices needs the inverse metric gⁱʲ.  We compute it with a
# cofactor / adjugate expansion rather than `LinearAlgebra.inv` so the core
# keeps its zero-non-stdlib-dependency invariant, and — just as importantly —
# so the routine is robust over a *symbolic* ring: cofactor expansion needs
# only ring +,−,× plus a single final division by det(g), and never has to
# decide whether a pivot is zero (which is undecidable for `Symbolics.Num`).
# Metric spaces are tiny, so the O(n!) expansion is irrelevant in practice.

function _minor(g::Matrix{R}, r::Int, c::Int) where R
    n    = size(g, 1)
    rows = Int[i for i in 1:n if i != r]
    cols = Int[j for j in 1:n if j != c]
    g[rows, cols]
end

function _det(g::Matrix{R}) where R
    n = size(g, 1)
    n == 0 && return one(R)
    n == 1 && return g[1, 1]
    n == 2 && return g[1, 1] * g[2, 2] - g[1, 2] * g[2, 1]
    acc = zero(R)
    for j in 1:n
        c = g[1, j]
        iszero(c) && continue
        term = c * _det(_minor(g, 1, j))
        acc  = isodd(1 + j) ? acc - term : acc + term   # sign (−1)^{1+j}
    end
    acc
end

# Exact inverse of a square matrix over R; throws if singular.
function _matrix_inverse(g::Matrix{R}) where R
    n = size(g, 1)
    n == 0 && return Matrix{R}(undef, 0, 0)
    d = _det(g)
    iszero(d) && throw(ArgumentError(
        "metric is degenerate (determinant = 0): the inverse metric does not " *
        "exist, so indices cannot be raised. A metric with null directions " *
        "(signature r > 0) is degenerate."))
    inv = Matrix{R}(undef, n, n)
    for i in 1:n, j in 1:n
        # adjugate is the transpose of the cofactor matrix: adjᵢⱼ = Cⱼᵢ
        cofactor = _det(_minor(g, j, i))
        signed   = isodd(i + j) ? -cofactor : cofactor
        inv[i, j] = signed / d
    end
    inv
end

"""
    inverse_metric(g::Metric{R}) -> Metric{R}

The inverse metric `gⁱʲ`, computed exactly via a cofactor/adjugate expansion
(no floating point, no non-stdlib dependency).  Throws `ArgumentError` if `g`
is degenerate (`det g = 0`, i.e. signature `r > 0`), since then `gⁱʲ` does not
exist.  The result is symmetric, like `g`.
"""
function inverse_metric(g::Metric{R}) where R
    Metric{R}(g.space, _matrix_inverse(g.g))
end

# ── Musical isomorphisms (need the metric) ────────────────────────────────────

function _check_metric_space(t::MixedTensor, g::Metric, op::Symbol)
    t.space == g.space || throw(ArgumentError(
        "Cannot $op: tensor space $(t.space) does not match metric space $(g.space)"))
end

"""
    lower(t::MixedTensor, slot::Int, g::Metric) -> MixedTensor

Lower (♭) the contravariant index at position `slot` using the metric:

    T…ⁱ…  ↦  Σⱼ g_{ij} T…ʲ…   (the slot becomes covariant)

Maps type (p,q) → (p−1, q+1).  Requires `slot` to be an `Up` slot; needs no
matrix inverse.

See also [`raise`](@ref) for the inverse operation.
"""
function lower(t::MixedTensor{R}, slot::Int, g::Metric{R}) where R
    _check_metric_space(t, g, :lower)
    pat = variance_pattern(t)
    (1 <= slot <= length(pat)) || throw(ArgumentError(
        "slot $slot out of range 1:$(length(pat))"))
    pat[slot] === Up() || throw(ArgumentError(
        "lower requires slot $slot to be contravariant (Up); it is $(pat[slot])"))

    n      = t.space.n
    result = Dict{MixedIndex, R}()
    for (slots, c) in t.terms
        i = slots[slot][1]
        for j in 1:n
            gij = g.g[i, j]
            iszero(gij) && continue
            newslots = copy(slots)
            newslots[slot] = (j, Down())
            _acc!(result, newslots, c * gij)
        end
    end
    MixedTensor{R}(t.space, result)
end

"""
    raise(t::MixedTensor, slot::Int, g::Metric) -> MixedTensor

Raise (♯) the covariant index at position `slot` using the **inverse** metric:

    T…ᵢ…  ↦  Σⱼ gⁱʲ T…ⱼ…   (the slot becomes contravariant)

Maps type (p,q) → (p+1, q−1).  Requires `slot` to be a `Down` slot and `g` to be
invertible; for a degenerate metric (`det g = 0`, signature `r > 0`) raising is
undefined and an `ArgumentError` is thrown (see [`inverse_metric`](@ref)).

`raise` and [`lower`](@ref) are mutually inverse on the same slot:
`raise(lower(t, s, g), s, g) == t` for every invertible `g`.
"""
function raise(t::MixedTensor{R}, slot::Int, g::Metric{R}) where R
    _check_metric_space(t, g, :raise)
    pat = variance_pattern(t)
    (1 <= slot <= length(pat)) || throw(ArgumentError(
        "slot $slot out of range 1:$(length(pat))"))
    pat[slot] === Down() || throw(ArgumentError(
        "raise requires slot $slot to be covariant (Down); it is $(pat[slot])"))

    ginv   = inverse_metric(g).g       # throws ArgumentError if g is degenerate
    n      = t.space.n
    result = Dict{MixedIndex, R}()
    for (slots, c) in t.terms
        i = slots[slot][1]
        for j in 1:n
            gij = ginv[i, j]
            iszero(gij) && continue
            newslots = copy(slots)
            newslots[slot] = (j, Up())
            _acc!(result, newslots, c * gij)
        end
    end
    MixedTensor{R}(t.space, result)
end

# ── Display ───────────────────────────────────────────────────────────────────
#
# A contravariant slot prints as its label (a vector eᵢ); a covariant slot is
# marked with a trailing `*` to flag the dual covector eⁱ ∈ V*.

function _slot_str(space::VectorSpace, slot::Slot)
    i, v = slot
    lbl  = string(space.labels[i])
    v === Down() ? lbl * "*" : lbl
end

function _mixed_idx_str(space::VectorSpace, slots::MixedIndex)
    isempty(slots) && return "𝟏"
    join((_slot_str(space, s) for s in slots), "⊗")
end

function Base.show(io::IO, t::MixedTensor{R}) where R
    if isempty(t.terms)
        print(io, "0")
        return
    end
    sorted = sort(collect(t.terms); by = kv -> (length(kv[1]), string(kv[1])))
    parts  = String[]
    for (slots, c) in sorted
        s = _mixed_idx_str(t.space, slots)
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

# ── Usage vignette ────────────────────────────────────────────────────────────
#
# A small Euclidean example tying the pieces together (g = I on ℝ³):
#
#   V = VectorSpace(3); R = Rational{BigInt}
#   g = signature_metric(V, R, 3, 0, 0)        # Euclidean metric
#   u = mixed_basis_vector(V, R, 1)            # e₁,  type (1,0)
#   v = mixed_basis_vector(V, R, 2)            # e₂,  type (1,0)
#
#   vlow = lower(v, 1, g)                      # e₂ ♭ → covector, type (0,1)
#   ip   = contract(u ⊗ vlow, 1, 2)            # ⟨u, v⟩ = g(e₁, e₂) = 0  (scalar)
#
#   raise(lower(v, 1, g), 1, g) == v           # ♯∘♭ = id  (round-trip)
#   contract(identity_tensor(V, R), 1, 2) == mixed_scalar(V, R(3))   # δ-trace = n
#
# Raising over a degenerate metric is an explicit error, never a silent result:
#   raise(mixed_basis_covector(V, R, 3), 1, signature_metric(V, R, 2, 0, 1))
#   # → ArgumentError: metric is degenerate (determinant = 0) ...

# ── Exports ───────────────────────────────────────────────────────────────────

export Variance, Up, Down,
       MixedTensor,
       mixed_basis_element, mixed_basis_vector, mixed_basis_covector,
       mixed_scalar, mixed_zero, mixed_one,
       identity_tensor, as_free_tensor,
       variance_pattern, tensor_type,
       contract, trace,
       lower, raise,
       inverse_metric
