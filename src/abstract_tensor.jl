# ── Abstract tensor element interface (Phase 6 refactor) ─────────────────────
#
# Every concrete element type in Tensorsmith — FreeTensor, AlgebraTensor,
# CliffordTensor, and (Phase 6) MixedTensor — is a sparse map from a key
# (some flavour of multi-index) to a coefficient in the scalar ring R.  They
# share a large slice of behaviour: grading, the zero test, equality, hashing,
# and graded projection.  That shared behaviour lives here, dispatched on the
# abstract supertype, so each concrete type contributes only what is genuinely
# specific to it.
#
# Each concrete type must provide three small hooks:
#
#   terms(t)       → the coefficient Dict           (default: the `.terms` field)
#   base_space(t)  → the underlying VectorSpace
#   _eq_key(t)     → the value that, together with `terms`, defines identity
#                    (the `.space` for Free/Algebra/Mixed, the `.metric` for
#                    Clifford — a Clifford element is not pinned down by its
#                    space alone)
#   _rebuild(t, d) → a new element of the *same* concrete type carrying Dict `d`
#
# Everything below is written once against `AbstractTensorElement` and reused.

"""
    AbstractTensorElement{R}

Supertype of every Tensorsmith element over scalar ring `R`:
[`FreeTensor`](@ref), [`AlgebraTensor`](@ref), [`CliffordTensor`](@ref), and
[`MixedTensor`](@ref).

All subtypes store a sparse `Dict` of (multi-index → coefficient) and share the
grading, equality, hashing, and graded-projection interface defined on this
supertype.  See [`grade`](@ref), [`grades`](@ref), and
[`homogeneous_component`](@ref).
"""
abstract type AbstractTensorElement{R} end

# ── Shared hooks (default implementations) ───────────────────────────────────

"""
    terms(t::AbstractTensorElement) -> Dict

The sparse coefficient dictionary backing `t` (multi-index → coefficient).
"""
terms(t::AbstractTensorElement) = t.terms

# ── Predicates, equality, hashing ────────────────────────────────────────────

Base.iszero(t::AbstractTensorElement) = isempty(terms(t))

# Two elements of the *same* concrete type are equal iff they agree on their
# identity key (space or metric) and on every stored coefficient.  Elements of
# different concrete types fall through to Base's `===` (i.e. not equal).
Base.:(==)(a::T, b::T) where {T <: AbstractTensorElement} =
    _eq_key(a) == _eq_key(b) && terms(a) == terms(b)

Base.hash(t::AbstractTensorElement, h::UInt) =
    hash(terms(t), hash(_eq_key(t), h))

# ── Grading ──────────────────────────────────────────────────────────────────
#
# Grade = the common length of every stored key.  For FreeTensor this is the
# tensor rank, for Sym the polynomial degree, for Λ/Cl the exterior degree, and
# for MixedTensor the total slot count p+q.

"""
    grade(t::AbstractTensorElement) -> Int

The grade of a *homogeneous* element — the common length of all its keys.

Throws `ArgumentError` if `t` is the zero element (grade undefined) or if `t`
mixes grades.  Use [`grades`](@ref) to inspect an arbitrary element and
[`homogeneous_component`](@ref) to extract a single grade.
"""
function grade(t::AbstractTensorElement)
    isempty(terms(t)) &&
        throw(ArgumentError("grade is undefined for the zero element"))
    gs = unique(length(k) for k in keys(terms(t)))
    length(gs) == 1 ||
        throw(ArgumentError(
            "Element is not homogeneous; grades present: $(sort(collect(gs)))"))
    first(gs)
end

"""
    grades(t::AbstractTensorElement) -> Vector{Int}

Sorted list of all grades present in `t`.  Returns `Int[]` for the zero element.
"""
grades(t::AbstractTensorElement) =
    sort(unique(length(k) for k in keys(terms(t))))

"""
    homogeneous_component(t::AbstractTensorElement, k::Int) -> (same type as t)

Extract the grade-`k` part of `t`, returning an element of the same concrete
type.  Returns the zero element if `t` has no grade-`k` terms.
"""
function homogeneous_component(t::AbstractTensorElement, k::Int)
    _rebuild(t, filter(p -> length(p.first) == k, terms(t)))
end

export AbstractTensorElement
