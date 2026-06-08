# ── Scalar ring abstraction ──────────────────────────────────────────────────
#
# Everything in Tensorsmith is parametric over a scalar ring R.
# R must satisfy the following duck-typed interface:
#
#   zero(R), one(R)                          additive and multiplicative identities
#   +(::R, ::R), -(::R, ::R), *(::R, ::R)   ring operations
#   -(::R)                                   additive inverse
#   ==(::R, ::R)                             exact equality
#   iszero(::R)                              test for zero
#   R(n::Integer)                            construction from integers
#
# ── Default exact ring: Rational{BigInt} ────────────────────────────────────
#
# DESIGN DECISION (flagged for Kainaan):
#
# The open question was: Symbolics.Num vs. a lighter exact polynomial ring as
# the Phase 0–4 default.  Choice: Rational{BigInt}.  Rationale:
#
#  1. Zero dependencies.  Rational{BigInt} is in Julia's stdlib.  Symbolics.jl
#     adds ~30 transitive packages and 30–60 s of first-load JIT time.  There
#     is no algebraic benefit to symbolic coefficients until Phase 5, where
#     build_function is needed.
#
#  2. Sufficient exactness.  The section maps (symmetrization and
#     antisymmetrization, Phase 4) divide by k!.  Rational{BigInt} contains ℚ
#     exactly, satisfying the mathematical requirement.
#
#  3. Clean upgrade path.  All types are parametric in R.  Symbolics.Num slides
#     in as another concrete R in Phase 5 with no architectural change.
#
#  4. Test clarity.  Rational literals make axiom tests self-evident.
#     Symbolic coefficients would require isequal(simplify(a - b), 0) instead
#     of a == b, which obscures the algebra.
#
# If you want symbolic coefficients now (to explore with free variables), you
# can pass R = Symbolics.Num manually to any constructor — everything is
# generic.  The default will be upgraded to Symbolics.Num in Phase 5.
#
# ── GASmith fixture delivery (flagged for Kainaan) ──────────────────────────
#
# DESIGN DECISION: JSON data files in test/fixtures/.
#
# Phase 3 will cross-check the Clifford leaf against GASmith outputs blade
# for blade.  The simplest approach is to have GASmith export a JSON file of
# pre-computed products (input blades → output blade + sign), which the Julia
# test suite loads and replays.  This requires no FFI, no C++ toolchain in CI,
# and the fixtures are version-controllable.
#
# When Phase 3 begins we will ask you to run a GASmith export script and drop
# the result in test/fixtures/gasmith_cl3.json (or similar).  If you later
# want live cross-checking via FFI (ccall into the shared library), that can be
# added as an optional code path without changing the architecture.

"""
    ExactRing

The default scalar ring for exact arithmetic: `Rational{BigInt}`.

Chosen for Phases 0–4 because it:
- is in Julia's stdlib (zero extra dependencies),
- contains ℚ (required for section maps in Phase 4),
- makes test assertions trivially exact.

See the design rationale comment at the top of `scalar_ring.jl`.
"""
const ExactRing = Rational{BigInt}

"""
    contains_rationals(::Type{R}) -> Bool

Return `true` if the scalar ring `R` contains the rationals (ℚ ⊆ R).

This is a *necessary* condition for the symmetrization and antisymmetrization
section maps introduced in Phase 4.  Calling those maps with a ring that does
not satisfy this predicate is a programming error and will throw.
"""
contains_rationals(::Type{<:Rational}) = true
contains_rationals(::Type)             = false

export ExactRing, contains_rationals
