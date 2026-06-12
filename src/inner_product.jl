# ── Phase L9.1: The Hodge (L²) inner product of cochains and the energy
#    functional ─────────────────────────────────────────────────────────────
#
# MATHEMATICAL CONTRACT (DESIGN.md §15.4, §17.1 — amended L9.1)
#
# For grade-k primal cochains α, β over a base with `can_hodge(b)`, the
# diagonal-Hodge L² pairing is
#
#     ⟨α, β⟩  =  Σ_{c ∈ cells(b,k)}  ⟨α(c) · reversion(β(c))⟩₀ · w_k(c)
#
# where ⟨·⟩₀ is the fibre-level scalar pairing (the grade-0 coefficient of the
# geometric product — `scalar_product(x, reversion(y))`, the standard GA
# multivector inner product) and w_k(c) is the diagonal Hodge weight
# (|dual cell| / |primal cell|), UNIT on the shipped orthogonal unit grids.
# The ⋆/dual-cell content of the continuum formula ∫ α ∧ ⋆β survives here as
# the WEIGHT, not as a paired value: non-unit dual volumes (unstructured
# meshes, a future phase) enter through `_hodge_weight` and nowhere else.
# The result is a scalar in the ring R.
#
# An earlier formulation that paired α(c) directly against the Hodge dual's
# value (⋆β)(dual_cell(c)) via `scalar_product` is RETRACTED as
# grade-degenerate: the grade-0 part of a grade-k × grade-(n−k) fibre product
# vanishes unless k = n−k (§15.4).
#
# THE FOUR CONTRACTS (each tested in test/test_inner_product.jl unless noted):
#
#   1. Bilinearity in both arguments (the fibre pairing is bilinear and the
#      sum is linear).
#   2. Symmetry on EVERY signature: ⟨x·reversion(y)⟩₀ = ⟨y·reversion(x)⟩₀
#      (reversion is an anti-automorphism and the grade-0 part is cyclic), so
#      ⟨α,β⟩ = ⟨β,α⟩ with no signature caveat.
#   3. Definiteness on Euclidean bases (signature (p,0,0)): ⟨α,α⟩ > 0 for
#      nonzero α, = 0 iff α is the zero field — every basis blade of Cl(p,0)
#      has positive square norm ⟨b·reversion(b)⟩₀ = +1.  On Lorentzian bases
#      the form is INDEFINITE: blades containing a negative direction have
#      ⟨b·reversion(b)⟩₀ = −1, so ⟨α,α⟩ may be ≤ 0 for nonzero α.  That is
#      physics (the EM field's ½(E²−B²)-type invariants), not a bug.
#   4. Adjointness: ⟨dα,β⟩ = ⟨α,δβ⟩ EXACTLY, on every signature, every grade
#      transition, boundary-touching configurations included — this is the
#      L8.2.1 definitional contract for δ, certified independently of this
#      file in test/test_adjointness.jl (which deliberately carries its own
#      local copy of the pairing so the operator-level test does not depend
#      on this layer).
#
# BOUNDARY BEHAVIOUR (no open-grid leakage — DESIGN.md §15.4):
# the engine's dual-complex d retains the one-sided dual boundary
# contributions (it is the exact transpose of the primal signed incidence),
# so δ is the full incidence transpose up to value maps and the adjointness
# identity holds exactly everywhere on the open grid.  The continuum boundary
# term ∮ α∧⋆β therefore does not reappear as a numerical discrepancy; it
# reappears as a MODELING COMMITMENT — this δ implicitly imposes the natural
# (Neumann-type) boundary condition.  Alternative boundary conditions are a
# future dedicated phase.
#
# THIS IS THE PAIRING, NOT THE CUP PRODUCT (§17.1): it pairs two cochains to
# a SCALAR — every integrated bilinear functional (total energy, quadratic
# action values, quadratic equations of motion via adjointness) is
# pairing-reachable.  There is no cochain-valued product of cochains here, no
# energy *density* field, and the cup product's exactness no-go never enters
# (integration kills the boundary-term-like local discrepancies).  The cup
# product is a dedicated post-L10 phase.
#
# BLACKBOARD NODE — DELIBERATELY SKIPPED.  An `InnerProductExpr` node would
# be scalar-valued: its "grade" is no grade at all, so it cannot participate
# in `Equation`'s (grade, residence, base) agreement contract without first
# giving scalar expressions a principled typing layer.  That is a coherent
# but separate design (a scalar-equation layer), which the L10 energy
# diagnostic does not need — it calls `total_energy` concretely per step
# through the observer hook.  Forcing the node now would be scope creep
# (the amended L9.1 contract's explicit out); the open n-ary node hierarchy
# means it can arrive purely additively later.

"""
    _hodge_weight(::Type{R}, b::BaseSpace, k::Integer, cell::Integer) -> R

The diagonal Hodge weight of a primal `k`-cell: the measure ratio
`|dual_cell(b,k,cell)| / |cell|`.  **Unit on the shipped orthogonal unit
grids** — this is where the `⋆`/dual-volume content of the continuum pairing
`∫ α ∧ ⋆β` lives, and where non-unit dual volumes (unstructured meshes, a
future phase) will enter; the paired fibre values never carry it.
"""
_hodge_weight(::Type{R}, b::BaseSpace, k::Integer, cell::Integer) where R = one(R)

"""
    inner_product(b::BaseSpace, α::Field, β::Field) -> R

The Hodge (L²) inner product of two same-grade primal cochains over `b`:

```julia
⟨α, β⟩ = Σ_{c ∈ cells(b,k)} scalar_product(α(c), reversion(β(c))) * w_k(c)
```

with the diagonal Hodge weight `w_k(c)` ([`_hodge_weight`](@ref), unit on the
shipped grids).  Returns a scalar in the ring `R` — exact over exact rings.

Contracts (see the file header for the full statements and derivations):
**bilinear** in both arguments; **symmetric** on every signature;
**positive-definite on Euclidean bases** (`⟨α,α⟩ > 0` for nonzero `α`, zero
iff `α` is the zero field) and **indefinite on Lorentzian bases** (nonzero
`α` may have `⟨α,α⟩ ≤ 0` — correct physics, not a bug); and the **adjointness
identity `⟨dα,β⟩ == ⟨α,δβ⟩` holds exactly** on every signature and grade
transition, boundary-touching configurations included (the L8.2.1
definitional contract for `δ`; certified in test/test_adjointness.jl).  The
transpose-based `δ` implicitly imposes the natural (Neumann-type) boundary
condition, so the continuum boundary term is a modeling commitment, not a
numerical leak.

This is the **pairing, not the cup product** (DESIGN.md §17.1): cochain ×
cochain → scalar.  No cochain-valued product and no energy-density field
exist here; those belong to the dedicated post-L10 DEC-product phase.

Requires `can_hodge(b) == true` — a bare graph honestly has no Hodge pairing
(`ArgumentError` naming the missing capability).  Both arguments must be
primal [`Field`](@ref)s over `b` itself, of equal grade; mismatches throw
`ArgumentError`s naming the disagreement.  [`HodgeDualField`](@ref)s are
deliberately not accepted: dual-cochain values carry the star's fibre
complement, so the only principled dual pairing is the pullback
`⟨⋆α,⋆β⟩ := ⟨α,β⟩`, which adds no information and invites signature-sign
confusion — pair the primal fields instead.
"""
function inner_product(b::B, α::Field{R,E,B}, β::Field{R,E,B}
                       ) where {R,E<:CliffordTensor{R},B<:BaseSpace}
    _require_hodge(b, "inner_product")
    α.base === b || throw(ArgumentError(
        "inner_product requires the supplied base to be the first field's own " *
        "base; α lives over a different $(typeof(α.base)) instance"))
    β.base === b || throw(ArgumentError(
        "inner_product requires the supplied base to be the second field's own " *
        "base; β lives over a different $(typeof(β.base)) instance"))
    field_grade(α) == field_grade(β) || throw(ArgumentError(
        "inner_product requires equal grades; α has grade $(field_grade(α)) " *
        "but β has grade $(field_grade(β))"))
    k = field_grade(α)
    s = zero(R)
    # Summing over the union of supports equals the contract's sum over all
    # k-cells: unstored cells evaluate to the fibre zero, whose pairing term
    # vanishes identically (weights cannot resurrect a zero term).
    for c in union(keys(α), keys(β))
        s = s + scalar_product(evaluate(α, c), reversion(evaluate(β, c))) *
                _hodge_weight(R, b, k, c)
    end
    s
end

"""
    field_norm2(b::BaseSpace, α::Field) -> R

The squared L² norm `⟨α, α⟩` ([`inner_product`](@ref) of a field with
itself).  No square root is taken — the value stays in `R` and is exact over
exact rings (named to avoid clashing with `LinearAlgebra.norm`; the root of
an exact scalar is generally irrational).  Positive for nonzero `α` on
Euclidean bases; may be ≤ 0 for nonzero `α` on Lorentzian bases (indefinite
fibre metric — physics, not a bug).
"""
field_norm2(b::BaseSpace, α::Field) = inner_product(b, α, α)

"""
    total_energy(b::BaseSpace, F::Field) -> R

The total-energy functional `½⟨F, F⟩` — the L10 conservation diagnostic for
the electromagnetic bivector field strength `F = dA` (a thin wrapper over
[`inner_product`](@ref); it accepts any grade, since `½⟨α,α⟩` is the natural
quadratic action value of any cochain).

The half is computed as `one(R)/R(2)`: **exact** on exact rings
(`1//2` over `Rational{BigInt}`) and on binary floats (division by 2 is
exact), symbolic over `Symbolics.Num`; the ring must support division by
`R(2)` (every shipped ring does).  On Euclidean bases the energy of a
nonzero `F` is strictly positive; on Lorentzian bases it may be ≤ 0
(the indefinite ½(E²−B²)-type invariant — correct physics).

This is a **scalar functional**, not an energy-density field: energy density
and Poynting flux need the cochain-valued cup product (the dedicated
post-L10 phase, DESIGN.md §17.1).
"""
total_energy(b::BaseSpace, F::Field{R,E,B}) where {R,E,B} =
    (one(R) / R(2)) * inner_product(b, F, F)

export inner_product, field_norm2, total_energy
