# ── Phase L8 (Tier 1): exterior derivative — topological coboundary ──────────
#
# MATHEMATICAL CONTRACT
#
# A grade-k Field is a discrete k-cochain: it assigns a fibre element to each
# k-cell of a BaseSpace.  The exterior derivative
#
#     d : Ωᵏ → Ωᵏ⁺¹
#
# implemented here is the COBoundary operator — the transpose of the signed
# boundary/incidence matrix supplied by the base.  For a (k+1)-cell c,
#
#     (dω)(c) = Σ_{(face, sign) ∈ boundary(b, k+1, c)} sign · ω(face).
#
# This is the discrete Stokes relation: evaluating the cochain dω on a cell is
# exactly evaluating ω on that cell's oriented boundary.  Since every valid base
# is required to satisfy ∂∘∂ = 0 with signs, d∘d = 0 follows as an exact operator
# identity — no tolerances, no metric, and no transport.
#
# TIER BOUNDARY (DESIGN.md §15)
#
# This file is deliberately metric-free and connection-free.  It uses ONLY
# boundary, cells, and fibre descriptors.  It does not consult metric, signature,
# has_metric, can_hodge, or transport.  The exterior derivative d and the
# covariant/geometric derivative ∇ are distinct discrete operators: d consumes
# topology (incidence), while ∇ consumes transport (a connection).  Their grade
# shadows converge in the continuum story, but the APIs remain separate here so
# topological bugs cannot be hidden behind geometric structure.

"""
    d(ω::Field) -> Field

The metric-free exterior derivative of a discrete field/cochain.

For a grade-`k` field `ω` over a base `b`, `d(ω)` is the grade-`k+1` field
whose value on each `(k+1)`-cell `c` is the signed boundary sum

```julia
(dω)(c) = sum(sign * ω[face] for (face, sign) in boundary(b, k+1, c))
```

Equivalently, `d` is the coboundary operator: the transpose of the base's signed
boundary/incidence operator.  This is the discrete-Stokes definition — evaluating
`dω` on a cell is evaluating `ω` on that cell's oriented boundary.  Because the
base contract requires `boundary ∘ boundary = 0` with signs, `d(d(ω))` is exactly
zero as an operator identity.

This Tier-1 operator is purely topological and exact.  It uses only
[`boundary`](@ref), [`cells`](@ref), and the [`FibreDescriptor`](@ref) machinery;
it never calls `metric`, `signature`, `transport`, `has_metric`, or `can_hodge`,
so it works on every [`BaseSpace`](@ref), including a bare [`GraphBase`](@ref).

`d` is deliberately separate from the covariant/geometric derivative `∇`: `d`
uses incidence, while `∇` (L8.1) uses transport/a connection.  Their continuum
shadows are related, but unifying them in the discrete API would hide the Tier
boundary from DESIGN.md §15.

If `k+1 > top_grade(b)`, there are no higher cells; the result is the sparse
zero field of grade `k+1`.
"""
function d(ω::Field{R,E,B}) where {R,E,B<:BaseSpace}
    b = ω.base
    kp = field_grade(ω) + 1

    # Honest top-grade behaviour: no cells above the topological dimension.
    # Preserve the element type of the input fibre so the zero field remains
    # concrete and type-stable even when no (k+1)-cell exists from which to infer
    # a descriptor.
    if kp > top_grade(b)
        return Field{R,E,B}(b, kp, Dict{Int,E}())
    end

    cs = cells(b, kp)
    vals = Dict{Int,E}()
    for c in cs
        fd = fibre(b, kp, c)
        fibre_eltype(fd) == E || throw(ArgumentError(
            "d cannot assemble a $(kp)-field with fibre element type " *
            "$(fibre_eltype(fd)) from a $(field_grade(ω))-field with element " *
            "type $E; heterogeneous grade fibres need an explicit transfer map"))
        acc = zero_fibre(fd)::E
        for (face, sign) in boundary(b, kp, c)
            acc = acc + sign * evaluate(ω, face)
        end
        iszero(acc) || (vals[Int(c)] = acc)
    end
    Field{R,E,B}(b, kp, vals)
end

"""
    grad(φ::Field) -> Field

Grade-specialized exterior derivative on 0-fields: `grad(φ) = d(φ)`.

This is the graph/grid cochain gradient, e.g. on an oriented graph edge
`tail → head`, `(grad φ)(edge) = φ(head) - φ(tail)`.  It is a thin documented
alias of [`d`](@ref), not a separate implementation, and is therefore exact,
metric-free, transport-free, and available on bare graphs.

`grad` is not the covariant/geometric derivative `∇`; that Tier-2 operator needs
transport/a connection and remains distinct in the API (DESIGN.md §15).
"""
function grad(φ::Field)
    field_grade(φ) == 0 || throw(ArgumentError(
        "grad is the grade-0 shadow of d; expected a 0-field, got grade $(field_grade(φ))"))
    d(φ)
end

"""
    curl(A::Field) -> Field

Grade-specialized exterior derivative on 1-fields: `curl(A) = d(A)`.

This is the metric-free 1-cochain → 2-cochain shadow of the discrete exterior
derivative.  It is a thin alias of [`d`](@ref), not a separate implementation; it
uses only signed incidence and therefore does not call metric, Hodge star,
transport, or any covariant derivative machinery.

No metric-free `div` is provided in Tier 1: divergence proper is the
codifferential/Hodge-adjoint story (`δ`) and belongs to L8.2, gated by metric and
dual-complex capability.
"""
function curl(A::Field)
    field_grade(A) == 1 || throw(ArgumentError(
        "curl is the grade-1 shadow of d; expected a 1-field, got grade $(field_grade(A))"))
    d(A)
end

export d, grad, curl
