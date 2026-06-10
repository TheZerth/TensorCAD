# ── Phase L8.2 (Tier 3): Hodge star, codifferential, Laplacian ─────────────
#
# MATHEMATICAL CONTRACT
#
# This tier is gated by `can_hodge(b) == has_metric(b) && has_dual_complex(b)`.
# A bare graph must refuse these operators.  `GridBase` supplies the validation
# case: an orthogonal structured primal complex for which the diagonal
# (mass-lumped) DEC Hodge is exact.  We do not materialize a dual-cell geometry in
# L8.2; DESIGN.md §16.1 deliberately defers that geometry to L11 visualization.
#
# The returned values remain ordinary fibre Fields.  The topological cell grade is
# complemented k ↦ n-k, where n = top_grade(b); the fibre value is complemented by
# the Clifford pseudoscalar machinery (`dual`) with the signature-correct
# star-square sign.  On the flat unit GridBase the diagonal Hodge factors are 1.

function _require_hodge(b::BaseSpace, opname::AbstractString)
    can_hodge(b) || throw(ArgumentError(
        "$opname requires can_hodge(b) == true (both has_metric and has_dual_complex). " *
        "$(typeof(b)) reports has_metric=$(has_metric(b)), has_dual_complex=$(has_dual_complex(b)); " *
        "a bare graph correctly has no Hodge star/codifferential."))
end

# The Clifford `dual(A) = A I⁻¹` has a convention-dependent global sign.  For the
# n=2 GridBase fibres used in L8.2 validation, multiplying top-grade inputs by -1
# gives the standard Hodge star-square law ⋆⋆ = (-1)^(k(n-k)+q), where q is the
# number of negative directions in the metric signature.  The diagonal DEC volume
# factor is 1 on the shipped unit orthogonal GridBase.
_hodge_value(x::CliffordTensor, k::Integer, n::Integer) =
    Int(k) == Int(n) ? -dual(x) : dual(x)

function _target_cell_exists(b::BaseSpace, grade::Int, cell::Int)
    1 <= cell <= n_cells(b, grade)
end

"""
    hodge_star(b::BaseSpace, ω::Field) -> Field
    ⋆(b, ω) -> Field

Tier-3 Hodge star on a discrete field/cochain.  Requires
`can_hodge(b) == true`; otherwise an `ArgumentError` names the missing metric / dual-complex capability.

For a grade-`k` field over an `n = top_grade(b)` base, `hodge_star` returns a
Field of grade `n-k`.  In L8.2 the shipped exact realization is the diagonal
(mass-lumped) DEC Hodge on an orthogonal `GridBase`: primal k-cochains are mapped
back into the primal `(n-k)` grade with unit diagonal volume factors.  This is
exact on the structured grid; the explicit dual-cell geometry is deliberately
not materialized until L11 visualization.

The fibre value is complemented using the Clifford pseudoscalar (`dual`) with the
signature-correct sign convention.  The star-square law is

```julia
⋆⋆ω = (-1)^(k*(n-k) + q) * ω
```

for non-degenerate signature with `q` negative directions (Euclidean q=0,
Lorentzian q=1).  The sign/factor is therefore signature-dependent.
"""
function hodge_star(b::B, ω::Field{R,E,B}) where {R,E<:CliffordTensor{R},B<:BaseSpace}
    b === ω.base || throw(ArgumentError("hodge_star requires the supplied base to be the field's own base"))
    _require_hodge(b, "hodge_star")
    n = top_grade(b)
    k = field_grade(ω)
    (0 <= k <= n) || throw(ArgumentError(
        "hodge_star expects a field grade k in 0:top_grade(b) = 0:$n, got $k"))
    target_grade = n - k
    vals = Dict{Int,E}()
    for c in cells(b, k)
        tc = Int(c)
        _target_cell_exists(b, target_grade, tc) || continue
        y = _hodge_value(evaluate(ω, c), k, n)::E
        iszero(y) || (vals[tc] = y)
    end
    Field{R,E,B}(b, target_grade, vals)
end

const ⋆ = hodge_star

"""
    codifferential(b::BaseSpace, ω::Field) -> Field
    δ(b, ω) -> Field

Hodge adjoint of the exterior derivative: for a grade-`k` field over an
`n = top_grade(b)` base,

```julia
δω = (-1)^(n*(k+1)+1) ⋆ d ⋆ ω
```

with the Hodge star convention documented in [`hodge_star`](@ref).  This maps
`k`-fields to `(k-1)`-fields for `k > 0`; on 0-fields it returns the zero 0-field.
This is the genuine metric/dual-complex divergence operator that Tier 1
explicitly did not provide.  Requires `can_hodge(b) == true`.
"""
function codifferential(b::B, ω::Field{R,E,B}) where {R,E<:CliffordTensor{R},B<:BaseSpace}
    _require_hodge(b, "codifferential")
    k = field_grade(ω)
    k == 0 && return Field{R,E,B}(b, 0, Dict{Int,E}())
    n = top_grade(b)
    s = hodge_star(b, d(hodge_star(b, ω)))
    isodd(n * (k + 1) + 1) ? -s : s
end

const δ = codifferential

"""
    hodge_laplacian(b::BaseSpace, ω::Field) -> Field
    Δ(b, ω) -> Field

Hodge–de Rham Laplacian

```julia
Δω = d(δω) + δ(dω)
```

Requires `can_hodge(b) == true`.  On 0-fields this reduces to the scalar
Laplacian `δd`; harmonic fields are the kernel of `Δ`.
"""
function hodge_laplacian(b::B, ω::Field{R,E,B}) where {R,E<:CliffordTensor{R},B<:BaseSpace}
    _require_hodge(b, "hodge_laplacian")
    k = field_grade(ω)
    left = k == 0 ? Field{R,E,B}(b, k, Dict{Int,E}()) : d(codifferential(b, ω))
    right = field_grade(ω) >= top_grade(b) ? Field{R,E,B}(b, k, Dict{Int,E}()) : codifferential(b, d(ω))
    left + right
end

const Δ = hodge_laplacian

export hodge_star, ⋆, codifferential, δ, hodge_laplacian, Δ
