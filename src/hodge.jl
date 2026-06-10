# ── Phase L8.2 (Tier 3): Hodge star, codifferential, Laplacian ─────────────
#
# MATHEMATICAL CONTRACT
#
# This tier is gated by `can_hodge(b) == has_metric(b) && has_dual_complex(b)`.
# A bare graph must refuse these operators.  `GridBase` supplies the validation
# case: an orthogonal structured primal complex with an exact, combinatorial dual
# correspondence.  DESIGN.md §16.1 splits the dual complex into three pieces:
#
#   1. correspondence / enumeration — needed here by `⋆`;
#   2. diagonal volume weights       — unit on the shipped orthogonal grid;
#   3. geometry / positions          — deferred to L11 visualization.
#
# The Hodge star maps primal Fields to HodgeDualFields and maps HodgeDualFields
# back to primal Fields.  `δ = ±⋆d⋆` crosses to the dual complex and returns a
# primal Field; `Δ = dδ + δd` remains primal-valued.

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

"""
    HodgeDualField{R,E,B}

A sparse section over the **dual** cell index set of a base.  This is not a
primal [`Field`](@ref): its keys are dual cell ids, with counts supplied by
[`dual_n_cells`](@ref), and its grade is a dual grade.  It exists so the Hodge
star is geometrically honest: a primal k-cochain maps to a dual `(n-k)`-cochain,
not to a relabeled primal cell.

The container mirrors the ordinary field interface where appropriate:
[`evaluate`](@ref) / indexing, `field_grade`, iteration over stored nonzero
entries, and exact-ring equality.  Entries are zero-pruned; values remain fibre
elements `E <: AbstractTensorElement{R}`.
"""
struct HodgeDualField{R,E<:AbstractTensorElement{R},B<:BaseSpace}
    base   :: B
    grade  :: Int
    values :: Dict{Int,E}

    function HodgeDualField{R,E,B}(base::B, grade::Int, values::Dict{Int,E}
                                  ) where {R,E<:AbstractTensorElement{R},B<:BaseSpace}
        grade >= 0 || throw(ArgumentError("dual field grade must be nonnegative, got $grade"))
        if grade > top_grade(base) && !isempty(values)
            throw(ArgumentError(
                "dual field grade $grade is above top_grade $(top_grade(base)) for $(typeof(base)); " *
                "only the empty zero dual field is valid above the topological dimension"))
        end
        maxid = dual_n_cells(base, grade)
        for (cid, x) in values
            1 <= cid <= maxid || throw(ArgumentError(
                "dual cell id $cid is not a grade-$grade dual cell of the base (valid range 1:$maxid)"))
            # The dual grade k corresponds to primal grade n-k; use that attached
            # fibre to validate value compatibility.  GridBase is uniform today,
            # but this keeps the contract explicit.
            primal_grade = top_grade(base) - grade
            fibre_matches(fibre(base, primal_grade, cid), x) || throw(ArgumentError(
                "the value assigned at dual cell $cid does not belong to the mirrored primal fibre"))
        end
        pruned = Dict{Int,E}(cid => x for (cid, x) in values if !iszero(x))
        new{R,E,B}(base, grade, pruned)
    end
end

function HodgeDualField(base::B, grade::Integer, values::Dict{Int,E}
                        ) where {B<:BaseSpace,R,E<:AbstractTensorElement{R}}
    HodgeDualField{R,E,B}(base, Int(grade), values)
end

function evaluate(η::HodgeDualField{R,E,B}, cell::Integer) where {R,E,B}
    c = Int(cell)
    haskey(η.values, c) && return η.values[c]
    primal_grade = top_grade(η.base) - η.grade
    zero_fibre(fibre(η.base, primal_grade, c))::E
end

Base.getindex(η::HodgeDualField, cell::Integer) = evaluate(η, cell)
field_grade(η::HodgeDualField) = η.grade
Base.length(η::HodgeDualField) = length(η.values)
Base.keys(η::HodgeDualField) = keys(η.values)
Base.values(η::HodgeDualField) = Base.values(η.values)
Base.pairs(η::HodgeDualField) = pairs(η.values)
Base.haskey(η::HodgeDualField, c::Integer) = haskey(η.values, Int(c))
Base.iterate(η::HodgeDualField, st...) = iterate(η.values, st...)
Base.eltype(::Type{HodgeDualField{R,E,B}}) where {R,E,B} = Pair{Int,E}

function Base.:(==)(a::HodgeDualField{R,E,B}, b::HodgeDualField{R,E,B}) where {R,E,B}
    a.base === b.base && a.grade == b.grade || return false
    for c in union(keys(a.values), keys(b.values))
        evaluate(a, c) == evaluate(b, c) || return false
    end
    return true
end

function Base.:-(η::HodgeDualField{R,E,B}) where {R,E,B}
    HodgeDualField{R,E,B}(η.base, η.grade, Dict{Int,E}(c => -x for (c, x) in η.values))
end

function Base.:*(c::R, η::HodgeDualField{R,E,B}) where {R,E,B}
    vals = Dict{Int,E}()
    for (cell, x) in η.values
        y = c * x
        iszero(y) || (vals[cell] = y)
    end
    HodgeDualField{R,E,B}(η.base, η.grade, vals)
end
Base.:*(η::HodgeDualField{R,E,B}, c::R) where {R,E,B} = c * η
Base.:*(n::Integer, η::HodgeDualField{R,E,B}) where {R,E,B} = R(n) * η
Base.:*(η::HodgeDualField{R,E,B}, n::Integer) where {R,E,B} = R(n) * η

# ── Hodge star ────────────────────────────────────────────────────────────────

"""
    hodge_star(b::BaseSpace, ω::Field) -> HodgeDualField
    hodge_star(b::BaseSpace, η::HodgeDualField) -> Field
    ⋆(b, ω)

Tier-3 Hodge star.  Requires `can_hodge(b) == true`; otherwise an
`ArgumentError` names the missing metric / dual-complex capability.

For a primal grade-`k` [`Field`](@ref) over `n = top_grade(b)`, `hodge_star`
returns a [`HodgeDualField`](@ref) of dual grade `n-k`.  Each primal k-cell `c`
is mapped to the corresponding dual cell `dual_cell(b,k,c)`; no primal id reuse
and no silent dropping occur.  Applying `hodge_star` to a `HodgeDualField` maps
back to an ordinary primal `Field`, so `⋆⋆` is well-defined.

On the shipped orthogonal `GridBase`, the diagonal DEC Hodge volume weights are
unit.  The dual **correspondence/enumeration** is operator structure needed here;
dual **volumes** beyond the unit-grid case and dual **geometry/positions** remain
out of scope (L11 for positions).

The fibre value is complemented using the Clifford pseudoscalar (`dual`) with the
signature-correct sign convention.  The star-square law is

```julia
⋆⋆ω = (-1)^(k*(n-k) + q) * ω
```

for non-degenerate signature with `q` negative directions (Euclidean q=0,
Lorentzian q=1).
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
        dc = dual_cell(b, k, c)
        y = _hodge_value(evaluate(ω, c), k, n)::E
        iszero(y) || (vals[dc] = y)
    end
    HodgeDualField{R,E,B}(b, target_grade, vals)
end

function hodge_star(b::B, η::HodgeDualField{R,E,B}) where {R,E<:CliffordTensor{R},B<:BaseSpace}
    b === η.base || throw(ArgumentError("hodge_star requires the supplied base to be the dual field's own base"))
    _require_hodge(b, "hodge_star")
    n = top_grade(b)
    l = field_grade(η)
    (0 <= l <= n) || throw(ArgumentError(
        "hodge_star expects a dual field grade in 0:top_grade(b) = 0:$n, got $l"))
    primal_grade = n - l
    vals = Dict{Int,E}()
    for pc in cells(b, primal_grade)
        dc = dual_cell(b, primal_grade, pc)
        y = _hodge_value(evaluate(η, dc), l, n)::E
        iszero(y) || (vals[Int(pc)] = y)
    end
    Field{R,E,B}(b, primal_grade, vals)
end

const ⋆ = hodge_star

# ── Exterior derivative on dual cochains ──────────────────────────────────────

"""
    d(η::HodgeDualField) -> HodgeDualField

Exterior derivative on dual cochains.  The dual incidence is derived from the
primal signed boundary and the `dual_cell` correspondence, not from independent
dual geometry.  If a dual grade-`l` cell corresponds to a primal `(n-l)` cell,
then a target dual `(l+1)` cell corresponds to a primal `(n-l-1)` cell; its
coboundary sum scans the primal cofaces whose boundary contains that cell.  The
sign is exactly the primal incidence sign: if `(face, s)` occurs in
`boundary(b, primal_source_grade, coface)`, then the dual source cell contributes
`s` to the dual target cell.  Boundary primal cells therefore produce one-sided
dual boundary terms rather than being dropped.
"""
function d(η::HodgeDualField{R,E,B}) where {R,E<:CliffordTensor{R},B<:BaseSpace}
    b = η.base
    l = field_grade(η)
    lp = l + 1
    n = top_grade(b)
    if lp > n
        return HodgeDualField{R,E,B}(b, lp, Dict{Int,E}())
    end
    primal_target_grade = n - lp
    primal_source_grade = n - l
    vals = Dict{Int,E}()
    for pc in cells(b, primal_target_grade)
        acc = zero_fibre(fibre(b, primal_target_grade, pc))::E
        for coface in cells(b, primal_source_grade)
            for (face, sign) in boundary(b, primal_source_grade, coface)
                if face == pc
                    acc = acc + sign * evaluate(η, dual_cell(b, primal_source_grade, coface))
                end
            end
        end
        iszero(acc) || (vals[dual_cell(b, primal_target_grade, pc)] = acc)
    end
    HodgeDualField{R,E,B}(b, lp, vals)
end

# ── Codifferential and Hodge Laplacian ────────────────────────────────────────

"""
    codifferential(b::BaseSpace, ω::Field) -> Field
    δ(b, ω) -> Field

Hodge adjoint of the exterior derivative: for a primal grade-`k` field over an
`n = top_grade(b)` base,

```julia
δω = (-1)^(n*(k+1)+1) ⋆ d ⋆ ω
```

where the first `⋆` maps to a [`HodgeDualField`](@ref), `d` uses the dual
incidence derived from primal boundary + dual correspondence, and the second `⋆`
returns to a primal [`Field`](@ref).  This maps `k`-fields to `(k-1)`-fields for
`k > 0`; on 0-fields it returns the zero 0-field.  This is the genuine
metric/dual-complex divergence operator that Tier 1 explicitly did not provide.
Requires `can_hodge(b) == true`.
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

export HodgeDualField, hodge_star, ⋆, codifferential, δ, hodge_laplacian, Δ
