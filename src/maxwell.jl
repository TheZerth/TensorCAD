# ── Phase L8.2: Maxwell assembly from potential-first field strength ─────────
#
# This file does not add new physics machinery.  It packages the L8/L8.2
# operators into exterior-calculus Maxwell checks on a `GridBase`: F = dA
# (potential primary), dF = 0 by d² = 0, and δF = J.

"""
    electromagnetic_field(A::Field) -> Field

Potential-first field strength `F = dA`.  This honors DESIGN.md §14: the
potential/cochain `A` is the input and the field strength is derived, never the
reverse.
"""
electromagnetic_field(A::Field) = d(A)

"""
    maxwell_bianchi(A::Field) -> Field

The source-free Bianchi identity for a potential-built field:
`dF = d(dA) = 0`, exact on every valid base because the signed boundary satisfies
`∂∘∂ = 0`.
"""
maxwell_bianchi(A::Field) = d(electromagnetic_field(A))

"""
    maxwell_current(b::BaseSpace, A::Field) -> Field

The sourced Maxwell current/cochain produced by a potential `A`:
`J = δF = δ(dA)`.  Requires the L8.2 Hodge gate through [`codifferential`](@ref).
"""
maxwell_current(b::BaseSpace, A::Field) = codifferential(b, electromagnetic_field(A))

"""
    source_equation_residual(b, F, J) -> Field

Residual of the sourced codifferential Maxwell equation `δF = J`.
"""
source_equation_residual(b::BaseSpace, F::Field, J::Field) = codifferential(b, F) - J

"""
    source_free_maxwell(b, F) -> Bool

Whether a field satisfies the source-free exterior-calculus Maxwell equations
`dF = 0` and `δF = 0`.  Requires `can_hodge(b)` through `δ`.
"""
source_free_maxwell(b::BaseSpace, F::Field) = iszero_field(d(F)) && iszero_field(codifferential(b, F))

iszero_field(ω::Field) = length(ω) == 0

"""
    PlaneWaveDemo

Return record for [`plane_wave_demo`](@ref): a potential `A`, derived field
`F = dA`, residuals `dF`, `δF`, `ΔF`, and the discrete standing-mode eigenvalue
`λ` such that `ΔF ≈ λF`.
"""
struct PlaneWaveDemo{A,F,D,C,L,V}
    A :: A
    F :: F
    dF :: D
    deltaF :: C
    laplacianF :: L
    eigenvalue :: V
    potential_first :: Bool
end

"""
    plane_wave_demo(grid::GridBase) -> PlaneWaveDemo

A non-constant oscillatory standing-mode validation on a `GridBase`, assembled
potential-first.  L8.2 does not implement time stepping or rendering; this is a
static discrete operator check.

For a floating-point unit grid, choose the Neumann path eigenmode

```julia
A(i,j) = cos(π*(i+1/2)/(nx+1))
```

on vertices and derive `F = dA`.  The field varies cell-to-cell, satisfies
`dF = d²A = 0` exactly, and satisfies the non-vacuous discrete eigenrelation
`ΔF = λF` with

```julia
λ = 2 - 2cos(π/(nx+1)).
```

The eigenvalue is **positive**: `Δ = dδ + δd` with the pairing-adjoint `δ`
(L8.2.1) is positive-semidefinite on Euclidean bases.  The historical value
`λ = -2 + 2cos(π/(nx+1))` was the analyst's-sign Laplacian produced by the
pre-L8.2.1 codifferential sign (`δ = -dᵀ` on Euclidean); the sign flip here is
the re-baseline mandated by DESIGN.md §15.4, not a change to the mode.

The codifferential `δF` is generally a nonzero source for this standing mode;
source-free `δF=0` remains checked separately by [`source_free_maxwell`](@ref)
when appropriate.
"""
function plane_wave_demo(grid::GridBase)
    _require_hodge(grid, "plane_wave_demo")
    m = grid.metric
    R = eltype(m.g)
    R <: AbstractFloat || throw(ArgumentError(
        "plane_wave_demo uses an oscillatory cosine mode and requires an AbstractFloat grid metric"))
    N = grid.nx + 1
    vertex_coord(v) = begin
        z = Int(v) - 1
        width = grid.nx + 1
        (z % width, z ÷ width)
    end
    A = Field(grid, 0, Dict{Int,CliffordTensor{R}}(
        c => clifford_scalar(m, R(cos(pi * (vertex_coord(c)[1] + 0.5) / N)))
        for c in cells(grid, 0)))
    F = electromagnetic_field(A)
    λ = R(2 - 2cos(pi / N))
    PlaneWaveDemo(A, F, d(F), codifferential(grid, F), hodge_laplacian(grid, F), λ, true)
end

export electromagnetic_field, maxwell_bianchi, maxwell_current,
       source_equation_residual, source_free_maxwell,
       PlaneWaveDemo, plane_wave_demo, iszero_field
