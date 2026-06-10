# ── Phase L8.2: Maxwell assembly from potential-first field strength ─────────
#
# This file does not add new physics machinery.  It packages the L8/L8.2
# operators into the standard exterior-calculus Maxwell checks on a `GridBase`:
# F = dA (potential primary), dF = 0 by d² = 0, and δF = J.

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
`F = dA`, and the three source-free residuals `dF`, `δF`, and `ΔF`.
"""
struct PlaneWaveDemo{A,F,D,C,L}
    A :: A
    F :: F
    dF :: D
    deltaF :: C
    laplacianF :: L
    potential_first :: Bool
end

"""
    plane_wave_demo(grid::GridBase) -> PlaneWaveDemo

A minimal source-free Maxwell validation on a `GridBase`, assembled
potential-first.  L8.2 does not implement time stepping or a rendered wave; those
belong to L10/L11.  This routine uses a nonzero affine pure-mode potential
`A(v) = (x(v)+y(v))e₁` on vertices.  Its derived field `F = dA` is the constant
1-cochain `e₁` on every grid edge: a discrete harmonic/source-free mode satisfying
`dF = 0`, `δF = 0`, and `ΔF = 0` exactly on the orthogonal grid.
"""
function plane_wave_demo(grid::GridBase)
    _require_hodge(grid, "plane_wave_demo")
    m = grid.metric
    R = eltype(m.g)
    e1 = clifford_basis_vector(m, 1)
    vertex_coord(v) = begin
        z = Int(v) - 1
        width = grid.nx + 1
        (z % width, z ÷ width)
    end
    A = Field(grid, 0, Dict{Int,CliffordTensor{R}}(
        c => R(sum(vertex_coord(c))) * e1 for c in cells(grid, 0)))
    F = electromagnetic_field(A)
    PlaneWaveDemo(A, F, d(F), codifferential(grid, F), hodge_laplacian(grid, F), true)
end

export electromagnetic_field, maxwell_bianchi, maxwell_current,
       source_equation_residual, source_free_maxwell,
       PlaneWaveDemo, plane_wave_demo, iszero_field
