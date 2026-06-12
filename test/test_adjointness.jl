# ─────────────────────────────────────────────────────────────────────────────
# Phase L8.2.1: δ is DEFINED as the pairing-adjoint of d (DESIGN.md §15.4)
#
# These are the permanent regression tests for the codifferential sign finding:
# the ⋆⋆ law provably underdetermines the star's paired-grade sign branch, so
# adjointness under the diagonal Hodge pairing is the constraint that pins δ.
# The historical σ = (-1)^(n(k+1)+1) made δ = -dᵀ on Euclidean signature; the
# corrected σ = (-1)^(n(k+1)+q) makes δ the true adjoint on every signature.
#
#   Gate 1 — ⟨dα,β⟩ == ⟨α,δβ⟩ EXACTLY over Rational{BigInt}, both signatures,
#            both grids, every grade transition, boundary-touching included
#            (the transpose-based dual-d retains one-sided boundary terms, so
#            the identity is exact everywhere — the natural-BC fact).
#   Gate 2 — the adjointness corollary ⟨α,Δα⟩ == ⟨dα,dα⟩ + ⟨δα,δα⟩ exactly,
#            and Euclidean positive-semidefiniteness of Δ.
#   Gate 3 — δ∘δ = 0 still exact.
#
# The pairing helper below is deliberately LOCAL and minimal (the diagonal
# Hodge pairing with unit grid weights): this operator-level test must not
# depend on the L9.1 `inner_product` layer it exists to certify independently.

R = Rational{BigInt}

# Diagonal Hodge pairing with unit weights: Σ_c ⟨α(c)·reversion(β(c))⟩₀.
function _adj_pairing(b, α::Field, β::Field)
    @assert field_grade(α) == field_grade(β)
    s = zero(R)
    for c in cells(b, field_grade(α))
        s += scalar_product(evaluate(α, c), reversion(evaluate(β, c)))
    end
    s
end

# A generic mixed-blade field with cell-dependent coefficients.  It populates
# EVERY k-cell, so boundary cells are always touched.
function _adj_field(b, m, g)
    e1 = clifford_basis_vector(m, 1)
    e2 = clifford_basis_vector(m, 2)
    e12 = clifford_basis_element(m, [1, 2])
    o = clifford_one(m)
    Field(b, g, Dict(Int(c) =>
        R(c) * o + R(2c + 1) * e1 + R(c^2 - 3) * e2 + R(7 - c) * e12
        for c in cells(b, g)))
end

# The signed-incidence transpose of d (unit dual volumes) — the object δ must
# equal by the definitional contract.
function _adj_dT(b, β::Field)
    g = field_grade(β)
    E = typeof(evaluate(β, first(cells(b, g))))
    vals = Dict{Int,E}()
    for f in cells(b, g - 1)
        acc = zero_fibre(fibre(b, g - 1, f))
        for c in cells(b, g)
            for (face, s) in boundary(b, g, c)
                face == f && (acc = acc + s * evaluate(β, c))
            end
        end
        iszero(acc) || (vals[Int(f)] = acc)
    end
    Field(b, g - 1, vals)
end

_adj_metrics() = (
    ("Euclidean", signature_metric(VectorSpace(2), R, 2, 0, 0)),
    ("Lorentzian", signature_metric(VectorSpace(2), R, 1, 1, 0)),
)

@testset "L8.2.1: δ equals the signed-incidence transpose of d on every signature" begin
    for (name, m) in _adj_metrics(), (nx, ny) in ((2, 2), (3, 2))
        grid = GridBase(nx, ny; metric = m)
        for g in 1:2
            β = _adj_field(grid, m, g)
            @test codifferential(grid, β) == _adj_dT(grid, β)
        end
    end
end

@testset "L8.2.1 gate 1: exact adjointness ⟨dα,β⟩ == ⟨α,δβ⟩, boundary-touching included" begin
    for (name, m) in _adj_metrics(), (nx, ny) in ((2, 2), (3, 2))
        grid = GridBase(nx, ny; metric = m)
        for k in 0:1
            # Fully-populated fields: every boundary cell carries a value.
            α = _adj_field(grid, m, k)
            β = _adj_field(grid, m, k + 1)
            @test _adj_pairing(grid, d(α), β) == _adj_pairing(grid, α, codifferential(grid, β))

            # Deliberately boundary-ONLY supports: a corner vertex / the first
            # (boundary) edge or face.  The identity must hold exactly there
            # too — the transpose-based dual-d retains the one-sided boundary
            # contributions (the natural boundary condition; DESIGN.md §15.4).
            αb = Field(grid, k, Dict(1 => evaluate(α, 1)))
            βb = Field(grid, k + 1, Dict(1 => evaluate(β, 1)))
            @test _adj_pairing(grid, d(αb), βb) == _adj_pairing(grid, αb, codifferential(grid, βb))
            @test _adj_pairing(grid, d(αb), β) == _adj_pairing(grid, αb, codifferential(grid, β))
            @test _adj_pairing(grid, d(α), βb) == _adj_pairing(grid, α, codifferential(grid, βb))
        end
    end
end

@testset "L8.2.1 gate 2: ⟨α,Δα⟩ == ⟨dα,dα⟩ + ⟨δα,δα⟩; Δ ⪰ 0 on Euclidean" begin
    # The corollary identity follows from adjointness alone, so it holds
    # exactly on every signature; positivity is Euclidean-only physics.
    for (name, m) in _adj_metrics(), (nx, ny) in ((2, 2), (3, 2))
        grid = GridBase(nx, ny; metric = m)
        for k in 0:2
            α = _adj_field(grid, m, k)
            lhs = _adj_pairing(grid, α, hodge_laplacian(grid, α))
            dα = d(α)
            δα = codifferential(grid, α)
            rhs = (k == 2 ? zero(R) : _adj_pairing(grid, dα, dα)) +
                  (k == 0 ? zero(R) : _adj_pairing(grid, δα, δα))
            @test lhs == rhs
            if name == "Euclidean"
                @test lhs >= 0
            end
        end
        if name == "Euclidean"
            # Strict positivity for a manifestly non-harmonic field: a single
            # bump 0-field has dα ≠ 0, so ⟨α,Δα⟩ = ⟨dα,dα⟩ > 0.
            bump = Field(grid, 0, Dict(1 => clifford_one(m)))
            @test _adj_pairing(grid, bump, hodge_laplacian(grid, bump)) > 0
        end
    end
end

@testset "L8.2.1 gate 3: δ∘δ = 0 remains exact" begin
    for (name, m) in _adj_metrics(), (nx, ny) in ((2, 2), (3, 2))
        grid = GridBase(nx, ny; metric = m)
        β = _adj_field(grid, m, 2)
        @test length(codifferential(grid, codifferential(grid, β))) == 0
    end
end
