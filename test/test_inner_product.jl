# ─────────────────────────────────────────────────────────────────────────────
# Phase L9.1: the Hodge (L²) inner product of cochains and the energy
# functional.
#
# Adjointness ⟨dα,β⟩ == ⟨α,δβ⟩ — the verification centerpiece — is certified
# in test_adjointness.jl (the L8.2.1 permanent gate, with its own local copy
# of the pairing so the operator test is independent of this layer); here it
# is only LINKED, not duplicated: one instance per signature ties the shipped
# `inner_product` to the same identity.
# ─────────────────────────────────────────────────────────────────────────────

R = Rational{BigInt}

# A generic mixed-blade field with cell-dependent coefficients (populates
# every k-cell, so boundary cells are always exercised).
function _ipt_field(b, m, g; seed::Int = 0)
    e1 = clifford_basis_vector(m, 1)
    e2 = clifford_basis_vector(m, 2)
    e12 = clifford_basis_element(m, [1, 2])
    o = clifford_one(m)
    Field(b, g, Dict(Int(c) =>
        R(c + seed) * o + R(2c - seed) * e1 + R(c^2 + seed) * e2 + R(5 - c - seed) * e12
        for c in cells(b, g)))
end

_ipt_metrics() = (
    ("Euclidean", signature_metric(VectorSpace(2), R, 2, 0, 0)),
    ("Lorentzian", signature_metric(VectorSpace(2), R, 1, 1, 0)),
)

@testset "inner_product: bilinearity and symmetry, exact, every grade, both grids" begin
    for (name, m) in _ipt_metrics(), (nx, ny) in ((2, 2), (3, 2))
        grid = GridBase(nx, ny; metric = m)
        a, bcoef = R(3 // 2), R(-7 // 3)
        for g in 0:2
            α = _ipt_field(grid, m, g; seed = 1)
            β = _ipt_field(grid, m, g; seed = 2)
            γ = _ipt_field(grid, m, g; seed = 5)

            # bilinearity, both slots
            @test inner_product(grid, a * α + bcoef * γ, β) ==
                  a * inner_product(grid, α, β) + bcoef * inner_product(grid, γ, β)
            @test inner_product(grid, α, a * β + bcoef * γ) ==
                  a * inner_product(grid, α, β) + bcoef * inner_product(grid, α, γ)

            # symmetry (no signature caveat: reversion is an anti-automorphism
            # and the grade-0 part is cyclic)
            @test inner_product(grid, α, β) == inner_product(grid, β, α)
        end
    end
end

@testset "inner_product: Euclidean definiteness, Lorentzian indefiniteness" begin
    m = signature_metric(VectorSpace(2), R, 2, 0, 0)
    grid = GridBase(2, 2; metric = m)
    for g in 0:2
        # several nonzero fields: single-cell scalar, single-cell blade mix,
        # fully populated generic
        c0 = first(cells(grid, g))
        singles = [
            Field(grid, g, Dict(Int(c0) => clifford_one(m))),
            Field(grid, g, Dict(Int(c0) =>
                clifford_basis_vector(m, 1) - R(3) * clifford_basis_element(m, [1, 2]))),
            _ipt_field(grid, m, g),
        ]
        for α in singles
            @test field_norm2(grid, α) > 0
        end
        # zero iff zero field
        E = typeof(clifford_one(m))
        @test field_norm2(grid, Field(grid, g, Dict{Int,E}())) == 0
    end

    # Lorentzian: a nonzero field with ⟨α,α⟩ < 0.  The fibre blade e₂ spans the
    # negative metric direction: ⟨e₂·reversion(e₂)⟩₀ = e₂² = -1, so a purely
    # e₂-valued cochain has negative squared norm — the indefinite fibre form
    # is physics (the ½(E²−B²)-type invariant), not a bug.
    lm = signature_metric(VectorSpace(2), R, 1, 1, 0)
    lgrid = GridBase(2, 2; metric = lm)
    αL = Field(lgrid, 1, Dict(1 => clifford_basis_vector(lm, 2)))
    @test field_norm2(lgrid, αL) < 0
    # and a null one: e₁ + e₂ squares to 1 - 1 = 0
    αN = Field(lgrid, 1, Dict(1 => clifford_basis_vector(lm, 1) + clifford_basis_vector(lm, 2)))
    @test length(αN) == 1                    # nonzero field…
    @test field_norm2(lgrid, αN) == 0        # …with vanishing norm²
end

@testset "inner_product: adjointness linkage to the L8.2.1 gate" begin
    # Full battery in test_adjointness.jl; this ties the SHIPPED pairing to it.
    for (name, m) in _ipt_metrics()
        grid = GridBase(3, 2; metric = m)
        for k in 0:1
            α = _ipt_field(grid, m, k; seed = 3)
            β = _ipt_field(grid, m, k + 1; seed = 4)
            @test inner_product(grid, d(α), β) ==
                  inner_product(grid, α, codifferential(grid, β))
        end
    end
end

@testset "total_energy: re-baselined Maxwell configurations" begin
    # Known-source exact configuration (test_maxwell.jl): A = e₁ on edge 1 of
    # GridBase(1,1), F = dA = e₁ on the single face.  ⟨F,F⟩ = ⟨e₁ẽ₁⟩₀ = 1, so
    # the energy is exactly 1//2.
    grid = GridBase(1, 1)
    m = grid.metric
    A = Field(grid, 1, Dict(1 => clifford_basis_vector(m, 1)))
    F = electromagnetic_field(A)
    @test total_energy(grid, F) == 1 // 2
    @test total_energy(grid, F) > 0
    @test total_energy(grid, F) == (one(R) / R(2)) * inner_product(grid, F, F)

    # Oscillatory standing mode (Float64): finite, positive, exactly ½⟨F,F⟩.
    fgrid = GridBase(3, 3; metric = signature_metric(VectorSpace(2), Float64, 2, 0, 0))
    demo = plane_wave_demo(fgrid)
    Ef = total_energy(fgrid, demo.F)
    @test isfinite(Ef)
    @test Ef > 0
    @test Ef == 0.5 * inner_product(fgrid, demo.F, demo.F)
end

@testset "inner_product: gating and mismatch errors" begin
    m = signature_metric(VectorSpace(2), R, 2, 0, 0)
    graph = GraphBase(2, [(1, 2)]; metric = m)
    gf = Field(graph, 0, Dict(1 => clifford_one(m)))
    err = try; inner_product(graph, gf, gf); nothing; catch e; e; end
    @test err isa ArgumentError && occursin("can_hodge", err.msg)

    grid = GridBase(2, 2; metric = m)
    α0 = _ipt_field(grid, m, 0)
    α1 = _ipt_field(grid, m, 1)
    err = try; inner_product(grid, α0, α1); nothing; catch e; e; end
    @test err isa ArgumentError && occursin("grade 0", err.msg) && occursin("grade 1", err.msg)

    other = GridBase(2, 2; metric = m)       # equal shape, different instance
    βother = _ipt_field(other, m, 0)
    @test_throws ArgumentError inner_product(grid, α0, βother)
    @test_throws ArgumentError inner_product(grid, βother, α0)

    # type-stable scalar return
    @test (@inferred inner_product(grid, α0, _ipt_field(grid, m, 0; seed = 2))) isa R
end

@testset "inner_product: symbolic-R bilinearity" begin
    symbolics_available = try
        @eval using Symbolics
        symbolic_metric(signature_metric(VectorSpace(2), R, 2, 0, 0))
        true
    catch err
        @info "Symbolics.jl unavailable or extension failed; symbolic inner-product test skipped." exception=(err, catch_backtrace())
        false
    end

    if symbolics_available
        S = Symbolics.Num
        m = symbolic_metric(signature_metric(VectorSpace(2), R, 2, 0, 0))
        grid = GridBase(2, 2; metric = m)
        a = Symbolics.variable(:a)
        b = Symbolics.variable(:b)
        e1 = clifford_basis_vector(m, 1)
        e12 = clifford_basis_element(m, [1, 2])
        α = Field(grid, 1, Dict(1 => a * e1, 2 => a * e12))
        β = Field(grid, 1, Dict(1 => b * e1, 3 => b * e12))
        γ = Field(grid, 1, Dict(2 => (a + b) * e1))

        # bilinearity over Num, decided by isequal_simplified on scalar wrappers
        lhs = inner_product(grid, a * α + b * γ, β)
        rhs = a * inner_product(grid, α, β) + b * inner_product(grid, γ, β)
        @test isequal_simplified(clifford_scalar(m, lhs), clifford_scalar(m, rhs))

        # symmetry over Num
        @test isequal_simplified(clifford_scalar(m, inner_product(grid, α, β)),
                                 clifford_scalar(m, inner_product(grid, β, α)))
    end
end
