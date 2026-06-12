# ─────────────────────────────────────────────────────────────────────────────
# Phase L8.2: Maxwell identities assembled from d, ⋆, δ, and Δ
# ─────────────────────────────────────────────────────────────────────────────

R = Rational{BigInt}

_scalar_coeff(A) = get(A.terms, Int[], zero(eltype(A.metric.g)))
_field_max_abs(fld::Field) = isempty(fld.values) ? 0.0 : maximum(abs(float(_scalar_coeff(evaluate(fld, c)))) for c in cells(fld.base, field_grade(fld)))
function _field_relation_error(a::Field, λ, b::Field)
    @assert a.base === b.base
    @assert field_grade(a) == field_grade(b)
    maximum(abs(float(_scalar_coeff(evaluate(a, c)) - λ * _scalar_coeff(evaluate(b, c))))
            for c in cells(a.base, field_grade(a)))
end

@testset "Maxwell: potential-first grade-crossing field and known nonzero source" begin
    grid = GridBase(1, 1)
    m = grid.metric
    e1 = clifford_basis_vector(m, 1)

    A = Field(grid, 1, Dict(1 => e1))
    F = electromagnetic_field(A)
    @test F == d(A)
    @test field_grade(F) == 2
    @test field_grade(hodge_star(grid, F)) == 0
    @test hodge_star(grid, F) isa HodgeDualField{R,CliffordTensor{R},typeof(grid)}
    @test length(maxwell_bianchi(A)) == 0

    # expected_J re-derived INDEPENDENTLY BY HAND from the L8.2.1-corrected δ
    # (DESIGN.md §15.4: never mechanically negate the old baseline).  Two
    # independent routes, worked blade-by-blade:
    #
    # GridBase(1,1) layout (constructor formulas): vertices 1..4; edges
    # 1 = bottom (h, j=0), 2 = top (h, j=1), 3 = left (v, i=0), 4 = right
    # (v, i=1); face 1 with CCW boundary (1,+1), (4,+1), (2,-1), (3,-1).
    # A = e₁ on edge 1 only, so F = dA has the single value
    #   F(face 1) = +A(1) + A(4) - A(2) - A(3) = e₁.
    #
    # Route 1 — the composite δ = (-1)^(n(k+1)+q) ⋆d⋆ with n=2, k=2, q=0,
    # so σ = (-1)^(2·3+0) = +1, in Cl(2,0) where I = e₁₂, I⁻¹ = -e₁₂:
    #   ⋆F   : grade-2 value map is -dual(x) = -x·I⁻¹:
    #          -e₁·(-e₁₂) = e₁e₁e₂ = e₂ at the dual vertex of face 1.
    #   d(⋆F): dual coboundary = primal incidence transpose: the dual edge of
    #          primal edge e receives sign(e in ∂face1)·e₂, i.e.
    #          (+e₂, -e₂, -e₂, +e₂) at the dual edges of edges (1,2,3,4).
    #   ⋆(d⋆F): dual-grade-1 value map is dual(y) = y·I⁻¹:
    #          e₂·(-e₁₂) = -e₂e₁e₂ = +e₁, so each ±e₂ becomes ±e₁.
    #   δF = σ·(...) = (+e₁, -e₁, -e₁, +e₁) on edges (1,2,3,4).
    #
    # Route 2 — the definitional contract δ = dᵀ (the signed-incidence
    # transpose; §15.4): J(e) = Σ_f sign(e in ∂f)·F(f) reading face 1's
    # boundary signs directly: J(1) = +e₁, J(2) = -e₁, J(3) = -e₁, J(4) = +e₁.
    #
    # Both routes agree.
    expected_J = Field(grid, 1, Dict(
        1 => e1,
        2 => -e1,
        3 => -e1,
        4 => e1,
    ))
    J = maxwell_current(grid, A)
    @test length(J) > 0
    @test J == expected_J
    @test source_equation_residual(grid, F, expected_J) ==
          zero_field(grid, field_grade(expected_J), fibre(grid, field_grade(expected_J), first(cells(grid, field_grade(expected_J)))))
end

@testset "Maxwell: oscillatory standing mode eigenrelation" begin
    grid = GridBase(3, 3; metric = signature_metric(VectorSpace(2), Float64, 2, 0, 0))
    demo = plane_wave_demo(grid)
    F = demo.F
    @test demo.potential_first
    @test F == electromagnetic_field(demo.A)
    @test length(F) > 0
    @test length(unique(round(_scalar_coeff(evaluate(F, e)); digits=12) for e in cells(grid, 1))) > 2
    @test length(demo.dF) == 0
    @test _field_max_abs(demo.deltaF) > 1e-6
    # L8.2.1 re-baseline: with the pairing-adjoint δ, Δ is positive-semidefinite
    # on Euclidean bases, so the standing-mode eigenvalue is the POSITIVE
    # λ = 2 - 2cos(π/N) (the Hodge sign; the old negative value was the
    # analyst's-sign Laplacian — DESIGN.md §15.4).
    @test demo.eigenvalue > 0
    @test demo.eigenvalue ≈ 2 - 2cos(pi / (3 + 1))
    @test _field_relation_error(demo.laplacianF, demo.eigenvalue, F) < 1e-12
end

@testset "Maxwell: capability gating and source-free predicate" begin
    m = signature_metric(VectorSpace(2), R, 2, 0, 0)
    graph = GraphBase(2, [(1, 2)]; metric = m)
    F = Field(graph, 1, Dict(1 => clifford_basis_vector(m, 1)))
    @test_throws ArgumentError source_free_maxwell(graph, F)

    grid = GridBase(2, 2)
    zeroF = Field(grid, 1, Dict{Int,CliffordTensor{R}}())
    @test source_free_maxwell(grid, zeroF)
end
