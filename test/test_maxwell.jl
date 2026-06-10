# ─────────────────────────────────────────────────────────────────────────────
# Phase L8.2: Maxwell identities assembled from d, ⋆, δ, and Δ
# ─────────────────────────────────────────────────────────────────────────────

R = Rational{BigInt}

@testset "Maxwell: potential-first field strength and known source" begin
    grid = GridBase(1, 1)
    m = grid.metric

    A = Field(grid, 1, Dict(1 => clifford_basis_vector(m, 1)))
    F = electromagnetic_field(A)
    @test F == d(A)
    @test length(maxwell_bianchi(A)) == 0

    J = maxwell_current(grid, A)
    @test J == codifferential(grid, F)
    @test source_equation_residual(grid, F, J) == zero_field(grid, field_grade(J), fibre(grid, field_grade(J), first(cells(grid, field_grade(J)))))
end

@testset "Maxwell: source-free plane-wave validation" begin
    grid = GridBase(2, 2)
    demo = plane_wave_demo(grid)
    F = demo.F
    @test demo.potential_first
    @test F == electromagnetic_field(demo.A)
    @test length(F) > 0
    @test length(demo.dF) == 0
    @test length(demo.deltaF) == 0
    @test length(demo.laplacianF) == 0
    @test source_free_maxwell(grid, F)
end

@testset "Maxwell: capability gating" begin
    m = signature_metric(VectorSpace(2), R, 2, 0, 0)
    graph = GraphBase(2, [(1, 2)]; metric = m)
    F = Field(graph, 1, Dict(1 => clifford_basis_vector(m, 1)))
    @test_throws ArgumentError source_free_maxwell(graph, F)
end
