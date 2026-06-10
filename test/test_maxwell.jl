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

@testset "Maxwell: potential-first field strength and known nonzero source" begin
    grid = GridBase(1, 1)
    m = grid.metric
    e1 = clifford_basis_vector(m, 1)

    A = Field(grid, 1, Dict(1 => e1))
    F = electromagnetic_field(A)
    @test F == d(A)
    @test length(maxwell_bianchi(A)) == 0

    expected_J = Field(grid, 1, Dict(
        1 => -e1,
        2 => e1,
        3 => e1,
        4 => -e1,
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
