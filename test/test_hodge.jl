# ─────────────────────────────────────────────────────────────────────────────
# Phase L8.2 (Tier 3): Hodge star, codifferential, Hodge-Laplacian
# ─────────────────────────────────────────────────────────────────────────────

R = Rational{BigInt}

_hscalar(m, x) = clifford_scalar(m, R(x))
_hcoeff(A, idx) = get(A.terms, idx, zero(eltype(A.metric.g)))

@testset "Hodge star: capability gating and star-square signs" begin
    m = signature_metric(VectorSpace(2), R, 2, 0, 0)
    graph = GraphBase(2, [(1, 2)]; metric = m)
    grid = GridBase(1, 1; metric = m)

    f0 = Field(grid, 0, Dict(1 => clifford_one(m)))
    f1 = Field(grid, 1, Dict(1 => clifford_basis_vector(m, 1)))
    f2 = Field(grid, 2, Dict(1 => clifford_basis_element(m, [1, 2])))

    @test_throws ArgumentError hodge_star(graph, Field(graph, 0, Dict(1 => clifford_one(m))))
    @test_throws ArgumentError codifferential(graph, Field(graph, 0, Dict(1 => clifford_one(m))))
    @test_throws ArgumentError hodge_laplacian(graph, Field(graph, 0, Dict(1 => clifford_one(m))))

    @test (@inferred hodge_star(grid, f0)) isa Field{R,CliffordTensor{R},typeof(grid)}
    @test hodge_star(grid, hodge_star(grid, f0)) == f0
    @test hodge_star(grid, hodge_star(grid, f1)) == -f1
    @test hodge_star(grid, hodge_star(grid, f2)) == f2

    lm = signature_metric(VectorSpace(2), R, 1, 1, 0)
    lgrid = GridBase(1, 1; metric = lm)
    l0 = Field(lgrid, 0, Dict(1 => clifford_one(lm)))
    l1 = Field(lgrid, 1, Dict(1 => clifford_basis_vector(lm, 1)))
    l2 = Field(lgrid, 2, Dict(1 => clifford_basis_element(lm, [1, 2])))
    @test hodge_star(lgrid, hodge_star(lgrid, l0)) == -l0
    @test hodge_star(lgrid, hodge_star(lgrid, l1)) == l1
    @test hodge_star(lgrid, hodge_star(lgrid, l2)) == -l2
end

@testset "Codifferential and Hodge-Laplacian identities" begin
    grid = GridBase(1, 1)
    m = grid.metric

    α = Field(grid, 1, Dict(
        1 => clifford_basis_vector(m, 1),
        2 => R(2) * clifford_basis_vector(m, 1),
        3 => R(3) * clifford_basis_vector(m, 2),
        4 => R(4) * clifford_basis_vector(m, 2),
    ))
    @test field_grade(codifferential(grid, α)) == 0
    @test length(codifferential(grid, codifferential(grid, α))) == 0

    left = @inferred hodge_laplacian(grid, α)
    right = d(codifferential(grid, α)) + codifferential(grid, d(α))
    @test left == right

    const0 = Field(grid, 0, Dict(c => clifford_one(m) for c in cells(grid, 0)))
    @test length(hodge_laplacian(grid, const0)) == 0
end

@testset "Hodge symbolic-R path" begin
    symbolics_available = try
        @eval using Symbolics
        symbolic_metric(signature_metric(VectorSpace(2), R, 2, 0, 0))
        true
    catch err
        @info "Symbolics.jl unavailable or extension failed; symbolic Hodge test skipped." exception=(err, catch_backtrace())
        false
    end

    if symbolics_available
        S = Symbolics.Num
        m = symbolic_metric(signature_metric(VectorSpace(2), R, 2, 0, 0))
        grid = GridBase(1, 1; metric = m)
        a = Symbolics.variable(:a)
        f0 = Field(grid, 0, Dict(1 => clifford_scalar(m, a)))
        @test isequal_simplified(hodge_star(grid, hodge_star(grid, f0))[1], f0[1])
        @test length(codifferential(grid, codifferential(grid, d(f0)))) == 0
    end
end
