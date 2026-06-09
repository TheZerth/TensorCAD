# ─────────────────────────────────────────────────────────────────────────────
# Phase L8 (Tier 1): topological exterior derivative d
#
# Validates that d is the metric-free coboundary (transpose of boundary), with
# exact d² = 0, discrete Stokes on a grid patch, grade shadows grad/curl, graph
# support without metric/transport, type stability, and symbolic-R safety.
# ─────────────────────────────────────────────────────────────────────────────

R = Rational{BigInt}

function _cl_scalar(m, x)
    clifford_scalar(m, R(x))
end

function _sum_values(fld)
    acc = zero_fibre(fibre(fld.base, field_grade(fld), first(cells(fld.base, field_grade(fld)))))
    for c in cells(fld.base, field_grade(fld))
        acc = acc + evaluate(fld, c)
    end
    acc
end

@testset "Exterior derivative: exact d² = 0 on GridBase" begin
    grid = GridBase(2, 2)
    m = grid.metric

    f0 = Field(grid, 0, Dict(
        1 => _cl_scalar(m, 2),
        2 => _cl_scalar(m, -3),
        4 => _cl_scalar(m, 5),
        9 => _cl_scalar(m, 7),
    ))
    dd0 = d(d(f0))
    @test field_grade(dd0) == 2
    @test length(dd0) == 0
    @test all(iszero(evaluate(dd0, face)) for face in cells(grid, 2))

    f1 = Field(grid, 1, Dict(
        1 => _cl_scalar(m, 1),
        2 => _cl_scalar(m, -2),
        5 => _cl_scalar(m, 3),
        8 => _cl_scalar(m, -4),
        12 => _cl_scalar(m, 5),
    ))
    dd1 = d(d(f1))
    @test field_grade(dd1) == 3
    @test length(dd1) == 0
end

@testset "Exterior derivative: discrete Stokes on a 2×2 grid" begin
    grid = GridBase(2, 2)
    m = grid.metric

    # Arbitrary edge cochain.  Summing dω over all four faces must telescope to
    # the signed contribution of boundary edges only; interior shared edges cancel.
    vals = Dict{Int,CliffordTensor{R}}(
        e => _cl_scalar(m, e^2 - 3e + 1) for e in cells(grid, 1)
    )
    omega = Field(grid, 1, vals)
    domega = d(omega)

    lhs = _sum_values(domega)

    boundary_weights = Dict{Int,Int}()
    for face in cells(grid, 2)
        for (edge, sign) in boundary(grid, 2, face)
            boundary_weights[edge] = get(boundary_weights, edge, 0) + sign
        end
    end
    rhs = clifford_zero(m)
    for (edge, weight) in boundary_weights
        rhs = rhs + weight * evaluate(omega, edge)
    end

    @test lhs == rhs
    # At least one interior edge appears with net zero weight, demonstrating the
    # telescoping cancellation in the patch sum.
    @test any(iszero(weight) for weight in values(boundary_weights))
    @test lhs == sum(evaluate(domega, face) for face in cells(grid, 2); init=clifford_zero(m))
end

@testset "Exterior derivative: grade behavior and graph gradient" begin
    grid = GridBase(2, 2)
    m = grid.metric

    f0 = Field(grid, 0, Dict(1 => _cl_scalar(m, 1), 2 => _cl_scalar(m, 4)))
    g0 = grad(f0)
    @test field_grade(g0) == 1
    @test g0 == d(f0)

    f1 = Field(grid, 1, Dict(1 => _cl_scalar(m, 2), 4 => _cl_scalar(m, -1)))
    c1 = curl(f1)
    @test field_grade(c1) == 2
    @test c1 == d(f1)

    top = Field(grid, top_grade(grid), Dict(1 => _cl_scalar(m, 8)))
    above = d(top)
    @test field_grade(above) == top_grade(grid) + 1
    @test length(above) == 0

    gm = signature_metric(VectorSpace(3), R, 3, 0, 0)
    graph = GraphBase(3, [(1, 2), (2, 3)]; metric = gm)
    @test has_metric(graph) == false
    phi = Field(graph, 0, Dict(
        1 => _cl_scalar(gm, 10),
        2 => _cl_scalar(gm, 13),
        3 => _cl_scalar(gm, 11),
    ))
    dphi = d(phi)
    @test field_grade(dphi) == 1
    @test evaluate(dphi, 1) == _cl_scalar(gm, 3)   # head − tail = 13 − 10
    @test evaluate(dphi, 2) == _cl_scalar(gm, -2)  # 11 − 13
    @test length(d(dphi)) == 0                     # no 2-cells on a graph
end

@testset "Exterior derivative: type stability" begin
    grid = GridBase(1, 1)
    m = grid.metric
    f0 = Field(grid, 0, Dict(1 => _cl_scalar(m, 1)))
    @test (@inferred d(f0)) isa Field{R,CliffordTensor{R},typeof(grid)}
end

@testset "Exterior derivative: symbolic-R field" begin
    symbolics_available = try
        @eval using Symbolics
        # Verify the Tensorsmith extension actually loaded; some local Julia
        # depots can have Symbolics optional-extension precompile failures.
        symbolic_metric(signature_metric(VectorSpace(3), R, 3, 0, 0))
        true
    catch err
        @info "Symbolics.jl unavailable or its extension failed to load — symbolic exterior derivative test skipped." exception=(err, catch_backtrace())
        false
    end

    if !symbolics_available
        @info "Symbolics.jl not installed — symbolic exterior derivative test skipped."
    else
        S = Symbolics.Num
        gsym = symbolic_metric(signature_metric(VectorSpace(3), R, 3, 0, 0))
        graph = GraphBase(2, [(1, 2)]; metric = gsym)
        a = Symbolics.variable(:a)
        b = Symbolics.variable(:b)
        phi = Field(graph, 0, Dict(
            1 => clifford_scalar(gsym, a),
            2 => clifford_scalar(gsym, b),
        ))
        dphi = d(phi)
        @test isequal_simplified(evaluate(dphi, 1), clifford_scalar(gsym, b - a))
    end
end
