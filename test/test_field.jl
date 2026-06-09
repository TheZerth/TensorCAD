# ─────────────────────────────────────────────────────────────────────────────
# Phase L7: Field — sections of a bundle
#
# A Field assigns fibre elements (existing AbstractTensorElement types) to the
# k-cells of a BaseSpace.  Tests cover construction (explicit + functional),
# validation, type-stable evaluation, iteration, pointwise arithmetic, a Cl(3)
# round-trip over GraphBase, and a symbolic-R fibre path (compared with
# isequal_simplified).
# ─────────────────────────────────────────────────────────────────────────────

R    = Rational{BigInt}
clf3 = signature_metric(VectorSpace(3), R, 3, 0, 0)
gb   = GraphBase(4, [(1, 2), (2, 3), (3, 4)]; metric = clf3)

e1  = clifford_basis_vector(clf3, 1)
e2  = clifford_basis_vector(clf3, 2)
e12 = clifford_basis_element(clf3, [1, 2])

# ─────────────────────────────────────────────────────────────────────────────
@testset "Construction & evaluation" begin
    f = Field(gb, 0, Dict(1 => e1, 3 => e2))
    @test field_grade(f) == 0
    @test length(f) == 2
    @test evaluate(f, 1) == e1
    @test f[3] == e2
    @test iszero(evaluate(f, 2))               # unstored cell → fibre zero
    @test iszero(f[4])

    # functional constructor over every cell of the grade
    g = Field(gb, 0, c -> e1)
    @test length(g) == n_cells(gb, 0)
    @test g[2] == e1

    # a section over the 1-cells (edges)
    h = Field(gb, 1, Dict(2 => e12))
    @test field_grade(h) == 1
    @test h[2] == e12
    @test iszero(h[1])

    # zero field
    z = zero_field(gb, 0, fibre(gb, 0, 1))
    @test length(z) == 0
    @test iszero(z[1])
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "Construction validation" begin
    @test_throws ArgumentError Field(gb, 2, Dict(1 => e1))      # grade out of range
    @test_throws ArgumentError Field(gb, 0, Dict(99 => e1))     # not a 0-cell
    # value from a different fibre (different metric)
    other = signature_metric(VectorSpace(3), R, 2, 1, 0)
    @test_throws ArgumentError Field(gb, 0, Dict(1 => clifford_basis_vector(other, 1)))
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "Pointwise arithmetic" begin
    a = Field(gb, 0, Dict(1 => e1, 2 => e2))
    b = Field(gb, 0, Dict(2 => e2, 3 => e1))

    s = a + b
    @test s[1] == e1
    @test s[2] == e2 + e2
    @test s[3] == e1

    @test length(a - a) == 0                   # everything cancels (sparse prune)
    @test (-a)[1] == -e1
    @test (-a)[2] == -e2

    @test (R(3) * a)[1] == R(3) * e1
    @test (a * R(2))[2] == R(2) * e2
    @test (2 * a)[1] == R(2) * e1
    @test (a * 2)[2] == R(2) * e2
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "Arithmetic compatibility checks" begin
    a = Field(gb, 0, Dict(1 => e1))
    other_base = GraphBase(4, [(1, 2)]; metric = clf3)
    @test_throws ArgumentError a + Field(other_base, 0, Dict(1 => e1))   # different base
    @test_throws ArgumentError a + Field(gb, 1, Dict(1 => e1))           # different grade
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "Cl(3)-valued field round-trips over GraphBase" begin
    vals = Dict(1 => e1 + e2, 2 => e12, 4 => R(3) * e1)
    f  = Field(gb, 0, vals)
    f2 = Field(gb, field_grade(f), Dict(c => evaluate(f, c) for c in keys(f)))
    @test f == f2
    for c in keys(f)
        @test evaluate(f2, c) == vals[c]
    end
    # iteration yields the stored cell => element pairs
    @test Dict(f) == vals
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "Type-stable element access" begin
    f = Field(gb, 0, Dict(1 => e1))
    @test (@inferred evaluate(f, 1)) isa CliffordTensor{R}
    @test (@inferred evaluate(f, 3)) isa CliffordTensor{R}
    @test eltype(typeof(f)) == Pair{Int, CliffordTensor{R}}
end

# ─────────────────────────────────────────────────────────────────────────────
# Symbolic (R = Symbolics.Num) fibre — compared with isequal_simplified, never ==.
@testset "Symbolic-R fibre field" begin
    symbolics_available = try
        @eval using Symbolics
        true
    catch
        false
    end

    if !symbolics_available
        @info "Symbolics.jl not installed — symbolic field test skipped."
    else
        S    = Symbolics.Num
        gsym = symbolic_metric(signature_metric(VectorSpace(3), R, 3, 0, 0))
        a    = symbolic_clifford_vector(gsym, :a)
        bvec = symbolic_clifford_vector(gsym, :b)
        gbs  = GraphBase(2, [(1, 2)]; metric = gsym)

        fa = Field(gbs, 0, Dict(1 => a, 2 => bvec))
        fb = Field(gbs, 0, Dict(1 => bvec, 2 => a))
        s  = fa + fb

        @test isequal_simplified(evaluate(s, 1), a + bvec)
        @test isequal_simplified(evaluate(s, 2), a + bvec)
        @test (@inferred evaluate(fa, 1)) isa CliffordTensor{S}
    end
end
