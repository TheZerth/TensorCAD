# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: Symbolics.Num as scalar ring
#
# Tests are skipped gracefully if Symbolics.jl is not installed.
# To install: julia --project=. -e 'using Pkg; Pkg.add("Symbolics")'
#
# Design note: we deliberately avoid the @variables macro so this file has no
# macro-expansion dependency on Symbolics being loaded.  All symbolic scalars
# are created via Symbolics.variable(:name) (plain function) or our own
# symbolic_vars() helper.
# ─────────────────────────────────────────────────────────────────────────────

symbolics_available = try
    @eval using Symbolics
    true
catch
    false
end

if !symbolics_available
    @info "Symbolics.jl not installed — Phase 5 tests skipped." *
          "\nTo enable: julia --project=. -e 'using Pkg; Pkg.add(\"Symbolics\")'"
else

S  = Symbolics.Num   # shorthand throughout
V2 = VectorSpace(2)
V3 = VectorSpace(3)

@testset "Phase 5: Symbolics.Num as scalar ring R" begin

# ─────────────────────────────────────────────────────────────────────────────
@testset "contains_rationals dispatch" begin
    @test Tensorsmith.contains_rationals(S)
    @test Tensorsmith.contains_rationals(Rational{BigInt})
    @test !Tensorsmith.contains_rationals(Float64)
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "symbolic_vars helper" begin
    xs = symbolic_vars(:x, 3)
    @test length(xs) == 3
    @test all(xs[i] isa S for i in 1:3)
    @test !isequal(xs[1], xs[2])
    @test !isequal(xs[1], xs[3])
    @test symbolic_vars(:y, 0) == S[]
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "FreeTensor{Symbolics.Num}" begin

    @testset "Construction and grade" begin
        e1 = basis_vector(V3, S, 1)
        @test e1 isa FreeTensor{S}
        @test grade(e1) == 1
        @test isequal(e1.terms[[1]], one(S))
    end

    @testset "Scalar multiplication" begin
        a  = Symbolics.variable(:a)
        e1 = basis_vector(V3, S, 1)
        t  = a * e1
        @test t isa FreeTensor{S}
        @test isequal(t.terms[[1]], a)
    end

    @testset "Tensor product with symbolic coefficients" begin
        a  = Symbolics.variable(:a)
        b  = Symbolics.variable(:b)
        e1 = basis_vector(V2, S, 1)
        e2 = basis_vector(V2, S, 2)
        lhs = (a * e1) * (b * e2)
        rhs = (b * e2) * (a * e1)
        @test isequal(lhs.terms[[1,2]], a * b)
        @test isequal(rhs.terms[[2,1]], a * b)
        @test !haskey(lhs.terms, [2,1])
        @test !haskey(rhs.terms, [1,2])
    end

    @testset "Grade of symbolic product" begin
        x  = Symbolics.variable(:x)
        y  = Symbolics.variable(:y)
        e1 = basis_vector(V3, S, 1)
        e2 = basis_vector(V3, S, 2)
        e3 = basis_vector(V3, S, 3)
        p  = (x * e1) * (y * e2) * e3
        @test grade(p) == 3
        @test isequal(p.terms[[1,2,3]], x * y)
    end

end  # FreeTensor

# ─────────────────────────────────────────────────────────────────────────────
@testset "ExteriorAlgebra{Symbolics.Num}" begin

    @testset "symbolic_element helper" begin
        a  = symbolic_element(V3, ExteriorAlgebra, :a)
        @test a isa AlgebraTensor{ExteriorAlgebra, S}
        @test grade(a) == 1
        @test length(a.terms) == 3
        xs = symbolic_vars(:a, 3)
        for i in 1:3
            @test isequal(a.terms[[i]], xs[i])
        end
    end

    @testset "Nilpotency: e_i ^ e_i = 0 over symbolic coefficients" begin
        c  = Symbolics.variable(:c)
        e1 = ext_basis_vector(V3, S, 1)
        @test iszero((c * e1) ∧ (c * e1))
    end

    @testset "Antisymmetry: e_i ^ e_j = -(e_j ^ e_i)" begin
        p  = Symbolics.variable(:p)
        q  = Symbolics.variable(:q)
        e1 = ext_basis_vector(V2, S, 1)
        e2 = ext_basis_vector(V2, S, 2)
        lhs = (p * e1) ∧ (q * e2)
        rhs = (q * e2) ∧ (p * e1)
        @test isequal(lhs.terms[[1,2]],  p * q)
        @test isequal(rhs.terms[[1,2]], -(p * q))
    end

    @testset "Wedge product of generic grade-1 elements" begin
        a    = symbolic_element(V2, ExteriorAlgebra, :a)
        b    = symbolic_element(V2, ExteriorAlgebra, :b)
        prod = a ∧ b
        a1, a2 = symbolic_vars(:a, 2)
        b1, b2 = symbolic_vars(:b, 2)
        coef     = prod.terms[[1,2]]
        expected = a1*b2 - a2*b1
        diff     = Symbolics.simplify(coef - expected)
        @test isequal(diff, zero(S))
    end

    @testset "project_ext of symbolic FreeTensor" begin
        x  = Symbolics.variable(:x)
        y  = Symbolics.variable(:y)
        e1 = basis_vector(V2, S, 1)
        e2 = basis_vector(V2, S, 2)
        t  = (x * e1) * (y * e2)
        p  = project_ext(t)
        @test p isa AlgebraTensor{ExteriorAlgebra, S}
        @test isequal(p.terms[[1,2]], x * y)
    end

end  # ExteriorAlgebra

# ─────────────────────────────────────────────────────────────────────────────
@testset "SymmetricAlgebra{Symbolics.Num}" begin

    @testset "symbolic_element helper" begin
        s = symbolic_element(V3, SymmetricAlgebra, :s)
        @test s isa AlgebraTensor{SymmetricAlgebra, S}
        @test grade(s) == 1
    end

    @testset "Commutativity over symbolic coefficients" begin
        p   = Symbolics.variable(:p)
        q   = Symbolics.variable(:q)
        es1 = sym_basis_vector(V2, S, 1)
        es2 = sym_basis_vector(V2, S, 2)
        lhs = (p * es1) * (q * es2)
        rhs = (q * es2) * (p * es1)
        @test isequal(lhs.terms[[1,2]], p * q)
        @test isequal(rhs.terms[[1,2]], p * q)
        # Structural `==` cannot decide equality of symbolic coefficients
        # (`Num == Num` is not a Bool); use the symbolic-equality decision.
        @test isequal_simplified(lhs, rhs)
    end

    @testset "Squaring a generic grade-1 element" begin
        a    = symbolic_element(V2, SymmetricAlgebra, :a)
        a2   = a * a
        a1v, a2v = symbolic_vars(:a, 2)
        @test haskey(a2.terms, [1,1])
        @test haskey(a2.terms, [1,2])
        @test haskey(a2.terms, [2,2])
        @test isequal(a2.terms[[1,1]], a1v^2)
        @test isequal(a2.terms[[2,2]], a2v^2)
        diff = Symbolics.simplify(a2.terms[[1,2]] - 2*a1v*a2v)
        @test isequal(diff, zero(S))
    end

end  # SymmetricAlgebra

# ─────────────────────────────────────────────────────────────────────────────
@testset "CliffordTensor{Symbolics.Num}" begin

    @testset "symbolic_metric and symbolic_clifford_vector" begin
        gE_exact = signature_metric(V3, Rational{BigInt}, 3, 0, 0)
        gE_sym   = symbolic_metric(gE_exact)
        @test gE_sym isa Metric{S}
        @test isequal(gE_sym.g[1,1], one(S))
        @test isequal(gE_sym.g[1,2], zero(S))
        a = symbolic_clifford_vector(gE_sym, :a)
        @test a isa CliffordTensor{S}
        @test grade(a) == 1
        @test length(a.terms) == 3
    end

    @testset "Fundamental Clifford relation over symbolic coefficients" begin
        gE_sym = symbolic_metric(signature_metric(V3, Rational{BigInt}, 3, 0, 0))
        ec = [clifford_basis_vector(gE_sym, S, i) for i in 1:3]
        # e_i^2 = 1 for Euclidean metric
        for i in 1:3
            sq = ec[i] * ec[i]
            @test isequal(sq.terms[Int[]], one(S))
        end
        # e_i*e_j = -e_j*e_i for i != j
        for i in 1:3, j in 1:3
            i == j && continue
            lhs = ec[i] * ec[j]
            rhs = ec[j] * ec[i]
            c_lhs = get(lhs.terms, [min(i,j), max(i,j)], zero(S))
            c_rhs = get(rhs.terms, [min(i,j), max(i,j)], zero(S))
            diff  = Symbolics.simplify(c_lhs + c_rhs)
            @test isequal(diff, zero(S))
        end
    end

    @testset "Symbolic geometric product (grade structure)" begin
        gE_sym = symbolic_metric(signature_metric(V2, Rational{BigInt}, 2, 0, 0))
        a    = symbolic_clifford_vector(gE_sym, :a)
        b    = symbolic_clifford_vector(gE_sym, :b)
        prod = a * b
        g    = grades(prod)
        @test 0 in g
        @test 2 in g
        a1, a2 = symbolic_vars(:a, 2)
        b1, b2 = symbolic_vars(:b, 2)
        scalar_coef    = prod.terms[Int[]]
        expected_scalar = a1*b1 + a2*b2
        diff = Symbolics.simplify(scalar_coef - expected_scalar)
        @test isequal(diff, zero(S))
    end

end  # CliffordTensor

# ─────────────────────────────────────────────────────────────────────────────
@testset "antisymmetrize over Symbolics.Num" begin

    @testset "Grade-1: identity" begin
        c  = Symbolics.variable(:c)
        e1 = ext_basis_vector(V3, S, 1)
        t  = c * e1
        result = antisymmetrize(t)
        @test result isa FreeTensor{S}
        @test isequal(result.terms[[1]], c)
    end

    @testset "Grade-2: (1/2)(e1xe2 - e2xe1) with symbolic coef" begin
        c    = Symbolics.variable(:c)
        e1   = ext_basis_vector(V2, S, 1)
        e2   = ext_basis_vector(V2, S, 2)
        blade  = (c * e1) ∧ e2
        result = antisymmetrize(blade)
        @test isequal(result.terms[[1,2]], c * S(1//2))
        @test isequal(result.terms[[2,1]], c * S(-1//2))
    end

    @testset "Round-trip: project_ext(antisymmetrize(t)) == t" begin
        c     = Symbolics.variable(:c)
        e1    = ext_basis_vector(V2, S, 1)
        e2    = ext_basis_vector(V2, S, 2)
        blade = c * (e1 ∧ e2)
        rt    = project_ext(antisymmetrize(blade))
        @test isequal(rt.terms[[1,2]], c)
    end

end  # antisymmetrize

# ─────────────────────────────────────────────────────────────────────────────
@testset "Symbol maps over Symbolics.Num" begin

    @testset "ext_to_cl and cl_to_ext preserve symbolic coefficients" begin
        c  = Symbolics.variable(:c)
        d  = Symbolics.variable(:d)
        e1 = ext_basis_vector(V2, S, 1)
        e2 = ext_basis_vector(V2, S, 2)
        blade  = c * e1 + d * (e1 ∧ e2)
        g0_sym = symbolic_metric(zero_metric(V2, Rational{BigInt}))
        cl = ext_to_cl(blade, g0_sym)
        rt = cl_to_ext(cl)
        @test isequal(rt.terms[[1]],   c)
        @test isequal(rt.terms[[1,2]], d)
    end

end  # Symbol maps

# ─────────────────────────────────────────────────────────────────────────────
@testset "show does not throw on symbolic coefficients" begin
    # Before the isequal fix, c == one(R) returned Num (not Bool) and threw.
    c  = Symbolics.variable(:c)
    e1 = basis_vector(V2, S, 1)
    e2 = basis_vector(V2, S, 2)
    t  = c * e1 + e2          # mixed: symbolic coef + unit coef
    @test_nowarn repr(t)      # FreeTensor show
    et = symbolic_element(V2, ExteriorAlgebra, :a)
    @test_nowarn repr(et)     # AlgebraTensor show
    gE_sym = symbolic_metric(signature_metric(V2, Rational{BigInt}, 2, 0, 0))
    cv = symbolic_clifford_vector(gE_sym, :v)
    @test_nowarn repr(cv)     # CliffordTensor show
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "isequal_simplified" begin
    gE_sym = symbolic_metric(signature_metric(V2, Rational{BigInt}, 2, 0, 0))
    a  = symbolic_clifford_vector(gE_sym, :a)
    b  = symbolic_clifford_vector(gE_sym, :b)

    # a*b + b*a = 2*(a1*b1 + a2*b2) * scalar  (commutator = 0 for grade-1)
    sum1 = a * b + b * a
    sum2 = b * a + a * b
    @test isequal_simplified(sum1, sum2)

    # An element is equal to itself
    @test isequal_simplified(a, a)

    # Distinct elements are not equal
    @test !isequal_simplified(a, b)

    # Commutativity of symbolic scalar multiplication
    x  = Symbolics.variable(:x)
    y  = Symbolics.variable(:y)
    e1 = ext_basis_vector(V2, S, 1)
    t1 = (x * e1) + (y * ext_basis_vector(V2, S, 2))
    t2 = (y * ext_basis_vector(V2, S, 2)) + (x * e1)
    # Structural == may fail (different insertion order); isequal_simplified must pass
    @test isequal_simplified(t1, t2)
end

end  # @testset Phase 5

end  # if symbolics_available
