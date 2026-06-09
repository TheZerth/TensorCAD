# ─────────────────────────────────────────────────────────────────────────────
# Phase 7: Number systems as Clifford (sub)algebras + dual-number autodiff
#
# DESIGN.md §6: ℂ, split-complex, dual numbers, and ℍ are not new types — they
# are Cl(p,q,r) for the right signature.  Dual numbers also serve as the scalar
# ring R, giving exact forward-mode AD across the whole library for free.
# ─────────────────────────────────────────────────────────────────────────────

R = Rational{BigInt}

# ─────────────────────────────────────────────────────────────────────────────
@testset "ℂ — complex numbers as Cl(0,1)" begin
    i = imaginary_unit(R)
    g = complex_metric(R)
    @test i * i == clifford_scalar(g, R(-1))                 # i² = −1
    z = complex_number(R(2), R(3))                            # 2 + 3i
    @test complex_real(z) == R(2)
    @test complex_imag(z) == R(3)
    @test complex_conjugate(z) == complex_number(R(2), R(-3)) # 2 − 3i
    # (a+bi)(c+di) = (ac−bd) + (ad+bc)i
    z1 = complex_number(R(1), R(2)); z2 = complex_number(R(3), R(4))
    @test z1 * z2 == complex_number(R(1*3 - 2*4), R(1*4 + 2*3))
    @test z * complex_conjugate(z) == clifford_scalar(g, R(13))  # |z|² = 4+9
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "Split-complex numbers as Cl(1,0)" begin
    g = split_complex_metric(R)
    j = clifford_basis_vector(g, 1)
    @test j * j == clifford_one(g)                           # j² = +1
    w = split_complex_number(R(2), R(3))
    # (2+3j)(2−3j) = 4 − 9 j² = 4 − 9 = −5
    @test w * split_complex_number(R(2), R(-3)) == clifford_scalar(g, R(-5))
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "Dual numbers as Cl(0,0,1)" begin
    g = dual_clifford_metric(R)
    ε = clifford_basis_vector(g, 1)
    @test iszero(ε * ε)                                      # ε² = 0
    d = dual_clifford_number(R(2), R(3))
    @test d * d == clifford_scalar(g, R(4)) + R(12) * ε      # (2+3ε)² = 4 + 12ε
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "Quaternions ℍ as even subalgebra Cl⁺(3,0)" begin
    one_, i, j, k = quaternion_basis(R)

    @testset "i² = j² = k² = ijk = −1" begin
        @test i * i == -one_
        @test j * j == -one_
        @test k * k == -one_
        @test i * j * k == -one_
    end

    @testset "Hamilton relations ij=k, jk=i, ki=j (and anticommute)" begin
        @test i * j == k
        @test j * k == i
        @test k * i == j
        @test j * i == -k
    end

    @testset "quaternion builder + Hamilton product" begin
        q1 = quaternion(R(1), R(2), R(3), R(4))
        q2 = quaternion(R(5), R(6), R(7), R(8))
        # reference Hamilton product
        a1,b1,c1,d1 = 1,2,3,4; a2,b2,c2,d2 = 5,6,7,8
        ref = quaternion(
            R(a1*a2 - b1*b2 - c1*c2 - d1*d2),
            R(a1*b2 + b1*a2 + c1*d2 - d1*c2),
            R(a1*c2 - b1*d2 + c1*a2 + d1*b2),
            R(a1*d2 + b1*c2 - c1*b2 + d1*a2))
        @test q1 * q2 == ref
        # norm: q q̄ for q=1+2i+3j+4k is 1+4+9+16 = 30 (q̄ = clifford conjugate)
        @test scalar_product(q1, reversion(q1)) == R(30)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "Dual{T} scalar ring — basic axioms" begin
    a = Dual{R}(R(2), R(1)); b = Dual{R}(R(3), R(-1))
    @test a + b == Dual{R}(R(5), R(0))
    @test a * b == Dual{R}(R(6), R(3 - 2))               # product rule: 1·3 + 2·(−1)
    @test zero(Dual{R}) == Dual{R}(R(0), R(0))
    @test one(Dual{R}) == Dual{R}(R(1), R(0))
    @test Dual{R}(7) == Dual{R}(R(7), R(0))              # from Integer
    @test iszero(zero(Dual{R}))
    @test dual_value(a) == R(2) && dual_deriv(a) == R(1)
    # quotient rule: d(1/x) = −1/x²  at x = 2
    @test one(Dual{R}) / dual_seed(R(2)) == Dual{R}(R(1)//2, R(-1)//4)
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "Forward-mode AD through the library (Dual as R)" begin
    # f(a + bε) = f(a) + f′(a)·b·ε, with f evaluated through Clifford operations.

    @testset "f(x) = x² via scalar_square of x·e₁ in Cl(1,0)" begin
        g  = signature_metric(VectorSpace(1), Dual{R}, 1, 0, 0)  # e₁² = +1
        e1 = clifford_basis_vector(g, 1)
        for a in (R(3), R(-2), R(5))
            x  = dual_seed(a) * e1
            ss = scalar_square(x)                # = x² (exact)
            @test ss == Dual{R}(a * a, R(2) * a) # value x², derivative 2x
        end
    end

    @testset "f(x) = x³ via the geometric product" begin
        g  = signature_metric(VectorSpace(1), Dual{R}, 1, 0, 0)
        e1 = clifford_basis_vector(g, 1)
        a  = R(2)
        x  = dual_seed(a) * e1
        cube = x * x * x                          # = x³ · e₁ (e₁³ = e₁)
        @test cube.terms[[1]] == Dual{R}(a^3, R(3) * a^2)   # 8 + 12ε
    end

    @testset "polynomial f(x) = 1 + x² via (1 + x e₁)² scalar part" begin
        g  = signature_metric(VectorSpace(1), Dual{R}, 1, 0, 0)
        e1 = clifford_basis_vector(g, 1)
        a  = R(4)
        p  = clifford_one(g) + dual_seed(a) * e1
        sq = p * p                                # (1 + x e₁)² = (1+x²) + 2x e₁
        @test sq.terms[Int[]] == Dual{R}(R(1) + a^2, R(2) * a)   # f=1+x², f′=2x
        @test sq.terms[[1]]   == Dual{R}(R(2) * a, R(2))
    end

    @testset "AD through magnitude over a Float dual ring (needs sqrt)" begin
        DF = Dual{Float64}
        @test has_sqrt(DF)
        g  = signature_metric(VectorSpace(2), DF, 2, 0, 0)
        # f(x) = √(x² + 16) at x = 3 ⇒ f = 5, f′ = x/√(x²+16) = 3/5 = 0.6
        v  = Dual{Float64}(3.0, 1.0) * clifford_basis_vector(g, 1) +
             Dual{Float64}(4.0, 0.0) * clifford_basis_vector(g, 2)
        m  = magnitude(v)
        @test isapprox(dual_value(m), 5.0; atol = 1e-12)
        @test isapprox(dual_deriv(m), 0.6; atol = 1e-12)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Symbolic path (skipped gracefully if Symbolics is unavailable).
# ─────────────────────────────────────────────────────────────────────────────
symbolics_available = try
    @eval using Symbolics
    true
catch
    false
end

if !symbolics_available
    @info "Symbolics.jl not installed — Phase 7 number-systems symbolic tests skipped."
else
    @testset "Phase 7 number systems symbolic path (Symbolics.Num)" begin
        S = Symbolics.Num
        @testset "complex multiplication with symbolic components" begin
            a, b, c, d = symbolic_vars(:z, 4)
            z1 = complex_number(a, b); z2 = complex_number(c, d)
            expected = complex_number(a*c - b*d, a*d + b*c)
            @test isequal_simplified(z1 * z2, expected)
        end
        @testset "quaternion units over symbolic ring still satisfy i²=−1" begin
            one_, i, _, _ = quaternion_basis(S)
            @test isequal_simplified(i * i, -one_)
        end
    end
end
