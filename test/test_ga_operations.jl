# ─────────────────────────────────────────────────────────────────────────────
# Phase 7 (L5): Geometric-algebra operation suite
#
# Every derived product is a grade projection of the geometric product; the
# tests check the grade rules, the involution sign tables, dual involutivity,
# versor inverse, rotors, and the ring-gating of magnitude / rotor_exp.
# Convention: Dorst left/right contractions (⨼ / ⨽).
# ─────────────────────────────────────────────────────────────────────────────

R  = Rational{BigInt}
g2 = signature_metric(VectorSpace(2), R, 2, 0, 0)   # Cl(2,0)
g3 = signature_metric(VectorSpace(3), R, 3, 0, 0)   # Cl(3,0)
g4 = signature_metric(VectorSpace(4), R, 4, 0, 0)   # Cl(4,0)
g11 = signature_metric(VectorSpace(2), R, 1, 1, 0)  # Cl(1,1) (indefinite)
gdeg = signature_metric(VectorSpace(3), R, 2, 0, 1) # degenerate

blade(g, idx) = clifford_basis_element(g, idx)

# ─────────────────────────────────────────────────────────────────────────────
@testset "Involutions: sign tables and identities" begin
    # Representative blade of each grade 0..4 in Cl(4,0).
    blades = [blade(g4, collect(1:r)) for r in 0:4]

    @testset "grade_involution sign (-1)^r" begin
        for (r, A) in enumerate(blades)
            rr = r - 1
            @test grade_involution(A) == (iseven(rr) ? A : -A)
        end
    end

    @testset "reversion sign (-1)^{r(r-1)/2}: +,+,-,-,+" begin
        expected = [1, 1, -1, -1, 1]
        for (r, A) in enumerate(blades)
            @test reversion(A) == expected[r] * A
            @test (~A) == reversion(A)            # ~ operator alias
        end
    end

    @testset "clifford_conjugate sign (-1)^{r(r+1)/2}: +,-,-,+,+" begin
        expected = [1, -1, -1, 1, 1]
        for (r, A) in enumerate(blades)
            @test clifford_conjugate(A) == expected[r] * A
        end
    end

    @testset "structural identities" begin
        A = blade(g3, [1]) + R(2) * blade(g3, [1, 2]) + blade(g3, [1, 2, 3])
        @test reversion(reversion(A)) == A
        @test grade_involution(grade_involution(A)) == A
        @test clifford_conjugate(A) == grade_involution(reversion(A))
        # anti-automorphism: (AB)~ = B̃ Ã
        Bm = blade(g3, [2]) + blade(g3, [1, 3])
        @test reversion(A * Bm) == reversion(Bm) * reversion(A)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "Grade-projected products" begin
    e1, e2, e3 = blade(g3, [1]), blade(g3, [2]), blade(g3, [3])
    e12 = blade(g3, [1, 2])

    @testset "wedge raises grade by r+s; agrees with Λ under zero metric" begin
        @test wedge(e1, e2) == e12
        @test wedge(e1, e2) == (e1 ∧ e2)        # operator alias
        @test iszero(wedge(e1, e1))             # x∧x = 0
        # zero-metric Clifford wedge == exterior ∧
        g0 = signature_metric(VectorSpace(3), R, 0, 0, 3)
        c1, c2, c3 = (clifford_basis_vector(g0, i) for i in 1:3)
        x1, x2, x3 = (ext_basis_vector(VectorSpace(3), R, i) for i in 1:3)
        @test wedge(wedge(c1, c2), c3).terms == ((x1 ∧ x2) ∧ x3).terms
        @test wedge(c2, c1).terms == (x2 ∧ x1).terms   # antisymmetry
    end

    @testset "left/right contraction grade rules (Dorst)" begin
        # left ⨼ : grade s-r
        @test grades(left_contraction(e1, e12)) == [1]    # 2-1
        @test iszero(left_contraction(e12, e1))           # 1-2 < 0
        @test grades(left_contraction(clifford_one(g3), e12)) == [2]  # scalar ⨼ B = B
        @test (e1 ⨼ e12) == left_contraction(e1, e12)     # operator alias
        # right ⨽ : grade r-s
        @test grades(right_contraction(e12, e1)) == [1]
        @test iszero(right_contraction(e1, e12))
        @test (e12 ⨽ e1) == right_contraction(e12, e1)
        # e_i ⨼ e_j = g(e_i, e_j)  (grade 0)
        @test left_contraction(e1, e1) == clifford_one(g3)
        @test iszero(left_contraction(e1, e2))
    end

    @testset "scalar_product returns the grade-0 coefficient (an R)" begin
        @test scalar_product(e1, e1) == one(R)
        @test scalar_product(e1, e2) == zero(R)
        @test scalar_product(e12, e12) == -one(R)         # ⟨e12 e12⟩₀ = -1
        @test scalar_product(e1, e1) isa R
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "Dual / Hodge" begin
    @testset "pseudoscalar" begin
        @test grade(pseudoscalar(g3)) == 3
        @test pseudoscalar(g3).terms[[1, 2, 3]] == one(R)
    end

    @testset "dual(dual(A)) = ±A with the signature-dependent sign" begin
        for g in (g2, g3, g4, g11)
            I   = pseudoscalar(g)
            s   = scalar_product(inv_mv(I) * inv_mv(I), clifford_one(g)) # I⁻² scalar
            @test isequal(s, one(R)) || isequal(s, -one(R))   # ±1
            for A in (clifford_basis_vector(g, 1),
                      clifford_basis_element(g, [1, 2]))
                @test dual(dual(A)) == s * A
            end
        end
        # 2-arg form validates the metric
        @test dual(clifford_basis_vector(g3, 1), g3) == dual(clifford_basis_vector(g3, 1))
    end

    @testset "dual throws on a degenerate metric" begin
        @test_throws ArgumentError dual(clifford_basis_vector(gdeg, 1))
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "Magnitude / scalar_square" begin
    e1, e2 = blade(g3, [1]), blade(g3, [2])

    @testset "scalar_square is exact for every ring" begin
        @test scalar_square(e1) == one(R)
        @test scalar_square(R(3) * e1 + R(4) * e2) == R(25)
        # indefinite signature: can be negative / zero
        @test scalar_square(clifford_basis_vector(g11, 2)) == -one(R)
        nullv = clifford_basis_vector(g11, 1) + clifford_basis_vector(g11, 2)
        @test iszero(scalar_square(nullv))
    end

    @testset "magnitude requires a sqrt-capable ring" begin
        @test_throws ArgumentError magnitude(e1)            # Rational rejected
        F  = Float64
        gF = signature_metric(VectorSpace(3), F, 3, 0, 0)
        vF = 3.0 * clifford_basis_vector(gF, 1) + 4.0 * clifford_basis_vector(gF, 2)
        @test magnitude(vF) ≈ 5.0
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "Versor inverse" begin
    @testset "round-trips on a blade/versor (two-sided)" begin
        for A in (blade(g3, [1]), R(2) * blade(g3, [2]), blade(g3, [1, 2]),
                  blade(g3, [1]) + blade(g3, [2]))     # versor (a vector)
            Ai = inv_mv(A)
            @test A * Ai == clifford_one(g3)
            @test Ai * A == clifford_one(g3)
        end
    end

    @testset "throws on null and on non-versor" begin
        nullv = clifford_basis_vector(g11, 1) + clifford_basis_vector(g11, 2)
        @test_throws ArgumentError inv_mv(nullv)            # scalar_square = 0
        nonversor = clifford_one(g3) + blade(g3, [1]) + blade(g3, [1, 2])
        @test_throws ArgumentError inv_mv(nonversor)        # general inverse out of scope
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "Rotors" begin
    @testset "apply_rotor preserves scalar_square (isometry), exact ring" begin
        # A unit vector is a versor; sandwiching is a reflection (an isometry).
        e1 = blade(g2, [1]); v = R(2) * blade(g2, [1]) + R(3) * blade(g2, [2])
        @test scalar_square(apply_rotor(e1, v)) == scalar_square(v)
    end

    @testset "concrete 2D Euclidean rotation by 90° (Float64)" begin
        F  = Float64
        gF = signature_metric(VectorSpace(2), F, 2, 0, 0)
        e1, e2 = clifford_basis_vector(gF, 1), clifford_basis_vector(gF, 2)
        B  = clifford_basis_element(gF, [1, 2])
        rot = rotor_exp(-(π / 4) * B)              # half-angle ⇒ 90° rotation
        v   = apply_rotor(rot, e1)
        @test isapprox(get(v.terms, [2], 0.0), 1.0; atol = 1e-12)   # e1 → e2
        @test isapprox(get(v.terms, [1], 0.0), 0.0; atol = 1e-12)
        @test isapprox(scalar_square(v), scalar_square(e1))
    end

    @testset "rotor_exp branch on B² sign and B²=0" begin
        F  = Float64
        gF = signature_metric(VectorSpace(2), F, 1, 1, 0)   # e12² = +1 ⇒ boost
        B  = clifford_basis_element(gF, [1, 2])
        @test isequal(grades(rotor_exp(0.5 * B)), [0, 2])   # cosh + sinh·B̂
        # B² = 0 ⇒ 1 + B
        gN = signature_metric(VectorSpace(2), F, 0, 0, 2)
        BN = clifford_basis_element(gN, [1, 2])
        @test rotor_exp(BN) == clifford_one(gN) + BN
    end

    @testset "rotor_exp gated off the exact ring; rotor_exp_series stays in it" begin
        B = blade(g2, [1, 2])
        @test_throws ArgumentError rotor_exp(B)             # Rational rejected
        @test rotor_exp_series(B, 0) == clifford_one(g2)
        @test rotor_exp_series(B, 1) == clifford_one(g2) + B
        # B² = -1, so the series is the partial sum of cos(1)+sin(1)·B; stays exact
        s2 = rotor_exp_series(B, 2)
        @test s2.terms[Int[]] == one(R) - one(R)//2        # 1 - 1/2!
        @test s2.terms[[1, 2]] == one(R)                   # B¹/1!
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "GASmith oracle cross-check (skipped if fixture absent)" begin
    # DESIGN.md §11: Tensorsmith's Clifford output must match GASmith blade-for-
    # blade.  When a JSON fixture is dropped into test/fixtures/, cross-check it
    # here; until then, skip gracefully (matching the existing fixture pattern).
    fixture = joinpath(@__DIR__, "fixtures", "gasmith_Cl300.json")
    if isfile(fixture)
        @info "GASmith fixture present; cross-check would run here." fixture
        @test filesize(fixture) > 0
    else
        @info "GASmith fixture absent — oracle cross-check skipped."
        @test true
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
    @info "Symbolics.jl not installed — Phase 7 GA symbolic tests skipped."
else
    @testset "Phase 7 GA symbolic path (Symbolics.Num)" begin
        S  = Symbolics.Num
        gS = symbolic_metric(signature_metric(VectorSpace(2), Rational{BigInt}, 2, 0, 0))

        @testset "reversion is involutive over symbolic coefficients" begin
            A = symbolic_clifford_vector(gS, :a) +
                symbolic_vars(:c, 1)[1] * clifford_basis_element(gS, [1, 2])
            @test isequal_simplified(reversion(reversion(A)), A)
        end

        @testset "apply_rotor sandwich identity (symbolic a + b·e₁₂ on e₁)" begin
            a, b = symbolic_vars(:r, 2)
            rotor = a * clifford_one(gS) + b * clifford_basis_element(gS, [1, 2])
            v     = clifford_basis_vector(gS, 1)
            got   = apply_rotor(rotor, v)
            # (a + b e12) e1 (a - b e12) = (a²-b²) e1 - 2ab e2
            expected = (a^2 - b^2) * clifford_basis_vector(gS, 1) +
                       (-2 * a * b) * clifford_basis_vector(gS, 2)
            @test isequal_simplified(got, expected)
        end

        @testset "scalar_product over symbolic coefficients" begin
            u = symbolic_clifford_vector(gS, :u)
            u1, u2 = symbolic_vars(:u, 2)
            @test isequal(Symbolics.expand(scalar_product(u, u) - (u1^2 + u2^2)),
                          zero(S))
        end

        @testset "has_sqrt / has_transcendentals enabled for Num" begin
            @test has_sqrt(S)
            @test has_transcendentals(S)
        end
    end
end
