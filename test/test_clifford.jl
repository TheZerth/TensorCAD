# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Metric + Clifford algebra tests
#
# Test strategy:
#   1. Metric construction and accessors
#   2. _clifford_normalize! directly (unit tests on the core algorithm)
#   3. Fundamental Clifford relation: eᵢeⱼ + eⱼeᵢ = 2g(eᵢ,eⱼ)·1
#   4. Specific signature algebras: Cl(1,0), Cl(0,1), Cl(0,0,1) = Exterior
#   5. MANDATORY: Cl(Q=0) reproduces Exterior exactly
#   6. Cl(3,0): quaternion structure, pseudoscalar, grade dimension
#   7. Ring and basis properties
#   8. General (non-diagonal) metric
#   9. GASmith cross-check (skipped gracefully if fixtures absent)
# ─────────────────────────────────────────────────────────────────────────────

R  = Rational{BigInt}
V1 = VectorSpace(1)
V2 = VectorSpace(2)
V3 = VectorSpace(3)
V4 = VectorSpace(4)

# Common metrics
g0_3  = zero_metric(V3, R)          # Q=0 on ℝ³ → isomorphic to Λ(ℝ³)
gE_3  = signature_metric(V3, R, 3, 0, 0)   # Euclidean Cl(3,0)
gL_3  = signature_metric(V3, R, 2, 1, 0)   # Lorentzian Cl(2,1)
gCl10 = signature_metric(V1, R, 1, 0, 0)   # ℂ: e₁² = +1
gCl01 = signature_metric(V1, R, 0, 1, 0)   # "split": e₁² = −1
gCl00 = signature_metric(V1, R, 0, 0, 1)   # null: e₁² = 0

# Clifford basis vectors for 3-dim Euclidean
ec = [clifford_basis_vector(gE_3, R, i) for i in 1:3]
# One scalar
cl1 = clifford_one(gE_3, R)

# ─────────────────────────────────────────────────────────────────────────────
@testset "Metric type — Phase 3" begin

    @testset "Construction from matrix" begin
        g = Matrix{R}([R(1) R(0); R(0) R(-1)])
        m = Metric{R}(V2, g)
        @test m.space == V2
        @test m.g == g
    end

    @testset "signature_metric" begin
        m = signature_metric(V3, R, 2, 1, 0)
        @test quadratic_form(m, 1) ==  one(R)
        @test quadratic_form(m, 2) ==  one(R)
        @test quadratic_form(m, 3) == -one(R)
        @test bilinear_form(m, 1, 2) == zero(R)
        @test bilinear_form(m, 1, 3) == zero(R)
    end

    @testset "diagonal_metric" begin
        d = diagonal_metric(V3, R, [R(1), R(-1), R(0)])
        @test quadratic_form(d, 1) ==  one(R)
        @test quadratic_form(d, 2) == -one(R)
        @test quadratic_form(d, 3) ==  zero(R)
        @test bilinear_form(d, 1, 2) == zero(R)
    end

    @testset "zero_metric" begin
        m = zero_metric(V3, R)
        for i in 1:3, j in 1:3
            @test bilinear_form(m, i, j) == zero(R)
        end
    end

    @testset "symmetry check on construction" begin
        # Non-symmetric matrix must be rejected
        bad = Matrix{R}([R(1) R(2); R(3) R(1)])   # g[1,2] ≠ g[2,1]
        @test_throws ArgumentError Metric{R}(V2, bad)
    end

    @testset "dimension mismatch check" begin
        bad_size = Matrix{R}([R(1) R(0) R(0); R(0) R(1) R(0); R(0) R(0) R(1)])
        @test_throws ArgumentError Metric{R}(V2, bad_size)  # 3×3 for 2-dim space
    end

    @testset "equality and hashing" begin
        m1 = signature_metric(V3, R, 3, 0, 0)
        m2 = signature_metric(V3, R, 3, 0, 0)
        m3 = signature_metric(V3, R, 2, 1, 0)
        @test m1 == m2
        @test m1 != m3
        @test hash(m1) == hash(m2)
    end

end  # Metric type

# ─────────────────────────────────────────────────────────────────────────────
@testset "_clifford_normalize! unit tests" begin

    function cl_norm(g_mat, idx, c)
        result = Dict{Vector{Int}, R}()
        Tensorsmith._clifford_normalize!(result, g_mat, idx, c)
        result
    end

    # ── Diagonal Euclidean metric (all +1) ───────────────────────────────────
    gE2_mat = gE_3.g   # 3×3 identity over R

    @testset "Already canonical: no change" begin
        @test cl_norm(gE2_mat, [1,2,3], R(1)) == Dict([1,2,3] => R(1))
        @test cl_norm(gE2_mat, Int[], R(5))    == Dict(Int[] => R(5))
        @test cl_norm(gE2_mat, [2], R(-1))     == Dict([2] => R(-1))
    end

    @testset "Single swap: [2,1] → −[1,2]" begin
        r = cl_norm(gE2_mat, [2,1], R(1))
        @test r == Dict([1,2] => R(-1))
    end

    @testset "Contract equal pair: [1,1] → g₁₁ = 1" begin
        r = cl_norm(gE2_mat, [1,1], R(1))
        @test r == Dict(Int[] => R(1))
    end

    @testset "Contract [2,2] with coef 3 → 3" begin
        r = cl_norm(gE2_mat, [2,2], R(3))
        @test r == Dict(Int[] => R(3))
    end

    @testset "Null metric: [1,1] → 0" begin
        g0_mat = zero_metric(V1, R).g
        r = cl_norm(g0_mat, [1,1], R(1))
        @test isempty(r)   # zero element
    end

    @testset "Contract + permute: [1,1,2] → [2]" begin
        # [1,1,2]: p=1 equal → contract eᵢ² = g₁₁ = 1, giving [2] with coef 1
        r = cl_norm(gE2_mat, [1,1,2], R(1))
        @test r == Dict([2] => R(1))
    end

    @testset "[2,1,1] → [2]" begin
        # [2,1,1]: p=1: 2>1 → swap to [1,2,1] coef -1, no cross-term (diag)
        #   [1,2,1]: p=2: 2>1 → swap to [1,1,2] coef +1, no cross-term
        #     [1,1,2]: contract → [2] coef +1
        r = cl_norm(gE2_mat, [2,1,1], R(1))
        @test r == Dict([2] => R(1))
    end

    @testset "Negative metric: [1,1] with g₁₁=−1 → −1" begin
        g_neg = signature_metric(V1, R, 0, 1, 0).g
        r = cl_norm(g_neg, [1,1], R(1))
        @test r == Dict(Int[] => R(-1))
    end

    @testset "Off-diagonal g: [2,1] with g₁₂=1//2" begin
        # Hyperbolic plane: g = [[0, 1//2],[1//2, 0]]
        g_hyp = Matrix{R}([R(0) R(1//2); R(1//2) R(0)])
        # [2,1]: swap → -[1,2] + 2·g(2,1)·[] = -[1,2] + 2·(1//2)·1·[] = -[1,2] + []
        r = cl_norm(g_hyp, [2,1], R(1))
        @test get(r, [1,2], zero(R)) == R(-1)
        @test get(r, Int[], zero(R)) == R(1)
        @test length(r) == 2
    end

end  # _clifford_normalize! unit tests

# ─────────────────────────────────────────────────────────────────────────────
@testset "Fundamental Clifford relation — Phase 3" begin

    # For any algebra the relation eᵢeⱼ + eⱼeᵢ = 2g(eᵢ,eⱼ)·1 must hold.

    @testset "Cl(3,0): all pairs eᵢeⱼ + eⱼeᵢ = 2gᵢⱼ·1" begin
        for i in 1:3, j in 1:3
            lhs = ec[i] * ec[j] + ec[j] * ec[i]
            expected_val = R(2) * bilinear_form(gE_3, i, j)
            rhs = clifford_scalar(gE_3, expected_val)
            @test lhs == rhs
        end
    end

    @testset "Cl(3,0): eᵢ² = 1 for i = 1,2,3" begin
        for i in 1:3
            @test ec[i] * ec[i] == cl1
        end
    end

    @testset "Cl(3,0): eᵢeⱼ = −eⱼeᵢ for i ≠ j" begin
        for i in 1:3, j in 1:3
            i == j && continue
            @test ec[i] * ec[j] == -(ec[j] * ec[i])
        end
    end

    @testset "Cl(2,1): mixed signature" begin
        eL = [clifford_basis_vector(gL_3, R, i) for i in 1:3]
        # e₁² = +1, e₂² = +1, e₃² = −1
        @test eL[1] * eL[1] == clifford_one(gL_3, R)
        @test eL[2] * eL[2] == clifford_one(gL_3, R)
        @test eL[3] * eL[3] == clifford_scalar(gL_3, R(-1))
        # Off-diagonal: anticommute (diagonal metric, g_{ij}=0 for i≠j)
        @test eL[1] * eL[2] == -(eL[2] * eL[1])
        @test eL[1] * eL[3] == -(eL[3] * eL[1])
    end

    @testset "Cl(1,0): e₁² = +1" begin
        e_pos = clifford_basis_vector(gCl10, R, 1)
        @test e_pos * e_pos == clifford_one(gCl10, R)
    end

    @testset "Cl(0,1): e₁² = −1" begin
        e_neg = clifford_basis_vector(gCl01, R, 1)
        @test e_neg * e_neg == clifford_scalar(gCl01, R(-1))
    end

    @testset "Cl(0,0,1): e₁² = 0 (null direction)" begin
        e_null = clifford_basis_vector(gCl00, R, 1)
        @test iszero(e_null * e_null)
    end

end  # Fundamental Clifford relation

# ─────────────────────────────────────────────────────────────────────────────
@testset "MANDATORY: Q=0 reproduces Exterior algebra exactly" begin
    # When g ≡ 0, Clifford reduces to Exterior.
    # This is the mandatory regression test: if Q=0 fails, the normalization
    # is wrong.

    ec0 = [clifford_basis_vector(g0_3, R, i) for i in 1:3]  # Clifford with g=0
    ee  = [ext_basis_vector(V3, R, i) for i in 1:3]          # Exterior (Phase 2)

    @testset "e₁² = 0 (null Clifford = Exterior)" begin
        for i in 1:3
            @test iszero(ec0[i] * ec0[i])
        end
    end

    @testset "Antisymmetry eᵢ∧eⱼ = −eⱼ∧eᵢ" begin
        for i in 1:3, j in 1:3
            cl_prod = ec0[i] * ec0[j]
            ext_prod = ee[i] ∧ ee[j]
            # Same sign structure
            @test iszero(cl_prod) == iszero(ext_prod)
            if !iszero(cl_prod) && !iszero(ext_prod)
                # The stored coefficient for the canonical index must match
                cl_key  = only(keys(cl_prod.terms))
                ext_key = only(keys(ext_prod.terms))
                @test cl_key  == ext_key
                @test cl_prod.terms[cl_key] == ext_prod.terms[ext_key]
            end
        end
    end

    @testset "All grade-2 products match" begin
        for i in 1:3, j in 1:3
            i == j && continue
            lhs = ec0[i] * ec0[j]
            rhs_sign = (i < j) ? R(1) : R(-1)
            canonical_key = [min(i,j), max(i,j)]
            @test !iszero(lhs)
            @test lhs.terms[canonical_key] == rhs_sign
        end
    end

    @testset "Triple product: e₃*e₁*e₂ with Q=0 matches Ext sign" begin
        cl_top = ec0[1] * ec0[2] * ec0[3]
        ext_top = ee[1] ∧ ee[2] ∧ ee[3]
        @test cl_top.terms[[1,2,3]] == ext_top.terms[[1,2,3]]
        @test cl_top.terms[[1,2,3]] == one(R)
    end

    @testset "Nilpotency for all basis pairs (Q=0)" begin
        for i in 1:3, j in 1:3
            i == j && continue
            # (eᵢ*eⱼ)*(eᵢ*eⱼ) should be zero (degree 4 in 3-dim space)
            # Actually, (eᵢeⱼ)² = eᵢeⱼeᵢeⱼ = −eᵢeᵢeⱼeⱼ = 0 for Exterior
            v = ec0[i] * ec0[j]
            @test iszero(v * v)
        end
    end

end  # Q=0 mandatory regression

# ─────────────────────────────────────────────────────────────────────────────
@testset "Cl(3,0) algebra structure" begin

    @testset "Associativity: all triples of grade-1 elements" begin
        for i in 1:3, j in 1:3, k in 1:3
            @test (ec[i] * ec[j]) * ec[k] == ec[i] * (ec[j] * ec[k])
        end
    end

    @testset "Left and right distributivity" begin
        for i in 1:3, j in 1:3, k in 1:3
            @test ec[i] * (ec[j] + ec[k]) == ec[i]*ec[j] + ec[i]*ec[k]
            @test (ec[i] + ec[j]) * ec[k] == ec[i]*ec[k] + ec[j]*ec[k]
        end
    end

    @testset "Scalar multiplicative identity" begin
        for i in 1:3
            @test cl1 * ec[i] == ec[i]
            @test ec[i] * cl1 == ec[i]
        end
    end

    @testset "Additive identity" begin
        z = clifford_zero(gE_3, R)
        for i in 1:3
            @test z + ec[i] == ec[i]
            @test ec[i] + z == ec[i]
            @test iszero(z)
        end
    end

    @testset "Scalar multiplication" begin
        for i in 1:3
            @test R(3) * ec[i] == ec[i] + ec[i] + ec[i]
            @test (R(0) * ec[i]) == clifford_zero(gE_3, R)
        end
    end

    @testset "Pseudoscalar e₁e₂e₃" begin
        # In Cl(3,0): I = e₁e₂e₃ is the pseudoscalar.
        I  = ec[1] * ec[2] * ec[3]
        # I² = −1  (Cl(3,0) pseudoscalar squares to −1)
        I2 = I * I
        @test I2 == clifford_scalar(gE_3, R(-1))
        # I anticommutes with grade-1 elements (since n=3 is odd... wait:
        # For Cl(3,0), Ie₁ = e₁e₂e₃e₁ = −e₁e₁e₂e₃ = −e₂e₃.
        # Actually let's just check I*I = -1.
        @test grade(I) == 3
        @test I.terms[[1,2,3]] == one(R)
    end

    @testset "Grade structure: each product of distinct grade-1 is grade 2" begin
        for i in 1:3, j in 1:3
            i == j && continue
            p = ec[i] * ec[j]
            @test grade(p) == 2
        end
    end

    @testset "Dimension 2ⁿ = 8 for Cl(3,0)" begin
        total = sum(length(clifford_homogeneous_basis(gE_3, R, k)) for k in 0:3)
        @test total == 8
    end

    @testset "Homogeneous basis dimensions: 1, 3, 3, 1" begin
        @test length(clifford_homogeneous_basis(gE_3, R, 0)) == 1
        @test length(clifford_homogeneous_basis(gE_3, R, 1)) == 3
        @test length(clifford_homogeneous_basis(gE_3, R, 2)) == 3
        @test length(clifford_homogeneous_basis(gE_3, R, 3)) == 1
        @test length(clifford_homogeneous_basis(gE_3, R, 4)) == 0
    end

    @testset "homogeneous_component" begin
        mixed = ec[1] + ec[1] * ec[2]    # grade 1 + grade 2
        @test grade(homogeneous_component(mixed, 1)) == 1
        @test grade(homogeneous_component(mixed, 2)) == 2
        @test iszero(homogeneous_component(mixed, 0))
    end

end  # Cl(3,0) structure

# ─────────────────────────────────────────────────────────────────────────────
@testset "Dimension 2ⁿ for all Cl(n) up to n=5" begin
    for n in 0:5
        V = VectorSpace(n)
        g = signature_metric(V, R, n, 0, 0)
        total = sum(
            length(clifford_homogeneous_basis(g, R, k)) for k in 0:n
        )
        @test total == 2^n
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "General (non-diagonal) metric" begin
    # Hyperbolic plane: g = [[0, 1//2],[1//2, 0]]
    # So e₁² = 0, e₂² = 0, g(e₁,e₂) = 1//2
    # Clifford relation: e₁e₂ + e₂e₁ = 2·(1//2)·1 = 1

    g_hyp = Metric{R}(V2, Matrix{R}([R(0) R(1//2); R(1//2) R(0)]))
    h1    = clifford_basis_vector(g_hyp, R, 1)
    h2    = clifford_basis_vector(g_hyp, R, 2)
    h_one = clifford_one(g_hyp, R)

    @testset "e₁² = e₂² = 0 (null directions)" begin
        @test iszero(h1 * h1)
        @test iszero(h2 * h2)
    end

    @testset "e₁e₂ + e₂e₁ = 2g(e₁,e₂)·1 = 1" begin
        @test h1 * h2 + h2 * h1 == h_one
    end

    @testset "e₂e₁ = 1 − e₁e₂  (explicit rewrite)" begin
        # e₂e₁ = 2g(e₂,e₁) - e₁e₂ = 1 - e₁e₂
        lhs = h2 * h1
        rhs = h_one + (-(h1 * h2))
        @test lhs == rhs
    end

    @testset "Associativity holds for general metric" begin
        @test (h1 * h2) * h1 == h1 * (h2 * h1)
        @test (h2 * h1) * h2 == h2 * (h1 * h2)
    end

end  # General metric

# ─────────────────────────────────────────────────────────────────────────────
@testset "Ring genericity (Float64)" begin
    Vf = VectorSpace(2)
    gf = signature_metric(Vf, Float64, 2, 0, 0)
    ef = [clifford_basis_vector(gf, Float64, i) for i in 1:2]
    # e₁² = 1.0 exactly
    @test ef[1] * ef[1] == clifford_one(gf, Float64)
    # e₁e₂ canonical form has coefficient 1.0
    p12 = ef[1] * ef[2]
    p21 = ef[2] * ef[1]
    @test p12.terms[[1,2]] == 1.0
    @test p21.terms[[1,2]] == -1.0
    # e₁e₂ + e₂e₁ = 0 (off-diagonal g = 0 for Euclidean diagonal metric)
    @test iszero(p12 + p21)
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "Incompatible metrics raise ArgumentError" begin
    g1 = signature_metric(V2, R, 2, 0, 0)
    g2 = signature_metric(V2, R, 1, 1, 0)
    e1 = clifford_basis_vector(g1, R, 1)
    e2 = clifford_basis_vector(g2, R, 1)
    @test_throws ArgumentError e1 * e2
    @test_throws ArgumentError e1 + e2
end


@testset "Integer literal scalar multiplication" begin
    R  = Rational{BigInt}
    V  = VectorSpace(3)
    g  = signature_metric(V, R, 3, 0, 0)
    e1 = clifford_basis_vector(g, R, 1)
    @test 2 * e1 == e1 + e1
    @test e1 * 2 == e1 + e1
    @test 3 * e1 == e1 + e1 + e1
    @test isequal((2 * e1).terms[[1]], R(2))
end
