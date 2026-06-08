# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Inter-algebra maps tests
#
# Test strategy:
#   1. _all_position_permutations and _perm_sign helpers
#   2. Projections: T(V) -> quotient  (homomorphism property + specific values)
#   3. Sections: antisymmetrize, symmetrize  (specific values + round-trips)
#   4. Ring guard: non-rational R raises ArgumentError
#   5. Symbol maps: ext_to_cl, cl_to_ext  (round-trips + Q=0 algebra iso)
#   6. Tower coherence: chained maps and cross-algebra consistency
# ─────────────────────────────────────────────────────────────────────────────

R  = Rational{BigInt}
V3 = VectorSpace(3)
V2 = VectorSpace(2)

# Free tensor basis
e  = [basis_vector(V3, R, i) for i in 1:3]

# Exterior basis
ee = [ext_basis_vector(V3, R, i) for i in 1:3]

# Symmetric basis
es = [sym_basis_vector(V3, R, i) for i in 1:3]

# Clifford (Euclidean) basis
gE = signature_metric(V3, R, 3, 0, 0)
ec = [clifford_basis_vector(gE, R, i) for i in 1:3]

# Zero metric (Clifford = Exterior as algebras)
g0 = zero_metric(V3, R)
ec0 = [clifford_basis_vector(g0, R, i) for i in 1:3]

# ─────────────────────────────────────────────────────────────────────────────
@testset "Internal helpers" begin

    @testset "_all_position_permutations" begin
        # n=0: one permutation (the empty one)
        @test Tensorsmith._all_position_permutations(0) == [Int[]]

        # n=1
        @test Tensorsmith._all_position_permutations(1) == [[1]]

        # n=2: exactly [[2,1],[1,2]] or [[1,2],[2,1]] (order may vary)
        perms2 = Tensorsmith._all_position_permutations(2)
        @test length(perms2) == 2
        @test sort(perms2) == [[1,2],[2,1]]

        # n=3: 6 permutations of {1,2,3}
        perms3 = Tensorsmith._all_position_permutations(3)
        @test length(perms3) == 6
        @test length(unique(perms3)) == 6
        expected = sort([[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]])
        @test sort(perms3) == expected

        # n=4: 24 permutations
        @test length(Tensorsmith._all_position_permutations(4)) == 24
    end

    @testset "_perm_sign" begin
        @test Tensorsmith._perm_sign(Int[])   ==  1   # empty: even
        @test Tensorsmith._perm_sign([1])     ==  1   # identity
        @test Tensorsmith._perm_sign([1,2])   ==  1   # identity
        @test Tensorsmith._perm_sign([2,1])   == -1   # one transposition
        @test Tensorsmith._perm_sign([1,2,3]) ==  1
        @test Tensorsmith._perm_sign([2,1,3]) == -1   # one inversion
        @test Tensorsmith._perm_sign([1,3,2]) == -1
        @test Tensorsmith._perm_sign([2,3,1]) ==  1   # two inversions
        @test Tensorsmith._perm_sign([3,1,2]) ==  1
        @test Tensorsmith._perm_sign([3,2,1]) == -1   # three inversions
    end

end  # Internal helpers

# ─────────────────────────────────────────────────────────────────────────────
@testset "Projections T(V) -> quotient" begin

    @testset "project_ext: grade-1 basis vectors unchanged" begin
        for i in 1:3
            @test project_ext(e[i]) == ee[i]
        end
    end

    @testset "project_sym: grade-1 basis vectors unchanged" begin
        for i in 1:3
            @test project_sym(e[i]) == es[i]
        end
    end

    @testset "project_cl: grade-1 basis vectors unchanged" begin
        for i in 1:3
            @test project_cl(e[i], gE) == ec[i]
        end
    end

    @testset "project_ext kills e_i x e_i" begin
        for i in 1:3
            @test iszero(project_ext(e[i] * e[i]))
        end
    end

    @testset "project_sym keeps e_i x e_i nonzero" begin
        for i in 1:3
            @test !iszero(project_sym(e[i] * e[i]))
        end
    end

    @testset "project_ext: e_j x e_i -> -(e_i ^ e_j)" begin
        for i in 1:3, j in 1:3
            i == j && continue
            lhs = project_ext(e[j] * e[i])
            rhs = -(ee[i] ∧ ee[j])
            @test lhs == rhs
        end
    end

    @testset "project_sym: e_j x e_i -> e_i * e_j (commutative)" begin
        for i in 1:3, j in 1:3
            @test project_sym(e[i] * e[j]) == project_sym(e[j] * e[i])
        end
    end

    @testset "project_ext is a ring hom: pi(a*b) == pi(a) ^ pi(b)" begin
        for i in 1:3, j in 1:3
            lhs = project_ext(e[i] * e[j])
            rhs = project_ext(e[i]) ∧ project_ext(e[j])
            @test lhs == rhs
        end
        # Also on sums
        a = e[1] + e[2]
        b = e[2] + e[3]
        @test project_ext(a * b) == project_ext(a) ∧ project_ext(b)
    end

    @testset "project_sym is a ring hom: pi(a*b) == pi(a) * pi(b)" begin
        for i in 1:3, j in 1:3
            lhs = project_sym(e[i] * e[j])
            rhs = project_sym(e[i]) * project_sym(e[j])
            @test lhs == rhs
        end
        a = e[1] + e[2]
        b = e[2] + e[3]
        @test project_sym(a * b) == project_sym(a) * project_sym(b)
    end

    @testset "project_cl is a ring hom: pi(a*b) == pi(a) * pi(b)" begin
        for i in 1:3, j in 1:3
            lhs = project_cl(e[i] * e[j], gE)
            rhs = project_cl(e[i], gE) * project_cl(e[j], gE)
            @test lhs == rhs
        end
    end

    @testset "project_cl with zero metric matches project_ext (Q=0)" begin
        for i in 1:3, j in 1:3
            cl  = project_cl(e[i] * e[j], g0)
            ext = project_ext(e[i] * e[j])
            # Both should have the same canonical terms
            @test cl_to_ext(cl) == ext
        end
    end

    @testset "project_cl space mismatch raises ArgumentError" begin
        V2 = VectorSpace(2)
        e2 = basis_vector(V2, R, 1)
        @test_throws ArgumentError project_cl(e2, gE)
    end

    @testset "Scalar (grade-0) passes through all projections" begin
        s = scalar_element(V3, R(7))
        @test project_ext(s).terms[Int[]] == R(7)
        @test project_sym(s).terms[Int[]] == R(7)
        @test project_cl(s, gE).terms[Int[]] == R(7)
    end

end  # Projections

# ─────────────────────────────────────────────────────────────────────────────
@testset "antisymmetrize -- section Lambda(V) -> T(V)" begin

    @testset "Grade-1: identity" begin
        for i in 1:3
            t = antisymmetrize(ee[i])
            @test t == e[i]
        end
    end

    @testset "Grade-0 scalar: identity" begin
        s = alg_scalar(V3, R(5), ExteriorAlgebra)
        t = antisymmetrize(s)
        @test t.terms[Int[]] == R(5)
    end

    @testset "Grade-2: specific coefficients" begin
        # antisym(e1^e2) = (1/2)(e1 x e2 - e2 x e1)
        t = antisymmetrize(ee[1] ∧ ee[2])
        @test t.terms[[1,2]] ==  R(1//2)
        @test t.terms[[2,1]] == R(-1//2)
        @test length(t.terms) == 2

        # antisym(e2^e3)
        t = antisymmetrize(ee[2] ∧ ee[3])
        @test t.terms[[2,3]] ==  R(1//2)
        @test t.terms[[3,2]] == R(-1//2)
    end

    @testset "Grade-3: 6 terms, each +/-1/6" begin
        t = antisymmetrize(ee[1] ∧ ee[2] ∧ ee[3])
        @test length(t.terms) == 6
        # All coefficients have absolute value 1/6
        for (_, c) in t.terms
            @test abs(c) == R(1//6)
        end
        # The even permutations get +1/6
        @test t.terms[[1,2,3]] == R( 1//6)
        @test t.terms[[2,3,1]] == R( 1//6)
        @test t.terms[[3,1,2]] == R( 1//6)
        # The odd permutations get -1/6
        @test t.terms[[2,1,3]] == R(-1//6)
        @test t.terms[[1,3,2]] == R(-1//6)
        @test t.terms[[3,2,1]] == R(-1//6)
    end

    @testset "Round-trip: project_ext(antisymmetrize(t)) == t" begin
        # Grade-1
        for i in 1:3
            @test project_ext(antisymmetrize(ee[i])) == ee[i]
        end
        # Grade-2: all pairs
        for i in 1:3, j in 1:3
            i == j && continue
            blade = ext_basis_vector(V3, R, i) ∧ ext_basis_vector(V3, R, j)
            # Note: if i > j, blade may be zero or negative; use canonical basis
        end
        # Canonical grade-2 basis
        for k in 0:2
            for b in ext_homogeneous_basis(V3, R, k)
                @test project_ext(antisymmetrize(b)) == b
            end
        end
        # Grade-3 (top form)
        top = ee[1] ∧ ee[2] ∧ ee[3]
        @test project_ext(antisymmetrize(top)) == top
    end

    @testset "Result lives in T(V): no normalization constraints" begin
        # antisymmetrize result is a FreeTensor, not an AlgebraTensor
        t = antisymmetrize(ee[1] ∧ ee[2])
        @test t isa FreeTensor{R}
        # T(V) keeps both [1,2] and [2,1] as distinct basis elements
        @test haskey(t.terms, [1,2])
        @test haskey(t.terms, [2,1])
    end

    @testset "Non-rational R raises ArgumentError" begin
        ee_f = [ext_basis_vector(V3, Float64, i) for i in 1:3]
        @test_throws ArgumentError antisymmetrize(ee_f[1] ∧ ee_f[2])
    end

end  # antisymmetrize

# ─────────────────────────────────────────────────────────────────────────────
@testset "symmetrize -- section Sym(V) -> T(V)" begin

    @testset "Grade-1: identity" begin
        for i in 1:3
            @test symmetrize(es[i]) == e[i]
        end
    end

    @testset "Grade-0 scalar: identity" begin
        s = alg_scalar(V3, R(3), SymmetricAlgebra)
        t = symmetrize(s)
        @test t.terms[Int[]] == R(3)
    end

    @testset "Grade-2 distinct: specific coefficients" begin
        # sym(e1*e2) = (1/2)(e1 x e2 + e2 x e1)
        t = symmetrize(es[1] * es[2])
        @test t.terms[[1,2]] == R(1//2)
        @test t.terms[[2,1]] == R(1//2)
        @test length(t.terms) == 2
    end

    @testset "Grade-2 repeated: e1*e1 -> e1 x e1" begin
        # sym(e1^2) = (1/2)(e1 x e1 + e1 x e1) = e1 x e1
        t = symmetrize(es[1] * es[1])
        @test t.terms[[1,1]] == R(1)
        @test length(t.terms) == 1
    end

    @testset "Grade-3: 6 terms, all +1/6" begin
        t = symmetrize(es[1] * es[2] * es[3])
        @test length(t.terms) == 6
        for (_, c) in t.terms
            @test c == R(1//6)
        end
    end

    @testset "Round-trip: project_sym(symmetrize(t)) == t" begin
        for k in 0:3
            for b in sym_homogeneous_basis(V3, R, k)
                @test project_sym(symmetrize(b)) == b
            end
        end
    end

    @testset "Result lives in T(V): FreeTensor type" begin
        t = symmetrize(es[1] * es[2])
        @test t isa FreeTensor{R}
        @test haskey(t.terms, [1,2])
        @test haskey(t.terms, [2,1])
    end

    @testset "Non-rational R raises ArgumentError" begin
        es_f = [sym_basis_vector(V3, Float64, i) for i in 1:3]
        @test_throws ArgumentError symmetrize(es_f[1] * es_f[2])
    end

end  # symmetrize

# ─────────────────────────────────────────────────────────────────────────────
@testset "Symbol maps Lambda(V) <--> Cl(V,g)" begin

    @testset "cl_to_ext: basis vectors are preserved" begin
        for i in 1:3
            @test cl_to_ext(ec[i]) == ee[i]
        end
        @test cl_to_ext(clifford_one(gE, R)) ==
              alg_scalar(V3, one(R), ExteriorAlgebra)
    end

    @testset "ext_to_cl: basis vectors are preserved" begin
        for i in 1:3
            @test ext_to_cl(ee[i], gE) == ec[i]
        end
        s = alg_scalar(V3, one(R), ExteriorAlgebra)
        @test ext_to_cl(s, gE) == clifford_one(gE, R)
    end

    @testset "cl_to_ext(ext_to_cl(t)) == t for all Ext basis elements" begin
        for k in 0:3
            for b in ext_homogeneous_basis(V3, R, k)
                @test cl_to_ext(ext_to_cl(b, gE)) == b
            end
        end
    end

    @testset "ext_to_cl(cl_to_ext(t)) == t for all Cl basis elements" begin
        for k in 0:3
            for b in clifford_homogeneous_basis(gE, R, k)
                @test ext_to_cl(cl_to_ext(b), gE) == b
            end
        end
    end

    @testset "Symbol map preserves coefficients exactly" begin
        # Mixed element: 3*e1 - 2*(e1^e2) + 5*(e1^e2^e3)
        t = R(3)*ee[1] + R(-2)*(ee[1]∧ee[2]) + R(5)*(ee[1]∧ee[2]∧ee[3])
        rt = cl_to_ext(ext_to_cl(t, gE))
        @test rt == t
    end

    @testset "Q=0: ext_to_cl is an algebra isomorphism" begin
        # When metric is zero, Cl(V,0) == Lambda(V) as algebras.
        # So ext_to_cl(a ^ b, g0) == ext_to_cl(a, g0) * ext_to_cl(b, g0).
        for i in 1:3, j in 1:3
            lhs = ext_to_cl(ee[i] ∧ ee[j], g0)
            rhs = ext_to_cl(ee[i], g0) * ext_to_cl(ee[j], g0)
            @test lhs == rhs
        end
        # And the triple product
        lhs = ext_to_cl(ee[1] ∧ ee[2] ∧ ee[3], g0)
        rhs = ext_to_cl(ee[1], g0) * ext_to_cl(ee[2], g0) * ext_to_cl(ee[3], g0)
        @test lhs == rhs
    end

    @testset "Q!=0: ext_to_cl is NOT an algebra iso in general" begin
        # For Euclidean metric, e1*e1 = 1 in Cl, but e1^e1 = 0 in Ext.
        # So ext_to_cl(e1^e1) = ext_to_cl(0) = 0, but
        # ext_to_cl(e1)*ext_to_cl(e1) = ec[1]*ec[1] = scalar(1) != 0.
        lhs = ext_to_cl(ee[1] ∧ ee[1], gE)           # = 0
        rhs = ext_to_cl(ee[1], gE) * ext_to_cl(ee[1], gE)  # = scalar 1
        @test iszero(lhs)
        @test !iszero(rhs)
        @test lhs != rhs
    end

    @testset "Space mismatch raises ArgumentError" begin
        V2 = VectorSpace(2)
        ee2 = ext_basis_vector(V2, R, 1)
        @test_throws ArgumentError ext_to_cl(ee2, gE)
    end

end  # Symbol maps

# ─────────────────────────────────────────────────────────────────────────────
@testset "Tower coherence" begin

    @testset "antisymmetrize image is antisymmetric in T(V)" begin
        # For a grade-2 blade e1^e2, antisymmetrize gives (e1xe2 - e2xe1)/2.
        # Swapping the two tensor slots negates the result.
        # We verify: the [1,2] coefficient equals minus the [2,1] coefficient.
        t = antisymmetrize(ee[1] ∧ ee[2])
        @test t.terms[[1,2]] == -t.terms[[2,1]]

        t3 = antisymmetrize(ee[1] ∧ ee[2] ∧ ee[3])
        # Any swap of adjacent slots negates: coef of [1,2,3] = -coef of [2,1,3]
        @test t3.terms[[1,2,3]] == -t3.terms[[2,1,3]]
        @test t3.terms[[1,2,3]] == -t3.terms[[1,3,2]]
    end

    @testset "symmetrize image is symmetric in T(V)" begin
        # For e1*e2, symmetrize gives (e1xe2 + e2xe1)/2.
        # Swapping slots preserves the result.
        t = symmetrize(es[1] * es[2])
        @test t.terms[[1,2]] == t.terms[[2,1]]
    end

    @testset "project_cl o antisymmetrize == ext_to_cl for Q=0" begin
        # With Q=0, Cl(V,0) == Lambda(V), so projecting an antisymmetrized
        # element into Cl should match ext_to_cl directly.
        for k in 0:3
            for b in ext_homogeneous_basis(V3, R, k)
                lhs = project_cl(antisymmetrize(b), g0)
                rhs = ext_to_cl(b, g0)
                @test lhs == rhs
            end
        end
    end

    @testset "Symmetrize preserves grade" begin
        for k in 0:3
            for b in sym_homogeneous_basis(V3, R, k)
                t = symmetrize(b)
                # Every term in the T(V) result has length k
                for (idx, _) in t.terms
                    @test length(idx) == k
                end
            end
        end
    end

    @testset "Antisymmetrize preserves grade" begin
        for k in 0:3
            for b in ext_homogeneous_basis(V3, R, k)
                t = antisymmetrize(b)
                for (idx, _) in t.terms
                    @test length(idx) == k
                end
            end
        end
    end

    @testset "project_sym o symmetrize == identity (all Sym grade 0-3)" begin
        for k in 0:3
            for b in sym_homogeneous_basis(V3, R, k)
                @test project_sym(symmetrize(b)) == b
            end
        end
    end

    @testset "project_ext o antisymmetrize == identity (all Ext grade 0-3)" begin
        for k in 0:3
            for b in ext_homogeneous_basis(V3, R, k)
                @test project_ext(antisymmetrize(b)) == b
            end
        end
    end

end  # Tower coherence
