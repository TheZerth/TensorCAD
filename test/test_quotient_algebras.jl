# Fixtures at file scope (outside @testset to avoid local-const issues)
R   = Rational{BigInt}
V3  = VectorSpace(3)
V4  = VectorSpace(4)

es  = [sym_basis_vector(V3, R, i) for i in 1:3]   # Symmetric basis
ee  = [ext_basis_vector(V3, R, i) for i in 1:3]   # Exterior basis

# ─────────────────────────────────────────────────────────────────────────────
@testset "SymmetricAlgebra — Phase 2" begin

    @testset "Commutativity: eᵢ·eⱼ = eⱼ·eᵢ" begin
        for i in 1:3, j in 1:3
            @test es[i] * es[j] == es[j] * es[i]
        end
        # Three-way commutativity follows from pairwise
        @test es[1] * es[2] * es[3] == es[3] * es[1] * es[2]
        @test es[1] * es[2] * es[3] == es[2] * es[3] * es[1]
    end

    @testset "Sym is NOT Exterior — eᵢ·eᵢ ≠ 0" begin
        # In Sym, eᵢ² represents the polynomial monomial eᵢ²; it is nonzero.
        for i in 1:3
            @test !iszero(es[i] * es[i])
        end
        # The canonical index for eᵢ·eᵢ is [i,i] (non-decreasing multiset)
        @test (es[1] * es[1]).terms[[1,1]] == one(R)
    end

    @testset "Associativity" begin
        for i in 1:3, j in 1:3, k in 1:3
            @test (es[i] * es[j]) * es[k] == es[i] * (es[j] * es[k])
        end
        a = es[1] + es[2]
        b = es[2] + es[3]
        c = es[1] + es[3]
        @test (a * b) * c == a * (b * c)
    end

    @testset "Left and right distributivity" begin
        for i in 1:3, j in 1:3, k in 1:3
            @test es[i] * (es[j] + es[k]) == es[i] * es[j] + es[i] * es[k]
            @test (es[i] + es[j]) * es[k] == es[i] * es[k] + es[j] * es[k]
        end
    end

    @testset "Canonical form: indices are non-decreasing" begin
        # e₂·e₁ should store as [1,2] not [2,1]
        t = es[2] * es[1]
        @test haskey(t.terms, [1,2])
        @test !haskey(t.terms, [2,1])
        # e₃·e₁·e₂ → canonical [1,2,3]
        t3 = es[3] * es[1] * es[2]
        @test haskey(t3.terms, [1,2,3])
    end

    @testset "Sym(V) is the polynomial ring R[e₁,…,eₙ]" begin
        # Scalar multiplication
        @test (R(2) * es[1]) * (R(3) * es[2]) == R(6) * (es[1] * es[2])
        # e₁² + 2·e₁·e₂ + e₂² = (e₁ + e₂)²
        lhs = (es[1] + es[2]) * (es[1] + es[2])
        rhs = es[1]*es[1] + R(2)*(es[1]*es[2]) + es[2]*es[2]
        @test lhs == rhs
    end

    @testset "Grade-k dimension = C(n+k−1, k)" begin
        for n in 0:4
            V = VectorSpace(n)
            for k in 0:5
                expected = sym_grade_dim(n, k)
                indices  = all_sym_grade_k_indices(V, k)
                @test length(indices) == expected
                @test length(unique(indices)) == expected
                @test all(length(idx) == k for idx in indices)
                # All indices are non-decreasing
                @test all(issorted(idx) for idx in indices)
            end
        end
    end

    @testset "sym_grade_dim matches binomial formula" begin
        for n in 0:5, k in 0:5
            @test sym_grade_dim(n, k) == (n == 0 && k > 0 ? 0 : binomial(n + k - 1, k))
        end
        # Spot checks
        @test sym_grade_dim(3, 0) == 1    # just the scalar
        @test sym_grade_dim(3, 1) == 3    # e₁, e₂, e₃
        @test sym_grade_dim(3, 2) == 6    # e₁², e₁e₂, e₁e₃, e₂², e₂e₃, e₃²
        @test sym_grade_dim(3, 3) == 10
        for k in 0:5
            @test sym_grade_dim(1, k) == 1   # 1 variable: exactly one monomial per degree
        end
    end

    @testset "Homogeneous basis for Sym" begin
        for k in 0:3
            basis = sym_homogeneous_basis(V3, R, k)
            @test length(basis) == sym_grade_dim(3, k)
            @test all(grade(b) == k for b in basis)
            # Each basis element stores exactly one term
            @test all(length(b.terms) == 1 for b in basis)
            # All stored indices are non-decreasing
            @test all(issorted(only(keys(b.terms))) for b in basis)
        end
    end

end  # SymmetricAlgebra

# ─────────────────────────────────────────────────────────────────────────────
@testset "ExteriorAlgebra — Phase 2" begin

    @testset "Nilpotency: eᵢ∧eᵢ = 0" begin
        for i in 1:3
            @test iszero(ee[i] ∧ ee[i])
        end
        # Higher-grade: (e₁∧e₂)∧e₁ = 0  (contains e₁ twice)
        @test iszero((ee[1] ∧ ee[2]) ∧ ee[1])
        @test iszero((ee[2] ∧ ee[3]) ∧ ee[2])
    end

    @testset "Antisymmetry: eᵢ∧eⱼ = −eⱼ∧eᵢ" begin
        for i in 1:3, j in 1:3
            @test ee[i] ∧ ee[j] == -(ee[j] ∧ ee[i])
        end
    end

    @testset "Associativity" begin
        for i in 1:3, j in 1:3, k in 1:3
            @test (ee[i] ∧ ee[j]) ∧ ee[k] == ee[i] ∧ (ee[j] ∧ ee[k])
        end
        a = ee[1] + ee[2]
        b = ee[2] + ee[3]
        c = ee[1] + ee[3]
        @test (a ∧ b) ∧ c == a ∧ (b ∧ c)
    end

    @testset "Left and right distributivity" begin
        for i in 1:3, j in 1:3, k in 1:3
            @test ee[i] ∧ (ee[j] + ee[k]) == ee[i]∧ee[j] + ee[i]∧ee[k]
            @test (ee[i] + ee[j]) ∧ ee[k] == ee[i]∧ee[k] + ee[j]∧ee[k]
        end
    end

    @testset "Sign correctness" begin
        # One swap: e₂∧e₁ = −e₁∧e₂
        @test ee[2] ∧ ee[1] == -(ee[1] ∧ ee[2])
        @test (ee[2] ∧ ee[1]).terms[[1,2]] == R(-1)

        # Two swaps: e₃∧e₁∧e₂ = +e₁∧e₂∧e₃
        # [3,1,2] → bubble sort: swap(3,1)→[1,3,2] flip, swap(3,2)→[1,2,3] flip²=no flip
        # Wait let me re-trace: sort [3,1,2]:
        #   i=1: j=1: 3>1? yes → [1,3,2] flip=true
        #         j=2: 3>2? yes → [1,2,3] flip=false
        #   i=2: j=1: 1>2? no
        # sign_flip=false → same sign → e₃∧e₁∧e₂ = +e₁∧e₂∧e₃
        @test ee[3] ∧ ee[1] ∧ ee[2] == ee[1] ∧ ee[2] ∧ ee[3]

        # Three swaps: e₃∧e₂∧e₁ = −e₁∧e₂∧e₃
        # [3,2,1] → sort: swap(3,2)→[2,3,1] flip, swap(3,1)→[2,1,3] no flip, swap(2,1)→[1,2,3] flip
        # sign_flip=true → negate → e₃∧e₂∧e₁ = −e₁∧e₂∧e₃
        @test ee[3] ∧ ee[2] ∧ ee[1] == -(ee[1] ∧ ee[2] ∧ ee[3])

        # Stored index is canonical (strictly increasing)
        t = ee[2] ∧ ee[1]
        @test haskey(t.terms, [1,2])
        @test t.terms[[1,2]] == R(-1)
    end

    @testset "Top form (pseudoscalar)" begin
        # For n=3, the top element e₁∧e₂∧e₃ is the unique grade-3 basis element
        top = ee[1] ∧ ee[2] ∧ ee[3]
        @test !iszero(top)
        @test grade(top) == 3
        @test top.terms[[1,2,3]] == one(R)
        # Any grade-4 product in a 3-dim space must be zero
        @test iszero(top ∧ ee[1])
        @test iszero(top ∧ ee[2])
    end

    @testset "Grade-k dimension = C(n, k)" begin
        for n in 0:5
            V = VectorSpace(n)
            for k in 0:6
                expected = ext_grade_dim(n, k)
                indices  = all_ext_grade_k_indices(V, k)
                @test length(indices) == expected
                @test length(unique(indices)) == expected
                @test all(length(idx) == k for idx in indices)
                # All indices are strictly increasing
                @test all(issorted(idx; lt = <=) for idx in indices)
            end
        end
    end

    @testset "ext_grade_dim matches binomial formula" begin
        for n in 0:5, k in 0:6
            @test ext_grade_dim(n, k) == binomial(n, k)
        end
        # Spot checks
        @test ext_grade_dim(3, 0) == 1
        @test ext_grade_dim(3, 1) == 3
        @test ext_grade_dim(3, 2) == 3    # e₁∧e₂, e₁∧e₃, e₂∧e₃
        @test ext_grade_dim(3, 3) == 1    # e₁∧e₂∧e₃ only
        @test ext_grade_dim(3, 4) == 0    # above top: zero
    end

    @testset "Total dimension = 2ⁿ" begin
        for n in 0:5
            V = VectorSpace(n)
            @test sum(ext_grade_dim(n, k) for k in 0:n) == 2^n
        end
    end

    @testset "Homogeneous basis for Ext" begin
        for k in 0:3
            basis = ext_homogeneous_basis(V3, R, k)
            @test length(basis) == ext_grade_dim(3, k)
            @test all(grade(b) == k for b in basis)
            # Each element has exactly one term with coefficient 1
            @test all(length(b.terms) == 1 for b in basis)
            @test all(only(values(b.terms)) == one(R) for b in basis)
            # All stored indices are strictly increasing
            @test all(issorted(only(keys(b.terms)); lt = <) for b in basis)
        end
    end

    @testset "Scalar ring genericity" begin
        V = VectorSpace(2)
        ef = [ext_basis_vector(V, Float64, i) for i in 1:2]
        @test iszero(ef[1] ∧ ef[1])
        @test ef[1] ∧ ef[2] == -(ef[2] ∧ ef[1])
        @test grade(ef[1] ∧ ef[2]) == 2
    end

end  # ExteriorAlgebra

# ─────────────────────────────────────────────────────────────────────────────
@testset "Cross-algebra isolation" begin

    @testset "Sym and Ext give different results for the same input" begin
        # In FreeTensor: e₁⊗e₁ ≠ 0 (no relation)
        e_free = [basis_vector(V3, R, i) for i in 1:3]
        @test !iszero(e_free[1] * e_free[1])

        # In Exterior: e₁∧e₁ = 0
        @test iszero(ee[1] ∧ ee[1])

        # In Symmetric: e₁·e₁ ≠ 0  (it's the monomial e₁²)
        @test !iszero(es[1] * es[1])

        # Sym and Ext agree on grade-1 basis vectors (they're the same)
        @test grade(es[1]) == 1
        @test grade(ee[1]) == 1

        # But disagree at grade 2
        @test !iszero(es[1] * es[1])    # Sym: e₁² ≠ 0
        @test  iszero(ee[1] ∧ ee[1])    # Ext: e₁∧e₁ = 0

        # And on order
        @test  (es[1] * es[2]) == (es[2] * es[1])    # Sym: commutative
        @test !(ee[1] ∧ ee[2] == ee[2] ∧ ee[1])      # Ext: anticommutative
    end

    @testset "AlgebraTensor and FreeTensor are distinct types" begin
        e_free = basis_vector(V3, R, 1)
        e_sym  = sym_basis_vector(V3, R, 1)
        e_ext  = ext_basis_vector(V3, R, 1)
        @test e_free isa FreeTensor{R}
        @test e_sym  isa AlgebraTensor{SymmetricAlgebra, R}
        @test e_ext  isa AlgebraTensor{ExteriorAlgebra, R}
    end

end  # Cross-algebra isolation

# ─────────────────────────────────────────────────────────────────────────────
@testset "normalize function directly" begin
    # FreeAlgebra: identity
    @test Tensorsmith.normalize(FreeAlgebra(), [2,1], R(3)) == ([2,1], R(3))

    # SymmetricAlgebra: sort, no sign
    @test Tensorsmith.normalize(SymmetricAlgebra(), [2,1], R(3)) == ([1,2], R(3))
    @test Tensorsmith.normalize(SymmetricAlgebra(), [3,1,2], R(1)) == ([1,2,3], R(1))
    @test Tensorsmith.normalize(SymmetricAlgebra(), Int[], R(5)) == (Int[], R(5))

    # ExteriorAlgebra: sort with sign, zero on repeat
    @test Tensorsmith.normalize(ExteriorAlgebra(), [1,1], R(1))   == (Int[], R(0))
    @test Tensorsmith.normalize(ExteriorAlgebra(), [2,1], R(1))   == ([1,2], R(-1))
    @test Tensorsmith.normalize(ExteriorAlgebra(), [1,2], R(1))   == ([1,2], R(1))
    @test Tensorsmith.normalize(ExteriorAlgebra(), [3,2,1], R(1)) == ([1,2,3], R(-1))
    @test Tensorsmith.normalize(ExteriorAlgebra(), [3,1,2], R(1)) == ([1,2,3], R(1))
    @test Tensorsmith.normalize(ExteriorAlgebra(), Int[], R(7))   == (Int[], R(7))
end
