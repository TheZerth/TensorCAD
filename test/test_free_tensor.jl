# ── Shared fixtures (module scope — safe to reference from any inner @testset) ──
# `const` inside @testset is local scope in Julia and would warn/error; define here.

R  = Rational{BigInt}
V3 = VectorSpace(3)
e  = [basis_vector(V3, R, i) for i in 1:3]

@testset "FreeTensor — Phase 1" begin

    # ── Construction ─────────────────────────────────────────────────────────
    @testset "Construction" begin
        # Basis vectors
        @test !iszero(e[1])
        @test grade(e[1]) == 1
        @test e[1].terms[[1]] == one(R)

        # basis_element: repetitions are allowed and stored — this is T(V), not Exterior
        t = basis_element(V3, R, [1, 1, 2])   # e₁⊗e₁⊗e₂
        @test grade(t) == 3
        @test t.terms[[1,1,2]] == one(R)

        # Scalar embedding
        s = scalar_element(V3, R(3))
        @test grade(s) == 0
        @test s.terms[Int[]] == R(3)

        # scalar_element of zero → zero element
        sz = scalar_element(V3, zero(R))
        @test iszero(sz)

        # zero / one
        z = zero(FreeTensor{R}, V3)
        @test iszero(z)
        @test_throws ArgumentError grade(z)

        o = one(FreeTensor{R}, V3)
        @test grade(o) == 0
        @test o.terms[Int[]] == one(R)
    end

    # ── T(V) has NO relations ─────────────────────────────────────────────────
    # This is what distinguishes T(V) from the quotient algebras we build later.
    @testset "No relations — T(V) is free" begin
        # eᵢ⊗eᵢ ≠ 0  (would be 0 in the Exterior algebra)
        @test !iszero(e[1] * e[1])
        @test !iszero(e[2] * e[2])
        @test !iszero(e[3] * e[3])

        # eᵢ⊗eⱼ ≠ eⱼ⊗eᵢ  (would be equal in the Symmetric algebra)
        @test e[1] * e[2] != e[2] * e[1]
        @test e[1] * e[3] != e[3] * e[1]
        @test e[2] * e[3] != e[3] * e[2]

        # Order of indices matters
        @test e[1] * e[2] * e[1] != e[1] * e[1] * e[2]
    end

    # ── Associativity: (a · b) · c == a · (b · c) ────────────────────────────
    @testset "Associativity" begin
        # All 3³ = 27 triples of basis vectors
        for i in 1:3, j in 1:3, k in 1:3
            @test (e[i] * e[j]) * e[k] == e[i] * (e[j] * e[k])
        end

        # With linear combinations
        a = e[1] + e[2]
        b = e[2] + e[3]
        c = e[1] + e[3]
        @test (a * b) * c == a * (b * c)

        # With repeated indices (crucial: these must NOT cancel in T(V))
        @test (e[1] * e[1]) * e[2] == e[1] * (e[1] * e[2])
        @test (e[2] * e[1]) * e[1] == e[2] * (e[1] * e[1])
    end

    # ── Left distributivity: a · (b + c) == a·b + a·c ────────────────────────
    @testset "Left distributivity" begin
        for i in 1:3, j in 1:3, k in 1:3
            @test e[i] * (e[j] + e[k]) == e[i] * e[j] + e[i] * e[k]
        end
        @test e[1] * (e[2] - e[3]) == e[1] * e[2] - e[1] * e[3]
    end

    # ── Right distributivity: (a + b) · c == a·c + b·c ──────────────────────
    @testset "Right distributivity" begin
        for i in 1:3, j in 1:3, k in 1:3
            @test (e[i] + e[j]) * e[k] == e[i] * e[k] + e[j] * e[k]
        end
    end

    # ── Grading ───────────────────────────────────────────────────────────────
    @testset "Grading" begin
        # Grade of basis vectors
        for i in 1:3
            @test grade(e[i]) == 1
        end
        # Grade-2 products (all 9 pairs)
        for i in 1:3, j in 1:3
            @test grade(e[i] * e[j]) == 2
        end
        # Grade-3 products
        for i in 1:3, j in 1:3, k in 1:3
            @test grade(e[i] * e[j] * e[k]) == 3
        end
        # Scalars are grade-0
        @test grade(one(FreeTensor{R}, V3)) == 0
        @test grade(scalar_element(V3, R(7))) == 0

        # grade(a * b) = grade(a) + grade(b) for homogeneous a, b
        for k1 in 0:3, k2 in 0:3
            bas1 = homogeneous_basis(V3, R, k1)
            bas2 = homogeneous_basis(V3, R, k2)
            isempty(bas1) && continue
            isempty(bas2) && continue
            @test grade(bas1[1] * bas2[1]) == k1 + k2
        end
    end

    # ── Grade-k dimension = n^k ───────────────────────────────────────────────
    # Core invariant of T(V): the grade-k basis has exactly n^k elements.
    @testset "Grade-k dimension n^k" begin
        for n in 0:4
            V = VectorSpace(n)
            for k in 0:5
                expected = n^k
                indices  = all_grade_k_indices(V, k)

                @test length(indices) == expected           # correct count
                @test length(unique(indices)) == expected   # no duplicates
                @test all(length(idx) == k for idx in indices)  # correct length
                @test all(all(1 <= j <= n for j in idx) for idx in indices)  # valid entries
            end
        end
    end

    # ── Homogeneous basis completeness ────────────────────────────────────────
    @testset "Homogeneous basis completeness" begin
        for k in 0:3
            basis = homogeneous_basis(V3, R, k)
            @test length(basis) == 3^k
            @test all(grade(b) == k for b in basis)
            # Each basis element has a distinct multi-index (one entry each)
            @test length(unique(collect(keys(b.terms))[1] for b in basis)) == 3^k
        end
    end

    # ── Homogeneous component extraction ─────────────────────────────────────
    @testset "Homogeneous component" begin
        V = VectorSpace(2)
        e1, e2 = basis_vector(V, R, 1), basis_vector(V, R, 2)

        t = e1 + e1 * e2    # grade-1 + grade-2 mixed element

        @test homogeneous_component(t, 1) == e1
        @test homogeneous_component(t, 2) == e1 * e2
        @test iszero(homogeneous_component(t, 0))
        @test iszero(homogeneous_component(t, 3))

        # Re-assembly
        @test homogeneous_component(t, 1) + homogeneous_component(t, 2) == t
    end

    # ── Scalar ring arithmetic ────────────────────────────────────────────────
    @testset "Scalar ring arithmetic" begin
        V = VectorSpace(2)
        e1 = basis_vector(V, R, 1)
        e2 = basis_vector(V, R, 2)
        half = R(1, 2)

        @test (half * e1).terms[[1]] == half
        @test (e1 * half).terms[[1]] == half
        @test 2 * e1 == e1 + e1                     # integer literal convenience

        @test iszero(e1 - e1)
        @test iszero(e1 + (-e1))
        @test e1 + e1 == R(2) * e1

        # Coefficient arithmetic in a product
        t = (R(2) * e1) * (R(3) * e2)
        @test t.terms[[1,2]] == R(6)
    end

    # ── Scalar ring genericity (Float64 and Int) ──────────────────────────────
    @testset "Scalar ring genericity" begin
        V = VectorSpace(2)

        ef = [basis_vector(V, Float64, i) for i in 1:2]
        @test grade(ef[1] * ef[2]) == 2
        @test !iszero(ef[1] * ef[1])
        @test ef[1] * ef[2] != ef[2] * ef[1]

        ei = [basis_vector(V, Int, i) for i in 1:2]
        @test grade(ei[1] * ei[2]) == 2
        @test !iszero(ei[1] * ei[1])
    end

    # ── ⊗ operator and tensor_product alias ──────────────────────────────────
    @testset "tensor_product / ⊗ alias" begin
        @test tensor_product(e[1], e[2]) == e[1] * e[2]
        @test (e[1] ⊗ e[2])             == e[1] * e[2]
        @test (e[1] ⊗ e[1])             == e[1] * e[1]  # no collapse — T(V) is free
    end

    # ── grade_dimension convenience ───────────────────────────────────────────
    @testset "grade_dimension" begin
        for n in 0:5, k in 0:5
            @test grade_dimension(VectorSpace(n), k) == n^k
        end
    end

end  # @testset "FreeTensor — Phase 1"
