# ─────────────────────────────────────────────────────────────────────────────
# Roadmap §9 item 3: linear maps and outermorphisms
#
# LinearMap stores an endomorphism f : V → V as a matrix whose column i is
# f(eᵢ).  The outermorphism is the unique grade-preserving extension to T(V),
# Λ(V), Cl(V,g), and mixed-variance tensors.
# ─────────────────────────────────────────────────────────────────────────────

R  = Rational{BigInt}
V2 = VectorSpace(2)
V3 = VectorSpace(3)

A3 = Matrix{R}([
    R(1)  R(2)  R(0);
    R(0)  R(1)  R(3);
    R(4)  R(0)  R(1)
])
B3 = Matrix{R}([
    R(2)  R(0)  R(1);
    R(1)  R(1)  R(0);
    R(0)  R(1)  R(1)
])
f3 = linear_map(V3, R, A3)
g3 = linear_map(V3, R, B3)

@testset "LinearMap construction, display, equality, and composition" begin
    @test f3 isa LinearMap{R}
    @test f3.space == V3
    @test f3.matrix == A3
    @test linear_map(V3, R, A3) == linear_map(VectorSpace(3, [:x, :y, :z]), R, A3)
    @test hash(linear_map(V3, R, A3)) == hash(linear_map(V3, R, A3))
    @test occursin("LinearMap", sprint(show, f3))
    expected_id = fill(zero(R), V3.n, V3.n)
    for i in 1:V3.n
        expected_id[i, i] = one(R)
    end
    @test identity_map(V3, R).matrix == expected_id
    @test (f3 ∘ g3).matrix == A3 * B3

    @test_throws ArgumentError linear_map(V3, R, Matrix{R}(undef, 2, 3))
    @test_throws ArgumentError linear_map(V3, R, Matrix{R}(undef, 2, 2))
end

@testset "apply_map on grade-1 FreeTensor matches matrix-vector product" begin
    v = R(2) * basis_vector(V3, R, 1) - R(3) * basis_vector(V3, R, 2) + basis_vector(V3, R, 3)
    w = apply_map(f3, v)
    coeffs = [R(2), R(-3), R(1)]
    expected = A3 * coeffs
    for j in 1:V3.n
        @test get(w.terms, [j], zero(R)) == expected[j]
    end
    @test iszero(apply_map(f3, zero(FreeTensor{R}, V3)))

    aext = R(2) * ext_basis_vector(V3, R, 1) - R(3) * ext_basis_vector(V3, R, 2) + ext_basis_vector(V3, R, 3)
    gE = signature_metric(V3, R, 3, 0, 0)
    acl = R(2) * clifford_basis_vector(gE, R, 1) - R(3) * clifford_basis_vector(gE, R, 2) + clifford_basis_vector(gE, R, 3)
    amix = R(2) * mixed_basis_vector(V3, R, 1) - R(3) * mixed_basis_vector(V3, R, 2) + mixed_basis_vector(V3, R, 3)
    @test apply_map(f3, aext) == outermorphism(f3, aext)
    @test apply_map(f3, acl) == outermorphism(f3, acl)
    @test apply_map(f3, amix) == outermorphism(f3, amix)

    @test_throws ArgumentError apply_map(f3, basis_vector(V3, R, 1) * basis_vector(V3, R, 2))
end

@testset "Outermorphism on T(V), Λ(V), and Cl(V,g)" begin
    e1 = ext_basis_vector(V3, R, 1)
    e2 = ext_basis_vector(V3, R, 2)
    e3 = ext_basis_vector(V3, R, 3)
    a  = R(2) * e1 - e2 + e3
    b  = e1 + R(3) * e2
    blade = e1 ∧ e2

    @test grades(outermorphism(f3, blade)) == [2]
    @test outermorphism(f3, a ∧ b) == outermorphism(f3, a) ∧ outermorphism(f3, b)

    Iext = e1 ∧ e2 ∧ e3
    @test outermorphism(f3, Iext) == determinant(f3) * Iext
    @test determinant(f3) == Tensorsmith._det(f3.matrix)

    xfree = basis_vector(V3, R, 1) * basis_vector(V3, R, 2)
    @test outermorphism(f3, xfree) == apply_map(f3, basis_vector(V3, R, 1)) *
                                      apply_map(f3, basis_vector(V3, R, 2))
    @test outermorphism(identity_map(V3, R), xfree) == xfree
    @test outermorphism(identity_map(V3, R), a + blade) == a + blade
    @test outermorphism(f3 ∘ g3, blade) == outermorphism(f3, outermorphism(g3, blade))
    @test outermorphism(f3 ∘ g3, xfree) == outermorphism(f3, outermorphism(g3, xfree))

    gE = signature_metric(V3, R, 3, 0, 0)
    c1 = clifford_basis_vector(gE, R, 1)
    c2 = clifford_basis_vector(gE, R, 2)
    c3 = clifford_basis_vector(gE, R, 3)
    cblade = wedge(c1, c2)
    @test grades(outermorphism(f3, cblade)) == [2]
    @test outermorphism(f3, wedge(c1 + R(2)*c2, c2 + c3)) ==
          wedge(outermorphism(f3, c1 + R(2)*c2), outermorphism(f3, c2 + c3))

    # In the exterior/Clifford/common grade-1 cases, the same matrix action appears.
    @test outermorphism(f3, e1 + R(2)*e3).terms == outermorphism(f3, c1 + R(2)*c3).terms
    @test outermorphism(f3, blade).terms == outermorphism(f3, cblade).terms
end

@testset "MixedTensor variance-aware outermorphism" begin
    A2 = Matrix{R}([R(2) R(1); R(0) R(3)])
    f2 = linear_map(V2, R, A2)
    finv2 = linear_map(V2, R, Tensorsmith._matrix_inverse(A2))

    u1 = mixed_basis_vector(V2, R, 1)
    u2 = mixed_basis_vector(V2, R, 2)
    d1 = mixed_basis_covector(V2, R, 1)
    d2 = mixed_basis_covector(V2, R, 2)

    t = R(2)*(u1 ⊗ d1) + R(3)*(u1 ⊗ d2) - (u2 ⊗ d1) + R(4)*(u2 ⊗ d2)
    @test contract(outermorphism(f2, t), 1, 2) == outermorphism(f2, contract(t, 1, 2))

    mixed = R(5)*(u1 ⊗ d2) - R(7)*(u2 ⊗ d1)
    @test outermorphism(finv2, outermorphism(f2, mixed)) == mixed

    free = basis_vector(V2, R, 1) * (R(3) * basis_vector(V2, R, 1) - basis_vector(V2, R, 2))
    @test as_free_tensor(outermorphism(f2, MixedTensor(free))) == outermorphism(f2, free)

    singular = linear_map(V2, R, Matrix{R}([R(1) R(0); R(0) R(0)]))
    @test_throws ArgumentError outermorphism(singular, d1)
    @test outermorphism(singular, u2) == mixed_zero(V2, R)
end

linear_maps_symbolics_available = try
    @eval using Symbolics
    true
catch
    false
end

if !linear_maps_symbolics_available
    @info "Symbolics.jl not installed — linear-map symbolic tests skipped."
else
    S = Symbolics.Num
    VS2 = VectorSpace(2)

    @testset "Linear maps over Symbolics.Num" begin
        a, b, c, d = symbolic_vars(:m, 4)
        MS = Matrix{S}([a b; c d])
        fs = linear_map(VS2, S, MS)
        e1 = ext_basis_vector(VS2, S, 1)
        e2 = ext_basis_vector(VS2, S, 2)
        I2 = e1 ∧ e2

        @test isequal_simplified(outermorphism(fs, I2), determinant(fs) * I2)
        @test isequal(Symbolics.expand(determinant(fs) - Tensorsmith._det(fs.matrix)), zero(S))

        xfree = basis_vector(VS2, S, 1) * basis_vector(VS2, S, 2)
        lhs = outermorphism(fs, xfree)
        rhs = apply_map(fs, basis_vector(VS2, S, 1)) * apply_map(fs, basis_vector(VS2, S, 2))
        @test isequal_simplified(lhs, rhs)

        # Use an invertible exact matrix lifted to S so inverse-transpose action is exact.
        Ns = Matrix{S}([S(2) S(1); S(0) S(3)])
        hs = linear_map(VS2, S, Ns)
        mt = mixed_basis_vector(VS2, S, 1) ⊗ mixed_basis_covector(VS2, S, 1) +
             mixed_basis_vector(VS2, S, 2) ⊗ mixed_basis_covector(VS2, S, 2)
        @test isequal_simplified(contract(outermorphism(hs, mt), 1, 2),
                                 outermorphism(hs, contract(mt, 1, 2)))
    end
end
