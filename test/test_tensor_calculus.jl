# ─────────────────────────────────────────────────────────────────────────────
# Phase 6: Tensor-calculus core
#
# Dual space, mixed variance, metric-free contraction, and the metric-induced
# musical isomorphisms (raise/lower).  Plus the AbstractTensorElement refactor
# and the new CliffordTensor display.
# ─────────────────────────────────────────────────────────────────────────────

R  = Rational{BigInt}
V0 = VectorSpace(0)
V2 = VectorSpace(2)
V3 = VectorSpace(3)

# A non-trivial invertible symmetric metric on V3 (positive definite):
#   g = [2 1 0; 1 2 0; 0 0 3]
gGen = Metric{R}(V3, Matrix{R}([R(2) R(1) R(0); R(1) R(2) R(0); R(0) R(0) R(3)]))
gE3  = signature_metric(V3, R, 3, 0, 0)          # Euclidean (identity)
gDeg = signature_metric(V3, R, 2, 0, 1)          # degenerate: one null direction

# ─────────────────────────────────────────────────────────────────────────────
@testset "AbstractTensorElement refactor" begin

    @testset "Subtyping" begin
        @test FreeTensor{R}            <: AbstractTensorElement{R}
        @test AlgebraTensor{ExteriorAlgebra, R} <: AbstractTensorElement{R}
        @test CliffordTensor{R}        <: AbstractTensorElement{R}
        @test MixedTensor{R}           <: AbstractTensorElement{R}
    end

    @testset "Lifted grade/grades/homogeneous_component still work" begin
        t = basis_element(V3, R, [1,2]) + basis_element(V3, R, [3])
        @test grades(t) == [1, 2]
        @test grade(homogeneous_component(t, 2)) == 2
        @test iszero(homogeneous_component(t, 0))
        @test_throws ArgumentError grade(t)                     # inhomogeneous
        @test_throws ArgumentError grade(zero(FreeTensor{R}, V3))
        @test grades(zero(FreeTensor{R}, V3)) == Int[]
    end

    @testset "== and hash inherited, Clifford keyed on metric not space" begin
        a = clifford_basis_vector(gE3, R, 1)
        b = clifford_basis_vector(signature_metric(V3, R, 2, 1, 0), R, 1)
        # same terms + same space, but different metric ⇒ not equal
        @test a != b
        @test a == clifford_basis_vector(gE3, R, 1)
        @test hash(a) == hash(clifford_basis_vector(gE3, R, 1))
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "CliffordTensor display" begin
    e1 = clifford_basis_vector(gE3, R, 1)
    e2 = clifford_basis_vector(gE3, R, 2)
    @test sprint(show, e1)               == "e1"
    @test sprint(show, e1 * e2)          == "e1∧e2"
    @test sprint(show, -(e1 * e2))       == "-e1∧e2"
    @test sprint(show, clifford_zero(gE3, R)) == "0"
    @test sprint(show, R(3) * e1)        == "(3//1)⋅e1"
    @test sprint(show, clifford_one(gE3, R))  == "𝟏"
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "VectorSpace equality is dimension-only" begin
    @test VectorSpace(3, [:x,:y,:z]) == VectorSpace(3)
    @test hash(VectorSpace(3, [:x,:y,:z])) == hash(VectorSpace(3))
    # tensors over same-dim differently-labelled spaces interoperate
    a = basis_vector(VectorSpace(3, [:x,:y,:z]), R, 1)
    b = basis_vector(VectorSpace(3), R, 2)
    @test grade(a * b) == 2
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "MixedTensor construction & type bookkeeping" begin
    u  = mixed_basis_vector(V3, R, 1)          # e₁,  (1,0)
    cu = mixed_basis_covector(V3, R, 2)        # e²,  (0,1)

    @test tensor_type(u)  == (1, 0)
    @test tensor_type(cu) == (0, 1)
    @test variance_pattern(u)  == [Up()]
    @test variance_pattern(cu) == [Down()]
    @test tensor_type(mixed_scalar(V3, R(5))) == (0, 0)

    t = u ⊗ cu                                  # e₁ ⊗ e²,  (1,1)
    @test tensor_type(t) == (1, 1)
    @test variance_pattern(t) == [Up(), Down()]
    @test grade(t) == 2                          # total slot count

    # out-of-range index
    @test_throws ArgumentError mixed_basis_vector(V3, R, 4)
    @test_throws ArgumentError mixed_basis_covector(V3, R, 0)

    # zero tensor: type / pattern undefined
    @test_throws ArgumentError variance_pattern(mixed_zero(V3, R))
    @test_throws ArgumentError tensor_type(mixed_zero(V3, R))

    # mixed variance pattern is not a homogeneous type
    bad = mixed_basis_element(V3, R, [(1, Up())]) +
          mixed_basis_element(V3, R, [(1, Down())])
    @test_throws ArgumentError tensor_type(bad)
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "Tensor product valence" begin
    a = mixed_basis_element(V3, R, [(1, Up()), (2, Down())])    # (1,1)
    b = mixed_basis_element(V3, R, [(3, Up())])                 # (1,0)
    c = a ⊗ b
    @test tensor_type(c) == (2, 1)
    @test variance_pattern(c) == [Up(), Down(), Up()]
    @test c == tensor_product(a, b) == a * b
    # zero absorbs
    @test iszero(a ⊗ mixed_zero(V3, R))
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "δ-trace: contracting the identity (1,1) tensor" begin
    for n in 0:4
        Vn = VectorSpace(n)
        id = identity_tensor(Vn, R)
        if n == 0
            @test iszero(id)                               # no slots ⇒ empty
        else
            @test tensor_type(id) == (1, 1)
            @test contract(id, 1, 2) == mixed_scalar(Vn, R(n))
            @test trace(id) == mixed_scalar(Vn, R(n))
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "Contraction: type bookkeeping & errors" begin
    # (2,1) tensor with matching indices on the contracted pair
    t = mixed_basis_element(V3, R, [(1, Up()), (2, Up()), (1, Down())])
    @test tensor_type(t) == (2, 1)

    c = contract(t, 1, 3)                       # upper slot 1 vs lower slot 3
    @test tensor_type(c) == (1, 0)              # (2,1) → (1,1-1)=(1,0)
    @test c == mixed_basis_vector(V3, R, 2)

    # non-matching indices contract to zero (δ picks the diagonal)
    t2 = mixed_basis_element(V3, R, [(1, Up()), (2, Up()), (3, Down())])
    @test iszero(contract(t2, 1, 3))
    @test iszero(contract(t2, 2, 3))

    # errors: slots must be exactly one upper and one lower, and distinct
    @test_throws ArgumentError contract(t, 1, 1)     # a == b
    @test_throws ArgumentError contract(t, 1, 2)     # both upper
    @test_throws ArgumentError contract(t, 3, 1)     # a is lower, b is upper
    @test_throws ArgumentError contract(t, 1, 9)     # out of range

    # trace only defined on (1,1)
    @test_throws ArgumentError trace(t)
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "Exact inverse metric" begin
    ginv = inverse_metric(gGen)
    @test ginv isa Metric{R}
    # g · g⁻¹ = I, checked entrywise
    n = V3.n
    for i in 1:n, j in 1:n
        s = sum(gGen.g[i,k] * ginv.g[k,j] for k in 1:n)
        @test s == (i == j ? one(R) : zero(R))
    end
    @test inverse_metric(gE3) == gE3              # I⁻¹ = I
    # degenerate metric has no inverse
    @test_throws ArgumentError inverse_metric(gDeg)
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "Musical isomorphisms: raise/lower" begin
    u = mixed_basis_vector(V3, R, 1)              # e₁, (1,0)

    @testset "lower flips variance, raise inverts it" begin
        ul = lower(u, 1, gGen)
        @test tensor_type(ul) == (0, 1)
        @test raise(ul, 1, gGen) == u                       # ♯∘♭ = id

        cu = mixed_basis_covector(V3, R, 2)
        cur = raise(cu, 1, gGen)
        @test tensor_type(cur) == (1, 0)
        @test lower(cur, 1, gGen) == cu                     # ♭∘♯ = id
    end

    @testset "round-trip on a higher-rank tensor, every slot" begin
        t = mixed_basis_element(V3, R, [(1, Up()), (2, Up())])   # (2,0)
        for s in 1:2
            @test raise(lower(t, s, gGen), s, gGen) == t
        end
    end

    @testset "Euclidean metric is a component-preserving no-op" begin
        # With g = I, lowering only flips the variance tag; coefficients unchanged.
        ul = lower(u, 1, gE3)
        @test ul.terms == Dict([(1, Down())] => one(R))
        # general element keeps its coefficients
        w  = R(2) * mixed_basis_vector(V3, R, 1) + R(-3) * mixed_basis_vector(V3, R, 3)
        wl = lower(w, 1, gE3)
        @test wl.terms == Dict([(1, Down())] => R(2), [(3, Down())] => R(-3))
        @test raise(wl, 1, gE3) == w
    end

    @testset "errors: wrong variance, degenerate metric, space mismatch" begin
        @test_throws ArgumentError lower(mixed_basis_covector(V3, R, 1), 1, gGen)  # slot is Down
        @test_throws ArgumentError raise(u, 1, gGen)                               # slot is Up
        @test_throws ArgumentError raise(mixed_basis_covector(V3, R, 1), 1, gDeg)  # degenerate
        @test_throws ArgumentError lower(mixed_basis_vector(V2, R, 1), 1, gGen)    # space mismatch
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "Inner-product recovery via contraction" begin
    # contract(u ⊗ lower(v)) reproduces the bilinear form g(u, v).
    for a in 1:3, b in 1:3
        u = mixed_basis_vector(V3, R, a)
        v = mixed_basis_vector(V3, R, b)
        ip = contract(u ⊗ lower(v, 1, gGen), 1, 2)
        @test ip == mixed_scalar(V3, bilinear_form(gGen, a, b))
    end

    # a genuine linear combination, checked against the quadratic form
    u = mixed_basis_vector(V3, R, 1) + R(2) * mixed_basis_vector(V3, R, 2)
    ip = contract(u ⊗ lower(u, 1, gGen), 1, 2)
    # g(e1+2e2, e1+2e2) = g11 + 4 g12 + 4 g22 = 2 + 4 + 8 = 14
    @test ip == mixed_scalar(V3, R(14))
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "Backward compatibility: all-Up MixedTensor ≡ FreeTensor" begin
    a = basis_element(V3, R, [1, 1, 2])
    b = basis_element(V3, R, [2, 3])

    @testset "embedding and projection round-trip" begin
        @test as_free_tensor(MixedTensor(a)) == a
        @test tensor_type(MixedTensor(a)) == (3, 0)
        @test grade(MixedTensor(a)) == grade(a)
    end

    @testset "tensor product matches T(V) concatenation" begin
        @test MixedTensor(a) * MixedTensor(b) == MixedTensor(a * b)
        @test as_free_tensor(MixedTensor(a) * MixedTensor(b)) == a * b
    end

    @testset "addition / scalar action match" begin
        @test MixedTensor(a) + MixedTensor(b) == MixedTensor(a + b)
        @test R(3) * MixedTensor(a) == MixedTensor(R(3) * a)
    end

    @testset "as_free_tensor rejects covariant slots" begin
        @test_throws ArgumentError as_free_tensor(mixed_basis_covector(V3, R, 1))
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "Degenerate-dimension edge cases (V0)" begin
    @test iszero(identity_tensor(V0, R))
    @test inverse_metric(signature_metric(V0, R, 0, 0, 0)) isa Metric{R}
    @test mixed_one(V0, R) == mixed_scalar(V0, one(R))
end

# ─────────────────────────────────────────────────────────────────────────────
# Symbolic path: repeat the key identities over R = Symbolics.Num, compared with
# isequal_simplified (skipped gracefully if Symbolics is unavailable).
# ─────────────────────────────────────────────────────────────────────────────
symbolics_available = try
    @eval using Symbolics
    true
catch
    false
end

if !symbolics_available
    @info "Symbolics.jl not installed — Phase 6 symbolic tests skipped."
else
    @testset "Phase 6 symbolic path (Symbolics.Num)" begin
        S  = Symbolics.Num
        Vs = VectorSpace(3)
        gS = symbolic_metric(signature_metric(Vs, Rational{BigInt}, 3, 0, 0))

        as = symbolic_vars(:a, 3)
        bs = symbolic_vars(:b, 3)
        u  = sum(as[i] * mixed_basis_vector(Vs, S, i) for i in 1:3)   # (1,0)
        w  = sum(bs[i] * mixed_basis_vector(Vs, S, i) for i in 1:3)

        @testset "raise/lower round-trip" begin
            @test isequal_simplified(raise(lower(u, 1, gS), 1, gS), u)
        end

        @testset "inner-product recovery (Euclidean ⇒ Σ aᵢbᵢ)" begin
            ip       = contract(u ⊗ lower(w, 1, gS), 1, 2)
            expected = mixed_scalar(Vs, sum(as[i] * bs[i] for i in 1:3))
            @test isequal_simplified(ip, expected)
        end

        @testset "δ-trace over symbolic ring" begin
            @test isequal_simplified(contract(identity_tensor(Vs, S), 1, 2),
                                     mixed_scalar(Vs, S(3)))
        end

        @testset "Metric{Num} from the matrix constructor (symbolic entries)" begin
            # The symmetry check must not choke on symbolic entries: `==` on Num
            # is not a Bool, so the constructor must compare with `isequal`.
            ga = Symbolics.variable(:ga)
            gb = Symbolics.variable(:gb)
            gc = Symbolics.variable(:gc)
            V2s = VectorSpace(2)
            gM  = Metric{S}(V2s, S[ga gb; gb gc])      # would throw under `==`
            ginv = inverse_metric(gM)
            @test ginv isa Metric{S}
            # g · g⁻¹ = I, entrywise, after symbolic simplification
            for i in 1:2, j in 1:2
                s = sum(gM.g[i,k] * ginv.g[k,j] for k in 1:2)
                @test isequal(Symbolics.simplify(s - (i == j ? one(S) : zero(S))),
                              zero(S))
            end
        end

        @testset "symbolic 3×3 inverse: no spurious asymmetry on re-validation" begin
            # The cofactor expressions for inv[i,j] and inv[j,i] are equal but
            # not structurally identical; the unchecked constructor in
            # inverse_metric must not reject the (symmetric) inverse.
            m = [Symbolics.variable(Symbol(:m, i)) for i in 1:6]
            V3s = VectorSpace(3)
            g3  = Metric{S}(V3s, S[m[1] m[2] m[3]; m[2] m[4] m[5]; m[3] m[5] m[6]])
            @test inverse_metric(g3) isa Metric{S}
        end
    end
end
