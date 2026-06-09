# ─────────────────────────────────────────────────────────────────────────────
# Phase L7: Base space / bundle interface
#
# Validates the four obligations + two capabilities against the three shipped
# realizations (GraphBase / GridBase / ManifoldChartBase): cell enumeration,
# the signed boundary as the discrete d (incl. d² = 0 and emptiness above
# top_grade), fibre descriptors (type-stable, dispatch-recoverable), transport
# (inverse + path-associativity / holonomy), capability gating, and the
# documented two-trait Hodge precondition.
# ─────────────────────────────────────────────────────────────────────────────

R    = Rational{BigInt}
clf3 = signature_metric(VectorSpace(3), R, 3, 0, 0)   # Cl(3,0), the QRCS fibre

# Unit-bivector versors (each a genuine versor / rotor) used as a connection.
e12 = clifford_basis_element(clf3, [1, 2])
e13 = clifford_basis_element(clf3, [1, 3])
e23 = clifford_basis_element(clf3, [2, 3])

# Triangle graph with a non-trivial rotor on every edge (a curved connection).
gb = GraphBase(3, [(1, 2), (2, 3), (3, 1)];
               metric  = clf3,
               versors = Dict(1 => e12, 2 => e13, 3 => e23))

grid = GridBase(2, 2)                                  # 9 verts, 12 edges, 4 faces

clf11 = signature_metric(VectorSpace(2), R, 1, 1, 0)  # Cl(1,1), Lorentzian fibre
b12   = clifford_basis_element(clf11, [1, 2])          # versor in Cl(1,1)
mb    = ManifoldChartBase([0.0, 1.0, 2.0], [(1, 2), (2, 3), (3, 1)];
                          metric    = clf11,
                          signature = (1, 1, 0),
                          versors   = Dict(1 => b12, 2 => b12))

# ─────────────────────────────────────────────────────────────────────────────
@testset "GraphBase: obligations & traits" begin
    @test top_grade(gb) == 1
    @test collect(cells(gb, 0)) == [1, 2, 3]
    @test collect(cells(gb, 1)) == [1, 2, 3]
    @test isempty(collect(cells(gb, 2)))           # nothing above top_grade
    @test n_cells(gb, 0) == 3
    @test n_cells(gb, 1) == 3

    # boundary = head − tail, orientation carried in the signs
    @test Set(boundary(gb, 1, 1)) == Set([(1, -1), (2, +1)])
    @test Set(boundary(gb, 1, 3)) == Set([(3, -1), (1, +1)])
    # empty for k == 0 and for k > top_grade (the contractual cases)
    @test isempty(boundary(gb, 0, 1))
    @test isempty(boundary(gb, 2, 1))

    # fibre attachment returns a descriptor (NOT a DataType, NOT a live element)
    d = fibre(gb, 0, 1)
    @test d isa CliffordFibre
    @test !(d isa DataType)
    @test fibre_eltype(d) == CliffordTensor{R}
    @test iszero(zero_fibre(d))
    @test fibre(gb, 1, 2) isa CliffordFibre          # uniform fibre on edges too

    # no metric, no dual complex — a bare graph cannot Hodge-dualize
    @test has_metric(gb) == false
    @test has_dual_complex(gb) == false
    @test can_hodge(gb) == false
    @test_throws ArgumentError metric(gb, 0, 1)
    @test_throws ArgumentError signature(gb)
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "GridBase: obligations, d² = 0, traits" begin
    @test top_grade(grid) == 2
    @test n_cells(grid, 0) == 9
    @test n_cells(grid, 1) == 12
    @test n_cells(grid, 2) == 4

    # discrete exterior derivative squared is zero: ∂∂(face) cancels with signs
    @testset "d² = 0 on every face" begin
        for f in cells(grid, 2)
            acc = Dict{Int,Int}()
            for (e, s1) in boundary(grid, 2, f)
                for (v, s2) in boundary(grid, 1, e)
                    acc[v] = get(acc, v, 0) + s1 * s2
                end
            end
            @test all(==(0), values(acc))
        end
    end

    # each face is bounded by 4 edges, each edge by 2 vertices
    @test length(boundary(grid, 2, 1)) == 4
    @test all(length(boundary(grid, 1, e)) == 2 for e in cells(grid, 1))
    @test isempty(boundary(grid, 0, 1))
    @test isempty(boundary(grid, 3, 1))            # above top_grade

    # metric + dual complex ⇒ Hodge available
    @test has_metric(grid) == true
    @test has_dual_complex(grid) == true
    @test can_hodge(grid) == true
    @test signature(grid) == (2, 0, 0)
    @test metric(grid, 2, 1) isa Metric
    @test fibre_eltype(fibre(grid, 2, 1)) == CliffordTensor{R}
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "ManifoldChartBase: connection + Lorentzian signature" begin
    @test top_grade(mb) == 1
    @test n_cells(mb, 0) == 3
    @test n_cells(mb, 1) == 3
    @test isempty(boundary(mb, 2, 1))              # above top_grade

    # declared, indefinite (Lorentzian) signature
    @test has_metric(mb) == true
    @test has_dual_complex(mb) == true
    @test can_hodge(mb) == true
    @test signature(mb) == (1, 1, 0)
    @test metric(mb, 0, 1) isa Metric{R}
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "Transport: inverse undoes it; path-associativity & holonomy" begin
    x  = clifford_basis_vector(clf3, 1)
    τ1 = transport(gb, 1)
    τ2 = transport(gb, 2)
    τ3 = transport(gb, 3)

    # the connection is non-trivial (a real rotor, not the identity)
    @test τ1(x) != x

    # reverse transport is inv(τ), both sides
    @test inv(τ1)(τ1(x)) == x
    @test τ1(inv(τ1)(x)) == x

    # path-associativity: the composition is associative, so loop holonomy is
    # well-defined and basis-independent
    @test ((τ3 ∘ τ2) ∘ τ1) == (τ3 ∘ (τ2 ∘ τ1))

    # transport along a composed path == composition of edge transports
    holo = τ3 ∘ τ2 ∘ τ1
    @test holo(x) == τ3(τ2(τ1(x)))

    # ── and on ManifoldChartBase (a genuine connection over a Lorentzian fibre)
    y  = clifford_basis_vector(clf11, 1)
    σ1 = transport(mb, 1)
    σ2 = transport(mb, 2)
    σ3 = transport(mb, 3)                            # identity (edge 3 unset)
    @test σ1(y) != y
    @test inv(σ1)(σ1(y)) == y
    @test ((σ3 ∘ σ2) ∘ σ1) == (σ3 ∘ (σ2 ∘ σ1))
    @test (σ3 ∘ σ2 ∘ σ1)(y) == σ3(σ2(σ1(y)))
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "Fibre descriptor is type-stable (dispatch, not DataType)" begin
    d = fibre(gb, 0, 1)
    @test !(d isa DataType)
    @test fibre_eltype(d) == CliffordTensor{R}
    # element type recovered by dispatch on the descriptor: @inferred on access
    fld = Field(gb, 0, Dict(1 => clifford_basis_vector(clf3, 1)))
    @test (@inferred evaluate(fld, 1)) isa CliffordTensor{R}   # stored branch
    @test (@inferred evaluate(fld, 2)) isa CliffordTensor{R}   # zero branch

    # the descriptor mechanism generalizes to other fibres (extension surface)
    td = TensorFibre{R}(VectorSpace(3))
    @test fibre_eltype(td) == MixedTensor{R}
    @test iszero(zero_fibre(td))
    @test !iszero(one_fibre(td))
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "Capability gating & the L8 Hodge precondition" begin
    # GraphBase is the case with NO dual complex; the others have one
    @test has_dual_complex(gb) == false
    @test has_dual_complex(grid) == true
    @test has_dual_complex(mb) == true

    # documented contract: ⋆ / δ (L8) require BOTH has_metric AND has_dual_complex
    for b in (gb, grid, mb)
        @test can_hodge(b) == (has_metric(b) && has_dual_complex(b))
    end
    @test can_hodge(gb) == false        # has the topology but no metric/dual cplx
    @test can_hodge(grid) == true
    @test can_hodge(mb) == true

    # metric/signature throw when has_metric == false
    @test_throws ArgumentError metric(gb, 0, 1)
    @test_throws ArgumentError signature(gb)
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "Construction validation" begin
    # edge referencing a non-existent node
    @test_throws ArgumentError GraphBase(2, [(1, 3)])
    # versor in the wrong algebra
    @test_throws ArgumentError GraphBase(2, [(1, 2)];
        metric = clf3, versors = Dict(1 => clifford_basis_element(clf11, [1, 2])))
    # grid must be at least 1×1
    @test_throws ArgumentError GridBase(0, 2)
    # declared signature must sum to the fibre dimension
    @test_throws ArgumentError ManifoldChartBase([0.0, 1.0], [(1, 2)];
        metric = clf11, signature = (3, 0, 0))
end
