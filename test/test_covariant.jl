# ─────────────────────────────────────────────────────────────────────────────
# Phase L8.1 (Tier 2): covariant derivative, holonomy, R/T/Q derived views
#
# Validates the settled DESIGN.md §15.1/§15.2 architecture: two first-class
# transports (two-sided VersorTransport and one-sided GaugeTransport), holonomy as
# the primitive loop function on every base, curvature/torsion as lossy derived
# face fields, and nonmetricity as a separate edge field from local metric change.
# ─────────────────────────────────────────────────────────────────────────────

R = Rational{BigInt}

_cl_scalar(m, x) = clifford_scalar(m, R(x))
_cl_coeff(A, idx) = get(A.terms, idx, zero(eltype(A.metric.g)))

import Tensorsmith: top_grade, cells, n_cells, fibre, transport,
                    has_metric, metric, signature, has_dual_complex

function _same_terms_approx(a::CliffordTensor{Float64}, b::CliffordTensor{Float64}; atol=1e-10)
    a.metric == b.metric || return false
    for idx in union(keys(a.terms), keys(b.terms))
        isapprox(get(a.terms, idx, 0.0), get(b.terms, idx, 0.0); atol=atol, rtol=atol) || return false
    end
    return true
end

# A tiny 2-complex used to test non-flat face holonomy without changing GridBase.
struct TriangleFaceBase{Rr} <: BaseSpace
    metric :: Metric{Rr}
    versors :: Dict{Int,CliffordTensor{Rr}}
end

top_grade(::TriangleFaceBase) = 2
cells(::TriangleFaceBase, k::Integer) = k == 0 ? (1:3) : k == 1 ? (1:3) : k == 2 ? (1:1) : (1:0)
n_cells(::TriangleFaceBase, k::Integer) = k == 0 ? 3 : k == 1 ? 3 : k == 2 ? 1 : 0
function Tensorsmith._boundary(::TriangleFaceBase, k::Int, cell::Integer)
    if k == 1
        edges = ((1, 2), (2, 3), (3, 1))
        t, h = edges[cell]
        return Tuple{Int,Int}[(t, -1), (h, +1)]
    end
    @assert k == 2
    cell == 1 || throw(ArgumentError("triangle has one face"))
    Tuple{Int,Int}[(1, +1), (2, +1), (3, +1)]
end
fibre(b::TriangleFaceBase{Rr}, k::Integer, cell) where Rr = CliffordFibre{Rr}(b.metric)
transport(b::TriangleFaceBase{Rr}, edge::Integer) where Rr =
    haskey(b.versors, edge) ? VersorTransport(b.versors[edge]) : identity_transport(b.metric)
has_metric(::TriangleFaceBase) = true
metric(b::TriangleFaceBase, k::Integer, cell) = b.metric
signature(b::TriangleFaceBase) = Tensorsmith._signature_of(b.metric)
has_dual_complex(::TriangleFaceBase) = true

# Gauge transport test base: same topology/fibre as a graph, but one-sided maps.
struct GaugeGraphForTest{Rr} <: BaseSpace
    n_nodes :: Int
    edges :: Vector{Tuple{Int,Int}}
    metric :: Metric{Rr}
    gauges :: Dict{Int,GaugeTransport{Rr}}
end

top_grade(::GaugeGraphForTest) = 1
cells(b::GaugeGraphForTest, k::Integer) = k == 0 ? (1:b.n_nodes) : k == 1 ? (1:length(b.edges)) : (1:0)
n_cells(b::GaugeGraphForTest, k::Integer) = k == 0 ? b.n_nodes : k == 1 ? length(b.edges) : 0
function Tensorsmith._boundary(b::GaugeGraphForTest, k::Int, edge::Integer)
    t, h = b.edges[edge]
    Tuple{Int,Int}[(t, -1), (h, +1)]
end
fibre(b::GaugeGraphForTest{Rr}, k::Integer, cell) where Rr = CliffordFibre{Rr}(b.metric)
transport(b::GaugeGraphForTest{Rr}, edge::Integer) where Rr =
    haskey(b.gauges, edge) ? b.gauges[edge] : identity_gauge_transport(b.metric)

# Metric-capable graph used solely for Q = inter-node metric variation.
struct MetricGraphForTest{Rr} <: BaseSpace
    edges :: Vector{Tuple{Int,Int}}
    fibres_metric :: Metric{Rr}
    node_metrics :: Vector{Metric{Rr}}
end

top_grade(::MetricGraphForTest) = 1
cells(b::MetricGraphForTest, k::Integer) = k == 0 ? (1:length(b.node_metrics)) : k == 1 ? (1:length(b.edges)) : (1:0)
n_cells(b::MetricGraphForTest, k::Integer) = length(cells(b, k))
function Tensorsmith._boundary(b::MetricGraphForTest, k::Int, edge::Integer)
    t, h = b.edges[edge]
    Tuple{Int,Int}[(t, -1), (h, +1)]
end
fibre(b::MetricGraphForTest{Rr}, k::Integer, cell) where Rr = CliffordFibre{Rr}(b.fibres_metric)
transport(b::MetricGraphForTest{Rr}, edge::Integer) where Rr = identity_transport(b.fibres_metric)
has_metric(::MetricGraphForTest) = true
metric(b::MetricGraphForTest, k::Integer, cell) = k == 0 ? b.node_metrics[cell] : b.fibres_metric
signature(b::MetricGraphForTest) = Tensorsmith._signature_of(b.fibres_metric)

@testset "Covariant derivative: flat connection and gauge one-sided action" begin
    m = signature_metric(VectorSpace(3), R, 3, 0, 0)
    graph = GraphBase(2, [(1, 2)]; metric = m)
    ψ = Field(graph, 0, Dict(1 => _cl_scalar(m, 2), 2 => _cl_scalar(m, 5)))

    ∇ψ = @inferred ∇(ψ)
    @test field_grade(∇ψ) == 1
    @test ∇ψ == d(ψ)
    @test evaluate(∇ψ, 1) == _cl_scalar(m, 3)

    e1 = clifford_basis_vector(m, 1)
    U = R(2) * clifford_one(m)
    Uinv = (one(R) // 2) * clifford_one(m)
    gauge = GaugeGraphForTest(2, [(1, 2)], m, Dict(1 => GaugeTransport(U, Uinv)))
    χ = Field(gauge, 0, Dict(1 => e1, 2 => clifford_zero(m)))
    @test transport(gauge, 1)(e1) == R(2) * e1           # one-sided U·ψ
    @test evaluate(∇(χ), 1) == -(R(2) * e1)              # not UψU⁻¹, which would be e1
end

@testset "Field keeps the L7 fibre-element bound" begin
    grid = GridBase(1, 1)
    @test !(Matrix{R} <: AbstractTensorElement{R})
    @test_throws TypeError Field{R,Matrix{R},typeof(grid)}(grid, 1, Dict{Int,Matrix{R}}())
end

@testset "Holonomy: graph support, orientation, trace, and non-abelian order" begin
    m = signature_metric(VectorSpace(3), R, 3, 0, 0)
    e12 = clifford_basis_element(m, [1, 2])
    e13 = clifford_basis_element(m, [1, 3])

    tri = GraphBase(3, [(1, 2), (2, 3), (3, 1)]; metric = m, versors = Dict(1 => e12))
    h = holonomy(tri, [1, 2, 3])
    @test h isa VersorTransport{R}
    @test h.versor == e12
    @test holonomy(tri, [-3, -2, -1]) == inv(h)
    @test holonomy_trace(tri, [1, 2, 3]) == holonomy_trace(tri, [2, 3, 1])

    one = clifford_one(m)
    v1 = R(3//5)  * one + R(4//5)   * e12
    v2 = R(5//13) * one + R(12//13) * e13
    v3 = R(7//25) * one + R(24//25) * clifford_basis_element(m, [2, 3])
    rebased = GraphBase(3, [(1, 2), (2, 3), (3, 1)];
        metric = m, versors = Dict(1 => v1, 2 => v2, 3 => v3))
    h123 = holonomy(rebased, [1, 2, 3])
    h231 = holonomy(rebased, [2, 3, 1])
    @test h123 != h231
    @test h231 == transport(rebased, 1) ∘ h123 ∘ inv(transport(rebased, 1))
    @test holonomy_trace(rebased, [1, 2, 3]) == holonomy_trace(rebased, [2, 3, 1])
    @test holonomy(rebased, [-3, -2, -1]) == inv(h123)
    @test holonomy_trace(rebased, [-3, -2, -1]) == holonomy_trace(rebased, [1, 2, 3])

    loops = GraphBase(1, [(1, 1), (1, 1)]; metric = m, versors = Dict(1 => e12, 2 => e13))
    @test holonomy(loops, [1, 2]) != holonomy(loops, [2, 1])

    bad_exact = clifford_one(m) + clifford_basis_vector(m, 1)
    @test_throws ArgumentError VersorTransport(bad_exact)

    mf_bad = signature_metric(VectorSpace(3), Float64, 3, 0, 0)
    bad_float = clifford_one(mf_bad) + clifford_basis_vector(mf_bad, 1)
    @test_throws ArgumentError VersorTransport(bad_float)

    ψ = Field(tri, 0, Dict(1 => clifford_basis_vector(m, 1)))
    @test ∇(ψ) isa Field
    @test holonomy_trace(tri, [1, 2, 3]) == zero(R)
    @test_throws ArgumentError holonomy_field(tri)
    @test_throws ArgumentError curvature(tri)
    @test_throws ArgumentError torsion(tri)
    @test_throws ArgumentError nonmetricity(tri)
end

@testset "Holonomy field, curvature, torsion, and trace invariance on 2-cells" begin
    grid = GridBase(1, 1)
    hf = @inferred holonomy_field(grid)
    @test hf isa HolonomyField
    @test !(hf isa Field)
    @test field_grade(hf) == 2
    @test evaluate(hf, 1) == identity_transport(grid.metric)
    @test holonomy_trace(grid, boundary(grid, 2, 1)) == holonomy_trace(grid, circshift(collect(boundary(grid, 2, 1)), -1))
    @test all(iszero(evaluate(curvature(grid), f)) for f in cells(grid, 2))
    @test all(iszero(evaluate(torsion(grid), f)) for f in cells(grid, 2))

    mf = signature_metric(VectorSpace(3), Float64, 3, 0, 0)
    B = 0.25 * clifford_basis_element(mf, [1, 2])
    r = rotor_exp(B)
    tri = TriangleFaceBase(mf, Dict(1 => r))
    h = holonomy(tri, boundary(tri, 2, 1))
    @test _same_terms_approx(h.versor, r)
    @test isapprox(holonomy_trace(tri, boundary(tri, 2, 1)), cos(0.25); atol=1e-10)
    curv = curvature(tri)
    @test isapprox(_cl_coeff(evaluate(curv, 1), [1, 2]), 0.25; atol=1e-10)

    m_exact = signature_metric(VectorSpace(2), R, 2, 0, 0)
    e12_exact = clifford_basis_element(m_exact, [1, 2])
    exact_versor = R(3//5) * clifford_one(m_exact) + R(4//5) * e12_exact
    exact_tri = TriangleFaceBase(m_exact, Dict(1 => exact_versor))
    @test _cl_coeff(evaluate(curvature(exact_tri), 1), [1, 2]) == R(4//5)

    m_float = signature_metric(VectorSpace(2), Float64, 2, 0, 0)
    e12_float = clifford_basis_element(m_float, [1, 2])
    float_versor = 0.6 * clifford_one(m_float) + 0.8 * e12_float
    float_tri = TriangleFaceBase(m_float, Dict(1 => float_versor))
    θ = _cl_coeff(evaluate(curvature(float_tri), 1), [1, 2])
    @test isapprox(θ, atan(0.8, 0.6); atol=1e-12)
    @test !isapprox(θ, 0.8; atol=1e-12)

    m4 = signature_metric(VectorSpace(4), Float64, 4, 0, 0)
    r12 = rotor_exp(0.1 * clifford_basis_element(m4, [1, 2]))
    r34 = rotor_exp(0.2 * clifford_basis_element(m4, [3, 4]))
    compound = r34 * r12
    @test_throws ArgumentError curvature(TriangleFaceBase(m4, Dict(1 => compound)))
end

@testset "Nonmetricity: separate metric variation, no holonomy" begin
    V = VectorSpace(2)
    fibre_m = signature_metric(V, R, 2, 0, 0)
    g1 = diagonal_metric(V, R, R[1, 1])
    g2 = diagonal_metric(V, R, R[2, 1])

    varying = MetricGraphForTest([(1, 2)], fibre_m, [g1, g2])
    Q = @inferred nonmetricity(varying)
    @test Q isa MetricVariationField
    @test !(Q isa Field)
    @test field_grade(Q) == 1
    @test evaluate(Q, 1) == R[1 0; 0 0]

    uniform = MetricGraphForTest([(1, 2)], fibre_m, [g1, g1])
    @test iszero(evaluate(nonmetricity(uniform), 1))
end

@testset "Covariant derivative: symbolic-R path" begin
    symbolics_available = try
        @eval using Symbolics
        symbolic_metric(signature_metric(VectorSpace(2), R, 2, 0, 0))
        true
    catch err
        @info "Symbolics.jl unavailable or extension failed; symbolic covariant test skipped." exception=(err, catch_backtrace())
        false
    end

    if symbolics_available
        S = Symbolics.Num
        m = symbolic_metric(signature_metric(VectorSpace(2), R, 2, 0, 0))
        graph = GraphBase(2, [(1, 2)]; metric = m)
        a = Symbolics.variable(:a)
        b = Symbolics.variable(:b)
        ψ = Field(graph, 0, Dict(1 => clifford_scalar(m, a), 2 => clifford_scalar(m, b)))
        @test isequal_simplified(evaluate(∇(ψ), 1), clifford_scalar(m, b - a))
    end
end
