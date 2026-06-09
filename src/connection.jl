# ── Phase L8.1 (Tier 2): connections, holonomy, and derived R/T/Q ───────────
#
# MATHEMATICAL CONTRACT
#
# DESIGN.md §15.1/§15.2 settles the architecture implemented here:
#
#   * Two first-class transport realizations, not one unified object:
#       - VersorTransport (defined in base_space.jl): two-sided geometric/frame
#         transport `M ↦ V M V⁻¹`, grade- and metric-preserving.
#       - GaugeTransport (below): one-sided representation/gauge transport
#         `ψ ↦ U ψ`.  It is a connection on a distinct bundle and must not be
#         forced through conjugation.
#
#   * The connection/potential is primary.  Holonomy is the primitive lossless
#     loop observable: an ordered, oriented, based list of edges is composed into
#     a transport map.  R/T/Q are derived views, never stored/cached state.
#
#   * Curvature and torsion are 2-cell Field-valued convenience views derived
#     from face holonomy.  Curvature is explicitly winding-lossy (a rotor log is
#     multivalued); holonomy remains the canonical observable.  Nonmetricity is
#     separate: a matrix-valued 1-cell MetricVariationField of inter-node metric
#     variation, not holonomy.

# ── One-sided gauge / representation transport ───────────────────────────────

"""
    GaugeTransport{R}(multiplier::CliffordTensor{R}, inverse::CliffordTensor{R})
    GaugeTransport(multiplier::CliffordTensor{R})

One-sided gauge/representation transport `ψ ↦ U·ψ`.

This is intentionally distinct from [`VersorTransport`](@ref).  A geometric/frame
transport acts two-sided by conjugation (`M ↦ V M V⁻¹`) because it preserves grade
and the Clifford quadratic form.  A gauge transport acts on a representation from
one side (`ψ ↦ Uψ`), as in a phase action `ψ ↦ exp(iθ)ψ`; forcing it through
conjugation would erase the distinction DESIGN.md §15.1 deliberately preserves.

The two transport kinds share the callable / `inv` / `∘` interface:

  - `(g)(ψ)  = U·ψ`,
  - `inv(g) = GaugeTransport(U⁻¹, U)`,
  - `(a ∘ b) = GaugeTransport(UₐU_b, U_b⁻¹Uₐ⁻¹)` (apply `b` then `a`).

Even-subalgebra or unitary restrictions are realization/model optimizations, not
interface constraints.
"""
struct GaugeTransport{R}
    multiplier :: CliffordTensor{R}
    inverse    :: CliffordTensor{R}
end

GaugeTransport(u::CliffordTensor{R}) where R = GaugeTransport{R}(u, inv_mv(u))

"""
    identity_gauge_transport(metric::Metric{R}) -> GaugeTransport{R}

The trivial one-sided gauge transport `ψ ↦ ψ`, represented by left multiplication
by the scalar Clifford unit.
"""
identity_gauge_transport(m::Metric{R}) where R =
    (o = clifford_one(m); GaugeTransport{R}(o, o))

(g::GaugeTransport{R})(x::CliffordTensor{R}) where R = g.multiplier * x

Base.inv(g::GaugeTransport{R}) where R = GaugeTransport{R}(g.inverse, g.multiplier)

# Apply `b` first, then `a`, matching Base's function-composition convention.
Base.:∘(a::GaugeTransport{R}, b::GaugeTransport{R}) where R =
    GaugeTransport{R}(a.multiplier * b.multiplier, b.inverse * a.inverse)

Base.:(==)(a::GaugeTransport{R}, b::GaugeTransport{R}) where R =
    a.multiplier == b.multiplier && a.inverse == b.inverse

Base.hash(g::GaugeTransport, h::UInt) = hash(g.multiplier, hash(g.inverse, h))

# ── Dedicated non-tensor derived-view containers ─────────────────────────────

"""
    HolonomyField{R,T,B}

A lightweight derived view over 2-cells whose values are transport maps (`T`,
e.g. [`VersorTransport`](@ref) or [`GaugeTransport`](@ref)), not fibre tensor
elements.  It intentionally is **not** a [`Field`](@ref): L7 `Field`s remain
sections valued in `AbstractTensorElement`s.  Use [`evaluate`](@ref) / `hf[cell]`
to read the face holonomy map, or call [`holonomy`](@ref) directly on a loop.
"""
struct HolonomyField{R,T,B<:BaseSpace}
    base    :: B
    grade   :: Int
    values  :: Dict{Int,T}
    default :: T
end

HolonomyField(base::B, values::Dict{Int,T}, default::T) where {B<:BaseSpace,T} =
    HolonomyField{_transport_ring(default),T,B}(base, 2, values, default)

"""
    MetricVariationField{R,B}

A lightweight matrix-valued edge view for nonmetricity `Q`.  Values are
`Matrix{R}` differences of local metric bilinear forms, so this is not a fibre
`Field`.  Access mirrors `Field`: [`evaluate`](@ref) / `q[edge]`, iteration over
stored nonzero entries, and `field_grade(q) == 1`.
"""
struct MetricVariationField{R,B<:BaseSpace}
    base    :: B
    grade   :: Int
    values  :: Dict{Int,Matrix{R}}
    default :: Matrix{R}
end

MetricVariationField(base::B, values::Dict{Int,Matrix{R}}, default::Matrix{R}) where {R,B<:BaseSpace} =
    MetricVariationField{R,B}(base, 1, values, default)

function evaluate(hf::HolonomyField{R,T,B}, cell::Integer) where {R,T,B}
    c = Int(cell)
    haskey(hf.values, c) ? hf.values[c] : hf.default
end

function evaluate(q::MetricVariationField{R,B}, cell::Integer) where {R,B}
    c = Int(cell)
    haskey(q.values, c) ? q.values[c] : q.default
end

Base.getindex(hf::HolonomyField, cell::Integer) = evaluate(hf, cell)
Base.getindex(q::MetricVariationField, cell::Integer) = evaluate(q, cell)

field_grade(hf::HolonomyField) = hf.grade
field_grade(q::MetricVariationField) = q.grade

for T in (:HolonomyField, :MetricVariationField)
    @eval begin
        Base.length(x::$T) = length(x.values)
        Base.keys(x::$T) = keys(x.values)
        Base.values(x::$T) = Base.values(x.values)
        Base.pairs(x::$T) = pairs(x.values)
        Base.haskey(x::$T, c::Integer) = haskey(x.values, Int(c))
        Base.iterate(x::$T, st...) = iterate(x.values, st...)
    end
end

Base.eltype(::Type{HolonomyField{R,T,B}}) where {R,T,B} = Pair{Int,T}
Base.eltype(::Type{MetricVariationField{R,B}}) where {R,B} = Pair{Int,Matrix{R}}

# ── Oriented edge parsing and common transport helpers ────────────────────────

_oriented_edge(spec::Integer) = (abs(Int(spec)), Int(spec) < 0 ? -1 : +1)

function _oriented_edge(spec::Tuple{<:Integer,<:Integer})
    e, s = spec
    ss = Int(s)
    (ss == 1 || ss == -1) || throw(ArgumentError(
        "oriented edge sign must be ±1, got $s for edge $e"))
    (Int(e), ss)
end

_oriented_transport(b::BaseSpace, spec) = begin
    e, s = _oriented_edge(spec)
    τ = transport(b, e)
    s == 1 ? τ : inv(τ)
end

_transport_ring(::VersorTransport{R}) where R = R
_transport_ring(::GaugeTransport{R}) where R = R

_transport_multiplier(t::VersorTransport) = t.versor
_transport_multiplier(t::GaugeTransport)  = t.multiplier

_scalar_part(A::CliffordTensor{R}) where R = get(A.terms, Int[], zero(R))

# ── Holonomy: primitive loop observable ───────────────────────────────────────

"""
    holonomy(b::BaseSpace, loop) -> transport map

Compose the transports around an ordered, oriented, based loop.

`loop` is an ordered iterable of oriented edges.  Each entry may be either
`edge_id` (positive = traverse the base edge orientation, negative = traverse it
backward) or `(edge_id, sign)` with `sign ∈ {+1,-1}`.  For a loop
`[e₁, e₂, …, eₙ]`, the result is

```julia
τₙ ∘ … ∘ τ₂ ∘ τ₁
```

where an edge traversed against its stored orientation contributes `inv(τᵢ)`.
Composition is generally non-abelian: order and orientation matter.  Rebasing a
closed loop conjugates its holonomy, so the holonomy map itself is basepoint-
convention-dependent while [`holonomy_trace`](@ref) is the conjugacy invariant.

This primitive is defined on every base with edge transport, including a bare
[`GraphBase`](@ref) with no faces.
"""
function holonomy(b::BaseSpace, loop)
    specs = collect(loop)
    isempty(specs) && throw(ArgumentError(
        "holonomy requires a non-empty ordered loop of oriented edges"))
    acc = _oriented_transport(b, first(specs))
    for spec in specs[2:end]
        acc = _oriented_transport(b, spec) ∘ acc
    end
    acc
end

"""
    holonomy_trace(b::BaseSpace, loop) -> scalar

Scalar invariant extracted from [`holonomy`](@ref), with transport-kind-specific
meaning.

For a geometric [`VersorTransport`](@ref), this returns the grade-0 coefficient
of the composed versor multiplier `V` (for a simple rotor `cos θ + B sin θ`, this
is `cos θ`).  It is the scalar part of the versor, not the matrix trace of the
conjugation operator `M ↦ V M V⁻¹`.  For a one-sided [`GaugeTransport`](@ref), it
returns the grade-0 coefficient of the composed gauge multiplier `U`, i.e. the
phase/representation scalar carried by the one-sided action.

The full holonomy depends on the ordered based loop and changes by conjugation
under rebasing; these scalar multiplier parts are invariant under that rebasing
for the shipped Clifford-valued transport realizations.
"""
holonomy_trace(b::BaseSpace, loop) = _scalar_part(_transport_multiplier(holonomy(b, loop)))

"""
    holonomy_field(b::BaseSpace) -> HolonomyField

Derived convenience view over 2-cells: evaluate [`holonomy`](@ref) on each face's
signed boundary loop using the base's boundary order as the canonical orientation
and basepoint convention.

This exists only for bases with `top_grade(b) ≥ 2`; on a graph/bare 1-complex it
throws an informative `ArgumentError`, because the primitive holonomy is loop-
valued and graph-valid, while this face view is merely convention-fixed.  The
returned [`HolonomyField`](@ref) is not a fibre [`Field`](@ref): its values are
transport maps, not `AbstractTensorElement`s.  The face maps themselves are
basepoint-convention-dependent; [`holonomy_trace`](@ref) gives the invariant
scalar content.
"""
function holonomy_field(b::B) where {B<:BaseSpace}
    top_grade(b) >= 2 || throw(ArgumentError(
        "holonomy_field requires 2-cells (top_grade(b) ≥ 2). Use holonomy(b, loop) " *
        "directly on a graph or other 1-complex."))
    fs = collect(cells(b, 2))
    isempty(fs) && throw(ArgumentError("holonomy_field requires at least one 2-cell"))
    first_h = holonomy(b, boundary(b, 2, first(fs)))
    T = typeof(first_h)
    vals = Dict{Int,T}(Int(first(fs)) => first_h)
    for f in fs[2:end]
        vals[Int(f)] = holonomy(b, boundary(b, 2, f))::T
    end
    HolonomyField(b, vals, first_h)
end

# ── Derived curvature and torsion views ───────────────────────────────────────

function _rotor_bivector_log(t::VersorTransport{R}) where R
    V = t.versor
    oneV = clifford_one(V.metric)
    iszero(V - oneV) && return clifford_zero(V.metric)

    Bpart = homogeneous_component(V, 2)
    iszero(Bpart) && return clifford_zero(V.metric)

    # Exact rings cannot represent a general rotor logarithm.  Return the
    # first-order bivector part as an in-ring local extraction; closed-form tests
    # use Float64, where the logarithm below recovers the generator.
    has_transcendentals(R) || return Bpart

    idxs = collect(keys(Bpart.terms))
    length(idxs) == 1 || throw(ArgumentError(
        "curvature log extraction currently supports simple single-plane rotors; " *
        "holonomy remains available as the lossless primitive"))
    idx = first(idxs)
    c = Bpart.terms[idx]
    blade = clifford_basis_element(V.metric, idx)
    q = _scalar_part(blade * blade)
    a = _scalar_part(V)

    if q < zero(R)              # circular rotor: cos θ + E sin θ
        θ = atan(c, a)
        return θ * blade
    elseif q > zero(R)          # hyperbolic rotor: cosh φ + E sinh φ
        return atanh(c / a) * blade
    else                        # nilpotent: exp(B) = 1 + B
        return Bpart
    end
end

_rotor_bivector_log(t::GaugeTransport) = throw(ArgumentError(
    "curvature(b) extracts Clifford bivectors from geometric/VersorTransport holonomy; " *
    "gauge holonomy is a separate one-sided representation connection"))

"""
    curvature(b::BaseSpace) -> Field

Derived curvature view over 2-cells, valued in Clifford bivectors.

For each face, this computes the face holonomy and extracts a bivector generator
`B` such that the holonomy rotor is locally `rotor_exp(B)` when that extraction is
in scope.  This extraction has three deliberate limitations:

  1. It is **winding-lossy and multivalued**: rotors with angles differing by
     `2π` have the same holonomy.
  2. Only simple single-plane rotors are supported by the closed-form extractor;
     compound/multi-plane rotors throw an `ArgumentError`.  Use [`holonomy`](@ref)
     directly for the lossless primitive in that case.
  3. On exact rings (for example `Rational{BigInt}`) there is no closed-form
     transcendental logarithm in the ring, so `curvature` returns the in-ring
     first-order bivector part of the holonomy multiplier, not a true rotor log.
     Use a transcendental-capable numeric ring (for example `Float64`) for the
     closed-form single-plane logarithm.

The holonomy returned by [`holonomy`](@ref) is therefore the canonical lossless
object; `curvature` is only a convenient derived Field and is never cached.

Requires `top_grade(b) ≥ 2`; use [`holonomy`](@ref) directly on graph loops.
"""
function curvature(b::B) where {B<:BaseSpace}
    top_grade(b) >= 2 || throw(ArgumentError(
        "curvature requires 2-cells/faces (top_grade(b) ≥ 2); graph holonomy is " *
        "available via holonomy(b, loop), but no face-derived curvature field exists."))
    fs = collect(cells(b, 2))
    isempty(fs) && throw(ArgumentError("curvature requires at least one 2-cell"))
    first_B = _rotor_bivector_log(holonomy(b, boundary(b, 2, first(fs))))
    R = eltype(first_B.metric.g)
    vals = Dict{Int,CliffordTensor{R}}()
    iszero(first_B) || (vals[Int(first(fs))] = first_B)
    for f in fs[2:end]
        Bf = _rotor_bivector_log(holonomy(b, boundary(b, 2, f)))
        iszero(Bf) || (vals[Int(f)] = Bf)
    end
    Field{R,CliffordTensor{R},B}(b, 2, vals)
end

function _edge_endpoints(b::BaseSpace, edge::Integer)
    bd = boundary(b, 1, edge)
    tail = nothing
    head = nothing
    for (v, s) in bd
        if s == -1
            tail = v
        elseif s == 1
            head = v
        end
    end
    (tail === nothing || head === nothing) && throw(ArgumentError(
        "edge $edge boundary must contain one tail sign -1 and one head sign +1"))
    (Int(tail), Int(head))
end

function _grid_vertex_coord(b::GridBase, v::Integer)
    z = Int(v) - 1
    width = b.nx + 1
    (z % width, z ÷ width)
end

function _edge_displacement(b::GridBase, edge::Integer)
    tail, head = _edge_endpoints(b, edge)
    ti, tj = _grid_vertex_coord(b, tail)
    hi, hj = _grid_vertex_coord(b, head)
    dx, dy = hi - ti, hj - tj
    dx * clifford_basis_vector(b.metric, 1) + dy * clifford_basis_vector(b.metric, 2)
end

_edge_displacement(b::BaseSpace, edge::Integer) = throw(ArgumentError(
    "torsion requires a base realization to provide edge displacement vectors; " *
    "GridBase supplies them, but $(typeof(b)) does not"))

"""
    torsion(b::BaseSpace) -> Field

Derived torsion/closure-failure view over 2-cells, valued in Clifford vectors.

For each face this sums the oriented edge displacement vectors around the face.
For the shipped flat cubical grid this is exactly zero on every face.  Like
curvature, this is a pure derived Field over face loops and is not cached.

Requires `top_grade(b) ≥ 2`; on a graph there are loop holonomies but no face
field, so an informative `ArgumentError` is thrown.
"""
function torsion(b::B) where {B<:BaseSpace}
    top_grade(b) >= 2 || throw(ArgumentError(
        "torsion requires 2-cells/faces (top_grade(b) ≥ 2); graph loops have " *
        "holonomy but no canonical face closure-failure field."))
    fs = collect(cells(b, 2))
    isempty(fs) && throw(ArgumentError("torsion requires at least one 2-cell"))
    d = fibre(b, 0, first(cells(b, 0)))
    zeroV = zero_fibre(d)
    R = eltype(zeroV.metric.g)
    vals = Dict{Int,CliffordTensor{R}}()
    for f in fs
        acc = zeroV
        for (edge, sign) in boundary(b, 2, f)
            dv = _edge_displacement(b, edge)
            acc = acc + (sign == 1 ? dv : -dv)
        end
        iszero(acc) || (vals[Int(f)] = acc)
    end
    Field{R,CliffordTensor{R},B}(b, 2, vals)
end

"""
    nonmetricity(b::BaseSpace) -> MetricVariationField

Separate nonmetricity view over 1-cells from inter-node metric variation.

For an oriented edge `u → v`, this returns the symmetric bilinear-form difference

```julia
Q[e] = metric(b, 0, v).g - metric(b, 0, u).g
```

as a dedicated matrix-valued [`MetricVariationField`](@ref) over edges.  It is
not a fibre [`Field`](@ref), because a raw bilinear-form matrix is not an
`AbstractTensorElement`.  This view does **not** use holonomy: per DESIGN.md
§15.1/§15.2, R/T come from edge-loop transport, while Q comes from node/local
metric variation.  Requires `has_metric(b) == true`; a bare graph throws the
documented capability-gating `ArgumentError`.
"""
function nonmetricity(b::B) where {B<:BaseSpace}
    has_metric(b) || throw(ArgumentError(
        "nonmetricity requires the metric capability (has_metric(b) == true): " *
        "Q is inter-node metric variation, not holonomy. A bare graph has no local metrics."))
    es = collect(cells(b, 1))
    isempty(es) && throw(ArgumentError("nonmetricity requires at least one 1-cell"))
    m0 = metric(b, 0, first(cells(b, 0))).g
    R = eltype(m0)
    default = zeros(R, size(m0))
    vals = Dict{Int,Matrix{R}}()
    for e in es
        tail, head = _edge_endpoints(b, e)
        diff = metric(b, 0, head).g - metric(b, 0, tail).g
        iszero(diff) || (vals[Int(e)] = diff)
    end
    MetricVariationField(b, vals, default)
end

export GaugeTransport, identity_gauge_transport,
       HolonomyField, MetricVariationField,
       holonomy, holonomy_trace, holonomy_field,
       curvature, torsion, nonmetricity
