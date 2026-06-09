# ── Phase L7: Base space / bundle interface ──────────────────────────────────
#
# MATHEMATICAL CONTRACT
#
# A `BaseSpace` is the *bundle* over which fields live: it supplies the
# topology (cells by grade), the orientation-carrying boundary operator (the
# discrete exterior derivative `d`), the fibre attached at each cell, and the
# transport of fibre elements along oriented 1-cells (the discrete connection).
# A `Field` (see field.jl) is a *section* of that bundle — it assigns a fibre
# element to each k-cell.  Keep STATE in the field; keep STRUCTURE in the base.
#
# This file is the authoritative L7 interface charter (DESIGN.md §13).  It is a
# documented, semver-governed EXTENSION SURFACE (DESIGN.md §8): third parties
# add a new base by subtyping `BaseSpace` and implementing the four obligations
# (and, optionally, the two capabilities), and add a new fibre by defining a new
# `FibreDescriptor` plus its `fibre_eltype` / `zero_fibre` hooks.  No existing
# code changes.
#
# FOUR OBLIGATIONS (every base must implement)
#
#   1. Cell enumeration by grade   — top_grade / cells / n_cells
#   2. Signed boundary (= d)        — boundary  (orientation lives in ±1 signs)
#   3. Fibre attachment             — fibre     (returns a FibreDescriptor)
#   4. Transport along a 1-cell     — transport (forward tail→head map; inv = reverse)
#
# TWO OPTIONAL CAPABILITIES (default off)
#
#   A. Pointwise metric  — has_metric / metric / signature   (LOCAL, per fibre)
#   B. Dual complex      — has_dual_complex                   (GLOBAL/topological)
#
# THE TWO-TRAIT GATING (state it now; `⋆`/`δ` are built in L8)
#
#   * Metric-FREE structure — boundary (`d`), cell enumeration, the natural
#     pairing/contraction — works on EVERY base.
#   * `metric` / `signature` require `has_metric`; calling them on a base with
#     `has_metric == false` throws a clear `ArgumentError`.
#   * The Hodge star `⋆` and codifferential `δ` (L8) require BOTH `has_metric`
#     AND `has_dual_complex`.  A base with a metric but no dual complex (a bare
#     weighted graph) honestly CANNOT Hodge-dualize — that is correct behaviour,
#     not a gap.  `can_hodge(b)` reports the conjunction (see its docstring).
#
# Capability B is STRICTLY STRONGER than A in intent: a dual complex is the
# global/topological structure `⋆` needs on top of the local metric.

# ── The abstract type ─────────────────────────────────────────────────────────

"""
    BaseSpace

Abstract supertype of every base space (the *bundle* a [`Field`](@ref) is a
section of).  A base supplies four obligations — cell enumeration by grade
([`top_grade`](@ref)/[`cells`](@ref)/[`n_cells`](@ref)), the signed boundary
operator [`boundary`](@ref) (which *is* the discrete exterior derivative `d`),
[`fibre`](@ref) attachment, and [`transport`](@ref) along oriented 1-cells —
plus two optional capabilities, a pointwise [`metric`](@ref) (gated by
[`has_metric`](@ref)) and a dual complex (gated by [`has_dual_complex`](@ref)).

See the file header for the full contract.  Concrete realizations shipped with
Tensorsmith: [`GraphBase`](@ref), [`GridBase`](@ref), [`ManifoldChartBase`](@ref).

# Extending
Subtype `BaseSpace` and implement `top_grade`, `cells`, `_boundary` (the
interior of `boundary`; see [`boundary`](@ref)), `fibre`, and `transport`.
Override `has_metric`/`metric`/`signature` and/or `has_dual_complex` to opt into
the capabilities.
"""
abstract type BaseSpace end

# ── Obligation 1: cells by grade ──────────────────────────────────────────────

"""
    top_grade(b::BaseSpace) -> Int

The highest cell dimension present in `b`.  A graph stops at `1` (nodes and
edges); a 2D cell complex returns `2`; an n-grid returns `n`.
"""
function top_grade end

"""
    cells(b::BaseSpace, k::Int)

Iterator over the ids of the `k`-cells of `b` (cell ids are `Int`).  Returns an
empty range for `k < 0` or `k > top_grade(b)`.  A *k-field* assigns fibre
elements to exactly these cells.
"""
function cells end

"""
    n_cells(b::BaseSpace, k::Int) -> Int

The number of `k`-cells.  Defaults to `length(cells(b, k))`; override if a base
can report the count more cheaply.
"""
n_cells(b::BaseSpace, k::Integer) = length(cells(b, Int(k)))

# ── Obligation 2: signed boundary (the discrete d) ────────────────────────────

"""
    boundary(b::BaseSpace, k::Int, cell) -> iterable of (face_id::Int, sign::Int)

The signed boundary of the `k`-cell `cell`: the `(k−1)`-cells that bound it,
each paired with an orientation sign `±1`.  **This operator is the discrete
exterior derivative `d`** (its transpose is the coboundary), so `d` — and
grad/curl/div as its grade shadows — come for free on any base implementing it.

**Orientation lives ENTIRELY in these signs.** There is no separate orientation
system: a non-orientable complex is handled by a base honestly reporting that
consistent signs are unavailable, never by a parallel mechanism that could
disagree.

By contract the boundary is **empty for `k == 0`** (a node has no faces) and for
`k > top_grade(b)`.  These two cases are enforced centrally here; concrete bases
implement the interior cases in [`_boundary`](@ref) and never need to special-case
the bounds.

The defining law `d² = 0` (boundary-of-boundary cancels with the signs) is a
property each base must satisfy and is verified by the test suite.
"""
function boundary(b::BaseSpace, k::Integer, cell)
    kk = Int(k)
    (kk <= 0 || kk > top_grade(b)) && return Tuple{Int,Int}[]
    _boundary(b, kk, cell)
end

"""
    _boundary(b::BaseSpace, k::Int, cell) -> iterable of (face_id::Int, sign::Int)

The interior of [`boundary`](@ref): the signed faces of a `k`-cell for
`1 ≤ k ≤ top_grade(b)`.  Concrete bases implement THIS method; the public
`boundary` wraps it and guarantees the contractual empty result for `k == 0`
and `k > top_grade`.
"""
function _boundary end

# ── Obligation 3: fibre attachment ────────────────────────────────────────────

"""
    fibre(b::BaseSpace, k::Int, cell) -> FibreDescriptor

The fibre algebra carried by the `k`-cell `cell`, returned as a concrete,
lightweight [`FibreDescriptor`](@ref) — **not** a live element and **not** a bare
`DataType`.  The element type and the zero/one constructors are recovered from
the descriptor by dispatch ([`fibre_eltype`](@ref), [`zero_fibre`](@ref)), which
keeps callers (notably [`Field`](@ref) access) type-stable.
"""
function fibre end

# ── Obligation 4: transport along a 1-cell ────────────────────────────────────

"""
    transport(b::BaseSpace, edge) -> a map fibre(tail) -> fibre(head)

The forward (tail → head) transport of a fibre element across the oriented
1-cell `edge` — the discrete connection.  The returned object is *callable*:
`transport(b, e)(x)` carries the fibre element `x` to the head frame.

**Reverse transport is `inv(τ)`** — there is no separate reverse-direction
argument, because orientation already lives in the edge / incidence signs and a
second system could disagree with them.

The load-bearing invariant is **path-associativity**: transports compose
associatively along paths (`τ₃ ∘ τ₂ ∘ τ₁`), so the holonomy of a closed loop is
well-defined and basis-independent.  Loop holonomy is what makes the three
failure modes of a general affine connection measurable (rotational holonomy =
curvature, translational closure failure = torsion, magnitude drift =
nonmetricity); extracting them is L8, but the interface that carries them is here.

See [`VersorTransport`](@ref) for the shipped Clifford-versor realization
(`x ↦ V x V⁻¹`), which is exactly path-associative because the geometric product
is associative.
"""
function transport end

# ── Capability A: pointwise metric (LOCAL) ────────────────────────────────────

"""
    has_metric(b::BaseSpace) -> Bool

Whether `b` provides the optional, *local*, *derived* metric capability
(DESIGN.md §13).  Defaults to `false` — a bare topological/graph base.  A base
that returns `true` MUST implement [`metric`](@ref) and [`signature`](@ref).

Parallels the scalar-ring traits `has_sqrt` / `has_transcendentals`
(scalar_ring.jl): a capability is *declared*, and the dependent operations are
gated on the declaration rather than failing obscurely.
"""
has_metric(::BaseSpace) = false

"""
    metric(b::BaseSpace, k::Int, cell) -> Metric

The bilinear form on the fibre attached at the `k`-cell `cell`.  The metric is
**local** (per cell/region), so a metric *gradient* between regions is
representable — a boundary between metric regions is exactly nonmetricity
`∇g ≠ 0` (DESIGN.md §13).

Requires the metric capability: calling this on a base with
`has_metric(b) == false` throws an `ArgumentError`.
"""
metric(b::BaseSpace, k::Integer, cell) = throw(ArgumentError(
    "$(typeof(b)) does not provide a metric (has_metric == false). The metric " *
    "is an optional capability; `metric`/`signature` require has_metric(b) == true. " *
    "A bare graph honestly has no fibre metric."))

"""
    signature(b::BaseSpace) -> Tuple{Int,Int,Int}

The declared signature `(p, q, r)` (positive, negative, null directions) of the
fibre metric — base-declarable, so a Lorentzian `(1, 3, 0)` spacetime fibre is
as valid as a Euclidean one (reuses the `Cl(p,q,r)` signature convention).

Requires the metric capability: throws an `ArgumentError` when
`has_metric(b) == false`.
"""
signature(b::BaseSpace) = throw(ArgumentError(
    "$(typeof(b)) does not provide a metric (has_metric == false), so it has no " *
    "signature. `signature` requires has_metric(b) == true."))

# ── Capability B: dual complex (GLOBAL) ───────────────────────────────────────

"""
    has_dual_complex(b::BaseSpace) -> Bool

Whether `b` carries the global/topological dual complex needed (on top of a
local metric) to form the Hodge star.  Defaults to `false`.  Strictly stronger
in intent than [`has_metric`](@ref): a bare weighted graph may have a per-fibre
metric yet no dual complex, and therefore cannot Hodge-dualize.
"""
has_dual_complex(::BaseSpace) = false

"""
    can_hodge(b::BaseSpace) -> Bool

Whether the Hodge star `⋆` and codifferential `δ` (to be built in L8) are
available on `b`.  Both require BOTH capabilities:

    can_hodge(b) == has_metric(b) && has_dual_complex(b)

This is the documented L8 precondition, surfaced now so the trait contract is
unambiguous: a base with a metric but no dual complex (e.g. [`GraphBase`](@ref))
returns `false`, and that is correct — Hodge duality genuinely does not exist
there.  Metric-free operators (`d`, incidence, the natural pairing) never consult
this.
"""
can_hodge(b::BaseSpace) = has_metric(b) && has_dual_complex(b)

# ── Fibre descriptors ─────────────────────────────────────────────────────────
#
# A descriptor is a concrete, lightweight value naming the fibre algebra at a
# cell.  The element type and its zero/one are recovered by DISPATCH on the
# descriptor — never by returning a bare `DataType` — so `Field` access stays
# type-stable (the element type is a type parameter the compiler can infer).

"""
    FibreDescriptor

Abstract supertype of the lightweight, concrete descriptors returned by
[`fibre`](@ref).  A descriptor names a fibre algebra; the associated element type
and constructors are recovered by dispatch:

  - [`fibre_eltype(d)`](@ref) → the concrete `AbstractTensorElement` subtype,
  - [`zero_fibre(d)`](@ref) / [`one_fibre(d)`](@ref) → its additive / scalar unit,
  - [`fibre_matches(d, x)`](@ref) → whether element `x` belongs to fibre `d`.

# Extending
Add a fibre by defining a new `FibreDescriptor` struct and the three hooks above
for it.  Shipped descriptors: [`CliffordFibre`](@ref), [`TensorFibre`](@ref).
"""
abstract type FibreDescriptor end

"""
    CliffordFibre{R}(metric::Metric{R})

Descriptor for a fibre that is the Clifford algebra `Cl(V, g)` of `metric`.  Its
elements are [`CliffordTensor{R}`](@ref).  This is the uniform `Cl(3)` fibre of
the QRCS-shaped [`GraphBase`](@ref) and the fibre of the other shipped bases.
"""
struct CliffordFibre{R} <: FibreDescriptor
    metric :: Metric{R}
end

"""
    TensorFibre{R}(space::VectorSpace)

Descriptor for a fibre of mixed-variance tensors over `space`; its elements are
[`MixedTensor{R}`](@ref).  Shipped to demonstrate that the descriptor mechanism
generalizes beyond Clifford fibres (DESIGN.md §8 extension surface).
"""
struct TensorFibre{R} <: FibreDescriptor
    space :: VectorSpace
end

"""
    fibre_eltype(d::FibreDescriptor) -> Type{<:AbstractTensorElement}

The concrete element type living in the fibre `d`, recovered by dispatch.  Must
be inferrable from `typeof(d)` alone (the descriptor carries the ring `R` as a
type parameter), which is what keeps [`Field`](@ref) access type-stable.
"""
fibre_eltype(::CliffordFibre{R}) where R = CliffordTensor{R}
fibre_eltype(::TensorFibre{R})   where R = MixedTensor{R}

"""
    zero_fibre(d::FibreDescriptor) -> AbstractTensorElement

The additive identity (zero element) of the fibre `d`, of type `fibre_eltype(d)`.
"""
zero_fibre(d::CliffordFibre{R}) where R = clifford_zero(d.metric)
zero_fibre(d::TensorFibre{R})   where R = mixed_zero(d.space, R)

"""
    one_fibre(d::FibreDescriptor) -> AbstractTensorElement

The multiplicative identity (scalar `1`) of the fibre `d`, of type
`fibre_eltype(d)`.
"""
one_fibre(d::CliffordFibre{R}) where R = clifford_one(d.metric)
one_fibre(d::TensorFibre{R})   where R = mixed_one(d.space, R)

"""
    fibre_matches(d::FibreDescriptor, x) -> Bool

Whether the element `x` genuinely belongs to the fibre described by `d` — used by
[`Field`](@ref) to validate sections honestly.  For a Clifford fibre this checks
both the element type and that `x` carries the descriptor's metric; the default
checks the element type only.
"""
fibre_matches(d::FibreDescriptor, x) = x isa fibre_eltype(d)
# `===` short-circuit first: when the element already carries the descriptor's
# own metric object (the common case) we avoid invoking `Metric ==`, which over a
# symbolic ring would compare `Num` matrices in a boolean context.
fibre_matches(d::CliffordFibre{R}, x::CliffordTensor{R}) where R =
    x.metric === d.metric || x.metric == d.metric
fibre_matches(d::TensorFibre{R}, x::MixedTensor{R}) where R =
    x.space === d.space || x.space == d.space

# ── Transport realization: Clifford versor conjugation ────────────────────────
#
# A connection that carries a fibre element across an edge by the sandwich
# `x ↦ V x V⁻¹` for an invertible versor `V`.  Conjugations compose to a
# conjugation (`W(VxV⁻¹)W⁻¹ = (WV)x(WV)⁻¹`) and invert to a conjugation, so the
# transport is EXACTLY path-associative — it inherits associativity from the
# associative geometric product, with no separate bookkeeping.

"""
    VersorTransport{R}(versor::CliffordTensor{R}, inverse::CliffordTensor{R})
    VersorTransport(versor::CliffordTensor{R})

A transport map realized as Clifford versor conjugation `x ↦ V x V⁻¹` (the
shipped [`transport`](@ref) realization, used by all three concrete bases).

The one-argument constructor computes `V⁻¹` via [`inv_mv`](@ref) (so `versor`
must be a versor/blade; a non-versor throws).  Callable on a `CliffordTensor` in
the same algebra; composes with `∘` and reverses with `inv`:

  - `(t)(x)            = V x V⁻¹`
  - `inv(t)            = VersorTransport(V⁻¹, V)`            (reverse transport)
  - `(a ∘ b)           = VersorTransport(Vₐ·V_b, V_b⁻¹·Vₐ⁻¹)` (apply `b` then `a`)

Composition is associative because the geometric product is, which is exactly
what makes loop holonomy well-defined.
"""
struct VersorTransport{R}
    versor  :: CliffordTensor{R}
    inverse :: CliffordTensor{R}
end

VersorTransport(v::CliffordTensor{R}) where R = VersorTransport{R}(v, inv_mv(v))

"""
    identity_transport(metric::Metric{R}) -> VersorTransport{R}

The trivial (flat) transport `x ↦ x`, i.e. conjugation by the scalar `1`.  The
default connection of a flat base.
"""
identity_transport(m::Metric{R}) where R =
    (o = clifford_one(m); VersorTransport{R}(o, o))

(t::VersorTransport{R})(x::CliffordTensor{R}) where R = t.versor * x * t.inverse

Base.inv(t::VersorTransport{R}) where R = VersorTransport{R}(t.inverse, t.versor)

# Apply `b` first, then `a` (function-composition order), matching `∘` on maps.
Base.:∘(a::VersorTransport{R}, b::VersorTransport{R}) where R =
    VersorTransport{R}(a.versor * b.versor, b.inverse * a.inverse)

Base.:(==)(a::VersorTransport{R}, b::VersorTransport{R}) where R =
    a.versor == b.versor && a.inverse == b.inverse

Base.hash(t::VersorTransport, h::UInt) = hash(t.versor, hash(t.inverse, h))

# =============================================================================
# REALIZATION 1 — GraphBase (the QRCS-shaped case)
# =============================================================================
#
# Nodes (0-cells) and oriented edges (1-cells); top_grade = 1.  Boundary from
# node–edge incidence: ∂(edge) = head − tail (the discrete d of a 0-cochain).
# Uniform Clifford fibre (default Cl(3,0)) attached to every cell.  Transport is
# a per-edge versor/rotor, identity where unspecified.  No metric capability and
# no dual complex: a bare graph cannot Hodge-dualize, and that is correct.

"""
    GraphBase(n_nodes, edges; metric = signature_metric(VectorSpace(3), ExactRing, 3,0,0),
              versors = Dict{Int,CliffordTensor}())

A base of `n_nodes` nodes and oriented `edges` (a vector of `(tail, head)`
node-id pairs, 1-based).  `top_grade == 1`.

Fibre: the uniform Clifford algebra of `metric` on every node and edge.
Transport: `versors[e]` (a versor in that algebra) on edge `e`, the identity
elsewhere.  This is the QRCS-shaped realization — it works **without faces and
without Hodge duality**: `has_metric == false` and `has_dual_complex == false`.
"""
# Extract the scalar ring `R` from a metric value at runtime, so a base's `R`
# type parameter need not be bound through a keyword argument (fragile pattern).
_ring(::Metric{R}) where R = R

struct GraphBase{R} <: BaseSpace
    n_nodes :: Int
    edges   :: Vector{Tuple{Int,Int}}
    metric  :: Metric{R}
    versors :: Dict{Int, CliffordTensor{R}}
end

function GraphBase(n_nodes::Integer, edges::Vector{Tuple{Int,Int}};
                   metric::Metric = signature_metric(VectorSpace(3), ExactRing, 3, 0, 0),
                   versors = nothing)
    R    = _ring(metric)
    vers = versors === nothing ? Dict{Int,CliffordTensor{R}}() :
                                 Dict{Int,CliffordTensor{R}}(versors)
    n_nodes >= 0 || throw(ArgumentError("n_nodes must be ≥ 0, got $n_nodes"))
    for (e, (t, h)) in enumerate(edges)
        (1 <= t <= n_nodes && 1 <= h <= n_nodes) || throw(ArgumentError(
            "edge $e = ($t, $h) references a node outside 1:$n_nodes"))
    end
    for (e, v) in vers
        (1 <= e <= length(edges)) || throw(ArgumentError(
            "versor key $e is not an edge id in 1:$(length(edges))"))
        (v.metric === metric || v.metric == metric) || throw(ArgumentError(
            "versor on edge $e lives in a different Clifford algebra than the fibre metric"))
    end
    GraphBase{R}(Int(n_nodes), edges, metric, vers)
end

top_grade(::GraphBase) = 1

cells(b::GraphBase, k::Integer) =
    k == 0 ? (1:b.n_nodes) :
    k == 1 ? (1:length(b.edges)) : (1:0)

n_cells(b::GraphBase, k::Integer) =
    k == 0 ? b.n_nodes : k == 1 ? length(b.edges) : 0

# ∂(edge) = head − tail.
function _boundary(b::GraphBase, k::Int, edge::Integer)
    # k == 1 only (the wrapper has already excluded k == 0 and k > 1).
    (1 <= edge <= length(b.edges)) || throw(ArgumentError(
        "edge id $edge out of range 1:$(length(b.edges))"))
    t, h = b.edges[edge]
    Tuple{Int,Int}[(t, -1), (h, +1)]
end

fibre(b::GraphBase{R}, k::Integer, cell) where R = CliffordFibre{R}(b.metric)

function transport(b::GraphBase{R}, edge::Integer) where R
    (1 <= edge <= length(b.edges)) || throw(ArgumentError(
        "edge id $edge out of range 1:$(length(b.edges))"))
    haskey(b.versors, edge) ? VersorTransport(b.versors[edge]) :
                              identity_transport(b.metric)
end

# has_metric / has_dual_complex use the BaseSpace defaults (both false), so
# `metric`/`signature` fall through to the gating errors — by design.

# =============================================================================
# REALIZATION 2 — GridBase (a structured cubical cell complex)
# =============================================================================
#
# A regular Nx×Ny grid of square 2-cells: 0-cells (vertices), 1-cells (edges),
# 2-cells (faces); top_grade = 2.  Signed incidence is the standard tensor-product
# cubical boundary, precomputed at construction so `d² = 0` holds by construction
# (∂∂face cancels termwise — verified in the suite).  Uniform Clifford fibre;
# flat transport (identity).  has_metric = true (flat fibre geometry) and
# has_dual_complex = true (a cubical complex has a well-defined dual).

"""
    GridBase(nx, ny; metric = signature_metric(VectorSpace(2), ExactRing, 2,0,0))

A structured 2D grid of `nx × ny` square cells (so `(nx+1)×(ny+1)` vertices,
horizontal+vertical edges, and `nx*ny` faces).  `top_grade == 2`.

Cell ids are contiguous per grade: vertices `1:(nx+1)(ny+1)`, edges
`1:n_edges` (all horizontal edges first, then vertical), faces `1:nx*ny`.
Signed incidence is the cubical boundary (oriented `bottom + right − top − left`
for a face, `head − tail` for an edge), precomputed so `d² = 0` holds exactly.

Fibre: the uniform Clifford algebra of `metric` (default flat Euclidean
`Cl(2,0)`) on every cell.  Transport: identity (flat).  `has_metric == true`,
`has_dual_complex == true`, so `can_hodge == true`.
"""
struct GridBase{R} <: BaseSpace
    nx :: Int
    ny :: Int
    nv :: Int
    ne :: Int
    nf :: Int
    edge_boundary :: Vector{Vector{Tuple{Int,Int}}}   # by edge id
    face_boundary :: Vector{Vector{Tuple{Int,Int}}}   # by face id
    metric :: Metric{R}
    sig    :: Tuple{Int,Int,Int}
end

function GridBase(nx::Integer, ny::Integer;
                  metric::Metric = signature_metric(VectorSpace(2), ExactRing, 2, 0, 0))
    R = _ring(metric)
    (nx >= 1 && ny >= 1) || throw(ArgumentError(
        "GridBase needs nx ≥ 1 and ny ≥ 1, got ($nx, $ny)"))

    vid(i, j) = i + j * (nx + 1) + 1                  # i∈0:nx, j∈0:ny
    nh = nx * (ny + 1)                                # horizontal edge count
    eid_h(i, j) = i + j * nx + 1                      # i∈0:nx-1, j∈0:ny
    eid_v(i, j) = nh + i + j * (nx + 1) + 1           # i∈0:nx,   j∈0:ny-1
    fid(i, j)   = i + j * nx + 1                      # i∈0:nx-1, j∈0:ny-1

    nv = (nx + 1) * (ny + 1)
    ne = nh + (nx + 1) * ny
    nf = nx * ny

    edge_boundary = Vector{Vector{Tuple{Int,Int}}}(undef, ne)
    # Horizontal edge (i,j)->(i+1,j):  ∂ = +(i+1,j) − (i,j)
    for j in 0:ny, i in 0:nx-1
        edge_boundary[eid_h(i, j)] = Tuple{Int,Int}[(vid(i, j), -1), (vid(i + 1, j), +1)]
    end
    # Vertical edge (i,j)->(i,j+1):    ∂ = +(i,j+1) − (i,j)
    for j in 0:ny-1, i in 0:nx
        edge_boundary[eid_v(i, j)] = Tuple{Int,Int}[(vid(i, j), -1), (vid(i, j + 1), +1)]
    end

    face_boundary = Vector{Vector{Tuple{Int,Int}}}(undef, nf)
    # Face with lower-left corner (i,j), CCW: bottom + right − top − left.
    for j in 0:ny-1, i in 0:nx-1
        bottom = eid_h(i, j)
        top    = eid_h(i, j + 1)
        left   = eid_v(i, j)
        right  = eid_v(i + 1, j)
        face_boundary[fid(i, j)] = Tuple{Int,Int}[
            (bottom, +1), (right, +1), (top, -1), (left, -1)]
    end

    GridBase{R}(Int(nx), Int(ny), nv, ne, nf, edge_boundary, face_boundary,
                metric, _signature_of(metric))
end

# Read a diagonal metric's (p,q,r) signature off its diagonal.
# Use `iszero` / `isequal(d, one(R))` rather than `==` so this stays safe
# when R = Symbolics.Num (where `Num == Num` is not a Bool).
function _signature_of(m::Metric{R}) where R
    p = q = r = 0
    for i in 1:m.space.n
        d = m.g[i, i]
        if iszero(d);                r += 1
        elseif isequal(d, one(R));   p += 1
        else;                        q += 1
        end
    end
    (p, q, r)
end

top_grade(::GridBase) = 2

cells(b::GridBase, k::Integer) =
    k == 0 ? (1:b.nv) :
    k == 1 ? (1:b.ne) :
    k == 2 ? (1:b.nf) : (1:0)

n_cells(b::GridBase, k::Integer) =
    k == 0 ? b.nv : k == 1 ? b.ne : k == 2 ? b.nf : 0

function _boundary(b::GridBase, k::Int, cell::Integer)
    if k == 1
        (1 <= cell <= b.ne) || throw(ArgumentError("edge id $cell out of range 1:$(b.ne)"))
        return b.edge_boundary[cell]
    else # k == 2
        (1 <= cell <= b.nf) || throw(ArgumentError("face id $cell out of range 1:$(b.nf)"))
        return b.face_boundary[cell]
    end
end

fibre(b::GridBase{R}, k::Integer, cell) where R = CliffordFibre{R}(b.metric)

transport(b::GridBase{R}, edge::Integer) where R = identity_transport(b.metric)

has_metric(::GridBase) = true
metric(b::GridBase, k::Integer, cell) = b.metric
signature(b::GridBase) = b.sig

has_dual_complex(::GridBase) = true

# =============================================================================
# REALIZATION 3 — ManifoldChartBase (minimal charted base with a connection)
# =============================================================================
#
# A 1D sampled/charted base: nodes are sample points along a curve (carrying
# chart coordinates), oriented edges connect them, optionally closing into a loop
# so holonomy is exercisable.  Each edge carries a non-trivial transport versor
# (a real discrete connection).  A per-cell metric with a DECLARABLE, possibly
# Lorentzian signature is provided (has_metric = true), and a dual complex is
# declared (has_dual_complex = true).  Deliberately minimal — just enough to
# prove the interface carries a real connection and an indefinite signature.

"""
    ManifoldChartBase(coords, edges; metric, signature, versors = Dict())

A minimal charted 1D base: `coords` is a vector of chart coordinates (one per
node; the "sampling"), `edges` a vector of `(tail, head)` node-id pairs (close a
loop to exercise holonomy).  `top_grade == 1`.

Carries a real connection — `versors[e]` is the transport versor on edge `e`
(identity elsewhere) — and the metric capability with a **declarable signature**:
`metric` is a `Metric{R}` on the fibre and `signature` its `(p,q,r)` (e.g. the
Lorentzian `(1,1,0)`).  `has_metric == true` and `has_dual_complex == true`.

This is the realization that demonstrates a non-trivial transport and an
indefinite (Lorentzian) signature; it is intentionally not a full
differential-geometry engine.
"""
struct ManifoldChartBase{R} <: BaseSpace
    coords  :: Vector{Float64}
    edges   :: Vector{Tuple{Int,Int}}
    metric  :: Metric{R}
    sig     :: Tuple{Int,Int,Int}
    versors :: Dict{Int, CliffordTensor{R}}
end

function ManifoldChartBase(coords::Vector{<:Real}, edges::Vector{Tuple{Int,Int}};
                           metric::Metric,
                           signature::Tuple{Int,Int,Int},
                           versors = nothing)
    R    = _ring(metric)
    vers = versors === nothing ? Dict{Int,CliffordTensor{R}}() :
                                 Dict{Int,CliffordTensor{R}}(versors)
    n = length(coords)
    sum(signature) == metric.space.n || throw(ArgumentError(
        "declared signature $signature must sum to the fibre dimension $(metric.space.n)"))
    for (e, (t, h)) in enumerate(edges)
        (1 <= t <= n && 1 <= h <= n) || throw(ArgumentError(
            "edge $e = ($t, $h) references a node outside 1:$n"))
    end
    for (e, v) in vers
        (1 <= e <= length(edges)) || throw(ArgumentError(
            "versor key $e is not an edge id in 1:$(length(edges))"))
        (v.metric === metric || v.metric == metric) || throw(ArgumentError(
            "versor on edge $e lives in a different Clifford algebra than the fibre metric"))
    end
    ManifoldChartBase{R}(Float64.(coords), edges, metric, signature, vers)
end

top_grade(::ManifoldChartBase) = 1

cells(b::ManifoldChartBase, k::Integer) =
    k == 0 ? (1:length(b.coords)) :
    k == 1 ? (1:length(b.edges)) : (1:0)

n_cells(b::ManifoldChartBase, k::Integer) =
    k == 0 ? length(b.coords) : k == 1 ? length(b.edges) : 0

function _boundary(b::ManifoldChartBase, k::Int, edge::Integer)
    (1 <= edge <= length(b.edges)) || throw(ArgumentError(
        "edge id $edge out of range 1:$(length(b.edges))"))
    t, h = b.edges[edge]
    Tuple{Int,Int}[(t, -1), (h, +1)]
end

fibre(b::ManifoldChartBase{R}, k::Integer, cell) where R = CliffordFibre{R}(b.metric)

function transport(b::ManifoldChartBase{R}, edge::Integer) where R
    (1 <= edge <= length(b.edges)) || throw(ArgumentError(
        "edge id $edge out of range 1:$(length(b.edges))"))
    haskey(b.versors, edge) ? VersorTransport(b.versors[edge]) :
                              identity_transport(b.metric)
end

has_metric(::ManifoldChartBase) = true
metric(b::ManifoldChartBase, k::Integer, cell) = b.metric
signature(b::ManifoldChartBase) = b.sig

has_dual_complex(::ManifoldChartBase) = true

# ── Exports ───────────────────────────────────────────────────────────────────

export BaseSpace,
       top_grade, cells, n_cells, boundary,
       fibre, transport,
       has_metric, metric, signature, has_dual_complex, can_hodge,
       FibreDescriptor, CliffordFibre, TensorFibre,
       fibre_eltype, zero_fibre, one_fibre, fibre_matches,
       VersorTransport, identity_transport,
       GraphBase, GridBase, ManifoldChartBase
