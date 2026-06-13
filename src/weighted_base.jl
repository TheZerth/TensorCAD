# ── Phase L10.1: variable diagonal Hodge weights — WeightedGridBase ──────────
#
# MATHEMATICAL CONTRACT (DESIGN.md §16.1 dual-volumes slot, §15.4 adjointness)
#
# A `WeightedGridBase` is a `GridBase` plus a per-grade, per-cell **positive**
# diagonal Hodge weight `w_k(c) :: R` — the discrete dual-volume ratio
# |dual cell| / |primal cell| that §16.1 deliberately deferred until a
# consumer existed.  The consumer is the medium capability: by the Plebanski
# correspondence (Ledger §5), Maxwell on a varying metric ≡ Maxwell on flat
# space in a medium with metric-determined ε, μ — so a spatially varying
# weight profile *is* a medium / metric-variation region.  That is what
# Experiment 2 (wave-packet refraction) consumes.
#
# WEIGHTS ARE POSITIVE VOLUMES; SIGNATURE LIVES IN THE FIBRE METRIC.  A
# weight may never be negative or zero: signs/signature have exactly one home
# (the fibre `Metric`), and a signature-carrying weight would duplicate that
# mechanism (honest mechanisms, one home per concept).  Positivity is
# enforced at construction.
#
# THE THREE-SEAM RULE (where weights enter the engine — exactly here):
#
#   1. `_hodge_weight(R, b, k, c)` (the single documented L9.1 entry point)
#      returns `w_k(c)` — the L9.1 pairing becomes the weighted inner
#      product ⟨α,β⟩_w = Σ_c w_k(c)·⟨α(c)·reversion(β(c))⟩₀ purely by
#      dispatch (zero edits to inner_product.jl).
#   2. The Hodge star's diagonal factor (`_apply_hodge_weight` /
#      `_apply_inverse_hodge_weight`, hodge.jl): primal→dual multiplies by
#      `w_k(c)`, dual→primal by `1/w_k(c)` — so `⋆⋆` is weight-independent
#      (positive weights cancel exactly; the sign law comes from signature
#      alone) and the codifferential follows AUTOMATICALLY through ⋆d⋆.
#   3. Nothing else.  `d`, `boundary`, the dual correspondence, transport/
#      connection, and the blackboard are metric-free or weight-agnostic and
#      are untouched.
#
# THE WEIGHTED-ADJOINT FACT (the §15.4 contract surviving — gate 1).
# Tracing the value maps exactly as in the L8.2.1 derivation, with weights:
#
#   ⋆β       at dual(c):  w_g(c) · h(β(c))            (g = grade of β)
#   d(⋆β)    at dual(f):  Σ_{c ∋ f} s_cf · w_g(c) · h(β(c))
#   ⋆d⋆β     at f:        (1/w_{g-1}(f)) · [sign composite] · Σ_c s_cf w_g(c) β(c)
#
# and the L8.2.1 sign composite is +1, so
#
#   δ_w  =  W_{g-1}⁻¹ · dᵀ · W_g
#
# — exactly the adjoint of d under the WEIGHTED pairing:
#
#   ⟨dα,β⟩_w = Σ_c w_g(c)⟨(dα)(c),β(c)⟩ = Σ_f w_{g-1}(f)⟨α(f),(δ_wβ)(f)⟩ = ⟨α,δ_wβ⟩_w
#
# exact over exact rings, every grade, boundary included (the dual-d is still
# the full incidence transpose, so the natural-BC fact carries over).
# Likewise δ_wδ_w = W⁻¹dᵀ(W W⁻¹)dᵀW = 0 by d² = 0.  The unit-weight case
# reduces to the pre-L10.1 engine bit-exactly: the star hooks default to
# identity, and `_hodge_weight` already returned `one(R)`.
#
# Scalar-ring requirements: weights must be positive (an ORDERED ring — the
# positivity gate is undecidable over `Symbolics.Num`) and invertible
# (`one(R)/w`); `Rational{BigInt}` and `Float64` both qualify.
#
# OUT OF SCOPE (§16.1, recorded): unstructured meshes / Galerkin Hodge,
# negative or signature-carrying weights, dual geometry/positions (L11),
# time-varying weights (a deliberate later decision).

"""
    WeightedGridBase(grid::GridBase{R}; weights = nothing)

A [`GridBase`](@ref) equipped with per-grade, per-cell **positive** diagonal
Hodge weights `w_k(c) :: R` — the §16.1 dual-volumes capability (see the
file header for the full contract).  `weights` is a 3-vector of per-grade
weight vectors `[w₀, w₁, w₂]` indexed by cell id (`weights[k+1][c]`);
omitted, every weight is `one(R)` and the base reduces to the plain grid's
behaviour **bit-exactly**.

A spatially varying weight profile is the medium / metric-variation
capability (Plebanski: metric variation ≡ ε,μ variation) — build one with
[`weight_profile`](@ref).  Weights are positive volume ratios; **signature
lives in the fibre metric, never in weights** (non-positive weights throw).

All base-space obligations and capabilities (cells, boundary, fibre,
transport, metric, signature, dual correspondence) forward to the wrapped
grid unchanged; only the L9.1 pairing weight and the Hodge star's diagonal
factor see the weights (the three-seam rule).  Consequently `δ` on this base
is the **weighted adjoint** `W_{k-1}⁻¹ dᵀ W_k` automatically, and
`⟨dα,β⟩_w = ⟨α,δβ⟩_w` holds exactly (gate-tested).
"""
struct WeightedGridBase{R} <: BaseSpace
    grid    :: GridBase{R}
    weights :: Vector{Vector{R}}            # [w₀, w₁, w₂] by cell id

    function WeightedGridBase(grid::GridBase{R};
                              weights::Union{Nothing,AbstractVector{<:AbstractVector}} = nothing
                              ) where R
        ws = weights === nothing ?
             [fill(one(R), n_cells(grid, k)) for k in 0:2] :
             [Vector{R}(weights[k + 1]) for k in 0:2]
        length(ws) == 3 || throw(ArgumentError(
            "weights must supply one vector per grade 0:2; got $(length(ws))"))
        for k in 0:2
            length(ws[k + 1]) == n_cells(grid, k) || throw(ArgumentError(
                "grade-$k weight vector has length $(length(ws[k + 1])) but the " *
                "grid has $(n_cells(grid, k)) grade-$k cells"))
            for (c, w) in enumerate(ws[k + 1])
                positive = try
                    w > zero(R)
                catch
                    throw(ArgumentError(
                        "Hodge weights require an ordered scalar ring (positivity " *
                        "w > 0 must be decidable); $R cannot be compared"))
                end
                positive || throw(ArgumentError(
                    "Hodge weights must be positive volume ratios; got " *
                    "w_$k($c) = $w. Signature lives in the fibre metric, " *
                    "never in weights"))
            end
        end
        new{R}(grid, ws)
    end
end

# ── Lattice coordinates of a cell (for profile construction) ─────────────────

# (i, j) lattice coordinates of a cell: a vertex's grid position, an edge's or
# face's origin (lower/left) corner.  Reuses the documented GridBase layouts.
function _wgb_coords(g::GridBase, k::Integer, c::Integer)
    z = Int(c) - 1
    if k == 0
        (z % (g.nx + 1), z ÷ (g.nx + 1))
    elseif k == 1
        _, i, j = _grid_decode_edge(g, Int(c))
        (i, j)
    else
        (z % g.nx, z ÷ g.nx)
    end
end

"""
    weight_profile(grid::GridBase{R}, f) -> WeightedGridBase{R}

Convenience constructor for a spatially varying weight profile — the medium
capability Experiment 2 consumes.  `f(i, j)` is evaluated at every cell's
integer lattice coordinates (a vertex's grid position; an edge's or face's
lower/left origin corner) and must return a **positive** value convertible
to `R`; the same profile is applied at every grade.  For grade-selective
weights, pass explicit per-grade vectors to [`WeightedGridBase`](@ref).
"""
weight_profile(grid::GridBase{R}, f) where R =
    WeightedGridBase(grid; weights =
        [[R(f(_wgb_coords(grid, k, c)...)) for c in cells(grid, k)] for k in 0:2])

# ── Base-space obligations and capabilities: forward to the wrapped grid ─────

top_grade(b::WeightedGridBase)                 = top_grade(b.grid)
cells(b::WeightedGridBase, k::Integer)         = cells(b.grid, k)
n_cells(b::WeightedGridBase, k::Integer)       = n_cells(b.grid, k)
_boundary(b::WeightedGridBase, k::Int, cell)   = _boundary(b.grid, k, cell)
fibre(b::WeightedGridBase, k::Integer, cell)   = fibre(b.grid, k, cell)
transport(b::WeightedGridBase, edge::Integer)  = transport(b.grid, edge)
has_metric(::WeightedGridBase)                 = true
metric(b::WeightedGridBase, k::Integer, cell)  = metric(b.grid, k, cell)
signature(b::WeightedGridBase)                 = signature(b.grid)
has_dual_complex(::WeightedGridBase)           = true
dual_n_cells(b::WeightedGridBase, k::Integer)  = dual_n_cells(b.grid, k)
dual_cell(b::WeightedGridBase, k::Integer, c::Integer) = dual_cell(b.grid, k, c)

# ── The three seams ───────────────────────────────────────────────────────────

# Seam 1 (L9.1 pairing): the single documented weight entry point, by dispatch.
_hodge_weight(::Type{R}, b::WeightedGridBase{R}, k::Integer, cell::Integer) where R =
    b.weights[Int(k) + 1][Int(cell)]

# Seam 2 (Hodge star diagonal factor): primal→dual by w_k(c), dual→primal by
# its inverse — both routed through _hodge_weight, never a second source.
_apply_hodge_weight(b::WeightedGridBase{R}, k::Int, c::Int, x::CliffordTensor{R}) where R =
    _hodge_weight(R, b, k, c) * x
_apply_inverse_hodge_weight(b::WeightedGridBase{R}, k::Int, c::Int, x::CliffordTensor{R}) where R =
    (one(R) / _hodge_weight(R, b, k, c)) * x

# Seam 3: nothing else.

# ── Display ───────────────────────────────────────────────────────────────────

function Base.show(io::IO, b::WeightedGridBase)
    nonunit = sum(count(w -> !isequal(w, one(w)), ws) for ws in b.weights)
    print(io, "WeightedGridBase(", b.grid.nx, "×", b.grid.ny, "; ",
          nonunit == 0 ? "unit weights" : "$nonunit non-unit weights", ")")
end

export WeightedGridBase, weight_profile
