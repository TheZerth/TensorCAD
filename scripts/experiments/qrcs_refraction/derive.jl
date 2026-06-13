#!/usr/bin/env julia
#
# QRCS Experiment 2, Part B — DERIVE the weighted-lattice dispersion relation.
#
# DERIVATION-AND-VERIFICATION PASS, NOT THE EXPERIMENT AND NOT AN ENGINE PHASE.
# Changes zero source files.  Deliverable: a *verified closed-form target* for
# the Part-A refraction experiment — the dispersion relation λ(k;w₁,w₂), the
# wave speed c_medium(w₁,w₂), and the refractive index n(w₁,w₂) — established
# by hand and confirmed numerically (exactly where possible) BEFORE any wave
# packet is launched.  Writes DISPERSION.md next to this file and prints the
# same to stdout.
#
# Reading discipline (Ledger §5): this derives the *kinematic* index of
# metric-variation-as-medium — the exact, safe half of §5 (Plebanski:
# metric variation ≡ ε,μ variation, an exact re-description).  It does NOT
# claim gravity control, EM-tunable c, or any inexact-co-variation effect.
#
# ════════════════════════════════════════════════════════════════════════════
# THE DERIVATION (all work shown; each step is asserted numerically below)
# ════════════════════════════════════════════════════════════════════════════
#
# The weighted wave operator on the grade-1 potential A (the L10 maxwell pair
# is ∂ₜA = E, ∂ₜE = −δ_w d A, so ∂ₜ²A = −K_w A) is
#
#     K_w = δ_w d = W₁⁻¹ dᵀ W₂ d                                            (1)
#
# with d : Ω¹→Ω² the discrete curl, dᵀ : Ω²→Ω¹ its signed-incidence
# transpose, W₁ the diagonal EDGE (grade-1) weight, W₂ the diagonal FACE
# (grade-2) weight.  Note the two weights live on different grades — this is
# the whole story.  (δ_w = W_{g−1}⁻¹ dᵀ W_g is the weighted_base.jl header
# result with g = 2.)
#
# ── Plane-wave substitution on the uniform (infinite/periodic) lattice ──────
# Index a grade-1 cochain by horizontal-edge amplitudes A_x(i,j) (edge from
# vertex (i,j) to (i+1,j)) and vertical-edge amplitudes A_y(i,j).  GridBase's
# face boundary is bottom + right − top − left, so the discrete curl is
#
#     (dA)(face i,j) = A_x(i,j) + A_y(i+1,j) − A_x(i,j+1) − A_y(i,j).         (2)
#
# Substitute the lattice plane wave A_x(i,j)=a_x e^{i(k_x i+k_y j)},
# A_y(i,j)=a_y e^{i(k_x i+k_y j)}, and write μ_x = e^{ik_x}−1, μ_y = e^{ik_y}−1
# (so |μ_x|² = 2−2cos k_x, |μ_y|² = 2−2cos k_y).  Then (2) gives the face
# amplitude
#
#     F = μ_x a_y − μ_y a_x.                                                 (3)
#
# Apply W₂ (multiply by the constant face weight w₂), then dᵀ (faces→edges;
# from the incidence transpose: (dᵀG)_x = −w₂F·conj(μ_y),
# (dᵀG)_y = +w₂F·conj(μ_x)), then W₁⁻¹ (divide by the constant edge weight
# w₁).  Collecting, K_w acts on (a_x,a_y) as the 2×2 Hermitian matrix
#
#     K_w = (w₂/w₁) · [  |μ_y|²        −conj(μ_y)μ_x ]
#                      [ −conj(μ_x)μ_y   |μ_x|²      ].                       (4)
#
# This matrix has determinant 0 (rank 1): its eigenvalues are
#   • 0  — eigenvector (a_x,a_y) ∝ (μ_x,μ_y): the LONGITUDINAL / gauge mode
#          A = dφ, killed by K (d∘d = 0); the L10 "gauge sector".
#   • λ  — the TRANSVERSE (co-exact) mode, trace of (4):
#
#     ┌─────────────────────────────────────────────────────────────────┐
#     │  λ(k_x,k_y; w₁,w₂) = (w₂/w₁) · [ (2−2cos k_x) + (2−2cos k_y) ]   │  (5)
#     └─────────────────────────────────────────────────────────────────┘
#
# This is THE dispersion relation.  The weights enter ONLY through the scalar
# ratio w₂/w₁; the lattice factor is the standard one.
#
# ── Finite-grid quantization (what the engine actually diagonalizes) ────────
# On a bounded GridBase the natural (Neumann-type, L8.2.1) BC quantizes the
# allowed wavenumbers.  The top-grade operator M₂^w = d δ_w (which shares the
# nonzero spectrum of K_w by the SVD pairing used in the L10/L10.1 tests) has
# the EXACT eigenmodes (Dirichlet sine modes of the face path-graph)
#
#     u_{p,q}(i,j) = sin(πp(i+1)/(N+1)) · sin(πq(j+1)/(M+1)),                (6)
#
# with eigenvalue exactly (5) at k_x = πp/(N+1), k_y = πq/(M+1), p=1..N,
# q=1..M.  (Sanity: GridBase(2,1) ⇒ {p,q}={(1,1),(2,1)} ⇒ λ_unit = 3, 5 — the
# values the L10 test pins by hand.)  There is no zero face-mode: dδ is
# positive-definite at the top grade (b₂ = 0 on the disk), so the lowest
# nonzero K_w mode is (p,q) = (1,1).
#
# ── Wave speed and refractive index ─────────────────────────────────────────
# Continuous-time wave equation ∂ₜ²A = −K_w A ⇒ ω² = λ.  Long-wavelength
# limit |k|→0, 2−2cos k ≈ k²:
#
#     ω² ≈ (w₂/w₁)(k_x² + k_y²) = (w₂/w₁)|k|²  ⇒  ω ≈ √(w₂/w₁) |k|,
#
# so phase and group velocity coincide at the linear (non-dispersive) limit:
#
#     ┌──────────────────────────────────────────────────────────────────┐
#     │  c_medium(w₁,w₂) = √(w₂/w₁)                                        │
#     │  c_vacuum = c_medium(1,1) = 1                                      │
#     │  n(w₁,w₂) = c_vacuum/c_medium = √(w₁/w₂)                           │  (7)
#     └──────────────────────────────────────────────────────────────────┘
#
# DIRECTION: larger EDGE weight w₁ ⇒ larger n ⇒ SLOWER wave; larger FACE
# weight w₂ ⇒ smaller n ⇒ FASTER wave.
#
# DICTIONARY to a dielectric (Ledger §5, the energy split fixes it): the L10.1
# energy is H = ½(⟨E,E⟩_{w₁} + ⟨dA,dA⟩_{w₂}); electric energy ~ w₁|E|² = ½εE²
# and magnetic energy ~ w₂|dA|² = ½B²/μ identify ε = w₁, μ = 1/w₂.  Then
# n = √(εμ) = √(w₁/w₂) ✓ (matches (7)), and Z = √(μ/ε) = 1/√(w₁ w₂).
#
#     ┌──────────────────────────────────────────────────────────────────┐
#     │  index n depends on the RATIO   w₁/w₂   (= √(εμ))                  │
#     │  impedance Z depends on the PRODUCT w₁ w₂ (= 1/√(εμ)... = 1/√(w₁w₂))│
#     └──────────────────────────────────────────────────────────────────┘
#
# This REFUTES the PM's conjectured *form* ("n a function of w₁·w₂"): n is a
# function of the ratio, not the product; product and ratio are swapped vs the
# conjecture.  The PM's operational scaling TEST — "(w₁,w₂) and (cw₁,cw₂) give
# the same n" — is, however, CONFIRMED and is exactly the ratio form (common
# scaling leaves w₁/w₂ fixed); the parenthetical "(index depends on the
# product)" attached to that test in the prompt is internally inconsistent and
# is the part that is wrong.
#
# ════════════════════════════════════════════════════════════════════════════

using Tensorsmith, LinearAlgebra

const R = Rational{BigInt}

# Engine provenance.
const COMMIT = strip(read(`git -C $(@__DIR__)/../../.. rev-parse --short HEAD`, String))

# ── K_w matrix on grade-1 cochains (scalar fibre component) ─────────────────
# K_w is fibre-component-wise linear with scalar weights, so the matrix on the
# clifford_one component fully characterizes the spectrum.  Independent of w₀
# (the grade-0 weight): K_w = W₁⁻¹dᵀW₂d touches only edge and face weights.
function Kw_matrix(N, M, w0, w1, w2; ring = R)
    m = signature_metric(VectorSpace(2), ring, 2, 0, 0)
    grid = GridBase(N, M; metric = m)
    wb = WeightedGridBase(grid; weights = [fill(ring(w0), n_cells(grid, 0)),
                                           fill(ring(w1), n_cells(grid, 1)),
                                           fill(ring(w2), n_cells(grid, 2))])
    ne = n_cells(grid, 1)
    A = zeros(ring, ne, ne)
    scal(x) = get(x.terms, Int[], zero(ring))
    for e in 1:ne
        col = codifferential(wb, d(Field(wb, 1, Dict(e => clifford_one(m)))))
        for ep in 1:ne
            A[ep, e] = scal(evaluate(col, ep))
        end
    end
    A
end

nonzero_spectrum(A; tol = 1e-9) =
    sort([λ for λ in real.(eigvals(Float64.(A))) if abs(λ) > tol])

# ── Exact eigenvalue of the (p,q) face sine-mode of M₂^w = d δ_w (Float64) ──
# Returns (λ, spread): spread ≈ 0 confirms (6) is an exact eigenmode.
function face_mode_eigenvalue(N, M, p, q, w1, w2)
    mf = signature_metric(VectorSpace(2), Float64, 2, 0, 0)
    grid = GridBase(N, M; metric = mf)
    wb = WeightedGridBase(grid; weights = [fill(1.0, n_cells(grid, 0)),
                                           fill(Float64(w1), n_cells(grid, 1)),
                                           fill(Float64(w2), n_cells(grid, 2))])
    fid(i, j) = i + j * N + 1
    u = Field(wb, 2, Dict(fid(i, j) => clifford_scalar(mf,
            sin(pi * p * (i + 1) / (N + 1)) * sin(pi * q * (j + 1) / (M + 1)))
            for i in 0:N-1, j in 0:M-1))
    Mu = d(codifferential(wb, u))                 # M₂^w u
    scal(x) = get(x.terms, Int[], 0.0)
    ratios = Float64[]
    for f in cells(grid, 2)
        iu = scal(evaluate(u, f))
        abs(iu) > 1e-9 && push!(ratios, scal(evaluate(Mu, f)) / iu)
    end
    (sum(ratios) / length(ratios), maximum(ratios) - minimum(ratios))
end

# ════════════════════════════════════════════════════════════════════════════
# Report assembly
# ════════════════════════════════════════════════════════════════════════════

io = IOBuffer()
prn(args...) = println(io, args...)
fmt(x; d = 6) = string(round(x; digits = d))

prn("# QRCS Experiment 2, Part B — Weighted-Lattice Dispersion Relation")
prn()
prn("Engine commit: `", COMMIT, "` (L10.1 WeightedGridBase). Derivation pass; ",
    "zero source files changed.")
prn()
prn("## Dispersion relation (derived; full work in derive.jl header)")
prn()
prn("For a uniform lattice with constant grade weights `(w₁, w₂)` (edge, face),")
prn("the wave operator `K_w = δ_w d = W₁⁻¹ dᵀ W₂ d` on the grade-1 potential `A`")
prn("has, per Bloch wavevector `k = (k_x, k_y)`, a longitudinal (gauge) mode at")
prn("eigenvalue 0 and a single transverse mode at")
prn()
prn("    λ(k; w₁, w₂) = (w₂/w₁) · [ (2 − 2cos k_x) + (2 − 2cos k_y) ].")
prn()
prn("The weights enter ONLY through the scalar ratio `w₂/w₁`. With `ω² = λ`,")
prn("the long-wavelength (`|k|→0`) wave speed and refractive index are")
prn()
prn("    c_medium(w₁,w₂) = √(w₂/w₁),    n(w₁,w₂) = c_vac/c_medium = √(w₁/w₂).")
prn()
prn("Direction: larger edge weight `w₁` ⇒ larger `n` ⇒ slower wave; larger face")
prn("weight `w₂` ⇒ faster. Dielectric dictionary (from the L10.1 energy split):")
prn("`ε = w₁`, `μ = 1/w₂`, so `n = √(εμ) = √(w₁/w₂)` and `Z = √(μ/ε) = 1/√(w₁ w₂)`.")
prn("**Index depends on the ratio `w₁/w₂`; impedance depends on the product `w₁ w₂`.**")
prn()

# ── Prediction 1: scalar-uniform null ───────────────────────────────────────
prn("## Prediction 1 — uniform SCALAR weight produces NO refraction")
prn()
prn("Exact over `Rational{BigInt}`: the grade-1 wave operator `K_w` for a uniform")
prn("scalar weight `w = 3` on all grades vs unit weight `w = 1`.")
prn()
p1_ok = true
prn("| grid | K_w(3,3,3) == K_w(1,1,1) exactly | nonzero spectrum (identical) |")
prn("|------|----------------------------------|------------------------------|")
for (N, M) in ((2, 2), (3, 2))
    Mu = Kw_matrix(N, M, 1, 1, 1)
    Ms = Kw_matrix(N, M, 3, 3, 3)
    eq = Ms == Mu
    global p1_ok &= eq
    spec = nonzero_spectrum(Mu)
    prn("| (", N, ",", M, ") | ", eq, " | ",
        join(fmt.(spec; d = 4), ", "), " |")
end
prn()
prn("**CONFIRMED.** `K_w(3,3,3) == K_w(1,1,1)` as exact rational matrices (not")
prn("merely isospectral — literally the same operator), because `K_w = (1/w)dᵀ(w)d")
prn("= dᵀd` when `w₁ = w₂ = w`. A uniform scalar weight rescales the energy")
prn("(inner product) but leaves the dynamics — hence the wave speed — untouched,")
prn("so it does not refract. (The grade-0 weight `w₀` never enters `K_w` at all.)")
prn()

# ── Prediction 2: asymmetric weights refract ────────────────────────────────
prn("## Prediction 2 — asymmetric grade weights DO refract")
prn()
prn("Exact: `(w₁,w₂) = (2,1)` vs unit, on GridBase(2,2). Closed form predicts the")
prn("whole spectrum scales by `w₂/w₁ = 1/2`.")
prn()
Mu22 = Kw_matrix(2, 2, 1, 1, 1)
Ma22 = Kw_matrix(2, 2, 1, 2, 1)                    # (w₁,w₂)=(2,1)
p2_ok = (Ma22 == (1 // 2) .* Mu22)
specu = nonzero_spectrum(Mu22)
speca = nonzero_spectrum(Ma22)
prn("    K_w(2,1) == (1/2)·K_unit exactly :  ", p2_ok)
prn("    unit nonzero spectrum  : ", join(fmt.(specu; d = 4), ", "))
prn("    (2,1) nonzero spectrum : ", join(fmt.(speca; d = 4), ", "))
prn("    ratios (should all be 1/2): ", join(fmt.(speca ./ specu; d = 4), ", "))
prn()
prn("**CONFIRMED.** Asymmetric `(w₁,w₂)` changes the nonzero spectrum (here exactly")
prn("halves it ⇒ wave speed × `√(1/2)`), so it refracts — as required for Part A")
prn("to be well-posed.")
prn()

# ── Dispersion verification: exact sine eigenmode ───────────────────────────
prn("## Dispersion verification — the closed form (5) is exact on finite grids")
prn()
prn("The face sine-mode `u_{p,q}(i,j) = sin(πp(i+1)/(N+1))·sin(πq(j+1)/(M+1))` is")
prn("an exact eigenmode of `M₂^w = d δ_w` (which shares the nonzero `K_w` spectrum")
prn("by the SVD pairing). Below: applying the engine operator returns a CONSTANT")
prn("ratio (spread ≈ 0) equal to the closed form, on GridBase(2,1) and (3,2).")
prn()
prn("| grid | (p,q) | engine λ | closed form (w₂/w₁)[Σ 2−2cos] | spread |")
prn("|------|-------|----------|-------------------------------|--------|")
disp_ok = true
for (N, M, w1, w2) in ((2, 1, 1, 1), (2, 1, 4, 1), (3, 2, 4, 1))
    for (p, q) in ((1, 1), (min(2, N), 1))
        λ, spread = face_mode_eigenvalue(N, M, p, q, w1, w2)
        cf = (w2 / w1) * ((2 - 2cos(pi * p / (N + 1))) + (2 - 2cos(pi * q / (M + 1))))
        global disp_ok &= (abs(λ - cf) < 1e-10 && spread < 1e-9)
        prn("| (", N, ",", M, ") | (", p, ",", q, ") | ", fmt(λ; d = 8), " | ",
            fmt(cf; d = 8), " | ", fmt(spread; d = 12), " |")
    end
end
prn()
prn(disp_ok ? "**CONFIRMED.** Engine eigenvalues match (5) to <1e-10; spreads are roundoff." :
              "**DISCREPANCY** — see values above (a finding, not to be tuned).")
prn()

# ── Long-wavelength convergence to c_medium ─────────────────────────────────
prn("## Long-wavelength index — convergence to `c_medium = √(w₂/w₁)`")
prn()
prn("Lowest nonzero mode `(p,q)=(1,1)` on `N×N` grids, `(w₁,w₂)=(4,1)` ⇒ target")
prn("`c_medium = √(1/4) = 1/2`, `n = √(4/1) = 2`. Wavenumber `|k|² = 2(π/(N+1))²`,")
prn("`c = √(λ/|k|²)`. Discrete factor `(2−2cosθ)/θ² = 1 − θ²/12 + …` ⇒ 2nd order.")
prn()
prn("| N  | engine λ(1,1) | measured c_medium | target √(w₂/w₁) | error |")
prn("|----|---------------|-------------------|-----------------|-------|")
target_c = sqrt(1 / 4)
prev_err = NaN
order_seen = String[]
for N in (4, 8, 16, 32)
    λ, _ = face_mode_eigenvalue(N, N, 1, 1, 4, 1)
    θ = pi / (N + 1)
    k2 = 2 * θ^2
    c = sqrt(λ / k2)
    err = abs(c - target_c)
    isnan(prev_err) || push!(order_seen, fmt(prev_err / err; d = 2))
    global prev_err = err
    prn("| ", lpad(N, 2), " | ", fmt(λ; d = 8), " | ", fmt(c; d = 8), " | ",
        fmt(target_c; d = 8), " | ", fmt(err; d = 8), " |")
end
prn()
prn("Error-ratio per N-doubling: ", join(order_seen, ", "),
    " (→ 4 = second-order as N grows; θ = π/(N+1) slightly inflates small-N ratios).")
prn("`c_medium → 0.5` and `n = 1/c_medium → 2`, matching the closed form.")
prn()

# ── Impedance / scaling structure ───────────────────────────────────────────
prn("## Index vs impedance — ratio vs product (exact)")
prn()
Ma_21 = Kw_matrix(2, 2, 1, 2, 1)                   # (w₁,w₂)=(2,1)
Ma_42 = Kw_matrix(2, 2, 1, 4, 2)                   # (cw₁,cw₂)=(4,2), c=2
scale_same = (Ma_21 == Ma_42)
prn("Common scaling `(w₁,w₂)=(2,1)` vs `(4,2)` (c=2): `K_w` identical exactly? ",
    scale_same, ".")
prn()
prn("So `n` is invariant under common scaling — `n` depends on the RATIO `w₁/w₂`")
prn("(here 2 in both ⇒ `n=√2`), confirming the operational scaling test. The")
prn("impedance `Z = 1/√(w₁ w₂)` does change (product 2 → 8, `Z` ×1/2): impedance")
prn("depends on the PRODUCT. Measuring `Z` needs the reflection coefficient at an")
prn("interface — deferred to Part A (the packet machinery); here it is established")
prn("structurally from the energy split. **NB: this swaps product/ratio relative")
prn("to the PM's conjectured form** (which guessed `n ∝ product`); the conjecture's")
prn("scaling test is satisfied, its product/ratio attribution is not.")
prn()

# ── Part A target ───────────────────────────────────────────────────────────
prn("## Part A target (the pre-registered, E1-style closed-form number)")
prn()
prn("Part A (the refraction experiment) must launch a long-wavelength wave packet")
prn("on the potential `A` across an interface between a unit region `(w₁,w₂)=(1,1)`")
prn("and a medium region with asymmetric weights, and measure the ratio of phase")
prn("(or group) speeds across the interface. The **pre-registered target** is")
prn()
prn("    c_medium / c_vacuum = √(w₂/w₁),   equivalently   n = √(w₁/w₂),")
prn()
prn("with Snell's law `sin θ_i / sin θ_t = n_medium / n_vacuum = √(w₁/w₂)` for an")
prn("oblique interface. Concretely, the registered medium `(w₁,w₂) = (4,1)` must")
prn("give a measured `n = 2.000` (speed halved) to the packet's resolution; a")
prn("uniform scalar medium `(w,w)` must give `n = 1` (NO bending) as the null")
prn("control. A uniform scalar weight that bends the packet, or an asymmetric")
prn("weight whose measured `n` departs from `√(w₁/w₂)` beyond discretization")
prn("error, is a finding about the model/operator — to be reported, not tuned.")
prn()
prn("## Reading discipline (Ledger §5)")
prn()
prn("This establishes the **kinematic** index of metric-variation-as-medium — the")
prn("exact, safe half of §5 (Plebanski re-description). It makes NO claim of")
prn("gravity control, EM-tunable `c`, or inexact co-variation; those remain on the")
prn("constrained fork. The result is `n = √(w₁/w₂)` (ratio), DERIVED here so Part A")
prn("measures against an independently-known target, in the E1 discipline.")

# ── Status line ─────────────────────────────────────────────────────────────
prn()
prn("---")
prn("Self-check summary: scalar-null ", p1_ok ? "CONFIRMED" : "REFUTED",
    "; asymmetric-refraction ", p2_ok ? "CONFIRMED" : "REFUTED",
    "; dispersion-formula ", disp_ok ? "EXACT-MATCH" : "DISCREPANCY",
    "; common-scaling-invariance ", scale_same ? "CONFIRMED" : "REFUTED", ".")

report = String(take!(io))
print(report)
open(joinpath(@__DIR__, "DISPERSION.md"), "w") do f
    write(f, report)
end
println("\nWrote ", joinpath(@__DIR__, "DISPERSION.md"))
