#!/usr/bin/env julia
#
# QRCS Experiment 2, Part A1 — phase-velocity refraction MEASUREMENT.
#
# MODEL-LAYER EXPERIMENT, NOT AN ENGINE PHASE. Changes zero source files.
# Deliverable: measure the wave speed `c` in uniform weighted regions by
# *evolving* packets with the L10 leapfrog stepper, and compare the speed
# RATIO to the derived, pre-registered target from Part B (DISPERSION.md,
# engine commit c1b9ef0):
#
#     n = √(w₁/w₂),   c_medium = √(w₂/w₁)            [Part B — CITED, not re-derived]
#
# This is the cheap, decisive A1 tier: it measures the speed RATIO directly
# (no interface, no angle, no centroid-across-a-boundary). The literal angled
# Snell bend is A2, built only if A1 passes — no A2 machinery here.
#
# WHY EVOLVE (vs reading Part B's static spectrum): A1's point is to confirm
# the TIME-DOMAIN L10 evolution propagates at the derived speed — a stronger,
# dynamical check than the static dispersion.
#
# METHOD (per uniform WeightedGridBase region):
#   • Initial data = Part B's verified grade-1 transverse eigenmode
#     Φ = δ_w(u), u_{1,1}(i,j) = sin(π(i+1)/(N+1))sin(π(j+1)/(N+1)) the lowest
#     (longest-wavelength) face sine-mode; set A = Φ, E = 0, so A(t) = cos(ωt)Φ
#     is a standing mode oscillating at ω = √λ (λ = K_w eigenvalue).
#   • Evolve with Leapfrog at dt safely under the WEIGHTED CFL bound
#     dt²·λ_max(w) ≤ 4, with λ_max(w) = (w₂/w₁)·λ_max(unit) (scales per region;
#     computed and documented below). A single small dt = 0.1 clears every
#     region's bound (worst case (1,4): z = dt²λ_max ≈ 0.31 ≪ 4).
#   • MEASURE ω from the recorded trajectory of a field component a_k =
#     A_k[peak edge]. For a single mode the discrete samples obey the exact
#     three-term cosine recurrence (a_{k-1}+a_{k+1})/(2a_k) = cos φ (φ = per-step
#     phase); averaging it over all interior steps yields φ (its spread over the
#     run ~1e-16 is the dynamical self-check: the evolution stays a clean single
#     tone at constant frequency, no drift). The integrator's known O(dt²) time
#     dispersion cos φ = 1 − dt²λ/2 is inverted exactly to recover the spatial
#     ω_cont = √λ_meas, λ_meas = 2(1−cos φ)/dt²; with dt = 0.1 the raw ω_d = φ/dt
#     and ω_cont agree to <0.1% anyway.
#   • c_measured = ω_cont / |k|, |k| = √2·π/(N+1) for the (1,1) mode.
#
# CLEAN RATIO: on a UNIFORM weighted base K_w = (w₂/w₁)·K_unit EXACTLY (Part B
# matrix-scaling result), so the discretized eigenvalues scale by exactly
# w₂/w₁ and the spatial-discretization bias CANCELS in the ratio at matched N
# (not merely to 2nd order). Float64; tolerance set by period-fit roundoff
# (~1e-13) — the absolute c carries the ~θ²/24 spatial bias, the ratio does not.
#
# Reading discipline (Ledger §5): kinematic index / Plebanski re-description
# ONLY. No claim of gravity control, EM-tunable c, or inexact co-variation.
# STOP-AND-REPORT (do not tune) if: the scalar null control changes speed; a
# measured n departs from √(w₁/w₂) beyond tolerance; or the ratio drifts with
# resolution (a discretization artifact). Any is a real finding.

using Tensorsmith

const COMMIT = strip(read(`git -C $(@__DIR__)/../../.. rev-parse --short HEAD`, String))
const DT = 0.1
const N_PERIODS = 6
const RATIO_TOL = 1e-5          # ratio deviation budget (period-fit + Float64 roundoff)

_sc(x) = get(x.terms, Int[], 0.0)

# Measure c in a uniform (w1,w2) region on an N×N grid by evolving the L10
# leapfrog on the Part-B (1,1) eigenmode and extracting ω from the trajectory.
function measure_region(N, w1, w2; dt = DT, n_periods = N_PERIODS)
    mf = signature_metric(VectorSpace(2), Float64, 2, 0, 0)
    grid = GridBase(N, N; metric = mf)
    wb = WeightedGridBase(grid; weights = [fill(1.0, n_cells(grid, 0)),
                                           fill(Float64(w1), n_cells(grid, 1)),
                                           fill(Float64(w2), n_cells(grid, 2))])
    fid(i, j) = i + j * N + 1
    u = Field(wb, 2, Dict(fid(i, j) => clifford_scalar(mf,
            sin(pi * (i + 1) / (N + 1)) * sin(pi * (j + 1) / (N + 1)))
            for i in 0:N-1, j in 0:N-1))
    Φ = codifferential(wb, u)                          # grade-1 transverse mode

    # Eigenmode precondition: K_w Φ = λ Φ (λspread ~ roundoff confirms it).
    KΦ = codifferential(wb, d(Φ))
    edges = [e for e in cells(grid, 1) if abs(_sc(evaluate(Φ, e))) > 1e-9]
    λs = [_sc(evaluate(KΦ, e)) / _sc(evaluate(Φ, e)) for e in edges]
    λ = sum(λs) / length(λs)
    λspread = maximum(λs) - minimum(λs)
    peak = edges[argmax([abs(_sc(evaluate(Φ, e))) for e in edges])]

    # Weighted CFL bound (λ_max scales with w₂/w₁; Part B verified dispersion).
    λmax = (w2 / w1) * 2 * (2 - 2cos(pi * N / (N + 1)))
    dtcfl = 2 / sqrt(λmax)
    z = dt^2 * λmax
    z < 4 || error("dt=$dt violates CFL in region ($w1,$w2): z=$z ≥ 4")

    # Evolve and record the peak-edge field component.
    nsteps = round(Int, n_periods * 2pi / (sqrt(λ) * dt))
    sys = maxwell_system(wb)
    E0 = zero_field(wb, 1, fibre(wb, 1, 1))
    traj = evolve(sys, SimState(sys, Dict(:A => Φ, :E => E0)), Leapfrog(), dt, nsteps;
                  observers = [:a => ((t, st) -> _sc(evaluate(st[:A], peak)))])
    a = Float64.(traj[:a])
    amax = maximum(abs, a)

    # ω from the exact three-term cosine recurrence (a single-tone estimator).
    rs = [(a[k-1] + a[k+1]) / (2a[k]) for k in 2:length(a)-1 if abs(a[k]) > 0.3amax]
    cosφ = sum(rs) / length(rs)
    φ_spread = maximum(rs) - minimum(rs)
    φ = acos(clamp(cosφ, -1, 1))
    ω_d = φ / dt                                       # raw discrete-trajectory ω
    λ_meas = 2 * (1 - cosφ) / dt^2                     # remove integrator O(dt²)
    ω_cont = sqrt(λ_meas)
    θ = pi / (N + 1)
    kmag = sqrt(2) * θ
    c = ω_cont / kmag

    (; N, w1, w2, λ, λspread, λmax, dtcfl, z, nsteps,
       φ_spread, ω_d, ω_cont, c, npts = length(rs))
end

# ── Report ──────────────────────────────────────────────────────────────────
io = IOBuffer()
prn(args...) = println(io, args...)
f6(x) = string(round(x; digits = 6))

prn("# QRCS Experiment 2, Part A1 — Phase-Velocity Refraction Measurement")
prn()
prn("Engine commit: `", COMMIT, "`. Model-layer experiment; zero source files changed.")
prn()
prn("Method: evolve the L10 leapfrog on Part B's verified grade-1 (1,1) eigenmode")
prn("`Φ = δ_w(u)` (A = Φ, E = 0; standing mode at ω = √λ), measure ω from the")
prn("recorded peak-edge field trajectory via the exact three-term cosine recurrence,")
prn("`c = ω/|k|`, `|k| = √2·π/(N+1)`. dt = ", DT, ", ", N_PERIODS, " periods, Float64.")
prn()
prn("## Cited target (Part B / DISPERSION.md, commit c1b9ef0 — NOT re-derived)")
prn()
prn("    n = √(w₁/w₂),   c_medium/c_vacuum = √(w₂/w₁).")
prn()
prn("On a uniform weighted base `K_w = (w₂/w₁)·K_unit` exactly, so the spatial")
prn("discretization bias cancels EXACTLY in the speed ratio at matched grid N")
prn("(the absolute c carries the ~θ²/24 bias; the ratio does not). Tolerance on")
prn("the ratio: ", RATIO_TOL, " (period-fit + Float64 roundoff).")
prn()

# Primary measurements at N = 8 (matched resolution; ratio vs the unit region).
N0 = 8
regions = [("unit (1,1)", 1, 1, 1.0),
           ("medium (4,1)", 4, 1, sqrt(1 / 4)),
           ("medium (1,4)", 1, 4, sqrt(4 / 1)),
           ("scalar (3,3) [NULL]", 3, 3, 1.0)]
res = Dict{String,Any}()
for (nm, w1, w2, _) in regions
    res[nm] = measure_region(N0, w1, w2)
end
cu = res["unit (1,1)"].c

prn("## Measured speeds and ratios (N = ", N0, ", matched resolution)")
prn()
prn("| region | measured c | ratio c/c_unit | predicted √(w₂/w₁) | deviation |")
prn("|--------|-----------|----------------|--------------------|-----------|")
maxdev = 0.0
for (nm, w1, w2, pred) in regions
    r = res[nm]
    ratio = r.c / cu
    dev = abs(ratio - pred)
    global maxdev = max(maxdev, dev)
    prn("| ", rpad(nm, 20), " | ", f6(r.c), " | ", f6(ratio), " | ",
        f6(pred), " | ", string(round(dev; sigdigits = 2)), " |")
end
prn()
prn("Max ratio deviation over all regions: **", string(round(maxdev; sigdigits = 2)),
    "** (tolerance ", RATIO_TOL, ").")
prn()

# ── Null control ─────────────────────────────────────────────────────────────
null_ratio = res["scalar (3,3) [NULL]"].c / cu
null_ok = abs(null_ratio - 1.0) < RATIO_TOL
prn("## Null control (headline)")
prn()
prn("Uniform SCALAR weight `(3,3)` ratio = **", f6(null_ratio), "** (target 1.000000): ",
    null_ok ? "**NO speed change** — PASS." : "**SPEED CHANGED — STOP-AND-REPORT.**")
prn("This attributes any measured effect to the grade ASYMMETRY, not the weight")
prn("magnitude. (Indeed `c[scalar(3,3)]` equals `c[unit]` to roundoff: ",
    f6(res["scalar (3,3) [NULL]"].c), " vs ", f6(cu), ".)")
prn()

# ── Direction check ──────────────────────────────────────────────────────────
r41 = res["medium (4,1)"].c / cu
r14 = res["medium (1,4)"].c / cu
dir_ok = r41 < 1 < r14
prn("## Direction check")
prn()
prn("`(4,1)` ratio = ", f6(r41), " (n = ", f6(1 / r41), " > 1, **slower**); ",
    "`(1,4)` ratio = ", f6(r14), " (n = ", f6(1 / r14), " < 1, **faster**): ",
    dir_ok ? "both directions correct — larger face-weight w₂ speeds up. PASS." :
             "DIRECTION WRONG — STOP-AND-REPORT.")
prn()

# ── Resolution stability ─────────────────────────────────────────────────────
prn("## Resolution-stability check (the ratio must NOT drift with N)")
prn()
prn("Refraction case `(4,1)` and the unit region at N = 8, 16. The RATIO should be")
prn("stable at √(1/4) = 0.5 (bias cancels) while the absolute c drifts toward the")
prn("continuum value c_vac = 1 (unit) / 0.5 (medium).")
prn()
prn("| N | c_unit | c_(4,1) | ratio | |ratio − 0.5| |")
prn("|---|--------|---------|-------|--------------|")
res_drift = Float64[]
for N in (8, 16)
    ru = measure_region(N, 1, 1)
    rm = measure_region(N, 4, 1)
    ratio = rm.c / ru.c
    push!(res_drift, ratio)
    prn("| ", N, " | ", f6(ru.c), " | ", f6(rm.c), " | ", f6(ratio), " | ",
        string(round(abs(ratio - 0.5); sigdigits = 2)), " |")
end
drift = abs(res_drift[2] - res_drift[1])
drift_ok = drift < RATIO_TOL
prn()
prn("Ratio drift N=8→16: **", string(round(drift; sigdigits = 2)), "** (",
    drift_ok ? "stable — PASS; absolute c rises toward continuum while ratio holds" :
               "DRIFTS — discretization artifact, STOP-AND-REPORT", ").")
prn()

# ── Self-check: evolved ω vs Part B's √λ ─────────────────────────────────────
ru8 = res["unit (1,1)"]
λ_partB = 2 * (2 - 2cos(pi / (N0 + 1)))            # unit (1,1) eigenvalue, Part B
ω_partB = sqrt(λ_partB)
self_ok = abs(ru8.ω_cont - ω_partB) < 1e-9
prn("## Self-check — evolved ω ties to Part B's static spectrum")
prn()
prn("Unit region, N = ", N0, ": evolved ω_cont = ", f6(ru8.ω_cont),
    " vs Part B √λ = √(2(2−2cos(π/(N+1)))) = ", f6(ω_partB), " → ",
    self_ok ? "MATCH" : "MISMATCH", " (≤1e-9).")
prn("Eigenmode quality: λspread = ", string(round(ru8.λspread; sigdigits = 2)),
    ", trajectory φ-spread = ", string(round(ru8.φ_spread; sigdigits = 2)),
    " (clean single tone). This ties the dynamical measurement to the L10")
prn("eigenmode oracle (leapfrog on an eigenvector is an exact rotation at ω=√λ).")
prn()

# ── CFL documentation ────────────────────────────────────────────────────────
prn("## Per-region CFL (weighted bound, documented)")
prn()
prn("| region | λ_max(w) | dt_CFL = 2/√λ_max | z = dt²λ_max (dt=", DT, ") |")
prn("|--------|----------|-------------------|---------------------|")
for (nm, w1, w2, _) in regions
    r = res[nm]
    prn("| ", rpad(nm, 20), " | ", f6(r.λmax), " | ", f6(r.dtcfl), " | ",
        f6(r.z), " |")
end
prn()
prn("All z ≪ 4: dt = ", DT, " is safely sub-CFL in every region (worst case (1,4)).")
prn()

# ── Status ───────────────────────────────────────────────────────────────────
all_ok = maxdev < RATIO_TOL && null_ok && dir_ok && drift_ok && self_ok
prn("---")
prn("## Status (Ledger §9 permitted shape)")
prn()
if all_ok
    prn("Metric-variation-as-medium **kinematics demonstrated dynamically** — the")
    prn("weighted L10 evolution propagates at the derived index `n = √(w₁/w₂)`")
    prn("(refraction `(4,1)` ⇒ n=2.000, `(1,4)` ⇒ n=0.500, scalar null ⇒ n=1.000,")
    prn("ratio resolution-stable to ", string(round(max(maxdev, drift); sigdigits = 2)),
        "). No claim of gravity control, EM-tunable c, or inexact co-variation; the")
    prn("kinematic Plebanski re-description only (Ledger §5). A1 PASSES ⇒ A2 (the")
    prn("literal angled-Snell bend) is now warranted.")
else
    prn("A1 did NOT cleanly pass — see the flagged section(s) above. Per the stop")
    prn("conditions this is a finding about the operator/evolution, reported as such,")
    prn("NOT tuned. A2 is NOT warranted until resolved.")
end

report = String(take!(io))
print(report)
open(joinpath(@__DIR__, "REFRACTION_A1.md"), "w") do f
    write(f, report)
end
println("\nWrote ", joinpath(@__DIR__, "REFRACTION_A1.md"))
