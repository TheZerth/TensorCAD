# QRCS Experiment 2, Part A1 — Phase-Velocity Refraction Measurement

Engine commit: `89fce27`. Model-layer experiment; zero source files changed.

Method: evolve the L10 leapfrog on Part B's verified grade-1 (1,1) eigenmode
`Φ = δ_w(u)` (A = Φ, E = 0; standing mode at ω = √λ), measure ω from the
recorded peak-edge field trajectory via the exact three-term cosine recurrence,
`c = ω/|k|`, `|k| = √2·π/(N+1)`. dt = 0.1, 6 periods, Float64.

## Cited target (Part B / DISPERSION.md, commit c1b9ef0 — NOT re-derived)

    n = √(w₁/w₂),   c_medium/c_vacuum = √(w₂/w₁).

On a uniform weighted base `K_w = (w₂/w₁)·K_unit` exactly, so the spatial
discretization bias cancels EXACTLY in the speed ratio at matched grid N
(the absolute c carries the ~θ²/24 bias; the ratio does not). Tolerance on
the ratio: 1.0e-5 (period-fit + Float64 roundoff).

## Measured speeds and ratios (N = 8, matched resolution)

| region | measured c | ratio c/c_unit | predicted √(w₂/w₁) | deviation |
|--------|-----------|----------------|--------------------|-----------|
| unit (1,1)           | 0.994931 | 1.0 | 1.0 | 0.0 |
| medium (4,1)         | 0.497465 | 0.5 | 0.5 | 6.2e-13 |
| medium (1,4)         | 1.989862 | 2.0 | 2.0 | 2.3e-14 |
| scalar (3,3) [NULL]  | 0.994931 | 1.0 | 1.0 | 0.0 |

Max ratio deviation over all regions: **6.2e-13** (tolerance 1.0e-5).

## Null control (headline)

Uniform SCALAR weight `(3,3)` ratio = **1.0** (target 1.000000): **NO speed change** — PASS.
This attributes any measured effect to the grade ASYMMETRY, not the weight
magnitude. (Indeed `c[scalar(3,3)]` equals `c[unit]` to roundoff: 0.994931 vs 0.994931.)

## Direction check

`(4,1)` ratio = 0.5 (n = 2.0 > 1, **slower**); `(1,4)` ratio = 2.0 (n = 0.5 < 1, **faster**): both directions correct — larger face-weight w₂ speeds up. PASS.

## Resolution-stability check (the ratio must NOT drift with N)

Refraction case `(4,1)` and the unit region at N = 8, 16. The RATIO should be
stable at √(1/4) = 0.5 (bias cancels) while the absolute c drifts toward the
continuum value c_vac = 1 (unit) / 0.5 (medium).

| N | c_unit | c_(4,1) | ratio | |ratio − 0.5| |
|---|--------|---------|-------|--------------|
| 8 | 0.994931 | 0.497465 | 0.5 | 6.2e-13 |
| 16 | 0.998578 | 0.499289 | 0.5 | 4.9e-13 |

Ratio drift N=8→16: **1.1e-12** (stable — PASS; absolute c rises toward continuum while ratio holds).

## Self-check — evolved ω ties to Part B's static spectrum

Unit region, N = 8: evolved ω_cont = 0.491151 vs Part B √λ = √(2(2−2cos(π/(N+1)))) = 0.491151 → MATCH (≤1e-9).
Eigenmode quality: λspread = 7.1e-15, trajectory φ-spread = 4.4e-16 (clean single tone). This ties the dynamical measurement to the L10
eigenmode oracle (leapfrog on an eigenvector is an exact rotation at ω=√λ).

## Per-region CFL (weighted bound, documented)

| region | λ_max(w) | dt_CFL = 2/√λ_max | z = dt²λ_max (dt=0.1) |
|--------|----------|-------------------|---------------------|
| unit (1,1)           | 7.75877 | 0.718015 | 0.077588 |
| medium (4,1)         | 1.939693 | 1.43603 | 0.019397 |
| medium (1,4)         | 31.035082 | 0.359008 | 0.310351 |
| scalar (3,3) [NULL]  | 7.75877 | 0.718015 | 0.077588 |

All z ≪ 4: dt = 0.1 is safely sub-CFL in every region (worst case (1,4)).

---
## Status (Ledger §9 permitted shape)

Metric-variation-as-medium **kinematics demonstrated dynamically** — the
weighted L10 evolution propagates at the derived index `n = √(w₁/w₂)`
(refraction `(4,1)` ⇒ n=2.000, `(1,4)` ⇒ n=0.500, scalar null ⇒ n=1.000,
ratio resolution-stable to 1.1e-12). No claim of gravity control, EM-tunable c, or inexact co-variation; the
kinematic Plebanski re-description only (Ledger §5). A1 PASSES ⇒ A2 (the
literal angled-Snell bend) is now warranted.
