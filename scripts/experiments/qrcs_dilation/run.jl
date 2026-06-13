#!/usr/bin/env julia
#
# QRCS Experiment 1 — the dilation-curve discrimination.
#
# MODEL-LAYER EXPERIMENT, NOT AN ENGINE PHASE.  This script changes zero
# source files; its deliverable is measured curves plus a results report
# (RESULTS.md, curves.csv, both written next to this file).  The experiment
# and its permitted reading are PRE-REGISTERED in QRCS_Research_Ledger.md §8;
# the reading is reproduced verbatim at the end of the report and this script
# does not interpret beyond it.
#
# THE EXPERIMENT.  A walker-pattern lives on a cycle GraphBase (a cycle, not
# a path — no boundary effects).  Each global tick it receives a unit compute
# budget split between MOTION and INTERNAL PHASE by an allocation rule:
#
#   motion: a deterministic accumulator — per tick, acc += v; when acc ≥ 1
#           the walker hops one edge of the cycle and acc -= 1.  No RNG, so
#           the long-run hop rate equals v exactly (to accumulator
#           granularity 1/T and float roundoff).
#   phase:  φ += r(v) per tick, where r is the rule's phase allotment.
#
# After T ticks the measured proper-time rate is dφ/dt = φ/T = r(v) BY
# CONSTRUCTION of the loop — so the script verifies that the loop mechanics
# reproduce r(v) (a self-check), and the physics comparison is r(v) against
# the relativistic target 1/γ(v) = √(1−v²), with c = 1 hop/tick.
#
# THREE ALLOCATION RULES, PRE-REGISTERED (Ledger §8 — none added or removed):
#   LINEAR        r(v) = 1 − v        (budget as an ordinary split)
#   PYTHAGOREAN   r(v) = √(1−v²)      (budget components composing as a norm)
#   QUADRATIC     r(v) = 1 − v²       (control)
#
# RING: Float64.  The Pythagorean rule and the target involve √(1−v²), which
# is irrational for most sweep values, so the exact ring is not applicable;
# Float64 granularity is far below the 1/T accumulator granularity that
# bounds the measurement anyway.
#
# IMPLEMENTATION SHAPE: a standalone loop is correct here.  The walker is
# not a Field evolution — its state is (node, accumulator, phase), not a
# cochain — and contorting EvolutionSystem/SimState to host it would be
# scope distortion, not reuse.  What IS reused from L10 is the *pattern*
# (DESIGN.md §18): time is the bare tick counter (the loop parameter and
# nothing more), and every recorded quantity — φ above all — is a DERIVED
# OBSERVABLE recorded by named per-tick observer callbacks, exactly the §18
# hook shape.  Physical-time language ("proper-time rate") is observer-layer
# model content, never engine-asserted.

using Tensorsmith

const T = 10_000                                   # global ticks per run
const N_NODES = 12                                 # cycle length (any n ≥ 3)
const SWEEP = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

# The three pre-registered allocation rules (Ledger §8).
const RULES = [
    :LINEAR      => v -> 1 - v,
    :PYTHAGOREAN => v -> sqrt(1 - v^2),
    :QUADRATIC   => v -> 1 - v^2,
]

inv_gamma(v) = sqrt(1 - v^2)                       # the relativistic target

# The substrate: a cycle graph, node i → node i%N+1.  The walker re-instantiates
# neighbor-ward along these edges (Ledger §1's glider reading); the cycle has
# no boundary, so the hop rate is homogeneous over the whole run.
function cycle_graph(n)
    GraphBase(n, [(i, i % n + 1) for i in 1:n])
end

# One walker run: tick loop with named observer callbacks (t, walker) -> value
# recorded per tick — the §18 observer hook pattern, standalone.
function run_walker(graph, v::Float64, r::Float64; ticks::Int = T,
                    observers = Pair{Symbol,Any}[])
    pos, acc, φ, hops = 1, 0.0, 0.0, 0
    series = Dict{Symbol,Vector{Any}}(name => Any[] for (name, _) in observers)
    for tick in 1:ticks
        acc += v
        if acc >= 1.0
            pos = graph.edges[pos][2]              # hop the cycle edge tail→head
            acc -= 1.0
            hops += 1
        end
        φ += r                                     # internal-phase allotment
        for (name, f) in observers
            push!(series[name], f(tick, (pos = pos, acc = acc, φ = φ, hops = hops)))
        end
    end
    (φ = φ, hops = hops, series = series)
end

# ── Sweep ─────────────────────────────────────────────────────────────────────

fmt(x; d = 6) = string(round(x; digits = d))
pad(s, w) = rpad(s, w)

graph = cycle_graph(N_NODES)

results = Dict{Symbol,Vector{NamedTuple}}()        # rule => sweep rows
selfcheck_rule = 0.0                               # max |φ/T − r(v)|
selfcheck_hops = 0.0                               # max |hops/T − v|

for (rulename, rule) in RULES
    rows = NamedTuple[]
    for v in SWEEP
        r = rule(v)
        out = run_walker(graph, v, r)
        measured = out.φ / T
        target = inv_gamma(v)
        global selfcheck_rule = max(selfcheck_rule, abs(measured - r))
        global selfcheck_hops = max(selfcheck_hops, abs(out.hops / T - v))
        push!(rows, (v = v, measured = measured, target = target,
                     deviation = measured - target, hops = out.hops))
    end
    results[rulename] = rows
end

# ── Report ────────────────────────────────────────────────────────────────────

report = IOBuffer()
prn(args...) = println(report, args...)

prn("# QRCS Experiment 1 — Dilation-Curve Discrimination: RESULTS")
prn()
prn("Setup: walker on a ", N_NODES, "-node cycle GraphBase; T = ", T,
    " global ticks per run; deterministic motion accumulator (no RNG); ",
    "Float64 (the √ in the Pythagorean rule and the target is irrational, ",
    "so the exact ring is not applicable); c = 1 hop/tick.")
prn()
prn("Measured proper-time rate dφ/dt = φ/T; target 1/γ(v) = √(1−v²).")
prn()

prn("## Self-checks (attribute the result to the rule, not the plumbing)")
prn()
loop_ok = selfcheck_rule <= 1e-11
hop_ok = selfcheck_hops <= 1 / T + 1e-9
prn("- Loop mechanics reproduce r(v):  max |φ/T − r(v)| = ", fmt(selfcheck_rule; d = 16),
    "  (bound 1e-11: T float additions)  → ", loop_ok ? "PASS" : "FAIL")
prn("- Hop rate matches v:  max |hops/T − v| = ", fmt(selfcheck_hops; d = 8),
    "  (accumulator granularity 1/T = ", fmt(1 / T; d = 6), ")  → ",
    hop_ok ? "PASS" : "FAIL")
prn()

maxdevs = Dict{Symbol,Float64}()
for (rulename, _) in RULES
    rows = results[rulename]
    maxdev = maximum(abs(row.deviation) for row in rows)
    maxdevs[rulename] = maxdev
    prn("## Rule ", rulename)
    prn()
    prn("| v    | measured dφ/dt | target 1/γ | deviation |")
    prn("|------|----------------|------------|-----------|")
    for row in rows
        prn("| ", pad(fmt(row.v; d = 2), 4),
            " | ", pad(fmt(row.measured), 14),
            " | ", pad(fmt(row.target), 10),
            " | ", pad(fmt(row.deviation), 9), " |")
    end
    prn()
    # Full precision here: the Pythagorean residual is float roundoff
    # (~1e-13), and rounding it to a literal 0.0 would overstate the match.
    prn("Max |deviation| over the sweep: **", fmt(maxdev; d = 16), "**")
    prn()
end

prn("## The three curves against 1/γ (ASCII; '.' = target 1/γ, '@' = PYTHAGOREAN")
prn("   sitting on the target, L = LINEAR, Q = QUADRATIC; at v = 0 all curves")
prn("   coincide at 1.0 and display as '@')")
prn()
levels = 20:-1:0                                   # dφ/dt = 1.00 … 0.00, step 0.05
rowidx(val) = clamp(round(Int, val / 0.05), 0, 20)
for li in levels
    line = string(pad(fmt(li * 0.05; d = 2), 5), "|")
    for (ci, v) in enumerate(SWEEP)
        marks = Set{Symbol}()
        rowidx(inv_gamma(v)) == li && push!(marks, :target)
        for (rulename, _) in RULES
            rowidx(results[rulename][ci].measured) == li && push!(marks, rulename)
        end
        c = ' '
        if :PYTHAGOREAN in marks && :target in marks
            c = '@'
        elseif :LINEAR in marks
            c = 'L'
        elseif :QUADRATIC in marks
            c = 'Q'
        elseif :target in marks
            c = '.'
        elseif :PYTHAGOREAN in marks
            c = 'P'
        end
        line *= string("  ", c, "  ")
    end
    prn(line)
end
prn("     +", repeat("-", 5 * length(SWEEP)))
prn("      ", join([pad(fmt(v; d = 2), 5) for v in SWEEP]))
prn("        (v, hops/tick)")
prn()
prn("Raw curves: curves.csv (v, target, and the three measured rates).")
prn()

prn("## Pre-registered reading (verbatim, QRCS_Research_Ledger.md §8)")
prn()
prn("> **Pre-registered reading:** the Pythagorean rule matching γ is")
prn("> *consistency-by-construction* (the √ was inserted); the scientific")
prn("> content is **discriminative** — the other rules *fail*, showing Lorentz")
prn("> dilation uniquely selects the norm-composition law. **Emergence is NOT")
prn("> demonstrated by this experiment** and may not be claimed; deriving the")
prn("> Pythagorean rule from substrate dynamics is the §2 debt, registered as")
prn("> the follow-up. Failure mode pre-stated: if even the Pythagorean rule")
prn("> misses γ (discretization effects), that is a finding about the walker")
prn("> model, reported as such.")
prn()
prn("Status of this run against that reading (Ledger §9 sentence shapes):")
prn("the discriminative dilation result — Lorentz kinematics selects the norm")
prn("law among the tested allocation rules (PYTHAGOREAN max |dev| ",
    fmt(maxdevs[:PYTHAGOREAN]; d = 16), ", i.e. float roundoff, vs LINEAR ",
    fmt(maxdevs[:LINEAR]), " and QUADRATIC ", fmt(maxdevs[:QUADRATIC]),
    "). Dilation is *reproduced",
    " under an inserted law*, not derived; no emergence is claimed.")

reporttext = String(take!(report))
print(reporttext)

# ── Artifacts ─────────────────────────────────────────────────────────────────

open(joinpath(@__DIR__, "RESULTS.md"), "w") do io
    write(io, reporttext)
end

open(joinpath(@__DIR__, "curves.csv"), "w") do io
    println(io, "v,target_inv_gamma,linear,pythagorean,quadratic")
    for (ci, v) in enumerate(SWEEP)
        println(io, join([fmt(v; d = 4), fmt(inv_gamma(v); d = 10),
                          fmt(results[:LINEAR][ci].measured; d = 10),
                          fmt(results[:PYTHAGOREAN][ci].measured; d = 10),
                          fmt(results[:QUADRATIC][ci].measured; d = 10)], ","))
    end
end

println()
println("Wrote ", joinpath(@__DIR__, "RESULTS.md"))
println("Wrote ", joinpath(@__DIR__, "curves.csv"))
