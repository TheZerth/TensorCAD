# ── Phase L10: The Simulation Engine — method-of-lines evolution ─────────────
#
# MATHEMATICAL / ARCHITECTURAL CONTRACT (DESIGN.md §18, §19 — settled Rev 2.6)
#
# Method of lines: space is the verified discrete complex (L7/L8), time is a
# bare loop parameter advanced by a PLUGGABLE integrator over a functional
# state → state update.  The three §18 commitments, realized here:
#
#   1. The stepper is state → state under a pluggable update rule
#      ([`step`](@ref) on an [`AbstractIntegrator`](@ref)); time is a bare
#      loop parameter — `t = k·dt` is bookkeeping for "which state comes
#      next" and NOTHING more.  Every physical-time interpretation (proper
#      time, internal phase, observed duration, dilation) is OBSERVER-computed
#      model content, never engine-asserted.
#   2. The per-step observer/diagnostic hook is first-class: named callbacks
#      `(t, state) -> value` recorded into the [`Trajectory`](@ref).
#   3. Spacetime-base problems BYPASS the stepper: a Lorentzian spacetime
#      grid is just another base used with the operator stack directly (§13,
#      §18) — the stepper is one driver of dynamics, not its definition.
#
# THE EVOLVED SYSTEM (potential-first by Hamiltonian structure, §14/§19):
# linear Maxwell in temporal gauge as the first-order pair
#
#     ∂ₜA = E,        ∂ₜE = −δ(d(A))  [+ J static]
#
# The STATE is the potential and its conjugate momentum; `F = dA` is a
# per-step *derived diagnostic*, never stored state.  The spatial operator
# `K = δd` is — by the L8.2.1 adjointness contract — self-adjoint and
# positive-semidefinite (`⟨α,Kα⟩ = ⟨dα,dα⟩ ≥ 0`), giving the conserved energy
# `H = ½(⟨E,E⟩ + ⟨dA,dA⟩)` and real frequencies `ω² = λ ≥ 0`.
#
# YEE IS DERIVED, NOT CHOSEN (§19): the classic Yee/FDTD scheme is exactly
# (DEC spatial discretization on staggered primal/dual cells — already ours)
# ∘ (leapfrog, which staggers the pair a half-step in time).  Plugging
# [`Leapfrog`](@ref) into [`maxwell_system`](@ref) *yields* Yee; there is no
# scheme fork to choose.
#
# ONE SOURCE OF TRUTH: an [`EvolutionSystem`](@ref) declares each variable's
# time derivative as a BLACKBOARD expression in `FieldVar`s naming the state
# variables.  Construction typechecks every right-hand side against its
# target variable through the blackboard's own (grade, residence, base)
# checker (an `Equation` per variable), and stepping evaluates the SAME
# expression objects via `evaluate(expr; bindings)` against the current
# state — the simulated equation IS the typechecked equation object, from
# blackboard to trajectory.
#
# PERFORMANCE POSTURE (§19 — three named optimization seams, deliberately
# NOT built; do not optimize before profiling):
#   seam 1: dense contiguous field layout for evolution (sparse dicts are
#           right for algebra, wrong for stepping every cell every tick);
#   seam 2: one-time assembly of the linear spatial operators into sparse
#           matrices (stepping = matvec, the natural threading/GPU unit);
#   seam 3: threading the matvec.
# Each seam is reachable because `step` does FUNCTIONAL whole-field updates
# (returns a new state, never mutates) with per-cell-independent operator
# structure — the parallelism-ready shape.  v1 is single-threaded on the
# existing Dict fields, adequate for the validation grids.
#
# TWO-MODE RING DUALITY (§19): the exact ring `Rational{BigInt}` is a
# SHORT-trajectory oracle by arithmetic, not by laziness — iterating a linear
# map compounds denominators ~exponentially in steps.  `Float64` through the
# SAME ring-generic code path is the workhorse.  Symbolic `R` is NOT
# supported for stepping (see [`evolve`](@ref)).
#
# EXPLICITLY OUT OF SCOPE (§19): time-dependent/driven sources and input-power
# bookkeeping (the named follow-on phase; a STATIC `J` is permitted because it
# costs one keyword); the cup product / nonlinear terms / energy density
# (post-L10, §17.1); adaptive timestepping and further integrators (the
# pluggable interface suffices); UI/rendering (L11); any QRCS/extended-ED
# model content — the observer hook is the seam, the physics is the user's.

# ── SimState ──────────────────────────────────────────────────────────────────

"""
    SimState(base::BaseSpace, values::AbstractDict{Symbol,<:Field})
    SimState(system::EvolutionSystem, values::AbstractDict{Symbol,<:Field})

An immutable named collection of same-base [`Field`](@ref)s — the evolved
state (e.g. `:A => potential, :E => momentum`).  Access by name with
`state[:A]`; iterate names with `keys`.  All fields must live over the same
base instance.  The system-validating constructor additionally checks the
names and grades against the [`EvolutionSystem`](@ref)'s declared variables
(an `ArgumentError` names any disagreement).

The state is treated as a VALUE: [`step`](@ref) returns a new `SimState` and
never mutates its input (the parallelism-ready functional shape, §19).  The
internal `Dict` is copied at construction; do not mutate it.

Equality is exact-ring fieldwise equality (the `Field` convention: over a
symbolic ring compare cellwise with `isequal_simplified` instead).
"""
struct SimState{R, E<:AbstractTensorElement{R}, B<:BaseSpace}
    base   :: B
    names  :: Vector{Symbol}                 # sorted; canonical iteration order
    fields :: Dict{Symbol, Field{R,E,B}}

    function SimState{R,E,B}(base::B, names::Vector{Symbol},
                             fields::Dict{Symbol,Field{R,E,B}}
                             ) where {R, E<:AbstractTensorElement{R}, B<:BaseSpace}
        for (nm, f) in fields
            f.base === base || throw(ArgumentError(
                "state field :$nm lives over a different base instance than the state"))
        end
        new{R,E,B}(base, names, fields)
    end
end

function SimState(base::B, values::AbstractDict{Symbol,Field{R,E,B}}
                  ) where {R, E<:AbstractTensorElement{R}, B<:BaseSpace}
    isempty(values) && throw(ArgumentError(
        "a SimState needs at least one named field"))
    fields = Dict{Symbol,Field{R,E,B}}(values)
    SimState{R,E,B}(base, sort!(collect(keys(fields))), fields)
end

function Base.getindex(s::SimState, name::Symbol)
    haskey(s.fields, name) || throw(ArgumentError(
        "the state has no field named :$name; it has $(s.names)"))
    s.fields[name]
end

Base.keys(s::SimState)   = s.names
Base.length(s::SimState) = length(s.names)

state_names(s::SimState) = s.names

function Base.:(==)(a::SimState{R,E,B}, b::SimState{R,E,B}) where {R,E,B}
    a.base === b.base && a.names == b.names || return false
    all(a.fields[n] == b.fields[n] for n in a.names)
end

function Base.show(io::IO, s::SimState)
    parts = ["$n (grade $(field_grade(s.fields[n])))" for n in s.names]
    print(io, "SimState(", join(parts, ", "), " over ", nameof(typeof(s.base)), ")")
end

# ── EvolutionSystem ───────────────────────────────────────────────────────────

# Collect every FieldVar leaf of a blackboard expression (the RHS's free
# variables), through the open generic traversal interface.
function _free_field_vars(e::BlackboardExpr)
    out = FieldVar[]
    _collect_field_vars!(out, e)
    out
end

function _collect_field_vars!(out::Vector{FieldVar}, e::BlackboardExpr)
    e isa FieldVar && push!(out, e)
    for k in expr_children(e)
        _collect_field_vars!(out, k)
    end
    out
end

"""
    EvolutionSystem(base::BaseSpace, eqs::AbstractVector{<:Pair})

A first-order evolution system: each pair `var => rhs` declares a state
variable (a primal [`FieldVar`](@ref) over `base`) and its time derivative
`∂ₜvar = rhs`, where `rhs` is a blackboard expression in `FieldVar`s naming
the state variables (a concrete `Field` is accepted as a literal term).

Construction **typechecks** through the blackboard's own checker: for each
variable an `Equation(var, rhs)` is built, so the RHS's (grade, residence,
base) must match its target variable's exactly — the blackboard's informative
errors are surfaced with the variable named.  Additionally every `FieldVar`
appearing in any RHS must be one of the declared state variables (same
declaration, not just the same name), so an unbound or inconsistently-typed
variable is rejected at construction, never at step time.  These typechecked
[`Equation`](@ref)s are stored and retrievable via [`equations`](@ref): the
simulated equation IS the typechecked equation object (§19, one source of
truth).

Time never appears in an RHS: the system is autonomous by construction —
time-dependent (driven) sources are the named follow-on phase (§19).
"""
struct EvolutionSystem{B<:BaseSpace}
    base      :: B
    names     :: Vector{Symbol}              # declaration order
    vars      :: Dict{Symbol, FieldVar}
    rhs       :: Dict{Symbol, BlackboardExpr}
    equations :: Vector{Equation}

    function EvolutionSystem(base::B, eqs::AbstractVector{<:Pair}
                             ) where {B<:BaseSpace}
        isempty(eqs) && throw(ArgumentError(
            "an EvolutionSystem needs at least one variable; got none"))
        names = Symbol[]
        vars  = Dict{Symbol,FieldVar}()
        rhs   = Dict{Symbol,BlackboardExpr}()
        eqns  = Equation[]
        for p in eqs
            v = p.first
            v isa FieldVar || throw(ArgumentError(
                "each left-hand side must be a FieldVar (the state variable); " *
                "got $(typeof(v))"))
            v.base === base || throw(ArgumentError(
                "state variable :$(v.name) is declared over a different base " *
                "instance than the system's"))
            v.residence === :primal || throw(ArgumentError(
                "state variable :$(v.name) must be primal — the evolved state " *
                "is a collection of primal Fields"))
            haskey(vars, v.name) && throw(ArgumentError(
                "duplicate state variable :$(v.name)"))
            r = _as_expr(p.second)
            # The blackboard's three-way (grade, residence, base) check, with
            # the offending variable named in the surfaced error.
            eq = try
                Equation(v, r)
            catch err
                err isa ArgumentError || rethrow()
                throw(ArgumentError(
                    "the right-hand side declared for ∂ₜ$(v.name) is rejected " *
                    "by the blackboard typecheck: $(err.msg)"))
            end
            push!(names, v.name)
            vars[v.name] = v
            rhs[v.name]  = r
            push!(eqns, eq)
        end
        # Every free variable of every RHS must be a declared state variable —
        # the very FieldVar declared, so grades/residence cannot disagree.
        for nm in names, fv in _free_field_vars(rhs[nm])
            haskey(vars, fv.name) || throw(ArgumentError(
                "∂ₜ$nm references FieldVar :$(fv.name), which is not a state " *
                "variable of this system (variables: $names)"))
            fv == vars[fv.name] || throw(ArgumentError(
                "∂ₜ$nm references a FieldVar :$(fv.name) that differs from the " *
                "declared state variable :$(fv.name) (grade, residence, or base)"))
        end
        new{B}(base, names, vars, rhs, eqns)
    end
end

"""
    equations(sys::EvolutionSystem) -> Vector{Equation}

The typechecked blackboard [`Equation`](@ref)s `var = rhs` (read: `∂ₜvar =
rhs`) the system was constructed from — the one source of truth the stepper
evaluates (§19).
"""
equations(sys::EvolutionSystem) = sys.equations

function Base.show(io::IO, sys::EvolutionSystem)
    print(io, "EvolutionSystem(")
    for (i, n) in enumerate(sys.names)
        i > 1 && print(io, "; ")
        print(io, "∂ₜ", n, " = ", sys.rhs[n])
    end
    print(io, ")")
end

# A state matches a system iff the name sets agree and each field matches its
# declared variable's grade (base identity is checked by both constructors).
function _check_state(sys::EvolutionSystem, state::SimState)
    state.base === sys.base || throw(ArgumentError(
        "the state lives over a different base instance than the system"))
    Set(state.names) == Set(sys.names) || throw(ArgumentError(
        "state/system variable mismatch: the state has $(state.names) but the " *
        "system declares $(sort(sys.names))"))
    for n in sys.names
        field_grade(state[n]) == sys.vars[n].grade || throw(ArgumentError(
            "state field :$n has grade $(field_grade(state[n])) but the system " *
            "declares grade $(sys.vars[n].grade)"))
    end
    nothing
end

function SimState(sys::EvolutionSystem, values::AbstractDict{Symbol,<:Field})
    s = SimState(sys.base, values)
    _check_state(sys, s)
    s
end

# ── The Maxwell pair ──────────────────────────────────────────────────────────

"""
    maxwell_system(grid::BaseSpace; J::Union{Nothing,Field} = nothing)
        -> EvolutionSystem

Source-free linear Maxwell in temporal gauge as the first-order pair
(DESIGN.md §19, potential-first by Hamiltonian structure — §14 dynamically
realized):

```julia
∂ₜA = E
∂ₜE = −δ(d(A))   [+ J]
```

The state is the grade-1 potential `:A` and its conjugate momentum `:E`; the
field strength `F = dA` is a per-step *derived diagnostic*, never stored
state.  The spatial operator `K = δd` is self-adjoint positive-semidefinite
by the L8.2.1 adjointness contract, so `H = ½(⟨E,E⟩ + ⟨dA,dA⟩)` is the
conserved energy (see [`energy_observer`](@ref)) and `δE` is conserved
exactly by nilpotency (see [`gauss_observer`](@ref)).

A **static** current `J` (a grade-1 `Field` over `grid`) may be supplied —
it costs one literal term in the `:E` equation.  Time-dependent/driven
sources and their input-power bookkeeping are deliberately the follow-on
phase (§19).

Requires `can_hodge(grid)` (through `δ`; the blackboard node construction
surfaces the gating error on a bare graph).

!!! note "Yee is derived, not chosen"
    DEC spatial discretization + [`Leapfrog`](@ref) on this pair *is* the
    classic Yee/FDTD scheme (§19): the staggered primal/dual placement is the
    complex's, and the half-step time stagger is the integrator's.  No scheme
    is hand-coded here.
"""
function maxwell_system(grid::BaseSpace; J::Union{Nothing,Field} = nothing)
    A = FieldVar(:A, grid, 1)
    E = FieldVar(:E, grid, 1)
    force = -(codifferential(grid, d(A)))
    J === nothing || (force = force + J)
    EvolutionSystem(grid, [A => E, E => force])
end

# ── Integrators ───────────────────────────────────────────────────────────────

"""
    AbstractIntegrator

Abstract supertype of the pluggable update rules (§18: "state + pluggable
update rule").  An integrator implements

```julia
step(integrator, system, state, dt) -> SimState
```

**functionally**: a NEW state is returned and the input state is never
mutated — the parallelism-ready shape behind the §19 optimization seams.
Shipped: [`ForwardEuler`](@ref) (reference) and [`Leapfrog`](@ref)
(validated default).  Further integrators (adaptive, higher-order) plug in
by subtyping; none ship in v1 (§19).
"""
abstract type AbstractIntegrator end

"""
    ForwardEuler()

The reference explicit Euler rule: every variable is advanced simultaneously
from the OLD state, `xᵢ' = xᵢ + dt·RHSᵢ(state)`.  Works for any
[`EvolutionSystem`](@ref) shape.

**Honesty note (§19): unconditionally unstable for oscillatory systems.**
For a mode of frequency `ω` the energy grows by the factor `(1 + dt²ω²)`
*every step*, for every `dt > 0` — there is no stable timestep.  It is
shipped as plumbing proof and as the contrast case of the validation battery,
not for production wave evolution; use [`Leapfrog`](@ref).
"""
struct ForwardEuler <: AbstractIntegrator end

"""
    Leapfrog()

Störmer–Verlet leapfrog in **kick-drift-kick (KDK / velocity-Verlet)** form,
for a partitioned pair (position-like `x`, momentum-like `p`):

```julia
p½ = p  + (dt/2)·F(x)        # half kick   (F = the declared RHS of p)
x' = x  + dt·p½              # full drift  (∂ₜx = p exactly)
p' = p½ + (dt/2)·F(x')       # half kick
```

The system must be a **recognizable partitioned pair**: exactly two
variables, one (`x`) whose RHS is exactly the other's bare `FieldVar`
(`∂ₜx = p`), and the other (`p`) whose RHS is independent of `p`.  Anything
else throws an informative `ArgumentError` rather than silently degrading
(honest mechanisms) — use [`ForwardEuler`](@ref) or add an integrator for
non-partitioned systems.

**Properties (§19):** symplectic and time-reversible.  Being symplectic, it
exactly conserves a *shadow* Hamiltonian `H̃` close to the true `H`, so the
true energy **oscillates boundedly with zero secular drift** — "energy
conserved" is those two precise claims, not a tolerance judgment.  For the
linear pair `∂ₜx = p, ∂ₜp = −Kx` the per-mode shadow energy is
`H̃ = ½p² + ½λ(1 − dt²λ/4)x²`, conserved to roundoff.

**Stability (CFL) bound:** stable iff `dt²·λ_max ≤ 4`, where `λ_max` is the
largest eigenvalue of the spatial operator (`K = δd` for Maxwell).  Above
the bound the highest mode amplifies exponentially; demonstrating both sides
of the boundary is part of the validation battery.

Requires the state's ring to support division by `R(2)` for the half-kick
(every shipped ring does).
"""
struct Leapfrog <: AbstractIntegrator end

integrator_name(::ForwardEuler) = :ForwardEuler
integrator_name(::Leapfrog)     = :Leapfrog

# Convert a user-supplied dt into the state's scalar ring, informatively.
function _ring_scalar(::Type{R}, dt) where R
    dt isa R && return dt
    try
        return R(dt)
    catch
        throw(ArgumentError(
            "dt of type $(typeof(dt)) cannot be represented in the state's " *
            "scalar ring $R; pass dt in the ring (e.g. 1//10 over " *
            "Rational{BigInt}, 0.01 over Float64)"))
    end
end

_bindings(state::SimState) =
    Dict{Symbol,Any}(n => state.fields[n] for n in state.names)

# Identify the (position, momentum) names of a partitioned pair, or throw.
function _partitioned_pair(sys::EvolutionSystem)
    shape = "Leapfrog requires a partitioned two-variable pair: a position-" *
            "like x with ∂ₜx = p EXACTLY (the bare FieldVar) and a momentum-" *
            "like p whose right-hand side is independent of p"
    length(sys.names) == 2 || throw(ArgumentError(
        "$shape; this system declares $(length(sys.names)) variable(s): " *
        "$(sys.names)"))
    for (x, p) in ((sys.names[1], sys.names[2]), (sys.names[2], sys.names[1]))
        rx = sys.rhs[x]
        (rx isa FieldVar && rx.name === p) || continue
        if all(fv -> fv.name !== p, _free_field_vars(sys.rhs[p]))
            return (x, p)
        end
    end
    throw(ArgumentError(
        "$shape; got ∂ₜ$(sys.names[1]) = $(sys.rhs[sys.names[1]]) and " *
        "∂ₜ$(sys.names[2]) = $(sys.rhs[sys.names[2]])"))
end

import Base: step

"""
    step(integrator::AbstractIntegrator, system::EvolutionSystem,
         state::SimState, dt) -> SimState

Advance the state by one step of size `dt` under the integrator's update
rule.  **Functional:** returns a new [`SimState`](@ref); the input state is
never mutated (§19's parallelism-ready shape).  The right-hand sides are the
system's typechecked blackboard expressions, evaluated against the current
state via the blackboard's `evaluate(expr; bindings)` — the same verified
operator path the rest of the suite certifies.

This extends `Base.step` (ranges) with the integrator methods; the two uses
do not collide.
"""
function step(::ForwardEuler, sys::EvolutionSystem{B}, state::SimState{R,E,B},
              dt) where {R,E,B}
    _check_state(sys, state)
    dtR = _ring_scalar(R, dt)
    b = _bindings(state)
    fields = Dict{Symbol,Field{R,E,B}}()
    for n in sys.names
        rhsval = evaluate(sys.rhs[n]; bindings = b)::Field{R,E,B}
        fields[n] = state.fields[n] + dtR * rhsval
    end
    SimState{R,E,B}(state.base, state.names, fields)
end

function step(::Leapfrog, sys::EvolutionSystem{B}, state::SimState{R,E,B},
              dt) where {R,E,B}
    _check_state(sys, state)
    x, p = _partitioned_pair(sys)
    dtR = _ring_scalar(R, dt)
    halfdt = dtR * (one(R) / R(2))
    force = sys.rhs[p]

    xf = state.fields[x]
    pf = state.fields[p]
    f0 = evaluate(force; bindings = _bindings(state))::Field{R,E,B}
    phalf = pf + halfdt * f0                                   # half kick
    xnew  = xf + dtR * phalf                                   # full drift
    f1 = evaluate(force;
                  bindings = Dict{Symbol,Any}(x => xnew, p => phalf))::Field{R,E,B}
    pnew = phalf + halfdt * f1                                 # half kick
    SimState{R,E,B}(state.base, state.names,
                    Dict{Symbol,Field{R,E,B}}(x => xnew, p => pnew))
end

# ── Observers (§18's per-step hook) ───────────────────────────────────────────

_ring(::Field{R,E,B}) where {R,E,B} = R

"""
    energy_observer(b::BaseSpace; position = :A, momentum = :E) -> Function

The built-in total-energy observer for a partitioned pair: the callback
`(t, state) -> H` with

```julia
H = ½(⟨E,E⟩ + ⟨dA,dA⟩)
```

via the L9.1 [`inner_product`](@ref).  Note `⟨dA,dA⟩`: the field strength is
**derived from the potential per step**, never read from stored state (§14,
§19 — `F = dA` is a diagnostic).  `K = δd` being self-adjoint
positive-semidefinite (L8.2.1) makes this the genuine conserved Hamiltonian
of [`maxwell_system`](@ref).

An observer is just a named callback `(t, state) -> value`; this builder
exists for convenience.  `t` is the bare loop parameter (§18) — any
physical-time interpretation belongs to user observers plugged into the same
hook.
"""
function energy_observer(b::BaseSpace; position::Symbol = :A, momentum::Symbol = :E)
    function (t, state::SimState)
        A = state[position]
        E = state[momentum]
        R = _ring(A)
        dA = d(A)
        (one(R) / R(2)) * (inner_product(b, E, E) + inner_product(b, dA, dA))
    end
end

"""
    gauss_observer(b::BaseSpace; momentum = :E) -> Function

The built-in Gauss-constraint observer: the callback `(t, state) -> δE`,
returning **the field `δE` itself** (not a scalar — the field is the
strictly stronger record; reduce with [`field_norm2`](@ref) if a scalar
series is wanted).

For [`maxwell_system`](@ref), `∂ₜ(δE) = −δδ(dA) = 0` **by nilpotency**: the
conservation is an algebraic identity of the signed incidence sums (§19) —
exact over exact rings, true to roundoff over `Float64` — so `δE = ρ` is
preserved by construction, independent of integrator and timestep.
"""
function gauss_observer(b::BaseSpace; momentum::Symbol = :E)
    (t, state::SimState) -> codifferential(b, state[momentum])
end

# ── Trajectory and the loop ───────────────────────────────────────────────────

"""
    Trajectory

The record returned by [`evolve`](@ref): integrator name, `dt`, `nsteps`,
the sample times `times` (the bare loop parameter `t = k·dt`, `k = 0:nsteps`
— §18), the per-observer series `observations :: Dict{Symbol,Vector{Any}}`
(one sample per recorded time; access a series with `traj[:name]`), the
`initial_state` and `final_state`, and — only when `evolve` was asked with
`record_states = true` — every intermediate state in `states` (default off:
memory).
"""
struct Trajectory{R, S<:SimState}
    integrator    :: Symbol
    dt            :: R
    nsteps        :: Int
    times         :: Vector{R}
    observations  :: Dict{Symbol, Vector{Any}}
    initial_state :: S
    final_state   :: S
    states        :: Union{Nothing, Vector{S}}
end

function Base.getindex(tr::Trajectory, name::Symbol)
    haskey(tr.observations, name) || throw(ArgumentError(
        "the trajectory recorded no observer named :$name; it has " *
        "$(sort(collect(keys(tr.observations))))"))
    tr.observations[name]
end

function Base.show(io::IO, tr::Trajectory)
    print(io, "Trajectory(", tr.integrator, ", dt = ", tr.dt, ", ",
          tr.nsteps, " steps, observers = ",
          sort(collect(keys(tr.observations))),
          tr.states === nothing ? "" : ", states recorded", ")")
end

"""
    evolve(system::EvolutionSystem, state0::SimState,
           integrator::AbstractIntegrator, dt, nsteps::Integer;
           observers = Pair{Symbol,Any}[], record_states = false) -> Trajectory

Run the method-of-lines loop: starting from `state0`, apply
`step(integrator, system, ·, dt)` `nsteps` times, recording every named
observer `(t, state) -> value` at `t = 0` and after every step
(`nsteps + 1` samples per series).  Returns a [`Trajectory`](@ref).

**Time is a bare loop parameter (§18):** `t = k·dt` is bookkeeping for
"which state comes next" and carries no physical-time assertion.  Proper
time, internal phase, observed duration — every such interpretation is
computed by observers plugged into this hook, never by the engine.
Spacetime-base problems (time as a literal Lorentzian grid dimension) bypass
this stepper entirely and use the operator stack directly (§18).

`observers` is a collection of `name::Symbol => callback` pairs; a
user-supplied callback is just another observer (this is the seam where,
e.g., per-pattern internal-phase tracking later plugs in).

**Ring regimes (§19):** exact rings are short-trajectory oracles —
iterating a linear map over `Rational{BigInt}` compounds denominators
roughly exponentially in the step count, so ten exact steps are cheap and
ten thousand are impossible regardless of optimization; `Float64` through
the same ring-generic code path is the long-run workhorse.  Symbolic `R`
(`Symbolics.Num`) is deliberately NOT supported for stepping: each step
substitutes the previous step's expressions into the RHS, growing the
symbolic state without bound — the symbolic ring's role is the
blackboard/compile path (coefficient-level symbolics on equations), not
trajectories.
"""
function evolve(sys::EvolutionSystem, state0::SimState{R,E,B},
                integrator::AbstractIntegrator, dt, nsteps::Integer;
                observers = Pair{Symbol,Any}[],
                record_states::Bool = false) where {R,E,B}
    nsteps >= 0 || throw(ArgumentError("nsteps must be ≥ 0, got $nsteps"))
    _check_state(sys, state0)
    dtR = _ring_scalar(R, dt)
    obs = collect(Pair{Symbol,Any}, observers)
    allunique(first.(obs)) || throw(ArgumentError(
        "observer names must be unique; got $(first.(obs))"))

    times  = [zero(dtR)]
    series = Dict{Symbol,Vector{Any}}(n => Any[] for (n, _) in obs)
    record = (t, st) -> for (n, f) in obs
        push!(series[n], f(t, st))
    end

    state = state0
    record(zero(dtR), state)
    states = record_states ? [state0] : nothing
    for k in 1:Int(nsteps)
        state = step(integrator, sys, state, dtR)
        t = R(k) * dtR
        push!(times, t)
        record(t, state)
        record_states && push!(states, state)
    end
    Trajectory{R,typeof(state0)}(integrator_name(integrator), dtR, Int(nsteps),
                                 times, series, state0, state, states)
end

# ── Exports ───────────────────────────────────────────────────────────────────

export SimState, state_names,
       EvolutionSystem, equations, maxwell_system,
       AbstractIntegrator, ForwardEuler, Leapfrog, integrator_name,
       energy_observer, gauss_observer,
       evolve, Trajectory
