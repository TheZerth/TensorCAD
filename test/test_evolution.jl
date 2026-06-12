# ─────────────────────────────────────────────────────────────────────────────
# Phase L10: the simulation engine — the §19 validation battery.
#
# Venue for the oracle: GridBase(2,1).  The spatial operator of the Maxwell
# pair is K = δd on grade-1 cochains; its NONZERO spectrum equals that of
# M₂ = dδ on grade-2 cochains (the SVD pairing: if M₂u = λu with λ > 0 then
# Φ = δu satisfies KΦ = δdδu = δ(M₂u) = λΦ).  On GridBase(2,1) the face row
# has M₂ = [[4,-1],[-1,4]] (each face has 4 boundary edges; adjacent faces
# share one vertical edge with opposite signs), so the exact RATIONAL
# eigenpairs are u = (1,1) ↦ λ = 3 and u = (1,-1) ↦ λ = 5 = λ_max.  Both
# eigenmodes Φ = δu have small-integer entries — exact over Rational{BigInt}.
#
# Why not the committed λ = 2−2cos(π/N) mode of test_maxwell.jl: that mode is
# F = dA₀ (A₀ the vertex cosine), an eigenvector of dδ at grade 1 — it lies
# in ker(δd), the GAUGE sector of the A-equation (K(dφ) = δddφ = 0), so as an
# initial `A` it is static, not oscillatory.  The gauge-freeze fact is itself
# asserted below; the oscillation oracle uses the co-exact δu modes.
# ─────────────────────────────────────────────────────────────────────────────

R = Rational{BigInt}

# The λ=3 (u = (1,1)) or λ=5 (u = (1,-1)) co-exact eigenmode of K = δd on a
# GridBase(2,1) with metric m, as a grade-1 field.
function _ev_mode(grid, m, sign2)
    o = clifford_one(m)
    u = Field(grid, 2, Dict(1 => o, 2 => sign2 * o))
    codifferential(grid, u)
end

_ev_zeroE(grid) = zero_field(grid, 1, fibre(grid, 1, 1))

@testset "L10 test 1: closed-form eigenmode oracle, exact over Rational{BigInt}" begin
    grid = GridBase(2, 1)
    m = grid.metric
    sys = maxwell_system(grid)

    # Eigenmode preconditions (the oracle's λ is verified, not assumed).
    Φ = _ev_mode(grid, m, one(R))
    λ = R(3)
    @test codifferential(grid, d(Φ)) == 3 * Φ
    @test length(Φ) == 6                       # middle vertical edge cancels
    @test field_norm2(grid, Φ) == 6

    # Gauge sector: any exact A = dφ has KA = 0 (d∘d = 0), so the state is
    # frozen — this is where the committed λ = 2−2cos(π/N) mode lives.
    φ = Field(grid, 0, Dict(1 => clifford_one(m), 4 => R(2) * clifford_one(m)))
    Ag = d(φ)
    @test length(codifferential(grid, d(Ag))) == 0
    gtraj = evolve(sys, SimState(sys, Dict(:A => Ag, :E => _ev_zeroE(grid))),
                   Leapfrog(), 1 // 10, 3; record_states = true)
    @test all(st[:A] == Ag && length(st[:E]) == 0 for st in gtraj.states)

    # HAND-DERIVED KDK RECURRENCE (the external ground truth).  On the
    # eigenmode, A = a·Φ, E = e·Φ, and kick-drift-kick with force F = −λa·Φ
    # reduces to the exact 2×2 map (z = dt²λ):
    #
    #   e½    = e − (dt·λ/2)·a
    #   a_{k+1} = a + dt·e½                = (1 − z/2)·a_k + dt·e_k
    #   e_{k+1} = e½ − (dt·λ/2)·a_{k+1}    = −dt·λ·(1 − z/4)·a_k + (1 − z/2)·e_k
    #
    # (determinant = (1−z/2)² + dt²λ(1−z/4) = 1 exactly: symplectic.)
    # The comparison values below come from THIS recurrence, not from running
    # the integrator twice.
    dt = R(1 // 10)
    z = dt^2 * λ
    nsteps = 6
    state0 = SimState(sys, Dict(:A => Φ, :E => _ev_zeroE(grid)))
    traj = evolve(sys, state0, Leapfrog(), dt, nsteps;
                  observers = [:H => energy_observer(grid)],
                  record_states = true)

    a, e = one(R), zero(R)
    # H on the mode: ⟨E,E⟩ = e²·‖Φ‖², ⟨dA,dA⟩ = a²·⟨Φ,KΦ⟩ = a²·λ‖Φ‖² (adjointness),
    # so H = ½‖Φ‖²(e² + λa²) = 3e² + 9a² exactly.
    @test traj[:H][1] == 3 * e^2 + 9 * a^2
    for k in 1:nsteps
        anew = (1 - z / 2) * a + dt * e
        enew = -dt * λ * (1 - z / 4) * a + (1 - z / 2) * e
        a, e = anew, enew
        @test traj.states[k + 1][:A] == a * Φ
        @test traj.states[k + 1][:E] == e * Φ
        @test traj[:H][k + 1] == 3 * e^2 + 9 * a^2
    end
    @test traj.final_state == traj.states[end]
end

@testset "L10 test 2: shadow energy — bounded oscillation, zero secular drift" begin
    fgrid = GridBase(2, 1; metric = signature_metric(VectorSpace(2), Float64, 2, 0, 0))
    fm = fgrid.metric
    fsys = maxwell_system(fgrid)
    Φ = _ev_mode(fgrid, fm, 1.0)               # λ = 3 mode
    λ = 3.0
    @test codifferential(fgrid, d(Φ)) == 3 * Φ # exact small-integer float arithmetic

    # dt at half the CFL limit: z = dt²λ = 1, ζ = z/4 = 1/4.  ALL modes are
    # stable at this dt (λ_max = 5: dt²·5 = 5/3 < 4), so roundoff leakage into
    # the other mode stays bounded.
    dt = 0.5 * (2 / sqrt(λ))
    ζ = dt^2 * λ / 4
    nsteps = 10_000
    state0 = SimState(fsys, Dict(:A => Φ, :E => _ev_zeroE(fgrid)))
    traj = evolve(fsys, state0, Leapfrog(), dt, nsteps;
                  observers = [:H => energy_observer(fgrid),
                               :G => gauss_observer(fgrid)])
    H = Float64.(traj[:H])
    H0 = H[1]
    @test H0 ≈ 9.0                              # ½·λ·‖Φ‖² = ½·3·6

    # BAND (derived): KDK conserves the shadow energy H̃ = ½‖Φ‖²(e² + λ(1−ζ)a²)
    # exactly; with e₀ = 0, H̃₀ = (1−ζ)H₀.  The true H = H̃₀ + ζ·½λ‖Φ‖²a² and
    # ½λ(1−ζ)a² ≤ H̃₀ bounds ½λa² ≤ H₀, so H ∈ [(1−ζ)H₀, H₀] — bounded
    # oscillation, the band fixed by dt, no tolerance judgment.
    tol = 1e-9 * H0
    @test all(h -> (1 - ζ) * H0 - tol <= h <= H0 + tol, H)

    # NO SECULAR DRIFT: H oscillates with period ~3 steps (cos θ = 1 − z/2 =
    # 0.5 ⟹ θ ≈ π/3, H period 2θ); a 1000-sample decile averages ~333 periods,
    # so decile means agree to O(amplitude·period/1000) ≈ 2.3e-3·amplitude ≈
    # 5e-3 absolute.  Assert well inside 1e-3·H₀ — and compare with the
    # ForwardEuler growth below to see what drift would look like.
    mean_first = sum(H[1:1000]) / 1000
    mean_last  = sum(H[end-999:end]) / 1000
    @test abs(mean_last - mean_first) < 1e-3 * H0

    # Gauss along the run: E stays in the mode span, δE ~ 0 to roundoff².
    @test all(g -> field_norm2(fgrid, g) <= 1e-20, traj[:G])

    # CONTRAST — ForwardEuler honestly failing: on a frequency-ω mode the
    # energy grows by the factor (1 + dt²λ) EVERY step (derived: e'² + λa'² =
    # (1 + dt²λ)(e² + λa²) identically), here a clean doubling (z = 1).
    ftraj = evolve(fsys, state0, ForwardEuler(), dt, 200;
                   observers = [:H => energy_observer(fgrid)])
    HF = Float64.(ftraj[:H])
    @test all(k -> HF[k + 1] > HF[k], 1:length(HF)-1)       # monotone growth
    @test all(k -> abs(HF[k + 1] / HF[k] - (1 + dt^2 * λ)) < 1e-6, 1:length(HF)-1)
    @test HF[end] > 1e10 * H0
end

@testset "L10 test 3: CFL boundary, both sides, computed from the verified λ" begin
    fgrid = GridBase(2, 1; metric = signature_metric(VectorSpace(2), Float64, 2, 0, 0))
    fm = fgrid.metric
    fsys = maxwell_system(fgrid)

    # The λ_max = 5 mode (u = (1,-1)), so that on the UNDER side every mode of
    # the operator is stable (dt²λ_max < 4 ⟹ dt²·3 < 4 too) and roundoff
    # leakage cannot blow up a long run.
    Φ5 = _ev_mode(fgrid, fm, -1.0)
    @test codifferential(fgrid, d(Φ5)) == 5 * Φ5
    @test field_norm2(fgrid, Φ5) == 10.0
    λ = 5.0
    dtcrit = 2 / sqrt(λ)                        # computed, not tuned
    state0 = SimState(fsys, Dict(:A => Φ5, :E => _ev_zeroE(fgrid)))
    H0 = 0.5 * λ * 10.0                         # ½λ‖Φ₅‖² = 25

    # UNDER: dt = 0.9·dt_crit (z = 3.24 < 4) — bounded forever (5000 steps),
    # inside the derived band [(1−ζ)H₀, H₀], ζ = z/4 = 0.81.
    dtu = 0.9 * dtcrit
    ζ = dtu^2 * λ / 4
    tru = evolve(fsys, state0, Leapfrog(), dtu, 5_000;
                 observers = [:H => energy_observer(fgrid)])
    Hu = Float64.(tru[:H])
    @test Hu[1] ≈ H0
    @test all(h -> (1 - ζ) * H0 - 1e-6 * H0 <= h <= H0 + 1e-6 * H0, Hu)

    # OVER: dt = 1.05·dt_crit (z = 4.41 > 4) — the initialized mode amplifies
    # exponentially (|μ| = (|tr| + √(tr²−4))/2 ≈ 1.88 per step); detect it fast.
    dto = 1.05 * dtcrit
    tro = evolve(fsys, state0, Leapfrog(), dto, 150;
                 observers = [:H => energy_observer(fgrid)])
    Ho = Float64.(tro[:H])
    @test Ho[end] > 1e6 * H0
    @test Ho[end] > Ho[76] > Ho[1]
end

@testset "L10 test 4: Gauss constraint conserved by nilpotency" begin
    # Exact: a generic state with δE ≠ 0.  E_{k+1} differs from E_k only by
    # multiples of δ(dA)-type fields, and δδ = 0 is an algebraic identity of
    # the signed sums — so δE is EXACTLY unchanged, step by step, for every
    # integrator and dt.
    grid = GridBase(2, 2)
    m = grid.metric
    sys = maxwell_system(grid)
    A0 = Field(grid, 1, Dict(Int(c) => R(c) * clifford_basis_vector(m, 1) +
                                      R(3 - c) * clifford_one(m)
                             for c in cells(grid, 1)))
    E0 = Field(grid, 1, Dict(Int(c) => R(c^2) * clifford_one(m) +
                                      R(c) * clifford_basis_vector(m, 2)
                             for c in cells(grid, 1)))
    @test length(codifferential(grid, E0)) > 0          # constraint nontrivial
    state0 = SimState(sys, Dict(:A => A0, :E => E0))
    for integ in (Leapfrog(), ForwardEuler())
        traj = evolve(sys, state0, integ, 1 // 7, 5;
                      observers = [:G => gauss_observer(grid)])
        gs = traj[:G]
        @test length(gs) == 6
        @test all(g == gs[1] for g in gs)               # exact, every step
    end

    # Float64, same shape: conserved to roundoff over a long run.
    fgrid = GridBase(2, 2; metric = signature_metric(VectorSpace(2), Float64, 2, 0, 0))
    fmet = fgrid.metric
    fsys = maxwell_system(fgrid)
    fA0 = Field(fgrid, 1, Dict(Int(c) => Float64(c) * clifford_basis_vector(fmet, 1)
                               for c in cells(fgrid, 1)))
    fE0 = Field(fgrid, 1, Dict(Int(c) => Float64(c^2) * clifford_one(fmet)
                               for c in cells(fgrid, 1)))
    ftraj = evolve(fsys, SimState(fsys, Dict(:A => fA0, :E => fE0)),
                   Leapfrog(), 0.05, 500;                # λ_max = 6 ⟹ dt_crit ≈ 0.82
                   observers = [:G => gauss_observer(fgrid)])
    g0 = ftraj[:G][1]
    scale = max(1.0, field_norm2(fgrid, g0))
    @test field_norm2(fgrid, g0) > 0
    @test all(g -> field_norm2(fgrid, g - g0) <= 1e-20 * scale, ftraj[:G])
end

@testset "L10 test 5: plumbing — typecheck, functional step, observers, metadata" begin
    grid = GridBase(2, 1)
    m = grid.metric
    Av = FieldVar(:A, grid, 1)
    Ev = FieldVar(:E, grid, 1)

    # Mis-graded RHS rejected with the blackboard's error, variable named.
    err = try; EvolutionSystem(grid, [Av => d(Av)]); nothing; catch e; e; end
    @test err isa ArgumentError && occursin("∂ₜA", err.msg) && occursin("grade", err.msg)

    # RHS referencing an undeclared variable is rejected at construction.
    err = try; EvolutionSystem(grid, [Av => FieldVar(:B, grid, 1)]); nothing; catch e; e; end
    @test err isa ArgumentError && occursin(":B", err.msg)

    # RHS referencing a same-named variable with a DIFFERENT declaration.
    err = try
        EvolutionSystem(grid, [Av => Ev,
                               Ev => -(codifferential(grid, FieldVar(:A, grid, 2)))])
        nothing
    catch e; e; end
    @test err isa ArgumentError && occursin("differs", err.msg)

    # Dual-residence state variables are rejected (state = primal Fields).
    Dv = FieldVar(:D, grid, 1; residence = :dual)
    @test_throws ArgumentError EvolutionSystem(grid, [Dv => Dv])

    # maxwell_system is gated by can_hodge through δ (bare graph refuses).
    graph = GraphBase(2, [(1, 2)]; metric = signature_metric(VectorSpace(2), R, 2, 0, 0))
    @test_throws ArgumentError maxwell_system(graph)

    sys = maxwell_system(grid)
    @test length(equations(sys)) == 2                  # the typechecked objects
    Φ = _ev_mode(grid, m, one(R))
    state0 = SimState(sys, Dict(:A => Φ, :E => _ev_zeroE(grid)))

    # SimState validation against the system.
    @test_throws ArgumentError SimState(sys, Dict(:A => Φ))             # missing :E
    bad = Field(grid, 0, Dict(1 => clifford_one(m)))
    @test_throws ArgumentError SimState(sys, Dict(:A => bad, :E => _ev_zeroE(grid)))

    # step is functional: new state out, inputs untouched (same objects).
    st1 = step(Leapfrog(), sys, state0, 1 // 10)
    @test st1 isa SimState && st1 !== state0
    @test state0.fields[:A] === Φ && state0[:A] == Φ
    @test length(state0[:E]) == 0
    @test st1[:A] != state0[:A]

    # @inferred on the step path (concrete SimState return type).
    @test (@inferred step(Leapfrog(), sys, state0, 1 // 10)) isa typeof(state0)
    @test (@inferred step(ForwardEuler(), sys, state0, 1 // 10)) isa typeof(state0)

    # Leapfrog refuses a non-partitioned shape (∂ₜA = 2E is not a bare drift).
    sys2 = EvolutionSystem(grid, [Av => 2 * Ev, Ev => -(codifferential(grid, d(Av)))])
    err = try; step(Leapfrog(), sys2, state0, 1 // 10); nothing; catch e; e; end
    @test err isa ArgumentError && occursin("partitioned", err.msg)
    # …but ForwardEuler handles any declared shape.
    @test step(ForwardEuler(), sys2, state0, 1 // 10) isa SimState

    # Static J costs one literal term: ∂ₜE = −δdA + J, verified against the
    # engine operators directly.
    Jf = Field(grid, 1, Dict(1 => R(2) * clifford_one(m)))
    sysJ = maxwell_system(grid; J = Jf)
    stJ = step(ForwardEuler(), sysJ, state0, 1 // 10)
    expectedE = state0[:E] + R(1 // 10) * (Jf - codifferential(grid, d(Φ)))
    @test stJ[:E] == expectedE

    # Observers: lengths, user-supplied callback, metadata, time = k·dt.
    traj = evolve(sys, state0, Leapfrog(), 1 // 10, 4;
                  observers = [:H => energy_observer(grid),
                               :t => (t, st) -> t])
    @test traj.integrator == :Leapfrog
    @test traj.dt == R(1 // 10) && traj.nsteps == 4
    @test length(traj.times) == 5 && length(traj[:H]) == 5
    @test traj.times == [R(k // 10) for k in 0:4]
    @test traj[:t] == traj.times                       # user observer works
    @test traj.states === nothing                      # record_states off by default
    @test_throws ArgumentError traj[:nope]
    @test_throws ArgumentError evolve(sys, state0, Leapfrog(), 1 // 10, -1)
    err = try; evolve(sys, state0, Leapfrog(), im, 3); nothing; catch e; e; end
    @test err isa ArgumentError && occursin("ring", err.msg)
end

@testset "L10 test 6: one ring-generic code path — exact oracle vs Float64" begin
    # The same maxwell_system + Leapfrog path over Rational{BigInt} and over
    # Float64, with a dt exact in BOTH rings (1//8 = 0.125 is a binary
    # rational), must produce the same trajectory up to float roundoff.
    # Symbolic R is deliberately NOT stepped: substituting each step's result
    # into the next RHS grows symbolic expressions without bound — the
    # symbolic ring's role is the blackboard/compile path, not trajectories
    # (documented at `evolve`).
    grid = GridBase(2, 1)
    fgrid = GridBase(2, 1; metric = signature_metric(VectorSpace(2), Float64, 2, 0, 0))
    sys, fsys = maxwell_system(grid), maxwell_system(fgrid)
    Φ = _ev_mode(grid, grid.metric, one(R))
    Φf = _ev_mode(fgrid, fgrid.metric, 1.0)
    tr = evolve(sys, SimState(sys, Dict(:A => Φ, :E => _ev_zeroE(grid))),
                Leapfrog(), 1 // 8, 5; observers = [:H => energy_observer(grid)])
    trf = evolve(fsys, SimState(fsys, Dict(:A => Φf, :E => _ev_zeroE(fgrid))),
                 Leapfrog(), 0.125, 5; observers = [:H => energy_observer(fgrid)])
    @test all(k -> abs(Float64(tr[:H][k]) - trf[:H][k]) < 1e-12, 1:6)
end
