# ─────────────────────────────────────────────────────────────────────────────
# Phase L10.1: variable diagonal Hodge weights — the gates.
#
#   Gate 1  weighted adjointness ⟨dα,β⟩_w == ⟨α,δβ⟩_w, exact, both grids,
#           both signatures, every grade transition, boundary included
#           (plus the structural fact δ_w == W_{k-1}⁻¹ dᵀ W_k).
#   Gate 2  ⋆⋆ sign law unchanged under non-unit weights (signs from
#           signature only; positive weights cancel through the inverse).
#   Gate 3  unit-weight regression: WeightedGridBase with default weights
#           reproduces the plain GridBase exactly (the full-suite half of the
#           gate is the suite run itself — no existing test is modified).
#   Gate 4  δ² = 0 exact with weights.
#   Gate 5  L10 leapfrog conserves the WEIGHTED H within the derived shadow
#           band; λ of the weighted operator hand-derived below.
#   Gate 6  the medium effect exists: a graded weight profile changes the
#           wave operator's local action (the Experiment-2 precondition; the
#           refraction run itself is E2's job, not this file's).
# ─────────────────────────────────────────────────────────────────────────────

R = Rational{BigInt}

_wb_metrics() = (
    ("Euclidean", signature_metric(VectorSpace(2), R, 2, 0, 0)),
    ("Lorentzian", signature_metric(VectorSpace(2), R, 1, 1, 0)),
)

# Generic mixed-blade field populating every k-cell (boundary always touched).
function _wb_field(b, m, g; seed::Int = 0)
    e1 = clifford_basis_vector(m, 1)
    e2 = clifford_basis_vector(m, 2)
    e12 = clifford_basis_element(m, [1, 2])
    o = clifford_one(m)
    Field(b, g, Dict(Int(c) =>
        R(c + seed) * o + R(2c - seed) * e1 + R(c^2 + seed) * e2 + R(7 - c) * e12
        for c in cells(b, g)))
end

# The weighted incidence transpose W_{g-1}⁻¹ dᵀ W_g — what δ_w must equal.
function _wb_weighted_dT(b::WeightedGridBase, β::Field)
    g = field_grade(β)
    E = typeof(evaluate(β, first(cells(b, g))))
    vals = Dict{Int,E}()
    for f in cells(b, g - 1)
        acc = zero_fibre(fibre(b, g - 1, f))
        for c in cells(b, g)
            for (face, s) in boundary(b, g, c)
                face == f && (acc = acc + s * (b.weights[g + 1][Int(c)] * evaluate(β, c)))
            end
        end
        acc = (one(R) / b.weights[g][Int(f)]) * acc
        iszero(acc) || (vals[Int(f)] = acc)
    end
    Field(b, g - 1, vals)
end

@testset "L10.1 construction: positivity, shapes, profile" begin
    grid = GridBase(2, 2)
    wb = WeightedGridBase(grid)                       # default unit
    @test all(all(isequal(w, one(R)) for w in ws) for ws in wb.weights)

    # the profile constructor and its lattice-coordinate convention
    wp = weight_profile(grid, (i, j) -> 1 + (i + j) // 4)
    @test wp.weights[1][1] == 1                       # vertex (0,0)
    @test wp.weights[3][end] == 1 + (1 + 1) // 4      # face (1,1)

    # positivity is enforced: weights are volumes, signature lives in the fibre
    @test_throws ArgumentError weight_profile(grid, (i, j) -> i - 1)       # 0 and negative
    badw = [fill(R(1), n_cells(grid, k)) for k in 0:2]
    badw[2][3] = R(-2)
    err = try; WeightedGridBase(grid; weights = badw); nothing; catch e; e; end
    @test err isa ArgumentError && occursin("positive", err.msg)
    # wrong per-grade length
    shortw = [fill(R(1), n_cells(grid, k)) for k in 0:2]
    pop!(shortw[1])
    @test_throws ArgumentError WeightedGridBase(grid; weights = shortw)

    # capabilities forward: the weighted grid can Hodge
    @test can_hodge(wb) && top_grade(wb) == 2
end

@testset "L10.1 gate 3: unit weights reproduce the plain grid exactly" begin
    for (name, m) in _wb_metrics()
        grid = GridBase(2, 2; metric = m)
        wb = WeightedGridBase(grid)
        for g in 0:2
            fg = _wb_field(grid, m, g; seed = 1)
            fw = Field(wb, g, Dict(Int(c) => evaluate(fg, c) for c in cells(grid, g)))

            # star values agree at every dual cell
            sg, sw = hodge_star(grid, fg), hodge_star(wb, fw)
            @test all(evaluate(sg, dual_cell(grid, g, c)) ==
                      evaluate(sw, dual_cell(wb, g, c)) for c in cells(grid, g))
            # codifferential, Laplacian, pairing agree cellwise / exactly
            if g >= 1
                @test all(evaluate(codifferential(grid, fg), c) ==
                          evaluate(codifferential(wb, fw), c) for c in cells(grid, g - 1))
            end
            @test all(evaluate(hodge_laplacian(grid, fg), c) ==
                      evaluate(hodge_laplacian(wb, fw), c) for c in cells(grid, g))
            @test inner_product(grid, fg, fg) == inner_product(wb, fw, fw)
            @test total_energy(grid, fg) == total_energy(wb, fw)
        end
    end
end

@testset "L10.1 gate 2: ⋆⋆ sign law unchanged under non-unit weights" begin
    for (name, m) in _wb_metrics()
        q = name == "Euclidean" ? 0 : 1
        wb = weight_profile(GridBase(2, 2; metric = m), (i, j) -> 1 + (i + j) // 4)
        for g in 0:2
            fld = _wb_field(wb, m, g; seed = 2)
            expected_sign = isodd(g * (2 - g) + q) ? -one(R) : one(R)
            @test hodge_star(wb, hodge_star(wb, fld)) == expected_sign * fld
        end
    end
end

@testset "L10.1 gate 1: weighted adjointness, exact, boundary-touching included" begin
    for (name, m) in _wb_metrics(), (nx, ny) in ((2, 2), (3, 2))
        wb = weight_profile(GridBase(nx, ny; metric = m), (i, j) -> 1 + (i + j) // 4)

        # δ_w IS the weighted transpose (the structural form of the contract)
        for g in 1:2
            β = _wb_field(wb, m, g; seed = 3)
            @test codifferential(wb, β) == _wb_weighted_dT(wb, β)
        end

        # ⟨dα,β⟩_w == ⟨α,δβ⟩_w through the weighted pairing (inner_product
        # picks up w_k(c) via the L9.1 _hodge_weight seam, by dispatch)
        for k in 0:1
            α = _wb_field(wb, m, k; seed = 4)
            β = _wb_field(wb, m, k + 1; seed = 5)
            @test inner_product(wb, d(α), β) ==
                  inner_product(wb, α, codifferential(wb, β))
            # boundary-only supports
            αb = Field(wb, k, Dict(1 => evaluate(α, 1)))
            βb = Field(wb, k + 1, Dict(1 => evaluate(β, 1)))
            @test inner_product(wb, d(αb), βb) ==
                  inner_product(wb, αb, codifferential(wb, βb))
            @test inner_product(wb, d(αb), β) ==
                  inner_product(wb, αb, codifferential(wb, β))
        end
    end
end

@testset "L10.1 gate 4: δ² = 0 exact with weights" begin
    for (name, m) in _wb_metrics()
        wb = weight_profile(GridBase(3, 2; metric = m), (i, j) -> 1 + (i + j) // 4)
        β = _wb_field(wb, m, 2; seed = 6)
        @test length(codifferential(wb, codifferential(wb, β))) == 0
    end
end

@testset "L10.1 gate 5: leapfrog conserves the weighted H in the shadow band" begin
    # Weighted operator eigenvalue, BY HAND.  On GridBase(1,1) (one face, four
    # edges: 1=bottom, 2=top, 3=left, 4=right; face boundary signs s =
    # (+1,−1,−1,+1)), the grade-2 operator M = dδ_w = D₁ W₁⁻¹ D₁ᵀ W₂ is 1×1:
    #
    #     M = w_face · Σ_e s_e²/w_e = w_face · Σ_e 1/w_e.
    #
    # Choose w₁ = (1, 2, 1, 2), w_face = 1:  λ = 1 + 1/2 + 1 + 1/2 = 3.
    # The K = δ_w d eigenmode is Φ = δ_w(u) for u = 1 on the face (SVD
    # pairing, as in L10):  Φ_e = s_e/w_e = (1, −1/2, −1, +1/2), and
    # ‖Φ‖²_w = Σ_e w_e Φ_e² = 1 + 1/2 + 1 + 1/2 = 3, so H₀ = ½λ‖Φ‖²_w = 9/2.
    w1 = [1, 2, 1, 2]

    # Exact preconditions + a 3-step exact recurrence check (the L10 KDK map
    # a' = (1−z/2)a + dt·e, e' = −dtλ(1−z/4)a + (1−z/2)e is weight-independent
    # once on an eigenmode).
    grid = GridBase(1, 1)
    wb = WeightedGridBase(grid; weights =
        [fill(R(1), 4), R.(w1), [R(1)]])
    m = grid.metric
    u = Field(wb, 2, Dict(1 => clifford_one(m)))
    Φ = codifferential(wb, u)
    λ = R(3)
    @test codifferential(wb, d(Φ)) == 3 * Φ
    @test field_norm2(wb, Φ) == 3
    sys = maxwell_system(wb)
    E0 = zero_field(wb, 1, fibre(wb, 1, 1))
    dt = R(1 // 10); z = dt^2 * λ
    traj = evolve(sys, SimState(sys, Dict(:A => Φ, :E => E0)), Leapfrog(), dt, 3;
                  observers = [:H => energy_observer(wb)], record_states = true)
    a, e = one(R), zero(R)
    @test traj[:H][1] == R(9 // 2)
    for k in 1:3
        a, e = (1 - z / 2) * a + dt * e, -dt * λ * (1 - z / 4) * a + (1 - z / 2) * e
        @test traj.states[k + 1][:A] == a * Φ
        @test traj.states[k + 1][:E] == e * Φ
        @test traj[:H][k + 1] == (e^2 * 3 + λ * a^2 * 3) / 2
    end

    # Float64 long run: bounded oscillation in the derived band, no drift.
    # As in L10 test 2: H̃ = ½‖Φ‖²_w(e² + λ(1−ζ)a²) is conserved (ζ = dt²λ/4),
    # so H ∈ [(1−ζ)H₀, H₀] with e₀ = 0.
    fgrid = GridBase(1, 1; metric = signature_metric(VectorSpace(2), Float64, 2, 0, 0))
    fwb = WeightedGridBase(fgrid; weights =
        [fill(1.0, 4), Float64.(w1), [1.0]])
    fm = fgrid.metric
    Φf = codifferential(fwb, Field(fwb, 2, Dict(1 => clifford_one(fm))))
    @test codifferential(fwb, d(Φf)) == 3 * Φf
    fsys = maxwell_system(fwb)
    fdt = 0.5 * (2 / sqrt(3.0)); ζ = fdt^2 * 3.0 / 4
    ftraj = evolve(fsys, SimState(fsys, Dict(:A => Φf, :E => zero_field(fwb, 1, fibre(fwb, 1, 1)))),
                   Leapfrog(), fdt, 10_000; observers = [:H => energy_observer(fwb)])
    H = Float64.(ftraj[:H])
    H0 = H[1]
    @test H0 ≈ 4.5
    tol = 1e-9 * H0
    @test all(h -> (1 - ζ) * H0 - tol <= h <= H0 + tol, H)
    @test abs(sum(H[end-999:end]) / 1000 - sum(H[1:1000]) / 1000) < 1e-3 * H0
end

@testset "L10.1 gate 6: a graded profile changes the local wave operator (E2 precondition)" begin
    # GridBase(4,1) with the graded profile w(i,j) = 1 + i.  Two unit bumps on
    # v-edges at mirrored relative positions, i = 1 (id 10) and i = 3 (id 12):
    # each v-edge bounds the two faces at i−1 and i, so the local Rayleigh
    # quotient of K = δ_w d is
    #
    #     q(i) = (w₂(i−1) + w₂(i)) / w₁(i) = ((i) + (i+1)) / (1+i)
    #
    # q(1) = 3/2 and q(3) = 7/4: the SAME local bump sees a different wave
    # operator in the two regions — the weights are a medium.  (The actual
    # refraction run across the gradient is Experiment 2's job, not this
    # test's.)
    grid = GridBase(4, 1)
    m = grid.metric
    wb = weight_profile(grid, (i, j) -> 1 + i)
    B1 = Field(wb, 1, Dict(10 => clifford_one(m)))    # v-edge i = 1
    B2 = Field(wb, 1, Dict(12 => clifford_one(m)))    # v-edge i = 3
    q1 = field_norm2(wb, d(B1)) // field_norm2(wb, B1)
    q2 = field_norm2(wb, d(B2)) // field_norm2(wb, B2)
    @test q1 == 3 // 2
    @test q2 == 7 // 4
    @test q1 != q2 && q1 > 0 && q2 > 0
    # and the operator's pointwise action differs too
    @test evaluate(codifferential(wb, d(B1)), 10) == R(3 // 2) * clifford_one(m)
    @test evaluate(codifferential(wb, d(B2)), 12) == R(7 // 4) * clifford_one(m)
end
