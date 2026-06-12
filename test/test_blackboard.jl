# ─────────────────────────────────────────────────────────────────────────────
# Phase L9: the equation blackboard — typed operator calculus
#
# Naming note: Symbolics (loaded by earlier test files when available) exports
# `simplify`, `substitute`, `expand`, and `Equation`; the blackboard names are
# therefore used QUALIFIED (`TS.…`) throughout this file so the suite runs
# identically with and without Symbolics in the session.
# ─────────────────────────────────────────────────────────────────────────────

R  = Rational{BigInt}
TS = Tensorsmith

@testset "Blackboard: inference propagates (grade, residence) per node type" begin
    grid = GridBase(2, 2)
    φ = FieldVar(:φ, grid, 0)
    A = FieldVar(:A, grid, 1)
    F = FieldVar(:F, grid, 2)

    # d : (k, primal) → (k+1, primal); above top grade = the zero-only space
    for (k, v) in ((0, φ), (1, A), (2, F))
        @test expr_grade(v) == k
        @test expr_residence(v) === :primal
        @test expr_base(v) === grid
        @test expr_grade(d(v)) == k + 1
        @test expr_residence(d(v)) === :primal
    end

    # ⋆ : (k, primal) ↔ (n-k, dual), and d acts on dual cochains too
    for (k, v) in ((0, φ), (1, A), (2, F))
        s = hodge_star(grid, v)
        @test expr_grade(s) == 2 - k
        @test expr_residence(s) === :dual
        ss = hodge_star(grid, s)
        @test expr_grade(ss) == k
        @test expr_residence(ss) === :primal
    end
    dual_dA = d(hodge_star(grid, F))     # dual grade 0 → dual grade 1
    @test expr_grade(dual_dA) == 1
    @test expr_residence(dual_dA) === :dual

    # δ : (k, primal) → (k-1, primal), grade-0 convention δφ = 0 (grade 0)
    @test expr_grade(codifferential(grid, F)) == 1
    @test expr_grade(codifferential(grid, A)) == 0
    @test expr_grade(codifferential(grid, φ)) == 0
    @test expr_residence(codifferential(grid, A)) === :primal

    # Δ : grade- and residence-preserving
    for v in (φ, A, F)
        @test expr_grade(hodge_laplacian(grid, v)) == expr_grade(v)
        @test expr_residence(hodge_laplacian(grid, v)) === :primal
    end

    # Linear combination: n-ary, all terms must agree
    lc = 2 * A + 3 * FieldVar(:B, grid, 1)
    @test lc isa LinearCombination
    @test length(lc.terms) == 2
    @test expr_grade(lc) == 1
    @test expr_residence(lc) === :primal
    @test_throws ArgumentError φ + A                          # grade mismatch
    @test_throws ArgumentError A + hodge_star(grid, A)        # residence mismatch
    @test_throws ArgumentError A + FieldVar(:C, GridBase(1, 1), 1)  # base mismatch

    # Leaf validation
    @test_throws ArgumentError FieldVar(:bad, grid, -1)
    @test_throws ArgumentError FieldVar(:bad, grid, 1; residence = :somewhere)
    @test zero_expr(grid, 3) isa ZeroExpr     # zero-only space above top grade is expressible
    @test expr_residence(zero_expr(grid, 1; residence = :dual)) === :dual
end

@testset "Blackboard: capability gates throw at node construction" begin
    grid = GridBase(2, 2)
    m = signature_metric(VectorSpace(2), R, 2, 0, 0)
    graph = GraphBase(2, [(1, 2)]; metric = m)
    x = FieldVar(:x, graph, 0)

    # d is metric-free and works on every base; ⋆/δ/Δ are gated by can_hodge
    @test d(x) isa DExpr
    @test_throws ArgumentError hodge_star(graph, x)
    @test_throws ArgumentError codifferential(graph, x)
    @test_throws ArgumentError hodge_laplacian(graph, x)

    # A dual placeholder needs a dual complex
    @test_throws ArgumentError FieldVar(:η, graph, 0; residence = :dual)

    # δ/Δ are primal-only; ⋆ refuses grades outside 0:n
    A = FieldVar(:A, grid, 1)
    @test_throws ArgumentError codifferential(grid, hodge_star(grid, A))
    @test_throws ArgumentError hodge_laplacian(grid, hodge_star(grid, A))
    @test_throws ArgumentError hodge_star(grid, d(FieldVar(:F, grid, 2)))  # grade 3 > n

    # Operators require the expression's own base
    other = GridBase(2, 2)
    @test_throws ArgumentError hodge_star(other, A)
end

@testset "Blackboard: Equation typechecking names the disagreement" begin
    grid  = GridBase(2, 2)
    grid2 = GridBase(2, 2)
    A = FieldVar(:A, grid, 1)

    err = try; TS.Equation(d(A), A); nothing; catch e; e; end
    @test err isa ArgumentError
    @test occursin("grade", err.msg)

    err = try; TS.Equation(hodge_star(grid, A), A); nothing; catch e; e; end
    @test err isa ArgumentError
    @test occursin("residence", err.msg)

    err = try; TS.Equation(A, FieldVar(:A, grid2, 1)); nothing; catch e; e; end
    @test err isa ArgumentError
    @test occursin("base", err.msg)

    # A bare number is rejected: it carries no (grade, residence, base)
    @test_throws ArgumentError TS.Equation(d(A), 0)

    eq = TS.Equation(d(A), zero_expr(grid, 2))
    @test TS.lhs(eq) == d(A)
    @test TS.rhs(eq) == zero_expr(grid, 2)
end

@testset "Blackboard: Bianchi and δδ fall out of the registry" begin
    grid = GridBase(2, 2)
    A = FieldVar(:A, grid, 1)
    @test TS.simplify(d(d(A))) == zero_expr(grid, 3)
    # d³A: the OUTER pair is also a d∘d (applied to dA), so the identity fires
    # at the root and yields the grade-4 typed zero.
    @test TS.simplify(d(d(d(A)))) == zero_expr(grid, 4)

    F = FieldVar(:F, grid, 2)
    @test TS.simplify(codifferential(grid, codifferential(grid, F))) == zero_expr(grid, 0)

    # Nested below another operator: the pass reaches inner nodes
    @test TS.simplify(hodge_star(grid, d(d(FieldVar(:φ, grid, 0))))) ==
          hodge_star(grid, zero_expr(grid, 2))
end

@testset "Blackboard: ⋆⋆ sign law matches the verified (-1)^(k(n-k)+q)" begin
    for (p, q) in ((2, 0), (1, 1))   # Euclidean and Lorentzian
        m = signature_metric(VectorSpace(2), R, p, q, 0)
        g = GridBase(1, 1; metric = m)
        @test signature(g) == (p, q, 0)
        for k in 0:2
            x = FieldVar(:x, g, k)
            s = TS.simplify(hodge_star(g, hodge_star(g, x)))
            if isodd(k * (2 - k) + q)
                @test s == (-1) * x
            else
                @test s == x
            end
        end
        # The same law through the dual side: ⋆⋆ on a dual cochain expression
        η = hodge_star(g, FieldVar(:y, g, 1))     # dual grade 1
        @test TS.simplify(hodge_star(g, hodge_star(g, η))) ==
              (isodd(1 * (2 - 1) + q) ? (-1) * η : η)
    end
end

@testset "Blackboard: linearity distributes and scalars pull out" begin
    grid = GridBase(2, 2)
    A = FieldVar(:A, grid, 1)
    B = FieldVar(:B, grid, 1)

    s = TS.simplify(d(R(2) * A + R(3) * B))
    @test s isa LinearCombination
    @test length(s.terms) == 2
    @test isequal(s.coeffs[1], R(2)) && isequal(s.coeffs[2], R(3))
    @test s.terms[1] == d(A) && s.terms[2] == d(B)

    # Scalar over an operator chain; unit-coefficient singletons collapse
    @test TS.simplify(1 * A) == A
    @test TS.simplify(codifferential(grid, 5 * A)) ==
          TS.simplify(5 * codifferential(grid, A))

    # Zero coefficients drop; the empty combination is the typed zero
    @test TS.simplify(0 * A + 0 * B) == zero_expr(grid, 1)
end

@testset "Blackboard: substitution typechecks and composes with simplify" begin
    grid = GridBase(1, 1)
    Avar = FieldVar(:A, grid, 1)
    Fvar = FieldVar(:F, grid, 2)

    eq  = TS.Equation(d(Fvar), zero_expr(grid, 3))
    eq2 = TS.substitute(eq, Fvar => d(Avar))
    @test TS.lhs(eq2) == d(d(Avar))
    seq = TS.simplify(eq2)
    @test TS.lhs(seq) == zero_expr(grid, 3)       # trivially-true equation
    @test TS.rhs(seq) == zero_expr(grid, 3)

    # Mismatched replacements throw, naming the disagreement
    err = try; TS.substitute(eq, Fvar => Avar); nothing; catch e; e; end
    @test err isa ArgumentError && occursin("grade", err.msg)
    err = try; TS.substitute(eq, Fvar => hodge_star(grid, Fvar)); nothing; catch e; e; end
    @test err isa ArgumentError
    err = try; TS.substitute(eq, Fvar => FieldVar(:G, GridBase(1, 1), 2)); nothing; catch e; e; end
    @test err isa ArgumentError && occursin("base", err.msg)

    # Substituting a concrete Field binds it as a literal leaf
    m = grid.metric
    concrete = Field(grid, 1, Dict(1 => clifford_basis_vector(m, 1)))
    ex = TS.substitute(d(Avar), Avar => concrete)
    @test evaluate(ex) == d(concrete)
end

@testset "Blackboard: definitional expansions are explicit, never automatic" begin
    grid = GridBase(1, 1)
    m = grid.metric
    A = FieldVar(:A, grid, 1)
    f = Field(grid, 1, Dict(1 => clifford_basis_vector(m, 1)))
    b = Dict{Symbol,Any}(:A => f)

    δA = codifferential(grid, A)
    @test TS.simplify(δA) == δA                       # not auto-rewritten
    ex = TS.expand(δA, :codifferential)
    @test ex != δA
    @test occursin("⋆", repr(ex))
    @test evaluate(ex; bindings = b) == codifferential(grid, f)   # engine agreement

    ΔA = hodge_laplacian(grid, A)
    @test TS.simplify(ΔA) == ΔA
    exL = TS.expand(ΔA, :laplacian)
    @test occursin("δ", repr(exL)) && occursin("d", repr(exL))
    @test evaluate(exL; bindings = b) == hodge_laplacian(grid, f)

    # Engine grade guards are mirrored at the edges of the grade range
    φ = FieldVar(:φ, grid, 0)
    f0 = Field(grid, 0, Dict(1 => clifford_one(m)))
    @test TS.expand(codifferential(grid, φ), :codifferential) == zero_expr(grid, 0)
    @test evaluate(TS.expand(hodge_laplacian(grid, φ), :laplacian);
                   bindings = Dict{Symbol,Any}(:φ => f0)) == hodge_laplacian(grid, f0)

    @test_throws ArgumentError TS.expand(δA, :nonsense)
end

@testset "Blackboard: Maxwell end-to-end on the L8.3-verified configuration" begin
    # Exactly the test_maxwell.jl known-source data (L8.2.1 re-baselined value;
    # the blade-by-blade hand re-derivation lives in test_maxwell.jl).
    grid = GridBase(1, 1)
    m = grid.metric
    e1 = clifford_basis_vector(m, 1)
    A = Field(grid, 1, Dict(1 => e1))
    expected_J = Field(grid, 1, Dict(
        1 => e1,
        2 => -e1,
        3 => -e1,
        4 => e1,
    ))

    Avar = FieldVar(:A, grid, 1)
    Jvar = FieldVar(:J, grid, 1)
    maxwell = TS.Equation(codifferential(grid, d(Avar)), Jvar)
    b = Dict{Symbol,Any}(:A => A, :J => expected_J)

    # One equation object: numeric check against the verified engine operators
    @test check(maxwell; bindings = b)
    @test length(residual(maxwell; bindings = b)) == 0     # exact zero residual
    @test evaluate(TS.lhs(maxwell); bindings = b) == maxwell_current(grid, A)

    # Bianchi numerically AND symbolically on the same equation type
    bianchi = TS.Equation(d(d(Avar)), zero_expr(grid, 3))
    @test check(bianchi; bindings = Dict{Symbol,Any}(:A => A))
    @test TS.lhs(TS.simplify(bianchi)) == zero_expr(grid, 3)

    # A wrong source is detected
    wrong = Dict{Symbol,Any}(:A => A, :J => Field(grid, 1, Dict(1 => e1)))
    @test !check(maxwell; bindings = wrong)

    # Unbound and mistyped bindings throw informative errors
    @test_throws ArgumentError evaluate(TS.lhs(maxwell))                     # unbound
    err = try
        check(maxwell; bindings = Dict{Symbol,Any}(:A => A,
              :J => Field(grid, 0, Dict(1 => clifford_one(m)))))
        nothing
    catch e; e; end
    @test err isa ArgumentError && occursin("grade", err.msg)
end

@testset "Blackboard: registry is data; strategy is genuinely pluggable" begin
    grid = GridBase(1, 1)
    X = FieldVar(:X, grid, 1)
    A = FieldVar(:A, grid, 1)

    rule = RewriteRule(:rename_X_to_Y,
        "test-only: rename the FieldVar :X to :Y (no identity claimed)",
        e -> (e isa FieldVar && e.name === :X) ?
             FieldVar(:Y, e.base, e.grade; residence = e.residence) : nothing)

    n_before = length(DEFAULT_RULES)
    register_rule!(rule)
    @test length(DEFAULT_RULES) == n_before + 1
    @test TS.simplify(X) == FieldVar(:Y, grid, 1)
    @test_throws ArgumentError register_rule!(rule)        # duplicate name
    unregister_rule!(:rename_X_to_Y)
    @test length(DEFAULT_RULES) == n_before
    @test TS.simplify(X) == X                              # prior behavior restored
    @test_throws ArgumentError unregister_rule!(:rename_X_to_Y)

    # A private registry never touches the global one
    @test TS.simplify(X; rules = RewriteRule[rule]) == FieldVar(:Y, grid, 1)
    @test TS.simplify(X) == X

    # Swapping the strategy for a no-op proves it is a real argument
    noop(e, rules) = e
    @test TS.simplify(d(d(A)); strategy = noop) == d(d(A))
end

@testset "Blackboard: show and latex output (smoke)" begin
    grid = GridBase(1, 1)
    A = FieldVar(:A, grid, 1)
    eq = TS.Equation(codifferential(grid, d(A)), FieldVar(:J, grid, 1))

    s = repr(eq)
    @test occursin("δ", s) && occursin("d", s) && occursin("=", s) && occursin("A", s)
    @test occursin("⋆", repr(hodge_star(grid, A)))
    @test occursin("Δ", repr(hodge_laplacian(grid, A)))
    @test repr(zero_expr(grid, 1)) == "0"
    @test occursin("+", repr(2 * A + 3 * FieldVar(:B, grid, 1)))
    @test occursin("⋅", repr(2 * A))

    l = latex(eq)
    @test !isempty(l)
    @test occursin("\\delta", l) && occursin("=", l)
    @test occursin("\\star", latex(hodge_star(grid, A)))
    @test occursin("\\Delta", latex(hodge_laplacian(grid, A)))
    @test occursin("\\cdot", latex(2 * A))
    @test latex(zero_expr(grid, 1)) == "0"
end

@testset "Blackboard: symbolic coefficients coexist (Symbolics.Num as R)" begin
    blackboard_symbolics_available = try
        @eval using Symbolics
        symbolic_metric(signature_metric(VectorSpace(2), R, 2, 0, 0))
        true
    catch err
        @info "Symbolics.jl unavailable; blackboard symbolic-coefficient tests skipped." exception=(err, catch_backtrace())
        false
    end

    if blackboard_symbolics_available
        m = symbolic_metric(signature_metric(VectorSpace(2), R, 2, 0, 0))
        grid = GridBase(1, 1; metric = m)
        a = Symbolics.variable(:a)
        b = Symbolics.variable(:b)

        φ = FieldVar(:φ, grid, 0)
        ψ = FieldVar(:ψ, grid, 0)

        # Symbolic coefficients on symbolic fields typecheck …
        lc = a * φ + b * ψ
        @test lc isa LinearCombination
        @test expr_grade(lc) == 0
        @test expr_residence(lc) === :primal

        # … and simplify by linearity, keeping the Num coefficients
        s = TS.simplify(d(lc))
        @test s isa LinearCombination
        @test isequal(s.coeffs[1], a) && isequal(s.coeffs[2], b)
        @test s.terms[1] == d(φ) && s.terms[2] == d(ψ)

        # Round-trip: evaluating the simplified tree against the engine's own
        # arithmetic agrees cellwise under isequal_simplified
        fφ = Field(grid, 0, Dict(1 => clifford_one(m)))
        fψ = Field(grid, 0, Dict(2 => clifford_one(m)))
        bind = Dict{Symbol,Any}(:φ => fφ, :ψ => fψ)
        lhs_val = evaluate(s; bindings = bind)
        rhs_val = a * d(fφ) + b * d(fψ)
        @test field_grade(lhs_val) == 1
        for c in cells(grid, 1)
            @test isequal_simplified(evaluate(lhs_val, c), evaluate(rhs_val, c))
        end

        # show does not throw on symbolic coefficients
        @test_nowarn repr(lc)
    end
end
