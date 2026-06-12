# ── Phase L9: The Equation Blackboard — a typed operator calculus ────────────
#
# MATHEMATICAL / ARCHITECTURAL CONTRACT (DESIGN.md §17, settled Rev 2.1)
#
# The blackboard is the field/operator-level symbolic layer: placeholders for
# fields (`FieldVar`), unevaluated operator applications (`d`, `⋆`, `δ`, `Δ`
# applied to expressions return expression NODES instead of computing),
# equations whose two sides are typechecked in (grade, residence, base), a tiny
# rewrite registry whose every rule is an externally verified identity, and an
# evaluator that binds placeholders to concrete `Field`s/`HodgeDualField`s and
# calls the real, verified L8 operators.
#
# Input is Julia itself — multiple dispatch is the parser.  LaTeX is OUTPUT
# ONLY (`latex`); there is no string or LaTeX parsing anywhere.
#
# TWO SYMBOLIC LAYERS, NEVER CONFLATED (§17): coefficient-level symbolics is
# `Symbolics.Num` as the scalar ring `R` (it already exists and stays at the
# ring layer); field/operator-level symbolics is THIS small expression layer.
# Field calculus is never encoded into Symbolics.jl; the two coexist in one
# expression (symbolic coefficients on symbolic fields).
#
# OPEN, N-ARY NODE HIERARCHY (§17.1 — architectural requirement): every node is
# a concrete subtype of `BlackboardExpr`, dispatch-extensible, NEVER a closed
# enum/union of operators, and arity is not assumed unary (the linear
# combination is n-ary).  A future binary product node (the DEC cup-product
# phase) and `∇` arrive purely additively: a new type + its inference methods
# + rules, with zero changes to existing nodes.
#
# EXPLICITLY OUT OF SCOPE (§17/§17.1 — do not add here): field products /
# wedge-of-fields / cup product (dedicated later phase), `∇` on the blackboard
# (additive later), equation solving, time evolution (L10), UI/wizard/LaTeX
# input (L11), e-graphs/equality saturation, and any CAS-style search
# rewriting (rejected — `Symbolics.Num` is already the scalar CAS at the
# correct layer; the blackboard stays a small typed operator calculus whose
# every rewrite is a verified identity).

# ── The abstract node type ────────────────────────────────────────────────────

"""
    BlackboardExpr

Abstract supertype of every equation-blackboard expression node (DESIGN.md
§17).  This hierarchy is **open and n-ary by contract** (§17.1):

  - Every node is a concrete subtype created by ordinary Julia dispatch.  The
    operator set is **never** a closed enum or a fixed `Union`; third parties
    (and later phases: the DEC cup product, `∇`) add node types purely
    additively, with zero changes to the existing nodes.
  - Node arity is not assumed unary: [`LinearCombination`](@ref) is n-ary, and
    the traversal machinery (`simplify`, `substitute`, `expand`, `evaluate`)
    works through the generic node interface below, so a future binary product
    node slots in without touching it.

# The node-author interface (the §8 extension surface)
A new node type `T <: BlackboardExpr` must implement:

  - **(grade, residence, base) inference** — methods [`expr_grade`](@ref),
    [`expr_residence`](@ref), [`expr_base`](@ref) propagating the verified
    operator signature of the node's operation.  Construction-time gates
    (capability checks such as `can_hodge`, grade-range checks) live in the
    node's constructor, so an ill-typed node can never exist.
  - **traversal** — [`expr_children`](@ref) and [`expr_rebuild`](@ref).
  - **structural equality** — a `Base.:(==)` method (must return `Bool` over
    every ring; compare coefficients with `isequal`, never `==`).
  - **output** — `Base.show` (Unicode) and [`latex`](@ref) (LaTeX string,
    output only).
  - **evaluation** — a `_evaluate(node, bindings)` method calling the real
    engine operators.

Rewrite rules for the new node are registered as data via
[`register_rule!`](@ref); nothing in the default registry needs to change.
"""
abstract type BlackboardExpr end

# ── Leaves ────────────────────────────────────────────────────────────────────

"""
    FieldVar(name::Symbol, base::BaseSpace, grade::Integer; residence = :primal)

A symbolic field placeholder: a named stand-in for a concrete grade-`grade`
section over `base`.  `residence` is `:primal` (a [`Field`](@ref)) or `:dual`
(a [`HodgeDualField`](@ref) cochain; requires `has_dual_complex(base)`, since a
dual cochain honestly does not exist on a bare graph).

A `FieldVar` is bound to a concrete section at evaluation time via the
`bindings` of [`evaluate`](@ref); the binding must match the variable's
(grade, residence, base) exactly or an informative `ArgumentError` is thrown.
"""
struct FieldVar{B<:BaseSpace} <: BlackboardExpr
    name      :: Symbol
    base      :: B
    grade     :: Int
    residence :: Symbol

    function FieldVar(name::Symbol, base::B, grade::Integer;
                      residence::Symbol = :primal) where {B<:BaseSpace}
        grade >= 0 || throw(ArgumentError(
            "FieldVar grade must be nonnegative, got $grade"))
        residence === :primal || residence === :dual || throw(ArgumentError(
            "FieldVar residence must be :primal or :dual, got :$residence"))
        if residence === :dual && !has_dual_complex(base)
            throw(ArgumentError(
                "a :dual FieldVar requires has_dual_complex(base) == true; " *
                "$(typeof(base)) has no dual complex, so dual cochains do not exist over it"))
        end
        new{B}(name, base, Int(grade), residence)
    end
end

"""
    FieldLit(f::Field)
    FieldLit(η::HodgeDualField)

A concrete section used directly as an expression leaf (a bound literal).  Its
(grade, residence, base) are read off the wrapped section.  [`evaluate`](@ref)
returns the wrapped section unchanged.

Structural equality of `FieldLit`s delegates to the section's `==`, which is
exact-ring only (over `Symbolics.Num` compare evaluations with
`isequal_simplified` instead — the same convention as `Field` itself).
"""
struct FieldLit{F} <: BlackboardExpr
    field :: F

    FieldLit(f::Field)          = new{typeof(f)}(f)
    FieldLit(η::HodgeDualField) = new{typeof(η)}(η)
end

"""
    ZeroExpr(base::BaseSpace, grade::Integer; residence = :primal)

The typed zero leaf: the everywhere-zero grade-`grade` section over `base` at
the given residence, as an expression.  Exists so that equations like
`Equation(d(F), zero_expr(b, k+1))` are expressible — a bare `0` carries no
(grade, residence, base) and is deliberately rejected.  Prefer the
[`zero_expr`](@ref) convenience constructor.
"""
struct ZeroExpr{B<:BaseSpace} <: BlackboardExpr
    base      :: B
    grade     :: Int
    residence :: Symbol

    function ZeroExpr(base::B, grade::Integer;
                      residence::Symbol = :primal) where {B<:BaseSpace}
        grade >= 0 || throw(ArgumentError(
            "ZeroExpr grade must be nonnegative, got $grade"))
        residence === :primal || residence === :dual || throw(ArgumentError(
            "ZeroExpr residence must be :primal or :dual, got :$residence"))
        if residence === :dual && !has_dual_complex(base)
            throw(ArgumentError(
                "a :dual ZeroExpr requires has_dual_complex(base) == true; " *
                "$(typeof(base)) has no dual complex"))
        end
        new{B}(base, Int(grade), residence)
    end
end

"""
    zero_expr(base::BaseSpace, grade::Integer; residence = :primal) -> ZeroExpr

The typed zero expression leaf (see [`ZeroExpr`](@ref)).  Evaluates to the
empty (everywhere-zero) sparse section of the stated grade and residence.
"""
zero_expr(base::BaseSpace, grade::Integer; residence::Symbol = :primal) =
    ZeroExpr(base, grade; residence = residence)

# ── Operator nodes ────────────────────────────────────────────────────────────
#
# Each node stores its argument(s) only; (grade, residence, base) are INFERRED
# by the per-type methods below, propagating the verified operator signatures.
# Capability gates (can_hodge) and grade-range checks run at CONSTRUCTION, so
# an expression that could not evaluate on its base can never be built.

"""
    DExpr(arg::BlackboardExpr)

Unevaluated exterior derivative node, built by `d(::BlackboardExpr)`.
Signature (verified, L8/L8.3): `(k, :primal) → (k+1, :primal)` and
`(l, :dual) → (l+1, :dual)` (the dual coboundary derived from the primal
incidence transpose).  Metric-free: no capability gate, valid on every base;
above the top grade it denotes the zero-only cochain space, exactly like the
engine's `d`.
"""
struct DExpr <: BlackboardExpr
    arg :: BlackboardExpr
end

"""
    StarExpr(arg::BlackboardExpr)

Unevaluated Hodge star node, built by `hodge_star(b, ::BlackboardExpr)` / `⋆`.
Signature (verified, L8.2/L8.3): `(k, :primal) ↔ (n-k, :dual)` with
`n = top_grade(base)`.  Construction requires `can_hodge(base)` (the
documented L8.2 gate — a bare graph throws) and `0 ≤ k ≤ n`.
"""
struct StarExpr <: BlackboardExpr
    arg :: BlackboardExpr

    function StarExpr(arg::BlackboardExpr)
        b = expr_base(arg)
        _require_hodge(b, "hodge_star")
        n = top_grade(b)
        k = expr_grade(arg)
        (0 <= k <= n) || throw(ArgumentError(
            "hodge_star expects an expression grade k in 0:top_grade(b) = 0:$n, got $k " *
            "(a d-result above the top grade has no Hodge dual)"))
        new(arg)
    end
end

"""
    CodiffExpr(arg::BlackboardExpr)

Unevaluated codifferential node, built by `codifferential(b, ::BlackboardExpr)`
/ `δ`.  Signature (verified, L8.2): `(k, :primal) → (k-1, :primal)` for
`k ≥ 1`; on grade 0 it denotes the zero 0-section (engine convention).
Construction requires `can_hodge(base)`, residence `:primal` (the engine
defines `δ` on primal fields only), and `k ≤ top_grade(base)`.
"""
struct CodiffExpr <: BlackboardExpr
    arg :: BlackboardExpr

    function CodiffExpr(arg::BlackboardExpr)
        b = expr_base(arg)
        _require_hodge(b, "codifferential")
        expr_residence(arg) === :primal || throw(ArgumentError(
            "the codifferential δ is defined on primal expressions only; " *
            "apply ⋆ first to return a dual expression to the primal complex"))
        n = top_grade(b)
        k = expr_grade(arg)
        k <= n || throw(ArgumentError(
            "codifferential expects an expression grade k ≤ top_grade(b) = $n, got $k " *
            "(the engine's δ is undefined above the top grade)"))
        new(arg)
    end
end

"""
    LaplacianExpr(arg::BlackboardExpr)

Unevaluated Hodge–de Rham Laplacian node, built by
`hodge_laplacian(b, ::BlackboardExpr)` / `Δ`.  Signature (verified, L8.2):
grade- and residence-preserving on primal expressions.  Construction requires
`can_hodge(base)`, residence `:primal`, and `0 ≤ k ≤ top_grade(base)`.
"""
struct LaplacianExpr <: BlackboardExpr
    arg :: BlackboardExpr

    function LaplacianExpr(arg::BlackboardExpr)
        b = expr_base(arg)
        _require_hodge(b, "hodge_laplacian")
        expr_residence(arg) === :primal || throw(ArgumentError(
            "the Hodge Laplacian Δ is defined on primal expressions only"))
        n = top_grade(b)
        k = expr_grade(arg)
        (0 <= k <= n) || throw(ArgumentError(
            "hodge_laplacian expects an expression grade k in 0:top_grade(b) = 0:$n, got $k"))
        new(arg)
    end
end

"""
    LinearCombination(coeffs::Vector{C}, terms::Vector{BlackboardExpr})

The **n-ary** linear-combination node `c₁⋅t₁ + ⋯ + cₘ⋅tₘ` (§17.1: the node
hierarchy is open and n-ary; this node is the live witness).  Built by `+`,
`-`, and scalar `*` on expressions; scalar coefficients live in the field's
scalar ring `R` (including `Symbolics.Num` — `Num <: Number`, so the existing
extension makes symbolic coefficients work with no blackboard-specific code)
or are plain `Integer`s.

All terms must agree in (grade, residence, base); a disagreement throws an
`ArgumentError` naming the offending term.  At least one term is required —
the empty combination is the typed zero, [`zero_expr`](@ref).
"""
struct LinearCombination{C} <: BlackboardExpr
    coeffs :: Vector{C}
    terms  :: Vector{BlackboardExpr}

    function LinearCombination{C}(coeffs::Vector{C},
                                  terms::Vector{BlackboardExpr}) where {C}
        length(coeffs) == length(terms) || throw(ArgumentError(
            "LinearCombination needs one coefficient per term; got " *
            "$(length(coeffs)) coefficients for $(length(terms)) terms"))
        isempty(terms) && throw(ArgumentError(
            "LinearCombination needs at least one term; " *
            "use zero_expr(base, grade) for the typed zero"))
        b = expr_base(terms[1])
        g = expr_grade(terms[1])
        r = expr_residence(terms[1])
        for (i, t) in enumerate(terms)
            expr_base(t) === b || throw(ArgumentError(
                "linear-combination terms disagree in base: term 1 is over one " *
                "$(typeof(b)) instance but term $i is over a different base instance"))
            expr_grade(t) == g || throw(ArgumentError(
                "linear-combination terms disagree in grade: term 1 has grade $g " *
                "but term $i has grade $(expr_grade(t))"))
            expr_residence(t) === r || throw(ArgumentError(
                "linear-combination terms disagree in residence: term 1 is $r " *
                "but term $i is $(expr_residence(t))"))
        end
        new{C}(coeffs, terms)
    end
end

LinearCombination(coeffs::Vector{C}, terms::Vector{<:BlackboardExpr}) where {C} =
    LinearCombination{C}(coeffs, Vector{BlackboardExpr}(terms))

# Tighten a possibly-heterogeneous coefficient vector to a common promoted type
# (Int + Symbolics.Num → Num, Int + Rational{BigInt} → Rational{BigInt}, …).
_promote_coeffs(cs::Vector) = isempty(cs) ? cs : collect(promote(cs...))

_lincomb(coeffs::Vector, terms::Vector{<:BlackboardExpr}) =
    LinearCombination(_promote_coeffs(coeffs), Vector{BlackboardExpr}(terms))

# ── (grade, residence, base) inference — one method set per node type ─────────

"""
    expr_base(e::BlackboardExpr) -> BaseSpace

The base space an expression lives over.  Part of the per-node-type inference
contract (see [`BlackboardExpr`](@ref)); every node type implements it.
"""
function expr_base end

"""
    expr_grade(e::BlackboardExpr) -> Int

The inferred grade of an expression, propagated through the verified operator
signatures: `d : k → k+1`; `⋆ : k → n-k`; `δ : k → k-1` (0 at grade 0);
`Δ : k → k`; linear combinations require all terms to agree.
"""
function expr_grade end

"""
    expr_residence(e::BlackboardExpr) -> Symbol

`:primal` or `:dual` — which cochain complex the expression's value inhabits.
`⋆` flips residence (primal k ↔ dual n-k); `d` preserves it; `δ` and `Δ` are
primal-only.
"""
function expr_residence end

expr_base(v::FieldVar)      = v.base
expr_grade(v::FieldVar)     = v.grade
expr_residence(v::FieldVar) = v.residence

expr_base(z::ZeroExpr)      = z.base
expr_grade(z::ZeroExpr)     = z.grade
expr_residence(z::ZeroExpr) = z.residence

expr_base(l::FieldLit)      = l.field.base
expr_grade(l::FieldLit)     = field_grade(l.field)
expr_residence(l::FieldLit) = l.field isa Field ? :primal : :dual

expr_base(e::DExpr)      = expr_base(e.arg)
expr_grade(e::DExpr)     = expr_grade(e.arg) + 1
expr_residence(e::DExpr) = expr_residence(e.arg)

expr_base(e::StarExpr)      = expr_base(e.arg)
expr_grade(e::StarExpr)     = top_grade(expr_base(e.arg)) - expr_grade(e.arg)
expr_residence(e::StarExpr) = expr_residence(e.arg) === :primal ? :dual : :primal

expr_base(e::CodiffExpr)      = expr_base(e.arg)
expr_grade(e::CodiffExpr)     = max(expr_grade(e.arg) - 1, 0)
expr_residence(e::CodiffExpr) = :primal

expr_base(e::LaplacianExpr)      = expr_base(e.arg)
expr_grade(e::LaplacianExpr)     = expr_grade(e.arg)
expr_residence(e::LaplacianExpr) = :primal

expr_base(e::LinearCombination)      = expr_base(e.terms[1])
expr_grade(e::LinearCombination)     = expr_grade(e.terms[1])
expr_residence(e::LinearCombination) = expr_residence(e.terms[1])

# ── Traversal interface ───────────────────────────────────────────────────────

"""
    expr_children(e::BlackboardExpr) -> Vector{BlackboardExpr}

The child expressions of a node (empty for leaves).  With
[`expr_rebuild`](@ref) this is the generic traversal interface `simplify`,
`substitute`, and `expand` are written against, so they extend to future node
types (a binary product, `∇`) with no changes.
"""
expr_children(::BlackboardExpr) = BlackboardExpr[]
expr_children(e::DExpr)             = BlackboardExpr[e.arg]
expr_children(e::StarExpr)          = BlackboardExpr[e.arg]
expr_children(e::CodiffExpr)        = BlackboardExpr[e.arg]
expr_children(e::LaplacianExpr)     = BlackboardExpr[e.arg]
expr_children(e::LinearCombination) = copy(e.terms)

"""
    expr_rebuild(e::BlackboardExpr, children::Vector{BlackboardExpr}) -> BlackboardExpr

Reconstruct a node of the same kind around new children (coefficients and any
other non-child payload are kept).  Reconstruction re-runs the node's
constructor, so the capability gates and typechecks are re-validated.  Leaves
require `children` to be empty and return themselves.
"""
function expr_rebuild(e::BlackboardExpr, children::Vector{BlackboardExpr})
    isempty(children) || throw(ArgumentError(
        "$(typeof(e)) is a leaf node; expr_rebuild expects no children, got $(length(children))"))
    e
end

function _expect_one_child(e, children)
    length(children) == 1 || throw(ArgumentError(
        "$(typeof(e)) is a unary node; expr_rebuild expects exactly 1 child, got $(length(children))"))
    children[1]
end

expr_rebuild(e::DExpr, children::Vector{BlackboardExpr})         = DExpr(_expect_one_child(e, children))
expr_rebuild(e::StarExpr, children::Vector{BlackboardExpr})      = StarExpr(_expect_one_child(e, children))
expr_rebuild(e::CodiffExpr, children::Vector{BlackboardExpr})    = CodiffExpr(_expect_one_child(e, children))
expr_rebuild(e::LaplacianExpr, children::Vector{BlackboardExpr}) = LaplacianExpr(_expect_one_child(e, children))

function expr_rebuild(e::LinearCombination, children::Vector{BlackboardExpr})
    length(children) == length(e.terms) || throw(ArgumentError(
        "LinearCombination expr_rebuild expects $(length(e.terms)) children, got $(length(children))"))
    LinearCombination(e.coeffs, children)
end

# ── Structural equality ───────────────────────────────────────────────────────
#
# Structural (syntactic) equality, always returning Bool over every ring:
# coefficients are compared with `isequal` (never `==`, which over
# Symbolics.Num is not a Bool).  Two semantically equal but syntactically
# different expressions (e.g. `x + y` vs `y + x` at the coefficient level)
# compare unequal — semantic equality of evaluations is decided by the
# engine's own equality (exact rings) or `isequal_simplified` (symbolic).

Base.:(==)(::BlackboardExpr, ::BlackboardExpr) = false
Base.:(==)(a::FieldVar, b::FieldVar) =
    a.name === b.name && a.base === b.base &&
    a.grade == b.grade && a.residence === b.residence
Base.:(==)(a::ZeroExpr, b::ZeroExpr) =
    a.base === b.base && a.grade == b.grade && a.residence === b.residence
Base.:(==)(a::FieldLit, b::FieldLit)           = a.field == b.field
Base.:(==)(a::DExpr, b::DExpr)                 = a.arg == b.arg
Base.:(==)(a::StarExpr, b::StarExpr)           = a.arg == b.arg
Base.:(==)(a::CodiffExpr, b::CodiffExpr)       = a.arg == b.arg
Base.:(==)(a::LaplacianExpr, b::LaplacianExpr) = a.arg == b.arg

function Base.:(==)(a::LinearCombination, b::LinearCombination)
    length(a.terms) == length(b.terms) || return false
    for i in eachindex(a.terms)
        isequal(a.coeffs[i], b.coeffs[i]) || return false
        a.terms[i] == b.terms[i] || return false
    end
    true
end

# ── Building expressions: new methods on the existing generic operators ───────
#
# These methods live HERE (the new file) and dispatch on the NEW blackboard
# types; the locked operator files are untouched.  Applied to expressions, the
# verified operator names return unevaluated nodes — dispatch is the parser.

"""
    d(x::BlackboardExpr) -> DExpr

Blackboard method of the exterior derivative: returns the unevaluated node
instead of computing (DESIGN.md §17 — the embedded-DSL input methodology).
"""
d(x::BlackboardExpr) = DExpr(x)

"""
    hodge_star(b::BaseSpace, x::BlackboardExpr) -> StarExpr
    ⋆(b, x)

Blackboard method of the Hodge star: returns the unevaluated node.  Requires
`b` to be the expression's own base and `can_hodge(b) == true` (the documented
gate throws at node construction — a `GraphBase` correctly refuses).
"""
function hodge_star(b::BaseSpace, x::BlackboardExpr)
    b === expr_base(x) || throw(ArgumentError(
        "hodge_star requires the supplied base to be the expression's own base"))
    StarExpr(x)
end

"""
    codifferential(b::BaseSpace, x::BlackboardExpr) -> CodiffExpr
    δ(b, x)

Blackboard method of the codifferential: returns the unevaluated node.
Requires `b` to be the expression's own base, `can_hodge(b)`, and a primal
expression.
"""
function codifferential(b::BaseSpace, x::BlackboardExpr)
    b === expr_base(x) || throw(ArgumentError(
        "codifferential requires the supplied base to be the expression's own base"))
    CodiffExpr(x)
end

"""
    hodge_laplacian(b::BaseSpace, x::BlackboardExpr) -> LaplacianExpr
    Δ(b, x)

Blackboard method of the Hodge–de Rham Laplacian: returns the unevaluated
node.  Requires `b` to be the expression's own base, `can_hodge(b)`, and a
primal expression.
"""
function hodge_laplacian(b::BaseSpace, x::BlackboardExpr)
    b === expr_base(x) || throw(ArgumentError(
        "hodge_laplacian requires the supplied base to be the expression's own base"))
    LaplacianExpr(x)
end

function Base.:+(a::BlackboardExpr, b::BlackboardExpr)
    ca, ta = a isa LinearCombination ? (a.coeffs, a.terms) : ([1], BlackboardExpr[a])
    cb, tb = b isa LinearCombination ? (b.coeffs, b.terms) : ([1], BlackboardExpr[b])
    _lincomb(vcat(ca, cb), vcat(ta, tb))
end

Base.:+(a::BlackboardExpr, f::Union{Field,HodgeDualField}) = a + FieldLit(f)
Base.:+(f::Union{Field,HodgeDualField}, a::BlackboardExpr) = FieldLit(f) + a

Base.:-(a::BlackboardExpr)                    = (-1) * a
Base.:-(a::BlackboardExpr, b::BlackboardExpr) = a + ((-1) * b)
Base.:-(a::BlackboardExpr, f::Union{Field,HodgeDualField}) = a - FieldLit(f)
Base.:-(f::Union{Field,HodgeDualField}, a::BlackboardExpr) = FieldLit(f) - a

"""
    *(c::Number, x::BlackboardExpr) -> LinearCombination

Scalar action on an expression, building (or folding into) the n-ary linear
node.  Coefficients live in the field's scalar ring `R` or are `Integer`s;
`Symbolics.Num <: Number`, so symbolic coefficients on symbolic fields work
through the existing extension with no blackboard-specific code (§17: the two
symbolic layers coexist).
"""
function Base.:*(c::Number, x::BlackboardExpr)
    x isa LinearCombination ?
        _lincomb([c * ci for ci in x.coeffs], copy(x.terms)) :
        _lincomb([c], BlackboardExpr[x])
end

Base.:*(x::BlackboardExpr, c::Number) = c * x

# ── Equation ──────────────────────────────────────────────────────────────────

"""
    Equation(lhs, rhs)

An equation between two blackboard expressions.  Construction **typechecks**:
both sides must agree in base (identical instance), grade, and residence;
each mismatch throws an `ArgumentError` naming the disagreement (§17:
"typechecking is the value-add").  Concrete `Field`s/`HodgeDualField`s are
accepted on either side and wrapped as [`FieldLit`](@ref) leaves; a bare
number is rejected — use [`zero_expr`](@ref) for a typed zero.

Accessors: [`lhs`](@ref), [`rhs`](@ref).  `show` pretty-prints with Unicode
operators; [`latex`](@ref) emits a LaTeX string (output only).

An `==`-style constructor (`a == b` returning an `Equation`) was considered
and rejected as unclean: `==` on expressions is structural equality returning
`Bool`, and overloading it to return an `Equation` would break every boolean
context.
"""
struct Equation
    lhs :: BlackboardExpr
    rhs :: BlackboardExpr

    function Equation(l::BlackboardExpr, r::BlackboardExpr)
        expr_base(l) === expr_base(r) || throw(ArgumentError(
            "equation sides disagree in base: lhs and rhs live over different " *
            "base instances ($(typeof(expr_base(l))) vs $(typeof(expr_base(r)))); " *
            "both sides must share one base"))
        expr_grade(l) == expr_grade(r) || throw(ArgumentError(
            "equation sides disagree in grade: lhs has grade $(expr_grade(l)) " *
            "but rhs has grade $(expr_grade(r))"))
        expr_residence(l) === expr_residence(r) || throw(ArgumentError(
            "equation sides disagree in residence: lhs is $(expr_residence(l)) " *
            "but rhs is $(expr_residence(r))"))
        new(l, r)
    end
end

_as_expr(x::BlackboardExpr)   = x
_as_expr(f::Field)            = FieldLit(f)
_as_expr(η::HodgeDualField)   = FieldLit(η)
_as_expr(x) = throw(ArgumentError(
    "expected a BlackboardExpr, Field, or HodgeDualField; got $(typeof(x)). " *
    "A bare number carries no (grade, residence, base) — use zero_expr(base, grade) " *
    "for a typed zero"))

Equation(l, r) = Equation(_as_expr(l), _as_expr(r))

"""
    lhs(eq::Equation) -> BlackboardExpr

The left-hand side of an equation.
"""
lhs(eq::Equation) = eq.lhs

"""
    rhs(eq::Equation) -> BlackboardExpr

The right-hand side of an equation.
"""
rhs(eq::Equation) = eq.rhs

Base.:(==)(a::Equation, b::Equation) = a.lhs == b.lhs && a.rhs == b.rhs

# ── Rewrite registry — rules as DATA, strategy swappable (§17.1) ──────────────

"""
    RewriteRule(name::Symbol, identity::String, apply::Function)

A rewrite rule as a first-class datum: a `name`, a docstring `identity` citing
the externally verified identity the rule encodes, and an `apply` function
`expr -> Union{Nothing, BlackboardExpr}` returning `nothing` when the rule
does not apply at the given node and the replacement expression when it does
(returning a non-`nothing` result counts as rewrite progress, so `apply` must
not return an expression structurally identical to its input).

Rules live in ordered registries (`Vector{RewriteRule}`); the shipped registry
is [`DEFAULT_RULES`](@ref) and is a documented §8 extension surface via
[`register_rule!`](@ref) / [`unregister_rule!`](@ref).  The rewriting
*strategy* is likewise swappable — see [`simplify`](@ref).  No e-graphs, no
search, no CAS (§17.1: rejected; if equality saturation is ever wanted it
integrates as a strategy swap, never as a blackboard rewrite).
"""
struct RewriteRule
    name     :: Symbol
    identity :: String
    apply    :: Function
end

Base.show(io::IO, r::RewriteRule) = print(io, "RewriteRule(:", r.name, ")")

# Rule bodies.  Each encodes exactly one externally verified identity.

function _rule_dd(e::BlackboardExpr)
    (e isa DExpr && e.arg isa DExpr) || return nothing
    ZeroExpr(expr_base(e), expr_grade(e); residence = expr_residence(e))
end

function _rule_codiff_codiff(e::BlackboardExpr)
    (e isa CodiffExpr && e.arg isa CodiffExpr) || return nothing
    ZeroExpr(expr_base(e), expr_grade(e); residence = :primal)
end

function _rule_star_star(e::BlackboardExpr)
    (e isa StarExpr && e.arg isa StarExpr) || return nothing
    x = e.arg.arg
    b = expr_base(e)
    n = top_grade(b)
    q = signature(b)[2]
    g = expr_grade(x)
    isodd(g * (n - g) + q) ? _lincomb([-1], BlackboardExpr[x]) : x
end

function _rule_linearity(e::BlackboardExpr)
    if e isa DExpr && e.arg isa LinearCombination
        lc = e.arg
        return _lincomb(copy(lc.coeffs), BlackboardExpr[DExpr(t) for t in lc.terms])
    elseif e isa StarExpr && e.arg isa LinearCombination
        lc = e.arg
        return _lincomb(copy(lc.coeffs), BlackboardExpr[StarExpr(t) for t in lc.terms])
    elseif e isa CodiffExpr && e.arg isa LinearCombination
        lc = e.arg
        return _lincomb(copy(lc.coeffs), BlackboardExpr[CodiffExpr(t) for t in lc.terms])
    elseif e isa LaplacianExpr && e.arg isa LinearCombination
        lc = e.arg
        return _lincomb(copy(lc.coeffs), BlackboardExpr[LaplacianExpr(t) for t in lc.terms])
    elseif e isa LinearCombination
        changed = false
        newc = Any[]
        newt = BlackboardExpr[]
        for (c, t) in zip(e.coeffs, e.terms)
            if iszero(c)
                changed = true                      # 0 ⋅ t  → dropped
            elseif t isa ZeroExpr
                changed = true                      # c ⋅ 0  → dropped
            elseif t isa LinearCombination
                changed = true                      # flatten nested linear nodes
                for (ci, ti) in zip(t.coeffs, t.terms)
                    push!(newc, c * ci)
                    push!(newt, ti)
                end
            else
                push!(newc, c)
                push!(newt, t)
            end
        end
        if !changed
            # Canonical collapse: a singleton with unit coefficient is its term.
            if length(e.terms) == 1 && isequal(e.coeffs[1], one(e.coeffs[1]))
                return e.terms[1]
            end
            return nothing
        end
        isempty(newt) && return ZeroExpr(expr_base(e), expr_grade(e);
                                         residence = expr_residence(e))
        if length(newt) == 1 && isequal(newc[1], one(newc[1]))
            return newt[1]
        end
        return _lincomb(newc, newt)
    end
    nothing
end

"""
    DEFAULT_RULES :: Vector{RewriteRule}

The shipped, ordered rewrite registry — exactly the verified-identity rules of
DESIGN.md §17, no more (smallness is the correctness guarantee):

 1. `:d_d_zero` — `d(d(x)) → 0`.
 2. `:codifferential_codifferential_zero` — `δ(δ(x)) → 0`.
 3. `:star_star_sign` — `⋆(⋆(x)) → (-1)^(k(n-k)+q) ⋅ x`.
 4. `:linearity` — operators distribute over the n-ary linear node; scalars
    pull out; the linear node canonicalizes (flatten, drop zero terms,
    collapse unit singletons).

The definitional substitutions `δ ↔ ±⋆d⋆` and `Δ ↔ dδ + δd` are deliberately
NOT in this registry: they are explicit, user-invoked expansions via
[`expand`](@ref), never automatic rewrites.

The registry is **data** (a §8 extension surface): mutate it with
[`register_rule!`](@ref) / [`unregister_rule!`](@ref), or pass any other
ordered rule vector to [`simplify`](@ref) via its `rules` keyword.
"""
const DEFAULT_RULES = RewriteRule[
    RewriteRule(:d_d_zero,
        "d∘d = 0 — the strict, exact operator identity of the topological " *
        "coboundary (L8: ∂∘∂ = 0 with signs, verified in the suite on every " *
        "shipped base, including dual cochains via the L8.3 transpose-derived " *
        "dual incidence).",
        _rule_dd),
    RewriteRule(:codifferential_codifferential_zero,
        "δ∘δ = 0 — the Hodge adjoint of d∘d = 0 (δ = ±⋆d⋆, so δδ = ±⋆d⋆⋆d⋆ " *
        "vanishes by d∘d = 0 and ⋆⋆ = ±1; verified numerically in the L8.2 " *
        "suite: codifferential(codifferential(α)) is the zero field).",
        _rule_codiff_codiff),
    RewriteRule(:star_star_sign,
        "⋆⋆ = (-1)^(k(n-k)+q) — the star-square sign law for non-degenerate " *
        "signature with q negative directions, externally verified across " *
        "Euclidean (q=0) and Lorentzian (q=1) in the L8.3 verification script " *
        "and the L8.2 suite.  The sign is computed from the inner expression's " *
        "inferred grade and the base's declared signature; k(n-k) is symmetric " *
        "under k ↔ n-k, so one formula serves both residences.",
        _rule_star_star),
    RewriteRule(:linearity,
        "d, ⋆, δ, Δ are linear operators (the coboundary is a signed sum; the " *
        "Hodge value map is cellwise linear) — they distribute over the n-ary " *
        "linear node and scalars pull out.  Includes the linear node's own " *
        "canonicalization: nested combinations flatten (c⋅(Σdᵢtᵢ) = Σ(c·dᵢ)tᵢ), " *
        "zero-coefficient and typed-zero terms drop (c⋅0 = 0), and a singleton " *
        "with unit coefficient collapses to its term.",
        _rule_linearity),
]

"""
    register_rule!(rule::RewriteRule; rules::Vector{RewriteRule} = DEFAULT_RULES)

Append a rule to a registry (default: the global [`DEFAULT_RULES`](@ref)).
Throws if a rule of the same name is already present.  Returns the registry.
This is the documented §8 extension surface for the blackboard: new node
types ship their rules by registering data, not by editing the simplifier.
"""
function register_rule!(rule::RewriteRule; rules::Vector{RewriteRule} = DEFAULT_RULES)
    any(r -> r.name === rule.name, rules) && throw(ArgumentError(
        "a rewrite rule named :$(rule.name) is already registered; " *
        "unregister it first or choose a different name"))
    push!(rules, rule)
    rules
end

"""
    unregister_rule!(name::Symbol; rules::Vector{RewriteRule} = DEFAULT_RULES)

Remove the rule named `name` from a registry (default: the global
[`DEFAULT_RULES`](@ref)); throws an `ArgumentError` if no such rule is
registered.  Returns the registry.  Because the registry is data, removal
exactly restores the prior rewriting behavior.
"""
function unregister_rule!(name::Symbol; rules::Vector{RewriteRule} = DEFAULT_RULES)
    i = findfirst(r -> r.name === name, rules)
    i === nothing && throw(ArgumentError(
        "no rewrite rule named :$name is registered"))
    deleteat!(rules, i)
    rules
end

# ── Simplification — deterministic strategy, swappable as a function ──────────

const SIMPLIFY_PASS_CAP = 256   # whole-tree passes per simplify call
const SIMPLIFY_NODE_CAP = 1024  # rule applications at a single node per visit

function _apply_at_node(e::BlackboardExpr, rules)
    changed = false
    for _ in 1:SIMPLIFY_NODE_CAP
        fired = false
        for r in rules
            e2 = r.apply(e)
            e2 === nothing && continue
            e = e2
            fired = true
            changed = true
            break
        end
        fired || return (e, changed)
    end
    throw(ArgumentError(
        "rewrite rules fired $SIMPLIFY_NODE_CAP times at a single node without " *
        "settling; a registered rule is likely non-terminating (its apply must " *
        "return nothing once its work is done)"))
end

function _rewrite_pass(e::BlackboardExpr, rules)
    e, changed = _apply_at_node(e, rules)
    kids = expr_children(e)
    isempty(kids) && return (e, changed)
    newkids = Vector{BlackboardExpr}(undef, length(kids))
    kidschanged = false
    for (i, k) in enumerate(kids)
        nk, ch = _rewrite_pass(k, rules)
        newkids[i] = nk
        kidschanged |= ch
    end
    kidschanged || return (e, changed)
    (expr_rebuild(e, newkids), true)
end

"""
    topdown_once_to_fixpoint(e::BlackboardExpr, rules) -> BlackboardExpr

The default [`simplify`](@ref) strategy: a simple, deterministic top-down
rewrite pass (apply the first matching rule at each node until none fires,
then descend), repeated until a whole pass makes no change, with an iteration
cap of $(SIMPLIFY_PASS_CAP) passes (an informative error is thrown if the cap
is hit — only a non-terminating registered rule can cause that).  No e-graphs,
no search, no CAS (§17.1).
"""
function topdown_once_to_fixpoint(e::BlackboardExpr, rules)
    for _ in 1:SIMPLIFY_PASS_CAP
        e, changed = _rewrite_pass(e, rules)
        changed || return e
    end
    throw(ArgumentError(
        "simplify did not reach a fixpoint within $SIMPLIFY_PASS_CAP passes; " *
        "a registered rule is likely cycling"))
end

"""
    simplify(e::BlackboardExpr; rules = DEFAULT_RULES,
             strategy = topdown_once_to_fixpoint) -> BlackboardExpr
    simplify(eq::Equation; kwargs...) -> Equation

Rewrite an expression with the verified-identity registry.  Both the rule set
(`rules`, ordered data — see [`DEFAULT_RULES`](@ref)) and the rewriting
`strategy` (a function `(expr, rules) -> expr`) are arguments, so each is
swappable independently (§17.1).  Applied to an [`Equation`](@ref), simplifies
both sides; the result of equation simplification re-typechecks by
construction.

!!! note
    `Symbolics` also exports a `simplify` (the coefficient-level scalar CAS —
    a different layer, §17).  With both packages loaded, qualify:
    `Tensorsmith.simplify` / `Symbolics.simplify`.
"""
simplify(e::BlackboardExpr; rules = DEFAULT_RULES,
         strategy = topdown_once_to_fixpoint) = strategy(e, rules)

simplify(eq::Equation; kwargs...) =
    Equation(simplify(eq.lhs; kwargs...), simplify(eq.rhs; kwargs...))

# ── Definitional expansions — explicit, user-invoked, never automatic ─────────

"""
    expand(e::BlackboardExpr, which::Symbol) -> BlackboardExpr
    expand(eq::Equation, which::Symbol) -> Equation

Apply one of the **definitional substitutions** everywhere in an expression —
explicitly, on user invocation only; these are deliberately not automatic
rewrite rules (§17):

  - `:codifferential` — `δ(x) → (-1)^(n(k+1)+q) ⋅ ⋆(d(⋆(x)))` (the engine's
    own verified definition, hodge.jl, with the L8.2.1 adjointness-pinned
    sign), where `k = expr_grade(x)`, `n = top_grade(base)`, and `q` is the
    number of negative directions in `signature(base)`.  On a grade-0 argument
    `δ` is the zero 0-section (engine convention), so the expansion is the
    typed zero.
  - `:laplacian` — `Δ(x) → d(δ(x)) + δ(d(x))` (the engine's verified
    definition).  Mirroring the engine's grade guards, the `d∘δ` term is the
    typed zero on grade 0 and the `δ∘d` term is the typed zero at the top
    grade, so the expansion never builds a node the engine could not evaluate.

Unknown `which` throws an `ArgumentError`.

!!! note
    `Symbolics` also exports an `expand` (coefficient-level).  With both
    packages loaded, qualify: `Tensorsmith.expand`.
"""
function expand(e::BlackboardExpr, which::Symbol)
    which === :codifferential || which === :laplacian || throw(ArgumentError(
        "expand recognizes :codifferential (δ ↔ ±⋆d⋆) and :laplacian " *
        "(Δ ↔ dδ + δd); got :$which"))
    _expand(e, which)
end

expand(eq::Equation, which::Symbol) =
    Equation(expand(eq.lhs, which), expand(eq.rhs, which))

# Generic recursion through the open node interface; the two definitional
# nodes specialize below.  A future node type participates automatically.
function _expand(e::BlackboardExpr, which::Symbol)
    kids = expr_children(e)
    isempty(kids) && return e
    expr_rebuild(e, BlackboardExpr[_expand(k, which) for k in kids])
end

function _expand(e::CodiffExpr, which::Symbol)
    a = _expand(e.arg, which)
    which === :codifferential || return CodiffExpr(a)
    b = expr_base(a)
    n = top_grade(b)
    k = expr_grade(a)
    k == 0 && return ZeroExpr(b, 0; residence = :primal)
    q = signature(b)[2]
    chain = StarExpr(DExpr(StarExpr(a)))
    isodd(n * (k + 1) + q) ? _lincomb([-1], BlackboardExpr[chain]) : chain
end

function _expand(e::LaplacianExpr, which::Symbol)
    a = _expand(e.arg, which)
    which === :laplacian || return LaplacianExpr(a)
    b = expr_base(a)
    n = top_grade(b)
    k = expr_grade(a)
    terms = BlackboardExpr[]
    k == 0 || push!(terms, DExpr(CodiffExpr(a)))     # dδ vanishes on 0-forms
    k >= n || push!(terms, CodiffExpr(DExpr(a)))     # δd vanishes at top grade
    isempty(terms) && return ZeroExpr(b, k; residence = :primal)
    length(terms) == 1 ? terms[1] : _lincomb([1, 1], terms)
end

# ── Substitution ──────────────────────────────────────────────────────────────

"""
    substitute(e::BlackboardExpr, var => replacement) -> BlackboardExpr
    substitute(eq::Equation, var => replacement) -> Equation

Replace every occurrence of the [`FieldVar`](@ref) `var` by `replacement` —
another expression or a concrete `Field`/`HodgeDualField` (wrapped as a
literal leaf).  The replacement is **typechecked**: it must match the
variable's (grade, residence, base) exactly, else an `ArgumentError` names the
disagreement.

!!! note
    `Symbolics` also exports a `substitute` (coefficient-level).  With both
    packages loaded, qualify: `Tensorsmith.substitute`.
"""
function substitute(e::BlackboardExpr, sub::Pair{<:FieldVar,<:Any})
    var = sub.first
    rep = _as_expr(sub.second)
    expr_base(rep) === var.base || throw(ArgumentError(
        "substitute: the replacement for :$(var.name) lives over a different " *
        "base instance than the variable"))
    expr_grade(rep) == var.grade || throw(ArgumentError(
        "substitute: the replacement for :$(var.name) has grade " *
        "$(expr_grade(rep)) but the variable declares grade $(var.grade)"))
    expr_residence(rep) === var.residence || throw(ArgumentError(
        "substitute: the replacement for :$(var.name) is " *
        "$(expr_residence(rep)) but the variable declares $(var.residence)"))
    _subst(e, var, rep)
end

substitute(eq::Equation, sub::Pair{<:FieldVar,<:Any}) =
    Equation(substitute(eq.lhs, sub), substitute(eq.rhs, sub))

function _subst(e::BlackboardExpr, var::FieldVar, rep::BlackboardExpr)
    e isa FieldVar && e == var && return rep
    kids = expr_children(e)
    isempty(kids) && return e
    expr_rebuild(e, BlackboardExpr[_subst(k, var, rep) for k in kids])
end

# ── Evaluation — bind placeholders, call the real verified operators ──────────

# Local helpers for combining evaluated sections.  Primal Fields delegate to
# the existing arithmetic; dual cochains are combined here cellwise (the same
# zero-pruned construction Field uses) so the blackboard adds no methods on
# the locked container types.
_add(a::Field, b::Field) = a + b

function _add(a::HodgeDualField{R,E,B}, b::HodgeDualField{R,E,B}) where {R,E,B}
    a.base === b.base || throw(ArgumentError(
        "cannot combine dual cochains over different base instances"))
    a.grade == b.grade || throw(ArgumentError(
        "cannot combine dual cochains of different grades ($(a.grade) vs $(b.grade))"))
    vals = Dict{Int,E}()
    for c in union(keys(a.values), keys(b.values))
        s = evaluate(a, c) + evaluate(b, c)
        iszero(s) || (vals[c] = s)
    end
    HodgeDualField{R,E,B}(a.base, a.grade, vals)
end

_sub(a, b) = _add(a, _scale(-1, b))

function _scale(c, f::Union{Field{R,E,B},HodgeDualField{R,E,B}}) where {R,E,B}
    c isa Integer && return c * f
    c isa R && return c * f
    cc = try
        R(c)
    catch
        throw(ArgumentError(
            "a coefficient of type $(typeof(c)) cannot act on a section with " *
            "scalar ring $R; use a coefficient in the ring (or an Integer)"))
    end
    cc * f
end

function _zero_section(base::BaseSpace, grade::Int, residence::Symbol)
    g = min(grade, top_grade(base))
    cs = cells(base, g)
    isempty(cs) && throw(ArgumentError(
        "cannot construct the typed zero section: the base has no grade-$g " *
        "cells from which to recover the fibre"))
    E = fibre_eltype(fibre(base, g, first(cs)))
    residence === :primal ? Field(base, grade, Dict{Int,E}()) :
                            HodgeDualField(base, grade, Dict{Int,E}())
end

"""
    evaluate(e::BlackboardExpr; bindings = Dict{Symbol,Any}()) -> Field | HodgeDualField

Evaluate an expression to a concrete section by binding every
[`FieldVar`](@ref) (by name) to a concrete `Field`/`HodgeDualField` and
calling the **real, verified** engine operators (`d`, `⋆`, `δ`, `Δ`) on the
results — the same operators the L8 suite certifies, so symbolic and numeric
agree on one equation object (§17).

Every variable must be bound to a section matching its (grade, residence,
base) exactly; an unbound or mistyped variable throws an informative
`ArgumentError` naming the variable and the mismatch.
"""
evaluate(e::BlackboardExpr;
         bindings::AbstractDict{Symbol,<:Any} = Dict{Symbol,Any}()) =
    _evaluate(e, bindings)

function _evaluate(v::FieldVar, bindings)
    haskey(bindings, v.name) || throw(ArgumentError(
        "FieldVar :$(v.name) is unbound; pass bindings = Dict(:$(v.name) => " *
        "a grade-$(v.grade) $(v.residence) section over the variable's base)"))
    val = bindings[v.name]
    if v.residence === :primal
        val isa Field || throw(ArgumentError(
            "the binding for :$(v.name) must be a primal Field; got $(typeof(val))"))
    else
        val isa HodgeDualField || throw(ArgumentError(
            "the binding for :$(v.name) must be a HodgeDualField (dual cochain); " *
            "got $(typeof(val))"))
    end
    val.base === v.base || throw(ArgumentError(
        "the binding for :$(v.name) lives over a different base instance than " *
        "the variable"))
    field_grade(val) == v.grade || throw(ArgumentError(
        "the binding for :$(v.name) has grade $(field_grade(val)) but the " *
        "variable declares grade $(v.grade)"))
    val
end

_evaluate(l::FieldLit, bindings) = l.field
_evaluate(z::ZeroExpr, bindings) = _zero_section(z.base, z.grade, z.residence)

_evaluate(e::DExpr, bindings)         = d(_evaluate(e.arg, bindings))
_evaluate(e::StarExpr, bindings)      = hodge_star(expr_base(e.arg), _evaluate(e.arg, bindings))
_evaluate(e::CodiffExpr, bindings)    = codifferential(expr_base(e.arg), _evaluate(e.arg, bindings))
_evaluate(e::LaplacianExpr, bindings) = hodge_laplacian(expr_base(e.arg), _evaluate(e.arg, bindings))

function _evaluate(e::LinearCombination, bindings)
    acc = _scale(e.coeffs[1], _evaluate(e.terms[1], bindings))
    for i in 2:length(e.terms)
        acc = _add(acc, _scale(e.coeffs[i], _evaluate(e.terms[i], bindings)))
    end
    acc
end

"""
    residual(eq::Equation; bindings = Dict{Symbol,Any}()) -> Field | HodgeDualField

`evaluate(lhs) - evaluate(rhs)` — the concrete section by which the bound
equation fails to hold.  The zero section (everywhere empty) means the
equation holds exactly.
"""
residual(eq::Equation; bindings::AbstractDict{Symbol,<:Any} = Dict{Symbol,Any}()) =
    _sub(_evaluate(eq.lhs, bindings), _evaluate(eq.rhs, bindings))

"""
    check(eq::Equation; bindings = Dict{Symbol,Any}()) -> Bool

Whether the bound equation holds: `true` iff the [`residual`](@ref) is the
zero section, decided by the field's own zero test (sections are zero-pruned
at construction, so an empty store *is* the zero section).

Over **exact rings** (the `Rational{BigInt}` default, and structurally zero
symbolic coefficients) this is an exact decision.  Over **floating-point
rings** the test is still *exact* zero — roundoff residuals of magnitude
`eps`-ish count as nonzero, so a float `check` failing may mean "true up to
roundoff"; inspect `residual` instead of loosening this test.
"""
check(eq::Equation; bindings::AbstractDict{Symbol,<:Any} = Dict{Symbol,Any}()) =
    length(residual(eq; bindings = bindings)) == 0

# ── Output formatting: Unicode show + LaTeX (output only) ─────────────────────

Base.show(io::IO, v::FieldVar) = print(io, v.name)
Base.show(io::IO, ::ZeroExpr)  = print(io, "0")
Base.show(io::IO, l::FieldLit) = print(io, "⟨", l.field, "⟩")
Base.show(io::IO, e::DExpr)         = print(io, "d(", e.arg, ")")
Base.show(io::IO, e::StarExpr)      = print(io, "⋆(", e.arg, ")")
Base.show(io::IO, e::CodiffExpr)    = print(io, "δ(", e.arg, ")")
Base.show(io::IO, e::LaplacianExpr) = print(io, "Δ(", e.arg, ")")

function Base.show(io::IO, e::LinearCombination)
    for i in eachindex(e.terms)
        i > 1 && print(io, " + ")
        c = e.coeffs[i]
        isequal(c, one(c)) || print(io, c, "⋅")
        t = e.terms[i]
        t isa LinearCombination ? print(io, "(", t, ")") : print(io, t)
    end
end

Base.show(io::IO, eq::Equation) = print(io, eq.lhs, " = ", eq.rhs)

"""
    latex(e::BlackboardExpr) -> String
    latex(eq::Equation) -> String

A LaTeX rendering of an expression or equation (`d`, `\\star`, `\\delta`,
`\\Delta`, `\\cdot`, coefficients as printed).  **Output only** — the
blackboard parses no LaTeX (and no strings at all); input is Julia itself
(DESIGN.md §17).
"""
latex(v::FieldVar) = string(v.name)
latex(::ZeroExpr)  = "0"
latex(l::FieldLit) = "\\langle\\text{bound field}\\rangle"
latex(e::DExpr)         = "d\\left(" * latex(e.arg) * "\\right)"
latex(e::StarExpr)      = "\\star\\left(" * latex(e.arg) * "\\right)"
latex(e::CodiffExpr)    = "\\delta\\left(" * latex(e.arg) * "\\right)"
latex(e::LaplacianExpr) = "\\Delta\\left(" * latex(e.arg) * "\\right)"

function latex(e::LinearCombination)
    parts = String[]
    for i in eachindex(e.terms)
        c = e.coeffs[i]
        t = e.terms[i]
        body = t isa LinearCombination ? "\\left(" * latex(t) * "\\right)" : latex(t)
        push!(parts, isequal(c, one(c)) ? body : string(c) * " \\cdot " * body)
    end
    join(parts, " + ")
end

latex(eq::Equation) = latex(eq.lhs) * " = " * latex(eq.rhs)

# ── Exports ───────────────────────────────────────────────────────────────────

export BlackboardExpr,
       FieldVar, FieldLit, ZeroExpr, zero_expr,
       DExpr, StarExpr, CodiffExpr, LaplacianExpr, LinearCombination,
       expr_base, expr_grade, expr_residence, expr_children, expr_rebuild,
       Equation, lhs, rhs,
       RewriteRule, DEFAULT_RULES, register_rule!, unregister_rule!,
       simplify, topdown_once_to_fixpoint, expand, substitute,
       residual, check, latex
