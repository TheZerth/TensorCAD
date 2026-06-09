# ── Phase L7: Field — a section of the bundle ─────────────────────────────────
#
# A `Field` is a SECTION over a `BaseSpace`: for a fixed grade `k` it assigns a
# fibre element (one of the existing `AbstractTensorElement` types) to each
# k-cell of the base.  The bundle-vs-section split is deliberate and load-bearing
# (DESIGN.md §13):
#
#     STATE  lives in the Field   (which element sits on each cell),
#     STRUCTURE lives in the BaseSpace (cells, boundary, fibre, transport).
#
# The field stores only the cells whose value is non-zero (a sparse section);
# `evaluate` returns the fibre zero on any unstored cell.  The fibre element type
# `E` is a type parameter, recovered from the base's `FibreDescriptor` by
# dispatch, so element access is type-stable (`evaluate`/`getindex` infer to `E`).
#
# Pointwise arithmetic is exactly that — cellwise: like-typed fields add and
# negate cell by cell, and a ring scalar scales every cell.  There is no product
# of fields here (that would need the differential/algebraic operators of L8).

"""
    Field{R, E, B}

A grade-`k` section over a [`BaseSpace`](@ref) `B`: a sparse assignment of fibre
elements of type `E <: AbstractTensorElement{R}` to the `k`-cells of the base.

Construct with [`Field`](@ref Field(::BaseSpace, ::Integer, ::Dict)) from an
explicit `Dict{Int,E}` (cell id → element) or from a function `cell -> element`.
Access is via [`evaluate`](@ref) / `field[cell]` (returns the fibre zero on an
unstored cell), iteration yields the stored `cell => element` pairs, and the
pointwise arithmetic `+`, `-`, and scalar `*` is supported.

The field is the *section*; the base is the *bundle* — state here, structure
there (DESIGN.md §13).
"""
struct Field{R, E<:AbstractTensorElement{R}, B<:BaseSpace}
    base   :: B
    grade  :: Int
    values :: Dict{Int, E}

    function Field{R,E,B}(base::B, grade::Int, values::Dict{Int,E}
                          ) where {R, E<:AbstractTensorElement{R}, B<:BaseSpace}
        0 <= grade <= top_grade(base) || throw(ArgumentError(
            "field grade $grade is out of range 0:$(top_grade(base)) for $(typeof(base))"))
        valid = cells(base, grade)
        for (cid, x) in values
            (cid in valid) || throw(ArgumentError(
                "cell id $cid is not a grade-$grade cell of the base"))
            fibre_matches(fibre(base, grade, cid), x) || throw(ArgumentError(
                "the value assigned at cell $cid does not belong to the fibre " *
                "attached there (wrong element type, metric, or space)"))
        end
        new{R,E,B}(base, grade, values)
    end
end

# ── Constructors ──────────────────────────────────────────────────────────────

"""
    Field(base::BaseSpace, grade::Integer, values::Dict{Int,E}) -> Field

Build a grade-`grade` section from an explicit map of cell id → fibre element.
Every key must be a `grade`-cell of `base`, and every value must belong to the
fibre attached there (checked via [`fibre_matches`](@ref)).
"""
function Field(base::B, grade::Integer, values::Dict{Int,E}
               ) where {B<:BaseSpace, R, E<:AbstractTensorElement{R}}
    Field{R,E,B}(base, Int(grade), values)
end

"""
    Field(base::BaseSpace, grade::Integer, f::Function) -> Field

Build a grade-`grade` section by evaluating `f(cell)` on every `grade`-cell of
`base`.  The element type is inferred from `f`'s return value, so the cell set
must be non-empty.
"""
function Field(base::BaseSpace, grade::Integer, f::Function)
    g  = Int(grade)
    cs = collect(cells(base, g))
    isempty(cs) && throw(ArgumentError(
        "cannot infer the fibre element type from an empty cell set; " *
        "pass an explicit Dict{Int,E} instead"))
    Field(base, g, Dict(c => f(c) for c in cs))
end

"""
    zero_field(base::BaseSpace, grade::Integer, descriptor::FibreDescriptor) -> Field

The everywhere-zero grade-`grade` section whose fibre is `descriptor` (an empty
sparse store; [`evaluate`](@ref) returns the fibre zero on every cell).
"""
function zero_field(base::BaseSpace, grade::Integer, d::FibreDescriptor)
    E = fibre_eltype(d)
    Field(base, Int(grade), Dict{Int,E}())
end

# ── Access ────────────────────────────────────────────────────────────────────

"""
    evaluate(fld::Field, cell::Integer) -> fibre element

The field's value at `cell`: the stored element if present, otherwise the fibre
zero ([`zero_fibre`](@ref)) of the fibre attached at that cell.  The return type
is the field's element parameter `E` (type-stable; recovered by dispatch on the
[`FibreDescriptor`](@ref)).  Also available as `fld[cell]`.
"""
function evaluate(fld::Field{R,E,B}, cell::Integer) where {R,E,B}
    c = Int(cell)
    haskey(fld.values, c) && return fld.values[c]
    return zero_fibre(fibre(fld.base, fld.grade, c))::E
end

Base.getindex(fld::Field, cell::Integer) = evaluate(fld, cell)

field_grade(fld::Field) = fld.grade

# ── Iteration / collection interface ──────────────────────────────────────────

Base.length(fld::Field)            = length(fld.values)
Base.keys(fld::Field)              = keys(fld.values)
Base.values(fld::Field)            = Base.values(fld.values)
Base.pairs(fld::Field)             = pairs(fld.values)
Base.haskey(fld::Field, c::Integer)= haskey(fld.values, Int(c))
Base.iterate(fld::Field, st...)    = iterate(fld.values, st...)
Base.eltype(::Type{Field{R,E,B}}) where {R,E,B} = Pair{Int,E}

# ── Equality ──────────────────────────────────────────────────────────────────
#
# Exact-ring equality only: over a symbolic ring (R = Symbolics.Num) `values ==`
# would invoke `Num == Num`, which is not a Bool.  For symbolic fields compare
# cellwise with `isequal_simplified(evaluate(a,c), evaluate(b,c))` instead
# (matching the convention used throughout the codebase).

Base.:(==)(a::Field{R,E,B}, b::Field{R,E,B}) where {R,E,B} =
    a.base === b.base && a.grade == b.grade && a.values == b.values

# ── Pointwise arithmetic ──────────────────────────────────────────────────────

function _check_field_compatible(a::Field, b::Field)
    a.base === b.base || throw(ArgumentError(
        "cannot combine fields over different base instances"))
    a.grade == b.grade || throw(ArgumentError(
        "cannot combine fields of different grades ($(a.grade) vs $(b.grade))"))
end

function Base.:+(a::Field{R,E,B}, b::Field{R,E,B}) where {R,E,B}
    _check_field_compatible(a, b)
    vals = Dict{Int,E}()
    for c in union(keys(a.values), keys(b.values))
        s = evaluate(a, c) + evaluate(b, c)
        iszero(s) || (vals[c] = s)
    end
    Field{R,E,B}(a.base, a.grade, vals)
end

Base.:-(t::Field{R,E,B}) where {R,E,B} =
    Field{R,E,B}(t.base, t.grade, Dict{Int,E}(c => -x for (c, x) in t.values))

Base.:-(a::Field{R,E,B}, b::Field{R,E,B}) where {R,E,B} = a + (-b)

function Base.:*(c::R, t::Field{R,E,B}) where {R,E,B}
    vals = Dict{Int,E}()
    for (cell, x) in t.values
        y = c * x
        iszero(y) || (vals[cell] = y)
    end
    Field{R,E,B}(t.base, t.grade, vals)
end

Base.:*(t::Field{R,E,B}, c::R) where {R,E,B}       = c * t
Base.:*(n::Integer, t::Field{R,E,B}) where {R,E,B} = R(n) * t
Base.:*(t::Field{R,E,B}, n::Integer) where {R,E,B} = R(n) * t

# ── Display ───────────────────────────────────────────────────────────────────

function Base.show(io::IO, fld::Field{R,E,B}) where {R,E,B}
    print(io, "Field(grade $(fld.grade); $(length(fld.values))/",
              "$(n_cells(fld.base, fld.grade)) cells populated; ",
              "fibre $E over $(nameof(B)))")
end

# ── Exports ───────────────────────────────────────────────────────────────────

export Field, evaluate, zero_field, field_grade
