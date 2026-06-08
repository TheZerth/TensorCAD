# ── Vector space ─────────────────────────────────────────────────────────────
#
# VectorSpace is a rank-n free R-module with named basis vectors.
#
# It carries NO metric — that is the Metric layer (Phase 3).  The labels are
# cosmetic: they affect display and nothing else.  Two VectorSpaces are equal
# iff they have the same rank and the same labels.
#
# Why store labels at all?  Because we want e₁⊗e₂ to print as "e1⊗e2", not
# "b_1⊗b_2", and the label lives closer to the space than to the tensor.
# If you define a space with custom labels (:x, :y, :z) your tensors will
# print in terms of those labels automatically.

"""
    VectorSpace(n, labels)
    VectorSpace(n)

A free R-module of rank `n` with ordered basis `{e₁, …, eₙ}`.

When called as `VectorSpace(n)`, labels default to `[:e1, :e2, …, :en]`.

# Examples
```julia
julia> VectorSpace(3)
VectorSpace(3; basis = [e1, e2, e3])

julia> VectorSpace(2, [:x, :y])
VectorSpace(2; basis = [x, y])
```
"""
struct VectorSpace
    n      :: Int
    labels :: Vector{Symbol}

    function VectorSpace(n::Int, labels::Vector{Symbol})
        n >= 0 ||
            throw(ArgumentError("Dimension must be ≥ 0, got $n"))
        length(labels) == n ||
            throw(ArgumentError(
                "Expected $n basis label(s), got $(length(labels))"))
        new(n, labels)
    end
end

VectorSpace(n::Int) = VectorSpace(n, [Symbol(:e, i) for i in 1:n])

Base.:(==)(a::VectorSpace, b::VectorSpace) = a.n == b.n && a.labels == b.labels
Base.hash(V::VectorSpace, h::UInt) = hash(V.labels, hash(V.n, h))

function Base.show(io::IO, V::VectorSpace)
    print(io, "VectorSpace($(V.n); basis = [$(join(V.labels, ", "))])")
end

export VectorSpace
