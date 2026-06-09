# ── Phase L8.1 (Tier 2): covariant derivative ∇ ──────────────────────────────
#
# MATHEMATICAL CONTRACT
#
# The discrete covariant derivative consumes the L7 connection/transport, not the
# boundary coboundary.  For an oriented edge `e : u → v` and a 0-field `ψ`, the
# convention implemented here is
#
#     (∇ψ)[e] = ψ[v] − transport(b, e)(ψ[u]).
#
# This works for either first-class transport realization from DESIGN.md §15.1:
# two-sided geometric/frame VersorTransport (`M ↦ VMV⁻¹`) and one-sided
# GaugeTransport (`ψ ↦ Uψ`).  It is deliberately distinct from `d`, which is the
# incidence/coboundary operator from exterior_derivative.jl.  The two converge in
# the continuum story but remain separate on a discrete base so topology and
# connection bugs cannot mask one another.

"""
    ∇(ψ::Field) -> Field
    covariant_derivative(ψ::Field) -> Field

Discrete covariant derivative of a 0-field using the base's edge transport.

For each oriented edge `e : u → v`, the returned 1-field is

```julia
(∇ψ)[e] = ψ[v] - transport(ψ.base, e)(ψ[u])
```

where reverse transport remains `inv(transport(...))` at the holonomy layer.  The
transport may be either the geometric two-sided [`VersorTransport`](@ref) or the
one-sided [`GaugeTransport`](@ref); `∇` is written only against the shared
callable interface.

`∇` is not [`d`](@ref): `d` is metric/connection-free incidence, while `∇` uses a
connection/potential.  With identity transport on a 0-field, this reduces exactly
to the bare edge difference (`d`/`grad` behavior): head value minus tail value.
"""
function ∇(ψ::Field{R,E,B}) where {R,E,B<:BaseSpace}
    field_grade(ψ) == 0 || throw(ArgumentError(
        "∇ currently acts on 0-fields as edge-wise covariant differences; got " *
        "a grade-$(field_grade(ψ)) field"))
    vals = Dict{Int,E}()
    b = ψ.base
    for e in cells(b, 1)
        tail, head = _edge_endpoints(b, e)
        diff = evaluate(ψ, head) - transport(b, e)(evaluate(ψ, tail))
        iszero(diff) || (vals[Int(e)] = diff)
    end
    Field{R,E,B}(b, 1, vals)
end

covariant_derivative(ψ::Field) = ∇(ψ)

export ∇, covariant_derivative
