#!/usr/bin/env julia

# Tensorsmith L8.3 dual-correspondence verification/inspection script.
#
# This script is intentionally print-only: no @test, no assertions.  Expected
# DEC facts below are derived from the primal GridBase boundary/cell structure
# plus explicitly coded textbook cubical-dual rules; they are not obtained by
# calling dual_cell/hodge_star and comparing those functions to themselves.

using Tensorsmith

const R = Rational{BigInt}

signstr(s::Integer) = s > 0 ? "+1" : s < 0 ? "-1" : "0"
matchstr(ok::Bool) = ok ? "[MATCH]" : "[MISMATCH]"
orientstr(o::Symbol) = o === :H ? "H" : o === :V ? "V" : string(o)

function vertex_ij(b::GridBase, v::Integer)
    z = Int(v) - 1
    return (z % (b.nx + 1), z ÷ (b.nx + 1))
end

function primal_edge_local_from_documented_layout(b::GridBase, e::Integer)
    # Public GridBase docs: horizontal edges first, then vertical edges.
    nh = b.nx * (b.ny + 1)
    ee = Int(e)
    if ee <= nh
        z = ee - 1
        return (:H, z % b.nx, z ÷ b.nx)
    else
        z = ee - nh - 1
        return (:V, z % (b.nx + 1), z ÷ (b.nx + 1))
    end
end

function dual_edge_orientation_from_documented_layout(b::GridBase, de::Integer)
    # Public dual_cell docs: dual 1-cell index space is horizontal dual edges
    # first (crossing primal verticals), followed by vertical dual edges
    # (crossing primal horizontals).  The horizontal-dual count is the number of
    # primal vertical edges: (nx+1)*ny.  Check against dual_n_cells for bounds.
    dde = Int(de)
    dual_total = dual_n_cells(b, 1)
    dual_horizontal_count = (b.nx + 1) * b.ny
    if !(1 <= dde <= dual_total)
        return :OUT_OF_RANGE
    end
    return dde <= dual_horizontal_count ? :H : :V
end

function expected_dual_edge_id_from_textbook_cubical_layout(b::GridBase, e::Integer)
    # Independent expectation: a primal horizontal edge is crossed by a vertical
    # dual edge at the same grid-local slot; a primal vertical edge is crossed by
    # a horizontal dual edge at the same grid-local slot.  This uses only the
    # documented primal and dual edge layouts, not dual_cell.
    orient, i, j = primal_edge_local_from_documented_layout(b, e)
    dual_horizontal_count = (b.nx + 1) * b.ny
    if orient === :H
        # vertical dual edges come after the horizontal-dual block; local range
        # i=0:nx-1, j=0:ny, count nx*(ny+1).
        return dual_horizontal_count + i + j * b.nx + 1
    else
        # horizontal dual edges local range i=0:nx, j=0:ny-1.
        return i + j * (b.nx + 1) + 1
    end
end

expected_dual_orientation_for_primal(o::Symbol) = o === :H ? :V : :H

function incident_edges_from_primal_boundary(b::GridBase, v::Integer)
    out = Tuple{Int,Int}[]
    for e in cells(b, 1)
        for (face, s) in boundary(b, 1, e)
            if face == v
                push!(out, (Int(e), Int(s)))
            end
        end
    end
    return out
end

function adjacent_faces_from_primal_boundary(b::GridBase, e::Integer)
    out = Tuple{Int,Int}[]
    for f in cells(b, 2)
        for (face, s) in boundary(b, 2, f)
            if face == e
                push!(out, (Int(f), Int(s)))
            end
        end
    end
    return out
end

function first_edge_with_adjacent_face_count(b::GridBase, count::Integer)
    for e in cells(b, 1)
        if length(adjacent_faces_from_primal_boundary(b, e)) == count
            return Int(e)
        end
    end
    return nothing
end

function chosen_vertices(b::GridBase)
    if b.nx == 2 && b.ny == 2
        return [
            ("center-ish vertex requested", 5),
            ("corner vertex", 1),
            ("edge-midpoint boundary vertex", 2),
        ]
    elseif b.nx == 3 && b.ny == 2
        # Interior vertices have 0<i<nx and 0<j<ny.  Pick (1,1).
        interior = 1 + 1 * (b.nx + 1) + 1
        return [
            ("interior vertex chosen", interior),
            ("corner vertex", 1),
            ("edge-midpoint boundary vertex", 2),
        ]
    else
        error("No chosen-vertex policy for GridBase($(b.nx),$(b.ny))")
    end
end

function tensor_unit_sign(m, x)
    one = clifford_one(m)
    if x == one
        return 1
    elseif x == -one
        return -1
    elseif iszero(x)
        return 0
    else
        return "NONUNIT($(x))"
    end
end

function impl_dual_face_boundary_sign(b::GridBase, dual_edge::Integer, dual_face::Integer)
    # Public implementation inspection: construct a basis dual 1-cochain on the
    # implementation's reported dual edge, apply public d to the dual cochain,
    # and read its value at the implementation's reported dual face.  This is not
    # used to build the expected sign; it is only printed/compared as impl output.
    η = HodgeDualField(b, 1, Dict(Int(dual_edge) => clifford_one(b.metric)))
    dη = d(η)
    return tensor_unit_sign(b.metric, evaluate(dη, Int(dual_face)))
end

function selected_edge_facts(b::GridBase)
    interior = first_edge_with_adjacent_face_count(b, 2)
    boundary_edge = first_edge_with_adjacent_face_count(b, 1)
    edges = Int[]
    interior !== nothing && push!(edges, interior)
    boundary_edge !== nothing && push!(edges, boundary_edge)
    faces = sort(unique([f for e in edges for (f, _s) in adjacent_faces_from_primal_boundary(b, e)]))
    return edges, faces, interior, boundary_edge
end

function print_primal_facts(b::GridBase)
    println("SECTION 1 -- primal facts (ground truth from cells/n_cells/boundary only)")
    println("GridBase(nx=$(b.nx), ny=$(b.ny))")
    for k in 0:2
        println("  n_cells(k=$k) = $(n_cells(b, k)); cells = $(collect(cells(b, k)))")
    end

    println("  Chosen vertices and incident primal edges (e incident to v iff v appears in boundary(grid,1,e)):")
    for (label, v) in chosen_vertices(b)
        i, j = vertex_ij(b, v)
        println("    $label: v=$v  (i,j)=($i,$j)")
        inc = incident_edges_from_primal_boundary(b, v)
        for (e, s) in inc
            orient, ei, ej = primal_edge_local_from_documented_layout(b, e)
            println("      edge e=$e  ∂e sign on v=$(signstr(s))  primal orient=$(orientstr(orient)) local=($ei,$ej)  boundary=$(boundary(b, 1, e))")
        end
    end

    edges, _faces, interior, boundary_edge = selected_edge_facts(b)
    println("  Chosen primal edges and adjacent primal faces (f adjacent iff e appears in boundary(grid,2,f)):")
    for e in edges
        kind = e == interior ? "interior primal edge" : e == boundary_edge ? "boundary primal edge" : "selected primal edge"
        orient, ei, ej = primal_edge_local_from_documented_layout(b, e)
        println("    $kind: e=$e  primal orient=$(orientstr(orient)) local=($ei,$ej)")
        adjs = adjacent_faces_from_primal_boundary(b, e)
        for (f, s) in adjs
            println("      face f=$f  ∂f sign on e=$(signstr(s))  face boundary=$(boundary(b, 2, f))")
        end
    end
end

function print_impl_dual_correspondence(b::GridBase)
    println("SECTION 2 -- implementation dual correspondence (dual_cell output)")
    dual_h = (b.nx + 1) * b.ny
    println("  dual_n_cells(k=0..2) = $([dual_n_cells(b, k) for k in 0:2])")
    println("  Dual-edge orientation determination: documented dual-1 layout has H ids 1:$dual_h (cross primal verticals), V ids $(dual_h + 1):$(dual_n_cells(b,1)); total dual_n_cells(grid,1)=$(dual_n_cells(b,1)).")

    println("  Chosen vertices:")
    for (label, v) in chosen_vertices(b)
        i, j = vertex_ij(b, v)
        df = dual_cell(b, 0, v)
        println("    $label: v=$v (i,j)=($i,$j) -> dual_cell(grid,0,v) = dual face id $df")
        for (e, _s) in incident_edges_from_primal_boundary(b, v)
            de = dual_cell(b, 1, e)
            dorient = dual_edge_orientation_from_documented_layout(b, de)
            println("      incident e=$e -> dual_cell(grid,1,e)=$de, impl dual orient=$(orientstr(dorient))")
        end
    end

    edges, faces, _interior, _boundary_edge = selected_edge_facts(b)
    println("  Chosen edges:")
    for e in edges
        de = dual_cell(b, 1, e)
        dorient = dual_edge_orientation_from_documented_layout(b, de)
        println("    e=$e -> dual_cell(grid,1,e)=$de, impl dual orient=$(orientstr(dorient))")
    end
    println("  Faces touched by the selected edge-adjacency checks:")
    for f in faces
        println("    f=$f -> dual_cell(grid,2,f) = dual vertex id $(dual_cell(b, 2, f))")
    end
end

function print_independent_dec_expectation(b::GridBase)
    println("SECTION 3 -- independent DEC expectation (from primal boundary + explicit transpose/perpendicular rules)")
    println("  Sign rule used: if vertex v has sign s in ∂e, then the dual edge crossing e has sign s in the boundary of the dual face around v (primal incidence transpose).")
    println("  Perpendicularity rule used: primal H edge -> expected V dual edge; primal V edge -> expected H dual edge.")
    println("  Expected dual edge ids are computed from the documented cubical layout, not from dual_cell.")
    for (label, v) in chosen_vertices(b)
        i, j = vertex_ij(b, v)
        expected_dual_face = v  # primal vertices enumerate dual faces one-for-one in textbook cubical correspondence
        println("    $label: v=$v (i,j)=($i,$j), expected dual face id=$expected_dual_face")
        for (e, s) in incident_edges_from_primal_boundary(b, v)
            porient, ei, ej = primal_edge_local_from_documented_layout(b, e)
            expected_orient = expected_dual_orientation_for_primal(porient)
            expected_de = expected_dual_edge_id_from_textbook_cubical_layout(b, e)
            impl_de = dual_cell(b, 1, e)
            impl_orient = dual_edge_orientation_from_documented_layout(b, impl_de)
            println("      primal e=$e local=($ei,$ej): ∂e sign on v=$(signstr(s)); primal orient=$(orientstr(porient)); expected dual edge id=$expected_de; expected dual orient=$(orientstr(expected_orient)); impl dual edge id=$impl_de; impl dual orient=$(orientstr(impl_orient))")
        end
    end
end

function print_verdicts(b::GridBase)
    println("SECTION 4 -- side-by-side verdict lines")
    total = 0
    mismatches = 0
    for (_label, v) in chosen_vertices(b)
        i, j = vertex_ij(b, v)
        impl_df = dual_cell(b, 0, v)
        expected_df = v
        println("vertex v=$v  (i,j)=($i,$j)")
        df_ok = impl_df == expected_df
        total += 1
        mismatches += df_ok ? 0 : 1
        println("  impl dual face id = $impl_df   expected dual face id = $expected_df   $(matchstr(df_ok))")
        for (e, s) in incident_edges_from_primal_boundary(b, v)
            porient, _ei, _ej = primal_edge_local_from_documented_layout(b, e)
            expected_orient = expected_dual_orientation_for_primal(porient)
            expected_de = expected_dual_edge_id_from_textbook_cubical_layout(b, e)
            impl_de = dual_cell(b, 1, e)
            impl_orient = dual_edge_orientation_from_documented_layout(b, impl_de)
            impl_sign = impl_dual_face_boundary_sign(b, impl_de, impl_df)
            expected_sign = s

            id_ok = impl_de == expected_de
            orient_ok = impl_orient == expected_orient
            sign_ok = impl_sign == expected_sign
            total += 3
            mismatches += (id_ok ? 0 : 1) + (orient_ok ? 0 : 1) + (sign_ok ? 0 : 1)

            println("  incident primal edge e=$e  ∂e sign on v = $(signstr(s))   primal orient=$(orientstr(porient))")
            println("     impl dual edge id = $impl_de   expected dual edge id = $expected_de        $(matchstr(id_ok))")
            println("     impl dual orient = $(orientstr(impl_orient))   expected dual orient = $(orientstr(expected_orient))        $(matchstr(orient_ok))")
            sign_txt = impl_sign isa Integer ? signstr(impl_sign) : string(impl_sign)
            println("     impl dual-face boundary sign = $sign_txt   expected (transpose) sign = $(signstr(expected_sign))        $(matchstr(sign_ok))")
        end
    end
    println("GRID SUMMARY: total checks = $total, mismatches = $mismatches")
end

function basis_value_for_grade(m, k::Integer)
    if k == 0
        return clifford_one(m)
    elseif k == 1
        return clifford_basis_vector(m, 1)
    elseif k == 2
        return clifford_basis_element(m, [1, 2])
    else
        error("basis_value_for_grade only supports k=0,1,2")
    end
end

function starstar_computed_sign(b::GridBase, k::Integer)
    c = first(cells(b, k))
    x = basis_value_for_grade(b.metric, k)
    fld = Field(b, k, Dict(Int(c) => x))
    ss = hodge_star(b, hodge_star(b, fld))
    if ss == fld
        return 1
    elseif ss == -fld
        return -1
    else
        return "not ± original basis cochain: $(ss)"
    end
end

function print_sign_law_spot_check()
    println("SECTION 5 -- sign-law spot check (independent inline expected formula)")
    grids = [
        ("Euclidean GridBase(2,2)", GridBase(2, 2), 0),
        ("Lorentzian GridBase(1,1) used in tests", GridBase(1, 1; metric = signature_metric(VectorSpace(2), R, 1, 1, 0)), 1),
    ]
    n = 2
    for (label, b, q) in grids
        println("  $label: n=$n, q=$q")
        for k in 0:2
            computed = starstar_computed_sign(b, k)
            expected = isodd(k * (n - k) + q) ? -1 : 1
            ok = computed == expected
            computed_txt = computed isa Integer ? signstr(computed) : string(computed)
            println("    k=$k: computed ⋆⋆ sign = $computed_txt; expected (-1)^(k*(n-k)+q) = $(signstr(expected))        $(matchstr(ok))")
        end
    end
end

function print_grid_report(b::GridBase)
    println("================================================================================")
    println("DUAL CORRESPONDENCE INSPECTION FOR GridBase($(b.nx),$(b.ny))")
    println("================================================================================")
    print_primal_facts(b)
    println()
    print_impl_dual_correspondence(b)
    println()
    print_independent_dec_expectation(b)
    println()
    print_verdicts(b)
    println()
end

println("Tensorsmith L8.3 dual correspondence verification/inspection")
println("Read-only note: this script prints public API outputs and independent expectations; it does not modify source/tests/design/QRCS files.")
println("Expectation note: expected dual-face boundaries are computed from primal boundary plus explicit DEC transpose/perpendicular rules, never by calling dual_cell/hodge_star.")
println()

print_grid_report(GridBase(2, 2))
print_grid_report(GridBase(3, 2))
print_sign_law_spot_check()
