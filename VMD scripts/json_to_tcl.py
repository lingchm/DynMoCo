# This script reads a JSON file containing community assignments over multiple timestamps
# and generates a single Tcl script for VMD that can dynamically update the visualization
# based on the current frame. Each community is assigned a unique color, and residues are
# selected based on their community membership at each timestamp.

import json

# User parameters
json_file = "results/a5b1_clamp/dynmoco/communities_clean.json"  # input JSON
output_tcl = "a5b1_clamp_example.tcl"
selection_type = "residue"
ntimestamps = 4 # number of timestamps in the JSON file

with open(json_file) as f:
    data = json.load(f)

# Gather all communities
communities_all = sorted({c for ts in data.values() for c in ts.keys()}, key=lambda x: int(x[1:]))
# Map communities to color IDs (0-16)
community_to_color = {comm: (i % 17) for i, comm in enumerate(communities_all)}

# Start building Tcl
tcl_lines = []
tcl_lines.append("set molID [molinfo top]")
tcl_lines.append("color Display Background white")

# Save residues per JSON timestamp per community
for ts, comms in data.items():
    for comm, residues in comms.items():
        if not residues:
            continue
        resid_list = " ".join(map(str, residues))
        tcl_lines.append(f"set community_data({ts},{comm}) {{{resid_list}}}")

# Store colors
for comm, cid in community_to_color.items():
    tcl_lines.append(f"set community_colors({comm}) {cid}")

# Sort communities in numeric order
tcl_lines.append("""
proc community_cmp {a b} {
    regexp {C(\\d+)} $a -> numA
    regexp {C(\\d+)} $b -> numB
    if {$numA < $numB} { return -1 }
    if {$numA > $numB} { return 1 }
    return 0
}
""")

# Assign residues to user variables per timestamp
user_vars = ["user", "user2", "user3", "user4"]
tcl_lines.append(f"""
set molid $molID

set ts_list {{}}
foreach key [array names community_data] {{
    lappend ts_list [lindex [split $key ,] 0]
}}
set ts_list [lsort -integer -unique $ts_list]

set user_vars [list {" ".join(user_vars)}]

for {{set i 0}} {{$i < [llength $ts_list]}} {{incr i}} {{
    set ts [lindex $ts_list $i]
    set user_var [lindex $user_vars $i]

    # loop over communities in this timestamp only
    foreach key [array names community_data] {{
        if {{[string equal [lindex [split $key ,] 0] $ts]}} {{
            set comm [lindex [split $key ,] 1]
            set resid_list $community_data($key)

            if {{[llength $resid_list] > 0}} {{
                set sel [atomselect $molid "{selection_type} [join $resid_list " "]"]
                regexp {{C(\d+)}} $comm -> comm_num
                $sel set $user_var $comm_num
                $sel delete
            }}
        }}
    }}
}}
""")

# Create representations for communities
tcl_lines.append("""
mol delrep all $molID
set repID 0
set all_comms [lsort -unique -command community_cmp [array names community_colors]]
foreach comm $all_comms {
    mol addrep $molID
    mol modselect $repID $molID "none"
    mol modcolor $repID $molID ColorID $community_colors($comm)
    mol modstyle $repID $molID NewCartoon
    set community_rep($comm) $repID
    incr repID
}
""")

# Frame assignment function
tcl_lines.append(f"""
proc assign_communities_to_frame {{json_ts}} {{
    global molID community_data community_rep community_colors
    set all_comms [lsort -unique -command community_cmp [array names community_rep]]
    foreach comm $all_comms {{
        set repID $community_rep($comm)
        set key "$json_ts,$comm"
        if {{[info exists community_data($key)] && [string length $community_data($key)] > 0}} {{
            mol modselect $repID $molID "{selection_type} $community_data($key)"
        }} else {{
            mol modselect $repID $molID "none"
        }}
    }}
}}
""")

# Community Annotation Legend function
tcl_lines.append(f"""
proc legend_comm {{json_ts}} {{
    global molID community_data community_colors
    draw delete all
    set all [atomselect $molID all]
    set system_center [measure center $all]
    $all delete
    foreach comm [array names community_colors] {{
        set key "$json_ts,$comm"
        if {{![info exists community_data($key)]}} {{ continue }}
        set resids $community_data($key)
        if {{[llength $resids] == 0}} {{ continue }}
        set sel [atomselect $molID "residue $resids"]
        if {{[$sel num] == 0}} {{ $sel delete ; continue }}
        set ccenter [measure center $sel]
        $sel delete
        set vx [expr {{[lindex $ccenter 0] - [lindex $system_center 0]}}]
        set vy [expr {{[lindex $ccenter 1] - [lindex $system_center 1]}}]
        set vz [expr {{[lindex $ccenter 2] - [lindex $system_center 2]}}]
        set norm [expr {{sqrt($vx*$vx + $vy*$vy + $vz*$vz)}}]
        if {{$norm < 1e-6}} {{ set norm 1.0 }}
        set offset_scale 0.2
        set offset [expr {{$norm * $offset_scale + 5.0}}]
        set vx [expr {{$vx / $norm * $offset}}]
        set vy [expr {{$vy / $norm * $offset}}]
        set vz [expr {{$vz / $norm * $offset}}]
        set shifted [list [expr {{[lindex $ccenter 0] + $vx}}] \\
                          [expr {{[lindex $ccenter 1] + $vy}}] \\
                          [expr {{[lindex $ccenter 2] + $vz}}]]
        set cid $community_colors($comm)
        draw color $cid
        draw text $shifted $comm
    }}
}}
""")

# Timestamp list and dynamic update
tcl_lines.append(f"""
set ts_list {{}}
foreach key [array names community_data] {{
    lappend ts_list [lindex [split $key ,] 0]
}}
set ts_list [lsort -integer -unique $ts_list]
set json_timestamps $ts_list

set nupdates {ntimestamps}

proc _update_for_frame {{args}} {{
    global molID json_timestamps nupdates
    set molid $molID
    set frame [molinfo $molid get frame]
    set nframes [molinfo $molid get numframes]
    if {{$nframes <= 0}} {{ return }}
    set chunk [expr {{int($frame * $nupdates / $nframes)}}]
    if {{$chunk < 0}} {{ set chunk 0 }}
    if {{$chunk >= $nupdates}} {{ set chunk [expr {{$nupdates - 1}}] }}
    set max_index [expr {{[llength $json_timestamps] - 1}}]
    if {{$max_index < 0}} {{ return }}
    set ts_index $chunk
    if {{$ts_index > $max_index}} {{ set ts_index $max_index }}
    set ts [lindex $json_timestamps $ts_index]
    display update off
    assign_communities_to_frame $ts
    legend_comm $ts
    display update on
}}

trace variable vmd_frame($molID) w _update_for_frame
_update_for_frame
""")

# Write to file
with open(output_tcl, "w") as f:
    f.write("\n".join(tcl_lines))

print(f"Wrote combined Tcl file: {output_tcl}")