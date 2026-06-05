#!/usr/bin/env python3
"""
generate_routes.py
Generates a SUMO routes file (.rou.xml) using <flow> elements with
randomly selected from/to fringe edges so vehicles spawn at random
entry points and travel to random exit points across the grid.

SUMO's built-in dijkstra router computes the full path at runtime.

Usage:
    python generate_routes.py --output sumo_network/routes.rou.xml
    python generate_routes.py --n-flows 60 --flow 400 --seed 99
"""
import argparse
import random
import xml.etree.ElementTree as ET
from xml.dom import minidom

# ---------------------------------------------------------------------------
# All 12 fringe entry edges grouped by which side they are on.
# Vehicles enter FROM these edges into the grid.
# ---------------------------------------------------------------------------
ENTRIES = {
    "top":   ["in_top_A0",   "in_top_A1",   "in_top_A2"],
    "bot":   ["in_bot_C0",   "in_bot_C1",   "in_bot_C2"],
    "left":  ["in_left_A0",  "in_left_B0",  "in_left_C0"],
    "right": ["in_right_A2", "in_right_B2", "in_right_C2"],
}

# All 12 fringe exit edges grouped by side.
# Vehicles exit TO these edges out of the grid.
EXITS = {
    "top":   ["out_top_A0",   "out_top_A1",   "out_top_A2"],
    "bot":   ["out_bot_C0",   "out_bot_C1",   "out_bot_C2"],
    "left":  ["out_left_A0",  "out_left_B0",  "out_left_C0"],
    "right": ["out_right_A2", "out_right_B2", "out_right_C2"],
}

ALL_ENTRIES = [e for edges in ENTRIES.values() for e in edges]
ALL_EXITS   = [e for edges in EXITS.values()   for e in edges]

# Map each entry edge to which side it is on (used to avoid same-side pairs)
ENTRY_SIDE = {e: side for side, edges in ENTRIES.items() for e in edges}
EXIT_SIDE  = {e: side for side, edges in EXITS.items()   for e in edges}


def random_od_pair(rng: random.Random, allow_same_side: bool = False):
    """
    Pick a random (entry, exit) pair.
    By default rejects pairs on the same side so vehicles always
    travel meaningfully through the grid.
    """
    for _ in range(100):   # retry loop to avoid same-side
        src = rng.choice(ALL_ENTRIES)
        dst = rng.choice(ALL_EXITS)
        if allow_same_side or ENTRY_SIDE[src] != EXIT_SIDE[dst]:
            return src, dst
    # Fallback: opposite side guaranteed
    src = rng.choice(ALL_ENTRIES)
    opposite = {"top": "bot", "bot": "top", "left": "right", "right": "left"}
    dst = rng.choice(EXITS[opposite[ENTRY_SIDE[src]]])
    return src, dst


def generate_routes(
    output_path     : str,
    sim_duration    : int   = 3600,
    base_flow       : int   = 300,   # vehicles/hour per flow definition
    n_flows         : int   = 50,    # number of random OD pairs to generate
    seed            : int   = 42,
    rush_hours      : bool  = True,
    allow_same_side : bool  = False,
):
    rng = random.Random(seed)
    root = ET.Element("routes")

    # ---- Vehicle type definitions ----
    vtypes = [
        # id,     accel, decel, sigma, length, maxSpeed, color
        ("car",   2.6,   4.5,   0.5,   5.0,   13.89,  "0.3,0.6,1.0"),
        ("truck", 1.5,   3.5,   0.4,   8.0,   10.00,  "0.8,0.5,0.2"),
        ("bus",   1.2,   3.0,   0.3,   12.0,   9.00,  "0.2,0.8,0.4"),
    ]
    for vt in vtypes:
        ET.SubElement(root, "vType", {
            "id":       vt[0],
            "accel":    str(vt[1]),
            "decel":    str(vt[2]),
            "sigma":    str(vt[3]),
            "length":   str(vt[4]),
            "maxSpeed": str(vt[5]),
            "color":    vt[6],
        })

    # ---- Generate random OD pairs and emit flow elements ----
    flow_id = 0
    pairs_used = []

    for _ in range(n_flows):
        src, dst = random_od_pair(rng, allow_same_side)
        pairs_used.append((src, dst))

        f = base_flow + rng.randint(-50, 50)

        if rush_hours:
            periods = [
                (0,    900,          int(f * 2.0), "car"),
                (0,    900,          int(f * 0.3), "truck"),
                (900,  2700,         int(f * 0.7), "car"),
                (900,  2700,         int(f * 0.1), "truck"),
                (2700, sim_duration, int(f * 1.8), "car"),
                (2700, sim_duration, int(f * 0.2), "truck"),
                (0,    sim_duration, max(1, int(f * 0.05)), "bus"),
            ]
        else:
            periods = [
                (0, sim_duration, f,               "car"),
                (0, sim_duration, max(1, f // 8),  "truck"),
                (0, sim_duration, max(1, f // 15), "bus"),
            ]

        for t_begin, t_end, vph, vtype in periods:
            if vph < 1 or t_end <= t_begin:
                continue
            ET.SubElement(root, "flow", {
                "id":          f"flow_{flow_id}",
                "type":        vtype,
                "from":        src,
                "to":          dst,
                "begin":       str(t_begin),
                "end":         str(t_end),
                "vehsPerHour": str(vph),
                "departLane":  "best",
                "departSpeed": "speedLimit",
            })
            flow_id += 1

    # ---- Write output ----
    rough    = ET.tostring(root, "unicode")
    reparsed = minidom.parseString(rough)
    lines    = reparsed.toprettyxml(indent="    ").splitlines()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write("\n".join(lines[1:]))

    # Print summary of which OD pairs were generated
    print(f"Generated {flow_id} flow definitions from {n_flows} random OD pairs -> {output_path}")
    print("\nOD pairs sampled:")
    for i, (src, dst) in enumerate(pairs_used):
        print(f"  {i+1:>3}. {src:<18} -> {dst}")

    return flow_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate random origin-destination traffic flows for SUMO"
    )
    parser.add_argument("--output",          default="sumo_network/routes.rou.xml")
    parser.add_argument("--duration",        type=int,  default=3600,
                        help="Simulation duration in seconds")
    parser.add_argument("--flow",            type=int,  default=300,
                        help="Base vehicles/hour per OD pair")
    parser.add_argument("--n-flows",         type=int,  default=50,
                        help="Number of random OD pairs to generate")
    parser.add_argument("--seed",            type=int,  default=42,
                        help="Random seed (change for different spawn patterns)")
    parser.add_argument("--no-rush",         action="store_true",
                        help="Disable rush-hour multipliers (flat uniform flow)")
    parser.add_argument("--allow-same-side", action="store_true",
                        help="Allow entry and exit on the same side of the grid")
    args = parser.parse_args()

    generate_routes(
        output_path     = args.output,
        sim_duration    = args.duration,
        base_flow       = args.flow,
        n_flows         = args.n_flows,
        seed            = args.seed,
        rush_hours      = not args.no_rush,
        allow_same_side = args.allow_same_side,
    )
