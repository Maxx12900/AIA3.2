#!/usr/bin/env python3
"""
setup_network.py
----------------
One-time setup: builds the SUMO network (.net.xml) from the
node and edge definition files, then generates the vehicle routes.

Run this once before training:
    python setup_network.py

Requirements
------------
- SUMO must be installed and SUMO_HOME must be set, OR
  netconvert/sumo must be on PATH.
  Install: https://sumo.dlr.de/docs/Installing/index.html
  Ubuntu:  sudo apt install sumo sumo-tools sumo-gui
  macOS:   brew install sumo
"""

import os
import sys
import subprocess
from pathlib import Path


def find_tool(name: str) -> str:
    """Find a SUMO tool binary."""
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        candidate = Path(sumo_home) / "bin" / name
        if candidate.exists():
            return str(candidate)
    for prefix in ["/usr/bin", "/usr/local/bin", "/opt/homebrew/bin"]:
        candidate = Path(prefix) / name
        if candidate.exists():
            return str(candidate)
    return name  # fall back to PATH


def run(cmd: list, desc: str):
    print(f"\n[Setup] {desc}")
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ✗ FAILED:\n{result.stderr}")
        sys.exit(1)
    print(f"  ✓ Done")
    if result.stdout.strip():
        print(f"  Output: {result.stdout.strip()[:200]}")


def main():
    net_dir = Path("sumo_network")
    net_dir.mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    net_file   = net_dir / "network.net.xml"
    node_file  = net_dir / "network.nod.xml"
    edge_file  = net_dir / "network.edg.xml"
    route_file = net_dir / "routes.rou.xml"

    # ---- Step 1: Build network ----
    netconvert = find_tool("netconvert")
    run([
        netconvert,
        "--node-files",        str(node_file),
        "--edge-files",        str(edge_file),
        "--output-file",       str(net_file),
        "--no-internal-links", "false",
        "--tls.default-type",  "static",
        "--tls.min-dur",       "10",
        "--tls.max-dur",       "60",
        "--junctions.corner-detail", "5",
        "--no-warnings",       "true",
    ], "Building SUMO network (netconvert)")

    # ---- Step 2: Generate routes ----
    run([
        sys.executable, "generate_routes.py",
        "--output", str(route_file),
        "--duration", "3600",
        "--flow", "350",
        "--seed", "42",
    ], "Generating vehicle routes")

    # ---- Verify ----
    for f in [net_file, route_file]:
        if f.exists():
            size = f.stat().st_size
            print(f"  ✓ {f}  ({size/1024:.1f} KB)")
        else:
            print(f"  ✗ Missing: {f}")
            sys.exit(1)

    print(f"""
{'='*60}
  Network setup complete!

  Files created:
    sumo_network/network.net.xml   — SUMO road network
    sumo_network/routes.rou.xml    — Vehicle demand
    sumo_network/grid.sumocfg      — Simulation config
    sumo_network/tls_programs.add.xml — TLS programs

  Next steps:

  1. Preview network in SUMO-GUI (optional):
       sumo-gui -c sumo_network/grid.sumocfg

  2. Train GNN + RL agent:
       python train.py --model gnn --episodes 200

  3. Train simple RL baseline for comparison:
       python train.py --model mlp --episodes 200 --port 8816 --save-dir models

  4. Evaluate and compare:
       python evaluate_all.py

  5. Watch trained agent live in SUMO-GUI:
       python evaluate_all.py --live
{'='*60}
""")


if __name__ == "__main__":
    main()
