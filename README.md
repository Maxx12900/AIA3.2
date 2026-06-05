# GNN + RL Traffic Signal Control with SUMO

A complete implementation of **Graph Neural Network + Reinforcement Learning**
for adaptive traffic signal control on a 3×3 intersection grid simulated in SUMO.

---

## Architecture Overview

```
SUMO Simulation (TraCI)
        │
        ▼
  SumoEnv (sumo_env.py)
  ┌─────────────────────────────────┐
  │  State: (9 nodes × 11 features) │  ← queue lengths, wait times,
  │    queue_N/S/E/W                │     phase, pressure
  │    wait_N/S/E/W                 │
  │    current_phase                │
  │    phase_duration               │
  │    pressure (|in - out|)        │
  └─────────────────────────────────┘
        │
        ▼
  TrafficGNN (gnn_model.py)
  ┌───────────────────────────────────────────────┐
  │                                               │
  │   Node features (9, 11)                       │
  │         │                                     │
  │   GAT Layer 1 (4 heads, hidden=64)            │
  │   ← message pass over road graph →            │
  │         │                                     │
  │   GAT Layer 2 (1 head, hidden=64)             │
  │         │                                     │
  │   Shared MLP + LayerNorm                      │
  │         │                                     │
  │   ┌─────┴──────┐                              │
  │ Actor        Critic                           │
  │ (9, 2)       (9, 1)                           │
  │ (keep/switch) (value)                         │
  └───────────────────────────────────────────────┘
        │
        ▼
  PPO Update (ppo_trainer.py)
  ← clipped surrogate loss + GAE advantages →
```

---

## Installation

### 1. Install SUMO

See https://sumo.dlr.de/docs/Downloads.php

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Step 1: Build the SUMO network (one-time)
python setup_network.py

# Step 2: Train the GNN + RL agent
python train.py --model gnn --episodes 200

# Step 3: (Optional) Train simple RL baseline for comparison
python train.py --model mlp --episodes 200 --port 8816

# Step 4: Evaluate and compare all methods
python evaluate_all.py

# Step 5: Watch the trained agent live in SUMO-GUI
python evaluate_all.py --live
```

---

## Project Structure

```
gnn_rl_traffic/
├── setup_network.py          # One-time SUMO network builder
├── train.py                  # Training entry point
├── evaluate_all.py           # Evaluation + comparison plots
├── generate_routes.py        # Vehicle demand generator
├── requirements.txt
│
├── sumo_network/
│   ├── network.nod.xml       # Intersection definitions
│   ├── network.edg.xml       # Road edge definitions
│   ├── network.net.xml       # ← built by setup_network.py
│   ├── routes.rou.xml        # ← built by setup_network.py
│   ├── tls_programs.add.xml  # Initial TLS programs
│   └── grid.sumocfg          # SUMO simulation config
│
├── src/
│   ├── sumo_env.py           # TraCI environment wrapper
│   ├── gnn_model.py          # GAT + actor-critic model
│   ├── ppo_trainer.py        # PPO training loop
│   └── evaluate.py           # Evaluation utilities
│
├── models/                   # Saved checkpoints (created during training)
│   ├── best_model.pt
│   ├── final_model.pt
│   └── training_log.json
│
└── results/                  # Plots and metrics
    ├── comparison.png
    ├── summary.xml
    └── tripinfo.xml
```

---

## Network Layout

```
  [top_A0]  [top_A1]  [top_A2]
      │          │          │
[left_A0]─── A0 ────── A1 ────── A2 ───[right_A2]
              │          │          │
[left_B0]─── B0 ────── B1 ────── B2 ───[right_B2]
              │          │          │
[left_C0]─── C0 ────── C1 ────── C2 ───[right_C2]
      │          │          │
  [bot_C0]  [bot_C1]  [bot_C2]
```

9 signalised intersections (A0–C2), each with 2 lanes per approach.
Entry/exit fringes inject and absorb vehicles.

---

## Training Arguments

```
python train.py --help

  --model     gnn|mlp        Model type (default: gnn)
  --cfg       PATH           Path to .sumocfg (default: sumo_network/grid.sumocfg)
  --episodes  INT            Training episodes (default: 200)
  --steps     INT            Rollout steps before PPO update (default: 256)
  --lr        FLOAT          Learning rate (default: 3e-4)
  --gamma     FLOAT          Discount factor (default: 0.99)
  --clip      FLOAT          PPO clip epsilon (default: 0.2)
  --epochs    INT            PPO update epochs (default: 10)
  --port      INT            TraCI port (default: 8813)
  --gui                      Open SUMO-GUI during training
  --save-dir  DIR            Checkpoint directory (default: models)
  --resume    PATH           Resume from checkpoint
```

---

## How the GNN Works

**Graph construction:**
- Each intersection is a node
- Road segments between intersections are edges
- Node features: local traffic state (queues, waits, phase, pressure)

**Graph Attention (GAT):**
```
For each intersection v:
  α_{vu} = softmax( LeakyReLU( a^T [Wh_v ‖ Wh_u] ) )
  h_v'   = ELU( Σ_{u ∈ N(v)} α_{vu} · W · h_u )
```
Each intersection learns to **attend** to its neighbours weighted by their
congestion level — a heavily loaded upstream neighbour gets higher attention,
allowing the agent to anticipate arriving platoons.

**Why GNN beats simple RL:**
- Simple RL: each intersection only sees its own local state → reactive
- GNN + RL: each intersection sees its neighbourhood → proactive
- Single model scales to any network size
- Transferable across different city layouts

---

## Reward Function

```
r_t = - (mean_waiting_time / 300)  +  0.1 × arrived_vehicles
```

- Penalises average waiting time across all vehicles
- Rewards vehicles completing their trips
- Normalised to keep values in [-1, 1] for stable training

---

## Comparison Methods

| Method       | Description                                    |
|--------------|------------------------------------------------|
| Fixed-time   | 30s green per phase, fixed cycle               |
| Webster      | Optimal fixed cycle from queue ratios          |
| Simple RL    | PPO with MLP policy, no graph communication    |
| **GNN + RL** | PPO with GAT encoder, full graph awareness     |

---

## Visualising in SUMO-GUI

When you run with `--gui` or `python evaluate_all.py --live`:
- Each intersection shows its current phase colour
- Vehicles are colour-coded by type (car=blue, truck=orange, bus=green)
- Queue lengths visible as waiting vehicles
- The agent switches phases in real time based on the GNN policy

You can also open the network without simulation:
```bash
sumo-gui sumo_network/network.net.xml
```

---

## Extending the Project

**Add more intersections:** Edit `network.nod.xml` and `network.edg.xml`,
then re-run `setup_network.py`. The GNN automatically adapts to any graph size.

**Different reward signals:**
- Pressure-based: `reward = -Σ |queue_in - queue_out|`
- Queue-based: `reward = -Σ queue_length`
- Delay-based: `reward = -Σ vehicle_delay`

**Multi-agent RL:** Make each intersection its own PPO agent. The GNN
embedding still provides shared context (centralised training / decentralised execution).

**Real-world maps:** Use `osmWebWizard.py` (included in SUMO tools) to
download and convert an OpenStreetMap area into a SUMO network.
