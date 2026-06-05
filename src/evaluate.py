"""
src/evaluate.py
---------------
Evaluate a trained GNN+RL model against baseline strategies and
produce comparison plots.

Also includes a live_run() function that opens SUMO-GUI so you can
watch the trained agent control the intersections in real time.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless by default; set to "TkAgg" for interactive
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

from src.gnn_model import TrafficGNN, SimpleMLP
from src.sumo_env import SumoEnv, TL_IDS, build_edge_index, NODE_FEATURE_DIM


# ---------------------------------------------------------------------------
# Fixed-time baseline controller
# ---------------------------------------------------------------------------

class FixedTimeController:
    """Classic fixed-cycle controller: 30s green, 4s yellow, alternating."""

    def __init__(self, cycle_ns: int = 30, cycle_ew: int = 30):
        self.cycle_ns = cycle_ns
        self.cycle_ew = cycle_ew
        self._timers  = {tid: 0 for tid in TL_IDS}
        self._phase   = {tid: 0 for tid in TL_IDS}

    def act(self, obs, step: int) -> Dict[str, int]:
        actions = {}
        for tid in TL_IDS:
            self._timers[tid] += 1
            cycle = self.cycle_ns if self._phase[tid] == 0 else self.cycle_ew
            if self._timers[tid] >= cycle:
                self._timers[tid] = 0
                self._phase[tid]  = 1 - self._phase[tid]
                actions[tid] = 1
            else:
                actions[tid] = 0
        return actions


# ---------------------------------------------------------------------------
# Webster controller (per-intersection optimal cycle)
# ---------------------------------------------------------------------------

class WebsterController:
    """Webster's method: optimal cycle derived from observed queue ratios."""

    def __init__(self, lost_time: float = 4.0, update_every: int = 60):
        self.lost_time    = lost_time
        self.update_every = update_every
        self._timers = {tid: 0 for tid in TL_IDS}
        self._phase  = {tid: 0 for tid in TL_IDS}
        self._cycles = {tid: 60 for tid in TL_IDS}   # initial 60s
        self._step   = 0

    def _update_cycles(self, obs: np.ndarray):
        """Recompute Webster cycle from current queue observations."""
        # obs shape: (N, FEAT) — queues in indices 0-3
        for i, tid in enumerate(TL_IDS):
            q_ns = obs[i, 0] + obs[i, 1]  # normalised queues for N-S
            q_ew = obs[i, 2] + obs[i, 3]  # normalised queues for E-W
            s    = 1.0                     # normalised saturation flow
            y_ns = q_ns / s if s > 0 else 0.05
            y_ew = q_ew / s if s > 0 else 0.05
            Y    = min(y_ns + y_ew, 0.95)
            L    = self.lost_time * 2
            C    = max(30, min(120, (1.5 * L + 5) / max(1 - Y, 0.05)))
            self._cycles[tid] = C

    def act(self, obs: np.ndarray, step: int) -> Dict[str, int]:
        self._step += 1
        if self._step % self.update_every == 0:
            self._update_cycles(obs)

        actions = {}
        for i, tid in enumerate(TL_IDS):
            self._timers[tid] += 1
            half = self._cycles[tid] / 2
            if self._timers[tid] >= half:
                self._timers[tid] = 0
                self._phase[tid]  = 1 - self._phase[tid]
                actions[tid] = 1
            else:
                actions[tid] = 0
        return actions


# ---------------------------------------------------------------------------
# Evaluate one controller for N episodes
# ---------------------------------------------------------------------------

def run_evaluation(
    controller,
    cfg_path    : str,
    n_episodes  : int = 5,
    traci_port  : int = 8814,
    use_gui     : bool = False,
) -> Dict:
    env = SumoEnv(
        cfg_path   = cfg_path,
        use_gui    = use_gui,
        max_steps  = 3600,
        delta_time = 5,
        traci_port = traci_port,
    )

    all_rewards, all_waits, all_thru = [], [], []

    for ep in range(n_episodes):
        obs    = env.reset()
        done   = False
        step   = 0
        ep_r   = 0.0

        while not done:
            if hasattr(controller, "act"):
                actions = controller.act(obs, step)
            else:
                # Neural model
                edge_index = torch.tensor(build_edge_index(), dtype=torch.long)
                obs_t      = torch.FloatTensor(obs)
                with torch.no_grad():
                    acts, _, _ = controller.act(obs_t, edge_index, deterministic=True)
                actions = {tid: int(acts[i].item()) for i, tid in enumerate(TL_IDS)}

            obs, reward, done, info = env.step(actions)
            ep_r += reward
            step += 1

        all_rewards.append(ep_r)
        all_waits.append(np.mean(env.episode_waits) if env.episode_waits else 0)
        all_thru.append(env.episode_thru)
        print(f"  Episode {ep+1}: reward={ep_r:.2f}  wait={all_waits[-1]:.1f}s  thru={all_thru[-1]}")

    env.close()
    return {
        "rewards"   : all_rewards,
        "mean_wait" : all_waits,
        "throughput": all_thru,
        "avg_reward": np.mean(all_rewards),
        "avg_wait"  : np.mean(all_waits),
        "avg_thru"  : np.mean(all_thru),
    }


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def plot_comparison(results: Dict, save_path: str = "results/comparison.png"):
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#0f0f12")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    COLORS = {
        "GNN + RL":  "#7F77DD",
        "Simple RL": "#378ADD",
        "Webster":   "#EF9F27",
        "Fixed":     "#888888",
    }
    methods = list(results.keys())
    colors  = [COLORS.get(m, "#aaa") for m in methods]

    def bar(ax, key, ylabel, title, fmt=".1f"):
        vals   = [results[m][key] for m in methods]
        bars   = ax.bar(methods, vals, color=colors, edgecolor="none", width=0.55)
        ax.set_title(title, color="#e0ddd6", fontsize=10, pad=8)
        ax.set_ylabel(ylabel, color="#888", fontsize=9)
        ax.tick_params(colors="#888", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.set_facecolor("#1a1a20")
        ax.yaxis.label.set_color("#888")
        for bar_, val in zip(bars, vals):
            ax.text(bar_.get_x() + bar_.get_width()/2, bar_.get_height() + max(vals)*0.01,
                    f"{val:{fmt}}", ha="center", va="bottom", color="#e0ddd6", fontsize=8)

    # Bar charts
    ax1 = fig.add_subplot(gs[0, 0]); bar(ax1, "avg_wait",   "seconds", "Avg. wait time ↓")
    ax2 = fig.add_subplot(gs[0, 1]); bar(ax2, "avg_thru",   "vehicles","Throughput ↑", fmt=".0f")
    ax3 = fig.add_subplot(gs[0, 2]); bar(ax3, "avg_reward", "",        "Avg. episode reward ↑")

    # Training curve (if GNN model log exists)
    ax4 = fig.add_subplot(gs[1, :])
    log_path = Path("models/training_log.json")
    if log_path.exists():
        with open(log_path) as f:
            log = json.load(f)
        eps   = [e["episode"]   for e in log]
        rews  = [e["reward"]    for e in log]
        waits = [e["mean_wait"] for e in log]

        # Smoothing
        def smooth(x, w=10):
            return np.convolve(x, np.ones(w)/w, mode="valid")

        ax4b = ax4.twinx()
        ax4.plot(eps, rews,  color="#7F77DD", alpha=0.3, linewidth=0.8)
        ax4.plot(range(5, len(rews)+1), smooth(rews, 10),
                 color="#7F77DD", linewidth=2, label="GNN+RL reward")
        ax4b.plot(eps, waits, color="#EF9F27", alpha=0.3, linewidth=0.8)
        ax4b.plot(range(5, len(waits)+1), smooth(waits, 10),
                  color="#EF9F27", linewidth=2, label="Avg wait (s)")

        ax4.set_xlabel("Episode", color="#888", fontsize=9)
        ax4.set_ylabel("Reward",  color="#7F77DD", fontsize=9)
        ax4b.set_ylabel("Avg wait (s)", color="#EF9F27", fontsize=9)
        ax4.set_title("GNN + RL training curve", color="#e0ddd6", fontsize=10)
        ax4.set_facecolor("#1a1a20")
        ax4.tick_params(colors="#888", labelsize=8)
        ax4b.tick_params(colors="#888", labelsize=8)
        for spine in ax4.spines.values(): spine.set_color("#333")
        for spine in ax4b.spines.values(): spine.set_color("#333")
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4b.get_legend_handles_labels()
        ax4.legend(lines1+lines2, labels1+labels2, facecolor="#1a1a20",
                   edgecolor="#333", labelcolor="#e0ddd6", fontsize=8)
    else:
        ax4.text(0.5, 0.5, "No training log found.\nRun train.py first.",
                 ha="center", va="center", color="#888", transform=ax4.transAxes)
        ax4.set_facecolor("#1a1a20")
        for spine in ax4.spines.values(): spine.set_color("#333")

    fig.suptitle("Traffic Signal Control — Method Comparison  (3×3 SUMO Grid)",
                 color="#e0ddd6", fontsize=13, y=0.97)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f0f12")
    print(f"[Eval] Saved comparison plot → {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Live run with SUMO-GUI
# ---------------------------------------------------------------------------

def live_run(
    model_path  : str = "models/best_model.pt",
    cfg_path    : str = "sumo_network/grid.sumocfg",
    traci_port  : int = 8815,
):
    """
    Load a trained model and run it with SUMO-GUI open.
    You will see the trained GNN+RL agent controlling the lights.
    """
    print(f"\n[Live] Loading model from {model_path}")
    ckpt  = torch.load(model_path, map_location="cpu")
    model = TrafficGNN(node_feat_dim=NODE_FEATURE_DIM)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    edge_index = torch.tensor(build_edge_index(), dtype=torch.long)

    env  = SumoEnv(cfg_path=cfg_path, use_gui=True, traci_port=traci_port)
    obs  = env.reset()
    done = False
    step = 0
    total_reward = 0.0

    print("[Live] SUMO-GUI should open. Watch the GNN+RL agent control traffic.")
    print("       Close the SUMO-GUI window to stop.\n")

    try:
        while not done:
            obs_t = torch.FloatTensor(obs)
            with torch.no_grad():
                acts, _, _ = model.act(obs_t, edge_index, deterministic=True)
            actions = {tid: int(acts[i].item()) for i, tid in enumerate(TL_IDS)}
            obs, reward, done, info = env.step(actions)
            total_reward += reward
            step += 1
            if step % 50 == 0:
                print(f"  Step {step:>4} | Reward: {total_reward:>8.2f} | "
                      f"Vehicles: {info['n_vehicles']:>3} | "
                      f"Avg wait: {info['mean_wait']:.1f}s | "
                      f"Throughput: {info['throughput']}")
    except Exception as e:
        print(f"[Live] Stopped: {e}")
    finally:
        env.close()
    print(f"\n[Live] Finished. Total reward: {total_reward:.2f}")
