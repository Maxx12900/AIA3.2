#!/usr/bin/env python3
"""
evaluate_all.py
---------------
Compare GNN+RL, Simple RL, Webster, and Fixed-time controllers
on the same 3x3 SUMO network and generate a comparison chart.

Usage
-----
  # After training both models:
  python evaluate_all.py

  # Watch the best model live in SUMO-GUI:
  python evaluate_all.py --live

  # Evaluate only (skip model loading if no checkpoint):
  python evaluate_all.py --baselines-only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.gnn_model import TrafficGNN, SimpleMLP
from src.sumo_env import NODE_FEATURE_DIM
from src.evaluate import (
    FixedTimeController, WebsterController,
    run_evaluation, plot_comparison, live_run
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg",            default="sumo_network/grid.sumocfg")
    p.add_argument("--n-eval",         type=int, default=3,
                   help="Episodes per method for evaluation")
    p.add_argument("--gnn-model",      default="models/best_model.pt")
    p.add_argument("--mlp-model",      default="models/best_mlp_model.pt")
    p.add_argument("--live",           action="store_true",
                   help="After evaluation, run GNN model in SUMO-GUI")
    p.add_argument("--baselines-only", action="store_true",
                   help="Only evaluate Fixed and Webster (no neural models)")
    p.add_argument("--port-base",      type=int, default=8814,
                   help="Base TraCI port; each method uses port+offset")
    p.add_argument("--output",         default="results/comparison.png")
    return p.parse_args()


def load_model(path: str, model_type: str):
    """Load a trained model from checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if model_type == "gnn":
        model = TrafficGNN(node_feat_dim=NODE_FEATURE_DIM)
    else:
        model = SimpleMLP(node_feat_dim=NODE_FEATURE_DIM)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main():
    args   = parse_args()
    results = {}

    # ---- Fixed-time baseline ----
    print("\n=== Evaluating: Fixed-time controller ===")
    results["Fixed"] = run_evaluation(
        FixedTimeController(), args.cfg,
        n_episodes=args.n_eval, traci_port=args.port_base
    )

    # ---- Webster baseline ----
    print("\n=== Evaluating: Webster controller ===")
    results["Webster"] = run_evaluation(
        WebsterController(), args.cfg,
        n_episodes=args.n_eval, traci_port=args.port_base + 1
    )

    if not args.baselines_only:
        # ---- Simple RL (MLP) ----
        mlp_path = Path(args.mlp_model)
        if mlp_path.exists():
            print(f"\n=== Evaluating: Simple RL (MLP) — {mlp_path} ===")
            mlp_model = load_model(str(mlp_path), "mlp")
            results["Simple RL"] = run_evaluation(
                mlp_model, args.cfg,
                n_episodes=args.n_eval, traci_port=args.port_base + 2
            )
        else:
            print(f"\n[Skip] MLP model not found at {mlp_path}. Train with:")
            print(f"       python train.py --model mlp --save-dir models --port 8816")

        # ---- GNN + RL ----
        gnn_path = Path(args.gnn_model)
        if gnn_path.exists():
            print(f"\n=== Evaluating: GNN + RL — {gnn_path} ===")
            gnn_model = load_model(str(gnn_path), "gnn")
            results["GNN + RL"] = run_evaluation(
                gnn_model, args.cfg,
                n_episodes=args.n_eval, traci_port=args.port_base + 3
            )
        else:
            print(f"\n[Skip] GNN model not found at {args.gnn_model}. Train with:")
            print(f"       python train.py --model gnn")

    # ---- Print summary table ----
    print("\n" + "="*65)
    print(f"{'Method':<15} {'Avg Reward':>12} {'Avg Wait (s)':>13} {'Throughput':>12}")
    print("-"*65)
    for method, res in results.items():
        print(f"{method:<15} {res['avg_reward']:>12.3f} "
              f"{res['avg_wait']:>13.1f} {res['avg_thru']:>12.0f}")
    print("="*65)

    # ---- Plot ----
    if len(results) >= 1:
        plot_comparison(results, save_path=args.output)

    # ---- Live demo ----
    if args.live:
        gnn_path = Path(args.gnn_model)
        if gnn_path.exists():
            live_run(model_path=str(gnn_path), cfg_path=args.cfg, traci_port=8818)
        else:
            print(f"[Live] No model found at {gnn_path}. Train first.")


if __name__ == "__main__":
    main()
