#!/usr/bin/env python3
"""
train.py
--------
Main entry point for training the GNN + RL traffic signal controller.

Quick start
-----------
  # Build the SUMO network first (one-time setup)
  python setup_network.py

  # Train GNN + RL agent
  python train.py --model gnn --episodes 200

  # Train simple RL baseline for comparison
  python train.py --model mlp --episodes 200 --port 8814

  # Resume from checkpoint
  python train.py --model gnn --resume models/checkpoint_ep100.pt

  # Evaluate and compare all methods
  python evaluate_all.py
"""

import argparse
import sys
from pathlib import Path

# Make src importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.ppo_trainer import PPOTrainer


def parse_args():
    p = argparse.ArgumentParser(description="Train GNN+RL traffic signal controller")
    p.add_argument("--model",    default="gnn",  choices=["gnn","mlp"],
                   help="Model type: gnn (GNN+RL) or mlp (simple RL baseline)")
    p.add_argument("--cfg",      default="sumo_network/grid.sumocfg",
                   help="Path to SUMO .sumocfg file")
    p.add_argument("--episodes", type=int, default=200,
                   help="Number of training episodes")
    p.add_argument("--steps",    type=int, default=256,
                   help="Rollout steps per PPO update")
    p.add_argument("--lr",       type=float, default=3e-4,
                   help="Learning rate")
    p.add_argument("--gamma",    type=float, default=0.99,
                   help="Discount factor")
    p.add_argument("--clip",     type=float, default=0.2,
                   help="PPO clip epsilon")
    p.add_argument("--epochs",   type=int, default=10,
                   help="PPO epochs per update")
    p.add_argument("--port",     type=int, default=8813,
                   help="TraCI port (change if running multiple instances)")
    p.add_argument("--gui",      action="store_true",
                   help="Launch SUMO-GUI during training (visual but slower)")
    p.add_argument("--save-dir", default="models",
                   help="Directory to save model checkpoints")
    p.add_argument("--resume",   default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


def main():
    args = parse_args()

    trainer = PPOTrainer(
        model_type   = args.model,
        cfg_path     = args.cfg,
        use_gui      = args.gui,
        n_episodes   = args.episodes,
        n_steps      = args.steps,
        lr           = args.lr,
        gamma        = args.gamma,
        clip_eps     = args.clip,
        n_epochs     = args.epochs,
        traci_port   = args.port,
        save_dir     = args.save_dir,
    )

    if args.resume:
        import torch
        print(f"[Train] Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        trainer.model.load_state_dict(ckpt["model_state"])
        trainer.optimizer.load_state_dict(ckpt["optim_state"])
        trainer.train_log = ckpt.get("train_log", [])
        print(f"[Train] Loaded checkpoint. Continuing training...")

    log = trainer.train()
    print(f"\n[Train] Done. {len(log)} episodes logged.")


if __name__ == "__main__":
    main()
