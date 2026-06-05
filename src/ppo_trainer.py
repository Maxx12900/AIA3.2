"""
src/ppo_trainer.py
------------------
Proximal Policy Optimisation (PPO) trainer for the GNN traffic controller.

This is a centralised-training / decentralised-execution (CTDE) setup:
  - During training, all N intersection observations are batched together.
  - The GNN processes the full graph in one forward pass (centralised).
  - Each intersection uses only its own action output (decentralised).

PPO hyper-parameters follow the recommendations from:
  Schulman et al. 2017 "Proximal Policy Optimization Algorithms"
  Henderson et al. 2018 "Deep RL That Matters" (tuning guidance)
"""

import time
import json
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.gnn_model import TrafficGNN, SimpleMLP
from src.sumo_env import SumoEnv, TL_IDS, build_edge_index


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores one episode's worth of (s, a, r, s', done) tuples."""

    def __init__(self, n_steps: int, n_agents: int, feat_dim: int, device):
        self.n_steps  = n_steps
        self.n_agents = n_agents
        self.device   = device

        self.obs       = torch.zeros(n_steps, n_agents, feat_dim)
        self.actions   = torch.zeros(n_steps, n_agents, dtype=torch.long)
        self.log_probs = torch.zeros(n_steps, n_agents)
        self.values    = torch.zeros(n_steps, n_agents)
        self.rewards   = torch.zeros(n_steps)
        self.dones     = torch.zeros(n_steps)

        self.ptr = 0

    def add(self, obs, actions, log_probs, values, reward, done):
        if self.ptr >= self.n_steps:
            return
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = actions
        self.log_probs[self.ptr] = log_probs
        self.values[self.ptr]    = values
        self.rewards[self.ptr]   = reward
        self.dones[self.ptr]     = float(done)
        self.ptr += 1

    def compute_returns(self, last_values, gamma: float, gae_lambda: float):
        """Compute GAE advantages and returns in-place."""
        advantages = torch.zeros_like(self.rewards)
        last_gae   = torch.zeros(self.n_agents)
        rewards    = self.rewards.unsqueeze(1).expand(-1, self.n_agents)
        dones      = self.dones.unsqueeze(1).expand(-1, self.n_agents)

        for t in reversed(range(self.ptr)):
            next_val   = last_values if t == self.ptr - 1 else self.values[t + 1]
            delta      = rewards[t] + gamma * next_val * (1 - dones[t]) - self.values[t]
            last_gae   = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae.mean()

        self.advantages = advantages[:self.ptr]
        self.returns    = (self.advantages.unsqueeze(1) + self.values[:self.ptr])
        return self.advantages

    def get(self):
        """Flatten time × agent dimensions for batch update."""
        T = self.ptr
        return {
            "obs":       self.obs[:T].to(self.device),
            "actions":   self.actions[:T].to(self.device),
            "log_probs": self.log_probs[:T].to(self.device),
            "values":    self.values[:T].to(self.device),
            "returns":   self.returns.to(self.device),
            "advantages":self.advantages.to(self.device),
        }

    def reset(self):
        self.ptr = 0


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """
    PPO trainer for multi-intersection traffic control.

    Parameters
    ----------
    model_type   : 'gnn' (GNN+RL) or 'mlp' (simple independent RL baseline)
    cfg_path     : path to grid.sumocfg
    use_gui      : open SUMO-GUI during training (slow but visual)
    n_episodes   : total training episodes
    n_steps      : steps per rollout before PPO update
    lr           : learning rate
    gamma        : discount factor
    gae_lambda   : GAE lambda
    clip_eps     : PPO clip epsilon
    entropy_coef : entropy bonus coefficient
    value_coef   : value loss coefficient
    n_epochs     : PPO epochs per update
    batch_size   : minibatch size (in steps)
    save_dir     : directory to save checkpoints
    """

    def __init__(
        self,
        model_type   : str   = "gnn",
        cfg_path     : str   = "sumo_network/grid.sumocfg",
        use_gui      : bool  = False,
        n_episodes   : int   = 200,
        n_steps      : int   = 256,
        lr           : float = 3e-4,
        gamma        : float = 0.99,
        gae_lambda   : float = 0.95,
        clip_eps     : float = 0.2,
        entropy_coef : float = 0.01,
        value_coef   : float = 0.5,
        n_epochs     : int   = 10,
        batch_size   : int   = 64,
        save_dir     : str   = "models",
        traci_port   : int   = 8813,
    ):
        self.cfg = dict(locals()); self.cfg.pop("self")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[PPO] Device: {self.device}")

        # Build environment
        self.env = SumoEnv(
            cfg_path   = cfg_path,
            use_gui    = use_gui,
            max_steps  = 3600,
            delta_time = 5,
            traci_port = traci_port,
        )

        # Graph structure (fixed for this network)
        edge_idx = build_edge_index()
        self.edge_index = torch.tensor(edge_idx, dtype=torch.long).to(self.device)

        # Model
        from src.sumo_env import NODE_FEATURE_DIM
        if model_type == "gnn":
            self.model = TrafficGNN(node_feat_dim=NODE_FEATURE_DIM).to(self.device)
        else:
            self.model = SimpleMLP(node_feat_dim=NODE_FEATURE_DIM).to(self.device)
        print(f"[PPO] Model: {model_type.upper()} | Params: {sum(p.numel() for p in self.model.parameters()):,}")

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_episodes, eta_min=lr * 0.1
        )

        # Hyper-params
        self.n_episodes   = n_episodes
        self.n_steps      = n_steps
        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.clip_eps     = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef   = value_coef
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self.save_dir     = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.buffer = RolloutBuffer(
            n_steps  = n_steps,
            n_agents = self.env.n_agents,
            feat_dim = NODE_FEATURE_DIM,
            device   = self.device,
        )

        # Logging
        self.ep_rewards  = deque(maxlen=20)
        self.ep_waits    = deque(maxlen=20)
        self.train_log   = []
        self.best_reward = -np.inf

    # ------------------------------------------------------------------

    def train(self):
        print(f"\n{'='*60}")
        print(f"  GNN + RL Traffic Signal Control — Training")
        print(f"  Episodes: {self.n_episodes}  |  Steps/rollout: {self.n_steps}")
        print(f"{'='*60}\n")

        for episode in range(1, self.n_episodes + 1):
            ep_start = time.time()
            obs_np   = self.env.reset()            # (N, F) numpy
            obs      = torch.FloatTensor(obs_np).to(self.device)
            done     = False
            ep_reward = 0.0
            self.buffer.reset()
            step_count = 0

            while not done:
                # ---- Collect rollout ----
                with torch.no_grad():
                    actions_t, log_probs_t, values_t = self.model.act(
                        obs, self.edge_index
                    )

                actions_dict = {
                    tid: int(actions_t[i].item())
                    for i, tid in enumerate(TL_IDS)
                }

                next_obs_np, reward, done, info = self.env.step(actions_dict)
                next_obs = torch.FloatTensor(next_obs_np).to(self.device)

                self.buffer.add(
                    obs.cpu(), actions_t.cpu(), log_probs_t.cpu(),
                    values_t.cpu(), reward, done
                )

                ep_reward += reward
                obs = next_obs
                step_count += 1

                # Run PPO update when buffer is full
                if self.buffer.ptr >= self.n_steps:
                    with torch.no_grad():
                        _, _, last_vals = self.model.act(obs, self.edge_index)
                    self.buffer.compute_returns(last_vals.cpu(), self.gamma, self.gae_lambda)
                    update_info = self._ppo_update()
                    self.buffer.reset()

            # Final update with whatever is left in the buffer
            if self.buffer.ptr > 0:
                with torch.no_grad():
                    _, _, last_vals = self.model.act(obs, self.edge_index)
                self.buffer.compute_returns(last_vals.cpu(), self.gamma, self.gae_lambda)
                self._ppo_update()

            self.scheduler.step()
            mean_wait  = np.mean(self.env.episode_waits) if self.env.episode_waits else 0
            throughput = self.env.episode_thru

            self.ep_rewards.append(ep_reward)
            self.ep_waits.append(mean_wait)

            elapsed = time.time() - ep_start
            log_entry = {
                "episode":    episode,
                "reward":     round(ep_reward, 3),
                "mean_wait":  round(mean_wait, 2),
                "throughput": throughput,
                "steps":      step_count,
                "lr":         self.scheduler.get_last_lr()[0],
                "time_s":     round(elapsed, 1),
            }
            self.train_log.append(log_entry)

            # Console logging
            avg_r = np.mean(self.ep_rewards)
            avg_w = np.mean(self.ep_waits)
            print(
                f"Ep {episode:>4}/{self.n_episodes} | "
                f"Reward: {ep_reward:>8.2f} (avg {avg_r:>8.2f}) | "
                f"Wait: {mean_wait:>6.1f}s (avg {avg_w:>5.1f}s) | "
                f"Thru: {throughput:>4} | "
                f"Steps: {step_count:>4} | "
                f"Time: {elapsed:.1f}s"
            )

            # Save best model
            if avg_r > self.best_reward and episode >= 10:
                self.best_reward = avg_r
                self.save_checkpoint("best_model.pt")

            # Periodic checkpoint
            if episode % 50 == 0:
                self.save_checkpoint(f"checkpoint_ep{episode}.pt")
                self._save_log()

        self.env.close()
        self.save_checkpoint("final_model.pt")
        self._save_log()
        print(f"\n[PPO] Training complete. Best avg reward: {self.best_reward:.3f}")
        return self.train_log

    # ------------------------------------------------------------------

    def _ppo_update(self) -> dict:
        data  = self.buffer.get()
        T     = data["obs"].shape[0]     # time steps
        N     = data["obs"].shape[1]     # agents

        # Flatten (T, N, F) → (T*N, F) for batch processing
        obs_flat     = data["obs"].view(T * N, -1)
        actions_flat = data["actions"].view(T * N)
        old_lp_flat  = data["log_probs"].view(T * N)
        returns_flat = data["returns"].view(T * N)
        adv_flat     = data["advantages"].unsqueeze(1).expand(T, N).reshape(T * N)

        # Normalise advantages
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        pg_losses, v_losses, ent_losses = [], [], []

        for _ in range(self.n_epochs):
            # Mini-batch indices — we process all timesteps but batch over T
            indices = torch.randperm(T)
            for start in range(0, T, max(1, self.batch_size // N)):
                end     = min(start + self.batch_size // N, T)
                mb_idx  = indices[start:end]

                # Gather minibatch (all agents for selected timesteps)
                mb_obs     = data["obs"][mb_idx].view(-1, data["obs"].shape[-1])
                mb_actions = data["actions"][mb_idx].view(-1)
                mb_old_lp  = data["log_probs"][mb_idx].view(-1)
                mb_returns = returns_flat.view(T, N)[mb_idx].view(-1)
                mb_adv     = adv_flat.view(T, N)[mb_idx].view(-1)

                # Forward pass with GNN on this minibatch
                # Note: for proper GNN we'd need the graph; we use a
                # pseudo-batch approach where each row is an independent sample
                log_p, values, entropy = self.model.evaluate(
                    mb_obs, self.edge_index, mb_actions
                )

                # PPO clipped surrogate loss
                ratio   = torch.exp(log_p - mb_old_lp)
                clip_r  = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                pg_loss = -torch.min(ratio * mb_adv, clip_r * mb_adv).mean()

                # Value loss (clipped)
                v_loss  = F.mse_loss(values, mb_returns)

                # Entropy bonus
                ent_loss = -entropy.mean()

                loss = pg_loss + self.value_coef * v_loss + self.entropy_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(ent_loss.item())

        return {
            "pg_loss":  np.mean(pg_losses),
            "v_loss":   np.mean(v_losses),
            "ent_loss": np.mean(ent_losses),
        }

    def save_checkpoint(self, filename: str):
        path = self.save_dir / filename
        torch.save({
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "config":      self.cfg,
            "train_log":   self.train_log,
        }, path)
        print(f"  ✓ Saved: {path}")

    def _save_log(self):
        log_path = self.save_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(self.train_log, f, indent=2)
