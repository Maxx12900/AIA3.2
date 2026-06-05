"""
src/gnn_model.py
----------------
Graph Attention Network (GAT) encoder + PPO actor-critic policy heads
for multi-intersection traffic signal control.

Architecture:
  Input node features  (N, F)
      │
  GAT Layer 1          (N, H*heads)   ← neighbourhood aggregation
      │
  GAT Layer 2          (N, H)         ← refined embeddings
      │
  ┌───┴───┐
  │       │
Policy   Value           ← per-node policy & value heads
(N, A)   (N, 1)

Where:
  N = number of intersections
  F = NODE_FEATURE_DIM  (11)
  H = hidden dimension  (64)
  A = action space      (2: keep / switch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Graph Attention Layer (single-head or multi-head)
# ---------------------------------------------------------------------------

class GATLayer(nn.Module):
    """
    One graph attention layer.

    For each node v:
        h_v' = σ( Σ_{u ∈ N(v)∪{v}}  α_{vu} · W · h_u )

    Attention coefficients:
        e_{vu}   = LeakyReLU( a^T · [W·h_v ‖ W·h_u] )
        α_{vu}   = softmax_u( e_{vu} )
    """

    def __init__(self, in_features: int, out_features: int, n_heads: int = 4,
                 dropout: float = 0.1, concat: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.n_heads      = n_heads
        self.concat       = concat
        self.dropout      = dropout

        self.W  = nn.Linear(in_features, out_features * n_heads, bias=False)
        self.a  = nn.Parameter(torch.Tensor(n_heads, 2 * out_features))
        self.leaky_relu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x           : (N, in_features)
        edge_index  : (2, E)  — source, target node indices

        Returns
        -------
        out : (N, out_features * n_heads) if concat else (N, out_features)
        """
        N = x.size(0)
        # Linear transform: (N, H*heads)
        Wx = self.W(x).view(N, self.n_heads, self.out_features)  # (N, heads, H)

        src, dst = edge_index  # each (E,)

        # Include self-loops
        self_src = torch.arange(N, device=x.device)
        src = torch.cat([src, self_src])
        dst = torch.cat([dst, self_src])

        # Compute attention scores
        feat_src = Wx[src]  # (E+N, heads, H)
        feat_dst = Wx[dst]  # (E+N, heads, H)
        cat      = torch.cat([feat_src, feat_dst], dim=-1)  # (E+N, heads, 2H)
        e        = self.leaky_relu((cat * self.a).sum(dim=-1))  # (E+N, heads)

        # Softmax per target node per head
        alpha = torch.zeros(N, self.n_heads, device=x.device)
        # exp(e) accumulated at dst
        exp_e = torch.exp(e - e.max(dim=0, keepdim=True).values)
        alpha_num = torch.zeros(N, self.n_heads, device=x.device)
        alpha_num.scatter_add_(0, dst.unsqueeze(1).expand_as(exp_e), exp_e)
        # normalise
        denom = alpha_num[dst] + 1e-9
        alpha_e = exp_e / denom  # (E+N, heads)

        # Dropout on attention weights
        alpha_e = F.dropout(alpha_e, p=self.dropout, training=self.training)

        # Weighted aggregation
        out = torch.zeros(N, self.n_heads, self.out_features, device=x.device)
        weighted = alpha_e.unsqueeze(-1) * feat_src  # (E+N, heads, H)
        out.scatter_add_(
            0,
            dst.unsqueeze(1).unsqueeze(2).expand_as(weighted),
            weighted
        )

        if self.concat:
            return F.elu(out.view(N, self.n_heads * self.out_features))
        else:
            return F.elu(out.mean(dim=1))


# ---------------------------------------------------------------------------
# Full GNN + Policy/Value heads
# ---------------------------------------------------------------------------

class TrafficGNN(nn.Module):
    """
    GNN actor-critic for multi-intersection traffic signal control.

    Inputs
    ------
    x           : (N, NODE_FEATURE_DIM)  — node features
    edge_index  : (2, E)                 — graph connectivity

    Outputs
    -------
    logits : (N, 2)   — action logits (keep / switch)
    values : (N, 1)   — state values
    attns  : list     — attention weights per layer (for visualisation)
    """

    def __init__(
        self,
        node_feat_dim : int   = 11,
        hidden_dim    : int   = 64,
        n_actions     : int   = 2,
        n_heads_1     : int   = 4,
        n_heads_2     : int   = 1,
        dropout       : float = 0.1,
    ):
        super().__init__()

        self.gat1 = GATLayer(
            node_feat_dim, hidden_dim,
            n_heads=n_heads_1, dropout=dropout, concat=True
        )
        self.gat2 = GATLayer(
            hidden_dim * n_heads_1, hidden_dim,
            n_heads=n_heads_2, dropout=dropout, concat=False
        )

        # Shared MLP after GNN
        self.shared_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ELU(),
            nn.Linear(32, n_actions),
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ELU(),
            nn.Linear(32, 1),
        )

        # Weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x          : Tensor,
        edge_index : Tensor,
    ):
        """
        Returns logits (N,2), values (N,1).
        """
        h = self.gat1(x, edge_index)
        h = self.gat2(h, edge_index)
        h = self.shared_mlp(h)

        logits = self.actor(h)   # (N, 2)
        values = self.critic(h)  # (N, 1)
        return logits, values

    def act(
        self,
        x          : Tensor,
        edge_index : Tensor,
        deterministic: bool = False,
    ):
        """
        Sample actions for all N intersections.

        Returns
        -------
        actions    : (N,)  int tensor
        log_probs  : (N,)  float tensor
        values     : (N,)  float tensor
        """
        logits, values = self.forward(x, edge_index)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, values.squeeze(-1)

    def evaluate(
        self,
        x          : Tensor,
        edge_index : Tensor,
        actions    : Tensor,
    ):
        """
        Evaluate log-probs and values for given (state, action) pairs.
        Used during PPO update.
        """
        logits, values = self.forward(x, edge_index)
        dist     = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        return log_prob, values.squeeze(-1), entropy


# ---------------------------------------------------------------------------
# Simple baseline: independent MLP agent (no GNN, no graph)
# ---------------------------------------------------------------------------

class SimpleMLP(nn.Module):
    """
    Baseline actor-critic without graph message passing.
    Each intersection is treated independently.
    Input: flattened local features (NODE_FEATURE_DIM,)
    """

    def __init__(self, node_feat_dim: int = 11, n_actions: int = 2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(node_feat_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),            nn.ReLU(),
        )
        self.actor  = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_actions))
        self.critic = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: Tensor, edge_index=None):
        h      = self.shared(x)
        logits = self.actor(h)
        values = self.critic(h)
        return logits, values

    def act(self, x, edge_index=None, deterministic=False):
        logits, values = self.forward(x)
        dist      = torch.distributions.Categorical(logits=logits)
        actions   = logits.argmax(-1) if deterministic else dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, values.squeeze(-1)

    def evaluate(self, x, edge_index, actions):
        logits, values = self.forward(x)
        dist     = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        return log_prob, values.squeeze(-1), entropy
