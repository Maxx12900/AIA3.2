"""
src/sumo_env.py
---------------
SUMO TraCI environment wrapper for the 3x3 grid network.
Provides a Gymnasium-compatible interface that the GNN+RL agent uses.

State per intersection (node features for GNN):
  [queue_N, queue_S, queue_E, queue_W,   -- lane queue lengths (normalised)
   wait_N,  wait_S,  wait_E,  wait_W,    -- cumulative wait (normalised)
   current_phase,                         -- 0 or 1
   phase_duration,                        -- normalised time in current phase
   pressure]                              -- |queue_in - queue_out| (normalised)

Action per intersection:
  0 = keep current phase
  1 = switch to next phase (with yellow transition)

Reward:
  - sum of waiting times across all vehicles (negative)
  + throughput bonus
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Locate SUMO installation
# ---------------------------------------------------------------------------
def _find_sumo() -> str:
    """Return path to sumo binary, checking SUMO_HOME first."""
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        candidate = Path(sumo_home) / "bin" / "sumo"
        if candidate.exists():
            return str(candidate)
    # Try common Linux/macOS install locations
    for path in ["/usr/bin/sumo", "/usr/local/bin/sumo", "/opt/homebrew/bin/sumo"]:
        if Path(path).exists():
            return path
    return "sumo"  # rely on PATH


def _find_sumo_gui() -> str:
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        candidate = Path(sumo_home) / "bin" / "sumo-gui"
        if candidate.exists():
            return str(candidate)
    for path in ["/usr/bin/sumo-gui", "/usr/local/bin/sumo-gui", "/opt/homebrew/bin/sumo-gui"]:
        if Path(path).exists():
            return path
    return "sumo-gui"


def _import_traci():
    """Import traci, adding SUMO_HOME/tools to sys.path if needed."""
    try:
        import traci
        return traci
    except ImportError:
        pass
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        tools = Path(sumo_home) / "tools"
        if str(tools) not in sys.path:
            sys.path.insert(0, str(tools))
        try:
            import traci
            return traci
        except ImportError:
            pass
    raise ImportError(
        "Could not import traci. Install SUMO and set SUMO_HOME, "
        "or install the 'traci' Python package: pip install traci"
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TL_IDS = ["A0", "A1", "A2", "B0", "B1", "B2", "C0", "C1", "C2"]
N_INTERSECTIONS = len(TL_IDS)

# Adjacency list (used externally by the GNN)
ADJACENCY = {
    "A0": ["A1", "B0"],
    "A1": ["A0", "A2", "B1"],
    "A2": ["A1", "B2"],
    "B0": ["A0", "B1", "C0"],
    "B1": ["B0", "B2", "A1", "C1"],
    "B2": ["B1", "A2", "C2"],
    "C0": ["B0", "C1"],
    "C1": ["C0", "C2", "B1"],
    "C2": ["C1", "B2"],
}

# Build edge_index tensor (COO format) - used by PyG
def build_edge_index() -> np.ndarray:
    """Returns shape (2, E) int array for PyTorch Geometric."""
    id2idx = {tid: i for i, tid in enumerate(TL_IDS)}
    src, dst = [], []
    for node, neighbors in ADJACENCY.items():
        for nb in neighbors:
            src.append(id2idx[node])
            dst.append(id2idx[nb])
    return np.array([src, dst], dtype=np.int64)


NODE_FEATURE_DIM = 11   # per intersection
MAX_QUEUE        = 30.0  # normalisation constant
MAX_WAIT         = 300.0
MAX_PRESSURE     = 20.0
MIN_GREEN        = 10    # seconds - minimum green before switch allowed
YELLOW_DURATION  = 4     # seconds of yellow inserted before phase switch


class SumoEnv:
    """
    Gym-style environment wrapping SUMO TraCI for the 3x3 grid.

    Parameters
    ----------
    cfg_path        : path to grid.sumocfg
    use_gui         : launch sumo-gui instead of sumo (visual mode)
    max_steps       : episode length in simulation seconds
    delta_time      : seconds per RL decision step
    traci_port      : TraCI server port (change if running multiple instances)
    """

    def __init__(
        self,
        cfg_path: str = "sumo_network/grid.sumocfg",
        use_gui: bool = False,
        max_steps: int = 3600,
        delta_time: int = 5,
        traci_port: int = 8813,
    ):
        self.cfg_path    = str(Path(cfg_path).resolve())
        self.use_gui     = use_gui
        self.max_steps   = max_steps
        self.delta_time  = delta_time
        self.traci_port  = traci_port

        self.traci = _import_traci()

        self.tl_ids       = TL_IDS
        self.n_agents     = N_INTERSECTIONS
        self.edge_index   = build_edge_index()

        # Per-intersection state tracking
        self._phase       = {tid: 0    for tid in self.tl_ids}
        self._phase_time  = {tid: 0    for tid in self.tl_ids}
        self._in_yellow   = {tid: False for tid in self.tl_ids}
        self._yellow_cnt  = {tid: 0    for tid in self.tl_ids}

        self._step_count  = 0
        self._running     = False

        # Metrics tracking
        self.episode_rewards : List[float] = []
        self.episode_waits   : List[float] = []
        self.episode_thru    : int = 0

    # ------------------------------------------------------------------
    # Gymnasium-style API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Start / restart SUMO. Returns initial observation."""
        if self._running:
            self._close()

        binary = _find_sumo_gui() if self.use_gui else _find_sumo()
        cmd = [
            binary,
            "-c", self.cfg_path,
            "--no-step-log",
            "--waiting-time-memory", "1000",
            "--no-warnings",
            "--random",
        ]

        self.traci.start(cmd, port=self.traci_port)
        self._running = True
        self._step_count = 0
        self.episode_rewards = []
        self.episode_waits   = []
        self.episode_thru    = 0

        # Reset internal trackers
        for tid in self.tl_ids:
            self._phase[tid]      = 0
            self._phase_time[tid] = 0
            self._in_yellow[tid]  = False
            self._yellow_cnt[tid] = 0
            self.traci.trafficlight.setPhase(tid, 0)

        # Advance a couple steps to populate detector data
        for _ in range(2):
            self.traci.simulationStep()

        return self._get_observations()

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Apply actions (dict tl_id -> 0/1) and advance simulation.

        Returns
        -------
        obs     : np.ndarray  shape (N, NODE_FEATURE_DIM)
        reward  : float       total reward for this step
        done    : bool
        info    : dict
        """
        # --- Apply actions ---
        for tid, action in actions.items():
            self._apply_action(tid, action)

        # --- Advance SUMO ---
        for _ in range(self.delta_time):
            self.traci.simulationStep()
            self._step_count += 1
            # Decrement yellow counters
            for tid in self.tl_ids:
                if self._in_yellow[tid]:
                    self._yellow_cnt[tid] += 1
                    if self._yellow_cnt[tid] >= YELLOW_DURATION:
                        self._in_yellow[tid]  = False
                        self._yellow_cnt[tid] = 0
                        new_phase = 2 if self._phase[tid] == 0 else 0
                        self._phase[tid] = new_phase
                        self.traci.trafficlight.setPhase(tid, new_phase)
                else:
                    self._phase_time[tid] += 1

        obs    = self._get_observations()
        reward = self._compute_reward()
        done   = self._step_count >= self.max_steps or \
                 self.traci.simulation.getMinExpectedNumber() == 0

        info = {
            "step":       self._step_count,
            "n_vehicles": self.traci.vehicle.getIDCount(),
            "mean_wait":  self._mean_waiting_time(),
            "throughput": self.episode_thru,
        }

        self.episode_rewards.append(reward)
        return obs, reward, done, info

    def close(self):
        self._close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_action(self, tid: str, action: int):
        """action=1 means switch phase (if min green satisfied)."""
        if action == 1 and not self._in_yellow[tid]:
            if self._phase_time[tid] >= MIN_GREEN:
                # Insert yellow phase
                yellow_phase = 1 if self._phase[tid] == 0 else 3
                self.traci.trafficlight.setPhase(tid, yellow_phase)
                self._in_yellow[tid]  = True
                self._yellow_cnt[tid] = 0
                self._phase_time[tid] = 0

    def _get_observations(self) -> np.ndarray:
        """
        Returns node feature matrix: shape (N_INTERSECTIONS, NODE_FEATURE_DIM).
        Feature order per node:
          [q_N, q_S, q_E, q_W, w_N, w_S, w_E, w_W, phase, phase_time, pressure]
        """
        obs = np.zeros((self.n_agents, NODE_FEATURE_DIM), dtype=np.float32)
        for i, tid in enumerate(self.tl_ids):
            lanes_in, lanes_out = self._get_lanes(tid)
            q_in  = [self._queue(l) for l in lanes_in[:4]]
            w_in  = [self._wait(l)  for l in lanes_in[:4]]
            # Pad to 4 if fewer lanes
            while len(q_in) < 4: q_in.append(0.0)
            while len(w_in) < 4: w_in.append(0.0)

            q_out = sum(self._queue(l) for l in lanes_out)
            pressure = (sum(q_in) - q_out) / MAX_PRESSURE

            obs[i, 0:4] = np.clip(np.array(q_in[:4])  / MAX_QUEUE, 0, 1)
            obs[i, 4:8] = np.clip(np.array(w_in[:4])  / MAX_WAIT,  0, 1)
            obs[i, 8]   = float(self._phase[tid] == 2)  # 0=NS green, 1=EW green
            obs[i, 9]   = min(self._phase_time[tid] / 60.0, 1.0)
            obs[i, 10]  = np.clip(pressure, -1, 1)
        return obs

    def _compute_reward(self) -> float:
        """
        Reward = -Σ waiting_time(vehicle) / n_vehicles  (per step)
        + 0.1 * vehicles_that_arrived_this_step
        """
        arrived = self.traci.simulation.getArrivedNumber()
        self.episode_thru += arrived

        veh_ids = self.traci.vehicle.getIDList()
        if not veh_ids:
            return 0.0
        total_wait = sum(
            self.traci.vehicle.getWaitingTime(v) for v in veh_ids
        )
        mean_wait = total_wait / len(veh_ids)
        reward = -mean_wait / MAX_WAIT + 0.1 * arrived
        self.episode_waits.append(mean_wait)
        return float(reward)

    def _mean_waiting_time(self) -> float:
        veh_ids = self.traci.vehicle.getIDList()
        if not veh_ids:
            return 0.0
        return sum(self.traci.vehicle.getWaitingTime(v) for v in veh_ids) / len(veh_ids)

    def _get_lanes(self, tl_id: str) -> Tuple[List[str], List[str]]:
        """Return (incoming_lanes, outgoing_lanes) for a traffic light."""
        try:
            controlled = self.traci.trafficlight.getControlledLanes(tl_id)
            incoming = list(dict.fromkeys(controlled))  # unique, preserve order
        except Exception:
            incoming = []

        try:
            links = self.traci.trafficlight.getControlledLinks(tl_id)
            outgoing = list(dict.fromkeys(
                link[0][1] for link in links if link
            ))
        except Exception:
            outgoing = []

        return incoming, outgoing

    def _queue(self, lane_id: str) -> float:
        try:
            return float(self.traci.lane.getLastStepHaltingNumber(lane_id))
        except Exception:
            return 0.0

    def _wait(self, lane_id: str) -> float:
        try:
            return float(self.traci.lane.getWaitingTime(lane_id))
        except Exception:
            return 0.0

    def _close(self):
        try:
            self.traci.close()
        except Exception:
            pass
        self._running = False
