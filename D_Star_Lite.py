# DStarLite_refactored.py
# ------------------------------------------------------------
# Edge-based D* Lite compatible with the Gaussian-field dataset
# (nodes.csv, edges_forecast.csv, edges_actual.csv).
#
# Key features:
# - Starts on forecast graph (expected costs).
# - Progressively "senses" actual costs for outgoing edges at the
#   current node and updates the graph online.
# - Replans via D* Lite without full recomputation.
# - Exports a hybrid edge set with discovered actual costs.
#
# Minimal required columns:
#   nodes.csv:          node_id, x, y
#   edges_forecast.csv: start, end, nominal        (initial costs)
#   edges_actual.csv:   start, end, nominal        (realized costs)
#
# Optional: robust_solution_breakdown.csv with path_node_id column
# (used only for comparison/logging; D* Lite does not require it).
# ------------------------------------------------------------

from __future__ import annotations

import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Set

import numpy as np
import pandas as pd
import heapq
from collections import defaultdict


Key = Tuple[float, float]
NodeId = int


@dataclass
class DStarResult:
    path: List[NodeId]
    total_cost: float
    replans: int
    expanded: int
    touched: int
    hybrid_edges_path: str
    summary_path: str


class DStarLite:
    """
    Edge-based D* Lite for directed graphs.

    Configurable options:
      - sense_radius (int): Sensing radius around the current node for edge cost discovery.
            0 = only outgoing edges from current node (default, legacy behavior).
            1 = all edges whose endpoints are within 1 grid step (8-neighborhood).
            2 = all edges whose endpoints are within 2 steps, etc.
      - use_robust_init (bool): If True (default), initialize value functions using the robust path as a warm start.
            If False, initialize D* Lite from scratch (no warm start).

    Graph representation:
      self.graph[u][v] = current known cost of edge (u, v)
    Actual costs:
      self.actual_costs[(u, v)] = ground-truth cost for edge (u, v)

    Online sensing:
      At each step, reveal the true costs for all edges whose endpoints are within the specified sense_radius of the current node,
      and update the graph. This triggers repairs via D* Lite.
    """

    def __init__(
        self,
        edges_forecast_df: pd.DataFrame,
        edges_actual_df: pd.DataFrame,
        nodes_df: pd.DataFrame,
        start: NodeId,
        goal: NodeId,
        robust_path_nodes: Optional[List[NodeId]] = None,
        sense_radius: int = 2,
        use_robust_init: bool = True,
    ):
        self.start: NodeId = int(start)
        self.goal: NodeId = int(goal)

        # --- Normalize column names from robust model ---
        rename_map = {"u": "start", "v": "end", "nominal_forecast": "nominal", "deviation_abs": "d"}
        edges_forecast_df = edges_forecast_df.rename(columns=rename_map)
        edges_actual_df = edges_actual_df.rename(columns=rename_map)

        # --- Validate inputs ---
        for col in ["node_id", "x", "y"]:
            if col not in nodes_df.columns:
                raise ValueError(f"nodes_df missing required column '{col}'")
        for col in ["start", "end", "nominal"]:
            if col not in edges_forecast_df.columns:
                raise ValueError(f"edges_forecast_df missing required column '{col}'")
            if col not in edges_actual_df.columns:
                raise ValueError(f"edges_actual_df missing required column '{col}'")

        # Index nodes by node_id for quick coord lookup
        self.nodes = nodes_df.set_index("node_id").sort_index()

        # Build initial graph from forecast (expected) costs
        self.graph: Dict[NodeId, Dict[NodeId, float]] = defaultdict(dict)
        self.succs: Dict[NodeId, List[NodeId]] = defaultdict(list)
        self.preds: Dict[NodeId, List[NodeId]] = defaultdict(list)

        for _, r in edges_forecast_df.iterrows():
            u = int(r["start"])
            v = int(r["end"])
            c = float(r["nominal"])
            self.graph[u][v] = c
            self.succs[u].append(v)
            self.preds[v].append(u)

        # Load actual (ground-truth) costs
        self.actual_costs: Dict[Tuple[NodeId, NodeId], float] = {}
        for _, r in edges_actual_df.iterrows():
            u = int(r["start"])
            v = int(r["end"])
            c = float(r["nominal"])
            self.actual_costs[(u, v)] = c

        # Track which edges have been updated to actual
        self.updated_to_actual: Set[Tuple[NodeId, NodeId]] = set()
        # Debug log of edge updates: (step, current_node, u, v, old_cost, new_cost)
        self.update_log: List[Tuple[int, int, int, int, float, float]] = []

        # D* Lite state
        self.g = defaultdict(lambda: math.inf)    # cost-to-go
        self.rhs = defaultdict(lambda: math.inf)  # one-step lookahead
        self.k_m: float = 0.0

        # Open list as a heap of (k1, k2, node)
        self.open: List[Tuple[float, float, NodeId]] = []
        self.in_open: Dict[NodeId, Tuple[float, float]] = {}  # lazy deletion map

        # Bookkeeping stats
        self.replans: int = 0
        self.expanded: int = 0
        self.touched: int = 0

        # --- Configurable sensing radius and robust initialization ---
        self.sense_radius: int = sense_radius
        """
        Sensing radius around the current node for edge cost discovery.
        0 = only outgoing edges from current node (legacy behavior).
        1 = all edges whose endpoints are within 1 grid step (8-neighbors).
        2 = all edges whose endpoints are within 2 steps, etc.
        """
        self.use_robust_init: bool = use_robust_init
        """
        Whether to initialize D* Lite value functions using the robust path as a warm start.
        If False, D* Lite starts from scratch (no warm start).
        """

        # Optional: initial robust path (for logging and initialization)
        self.robust_path_nodes = list(robust_path_nodes) if robust_path_nodes else None

        # --- Initialization: support robust_path_nodes as warm start ---
        if self.use_robust_init and self.robust_path_nodes and len(self.robust_path_nodes) >= 2:
            # Initialize g and rhs along robust_path_nodes using forecast costs
            path_nodes = self.robust_path_nodes
            # Set rhs at goal to 0
            self.rhs[self.goal] = 0.0
            self.g[self.goal] = math.inf
            # Back-propagate rhs and g along the robust path
            for i in reversed(range(len(path_nodes)-1)):
                u = path_nodes[i]
                v = path_nodes[i+1]
                cost = self.graph[u][v]
                # For robust path: g[u] = cost + g[v] (if v is on path), else inf
                self.g[u] = cost + (self.g[v] if v in self.g and not math.isinf(self.g[v]) else 0.0 if v == self.goal else math.inf)
                self.rhs[u] = cost + (self.g[v] if v in self.g and not math.isinf(self.g[v]) else 0.0 if v == self.goal else math.inf)
            # For all nodes on the robust path except goal, push into open if inconsistent
            for u in path_nodes[:-1]:
                if self.g[u] != self.rhs[u]:
                    self._push_open(u, self._calculate_key(u))
            # Always push goal
            self._push_open(self.goal, self._calculate_key(self.goal))
            print(f"[DSTAR INIT] Loaded robust initial path with {len(self.robust_path_nodes)} nodes.")
        else:
            # Default: Initialize rhs at goal, push goal in OPEN
            self.rhs[self.goal] = 0.0
            self._push_open(self.goal, self._calculate_key(self.goal))

    # ----------------------------
    # Heuristic and keys
    # ----------------------------
    def _heuristic(self, a: NodeId, b: NodeId) -> float:
        # Euclidean heuristic on coordinates
        xa, ya = self.nodes.loc[a, ["x", "y"]]
        xb, yb = self.nodes.loc[b, ["x", "y"]]
        return math.hypot(xa - xb, ya - yb)

    def _calculate_key(self, s: NodeId) -> Key:
        g_rhs = min(self.g[s], self.rhs[s])
        return (g_rhs + self._heuristic(self.start, s) + self.k_m, g_rhs)

    # ----------------------------
    # OPEN management (lazy delete)
    # ----------------------------
    def _push_open(self, s: NodeId, key: Key) -> None:
        heapq.heappush(self.open, (key[0], key[1], s))
        self.in_open[s] = key

    def _pop_open(self) -> Tuple[Key, NodeId]:
        while self.open:
            k1, k2, s = heapq.heappop(self.open)
            if s in self.in_open and self.in_open[s] == (k1, k2):
                del self.in_open[s]
                return (k1, k2), s
        return (math.inf, math.inf), -1

    def _remove_from_open(self, s: NodeId) -> None:
        if s in self.in_open:
            del self.in_open[s]  # lazy delete; heap entry remains but is ignored

    # ----------------------------
    # Core D* Lite operators
    # ----------------------------
    def _predecessors(self, u: NodeId) -> List[NodeId]:
        return self.preds.get(u, [])

    def _successors(self, u: NodeId) -> List[NodeId]:
        return self.succs.get(u, [])

    def _update_vertex(self, u: NodeId) -> None:
        self.touched += 1
        if u != self.goal:
            succ = self._successors(u)
            if len(succ) == 0:
                self.rhs[u] = math.inf
            else:
                self.rhs[u] = min(self.graph[u][v] + self.g[v] for v in succ)
        if self.g[u] != self.rhs[u]:
            self._push_open(u, self._calculate_key(u))
        else:
            self._remove_from_open(u)

    def _compute_shortest_path(self) -> None:
        while True:
            if not self.in_open:
                break
            top_key = min(self.in_open.values())
            start_key = self._calculate_key(self.start)
            if not (top_key < start_key or self.rhs[self.start] != self.g[self.start]):
                break

            _, u = self._pop_open()
            if u == -1:
                break
            self.expanded += 1

            if self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for p in self._predecessors(u):
                    self._update_vertex(p)
            else:
                old_g = self.g[u]
                self.g[u] = math.inf
                succ = self._successors(u)
                for p in self._predecessors(u) + [u]:
                    if p != self.goal and len(self._successors(p)) > 0:
                        # If u was on p's best successor, increasing g[u] may increase rhs[p]
                        # Recompute rhs[p] robustly by considering all successors
                        best = min(self.graph[p][v] + self.g[v] for v in self._successors(p))
                        self.rhs[p] = best
                    self._update_vertex(p)

    # ----------------------------
    # Online sensing of actual costs
    # ----------------------------
    def _sense_outgoing(self, current: NodeId, step: Optional[int] = None) -> int:
        """
        Reveal actual costs for edges whose endpoints are within the
        configured sense_radius of the current node.

        sense_radius=0: Only outgoing edges from `current` (legacy behavior).
        sense_radius=1: All edges whose BOTH endpoints are within 1 grid step (8-neighborhood).
        sense_radius=k: All edges whose BOTH endpoints are within k steps (2k+1 x 2k+1 square).

        Returns number of edges whose costs changed.
        """
        changes = 0
        updated_edges = []
        # Find all node ids within sense_radius of current node
        if self.sense_radius == 0:
            # Legacy: only outgoing edges from current node
            nodes_in_radius = set([current])
        else:
            # Use Euclidean distance on node coordinates to find nodes within radius
            x0, y0 = self.nodes.loc[current, ["x", "y"]]
            # Nodes whose grid distance <= sense_radius
            nodes_in_radius = set(
                self.nodes[
                    ((self.nodes["x"] - x0).abs() <= self.sense_radius)
                    & ((self.nodes["y"] - y0).abs() <= self.sense_radius)
                ].index
            )

        # For all edges in the graph, update if both endpoints in radius
        for u in self.graph:
            for v in self.graph[u]:
                if u in nodes_in_radius and v in nodes_in_radius:
                    key = (u, v)
                    if key in self.actual_costs:
                        true_cost = self.actual_costs[key]
                        prev_cost = self.graph[u][v]
                        if abs(true_cost - prev_cost) > 1e-4:  # detect even small GRF deviations
                            # Update edge to ground truth
                            self.graph[u][v] = true_cost
                            self.updated_to_actual.add(key)
                            # Update consistency of `u`
                            self._update_vertex(u)
                            changes += 1
                            updated_edges.append((u, v, prev_cost, true_cost))
                            # Log the update to self.update_log
                            log_step = step if step is not None else -1
                            self.update_log.append((log_step, current, u, v, prev_cost, true_cost))
        if updated_edges:
            print(f"[DSTAR DEBUG] Sensing at node {current} (sense_radius={self.sense_radius}):")
            for u, v, old, new in updated_edges:
                print(f"    [DSTAR DEBUG] Edge ({u}->{v}) updated: forecast={old:.3f}, actual={new:.3f}")
        if changes > 0:
            print(f"[DSTAR DEBUG] {changes} edges updated to actual values (radius={self.sense_radius})")
            self.replans += 1
        else:
            print(f"[DSTAR DEBUG] No changes detected (radius={self.sense_radius})")
        return changes

    # ----------------------------
    # Run episode
    # ----------------------------
    def run(self, max_steps: int = 10_000) -> DStarResult:
        current = self.start
        path: List[NodeId] = [current]
        cumulative_cost = 0.0

        print("[DSTAR DEBUG] Initial D* Lite planning from start to goal...")
        self._compute_shortest_path()
        print(f"[DSTAR DEBUG] Initial g/rhs values at start node {self.start}: g={self.g[self.start]:.3f}, rhs={self.rhs[self.start]:.3f}")

        steps = 0
        num_replans = 0
        total_edge_changes = 0
        while current != self.goal and steps < max_steps:
            steps += 1
            print(f"\n[DSTAR DEBUG] Step {steps}: At node {current}")
            print(f"[DSTAR DEBUG]    g[{current}]={self.g[current]:.3f}, rhs[{current}]={self.rhs[current]:.3f}")

            # Sense true costs around the current node; update graph & repair policy
            changed = self._sense_outgoing(current, step=steps)
            total_edge_changes += changed
            print(f"[DSTAR DEBUG]    Sensed outgoing edges from node {current}: {changed} edge(s) changed.")
            if changed > 0:
                print(f"[DSTAR DEBUG]    Replanning due to sensed edge changes at node {current}...")
                self._compute_shortest_path()
                num_replans += 1
                print(f"[DSTAR DEBUG]    Replan {num_replans} complete. g[{current}]={self.g[current]:.3f}, rhs[{current}]={self.rhs[current]:.3f}")

            # Choose next hop that minimizes c(u,v) + g[v]
            succ = self._successors(current)
            if len(succ) == 0:
                # Dead-end (should not happen on a connected DAG/grid)
                print("[DSTAR DEBUG]    No successors from current node. Dead-end encountered.")
                break
            costs = [(self.graph[current][v] + self.g[v], v) for v in succ]
            min_cost, nxt = min(costs, key=lambda t: t[0])

            move_cost = self.graph[current][nxt]
            cumulative_cost += move_cost
            print(f"[DSTAR DEBUG]    Moving to node {nxt} via ({current}->{nxt}), edge cost={move_cost:.3f}, cumulative cost={cumulative_cost:.3f}")

            current = nxt
            path.append(current)

            # Early exit if g[start] is inf (no path)
            if math.isinf(self.g[self.start]) and current != self.goal:
                print("[DSTAR DEBUG]    No path to goal exists from current node.")
                break

        total_cost = sum(self.graph[u][v] for u, v in zip(path[:-1], path[1:]))
        # Persist hybrid edges, summary, and update log
        hybrid_edges_path = self._export_hybrid_edges()
        summary_path = self._export_summary(path, total_cost)
        self._export_update_log()

        print("\n[DSTAR DEBUG] === D* Lite Episode Complete ===")
        print(f"[DSTAR DEBUG] Path: {path}")
        print(f"[DSTAR DEBUG] Path length (nodes): {len(path)}")
        print(f"[DSTAR DEBUG] Final cost: {total_cost:.3f}")
        print(f"[DSTAR DEBUG] Sensed/updated edges (actual): {len(self.updated_to_actual)}")
        print(f"[DSTAR DEBUG] Total edge changes discovered: {total_edge_changes}")
        print(f"[DSTAR DEBUG] Number of replans: {num_replans} (internal counter: {self.replans})")
        print(f"[DSTAR DEBUG] Search nodes expanded: {self.expanded}, touched: {self.touched}")
        print(f"[DSTAR DEBUG] Hybrid edges CSV: {hybrid_edges_path}")
        print(f"[DSTAR DEBUG] Summary JSON: {summary_path}")

        return DStarResult(
            path=path,
            total_cost=total_cost,
            replans=self.replans,
            expanded=self.expanded,
            touched=self.touched,
            hybrid_edges_path=hybrid_edges_path,
            summary_path=summary_path,
        )
    def _export_update_log(self) -> str:
        """
        Export the edge update log to CSV.
        Columns: step, current_node, u, v, old_cost, new_cost
        """
        import pandas as pd
        if not self.update_log:
            # No updates; still write header
            df = pd.DataFrame(columns=["step", "current_node", "u", "v", "old_cost", "new_cost"])
        else:
            df = pd.DataFrame(self.update_log, columns=["step", "current_node", "u", "v", "old_cost", "new_cost"])
        path = "Data/dstar_edge_updates.csv"
        df.to_csv(path, index=False)
        return path

    # ----------------------------
    # Exports
    # ----------------------------
    def _export_hybrid_edges(self) -> str:
        rows = []
        for u, nbrs in self.graph.items():
            for v, c in nbrs.items():
                origin = "actual" if (u, v) in self.updated_to_actual else "forecast"
                color = "red" if origin == "actual" else "grey"
                rows.append({"start": u, "end": v, "cost": c, "source": origin, "color_code": color})
        df = pd.DataFrame(rows)
        path = "Data/dstar_hybrid_edges.csv"
        df.to_csv(path, index=False)
        return path

    def _export_summary(self, path_nodes: List[NodeId], total_cost: float) -> str:
        robust_info = None
        if self.robust_path_nodes is not None:
            robust_info = {
                "count": len(self.robust_path_nodes),
                "path": self.robust_path_nodes
            }
        data = {
            "start": self.start,
            "goal": self.goal,
            "path_nodes": path_nodes,
            "total_cost": float(total_cost),
            "total_realized_cost": float(total_cost),  # for visualizer compatibility
            "nominal_cost": None,                      # can be linked later to robust model nominal value
            "robust_objective": None,                  # can be linked later to robust model robust objective
            "replans": int(self.replans),
            "expanded": int(self.expanded),
            "touched": int(self.touched),
            "updated_edges_count": int(len(self.updated_to_actual)),
            "robust_path_nodes": robust_info,
        }
        out = "Data/dstar_run_summary.json"
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
        return out


# ----------------------------
# CLI for quick testing
# ----------------------------
def _load_optional_robust_path(robust_csv_path: Optional[str]) -> Optional[List[NodeId]]:
    if robust_csv_path is None:
        return None
    try:
        df = pd.read_csv(robust_csv_path)
        col = None
        for candidate in ["path_node_id", "node", "node_id"]:
            if candidate in df.columns:
                col = candidate
                break
        if col is None:
            return None
        return [int(x) for x in df[col].tolist()]
    except Exception:
        return None



import os

if __name__ == "__main__":
    # --- Configurable experiment options ---
    # Set these parameters below to experiment with D* Lite behavior:
    SCENARIO_DIR = "Data/scenarios/scenario_1"
    START = 0
    GOAL = 1599
    PRIMARY_ROBUST_PATH_CSV = "Data/scenarios/scenario_1/robust_path_nodes.csv"
    LEGACY_ROBUST_PATH_CSV = "Data/scenarios/scenario_1/robust_solution_breakdown.csv"
    # --- Sensing radius: 0 = only outgoing edges, 1 = 8-neighborhood, etc. ---
    SENSE_RADIUS = 1  # Change to 1, 2, ... for wider sensing
    # --- Use robust path as warm start? (True = robust init, False = scratch) ---
    USE_ROBUST_INIT = True  # Change to False for no warm start

    # --- Load data ---
    nodes = pd.read_csv(f"{SCENARIO_DIR}/nodes.csv")
    edges_forecast = pd.read_csv(f"{SCENARIO_DIR}/edges_forecast.csv")
    edges_actual = pd.read_csv(f"{SCENARIO_DIR}/edges_actual.csv")

    # Try to load robust_path_nodes.csv first, then robust_solution_breakdown.csv
    if os.path.exists(PRIMARY_ROBUST_PATH_CSV):
        robust_nodes = _load_optional_robust_path(PRIMARY_ROBUST_PATH_CSV)
        used_robust_file = PRIMARY_ROBUST_PATH_CSV
    elif os.path.exists(LEGACY_ROBUST_PATH_CSV):
        robust_nodes = _load_optional_robust_path(LEGACY_ROBUST_PATH_CSV)
        used_robust_file = LEGACY_ROBUST_PATH_CSV
    else:
        # robust_nodes = None
        # used_robust_file = None
        # print("[MAIN] No robust initialization path found. Running D* Lite without robust warm start.")
        raise FileNotFoundError("[ERROR] No robust initialization path found. Expected either 'robust_path_nodes.csv' or 'robust_solution_breakdown.csv'.")

    if robust_nodes is not None and used_robust_file is not None:
        print(f"[MAIN] Using robust initialization path from '{used_robust_file}' ({len(robust_nodes)} nodes).")
    # else:
    #     print("[MAIN] No robust initialization path found. Running D* Lite without robust warm start.")

    # --- Instantiate D* Lite ---
    dsl = DStarLite(
        edges_forecast_df=edges_forecast,
        edges_actual_df=edges_actual,
        nodes_df=nodes,
        start=START,
        goal=GOAL,
        robust_path_nodes=robust_nodes,
        sense_radius=SENSE_RADIUS,
        use_robust_init=USE_ROBUST_INIT,
    )
    # --- Run D* Lite episode ---
    result = dsl.run()

    print("\n=== D* Lite Run ===")
    print(f"Path length (nodes): {len(result.path)}")
    print(f"Total realized cost: {result.total_cost:.3f}")
    print(f"Replans: {result.replans} | Expanded: {result.expanded} | Touched: {result.touched}")
    print(f"Hybrid edges saved to: {result.hybrid_edges_path}")
    print(f"Summary saved to:     {result.summary_path}")