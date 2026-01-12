"""
====================================================================
   ROBUST SHORTEST PATH OPTIMIZER
   Compatible with STRFGenerator and default.json
====================================================================
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gurobipy import GRB, Model


# ==========================================================
# DATA CONTAINERS
# ==========================================================

@dataclass(frozen=True)
class ScenarioData:
    """Container for STRF scenarios used by the robust optimizer."""

    nodes: pd.DataFrame
    edge_lists: List[pd.DataFrame]
    fields: List[np.ndarray]

    def num_edges(self) -> int:
        return len(self.edge_lists[0])

    def num_scenarios(self) -> int:
        return len(self.edge_lists)


@dataclass(frozen=True)
class RobustOptimizationResult:
    """Simple struct returned by the MILP solver."""

    edge_indices: List[int]
    objective_value: float


# ==========================================================
# UTILITIES
# ==========================================================

DEFAULT_CONFIG = "/Users/philippstockerl/BachelorThesis/project/config/default.json"


def load_config(path: str | os.PathLike[str] = DEFAULT_CONFIG) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


# ==========================================================
# LOAD GRAPH DATA
# ==========================================================

def load_scenarios(data_root: str | os.PathLike[str]) -> ScenarioData:
    """
    Loads nodes.csv and all scenario_<t> folders produced by STRFGenerator.
    The edge tables are sorted by edge_id so that scenario costs align.
    """

    root = Path(data_root)
    nodes_path = root / "nodes.csv"
    if not nodes_path.exists():
        raise FileNotFoundError(f"No nodes.csv found in {root}")

    nodes = pd.read_csv(nodes_path)
    if "node_id" not in nodes.columns:
        raise ValueError("nodes.csv must contain a 'node_id' column.")
    nodes = nodes.sort_values("node_id").reset_index(drop=True)
    nodes["node_id"] = nodes["node_id"].astype(int)

    scenario_dirs = sorted([p for p in root.glob("scenario_*") if p.is_dir()])
    if not scenario_dirs:
        raise FileNotFoundError(f"No scenario_* folders found in {root}")

    edge_lists: List[pd.DataFrame] = []
    fields: List[np.ndarray] = []

    for scen_path in scenario_dirs:
        edges_csv = scen_path / "edges.csv"
        field_npy = scen_path / "field.npy"
        if not edges_csv.exists() or not field_npy.exists():
            raise FileNotFoundError(f"Scenario folder {scen_path} is missing edges.csv or field.npy")

        edges = pd.read_csv(edges_csv)
        if "edge_id" not in edges.columns:
            edges["edge_id"] = np.arange(len(edges))

        edges = edges.sort_values("edge_id").reset_index(drop=True)
        edges["edge_index"] = np.arange(len(edges))

        edge_lists.append(edges)
        fields.append(np.load(field_npy))

    _validate_edge_topology(edge_lists)

    print(f"Loaded {len(edge_lists)} scenarios from {root}.")
    return ScenarioData(nodes=nodes, edge_lists=edge_lists, fields=fields)


def _validate_edge_topology(edge_lists: Sequence[pd.DataFrame]) -> None:
    """Ensures every scenario shares the same (u, v, edge_id) ordering."""

    template = edge_lists[0][["edge_id", "u", "v"]].reset_index(drop=True)

    for idx, edges in enumerate(edge_lists[1:], start=1):
        candidate = edges[["edge_id", "u", "v"]].reset_index(drop=True)
        if not template.equals(candidate):
            raise ValueError(
                f"Edge structure mismatch between scenario 0 and scenario {idx}. "
                "Cannot build a shared decision variable vector."
            )


# ==========================================================
# ROBUST MILP
# ==========================================================

def solve_robust_model(
    nodes: pd.DataFrame,
    edge_lists: Sequence[pd.DataFrame],
    start_node: int,
    goal_node: int,
    gurobi_output: bool = True,
) -> RobustOptimizationResult:
    """
    Builds and solves the robust shortest path model:
        min z
        s.t.  z >= sum_e c_e^s * x_e   ∀ scenarios s
              flow_balance(node) == {-1,0,+1} depending on node role
              x_e ∈ {0, 1}
    """

    if not edge_lists:
        raise ValueError("At least one scenario is required.")

    node_ids = nodes["node_id"].tolist()
    if start_node not in node_ids or goal_node not in node_ids:
        raise ValueError("Start or goal node is not present in nodes.csv.")

    num_edges = len(edge_lists[0])
    num_scenarios = len(edge_lists)

    cost_tensor = _build_cost_tensor(edge_lists)
    out_incidence, in_incidence = _build_incidence_maps(nodes, edge_lists[0])

    model = Model("robust_shortest_path")
    model.Params.OutputFlag = 1 if gurobi_output else 0

    x = model.addVars(num_edges, vtype=GRB.BINARY, name="edge_selected")
    z = model.addVar(vtype=GRB.CONTINUOUS, name="worst_case_cost")

    model.setObjective(z, GRB.MINIMIZE)

    for scen_idx in range(num_scenarios):
        model.addConstr(
            z >= sum(cost_tensor[scen_idx, e] * x[e] for e in range(num_edges)),
            name=f"worstcase_{scen_idx}",
        )

    for node in node_ids:
        outgoing = sum(x[e] for e in out_incidence[node])
        incoming = sum(x[e] for e in in_incidence[node])

        if node == start_node:
            rhs = 1
            name = "flow_start"
        elif node == goal_node:
            rhs = -1
            name = "flow_goal"
        else:
            rhs = 0
            name = f"flow_{node}"

        model.addConstr(outgoing - incoming == rhs, name=name)

    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi terminated with status {model.Status}. Cannot extract solution.")

    chosen_edges = [idx for idx in range(num_edges) if x[idx].X > 0.5]
    return RobustOptimizationResult(edge_indices=chosen_edges, objective_value=z.X)


def _build_cost_tensor(edge_lists: Sequence[pd.DataFrame]) -> np.ndarray:
    """Stacks the scenario-specific cost columns into a (S, E) tensor."""

    num_scenarios = len(edge_lists)
    num_edges = len(edge_lists[0])

    tensor = np.zeros((num_scenarios, num_edges), dtype=float)
    for s_idx, edges in enumerate(edge_lists):
        tensor[s_idx, :] = edges["cost"].to_numpy(dtype=float)
    return tensor


def _build_incidence_maps(
    nodes: pd.DataFrame, edges: pd.DataFrame
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """Returns dictionaries mapping node_id to outgoing/incoming edge indices."""

    node_ids = nodes["node_id"].tolist()
    node_out: Dict[int, List[int]] = {node: [] for node in node_ids}
    node_in: Dict[int, List[int]] = {node: [] for node in node_ids}

    for idx, row in edges.iterrows():
        u = int(row["u"])
        v = int(row["v"])
        node_out[u].append(idx)
        node_in[v].append(idx)

    return node_out, node_in


# ==========================================================
# PATH → NODE SEQUENCE
# ==========================================================

def edge_path_to_nodes(
    edges_df: pd.DataFrame,
    chosen_edges: Sequence[int],
    start: int,
    goal: int,
) -> List[int]:
    """
    Converts the selected edge indices into an ordered node path.
    Raises a ValueError if the path cannot be reconstructed.
    """

    successor: Dict[int, int] = {}
    for edge_idx in chosen_edges:
        row = edges_df.iloc[edge_idx]
        u = int(row["u"])
        v = int(row["v"])
        successor[u] = v

    path = [start]
    current = start
    max_steps = len(edges_df) + 1

    for _ in range(max_steps):
        if current == goal:
            return path

        if current not in successor:
            break

        current = successor[current]
        path.append(current)

    raise ValueError(
        "Failed to reconstruct a start-to-goal path from the selected edges. "
        "Check the MILP solution and graph connectivity."
    )


# ==========================================================
# OVERLAYS
# ==========================================================

def export_overlays(
    nodes: pd.DataFrame,
    fields: Sequence[np.ndarray],
    edge_lists: Sequence[pd.DataFrame],
    chosen_edges: Sequence[int],
    out_dir: str | os.PathLike[str],
) -> Path:
    """Exports per-scenario overlays and a GIF summarizing the robust path."""

    out_path = ensure_dir(out_dir)
    nodes_by_id = nodes.set_index("node_id")[["x", "y"]]
    frames = []

    for t_idx, field in enumerate(fields):
        scen_id = f"scenario_{t_idx:03d}"
        out_png = out_path / f"{scen_id}_overlay.png"

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(field, cmap="viridis", origin="lower")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        edges = edge_lists[t_idx].iloc[chosen_edges]
        for _, edge in edges.iterrows():
            u = int(edge["u"])
            v = int(edge["v"])
            x1, y1 = nodes_by_id.loc[u, ["x", "y"]]
            x2, y2 = nodes_by_id.loc[v, ["x", "y"]]
            ax.plot([x1, x2], [y1, y2], color="red", linewidth=1.5)

        ax.set_title(f"Robust Path Overlay — Scenario {t_idx}")
        fig.tight_layout()
        fig.savefig(out_png, dpi=120)
        plt.close(fig)

        frames.append(imageio.imread(out_png))

    gif_path = out_path / "robust_path_overlay_animation.gif"
    imageio.mimsave(gif_path, frames, fps=2, loop=0)

    print(f"\nSaved overlay animation:\n{gif_path.resolve()}")
    return gif_path


# ==========================================================
# MAIN ENTRYPOINT
# ==========================================================

def run_robust_optimizer(config_path: str | os.PathLike[str] = DEFAULT_CONFIG) -> None:
    cfg = load_config(config_path)

    data_root = Path(cfg["paths"]["data_root"]).expanduser()
    result_root = ensure_dir(data_root)
    seed = cfg.get("seed", "unknown")
    robust_root = ensure_dir(result_root / "Robust")
    print(f"[DEBUG] Result directory: {robust_root.resolve()}")

    start_node = int(cfg["robust_model"]["start_node"])
    goal_node = cfg["robust_model"]["goal_node"]
    if goal_node == "auto":
        nodes_test = pd.read_csv(Path(data_root) / "nodes.csv")
        goal_node = int(nodes_test["node_id"].max())
    else:
        goal_node = int(goal_node)

    print(f"Start = {start_node}, Goal = {goal_node}")

    scenario_data = load_scenarios(str(data_root))

    result = solve_robust_model(
        scenario_data.nodes,
        scenario_data.edge_lists,
        start_node=start_node,
        goal_node=goal_node,
    )

    print(f"\nChosen robust edges: {result.edge_indices}")
    print(f"Worst-case cost: {result.objective_value:.3f}")

    def export_dataframe(df: pd.DataFrame, path: Path, label: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        if path.exists():
            print(f"[DEBUG] Exported {label}: {path.resolve()}")
        else:
            raise RuntimeError(f"Failed to export {label} to {path}")

    edges_export_path = robust_root / "robust_path_edges.csv"
    selected_edges = scenario_data.edge_lists[0].iloc[result.edge_indices].copy()
    selected_edges.insert(0, "edge_index", selected_edges.index)
    export_dataframe(selected_edges, edges_export_path, "robust edge solution")

    node_path = edge_path_to_nodes(
        scenario_data.edge_lists[0], result.edge_indices, start_node, goal_node
    )
    nodes_export_path = robust_root / "robust_path_nodes.csv"
    export_dataframe(pd.DataFrame({"node_id": node_path}), nodes_export_path, "robust node path")

    if cfg["robust_model"]["export_overlays"]:
        overlay_path = export_overlays(
            scenario_data.nodes,
            scenario_data.fields,
            scenario_data.edge_lists,
            result.edge_indices,
            robust_root,
        )
        print(f"[DEBUG] Overlay frames written under {robust_root.resolve()}")
        print(f"[DEBUG] Overlay animation exported to {Path(overlay_path).resolve()}")

    print("\n✓ Robust optimization complete.\n")


# ==========================================================
# SCRIPT MODE
# ==========================================================

if __name__ == "__main__":
    run_robust_optimizer(DEFAULT_CONFIG)
