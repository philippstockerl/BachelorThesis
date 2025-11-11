# %% [markdown]
# <h1 style="color:#2E86C1; font-family: 'Helvetica';">d*liteOptimized_refactored.py</h2>
# 
# <p style="font-size: 16px; line-height: 1.5; color:#333;">
# Clean implementation of the <b>D* Lite</b> dynamic path replanning algorithm — without global variables.
# </p>

# %%
import os
import math
import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% ------------------------------------------------------------
# (1) DStarLiteState Dataclass
# ------------------------------------------------------------

@dataclass
class DStarLiteState:
    S: set
    succ: dict
    pred: dict
    cost_forecast: dict
    cost_nominal: dict
    node_positions: dict
    g: dict = field(default_factory=lambda: defaultdict(lambda: float('inf')))
    rhs: dict = field(default_factory=lambda: defaultdict(lambda: float('inf')))
    U: list = field(default_factory=list)
    k_m: float = 0.0
    s_start: int = None
    s_goal: int = None


# %% ------------------------------------------------------------
# (2) Helper Functions
# ------------------------------------------------------------

def load_graph_data(nodes_csv: str, edges_csv: str):
    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)
    id_col = "id" if "id" in nodes.columns else "node_id"

    S = set(nodes[id_col].astype(int).tolist())
    node_positions = {int(row[id_col]): (row["x"], row["y"]) for _, row in nodes.iterrows()}
    cost_forecast = {(int(row.u), int(row.v)): float(row.cost) for _, row in edges.iterrows()}
    cost_nominal = {(int(row.u), int(row.v)): float(row.nominal) for _, row in edges.iterrows()}

    succ = defaultdict(list)
    pred = defaultdict(list)
    for _, row in edges.iterrows():
        u, v = int(row.u), int(row.v)
        succ[u].append(v)
        pred[v].append(u)

    return S, succ, pred, cost_forecast, cost_nominal, node_positions


def c(u, v, cost_dict):
    return cost_dict.get((u, v), float('inf'))


def h(a, b, node_positions):
    ax, ay = node_positions[a]
    bx, by = node_positions[b]
    return abs(ax - bx) + abs(ay - by)


def KeyLessThan(k1, k2):
    eps = 1e-12
    if k1[0] < k2[0] - eps: return True
    if k1[0] > k2[0] + eps: return False
    return k1[1] < k2[1] - eps


# %% ------------------------------------------------------------
# (3) Priority Queue Operations
# ------------------------------------------------------------

def U_Remove(U, s):
    U[:] = [(key, node) for key, node in U if node != s]
    heapq.heapify(U)


def U_Insert(U, u, k):
    heapq.heappush(U, (k, u))


def U_Update(U, s, k):
    U_Remove(U, s)
    heapq.heappush(U, (k, s))


def U_TopKey(U):
    return U[0][0] if U else (float('inf'), float('inf'))


def U_Top(U):
    return U[0][1] if U else None


# %% ------------------------------------------------------------
# (4) Core D* Lite Algorithm
# ------------------------------------------------------------

def CalculateKey(state: DStarLiteState, s):
    val = min(state.g[s], state.rhs[s])
    return (val + h(state.s_start, s, state.node_positions) + state.k_m, val)


def UpdateVertex(state: DStarLiteState, u):
    in_queue = any(node == u for _, node in state.U)
    if state.g[u] != state.rhs[u] and in_queue:
        U_Update(state.U, u, CalculateKey(state, u))
    elif state.g[u] != state.rhs[u] and not in_queue:
        U_Insert(state.U, u, CalculateKey(state, u))
    elif state.g[u] == state.rhs[u] and in_queue:
        U_Remove(state.U, u)


def Initialize(state: DStarLiteState):
    state.U.clear()
    state.k_m = 0.0
    for s in state.S:
        state.g[s] = float('inf')
        state.rhs[s] = float('inf')

    state.rhs[state.s_goal] = 0.0
    U_Insert(state.U, state.s_goal, CalculateKey(state, state.s_goal))
    UpdateVertex(state, state.s_start)


def ComputeShortestPath(state: DStarLiteState):
    while (KeyLessThan(U_TopKey(state.U), CalculateKey(state, state.s_start))
           or state.rhs[state.s_start] != state.g[state.s_start]):

        if not state.U:
            break

        k_old, u = heapq.heappop(state.U)
        k_new = CalculateKey(state, u)

        if KeyLessThan(k_old, k_new):
            U_Insert(state.U, u, k_new)
            continue

        if state.g[u] > state.rhs[u]:
            state.g[u] = state.rhs[u]
            for s in state.pred[u]:
                if s != state.s_goal:
                    state.rhs[s] = min(state.rhs[s], c(s, u, state.cost_forecast) + state.g[u])
                UpdateVertex(state, s)
        else:
            g_old = state.g[u]
            state.g[u] = float('inf')
            for s in (list(state.pred[u]) + [u]):
                if state.rhs[s] == c(s, u, state.cost_forecast) + g_old:
                    if s != state.s_goal:
                        state.rhs[s] = min(
                            (c(s, sp, state.cost_forecast) + state.g[sp]) for sp in state.succ[s]
                        )
                UpdateVertex(state, s)


def scan_for_edge_changes(state: DStarLiteState, threshold=1e-6, current_node=None):
    changed = []
    if current_node is None:
        return changed
    for v in state.succ[current_node]:
        forecast_val = state.cost_forecast.get((current_node, v))
        nominal_val = state.cost_nominal.get((current_node, v))
        if nominal_val is None:
            continue
        if abs(forecast_val - nominal_val) > threshold:
            changed.append((current_node, v))
    return changed


# %% ------------------------------------------------------------
# (5) Reporting & Visualization
# ------------------------------------------------------------

def build_dstar_breakdown(g, rhs, U, path_nodes, succ, cost_forecast, cost_nominal):
    rows, total_forecast, total_nominal = [], 0.0, 0.0
    for idx in range(len(path_nodes) - 1):
        u, v = path_nodes[idx], path_nodes[idx + 1]
        forecast_cost = cost_forecast.get((u, v), float('inf'))
        actual_cost = cost_nominal.get((u, v), float('inf'))
        deviation = actual_cost - forecast_cost
        total_forecast += forecast_cost
        total_nominal += actual_cost
        rows.append({
            "order": idx + 1,
            "u": u, "v": v,
            "forecast_cost": forecast_cost,
            "actual_cost": actual_cost,
            "deviation": deviation,
            "g_value": g[u],
            "rhs_value": rhs[u],
            "in_queue": any(node == u for _, node in U)
        })
    rows.append({
        "order": "TOTAL_FORECAST_COST",
        "forecast_cost": total_forecast,
        "actual_cost": "",
        "deviation": "",
        "g_value": "",
        "rhs_value": "",
        "in_queue": ""
    })
    return pd.DataFrame(rows), total_forecast, total_nominal


def save_dstar_results(out_dir, breakdown_df, path_nodes):
    os.makedirs(out_dir, exist_ok=True)
    breakdown_csv = os.path.join(out_dir, "dstar_solution_breakdown.csv")
    path_csv = os.path.join(out_dir, "dstar_path_nodes.csv")
    breakdown_df.to_csv(breakdown_csv, index=False)
    pd.DataFrame({"path_node_id": path_nodes}).to_csv(path_csv, index=False)
    print(f"Saved D* Lite breakdown → {breakdown_csv}")
    print(f"Saved D* Lite path nodes → {path_csv}")
    return breakdown_csv, path_csv


def plot_dstar_path_overlay(nodes_csv, edges_csv, cost_forecast, path_nodes, output_dir, total_forecast=None, total_nominal=None):
    forecast_npy = edges_csv.replace("edges.csv", "actual.npy")
    cost_field = np.load(forecast_npy).T

    nodes = pd.read_csv(nodes_csv)
    id_to_idx = {int(row.id if "id" in row else row.node_id): i for i, row in nodes.iterrows()}
    coords = nodes[["x", "y"]].to_numpy()
    path_xy = np.array([coords[id_to_idx[n]] for n in path_nodes])

    plt.figure(figsize=(8, 8))
    im = plt.imshow(cost_field, origin="lower", cmap="viridis")
    plt.colorbar(im, label="Forecast Cost (normalized)")
    plt.plot(path_xy[:, 0], path_xy[:, 1], color="red", linewidth=2, label="D* Lite Path")
    plt.scatter(path_xy[0, 0], path_xy[0, 1], color="orange", s=60, label="Start")
    plt.scatter(path_xy[-1, 0], path_xy[-1, 1], color="cyan", s=60, label="Goal")
    plt.legend()
    title = "D* Lite Path Overlay (forecast background, actual discovered)"
    if total_forecast is not None and total_nominal is not None:
        title += f"\nTotal Forecast Cost: {total_forecast:.2f} | Total Nominal Cost: {total_nominal:.2f}"
    plt.title(title)
    plt.tight_layout()
    out_png = os.path.join(output_dir, "dstar_path_overlay.png")
    plt.savefig(out_png)
    plt.close()
    print(f"Saved D* Lite visualization → {out_png}")
    return out_png


# %% ------------------------------------------------------------
# (6) Main Execution Function
# ------------------------------------------------------------

def Main(state: DStarLiteState, output_dir=None, visualize=True, nodes_csv=None, edges_csv=None, cost_forecast_initial=None):
    s_last = state.s_start
    Initialize(state)
    ComputeShortestPath(state)

    path_nodes, step = [state.s_start], 0
    while state.s_start != state.s_goal:
        if state.g[state.s_start] == float('inf'):
            print("No valid path to goal!")
            break
        succ_nodes = state.succ[state.s_start]
        if not succ_nodes:
            print("No successors available from start!")
            break
        state.s_start = min(succ_nodes, key=lambda sp: c(state.s_start, sp, state.cost_forecast) + state.g[sp])
        path_nodes.append(state.s_start)
        step += 1

        changed_edges = scan_for_edge_changes(state, current_node=state.s_start)
        if changed_edges:
            state.k_m += h(s_last, state.s_start, state.node_positions)
            s_last = state.s_start
            for (u, v) in changed_edges:
                c_old = c(u, v, state.cost_forecast)
                new_cost = c(u, v, state.cost_nominal)
                state.cost_forecast[(u, v)] = new_cost
                if c_old > new_cost:
                    if u != state.s_goal:
                        state.rhs[u] = min(state.rhs[u], new_cost + state.g[v])
                elif state.rhs[u] == c_old + state.g[v]:
                    if u != state.s_goal:
                        state.rhs[u] = min(c(u, sp, state.cost_forecast) + state.g[sp] for sp in state.succ[u])
                UpdateVertex(state, u)
            ComputeShortestPath(state)

    breakdown_df, total_forecast, total_nominal = build_dstar_breakdown(
        state.g, state.rhs, state.U, path_nodes, state.succ, cost_forecast_initial, state.cost_nominal
    )
    breakdown_csv, path_csv = save_dstar_results(output_dir, breakdown_df, path_nodes)
    if visualize:
        plot_dstar_path_overlay(nodes_csv, edges_csv, state.cost_forecast, path_nodes, output_dir, total_forecast, total_nominal)
    print(f"\nFinal D* Lite path has {len(path_nodes)} nodes.")
    print(f"Total Nominal Cost = {total_nominal:.3f}, Forecast Cost = {total_forecast:.3f}")
    return {
        "path_nodes": path_nodes,
        "breakdown_csv": breakdown_csv,
        "path_csv": path_csv,
        "total_nominal": total_nominal,
        "total_forecast": total_forecast
    }


# %% ------------------------------------------------------------
# (7) Entry Point
# ------------------------------------------------------------

if __name__ == "__main__":
    dir = "/Users/philippstockerl/BachelorThesis/Data/seed_35684_grid100_Stable_ls30.0_var1.0_nug0.05/"
    nodes_csv = f"{dir}/nodes.csv"
    edges_csv = f"{dir}/edges.csv"

    S, succ, pred, cost_forecast, cost_nominal, node_positions = load_graph_data(nodes_csv, edges_csv)
    cost_forecast_initial = dict(cost_forecast)
    state = DStarLiteState(S, succ, pred, cost_forecast, cost_nominal, node_positions)
    state.s_start = min(S)
    state.s_goal = max(S)

    out_dir = os.path.join(dir, "dstar_results")
    result = Main(state, output_dir=out_dir, visualize=True, nodes_csv=nodes_csv, edges_csv=edges_csv, cost_forecast_initial=cost_forecast_initial)
    print("\n=== D* Lite Export Summary ===")
    print(f"Path CSV: {result['path_csv']}")
    print(f"Breakdown CSV: {result['breakdown_csv']}")
    print(f"Total Nominal Cost: {result['total_nominal']:.3f}")
    print(f"Total Forecast Cost: {result['total_forecast']:.3f}")