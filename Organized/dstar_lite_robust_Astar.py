"""
D* Lite (Optimized Version, Koenig & Likhachev 2002/2005)
----------------------------------------------------------
Modular and extensible implementation for dynamic path replanning.
Supports both:
    • Standard D* Lite (backward A* initialization)
    • Robust-Warm D* Lite (backward robust path initialization)

Author: Philipp Stockerl (Bachelor Thesis, University of Passau)
"""

import os, math, heapq, json, time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class DStarLiteState:
    S: set
    succ: dict
    pred: dict
    cost_forecast: dict
    cost_nominal: dict
    node_positions: dict
    g: dict = field(default_factory=lambda: defaultdict(lambda: float("inf")))
    rhs: dict = field(default_factory=lambda: defaultdict(lambda: float("inf")))
    U: list = field(default_factory=list)
    k_m: float = 0.0
    s_start: int = None
    s_goal: int = None
    R_SENSE: int = 1
    robust_initialized: bool = False
    replan_calls: int = 0


def load_graph_data(nodes_csv: str, edges_csv: str):
    """Load nodes/edges and build successor/predecessor dictionaries."""
    nodes, edges = pd.read_csv(nodes_csv), pd.read_csv(edges_csv)
    id_col = "id" if "id" in nodes.columns else "node_id"
    S = set(nodes[id_col].astype(int).tolist())
    node_positions = {int(r[id_col]): (r["x"], r["y"]) for _, r in nodes.iterrows()}
    cost_forecast = {(int(r.u), int(r.v)): float(r.cost) for _, r in edges.iterrows()}
    cost_nominal  = {(int(r.u), int(r.v)): float(r.nominal) for _, r in edges.iterrows()}

    succ, pred = defaultdict(list), defaultdict(list)
    for _, r in edges.iterrows():
        u, v = int(r.u), int(r.v)
        succ[u].append(v)
        pred[v].append(u)
    return S, succ, pred, cost_forecast, cost_nominal, node_positions


def load_robust_path(path_file: str) -> List[int]:
    df = pd.read_csv(path_file)
    col = df.columns[0]
    path = df[col].astype(int).tolist()
    print(f"Loaded robust path ({len(path)} nodes) from {path_file}")
    return path

def U_Insert(U, s, k): heapq.heappush(U, (k, s))
def U_Remove(U, s): U[:] = [(k, n) for k, n in U if n != s]; heapq.heapify(U)
def U_Update(U, s, k): U_Remove(U, s); heapq.heappush(U, (k, s))
def U_Top(U): return U[0][1] if U else None
def U_TopKey(U): return U[0][0] if U else (float("inf"), float("inf"))

def c(u, v, cost_dict): return cost_dict.get((u, v), float("inf"))

def h(a, b, pos):
    ax, ay = pos[a]; bx, by = pos[b]
    return abs(ax - bx) + abs(ay - by)

def KeyLessThan(k1, k2):
    eps = 1e-9
    return k1[0] < k2[0] - eps or (abs(k1[0]-k2[0]) < eps and k1[1] < k2[1] - eps)

def CalculateKey(state, s):
    val = min(state.g[s], state.rhs[s])
    return (val + h(state.s_start, s, state.node_positions) + state.k_m, val)


def Initialize(state: DStarLiteState):
    state.U.clear(); state.k_m = 0.0
    for s in state.S:
        state.g[s] = float("inf")
        state.rhs[s] = float("inf")
    state.rhs[state.s_goal] = 0.0
    U_Insert(state.U, state.s_goal, CalculateKey(state, state.s_goal))


def UpdateVertex(state: DStarLiteState, u):
    in_queue = any(node == u for _, node in state.U)
    if state.g[u] != state.rhs[u] and not in_queue:
        U_Insert(state.U, u, CalculateKey(state, u))
    elif state.g[u] != state.rhs[u] and in_queue:
        U_Update(state.U, u, CalculateKey(state, u))
    elif state.g[u] == state.rhs[u] and in_queue:
        U_Remove(state.U, u)


def ComputeShortestPath(state: DStarLiteState):
    """Lines 10–28 of the optimized pseudocode."""
    while (KeyLessThan(U_TopKey(state.U), CalculateKey(state, state.s_start))
           or state.rhs[state.s_start] != state.g[state.s_start]):

        if not state.U: break
        k_old, u = heapq.heappop(state.U)
        k_new = CalculateKey(state, u)

        if KeyLessThan(k_old, k_new):
            U_Insert(state.U, u, k_new); continue

        if state.g[u] > state.rhs[u]:
            state.g[u] = state.rhs[u]
            for s in state.pred[u]:
                if s != state.s_goal:
                    state.rhs[s] = min(state.rhs[s], c(s,u,state.cost_forecast)+state.g[u])
                UpdateVertex(state, s)
        else:
            g_old = state.g[u]; state.g[u] = float("inf")
            for s in list(state.pred[u]) + [u]:
                if state.rhs[s] == c(s,u,state.cost_forecast)+g_old:
                    if s != state.s_goal:
                        state.rhs[s] = min(
                            c(s,sp,state.cost_forecast)+state.g[sp]
                            for sp in state.succ[s]
                        )
                UpdateVertex(state, s)
    state.replan_calls += 1


def scan_for_edge_changes(state, current_node, radius=1, threshold=1e-6):
    """Detect local cost deviations (forecast vs nominal)."""
    changed = []; frontier = [current_node]; visited = {current_node}
    for _ in range(radius):
        new_frontier = []
        for u in frontier:
            for v in state.succ[u]:
                f, n = state.cost_forecast.get((u,v)), state.cost_nominal.get((u,v))
                if n is None: continue
                if abs(f - n) > threshold: changed.append((u,v))
                if v not in visited: visited.add(v); new_frontier.append(v)
        frontier = new_frontier
    return changed


def InitializeFromRobustPath(state: DStarLiteState, robust_path: List[int]):
    """Walk robust path (goal→start) to initialize rhs/g values."""
    state.robust_initialized = True
    state.U.clear(); state.k_m = 0.0
    for s in state.S:
        state.g[s] = float("inf"); state.rhs[s] = float("inf")

    goal = robust_path[-1]
    state.s_goal = goal
    state.rhs[goal] = 0.0
    U_Insert(state.U, goal, CalculateKey(state, goal))

    for i in range(len(robust_path)-2, -1, -1):
        u, v = robust_path[i], robust_path[i+1]
        cost = c(u, v, state.cost_forecast)
        state.rhs[u] = min(state.rhs[u], cost + state.g[v])
        UpdateVertex(state, u)
        ComputeShortestPath(state)
    print(f"✅ Initialized along robust path ({len(robust_path)} nodes).")



def DStarMain(state: DStarLiteState):
    """Implements Figure 4 (Main procedure)."""
    s_last = state.s_start
    if not state.robust_initialized:
        Initialize(state)
        print_value_stats(state, "Before ComputeShortestPath (Baseline)")
        ComputeShortestPath(state)
        print_value_stats(state, "After ComputeShortestPath (Baseline)")
    else:
        print("⚙️ Using robust-warm start initialization.")
        print_value_stats(state, "Before ComputeShortestPath (Robust)")
        ComputeShortestPath(state)
        print_value_stats(state, "After ComputeShortestPath (Robust)")

    path = [state.s_start]

    while state.s_start != state.s_goal:
        if state.g[state.s_start] == float("inf"):
            print("⚠️ No known path to goal."); break

        succ_nodes = state.succ[state.s_start]
        if not succ_nodes: break
        state.s_start = min(
            succ_nodes,
            key=lambda sp: c(path[-1], sp, state.cost_forecast) + state.g[sp]
        )
        path.append(state.s_start)

        changed_edges = scan_for_edge_changes(state, state.s_start, radius=state.R_SENSE)
        if changed_edges:
            print(f"Discovered {len(changed_edges)} changed edges (up to 5 samples): {changed_edges[:5]}")
            state.k_m += h(s_last, state.s_start, state.node_positions)
            s_last = state.s_start
            for (u,v) in changed_edges:
                c_old, c_new = c(u,v,state.cost_forecast), c(u,v,state.cost_nominal)
                state.cost_forecast[(u,v)] = c_new
                if c_old > c_new:
                    if u != state.s_goal:
                        state.rhs[u] = min(state.rhs[u], c_new + state.g[v])
                elif state.rhs[u] == c_old + state.g[v]:
                    if u != state.s_goal:
                        state.rhs[u] = min(
                            c(u,sp,state.cost_forecast)+state.g[sp]
                            for sp in state.succ[u]
                        )
                UpdateVertex(state, u)
            ComputeShortestPath(state)

    print(f"✅ D* Lite completed in {state.replan_calls} replan calls.")
    return path



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

def plot_dstar_path_overlay(nodes_csv, edges_csv, cost_forecast, path_nodes, output_dir, total_forecast=None, total_nominal=None, robust_path=None):
    forecast_npy = edges_csv.replace("edges.csv", "actual.npy")
    cost_field = np.load(forecast_npy).T

    nodes = pd.read_csv(nodes_csv)
    id_to_idx = {int(row.id if "id" in row else row.node_id): i for i, row in nodes.iterrows()}
    coords = nodes[["x", "y"]].to_numpy()
    path_xy = np.array([coords[id_to_idx[n]] for n in path_nodes])

    plt.figure(figsize=(8, 8))
    im = plt.imshow(cost_field, origin="lower", cmap="viridis")
    plt.colorbar(im, label="Forecast Cost (normalized)")

    # --- D* Lite path ---
    plt.plot(path_xy[:, 0], path_xy[:, 1], color="red", linewidth=2, label="D* Lite Path")
    plt.scatter(path_xy[0, 0], path_xy[0, 1], color="green", s=200, label="Start")
    plt.scatter(path_xy[-1, 0], path_xy[-1, 1], color="red", s=200, label="Goal")

    # --- Robust overlay (blue dashed) ---
    if robust_path is not None:
        robust_xy = np.array([coords[id_to_idx[n]] for n in robust_path])
        plt.plot(
            robust_xy[:, 0], robust_xy[:, 1],
            color="blue", linestyle="--", linewidth=2.0, label="Robust Path"
        )

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

# Debug helper functions

def first_divergence(path1: List[int], path2: List[int]) -> Tuple[int, int, int]:
    """Find the first index where two paths diverge and the nodes at divergence."""
    min_len = min(len(path1), len(path2))
    for i in range(min_len):
        if path1[i] != path2[i]:
            return i, path1[i], path2[i]
    if len(path1) != len(path2):
        return min_len, path1[min_len] if len(path1) > min_len else None, path2[min_len] if len(path2) > min_len else None
    return -1, None, None  # paths are identical

def stats_after_init(state: DStarLiteState, label: str):
    print(f"--- {label} ---")
    print(f"Number of nodes in open queue U: {len(state.U)}")
    print(f"g-values (sample): {dict(list(state.g.items())[:5])}")
    print(f"rhs-values (sample): {dict(list(state.rhs.items())[:5])}")
    print(f"k_m: {state.k_m}")
    print(f"robust_initialized: {state.robust_initialized}")
    print("-------------------")

def print_value_stats(state: DStarLiteState, tag: str):
    g_vals = np.array([v for v in state.g.values() if v != float("inf")])
    rhs_vals = np.array([v for v in state.rhs.values() if v != float("inf")])
    print(f"--- {tag} ---")
    print(f"g-values: mean={np.mean(g_vals) if g_vals.size > 0 else 'N/A'}, max={np.max(g_vals) if g_vals.size > 0 else 'N/A'}")
    print(f"rhs-values: mean={np.mean(rhs_vals) if rhs_vals.size > 0 else 'N/A'}, max={np.max(rhs_vals) if rhs_vals.size > 0 else 'N/A'}")
    print("-------------------")


def run_dstar():
    dir = "/Users/philippstockerl/BachelorThesis/Data/master78/seed_39622_grid100_Stable_ls30.0_var1.0_nug0.05"
    nodes_csv, edges_csv = f"{dir}/nodes.csv", f"{dir}/edges.csv"
    params_json = "/Users/philippstockerl/BachelorThesis/Organized/parameters.json"

    with open(params_json) as f: params = json.load(f)
    gen, dstarp = params.get("general", {}), params.get("dstar", {})

    S, succ, pred, cf, cn, pos = load_graph_data(nodes_csv, edges_csv)
    cf_init = dict(cf)

    # Compute and print mean, max, and number of nonzero differences between cf and cn
    diffs = []
    for key in cf:
        if key in cn:
            diff = abs(cf[key] - cn[key])
            if diff > 1e-9:
                diffs.append(diff)
    if diffs:
        print(f"Cost difference stats: mean={np.mean(diffs):.6f}, max={np.max(diffs):.6f}, nonzero count={len(diffs)}")
    else:
        print("No nonzero cost differences found between forecast and nominal edges.")

    state = DStarLiteState(S, succ, pred, cf, cn, pos,
                           R_SENSE=dstarp.get("sensing_radius",1))
    state.s_start, state.s_goal = min(S), max(S)

    robust_path = None
    path_baseline = None
    path_warm = None

    if dstarp.get("use_robust_path", False):
        robust_path = load_robust_path(
            dstarp.get("robust_path_file",
                f"{dir}/robust_Gamma_39600.0/robust_path_nodes.csv"))
        InitializeFromRobustPath(state, robust_path)
        stats_after_init(state, "After Init (Robust Path)")
        path_warm = DStarMain(state)
    else:
        stats_after_init(state, "After Init (Baseline)")
        path_baseline = DStarMain(state)

    # If both paths exist, compare divergence
    if path_baseline is not None and path_warm is not None:
        idx, node_base, node_warm = first_divergence(path_baseline, path_warm)
        if idx == -1:
            print("Paths are identical.")
        else:
            print(f"Paths diverge at index {idx}: baseline node {node_base}, warm node {node_warm}")

    # export + plot
    out_dir = os.path.join(dir, dstarp.get("output_dir","dstar_results"))
    os.makedirs(out_dir, exist_ok=True)
    if path_baseline is not None:
        df, fcost, ncost = build_dstar_breakdown(
            state.g, state.rhs, state.U, path_baseline, succ, cf_init, cn)
        save_dstar_results(out_dir, df, path_baseline)
        if dstarp.get("visualize", gen.get("visualize", True)):
            plot_dstar_path_overlay(nodes_csv, edges_csv, cf, path_baseline,
                                    out_dir, fcost, ncost, robust_path)
        print(f"Final Nominal Cost = {ncost:.3f}, Forecast Cost = {fcost:.3f}")
    if path_warm is not None:
        df, fcost, ncost = build_dstar_breakdown(
            state.g, state.rhs, state.U, path_warm, succ, cf_init, cn)
        save_dstar_results(out_dir, df, path_warm)
        if dstarp.get("visualize", gen.get("visualize", True)):
            plot_dstar_path_overlay(nodes_csv, edges_csv, cf, path_warm,
                                    out_dir, fcost, ncost, robust_path)
        print(f"Final Nominal Cost = {ncost:.3f}, Forecast Cost = {fcost:.3f}")

if __name__ == "__main__":
    run_dstar()
