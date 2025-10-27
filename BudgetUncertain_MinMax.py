import os
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def robust_shortest_path(
    nodes_csv: str,
    edges_csv: str,
    start: int,
    goal: int,
    Gamma: float = 0.0,
    time_limit: float | None = None,
):
    """
    Bertsimas–Sim robust shortest path / unit-flow MCF.

    Parameters
    ----------
    nodes_csv : path to nodes.csv  (columns: node_id,x,y)
    edges_csv : path to edges_forecast.csv (columns: edge_id,u,v,nominal_forecast,deviation_abs)
    start     : source node id
    goal      : sink node id
    Gamma     : uncertainty budget
    time_limit: optional Gurobi time limit (seconds)

    Returns
    -------
    dict with:
      - 'path_nodes': list[int]
      - 'used_edges': list[int] (edge_id)
      - 'robust_obj': float
      - 'nominal_cost_on_path': float
      - 'pi': float
      - 'sum_rho': float
      - 'x': dict[edge_id -> 0/1]
    """
    # ---- Load & validate ----
    Nd = pd.read_csv(nodes_csv)
    Ed = pd.read_csv(edges_csv)

    required_nodes = {"node_id", "x", "y"}
    required_edges = {"edge_id", "u", "v", "nominal_forecast", "deviation_abs"}

    if not required_nodes.issubset(set(Nd.columns)):
        raise ValueError(f"nodes_csv missing columns {required_nodes - set(Nd.columns)}")

    if not required_edges.issubset(set(Ed.columns)):
        raise ValueError(f"edges_csv missing columns {required_edges - set(Ed.columns)}")

    Nd = Nd.copy()
    Nd["node_id"] = Nd["node_id"].astype(int)

    Ed = Ed.copy()
    for col in ["edge_id", "u", "v"]:
        Ed[col] = Ed[col].astype(int)
    for col in ["nominal_forecast", "deviation_abs"]:
        Ed[col] = Ed[col].astype(float)

    V = set(Nd["node_id"].tolist())
    if start not in V or goal not in V:
        raise ValueError(f"start ({start}) or goal ({goal}) not in nodes.csv")

    # Useful maps
    edges = list(Ed["edge_id"])
    idx_by_eid = {eid: i for i, eid in enumerate(edges)}
    out_by_node = {i: [] for i in V}
    in_by_node  = {i: [] for i in V}
    for _, r in Ed.iterrows():
        out_by_node[r.u].append(int(r.edge_id))
        in_by_node[r.v].append(int(r.edge_id))

    # ---- Build model ----
    m = gp.Model("RobustMCF_BertsimasSim")
    if time_limit is not None:
        m.Params.TimeLimit = time_limit
    m.Params.OutputFlag = 1

    # Decision variables
    x = m.addVars(edges, vtype=GRB.BINARY, name="x")       # choose edges
    pi = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="pi") # global penalty
    rho = m.addVars(edges, lb=0.0, vtype=GRB.CONTINUOUS, name="rho")

    # Objective: sum c x + Gamma*pi + sum rho
    obj = gp.quicksum(Ed.loc[idx_by_eid[eid], "nominal_forecast"] * x[eid] for eid in edges) \
        + Gamma * pi \
        + gp.quicksum(rho[eid] for eid in edges)
    m.setObjective(obj, GRB.MINIMIZE)

    # Robust constraints: pi + rho_e >= d_e * x_e
    for eid in edges:
        d_e = Ed.loc[idx_by_eid[eid], "deviation_abs"]
        m.addConstr(pi + rho[eid] >= d_e * x[eid], name=f"robust_{eid}")

    # Flow balance (unit from start to goal)
    # sum_out(i) - sum_in(i) = b_i
    # b_start = 1, b_goal = -1, others = 0
    for i in V:
        b_i = 1 if i == start else (-1 if i == goal else 0)
        m.addConstr(
            gp.quicksum(x[eid] for eid in out_by_node[i]) -
            gp.quicksum(x[eid] for eid in in_by_node[i]) == b_i,
            name=f"flow_{i}"
        )

    m.optimize()

    # ---- Extract solution ----
    if m.Status != GRB.OPTIMAL and m.Status != GRB.TIME_LIMIT:
        raise RuntimeError(f"Gurobi status {m.Status}: no solution found")

    # Selected edges
    x_sol = {eid: int(round(x[eid].X)) for eid in edges if x[eid].X > 0.5}
    chosen_eids = list(x_sol.keys())

    # Reconstruct path (traverse from start)
    succ = {}
    for _, r in Ed.iterrows():
        if int(r.edge_id) in x_sol:
            succ[int(r.u)] = int(r.v)

    path_nodes = [start]
    cur = start
    visited = set([start])
    while cur in succ:
        nxt = succ[cur]
        path_nodes.append(nxt)
        if nxt in visited:   # safety check against cycles
            break
        visited.add(nxt)
        cur = nxt
        if cur == goal:
            break

    # Costs
    nominal_cost_on_path = float(sum(Ed.loc[idx_by_eid[eid], "nominal_forecast"] for eid in chosen_eids))
    pi_val = float(pi.X)
    sum_rho = float(sum(rho[eid].X for eid in edges))
    robust_obj = float(m.ObjVal)

    # --- Path-ordered edge breakdown ---
    # Build path-ordered list of edge_ids
    path_edge_ids = []
    for i in range(len(path_nodes) - 1):
        from_node = path_nodes[i]
        to_node = path_nodes[i+1]
        # Find edge in Ed where u==from_node and v==to_node and edge_id in chosen_eids
        row = Ed[(Ed["u"] == from_node) & (Ed["v"] == to_node) & (Ed["edge_id"].isin(chosen_eids))]
        if not row.empty:
            eid = int(row.iloc[0]["edge_id"])
            path_edge_ids.append(eid)
        else:
            path_edge_ids.append(None)  # Safety: should not happen

    # Build breakdown DataFrame
    breakdown_rows = []
    for idx, eid in enumerate(path_edge_ids):
        if eid is None:
            continue
        edge_row = Ed.loc[idx_by_eid[eid]]
        deviation_abs = edge_row["deviation_abs"]
        covered_by_pi = (deviation_abs <= pi_val + 1e-9)
        rho_implied = max(deviation_abs - pi_val, 0.0)
        breakdown_rows.append({
            "order": idx + 1,
            "edge_id": eid,
            "u": int(edge_row["u"]),
            "v": int(edge_row["v"]),
            "nominal_forecast": edge_row["nominal_forecast"],
            "deviation_abs": deviation_abs,
            "covered_by_pi": covered_by_pi,
            "rho_implied": rho_implied,
            "robust_cost_component": rho_implied,
        })
    df_breakdown = pd.DataFrame(breakdown_rows, columns=[
        "order", "edge_id", "u", "v", "nominal_forecast", "deviation_abs",
        "covered_by_pi", "rho_implied", "robust_cost_component"
    ])

    # Totals (footer, not appended to DataFrame)
    totals = {
        "nominal_cost_on_path": nominal_cost_on_path,
        "Gamma_times_pi": Gamma * pi_val,
        "sum_rho": df_breakdown["rho_implied"].sum(),
        "robust_objective": robust_obj,
        "check_nominal_plus_surcharge": nominal_cost_on_path + Gamma * pi_val + df_breakdown["rho_implied"].sum(),
    }

    # Print breakdown report
    print("\n--- Robust surcharge breakdown (path order) ---")
    with pd.option_context('display.float_format', '{:.3f}'.format):
        print(df_breakdown.to_string(index=False))
    print("\n--- Totals ---")
    for k, v in totals.items():
        print(f"{k}: {v:.3f}")

    # Save breakdown DataFrame to CSV
    out_dir = os.path.dirname(edges_csv)
    breakdown_csv = os.path.join(out_dir, "robust_solution_breakdown.csv")

    # Add nominal and robust objective values as extra rows for clarity
    extra_rows = pd.DataFrame([
        {"order": None, "edge_id": None, "u": None, "v": None, "nominal_forecast": None,
         "deviation_abs": None, "covered_by_pi": None, "rho_implied": None, "robust_cost_component": None},
        {"order": "TOTAL_NOMINAL_COST", "edge_id": None, "u": None, "v": None, "nominal_forecast": nominal_cost_on_path,
         "deviation_abs": None, "covered_by_pi": None, "rho_implied": None, "robust_cost_component": None},
        {"order": "TOTAL_ROBUST_OBJECTIVE", "edge_id": None, "u": None, "v": None, "nominal_forecast": robust_obj,
         "deviation_abs": None, "covered_by_pi": None, "rho_implied": None, "robust_cost_component": None},
    ])
    df_breakdown_to_save = pd.concat([df_breakdown, extra_rows], ignore_index=True)
    df_breakdown_to_save.to_csv(breakdown_csv, index=False)

    # --- Export robust path nodes as CSV for D* Lite compatibility ---
    robust_path_nodes_csv = os.path.join(out_dir, "robust_path_nodes.csv")
    # Save as a single-column CSV: path_node_id
    df_path_nodes = pd.DataFrame({"path_node_id": path_nodes})
    df_path_nodes.to_csv(robust_path_nodes_csv, index=False)

    # Print confirmation for both files
    print(f"\n✅ Robust path breakdown exported to: {breakdown_csv}")
    print(f"✅ Robust path node sequence exported to: {robust_path_nodes_csv}")

    return {
        "path_nodes": path_nodes,
        "used_edges": chosen_eids,
        "robust_obj": robust_obj,
        "nominal_cost_on_path": nominal_cost_on_path,
        "pi": pi_val,
        "sum_rho": sum_rho,
        "x": x_sol,
        "breakdown_csv": breakdown_csv,
        "path_edge_ids": path_edge_ids,
        "robust_path_nodes_csv": robust_path_nodes_csv,
    }


if __name__ == "__main__":
    # Example hard-coded call — adapt paths as needed
    result = robust_shortest_path(
        nodes_csv="Data/scenarios/scenario_1/nodes.csv",
        edges_csv="Data/scenarios/scenario_1/edges_forecast.csv",
        start=0,
        goal=1599,
        Gamma=20,
    )
    print("\n=== Robust Result ===")
    print(f"Robust objective: {result['robust_obj']:.3f}")
    print(f"Nominal cost on path: {result['nominal_cost_on_path']:.3f}")
    print(f"pi: {result['pi']:.6f}   sum_rho: {result['sum_rho']:.6f}")
    print(f"Path nodes: {result['path_nodes']}")
    print(f"Used edges: {result['used_edges']}")
    print(f"Breakdown CSV saved to: {result['breakdown_csv']}")
    print(f"Robust path nodes CSV saved to: {result['robust_path_nodes_csv']}")