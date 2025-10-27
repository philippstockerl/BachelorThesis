import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_node_coords(nodes_csv: str) -> tuple[np.ndarray, dict]:
    """
    Returns:
      - coords_arr: np.ndarray of shape (N,2) indexed by a dense 0..N-1 index
      - id_to_idx: dict mapping node_id -> dense index usable to slice coords_arr
    This is robust even if node_id are not contiguous.
    """
    df = pd.read_csv(nodes_csv)
    # Determine id column
    id_col = "node_id" if "node_id" in df.columns else ("id" if "id" in df.columns else None)
    if id_col is None:
        raise ValueError("nodes.csv must have a 'node_id' (or 'id') column")
    df = df.sort_values(id_col).reset_index(drop=True)

    node_ids = df[id_col].astype(int).tolist()
    coords = df[["x", "y"]].to_numpy()
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    return coords, id_to_idx


def load_path_nodes(scn_dir: str) -> list[int]:
    """
    Prefer robust_path_nodes.csv (single column 'path_node_id').
    Fallback to robust_solution_breakdown.csv:
      - use 'path_node_id' column if present
      - else reconstruct from 'u','v' (take all u and the last v).
    """
    path_nodes_csv = os.path.join(scn_dir, "robust_path_nodes.csv")
    breakdown_csv = os.path.join(scn_dir, "robust_solution_breakdown.csv")

    if os.path.exists(path_nodes_csv):
        df = pd.read_csv(path_nodes_csv)
        col = "path_node_id" if "path_node_id" in df.columns else df.columns[0]
        ids = df[col].dropna().astype(int).tolist()
        if not ids:
            raise ValueError("robust_path_nodes.csv is empty after dropping NaNs.")
        return ids

    # fallback: use breakdown csv
    if not os.path.exists(breakdown_csv):
        raise FileNotFoundError("Could not find robust_path_nodes.csv nor robust_solution_breakdown.csv")
    dfb = pd.read_csv(breakdown_csv)

    if "path_node_id" in dfb.columns:
        ids = dfb["path_node_id"].dropna().astype(int).tolist()
        if ids:
            return ids

    if {"u", "v"}.issubset(dfb.columns):
        valid = dfb.dropna(subset=["u", "v"])
        if valid.empty:
            raise ValueError("robust_solution_breakdown.csv has no valid rows with 'u' and 'v'.")
        u_seq = valid["u"].astype(int).tolist()
        last_v = int(valid["v"].iloc[-1])
        return u_seq + [last_v]

    raise ValueError("robust_solution_breakdown.csv missing expected columns ('path_node_id') or ('u','v').")


def load_totals_from_breakdown(scn_dir: str) -> tuple[float | None, float | None]:
    """
    Read nominal and robust totals from robust_solution_breakdown.csv where
    'order' equals 'TOTAL_NOMINAL_COST' and 'TOTAL_ROBUST_OBJECTIVE'.
    Values are stored in the column 'nominal_forecast' by the solver exporter.
    """
    breakdown_csv = os.path.join(scn_dir, "robust_solution_breakdown.csv")
    nominal_cost = None
    robust_objective = None

    if os.path.exists(breakdown_csv):
        df = pd.read_csv(breakdown_csv)
        if "order" in df.columns and "nominal_forecast" in df.columns:
            nom = df.loc[df["order"].astype(str) == "TOTAL_NOMINAL_COST", "nominal_forecast"]
            rob = df.loc[df["order"].astype(str) == "TOTAL_ROBUST_OBJECTIVE", "nominal_forecast"]
            if not nom.empty and pd.notna(nom.iloc[0]):
                nominal_cost = float(nom.iloc[0])
            if not rob.empty and pd.notna(rob.iloc[0]):
                robust_objective = float(rob.iloc[0])

        # Gentle fallback (last resort): if no totals rows exist, try single-value columns
        if robust_objective is None and "robust_objective" in df.columns:
            robust_objective = float(df["robust_objective"].dropna().iloc[0])
        if nominal_cost is None and "nominal_cost" in df.columns:
            nominal_cost = float(df["nominal_cost"].dropna().iloc[0])

        # WARNING: summing 'robust_cost_component' only yields Σρ, not the full robust objective.
    return nominal_cost, robust_objective


def main():
    parser = argparse.ArgumentParser(description="Visualize robust path on forecast field")
    parser.add_argument("--scenario", type=str, default="Data/scenarios/scenario_1",
                        help="Path to scenario folder containing forecast_field.npy etc.")
    args = parser.parse_args()
    scn_dir = args.scenario

    # --- Load artifacts ---
    cost_field = np.load(os.path.join(scn_dir, "forecast_field.npy"))
    coords, id_to_idx = load_node_coords(os.path.join(scn_dir, "nodes.csv"))
    path_node_ids = load_path_nodes(scn_dir)
    nominal_cost, robust_objective = load_totals_from_breakdown(scn_dir)

    # Map node IDs to dense indices for robust slicing
    try:
        idx_seq = [id_to_idx[int(nid)] for nid in path_node_ids]
    except KeyError as e:
        raise KeyError(f"Node id {e.args[0]} from the path is not present in nodes.csv") from None
    path_xy = coords[idx_seq, :]

    # --- Plot ---
    plt.figure(figsize=(8, 8))
    im = plt.imshow(cost_field, origin="lower", cmap="viridis")
    plt.colorbar(im, label="Forecast Cost")

    plt.plot(path_xy[:, 0], path_xy[:, 1], color="red", linewidth=2, label="Robust Path")

    start_xy = path_xy[0]
    goal_xy = path_xy[-1]
    plt.scatter([start_xy[0]], [start_xy[1]], color="green", s=80, marker="o", label="Start")
    plt.scatter([goal_xy[0]], [goal_xy[1]], color="blue", s=80, marker="o", label="Goal")

    scenario_name = os.path.basename(scn_dir)
    title = f"Robust Path Overlay - {scenario_name}"
    # Put both objectives in the title, using totals if available
    obj_bits = []
    if nominal_cost is not None:
        obj_bits.append(f"Nominal Cost: {nominal_cost:.2f}")
    if robust_objective is not None:
        obj_bits.append(f"Robust Objective: {robust_objective:.2f}")
    if obj_bits:
        title += "\n" + " | ".join(obj_bits)
    plt.title(title)
    plt.legend(loc="upper left")

    out_png = os.path.join(scn_dir, "robust_path_overlay.png")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"✅ Saved robust overlay: {out_png}")


if __name__ == "__main__":
    main()