# dstar_visualizer.py
# ------------------------------------------------------------
# Visualizes D* Lite's final perceived environment, the D* path,
# and the initial robust path overlayed on a Gaussian-like map.
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

def visualize_dstar_environment(
    scenario_dir="Data/scenarios/scenario_1",
    hybrid_edges_path="Data/dstar_hybrid_edges.csv",
    summary_json_path="Data/dstar_run_summary.json",
    robust_csv_path="Data/robust_solution_breakdown.csv",
    output_path="Data/dstar_visualization.png",
    edge_update_log_path="Data/dstar_edge_updates.csv",
    detailed_output_path="Data/dstar_visualization_detailed.png",
):
    # --- Load data ---
    nodes_df = pd.read_csv(f"{scenario_dir}/nodes.csv")
    edges_df = pd.read_csv(hybrid_edges_path)
    with open(summary_json_path, "r") as f:
        summary = json.load(f)

    robust_path = []
    if os.path.exists(robust_csv_path):
        try:
            r_df = pd.read_csv(robust_csv_path)
            for col in ["path_node_id", "node", "node_id"]:
                if col in r_df.columns:
                    robust_path = r_df[col].astype(int).tolist()
                    break
        except Exception:
            pass
    if not robust_path and "robust_path_nodes" in summary:
        if isinstance(summary["robust_path_nodes"], list):
            robust_path = summary["robust_path_nodes"]
        elif isinstance(summary["robust_path_nodes"], dict):
            if "path" in summary["robust_path_nodes"]:
                robust_path = summary["robust_path_nodes"]["path"]

    # --- Create background map (using node positions) ---
    # Determine grid size from node coordinates
    x_coords = nodes_df["x"].astype(int)
    y_coords = nodes_df["y"].astype(int)
    ncols = x_coords.max() + 1
    nrows = y_coords.max() + 1
    grid = np.full((nrows, ncols), np.nan)

    # For each node, compute mean cost of its outgoing edges (perceived)
    node_costs = {}
    for nid in nodes_df["node_id"]:
        out_edges = edges_df[edges_df["start"] == nid]
        if len(out_edges) > 0:
            node_costs[nid] = out_edges["cost"].mean()
        else:
            node_costs[nid] = 0.0
    # Fill grid with cost mean for each node
    for idx, row in nodes_df.iterrows():
        x, y = int(row["x"]), int(row["y"])
        nid = int(row["node_id"])
        grid[y, x] = node_costs[nid]
    # Interpolate missing (nan) values if any
    mask = np.isnan(grid)
    if np.any(mask):
        # Simple nearest fill for nans
        from scipy.ndimage import distance_transform_edt
        grid_filled = grid.copy()
        inds = np.array(np.nonzero(~mask)).T
        for y in range(nrows):
            for x in range(ncols):
                if mask[y, x]:
                    # Find nearest non-nan
                    dists = np.sqrt((inds[:,0]-y)**2 + (inds[:,1]-x)**2)
                    nearest = inds[np.argmin(dists)]
                    grid_filled[y, x] = grid[nearest[0], nearest[1]]
        grid = grid_filled
    # grid = grid / np.nanmax(grid)  # old normalization (relative scale 0–1)
    # New normalization (absolute scale 1–10 to match robust visualizer)
    grid = 1 + 9 * (grid - np.nanmin(grid)) / (np.nanmax(grid) - np.nanmin(grid))

    # --- SIMPLE VISUALIZATION: Only grid, robust path (red), D* Lite path (orange), start/goal, simple legend ---
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap="viridis", origin="lower")
    # plt.suptitle(f"D* Lite Path vs. Robust Path\nRobust Objective: {summary.get('robust_objective', 'N/A')}")
    dstar_cost = summary.get('total_realized_cost', 'N/A')
    nominal_cost = summary.get('nominal_cost', None)
    robust_obj = summary.get('robust_objective', 'N/A')
    title_lines = [f"D* Lite Path vs. Robust Path"]
    title_lines.append(f"D* Lite Total Cost: {dstar_cost}")
    if nominal_cost is not None:
        title_lines.append(f"Nominal Cost: {nominal_cost}")
    title_lines.append(f"Robust Objective: {robust_obj}")
    plt.title("\n".join(title_lines), pad=10)

    # Plot robust path (red)
    robust_line = None
    if robust_path:
        for i in range(len(robust_path) - 1):
            u, v = robust_path[i], robust_path[i + 1]
            xu, yu = nodes_df.loc[nodes_df["node_id"] == u, ["x", "y"]].values[0]
            xv, yv = nodes_df.loc[nodes_df["node_id"] == v, ["x", "y"]].values[0]
            robust_line = plt.plot([xu, xv], [yu, yv], color="red", linewidth=2.5, alpha=0.9, label="Robust Path" if i == 0 else "")

    # Plot D* Lite final path (orange)
    dstar_path = summary["path_nodes"]
    dstar_line = None
    for i in range(len(dstar_path) - 1):
        u, v = dstar_path[i], dstar_path[i + 1]
        xu, yu = nodes_df.loc[nodes_df["node_id"] == u, ["x", "y"]].values[0]
        xv, yv = nodes_df.loc[nodes_df["node_id"] == v, ["x", "y"]].values[0]
        dstar_line = plt.plot([xu, xv], [yu, yv], color="orange", linewidth=2.5, alpha=0.95, label="D* Lite Path" if i == 0 else "")

    # Mark start and goal
    start, goal = summary["start"], summary["goal"]
    xs, ys = nodes_df.loc[nodes_df["node_id"] == start, ["x", "y"]].values[0]
    xg, yg = nodes_df.loc[nodes_df["node_id"] == goal, ["x", "y"]].values[0]
    plt.scatter(xs, ys, color="lime", s=80, edgecolors="black", label="Start", zorder=10)
    plt.scatter(xg, yg, color="magenta", s=80, edgecolors="black", label="Goal", zorder=10)

    # Legend: only robust path, D* Lite path, Start, Goal
    handles, labels = plt.gca().get_legend_handles_labels()
    seen = set()
    new_handles = []
    new_labels = []
    for h, l in zip(handles, labels):
        if l not in seen and l != "":
            new_handles.append(h)
            new_labels.append(l)
            seen.add(l)
    plt.legend(new_handles, new_labels, loc="upper left", fontsize=10, frameon=True)

    plt.colorbar(label="Normalized perceived edge cost")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ Visualization saved to {output_path}")

    # --- DETAILED VISUALIZATION: As before, with updated edges/counts ---
    # Only create detailed if edge update log exists
    import collections
    edge_update_counts = collections.Counter()
    edge_update_steps = collections.defaultdict(list)
    edge_update_log_exists = os.path.exists(edge_update_log_path)
    if edge_update_log_exists:
        plt.figure(figsize=(8, 8))
        plt.imshow(grid, cmap="viridis", origin="lower")
        plt.title("D* Lite Perceived Environment with Robust and D* Paths")

        # Plot edges that were updated to actual
        updated_edges = edges_df[edges_df["source"] == "actual"]
        for idx, r in updated_edges.iterrows():
            u, v = int(r["start"]), int(r["end"])
            xu, yu = nodes_df.loc[nodes_df["node_id"] == u, ["x", "y"]].values[0]
            xv, yv = nodes_df.loc[nodes_df["node_id"] == v, ["x", "y"]].values[0]
            plt.plot([xu, xv], [yu, yv], color="red", linewidth=1.2, alpha=0.7, label="Actual Edge" if idx == updated_edges.index[0] else "")

        update_log_df = pd.read_csv(edge_update_log_path)
        for _, row in update_log_df.iterrows():
            u, v = int(row["u"]), int(row["v"])
            edge_update_counts[(u, v)] += 1
            edge_update_steps[(u, v)].append(int(row["step"]))
        if edge_update_counts:
            max_count = max(edge_update_counts.values())
        else:
            max_count = 1
        import matplotlib as mpl
        red_cmap = mpl.colormaps["Reds"]
        norm = mpl.colors.Normalize(vmin=1, vmax=max_count)
        for (u, v), count in edge_update_counts.items():
            xu, yu = nodes_df.loc[nodes_df["node_id"] == u, ["x", "y"]].values[0]
            xv, yv = nodes_df.loc[nodes_df["node_id"] == v, ["x", "y"]].values[0]
            color = red_cmap(norm(count))
            plt.plot([xu, xv], [yu, yv], color=color, linewidth=3.0, alpha=0.95, zorder=8, label=None)
            mx, my = (xu + xv) / 2, (yu + yv) / 2
            plt.text(mx, my, str(count), color="black", fontsize=8, ha="center", va="center", zorder=20, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
        from matplotlib.lines import Line2D
        proxy = Line2D([0], [0], color=red_cmap(norm(max_count)), linewidth=3.0, label="Updated Edges (Actual discovered)")
        # Plot robust path (blue)
        robust_line = None
        if robust_path:
            for i in range(len(robust_path) - 1):
                u, v = robust_path[i], robust_path[i + 1]
                xu, yu = nodes_df.loc[nodes_df["node_id"] == u, ["x", "y"]].values[0]
                xv, yv = nodes_df.loc[nodes_df["node_id"] == v, ["x", "y"]].values[0]
                robust_line = plt.plot([xu, xv], [yu, yv], color="blue", linewidth=2.0, alpha=0.8, label="Robust Path" if i == 0 else "")
        # Plot D* Lite path (orange)
        dstar_line = None
        for i in range(len(dstar_path) - 1):
            u, v = dstar_path[i], dstar_path[i + 1]
            xu, yu = nodes_df.loc[nodes_df["node_id"] == u, ["x", "y"]].values[0]
            xv, yv = nodes_df.loc[nodes_df["node_id"] == v, ["x", "y"]].values[0]
            dstar_line = plt.plot([xu, xv], [yu, yv], color="orange", linewidth=2.5, alpha=0.9, label="D* Lite Path" if i == 0 else "")
        # Mark start and goal
        plt.scatter(xs, ys, color="lime", s=80, edgecolors="black", label="Start", zorder=10)
        plt.scatter(xg, yg, color="magenta", s=80, edgecolors="black", label="Goal", zorder=10)
        # Legend
        handles, labels = plt.gca().get_legend_handles_labels()
        seen = set()
        new_handles = []
        new_labels = []
        for h, l in zip(handles, labels):
            if l not in seen and l != "":
                new_handles.append(h)
                new_labels.append(l)
                seen.add(l)
        new_handles.append(proxy)
        new_labels.append("Updated Edges (Actual discovered)")
        plt.legend(new_handles, new_labels, loc="upper right", fontsize=10, frameon=True)
        plt.colorbar(label="Normalized perceived edge cost")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.tight_layout()
        plt.savefig(detailed_output_path, dpi=300)
        plt.close()
        print(f"✅ Visualization with updated edges saved to {detailed_output_path}")

if __name__ == "__main__":
    visualize_dstar_environment()
