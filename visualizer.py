

#!/usr/bin/env python3
"""
Unified Visualization: Robust vs. D* Lite Path Comparison
---------------------------------------------------------
Visualizes and compares the robust path (Bertsimas–Sim) and the online D* Lite path
on the same Gaussian field scenario.

Usage:
    python visualization.py --scenario_dir <scenario_dir> --Gamma <Gamma> --start <start_id> --goal <goal_id>
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Import robust solver
from BudgetUncertain_MinMax import build_and_solve
# Import D* Lite and helpers
from D_Star_Lite import DStarLite, load_graph_from_csv, parse_path_txt

def load_node_coords(nodes_csv):
    df = pd.read_csv(nodes_csv)
    coords = {int(row['id'] if 'id' in row else row['node_id']): (row['x'], row['y']) for _, row in df.iterrows()}
    return coords

def main():
    # Hardcoded variables
    scenario_dir = Path("Data/scenarios/scenario_01")
    Gamma = 20
    start = 0
    goal = 224

    edges_forecast_csv = scenario_dir / "edges_forecast.csv"
    edges_realized_csv = scenario_dir / "edges_realized.csv"
    nodes_csv = scenario_dir / "nodes.csv"
    forecast_map_png = scenario_dir / "forecast_map.png"

    # --- Robust path (Bertsimas–Sim) ---
    robust_result = build_and_solve(
        str(edges_forecast_csv),
        str(nodes_csv),
        start,
        goal,
        Gamma
    )
    robust_path = robust_result["Path Nodes"]
    robust_cost = robust_result["Robust Objective"]

    # --- D* Lite path (online, realized) ---
    # Load graph with realized edge costs
    g_realized = load_graph_from_csv(str(edges_realized_csv), str(nodes_csv), realization="worstcase")
    planner = DStarLite(g_realized, start, goal)
    # Run with no preferred path (pure D* Lite)
    dstar_path, dstar_cost, dstar_replans = planner.run(prefer_path=None, verbose=False)

    # --- Visualization ---
    coords = load_node_coords(str(nodes_csv))

    # Load both background maps
    img_forecast = plt.imread(str(forecast_map_png))
    img_realized = plt.imread(str(scenario_dir / "realized_map.png"))

    # ===============================================================
    # FIGURE 1: Robust path ONLY on forecast (blurred) map
    # ===============================================================
    fig1, ax1 = plt.subplots(figsize=(6,6))
    ax1.imshow(img_forecast, origin='upper')

    robust_xy = [coords[n] for n in robust_path if n in coords]
    if len(robust_xy) >= 2:
        xs, ys = zip(*robust_xy)
        ax1.plot(ys, xs, 'r--', linewidth=2.5, label=f"Robust Path (Γ={Gamma})")

    if start in coords:
        ax1.scatter(coords[start][1], coords[start][0], color='lime', s=100, marker='s', label='Start')
    if goal in coords:
        ax1.scatter(coords[goal][1], coords[goal][0], color='red', s=120, marker='*', label='Goal')

    ax1.set_title(f"Robust Path on Forecast Map (Γ={Gamma})\nCost: {robust_cost:.3f}")
    ax1.axis('off')
    ax1.legend()
    plt.tight_layout()

    # ===============================================================
    # FIGURE 2: Robust vs D* Lite comparison on realized map
    # ===============================================================
    fig2, ax2 = plt.subplots(figsize=(6,6))
    ax2.imshow(img_realized, origin='upper')

    # Robust path (red dashed)
    if len(robust_xy) >= 2:
        xs, ys = zip(*robust_xy)
        ax2.plot(ys, xs, 'r--', linewidth=2.5, label="Robust Path")

    # D* Lite path (blue solid)
    dstar_xy = [coords[n] for n in dstar_path if n in coords]
    if len(dstar_xy) >= 2:
        xs2, ys2 = zip(*dstar_xy)
        ax2.plot(ys2, xs2, 'b-', linewidth=2.5, label="D* Lite Path")

    if start in coords:
        ax2.scatter(coords[start][1], coords[start][0], color='lime', s=100, marker='s', label='Start')
    if goal in coords:
        ax2.scatter(coords[goal][1], coords[goal][0], color='red', s=120, marker='*', label='Goal')

    ax2.set_title("Comparison on Realized Map\n"
                f"Robust Cost: {robust_cost:.3f} | D* Lite Cost: {dstar_cost:.3f} | Replans: {dstar_replans}")
    ax2.axis('off')
    ax2.legend()
    plt.tight_layout()

    # --- Save both figures to the scenario folder ---
    fig1.savefig(scenario_dir / "robust_forecast_vis.png", dpi=300)
    fig2.savefig(scenario_dir / "comparison_realized_vis.png", dpi=300)
    print(f"\n✅ Saved visualization images to: {scenario_dir}")

    plt.show()

    # --- Print summary ---
    print("\n=== Robust Path ===")
    print("Nodes:", robust_path)
    print(f"Robust Objective: {robust_cost:.3f}")
    print("\n=== D* Lite Path ===")
    print("Nodes:", dstar_path)
    print(f"Total Realized Cost: {dstar_cost:.3f}")
    print(f"Replans: {dstar_replans}")

if __name__ == "__main__":
    main()