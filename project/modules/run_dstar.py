"""
run_dstar.py
Fully modular runner for the D* Lite module.
Reads default.json, loads STRF-generated graph data,
optionally loads robust initialization path, and runs D* Lite.
Exports:
    - taken path CSV
    - initialization path CSV (A* or robust warm start)
    - overlays for every scenario frame
    - animated GIF
"""

import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from modules.dstar_lite import DStarLite


# ---------------------------------------------------------
# Utility: Load robust initialization path
# ---------------------------------------------------------
def load_robust_path(result_root):
    base = Path(result_root)
    candidates = [
        base / "Robust" / "robust_path_nodes.csv",
        base / "robust_path_nodes.csv",
    ]
    legacy_dir = base / "Robust"
    if legacy_dir.exists():
        for sub in sorted(legacy_dir.glob("Set_*/robust_path_nodes.csv"), reverse=True):
            candidates.append(sub)
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            return df["node_id"].tolist()

    print("⚠ No robust_path_nodes.csv found -> cannot warm-start D* Lite.")
    return None


# ---------------------------------------------------------
# Utility: Load scenario count from STRF output directory
# ---------------------------------------------------------
def count_scenarios(data_root):
    count = 0
    while True:
        folder = os.path.join(data_root, f"scenario_{count:03d}")
        if not os.path.exists(folder):
            break
        count += 1
    return count


# ---------------------------------------------------------
# Utility: overlay path over single scenario field image
# ---------------------------------------------------------
def export_overlay_image(field_path, path_nodes, nodes_df, out_png):
    field = np.load(field_path)

    plt.figure(figsize=(6,6))
    plt.imshow(field, origin="lower", cmap="viridis")

    xs = [nodes_df.loc[n, "x"] for n in path_nodes]
    ys = [nodes_df.loc[n, "y"] for n in path_nodes]

    plt.plot(xs, ys, color="red", linewidth=2.0, label="Taken Path")
    plt.scatter(xs[0], ys[0], color="yellow", s=80, label="Start")
    plt.scatter(xs[-1], ys[-1], color="green", s=80, label="Goal")
    plt.legend()

    plt.title("D* Lite Path Overlay")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ---------------------------------------------------------
# Main runner
# ---------------------------------------------------------
def main():

    # ---------------------------------------------------------
    # Load default.json
    # ---------------------------------------------------------
    CFG_FILE = "/Users/philippstockerl/BachelorThesis/project/config/default.json"

    with open(CFG_FILE, "r") as f:
        cfg = json.load(f)

    seed = cfg["seed"]

    data_root   = Path(cfg["paths"]["data_root"]).expanduser()
    result_root = Path(cfg["paths"]["result_root"]).expanduser()
    seed = cfg.get("seed", "unknown")
    dstar_root = result_root / "DStar"

    graph_connectivity = cfg["graph"]["connectivity"]

    dstar_cfg = cfg["dstar_lite"]

    start_node = cfg["robust_model"]["start_node"]
    goal_node  = cfg["robust_model"]["goal_node"]

    if goal_node == "auto":
        # largest node ID = farthest corner
        nodes_df = pd.read_csv(os.path.join(data_root, "nodes.csv"))
        goal_node = nodes_df["node_id"].max()

    print("============================================")
    print("=== Running D* Lite with modular config ===")
    print("============================================")
    print(f"Start node: {start_node}")
    print(f"Goal node : {goal_node}")
    print(f"Data root : {data_root}")
    print(f"Result root: {dstar_root}")
    print("============================================\n")

    # ---------------------------------------------------------
    # Ensure result directory exists
    # ---------------------------------------------------------
    dstar_root.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Robust initialization path (optional)
    # ---------------------------------------------------------
    robust_path_nodes = None
    if dstar_cfg["use_robust_init"]:
        robust_path_nodes = load_robust_path(result_root)
        if robust_path_nodes is not None:
            print("Using robust warm-start initialization.")
        else:
            print("⚠ Warm-start requested but no robust path found. Falling back to A* init.")

    # ---------------------------------------------------------
    # Count scenarios
    # ---------------------------------------------------------
    T = count_scenarios(str(data_root))
    if T == 0:
        raise RuntimeError("Found zero scenarios. Run STRF generator first.")

    print(f"Detected {T} scenarios.")

    # ---------------------------------------------------------
    # Instantiate D* Lite
    # ---------------------------------------------------------
    dstar = DStarLite(
        graph_root=data_root,
        start=start_node,
        goal=goal_node,
        scenario_frames=str(data_root),
        mode=dstar_cfg["mode"],
        use_robust_init=dstar_cfg["use_robust_init"],
        robust_path_nodes=robust_path_nodes,
        change_interval_steps=dstar_cfg["change_interval_steps"],
        export_overlays=dstar_cfg["export_overlays"],
    )

    # ---------------------------------------------------------
    # Run D* Lite
    # ---------------------------------------------------------
    taken_path = dstar.run(max_steps=2000)

    print("\n=== D* Lite run completed ===")
    print("Path length:", len(taken_path))

    # ---------------------------------------------------------
    # Export path result
    # ---------------------------------------------------------
    out_path_file = dstar_root / "dstar_taken_path.csv"
    pd.DataFrame({"node_id": taken_path}).to_csv(out_path_file, index=False)
    print(f"Saved taken path -> {out_path_file}")

    # ---------------------------------------------------------
    # Export initialization path
    # ---------------------------------------------------------
    if robust_path_nodes is not None:
        init_path = robust_path_nodes.copy()
    else:
        # For now placeholder until you implement backwards A*
        init_path = [goal_node]

    init_file = dstar_root / "dstar_initialization_path.csv"
    pd.DataFrame({"node_id": init_path}).to_csv(init_file, index=False)
    print(f"Saved initialization path -> {init_file}")

    # ---------------------------------------------------------
    # Export overlays (if enabled)
    # ---------------------------------------------------------
    if dstar_cfg["export_overlays"]:
        print("\nGenerating overlays...")

        overlay_dir = dstar_root / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)

        nodes_df = pd.read_csv(data_root / "nodes.csv")

        overlay_paths = []

        for t in range(T):
            scen_dir = data_root / f"scenario_{t:03d}"
            #frame_png = os.path.join(scen_dir, "frame.png")
            field_path = scen_dir / "field.npy"
            out_png = overlay_dir / f"overlay_{t:03d}.png"
            export_overlay_image(field_path, taken_path, nodes_df, out_png)
            overlay_paths.append(out_png)

        # build animated GIF
        gif_path = dstar_root / "dstar_overlay_animation.gif"
        imgs = [imageio.imread(p) for p in overlay_paths]
        imageio.mimsave(gif_path, imgs, fps=2, loop=0)

        print(f"Saved overlays -> {overlay_dir}")
        print(f"Saved animated GIF -> {gif_path}")

    print("\nAll exports completed.")
    print("============================================")
    print("           D* Lite run finished")
    print("============================================")


# ---------------------------------------------------------
# Pipeline wrapper for API (FastAPI / NiceGUI)
# ---------------------------------------------------------
def run_dstar_pipeline(cfg):
    """
    Wrapper so the API can call the D* Lite run without using CLI main().
    cfg: loaded default.json dict
    """
    data_root   = Path(cfg["paths"]["data_root"]).expanduser()
    result_root = Path(cfg["paths"]["result_root"]).expanduser()
    seed = cfg.get("seed", "unknown")
    dstar_root = result_root / "DStar"

    # Ensure result directory
    dstar_root.mkdir(parents=True, exist_ok=True)

    # Robust warm-start
    robust_path_nodes = None
    if cfg["dstar_lite"]["use_robust_init"]:
        robust_path_nodes = load_robust_path(result_root)

    # Count scenarios
    T = count_scenarios(str(data_root))
    if T == 0:
        raise RuntimeError("Found zero scenarios. Run STRF generator first.")

    # Determine goal node
    start_node = cfg["robust_model"]["start_node"]
    goal_node  = cfg["robust_model"]["goal_node"]
    if goal_node == "auto":
        nodes_df = pd.read_csv(os.path.join(data_root, "nodes.csv"))
        goal_node = nodes_df["node_id"].max()

    # Instantiate D* Lite
    dstar = DStarLite(
        graph_root=data_root,
        start=start_node,
        goal=goal_node,
        scenario_frames=data_root,
        mode=cfg["dstar_lite"]["mode"],
        use_robust_init=cfg["dstar_lite"]["use_robust_init"],
        robust_path_nodes=robust_path_nodes,
        change_interval_steps=cfg["dstar_lite"]["change_interval_steps"],
        export_overlays=cfg["dstar_lite"]["export_overlays"],
    )

    taken_path = dstar.run(max_steps=2000)

    # Save results
    out_taken = dstar_root / "dstar_taken_path.csv"
    pd.DataFrame({"node_id": taken_path}).to_csv(out_taken, index=False)

    if robust_path_nodes is not None:
        init_path = robust_path_nodes.copy()
    else:
        init_path = [goal_node]

    out_init = dstar_root / "dstar_initialization_path.csv"
    pd.DataFrame({"node_id": init_path}).to_csv(out_init, index=False)

    # Overlays
    overlay_dir = None
    if cfg["dstar_lite"]["export_overlays"]:
        overlay_dir = dstar_root / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        nodes_df = pd.read_csv(data_root / "nodes.csv")

        overlay_paths = []
        for t in range(T):
            scen_dir = data_root / f"scenario_{t:03d}"
            #frame_png = os.path.join(scen_dir, "frame.png")
            field_path = scen_dir / "field.npy"
            out_png = overlay_dir / f"overlay_{t:03d}.png"
            export_overlay_image(field_path, taken_path, nodes_df, out_png)
            overlay_paths.append(out_png)

        gif_path = dstar_root / "dstar_overlay_animation.gif"
        imgs = [imageio.imread(p) for p in overlay_paths]
        imageio.mimsave(gif_path, imgs, fps=2)

    return {
        "taken_path": out_taken,
        "initialization_path": out_init,
        "overlays_dir": overlay_dir
    }


if __name__ == "__main__":
    main()
