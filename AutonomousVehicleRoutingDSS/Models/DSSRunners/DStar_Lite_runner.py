import json
from pathlib import Path
from Models.DStar_Lite import Preprocessing, DStarLite


def find_robust_path_file(base: Path, mode: str) -> Path | None:
    if mode == "discrete":
        p = base / "DiscreteUncertainty" / "discrete_uncertainty_path.csv"
        return p if p.exists() else None
    if mode == "discrete_adaptive":
        p = base / "DiscreteUncertaintyAdaptive" / "discrete_uncertainty_adaptive_path.csv"
        return p if p.exists() else None
    if mode == "budgeted":
        p = base / "BudgetedUncertainty" / "budgeted_uncertainty_path.csv"
        return p if p.exists() else None
    return None


def run_dstar_pipeline(cfg):
    base = Path(cfg["paths"]["data_root"]).expanduser()
    mode = cfg["dstar_lite"].get("warmstart_mode", "none")
    if mode == "lbu":  # legacy alias
        mode = "budgeted"
    robust_cfg = cfg.get("robust_model", {})
    start_node = int(robust_cfg.get("start_node", 0))
    goal_cfg = robust_cfg.get("goal_node", "auto")
    goal_node = None if str(goal_cfg).lower() == "auto" else int(goal_cfg)

    nodes = str(base / "nodes.csv")
    edges = str(base / "scenario_000" / "edges.csv")

    robust_file = None if mode == "none" else find_robust_path_file(base, mode)

    nodes_map, graph, pred, scen_changes, robust_list = Preprocessing(
        nodes, edges, str(base), robust_file
    )

    dstar_cfg = cfg.get("dstar_lite", {})
    beacon_cap = dstar_cfg.get("max_milestones", 10)
    debug = bool(dstar_cfg.get("debug", False))
    debug_stride = int(dstar_cfg.get("debug_stride", 0) or 0)
    max_steps = dstar_cfg.get("max_steps")
    dstar = DStarLite(
        nodes_map,
        graph,
        pred,
        scen_changes,
        robust_list,
        start_node=start_node,
        goal_node=goal_node,
        beacon_cap=beacon_cap,
        debug=debug,
        debug_stride=debug_stride,
        max_steps=max_steps,
    )
    dstar.DStarMain()

    # ----------------------------------------------------------
    # EXPORT OVERLAY GIF (optional)
    # ----------------------------------------------------------
    export_overlays = cfg.get("visualizations", {}).get("export_overlays", True)
    out_folder = None
    if export_overlays:
        out_folder = base / (
            "DStarLiteDiscreteUncertainty" if mode == "discrete"
            else "DStarLiteDiscreteAdaptiveUncertainty" if mode == "discrete_adaptive"
            else "DStarLiteBudgetedUncertainty" if mode == "budgeted"
            else "DStarLite"
        ) / "overlays"
        out_folder.mkdir(parents=True, exist_ok=True)

        # Call animation export
        dstar.ExportPathOverlay(str(base), out_folder)

    # ----------------------------------------------------------
    # CREATE RESULT JSON
    # ----------------------------------------------------------
    result_full = dstar.result
    result_full["data_root"] = str(base)
    if out_folder is not None:
        result_full["animated_overlay"] = str(out_folder)

    json_name = (
        "DStarLiteDiscreteUncertainty_result.json"
        if mode == "discrete"
        else "DStarLiteDiscreteAdaptiveUncertainty_result.json"
        if mode == "discrete_adaptive"
        else "DStarLiteBudgetedUncertainty_result.json"
        if mode == "budgeted"
        else "DStarLite_result.json"
    )

    comparison_path = base / "ComparisonData"
    comparison_path.mkdir(parents=True, exist_ok=True)
    json_path = comparison_path / json_name

    with open(json_path, "w") as jf:
        json.dump(result_full, jf, indent=2)


    return {
        "result": result_full
    }
