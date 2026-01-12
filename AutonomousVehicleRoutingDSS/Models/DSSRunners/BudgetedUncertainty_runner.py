from pathlib import Path

from Models.BudgetedUncertainty import BudgetedUncertainty


def run_budgeted_uncertainty(cfg):
    base = Path(cfg["paths"]["data_root"]).expanduser()

    gamma_value = cfg["budgeted_model"].get("gamma_value")
    if gamma_value is None:
        gamma_value = cfg["budgeted_model"].get("gamma", 0.0)
    try:
        gamma_value = float(gamma_value)
    except Exception:
        pass

    try:
        time_limit = cfg["budgeted_model"].get("time_limit")
        time_limit = float(time_limit) if time_limit is not None else None
    except Exception:
        time_limit = None
    # Safety cap so runs do not hang forever; override via config if needed.
    if time_limit is None:
        time_limit = 180.0

    nominal_rule = str(cfg["budgeted_model"].get("nominal_rule", "min")).lower()
    if nominal_rule not in {"min", "avg"}:
        nominal_rule = "min"

    start_node = int(cfg["robust_model"].get("start_node", 0))
    goal_cfg = cfg["robust_model"].get("goal_node", "auto")
    goal_node = None if str(goal_cfg).lower() == "auto" else int(goal_cfg)

    print(
        f"[BudgetedRunner] Starting Budgeted Uncertainty | data_root={base} | "
        f"gamma={gamma_value} | nominal_rule={nominal_rule} | start={start_node} | "
        f"goal={goal_node or 'max'} | time_limit={time_limit}"
    )

    model_obj = BudgetedUncertainty(
        data_root=base,
        gamma=gamma_value,
        nominal_rule=nominal_rule,
        time_limit=time_limit,
    )
    model_obj.Preprocessing()
    print("[BudgetedRunner] Preprocessing finished.", flush=True)
    model_obj.OptimizationModel(start_node=start_node, goal_node=goal_node)
    print("[BudgetedRunner] Optimization complete, extracting results...", flush=True)

    model_obj.GetResultData()
    model_obj.ExportRobustPathCSV(base)
    export_overlays = cfg.get("visualizations", {}).get("export_overlays", True)
    if export_overlays:
        model_obj.ExportPathOverlay(base)
    model_obj.ExportResultsJSON(base)
    print("[BudgetedRunner] Finished exports.", flush=True)

    return model_obj.result
