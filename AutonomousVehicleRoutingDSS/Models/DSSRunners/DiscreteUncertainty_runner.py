from pathlib import Path
from Models.DiscreteUncertainty import Preprocessing, DiscreteUncertainty


def run_discrete_pipeline(cfg):
    base = Path(cfg["paths"]["data_root"]).expanduser()
    robust_cfg = cfg.get("robust_model", {})
    discrete_mode = robust_cfg.get("discrete_mode", "classic")
    adaptive_window = int(robust_cfg.get("adaptive_window", 1))
    adaptive_commit = robust_cfg.get("adaptive_commit", None)

    nodes = base / "nodes.csv"
    edges = base / "scenario_000" / "edges.csv"

    coords, node_ids, edges_set, costs, graph_paths = Preprocessing(
        str(nodes), str(edges), str(base)
    )

    model_obj = DiscreteUncertainty(coords, node_ids, edges_set, costs, graph_paths)
    model_obj.data_root = str(base)
    model_obj.start = int(robust_cfg.get("start_node", min(coords.keys())))
    goal_cfg = robust_cfg.get("goal_node", "auto")
    if str(goal_cfg).lower() == "auto":
        model_obj.goal = max(coords.keys())
    else:
        model_obj.goal = int(goal_cfg)

    if str(discrete_mode).lower() == "adaptive":
        # rolling-window adaptive solve
        model_obj.run_adaptive(
            window_size=adaptive_window,
            commit_length=adaptive_commit if adaptive_commit is None else int(adaptive_commit),
            start_node=model_obj.start,
            goal_node=model_obj.goal,
            log_to_console=1
        )
    else:
        # classic single-shot min-max
        model = model_obj.OptimizationModel()
        model_obj.GetResultData(model)

    model_obj.ExportRobustPathCSV(base)
    export_overlays = cfg.get("visualizations", {}).get("export_overlays", True)
    if export_overlays:
        model_obj.ExportPathOverlay(base)
    model_obj.ExportResultsJSON(base)

    return model_obj.result
