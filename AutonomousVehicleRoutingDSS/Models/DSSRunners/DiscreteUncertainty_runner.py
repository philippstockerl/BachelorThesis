from pathlib import Path
from Models.DiscreteUncertainty import Preprocessing, DiscreteUncertainty


def run_discrete_pipeline(cfg):
    base = Path(cfg["paths"]["data_root"]).expanduser()
    robust_cfg = cfg.get("robust_model", {})

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

    model = model_obj.OptimizationModel()
    model_obj.GetResultData(model)

    model_obj.ExportRobustPathCSV(base)
    export_overlays = cfg.get("visualizations", {}).get("export_overlays", True)
    if export_overlays:
        model_obj.ExportPathOverlay(base)
    model_obj.ExportResultsJSON(base)

    return model_obj.result
