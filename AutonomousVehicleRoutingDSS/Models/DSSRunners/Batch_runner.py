from __future__ import annotations

import copy
import csv
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from Models.DSSRunners.BudgetedUncertainty_runner import run_budgeted_uncertainty
from Models.DSSRunners.DStar_Lite_runner import run_dstar_pipeline
from Models.DSSRunners.DiscreteUncertainty_runner import run_discrete_pipeline
from Models.DSSRunners.STGRF_runner import run_srf_generator


Runner = Callable[[Dict[str, Any]], Dict[str, Any]]


ALGORITHM_REGISTRY: Dict[str, Dict[str, Any]] = {
    "discrete": {
        "label": "Discrete Uncertainty",
        "runner": run_discrete_pipeline,
    },
    "budgeted": {
        "label": "Budgeted Uncertainty",
        "runner": run_budgeted_uncertainty,
    },
    "dstar": {
        "label": "DStar Lite",
        "runner": run_dstar_pipeline,
        "warmstart_mode": "none",
    },
    "dstar_discrete": {
        "label": "DStar Lite (Discrete)",
        "runner": run_dstar_pipeline,
        "warmstart_mode": "discrete",
    },
    "dstar_budgeted": {
        "label": "DStar Lite (Budgeted)",
        "runner": run_dstar_pipeline,
        "warmstart_mode": "budgeted",
    },
}


def _deep_update(target: Dict[str, Any], updates: Dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _as_tuple(value: Any, length: int = 3, default: float = 1.0) -> Tuple[float, ...]:
    if isinstance(value, (list, tuple)):
        seq = [float(v) for v in value] if value else [default]
    elif value is None:
        seq = [default]
    else:
        seq = [float(value)]

    if len(seq) < length:
        seq.extend([seq[-1]] * (length - len(seq)))
    return tuple(seq[:length])


def _unwrap_result(result: Any) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    if "result" in result and isinstance(result["result"], dict):
        return result["result"]
    return result


def _coerce_int_list(values: Any) -> List[int]:
    if values is None:
        return []
    if isinstance(values, list):
        return [int(v) for v in values]
    if isinstance(values, str):
        parts = [p.strip() for p in values.split(",") if p.strip()]
        return [int(p) for p in parts]
    return [int(values)]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _append_csv_row(path: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _compute_realized_cost(result: Dict[str, Any]) -> Optional[float]:
    if not result:
        return None
    step_costs = result.get("step_cost_details")
    if isinstance(step_costs, list) and step_costs:
        try:
            return float(sum(float(step["cost"]) for step in step_costs))
        except Exception:
            pass
    arc_costs = result.get("arc_costs_per_scenario")
    scenario_steps = result.get("scenario_per_step")
    if isinstance(arc_costs, list) and isinstance(scenario_steps, list):
        total = 0.0
        for arc_entry, scen in zip(arc_costs, scenario_steps):
            scenarios = arc_entry.get("scenarios", {})
            if scen in scenarios:
                total += float(scenarios[scen])
        return float(total)
    for key in ("cost", "total_cost"):
        if key in result:
            try:
                return float(result[key])
            except Exception:
                return None
    return None


def _extract_path_length(result: Dict[str, Any]) -> Optional[int]:
    path = result.get("node_path")
    if isinstance(path, list):
        return max(0, len(path) - 1)
    return None


def _boundary_edge_share(result: Dict[str, Any]) -> Optional[float]:
    path = result.get("node_path")
    if not isinstance(path, list):
        return None
    total_edges = max(0, len(path) - 1)
    if total_edges == 0:
        return 0.0

    coords = result.get("coords")
    if not isinstance(coords, dict) or not coords:
        return None

    points = []
    for value in coords.values():
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                points.append((float(value[0]), float(value[1])))
            except Exception:
                continue
    if not points:
        return None

    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    def lookup(node_id: Any) -> Optional[Tuple[float, float]]:
        if node_id in coords:
            value = coords.get(node_id)
        else:
            value = None
            if isinstance(node_id, str):
                try:
                    value = coords.get(int(node_id))
                except Exception:
                    value = None
            elif isinstance(node_id, int):
                value = coords.get(str(node_id))
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return None
        try:
            return float(value[0]), float(value[1])
        except Exception:
            return None

    def is_boundary(node_id: Any) -> bool:
        coord = lookup(node_id)
        if coord is None:
            return False
        x, y = coord
        return x in (min_x, max_x) or y in (min_y, max_y)

    boundary_edges = 0
    for u, v in zip(path[:-1], path[1:]):
        # Edge is counted if both endpoints lie on the boundary.
        if is_boundary(u) and is_boundary(v):
            boundary_edges += 1
    return boundary_edges / total_edges if total_edges else 0.0


def _extra_metrics(result: Dict[str, Any], algorithm: str) -> Dict[str, Any]:
    extra: Dict[str, Any] = {}
    if not result:
        return extra
    if "warmstart_used" in result:
        extra["warmstart_used"] = bool(result["warmstart_used"])
    if "beacon_hits" in result and isinstance(result["beacon_hits"], list):
        extra["beacon_hit_count"] = len(result["beacon_hits"])
    if "beacon_sequence" in result and isinstance(result["beacon_sequence"], list):
        extra["beacon_sequence_len"] = len(result["beacon_sequence"])
    if "replans" in result:
        extra["replans"] = result["replans"]
    if "nominal_cost" in result:
        extra["nominal_cost"] = result["nominal_cost"]
    if "robust_cost" in result:
        extra["robust_cost"] = result["robust_cost"]
    if "nominal_cost" in result and "robust_cost" in result:
        try:
            nominal = float(result["nominal_cost"])
            robust = float(result["robust_cost"])
            premium = robust - nominal
            extra["robustness_premium"] = premium
            if nominal != 0:
                extra["robustness_premium_ratio"] = premium / nominal
        except Exception:
            pass
    if "baseline_cost" in result:
        extra["baseline_cost"] = result["baseline_cost"]
    if "model_type" in result:
        extra["model_type"] = result["model_type"]
    boundary_share = _boundary_edge_share(result)
    if boundary_share is not None:
        extra["boundary_edge_share"] = boundary_share
    scen_steps = result.get("scenario_per_step")
    if isinstance(scen_steps, list) and scen_steps:
        switches = sum(1 for i in range(1, len(scen_steps)) if scen_steps[i] != scen_steps[i - 1])
        extra["scenario_switches"] = switches
    warm_expected = algorithm.startswith("dstar_") and algorithm != "dstar"
    if warm_expected:
        extra["warmstart_expected"] = True
    return extra


def _status_from_exception(exc: Exception) -> str:
    msg = str(exc).lower()
    if "infeasible" in msg or "no feasible" in msg:
        return "infeasible"
    if "timeout" in msg or "time limit" in msg:
        return "timeout"
    return "error"


def _apply_batch_overrides(base_cfg: Dict[str, Any], batch_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    for section in ("srf", "graph", "robust_model", "budgeted_model", "dstar_lite", "visualizations"):
        if isinstance(batch_cfg.get(section), dict):
            cfg.setdefault(section, {})
            _deep_update(cfg[section], batch_cfg[section])
    return cfg


def _build_algorithm_cfg(base_cfg: Dict[str, Any], algorithm: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    spec = ALGORITHM_REGISTRY.get(algorithm, {})
    if "warmstart_mode" in spec:
        cfg.setdefault("dstar_lite", {})["warmstart_mode"] = spec["warmstart_mode"]
    return cfg


def run_batch_experiments(
    base_cfg: Dict[str, Any],
    batch_cfg: Dict[str, Any],
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    def log(msg: str) -> None:
        if log_fn is None:
            print(msg, flush=True)
        else:
            log_fn(msg)

    config_id = str(batch_cfg.get("config_id", "batch_default")).strip() or "batch_default"
    experiments_root = Path(batch_cfg.get("experiments_root", "experiments")).expanduser().resolve()
    results_name = str(batch_cfg.get("results_csv", "batch_results.csv")).strip() or "batch_results.csv"

    config_root = experiments_root / config_id
    data_root_base = config_root / "data"
    results_path = Path(results_name)
    if not results_path.is_absolute():
        results_path = config_root / results_path

    algorithms = [str(a) for a in batch_cfg.get("algorithms", []) if str(a)]
    seeds = _coerce_int_list(batch_cfg.get("seeds", []))

    if not algorithms:
        raise ValueError("Batch config has no algorithms selected.")
    if not seeds:
        raise ValueError("Batch config has no seeds configured.")

    base_cfg = _apply_batch_overrides(base_cfg, batch_cfg)
    base_cfg.setdefault("visualizations", {})
    base_cfg["visualizations"]["export_metadata"] = True

    _write_json(config_root / "batch_snapshot.json", batch_cfg)
    _write_json(config_root / "config_resolved.json", base_cfg)

    fields = [
        "timestamp",
        "config_id",
        "seed",
        "algorithm",
        "status",
        "realized_cost",
        "path_length",
        "runtime_ms",
    ]

    summary = {
        "config_id": config_id,
        "results_csv": str(results_path),
        "runs_total": 0,
        "runs_success": 0,
        "runs_failed": 0,
    }

    log(f"[Batch] Starting batch run | config_id={config_id} | seeds={len(seeds)} | algos={len(algorithms)}")
    for seed in seeds:
        seed_cfg = copy.deepcopy(base_cfg)
        seed_cfg["seed"] = int(seed)
        seed_cfg.setdefault("paths", {})
        data_root = data_root_base / f"seed_{int(seed):04d}"
        seed_cfg["paths"]["data_root"] = str(data_root)
        seed_cfg.setdefault("visualizations", {})
        seed_cfg["visualizations"]["export_metadata"] = True

        _write_json(config_root / "configs" / f"{config_id}_seed_{int(seed):04d}.json", seed_cfg)

        log(f"[Batch] Generating STRF | seed={seed} | data_root={data_root}")
        try:
            run_srf_generator(seed_cfg)
        except Exception as exc:
            status = _status_from_exception(exc)
            tb = traceback.format_exc(limit=6)
            for algo in algorithms:
                algo_cfg = _build_algorithm_cfg(seed_cfg, algo)
                row = _build_row(
                    algo_cfg,
                    config_id,
                    algo,
                    result=None,
                    runtime_ms=0.0,
                    status=status,
                    error_message=str(exc),
                    error_traceback=tb,
                )
                _append_csv_row(results_path, fields, row)
                summary["runs_total"] += 1
                summary["runs_failed"] += 1
            log(f"[Batch] STRF generation failed for seed={seed}: {exc}")
            continue

        for algo in algorithms:
            spec = ALGORITHM_REGISTRY.get(algo)
            if spec is None:
                log(f"[Batch] Unknown algorithm '{algo}', recording error.")
                row = _build_row(
                    seed_cfg,
                    config_id,
                    algo,
                    result=None,
                    runtime_ms=0.0,
                    status="error",
                    error_message=f"Unknown algorithm '{algo}'",
                    error_traceback="",
                )
                _append_csv_row(results_path, fields, row)
                summary["runs_total"] += 1
                summary["runs_failed"] += 1
                continue
            runner: Runner = spec["runner"]
            algo_cfg = _build_algorithm_cfg(seed_cfg, algo)
            start = time.perf_counter()
            try:
                log(f"[Batch] Running {algo} | seed={seed}")
                raw = runner(algo_cfg)
                result = _unwrap_result(raw)
                runtime_ms = (time.perf_counter() - start) * 1000.0
                row = _build_row(
                    algo_cfg,
                    config_id,
                    algo,
                    result=result,
                    runtime_ms=runtime_ms,
                    status="success",
                    error_message="",
                    error_traceback="",
                )
                _append_csv_row(results_path, fields, row)
                summary["runs_total"] += 1
                summary["runs_success"] += 1
            except Exception as exc:
                runtime_ms = (time.perf_counter() - start) * 1000.0
                status = _status_from_exception(exc)
                row = _build_row(
                    algo_cfg,
                    config_id,
                    algo,
                    result=None,
                    runtime_ms=runtime_ms,
                    status=status,
                    error_message=str(exc),
                    error_traceback=traceback.format_exc(limit=6),
                )
                _append_csv_row(results_path, fields, row)
                summary["runs_total"] += 1
                summary["runs_failed"] += 1
                log(f"[Batch] {algo} failed for seed={seed}: {exc}")

    log(f"[Batch] Finished | success={summary['runs_success']} | failed={summary['runs_failed']}")
    return summary


def _build_row(
    cfg: Dict[str, Any],
    config_id: str,
    algorithm: str,
    result: Optional[Dict[str, Any]],
    runtime_ms: float,
    status: str,
    error_message: str,
    error_traceback: str,
) -> Dict[str, Any]:
    realized_cost = _compute_realized_cost(result or {})
    path_length = _extract_path_length(result or {})
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_id": config_id,
        "seed": cfg.get("seed"),
        "algorithm": algorithm,
        "status": status,
        "realized_cost": realized_cost,
        "path_length": path_length,
        "runtime_ms": runtime_ms,
    }
