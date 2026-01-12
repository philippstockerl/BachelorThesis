from __future__ import annotations


import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import hashlib


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

DEFAULT_BASE_CONFIG_PATH = Path(
    os.environ.get("DRDSS_CONFIG", APP_ROOT / "config" / "default.json")
)
DEFAULT_BATCH_CONFIG_PATH = Path(
    os.environ.get("DRDSS_BATCH_CONFIG", APP_ROOT / "config" / "batch.json")
)


REQUIRED_COLUMNS = [
    "config_id",
    "seed",
    "algorithm",
    "status",
    "realized_cost",
    "path_length",
    "runtime_ms",
]

DEFAULT_OK_STATUSES = {"ok", "success"}

DEFAULT_META_COLUMNS = [
    "grid_size",
    "cell_size",
    "num_scenarios",
    "kernel",
    "variance",
    "length_scale_x",
    "length_scale_y",
    "length_scale_t",
    "anis_x",
    "anis_y",
    "anis_t",
    "connectivity",
    "start_node",
    "goal_node",
    "gamma",
    "adaptive_window",
    "adaptive_commit",
    "max_milestones",
    "warmstart_mode",
]

DEFAULT_COLUMN_MAP = {
    "configuration_id": "config_id",
    "config": "config_id",
    "algorithm_name": "algorithm",
    "execution_status": "status",
    "runtime": "runtime_ms",
    "runtime_ms": "runtime_ms",
    "path_cost": "realized_cost",
    "cost": "realized_cost",
    "path_length_steps": "path_length",
}


def _seed_from_key(*parts: str, modulus: int = 2**32) -> int:
    """Derive a stable seed from string parts."""
    raw = "|".join(parts).encode("utf-8", errors="ignore")
    digest = hashlib.blake2b(raw, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") % modulus


def collect_csv_files(
    paths: Sequence[str | Path],
    recursive: bool = True,
    exclude_prefixes: Sequence[str] = ("summary_", "time_resolved_"),
) -> List[Path]:
    """Collect CSV files from files or directories."""
    files: List[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_file() and path.suffix.lower() == ".csv":
            files.append(path)
            continue
        if path.is_dir():
            pattern = "**/*.csv" if recursive else "*.csv"
            for fp in path.glob(pattern):
                if fp.name.startswith(tuple(exclude_prefixes)):
                    continue
                files.append(fp)
    return sorted(set(files))


def load_results(
    paths: Sequence[str | Path],
    column_map: Optional[Dict[str, str]] = None,
    add_source_column: bool = False,
) -> pd.DataFrame:
    """Load one or more CSV files into a single DataFrame."""
    files = collect_csv_files(paths)
    if not files:
        raise FileNotFoundError("No CSV files found for aggregation.")
    frames: List[pd.DataFrame] = []
    for fp in files:
        df = pd.read_csv(fp)
        if add_source_column:
            df["source_file"] = str(fp)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    return standardize_columns(data, column_map or {})


def standardize_columns(
    df: pd.DataFrame,
    column_map: Dict[str, str],
) -> pd.DataFrame:
    """Rename columns using a provided mapping and known defaults."""
    mapping = dict(DEFAULT_COLUMN_MAP)
    mapping.update(column_map)
    columns = {col: mapping[col] for col in df.columns if col in mapping}
    return df.rename(columns=columns)


def validate_required_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def expand_extra_metrics(
    df: pd.DataFrame,
    column: str = "extra_metrics",
    prefix: str = "metric_",
) -> pd.DataFrame:
    """Expand a JSON column of extra metrics into prefixed columns."""
    if column not in df.columns:
        return df
    parsed: List[Dict[str, Any]] = []
    for raw in df[column].fillna(""):
        if isinstance(raw, dict):
            parsed.append(raw)
            continue
        if not raw:
            parsed.append({})
            continue
        try:
            parsed.append(json.loads(raw))
        except Exception:
            parsed.append({})
    metrics = pd.json_normalize(parsed)
    if metrics.empty:
        return df
    metrics = metrics.add_prefix(prefix)
    return pd.concat([df.reset_index(drop=True), metrics.reset_index(drop=True)], axis=1)


def clean_results(
    df: pd.DataFrame,
    required_columns: Sequence[str] = REQUIRED_COLUMNS,
    status_column: str = "status",
) -> pd.DataFrame:
    """Drop rows with malformed required fields and coerce numeric columns."""
    validate_required_columns(df, required_columns)
    cleaned = df.copy()

    for col in ("seed", "realized_cost", "path_length", "runtime_ms"):
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned[status_column] = cleaned[status_column].astype(str)
    cleaned["algorithm"] = cleaned["algorithm"].astype(str)
    cleaned["config_id"] = cleaned["config_id"].astype(str)

    cleaned = cleaned.dropna(subset=required_columns)
    cleaned = cleaned[cleaned["algorithm"].str.strip() != ""]
    cleaned = cleaned[cleaned["config_id"].str.strip() != ""]
    return cleaned.reset_index(drop=True)


def filter_paired_seeds(
    df: pd.DataFrame,
    config_column: str = "config_id",
    algorithm_column: str = "algorithm",
    seed_column: str = "seed",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Keep only seeds that contain all algorithms within each configuration."""
    keep_frames: List[pd.DataFrame] = []
    stats: Dict[str, Dict[str, Any]] = {}
    for config_id, group in df.groupby(config_column):
        algos = sorted(group[algorithm_column].unique())
        if not algos:
            continue
        counts = group.groupby(seed_column)[algorithm_column].nunique()
        keep_seeds = counts[counts == len(algos)].index
        filtered = group[group[seed_column].isin(keep_seeds)]
        stats[str(config_id)] = {
            "algorithms": algos,
            "seeds_total": int(counts.size),
            "seeds_used": int(len(keep_seeds)),
        }
        keep_frames.append(filtered)
    if not keep_frames:
        return df.iloc[0:0].copy(), stats
    return pd.concat(keep_frames, ignore_index=True), stats


def bootstrap_ci(
    values: Sequence[float],
    stat_func=np.mean,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Non-parametric bootstrap confidence interval for a statistic."""
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size < 2:
        return float("nan"), float("nan")
    rng = rng or np.random.default_rng(12345)
    idx = rng.integers(0, data.size, size=(n_boot, data.size))
    samples = data[idx]
    try:
        stats = stat_func(samples, axis=1)
    except Exception:
        stats = np.array([stat_func(sample) for sample in samples], dtype=float)
    lower = float(np.quantile(stats, alpha / 2))
    upper = float(np.quantile(stats, 1 - alpha / 2))
    return lower, upper


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def aggregate_summary(
    df: pd.DataFrame,
    config_column: str = "config_id",
    algorithm_column: str = "algorithm",
    seed_column: str = "seed",
    status_column: str = "status",
    ok_statuses: Sequence[str] = DEFAULT_OK_STATUSES,
    require_paired: bool = True,
    cost_column: str = "realized_cost",
    path_column: str = "path_length",
    runtime_column: str = "runtime_ms",
    runtime_quantile: float = 0.95,
    include_log_runtime: bool = True,
    log_base: float = 10.0,
    n_boot: int = 1000,
    ci_alpha: float = 0.05,
    cost_quantile: float = 0.9,
    algorithm_order: Optional[Sequence[str]] = None,
    meta_columns: Optional[Sequence[str]] = None,
    metric_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Aggregate median cost/runtime/path length and cost CI per configuration and algorithm."""
    df = df.copy()
    df[status_column] = df[status_column].astype(str).str.lower()
    ok_set = {s.lower() for s in ok_statuses}
    ok_mask = df[status_column].isin(ok_set)
    ok_df = df[ok_mask].copy()

    paired_df = ok_df
    pairing_stats = {}
    if require_paired:
        paired_df, pairing_stats = filter_paired_seeds(
            ok_df,
            config_column=config_column,
            algorithm_column=algorithm_column,
            seed_column=seed_column,
        )

    summary_rows: List[Dict[str, Any]] = []

    configs = sorted(df[config_column].dropna().unique())
    for config_id in configs:
        config_df = df[df[config_column] == config_id]
        algos = list(config_df[algorithm_column].dropna().unique())
        if algorithm_order:
            ordered_algos = [a for a in algorithm_order if a in algos]
            ordered_algos += [a for a in algos if a not in ordered_algos]
        else:
            ordered_algos = sorted(algos)

        for algorithm in ordered_algos:
            group = config_df[config_df[algorithm_column] == algorithm]
            total_runs = int(len(group))
            ok_group = group[group[status_column].isin(ok_set)]
            failed_runs = int(total_runs - len(ok_group))
            failure_rate = float(failed_runs / total_runs) if total_runs else float("nan")
            timeout_runs = int((group[status_column] == "timeout").sum())
            success_rate = float(len(ok_group) / total_runs) if total_runs else float("nan")
            timeout_rate = float(timeout_runs / total_runs) if total_runs else float("nan")

            used_group = paired_df[
                (paired_df[config_column] == config_id)
                & (paired_df[algorithm_column] == algorithm)
            ]
            if used_group.empty:
                used_group = ok_group if not require_paired else used_group

            cost_values = _safe_numeric(used_group.get(cost_column, pd.Series(dtype=float))).dropna()
            path_values = _safe_numeric(used_group.get(path_column, pd.Series(dtype=float))).dropna()
            runtime_values = _safe_numeric(used_group.get(runtime_column, pd.Series(dtype=float))).dropna()

            seed = _seed_from_key(str(config_id), str(algorithm))
            rng = np.random.default_rng(seed)
            cost_ci_low, cost_ci_high = bootstrap_ci(
                cost_values.values,
                stat_func=np.median,
                n_boot=n_boot,
                alpha=ci_alpha,
                rng=rng,
            )

            row: Dict[str, Any] = {
                config_column: config_id,
                algorithm_column: algorithm,
                "n_seeds": int(used_group[seed_column].nunique()) if not used_group.empty else 0,
                "cost_median": float(cost_values.median()) if not cost_values.empty else float("nan"),
                "cost_ci_low": cost_ci_low,
                "cost_ci_high": cost_ci_high,
                "runtime_median_ms": float(runtime_values.median()) if not runtime_values.empty else float("nan"),
                "path_length_median": float(path_values.median()) if not path_values.empty else float("nan"),
            }

            summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def aggregate_time_resolved(
    df: pd.DataFrame,
    time_column: str,
    config_column: str = "config_id",
    algorithm_column: str = "algorithm",
    seed_column: str = "seed",
    status_column: str = "status",
    ok_statuses: Sequence[str] = DEFAULT_OK_STATUSES,
    require_paired: bool = True,
    cost_column: str = "realized_cost",
    n_boot: int = 1000,
    ci_alpha: float = 0.05,
    algorithm_order: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Aggregate per-time-step metrics if a time column is present."""
    if time_column not in df.columns:
        return pd.DataFrame()

    data = df.copy()
    data[status_column] = data[status_column].astype(str).str.lower()
    ok_set = {s.lower() for s in ok_statuses}
    ok_df = data[data[status_column].isin(ok_set)].copy()

    if require_paired:
        ok_df, _ = filter_paired_seeds(
            ok_df,
            config_column=config_column,
            algorithm_column=algorithm_column,
            seed_column=seed_column,
        )

    ok_df[time_column] = pd.to_numeric(ok_df[time_column], errors="coerce")
    ok_df[cost_column] = pd.to_numeric(ok_df[cost_column], errors="coerce")
    ok_df = ok_df.dropna(subset=[time_column, cost_column])

    rows: List[Dict[str, Any]] = []
    configs = sorted(ok_df[config_column].dropna().unique())
    for config_id in configs:
        config_df = ok_df[ok_df[config_column] == config_id]
        algos = list(config_df[algorithm_column].dropna().unique())
        if algorithm_order:
            ordered_algos = [a for a in algorithm_order if a in algos]
            ordered_algos += [a for a in algos if a not in ordered_algos]
        else:
            ordered_algos = sorted(algos)
        for algorithm in ordered_algos:
            algo_df = config_df[config_df[algorithm_column] == algorithm]
            for time_step, group in algo_df.groupby(time_column):
                values = group[cost_column].values
                seed = _seed_from_key(str(config_id), str(algorithm), str(time_step))
                rng = np.random.default_rng(seed)
                ci_low, ci_high = bootstrap_ci(
                    values,
                    stat_func=np.mean,
                    n_boot=n_boot,
                    alpha=ci_alpha,
                    rng=rng,
                )
                rows.append(
                    {
                        config_column: config_id,
                        algorithm_column: algorithm,
                        time_column: time_step,
                        "cost_mean": float(np.mean(values)) if values.size else float("nan"),
                        "cost_median": float(np.median(values)) if values.size else float("nan"),
                        "cost_ci_low": ci_low,
                        "cost_ci_high": ci_high,
                        "runs_used": int(len(values)),
                    }
                )
    return pd.DataFrame(rows)


def run_aggregation(
    paths: Sequence[str | Path],
    output_dir: Optional[str | Path] = None,
    summary_name: str = "summary.csv",
    time_name: str = "time_resolved.csv",
    time_column: Optional[str] = None,
    ok_statuses: Sequence[str] = DEFAULT_OK_STATUSES,
    require_paired: bool = True,
    n_boot: int = 1000,
    ci_alpha: float = 0.05,
    cost_quantile: float = 0.9,
    algorithm_order: Optional[Sequence[str]] = None,
    include_log_runtime: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load, clean, aggregate, and optionally export summary/time-resolved CSVs.
    Returns (summary_df, time_df).
    """
    df = load_results(paths)
    df = expand_extra_metrics(df)
    df = clean_results(df)

    summary_df = aggregate_summary(
        df,
        ok_statuses=ok_statuses,
        require_paired=require_paired,
        n_boot=n_boot,
        ci_alpha=ci_alpha,
        cost_quantile=cost_quantile,
        algorithm_order=algorithm_order,
        include_log_runtime=include_log_runtime,
    )

    time_df = None
    if time_column:
        time_df = aggregate_time_resolved(
            df,
            time_column=time_column,
            ok_statuses=ok_statuses,
            require_paired=require_paired,
            n_boot=n_boot,
            ci_alpha=ci_alpha,
            algorithm_order=algorithm_order,
        )

    if output_dir:
        out_dir = Path(output_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_dir / summary_name, index=False)
        if time_df is not None and not time_df.empty:
            time_df.to_csv(out_dir / time_name, index=False)

    return summary_df, time_df


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for plotting.") from exc
    return plt


def _ensure_output_dir(path: Optional[str | Path], summary_path: Path) -> Path:
    if path:
        out_dir = Path(path).expanduser()
    else:
        out_dir = summary_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _algorithm_order(df: pd.DataFrame, preferred: Optional[Sequence[str]]) -> List[str]:
    algos = list(df["algorithm"].dropna().astype(str).unique())
    if preferred:
        order = [a for a in preferred if a in algos]
        order += [a for a in algos if a not in order]
        return order
    return sorted(algos)


def _numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy()


def _plot_bar_with_ci(
    group: pd.DataFrame,
    value_col: str,
    ci_low_col: Optional[str],
    ci_high_col: Optional[str],
    ylabel: str,
    title: str,
    output_path: Path,
    algorithm_order: Optional[Sequence[str]] = None,
) -> None:
    if group.empty:
        return
    plt = _require_matplotlib()
    order = _algorithm_order(group, algorithm_order)
    plot_df = group.set_index("algorithm").reindex(order).reset_index()
    values = _numeric(plot_df[value_col])
    labels = plot_df["algorithm"].astype(str).tolist()

    yerr = None
    if ci_low_col and ci_high_col and ci_low_col in plot_df.columns and ci_high_col in plot_df.columns:
        lows = _numeric(plot_df[ci_low_col])
        highs = _numeric(plot_df[ci_high_col])
        err_low = values - lows
        err_high = highs - values
        err_low = np.where(np.isfinite(err_low), err_low, 0.0)
        err_high = np.where(np.isfinite(err_high), err_high, 0.0)
        yerr = np.vstack([err_low, err_high])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=yerr, capsize=4, color="#4C78A8")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_bar(
    group: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
    algorithm_order: Optional[Sequence[str]] = None,
) -> None:
    if group.empty:
        return
    plt = _require_matplotlib()
    order = _algorithm_order(group, algorithm_order)
    plot_df = group.set_index("algorithm").reindex(order).reset_index()
    values = _numeric(plot_df[value_col])
    labels = plot_df["algorithm"].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, values, color="#F58518")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_summary(
    summary_df: pd.DataFrame,
    output_dir: Path,
    config_filter: Optional[str] = None,
    algorithm_order: Optional[Sequence[str]] = None,
    runtime_stat: str = "q",
    file_ext: str = "png",
) -> None:
    df = summary_df.copy()
    if config_filter:
        df = df[df["config_id"] == config_filter]

    for config_id, group in df.groupby("config_id"):
        runtime_map = {
            "mean": "runtime_mean_ms",
            "median": "runtime_median_ms",
            "q": "runtime_q_ms",
        }
        runtime_col = runtime_map.get(runtime_stat, "runtime_median_ms")
        if runtime_col not in group.columns:
            runtime_col = "runtime_median_ms"
            runtime_stat = "median"

        _plot_bar_with_ci(
            group,
            value_col="cost_median",
            ci_low_col="cost_ci_low",
            ci_high_col="cost_ci_high",
            ylabel="Realized Cost (median)",
            title=f"Cost median with 95% CI | {config_id}",
            output_path=output_dir / f"cost_ci_{config_id}.{file_ext}",
            algorithm_order=algorithm_order,
        )

        _plot_bar(
            group,
            value_col=runtime_col,
            ylabel=f"Runtime (ms, {runtime_stat})",
            title=f"Runtime ({runtime_stat}) | {config_id}",
            output_path=output_dir / f"runtime_{runtime_stat}_{config_id}.{file_ext}",
            algorithm_order=algorithm_order,
        )

        if "failure_rate" in group.columns:
            failure = group.copy()
            failure["failure_rate_pct"] = _numeric(failure["failure_rate"]) * 100.0
            _plot_bar(
                failure,
                value_col="failure_rate_pct",
                ylabel="Failure Rate (%)",
                title=f"Failure Rate | {config_id}",
                output_path=output_dir / f"failure_rate_{config_id}.{file_ext}",
                algorithm_order=algorithm_order,
            )


def plot_paired_deltas(
    raw_paths: Sequence[str | Path],
    output_dir: Path,
    baseline_algorithm: str,
    metric: str = "realized_cost",
    config_filter: Optional[str] = None,
    ok_statuses: Sequence[str] = ("success", "ok"),
    file_ext: str = "png",
) -> None:
    df = load_results(raw_paths)
    df = clean_results(df)
    df["status"] = df["status"].astype(str).str.lower()
    ok_set = {s.lower() for s in ok_statuses}
    df = df[df["status"].isin(ok_set)]

    if config_filter:
        df = df[df["config_id"] == config_filter]

    df, _ = filter_paired_seeds(df)

    if metric not in df.columns:
        return

    plt = _require_matplotlib()
    for config_id, group in df.groupby("config_id"):
        pivot = group.pivot_table(
            index="seed",
            columns="algorithm",
            values=metric,
            aggfunc="first",
        )
        if baseline_algorithm not in pivot.columns:
            continue
        deltas = pivot.subtract(pivot[baseline_algorithm], axis=0)
        deltas = deltas.drop(columns=[baseline_algorithm], errors="ignore")
        if deltas.empty:
            continue

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.boxplot(
            [deltas[col].dropna().values for col in deltas.columns],
            labels=[str(c) for c in deltas.columns],
            showmeans=True,
        )
        ax.axhline(0.0, color="#999999", linewidth=1.0, linestyle="--")
        ax.set_ylabel(f"{metric} delta vs {baseline_algorithm}")
        ax.set_title(f"Paired deltas | {config_id}")
        ax.set_xticklabels([str(c) for c in deltas.columns], rotation=25, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(output_dir / f"paired_deltas_{config_id}.{file_ext}", dpi=200)
        plt.close(fig)


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _load_json_or_dict(
    value: Dict[str, Any] | str | Path | None,
    default_path: Path,
) -> Dict[str, Any]:
    if value is None:
        return load_json(default_path)
    if isinstance(value, (str, Path)):
        return load_json(value)
    if isinstance(value, dict):
        return value
    raise TypeError("Expected dict or path for config input.")


def default_summary_path(batch_cfg: Dict[str, Any]) -> Path:
    root = Path(batch_cfg.get("experiments_root", APP_ROOT / "experiments")).expanduser().resolve()
    config_id = str(batch_cfg.get("config_id", "batch_default")).strip() or "batch_default"
    return root / config_id / "batch_summary.json"


def default_summary_dir(batch_cfg: Dict[str, Any], results_csv: str | None) -> Path:
    cfg_dir = batch_cfg.get("summary_dir")
    if cfg_dir:
        return Path(cfg_dir).expanduser()
    if results_csv:
        return Path(results_csv).expanduser().resolve().parent / "summary"
    root = Path(batch_cfg.get("experiments_root", APP_ROOT / "experiments")).expanduser().resolve()
    config_id = str(batch_cfg.get("config_id", "batch_default")).strip() or "batch_default"
    return root / config_id / "summary"


class _LogWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        stamp = datetime.now(timezone.utc).isoformat()
        self._fh.write(f"[{stamp}] {msg}\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def run_batch_pipeline(
    base_cfg: Dict[str, Any] | str | Path | None = None,
    batch_cfg: Dict[str, Any] | str | Path | None = None,
    summary_out: Optional[str | Path] = None,
    log_path: Optional[str | Path] = None,
    plot: bool = True,
) -> Dict[str, Any]:
    """
    Run batch experiments, aggregate results, and optionally generate plots.

    Config paths default to `config/default.json` and `config/batch.json`.
    """
    from Models.DSSRunners.Batch_runner import run_batch_experiments

    base_cfg_obj = _load_json_or_dict(base_cfg, DEFAULT_BASE_CONFIG_PATH)
    batch_cfg_obj = _load_json_or_dict(batch_cfg, DEFAULT_BATCH_CONFIG_PATH)

    logger = _LogWriter(Path(log_path).expanduser()) if log_path else None
    summary: Dict[str, Any] = {}
    try:
        summary = run_batch_experiments(base_cfg_obj, batch_cfg_obj, log_fn=logger)
        results_csv = summary.get("results_csv")
        algo_order = [str(a) for a in batch_cfg_obj.get("algorithms", []) if str(a)]
        summary_dir = default_summary_dir(batch_cfg_obj, results_csv)
        summary["summary_dir"] = str(summary_dir)
        if log_path:
            summary["log_path"] = str(Path(log_path).expanduser())

        if results_csv and Path(results_csv).expanduser().exists():
            try:
                summary_df, time_df = run_aggregation(
                    [results_csv],
                    output_dir=summary_dir,
                    algorithm_order=algo_order or None,
                )
                summary_csv = summary_dir / "summary.csv"
                if summary_csv.exists():
                    summary["summary_csv"] = str(summary_csv)
                if time_df is not None and not time_df.empty:
                    time_csv = summary_dir / "time_resolved.csv"
                    if time_csv.exists():
                        summary["time_csv"] = str(time_csv)
                if plot:
                    plot_output_dir = Path(results_csv).expanduser().resolve().parent
                    try:
                        summary_csv = summary_dir / "summary.csv"
                        if summary_csv.exists():
                            from analysis.plot_experiment_results import run_all_plots

                            run_all_plots(
                                summary_csv,
                                output_dir=plot_output_dir,
                                raw_csv=results_csv,
                            )
                            summary["plots_dir"] = str(plot_output_dir)
                    except RuntimeError as exc:
                        summary["experiment_plot_error"] = str(exc)
                    except Exception as exc:
                        summary["experiment_plot_error"] = str(exc)
            except Exception as exc:
                summary["aggregation_error"] = str(exc)
        else:
            summary["aggregation_error"] = "results_csv missing or not found"
    except Exception as exc:
        summary = {
            "status": "error",
            "error": str(exc),
        }
        if logger:
            logger(f"[Batch] Fatal error: {exc}")
    finally:
        if logger:
            logger.close()

    summary_path = Path(summary_out).expanduser() if summary_out else default_summary_path(batch_cfg_obj)
    write_json(summary_path, summary)
    return summary
