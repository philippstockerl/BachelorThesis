from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for plotting.") from exc
    return plt


def set_plot_style() -> None:
    plt = _require_matplotlib()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "TeX Gyre Termes",
                "STIXGeneral",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value))
    cleaned = cleaned.strip("_")
    return cleaned or "unnamed"


def _format_algorithm_label(algo: str) -> str:
    text = str(algo).replace("_", " ").strip()
    if not text:
        return str(algo)
    return " ".join(word[:1].upper() + word[1:] for word in text.split())


def _save_figure(fig, output_dir: Path, filename_base: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{filename_base}.png"
    pdf_path = output_dir / f"{filename_base}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")


def _algorithm_order(df: pd.DataFrame, preferred: Optional[Sequence[str]] = None) -> List[str]:
    algos = list(df["algorithm"].dropna().astype(str).unique())
    if preferred:
        order = [a for a in preferred if a in algos]
        order += [a for a in algos if a not in order]
        return order
    return sorted(algos)


def _color_map(algorithms: Sequence[str], cmap_name: str = "tab10") -> Dict[str, Any]:
    plt = _require_matplotlib()
    cmap = plt.get_cmap(cmap_name)
    colors = {}
    for idx, algo in enumerate(algorithms):
        colors[algo] = cmap(idx % cmap.N)
    return colors


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _resolve_raw_csv(summary_path: Path, raw_csv: Optional[str | Path]) -> Optional[Path]:
    if raw_csv:
        candidate = Path(raw_csv).expanduser()
        return candidate if candidate.exists() else None
    parent = summary_path.parent.parent if summary_path.parent else None
    if parent and parent.exists():
        preferred = parent / "batch_results.csv"
        if preferred.exists():
            return preferred
        for fp in parent.glob("*.csv"):
            if fp.name in {"summary.csv", "time_resolved.csv"}:
                continue
            return fp
    return None


def _load_raw_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    column_map = {
        "configuration_id": "config_id",
        "config": "config_id",
        "algorithm_name": "algorithm",
        "execution_status": "status",
        "runtime": "runtime_ms",
        "path_cost": "realized_cost",
        "cost": "realized_cost",
        "path_length_steps": "path_length",
    }
    rename = {col: column_map[col] for col in df.columns if col in column_map}
    if rename:
        df = df.rename(columns=rename)
    return df


def _parse_list_field(value: Any) -> Optional[List[float]]:
    if isinstance(value, list):
        return [float(v) for v in value if _is_number(v)]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except Exception:
            return None
        if isinstance(parsed, list):
            return [float(v) for v in parsed if _is_number(v)]
    return None


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def _extract_samples_by_algorithm(
    group: pd.DataFrame,
    column_candidates: Sequence[str],
) -> Tuple[Optional[str], Dict[str, List[float]]]:
    for col in column_candidates:
        if col not in group.columns:
            continue
        samples: Dict[str, List[float]] = {}
        for _, row in group.iterrows():
            algo = str(row.get("algorithm", "")).strip()
            if not algo:
                continue
            values = _parse_list_field(row.get(col))
            if values:
                samples[algo] = values
        if samples:
            return col, samples
    return None, {}


def plot_cost_ci_dot(
    summary_df: pd.DataFrame,
    output_dir: Path,
    algorithm_order: Optional[Sequence[str]] = None,
    colors: Optional[Dict[str, Any]] = None,
) -> None:
    if "cost_median" not in summary_df.columns:
        print("[plot] cost_median missing; skipping cost CI dot plots.")
        return
    summary_df = _coerce_numeric(summary_df, ["cost_median", "cost_ci_low", "cost_ci_high"])

    for config_id, group in summary_df.groupby("config_id"):
        group = group.dropna(subset=["cost_median"])
        if group.empty:
            continue
        order = _algorithm_order(group, algorithm_order)
        colors = colors or _color_map(order)
        fig, ax = _require_matplotlib().subplots(figsize=(7.5, 4.0))
        for idx, algo in enumerate(order):
            row = group[group["algorithm"] == algo]
            if row.empty:
                continue
            y = float(row["cost_median"].iloc[0])
            low = row.get("cost_ci_low", pd.Series([math.nan])).iloc[0]
            high = row.get("cost_ci_high", pd.Series([math.nan])).iloc[0]
            if _is_number(low) and _is_number(high):
                err_low = max(0.0, y - float(low))
                err_high = max(0.0, float(high) - y)
                yerr = [[err_low], [err_high]]
            else:
                yerr = None
            ax.errorbar(
                idx,
                y,
                yerr=yerr,
                fmt="o",
                color=colors.get(algo),
                ecolor=colors.get(algo),
                capsize=3,
            )

        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=25, ha="right")
        ax.set_ylabel("Cost")
        ax.set_title(f"Cost Median with 95% CI | {config_id}")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        filename = f"{_safe_name(config_id)}_cost_median_ci_dot"
        _save_figure(fig, output_dir, filename)
        _require_matplotlib().close(fig)


def plot_cost_distribution(
    summary_df: pd.DataFrame,
    output_dir: Path,
    algorithm_order: Optional[Sequence[str]] = None,
    colors: Optional[Dict[str, Any]] = None,
) -> None:
    column, samples = _extract_samples_by_algorithm(
        summary_df,
        ["cost_values", "cost_samples", "realized_cost_samples"],
    )
    if not samples:
        print("[plot] No per-seed cost samples in summary; skipping cost distribution plots.")
        return

    for config_id, group in summary_df.groupby("config_id"):
        _, per_algo = _extract_samples_by_algorithm(
            group,
            [column] if column else ["cost_values", "cost_samples", "realized_cost_samples"],
        )
        if not per_algo:
            continue
        order = _algorithm_order(group, algorithm_order)
        colors = colors or _color_map(order)
        data = [per_algo.get(algo, []) for algo in order if algo in per_algo]
        labels = [algo for algo in order if algo in per_algo]
        if not data:
            continue
        fig, ax = _require_matplotlib().subplots(figsize=(7.5, 4.0))
        box = ax.boxplot(data, labels=labels, showmeans=True, patch_artist=True)
        for patch, algo in zip(box["boxes"], labels):
            patch.set_facecolor(colors.get(algo))
            patch.set_alpha(0.6)
        ax.set_ylabel("Total cost")
        ax.set_title(f"Cost Distribution | {config_id}")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        filename = f"{_safe_name(config_id)}_cost_distribution_box"
        _save_figure(fig, output_dir, filename)
        _require_matplotlib().close(fig)


def plot_cost_distribution_raw(
    raw_df: pd.DataFrame,
    output_dir: Path,
    algorithm_order: Optional[Sequence[str]] = None,
    colors: Optional[Dict[str, Any]] = None,
    ok_statuses: Sequence[str] = ("success", "ok"),
) -> None:
    if "realized_cost" not in raw_df.columns:
        print("[plot] realized_cost missing; skipping cost distribution plots.")
        return

    df = raw_df.copy()
    if "status" in df.columns:
        ok_set = {s.lower() for s in ok_statuses}
        df["status"] = df["status"].astype(str).str.lower()
        df = df[df["status"].isin(ok_set)]

    df = _coerce_numeric(df, ["realized_cost"])
    df = df.dropna(subset=["realized_cost"])
    if df.empty:
        print("[plot] No valid cost data for distribution plots.")
        return

    for config_id, group in df.groupby("config_id"):
        median_costs = (
            group.groupby("algorithm")["realized_cost"].median().sort_values()
        )
        order = median_costs.index.tolist()
        colors = colors or _color_map(order)
        data = [group[group["algorithm"] == algo]["realized_cost"].values for algo in order]
        labels = [algo for algo in order]
        if not any(len(values) for values in data):
            continue
        fig, ax = _require_matplotlib().subplots(figsize=(7.5, 4.0))
        flierprops = {
            "markersize": 2,
            "markerfacecolor": "none",
            "markeredgecolor": "#555555",
            "alpha": 0.6,
        }
        box = ax.boxplot(
            data,
            labels=labels,
            showmeans=True,
            patch_artist=True,
            flierprops=flierprops,
        )
        for patch, algo in zip(box["boxes"], labels):
            patch.set_facecolor(colors.get(algo))
            patch.set_alpha(0.6)
        ax.set_ylabel("Total cost")
        ax.set_title(f"Cost Distribution | {config_id}")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        filename = f"{_safe_name(config_id)}_cost_distribution_box"
        _save_figure(fig, output_dir, filename)
        _require_matplotlib().close(fig)


def plot_runtime_distribution(
    summary_df: pd.DataFrame,
    output_dir: Path,
    algorithm_order: Optional[Sequence[str]] = None,
    colors: Optional[Dict[str, Any]] = None,
) -> None:
    column, samples = _extract_samples_by_algorithm(
        summary_df,
        ["runtime_values", "runtime_samples", "runtime_ms_samples"],
    )
    if not samples:
        print("[plot] No per-seed runtime samples in summary; skipping runtime distribution plots.")
        return

    for config_id, group in summary_df.groupby("config_id"):
        _, per_algo = _extract_samples_by_algorithm(
            group,
            [column] if column else ["runtime_values", "runtime_samples", "runtime_ms_samples"],
        )
        if not per_algo:
            continue
        order = _algorithm_order(group, algorithm_order)
        colors = colors or _color_map(order)
        data = [per_algo.get(algo, []) for algo in order if algo in per_algo]
        labels = [algo for algo in order if algo in per_algo]
        if not data:
            continue
        fig, ax = _require_matplotlib().subplots(figsize=(7.5, 4.0))
        box = ax.boxplot(data, labels=labels, showmeans=True, patch_artist=True)
        for patch, algo in zip(box["boxes"], labels):
            patch.set_facecolor(colors.get(algo))
            patch.set_alpha(0.6)
        ax.set_ylabel("Runtime (ms)")
        ax.set_yscale("log")
        ax.set_title(f"Runtime Distribution (log) | {config_id}")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        filename = f"{_safe_name(config_id)}_runtime_distribution_box_log"
        _save_figure(fig, output_dir, filename)
        _require_matplotlib().close(fig)


def plot_runtime_median(
    summary_df: pd.DataFrame,
    output_dir: Path,
    algorithm_order: Optional[Sequence[str]] = None,
    colors: Optional[Dict[str, Any]] = None,
) -> None:
    if "runtime_median_ms" not in summary_df.columns:
        print("[plot] runtime_median_ms missing; skipping runtime median plots.")
        return
    summary_df = _coerce_numeric(summary_df, ["runtime_median_ms"])

    for config_id, group in summary_df.groupby("config_id"):
        group = group.dropna(subset=["runtime_median_ms"])
        if group.empty:
            continue
        order = _algorithm_order(group, algorithm_order)
        colors = colors or _color_map(order)
        fig, ax = _require_matplotlib().subplots(figsize=(7.5, 4.0))
        for idx, algo in enumerate(order):
            row = group[group["algorithm"] == algo]
            if row.empty:
                continue
            y = float(row["runtime_median_ms"].iloc[0])
            ax.bar(idx, y, color=colors.get(algo), alpha=0.75)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=25, ha="right")
        ax.set_ylabel("Runtime median (ms)")
        ax.set_title(f"Runtime Median | {config_id}")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        filename = f"{_safe_name(config_id)}_runtime_median_bar"
        _save_figure(fig, output_dir, filename)
        _require_matplotlib().close(fig)


def plot_runtime_quantiles(
    summary_df: pd.DataFrame,
    output_dir: Path,
    algorithm_order: Optional[Sequence[str]] = None,
    colors: Optional[Dict[str, Any]] = None,
) -> None:
    if "runtime_median_ms" not in summary_df.columns or "runtime_q_ms" not in summary_df.columns:
        print("[plot] runtime_median_ms/runtime_q_ms missing; skipping runtime quantile plots.")
        return
    summary_df = _coerce_numeric(summary_df, ["runtime_median_ms", "runtime_q_ms"])

    for config_id, group in summary_df.groupby("config_id"):
        order = _algorithm_order(group, algorithm_order)
        colors = colors or _color_map(order)
        plot_df = group.set_index("algorithm").reindex(order).reset_index()
        med = pd.to_numeric(plot_df["runtime_median_ms"], errors="coerce")
        q = pd.to_numeric(plot_df["runtime_q_ms"], errors="coerce")
        if med.isna().all() or q.isna().all():
            continue

        fig, ax = _require_matplotlib().subplots(figsize=(7.5, 4.0))
        x = np.arange(len(order))
        width = 0.38
        for idx, algo in enumerate(order):
            color = colors.get(algo)
            ax.bar(
                x[idx] - width / 2,
                med.iloc[idx],
                width=width,
                color=color,
                alpha=0.85,
                label="Median" if idx == 0 else None,
            )
            ax.bar(
                x[idx] + width / 2,
                q.iloc[idx],
                width=width,
                color=color,
                alpha=0.35,
                hatch="//",
                label="95th percentile" if idx == 0 else None,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=25, ha="right")
        ax.set_ylabel("Runtime (ms)")
        ax.set_yscale("log")
        ax.set_title(f"Runtime Median vs 95th Percentile | {config_id}")
        ax.legend(frameon=False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        filename = f"{_safe_name(config_id)}_runtime_quantiles_bar_log"
        _save_figure(fig, output_dir, filename)
        _require_matplotlib().close(fig)


def _expanded_nodes_column(summary_df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "expanded_nodes_median",
        "expanded_nodes_mean",
        "metric_expanded_nodes_median",
        "metric_expanded_nodes_mean",
    ]
    for col in candidates:
        if col in summary_df.columns:
            return col
    return None


def plot_search_effort(
    summary_df: pd.DataFrame,
    output_dir: Path,
    algorithm_order: Optional[Sequence[str]] = None,
    colors: Optional[Dict[str, Any]] = None,
) -> None:
    col = _expanded_nodes_column(summary_df)
    if not col:
        print("[plot] Expanded node metric missing; skipping search effort plots.")
        return
    summary_df = _coerce_numeric(summary_df, [col])
    for config_id, group in summary_df.groupby("config_id"):
        group = group.dropna(subset=[col])
        if group.empty:
            continue
        order = _algorithm_order(group, algorithm_order)
        colors = colors or _color_map(order)
        fig, ax = _require_matplotlib().subplots(figsize=(7.5, 4.0))
        for idx, algo in enumerate(order):
            row = group[group["algorithm"] == algo]
            if row.empty:
                continue
            y = float(row[col].iloc[0])
            ax.scatter(idx, y, color=colors.get(algo))
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=25, ha="right")
        ax.set_ylabel("Expanded nodes (count)")
        ax.set_title(f"Search Effort | {config_id}")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        filename = f"{_safe_name(config_id)}_expanded_nodes_dot"
        _save_figure(fig, output_dir, filename)
        _require_matplotlib().close(fig)


def plot_cost_decomposition(
    summary_df: pd.DataFrame,
    output_dir: Path,
    algorithm_order: Optional[Sequence[str]] = None,
    colors: Optional[Dict[str, Any]] = None,
) -> None:
    needed = ["metric_nominal_cost_median", "metric_robust_cost_median"]
    if not all(col in summary_df.columns for col in needed):
        print("[plot] Nominal/robust cost metrics missing; skipping cost decomposition plots.")
        return
    summary_df = _coerce_numeric(summary_df, needed)

    for config_id, group in summary_df.groupby("config_id"):
        group = group.dropna(subset=needed)
        if group.empty:
            continue
        order = _algorithm_order(group, algorithm_order)
        colors = colors or _color_map(order)
        plot_df = group.set_index("algorithm").reindex(order).reset_index()
        nominal = plot_df["metric_nominal_cost_median"].to_numpy()
        robust = plot_df["metric_robust_cost_median"].to_numpy()

        fig, ax = _require_matplotlib().subplots(figsize=(7.5, 4.0))
        x = np.arange(len(order))
        width = 0.38
        for idx, algo in enumerate(order):
            color = colors.get(algo)
            ax.bar(
                x[idx] - width / 2,
                nominal[idx],
                width=width,
                color=color,
                alpha=0.5,
                label="Nominal" if idx == 0 else None,
            )
            ax.bar(
                x[idx] + width / 2,
                robust[idx],
                width=width,
                color=color,
                alpha=0.9,
                hatch="//",
                label="Robust" if idx == 0 else None,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=25, ha="right")
        ax.set_ylabel("Cost")
        ax.set_title(f"Nominal vs Robust Cost | {config_id}")
        ax.legend(frameon=False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        filename = f"{_safe_name(config_id)}_cost_nominal_robust_bar"
        _save_figure(fig, output_dir, filename)
        _require_matplotlib().close(fig)

        premium = robust - nominal
        fig, ax = _require_matplotlib().subplots(figsize=(7.5, 4.0))
        for idx, algo in enumerate(order):
            ax.bar(idx, premium[idx], color=colors.get(algo))
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=25, ha="right")
        ax.set_ylabel("Robustness premium (cost)")
        ax.set_title(f"Robustness Premium | {config_id}")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        filename = f"{_safe_name(config_id)}_robustness_premium_bar"
        _save_figure(fig, output_dir, filename)
        _require_matplotlib().close(fig)


def _aggregate_param_sweep(
    df: pd.DataFrame,
    param: str,
    metric: str,
    ci_low: Optional[str] = None,
    ci_high: Optional[str] = None,
) -> pd.DataFrame:
    agg: Dict[str, Any] = {
        metric: "mean",
    }
    if ci_low and ci_low in df.columns:
        agg[ci_low] = "min"
    if ci_high and ci_high in df.columns:
        agg[ci_high] = "max"
    grouped = df.groupby(["algorithm", param], as_index=False).agg(agg)
    return grouped


def plot_parameter_sweeps(
    summary_df: pd.DataFrame,
    output_dir: Path,
    algorithm_order: Optional[Sequence[str]] = None,
    colors: Optional[Dict[str, Any]] = None,
) -> None:
    params = [
        "grid_size",
        "length_scale_x",
        "length_scale_y",
        "length_scale_t",
        "variance",
        "gamma",
        "num_scenarios",
    ]
    for param in params:
        if param not in summary_df.columns:
            continue
        values = pd.to_numeric(summary_df[param], errors="coerce")
        if values.nunique(dropna=True) < 2:
            continue
        sweep_df = summary_df.copy()
        sweep_df[param] = values
        sweep_df = _coerce_numeric(sweep_df, ["cost_median", "cost_ci_low", "cost_ci_high"])
        sweep_df = sweep_df.dropna(subset=[param, "cost_median"])
        if sweep_df.empty:
            continue
        order = _algorithm_order(sweep_df, algorithm_order)
        colors = colors or _color_map(order)

        agg_df = _aggregate_param_sweep(
            sweep_df,
            param=param,
            metric="cost_median",
            ci_low="cost_ci_low",
            ci_high="cost_ci_high",
        )

        fig, ax = _require_matplotlib().subplots(figsize=(7.5, 4.0))
        for algo in order:
            algo_df = agg_df[agg_df["algorithm"] == algo].sort_values(param)
            if algo_df.empty:
                continue
            x = algo_df[param].to_numpy()
            y = algo_df["cost_median"].to_numpy()
            ax.plot(x, y, marker="o", color=colors.get(algo), label=algo)
            if "cost_ci_low" in algo_df.columns and "cost_ci_high" in algo_df.columns:
                low = algo_df["cost_ci_low"].to_numpy()
                high = algo_df["cost_ci_high"].to_numpy()
                if np.isfinite(low).any() and np.isfinite(high).any():
                    ax.fill_between(x, low, high, color=colors.get(algo), alpha=0.15)

        ax.set_xlabel(param.replace("_", " "))
        ax.set_ylabel("Cost")
        ax.set_title(f"Cost vs {param.replace('_', ' ')}")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        filename = f"all_{_safe_name(param)}_cost_line"
        _save_figure(fig, output_dir, filename)
        _require_matplotlib().close(fig)

        if "runtime_median_ms" in sweep_df.columns:
            runtime_df = _coerce_numeric(sweep_df, ["runtime_median_ms"])
            runtime_df = runtime_df.dropna(subset=["runtime_median_ms", param])
            if runtime_df.empty:
                continue
            runtime_agg = _aggregate_param_sweep(
                runtime_df,
                param=param,
                metric="runtime_median_ms",
            )
            fig, ax = _require_matplotlib().subplots(figsize=(7.5, 4.0))
            for algo in order:
                algo_df = runtime_agg[runtime_agg["algorithm"] == algo].sort_values(param)
                if algo_df.empty:
                    continue
                x = algo_df[param].to_numpy()
                y = algo_df["runtime_median_ms"].to_numpy()
                ax.plot(x, y, marker="o", color=colors.get(algo), label=algo)
            ax.set_xlabel(param.replace("_", " "))
            ax.set_ylabel("Runtime (ms)")
            ax.set_title(f"Runtime vs {param.replace('_', ' ')}")
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.legend(frameon=False)
            fig.tight_layout()
            filename = f"all_{_safe_name(param)}_runtime_line"
            _save_figure(fig, output_dir, filename)
            _require_matplotlib().close(fig)


def plot_tradeoff_scatter(
    summary_df: pd.DataFrame,
    output_dir: Path,
    runtime_metric: str = "runtime_median_ms",
    algorithm_order: Optional[Sequence[str]] = None,
    colors: Optional[Dict[str, Any]] = None,
    annotate: bool = False,
) -> None:
    if runtime_metric not in summary_df.columns or "cost_median" not in summary_df.columns:
        print("[plot] Missing runtime or cost metrics; skipping trade-off scatter plot.")
        return
    plot_df = summary_df.copy()
    plot_df = _coerce_numeric(plot_df, ["cost_median", runtime_metric])
    plot_df = plot_df.dropna(subset=["cost_median", runtime_metric])
    plot_df = plot_df[plot_df[runtime_metric] > 0]
    if plot_df.empty:
        return

    order = _algorithm_order(plot_df, algorithm_order)
    colors = colors or _color_map(order)

    fig, ax = _require_matplotlib().subplots(figsize=(7.0, 4.5))
    for algo in order:
        algo_df = plot_df[plot_df["algorithm"] == algo]
        if algo_df.empty:
            continue
        label = _format_algorithm_label(algo)
        ax.scatter(
            algo_df[runtime_metric].to_numpy(),
            algo_df["cost_median"].to_numpy(),
            label=label,
            color=colors.get(algo),
            alpha=0.6,
            zorder=2,
        )
        if annotate:
            for xv, yv in zip(
                algo_df[runtime_metric].to_numpy(),
                algo_df["cost_median"].to_numpy(),
            ):
                ax.annotate(label, (xv, yv), textcoords="offset points", xytext=(4, 4), fontsize=9)

    ax.set_xscale("log")
    ax.set_xlabel("Runtime (ms, log scale)")
    ax.set_ylabel("Cost")
    ax.set_title("Quality-Effort Trade-off")
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    filename = f"all_tradeoff_{_safe_name(runtime_metric)}_scatter"
    _save_figure(fig, output_dir, filename)
    _require_matplotlib().close(fig)


def plot_replans_vs_cost(
    raw_df: pd.DataFrame,
    output_dir: Path,
    experiments_root: Path,
    algorithm_order: Optional[Sequence[str]] = None,
    colors: Optional[Dict[str, Any]] = None,
    ok_statuses: Sequence[str] = ("success", "ok"),
) -> None:
    if "realized_cost" not in raw_df.columns:
        print("[plot] realized_cost missing; skipping replans vs cost plots.")
        return
    if "seed" not in raw_df.columns or "algorithm" not in raw_df.columns or "config_id" not in raw_df.columns:
        print("[plot] seed/algorithm/config_id missing; skipping replans vs cost plots.")
        return

    algo_to_file = {
        "dstar": "DStarLite_result.json",
        "dstar_discrete": "DStarLiteDiscreteUncertainty_result.json",
        "dstar_discrete_adaptive": "DStarLiteDiscreteAdaptiveUncertainty_result.json",
        "dstar_budgeted": "DStarLiteBudgetedUncertainty_result.json",
    }

    df = raw_df.copy()
    if "status" in df.columns:
        ok_set = {s.lower() for s in ok_statuses}
        df["status"] = df["status"].astype(str).str.lower()
        df = df[df["status"].isin(ok_set)]

    df = df[df["algorithm"].isin(algo_to_file)]
    df = _coerce_numeric(df, ["realized_cost", "seed"])
    df = df.dropna(subset=["realized_cost", "seed"])
    if df.empty:
        print("[plot] No D* Lite rows available for replans vs cost plots.")
        return

    replans_cache: Dict[Tuple[str, int, str], Optional[float]] = {}

    def lookup_replans(config_id: str, seed: int, algorithm: str) -> Optional[float]:
        key = (config_id, seed, algorithm)
        if key in replans_cache:
            return replans_cache[key]
        json_name = algo_to_file.get(algorithm)
        if not json_name:
            replans_cache[key] = None
            return None
        base = experiments_root / config_id / "data" / f"seed_{seed:04d}" / "ComparisonData"
        path = base / json_name
        if not path.exists():
            replans_cache[key] = None
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            replans_cache[key] = None
            return None
        value = payload.get("replans")
        try:
            replans_cache[key] = float(value) if value is not None else None
        except Exception:
            replans_cache[key] = None
        return replans_cache[key]

    for config_id, group in df.groupby("config_id"):
        rows = []
        for _, row in group.iterrows():
            seed = int(row["seed"])
            algorithm = str(row["algorithm"])
            replans = lookup_replans(str(config_id), seed, algorithm)
            if replans is None:
                continue
            rows.append(
                {
                    "algorithm": algorithm,
                    "realized_cost": float(row["realized_cost"]),
                    "replans": replans,
                }
            )
        if not rows:
            continue
        plot_df = pd.DataFrame(rows)
        order = _algorithm_order(plot_df, algorithm_order)
        colors = colors or _color_map(order)
        rng = np.random.default_rng(abs(hash(str(config_id))) % (2**32))

        fig, ax = _require_matplotlib().subplots(figsize=(7.0, 4.5))
        for algo in order:
            algo_df = plot_df[plot_df["algorithm"] == algo]
            if algo_df.empty:
                continue
            jitter = rng.uniform(-0.25, 0.25, size=len(algo_df))
            ax.scatter(
                algo_df["realized_cost"].to_numpy(),
                algo_df["replans"].to_numpy() + jitter,
                label=algo,
                color=colors.get(algo),
                alpha=0.85,
            )
            ax.scatter(
                [float(algo_df["realized_cost"].median())],
                [float(algo_df["replans"].median())],
                marker="X",
                s=60,
                color=colors.get(algo),
                edgecolors="black",
                linewidths=0.4,
                alpha=0.9,
                label=None,
            )
        ax.set_xlabel("Cost")
        ax.set_ylabel("Replans (jittered)")
        ax.set_title(f"Replans vs Cost | {config_id}")
        ax.grid(axis="both", linestyle="--", alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        filename = f"{_safe_name(config_id)}_replans_vs_cost_scatter"
        _save_figure(fig, output_dir, filename)
        _require_matplotlib().close(fig)


def plot_path_geometry(
    summary_df: pd.DataFrame,
    output_dir: Path,
    algorithm_order: Optional[Sequence[str]] = None,
    colors: Optional[Dict[str, Any]] = None,
) -> None:
    candidates = [
        "metric_boundary_edge_share_median",
        "metric_boundary_edge_share_mean",
    ]
    column = next((c for c in candidates if c in summary_df.columns), None)
    if not column:
        print("[plot] Boundary-edge metric missing; skipping path geometry plots.")
        return
    summary_df = _coerce_numeric(summary_df, [column])
    for config_id, group in summary_df.groupby("config_id"):
        group = group.dropna(subset=[column])
        if group.empty:
            continue
        order = _algorithm_order(group, algorithm_order)
        colors = colors or _color_map(order)
        fig, ax = _require_matplotlib().subplots(figsize=(7.5, 4.0))
        for idx, algo in enumerate(order):
            row = group[group["algorithm"] == algo]
            if row.empty:
                continue
            y = float(row[column].iloc[0])
            ax.bar(idx, y, color=colors.get(algo))
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=25, ha="right")
        ax.set_ylabel("Boundary-edge share")
        ax.set_title(f"Path Geometry | {config_id}")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        filename = f"{_safe_name(config_id)}_boundary_edge_share_bar"
        _save_figure(fig, output_dir, filename)
        _require_matplotlib().close(fig)


def run_all_plots(
    summary_csv: str | Path,
    output_dir: Optional[str | Path] = None,
    runtime_metric: str = "runtime_median_ms",
    annotate_tradeoff: bool = False,
    raw_csv: Optional[str | Path] = None,
) -> None:
    set_plot_style()
    summary_path = Path(summary_csv).expanduser()
    summary_df = pd.read_csv(summary_path)
    output_dir = Path(output_dir or (Path.cwd() / "figures" / "experiment_results")).expanduser()

    algorithm_order = _algorithm_order(summary_df)
    colors = _color_map(algorithm_order)

    plot_tradeoff_scatter(
        summary_df,
        output_dir,
        runtime_metric=runtime_metric,
        algorithm_order=algorithm_order,
        colors=colors,
        annotate=annotate_tradeoff,
    )
    plot_runtime_median(summary_df, output_dir, algorithm_order=algorithm_order, colors=colors)
    plot_cost_ci_dot(summary_df, output_dir, algorithm_order=algorithm_order, colors=colors)

    raw_path = _resolve_raw_csv(summary_path, raw_csv)
    raw_df = _load_raw_results(raw_path) if raw_path else pd.DataFrame()
    if not raw_df.empty:
        plot_cost_distribution_raw(
            raw_df,
            output_dir,
            algorithm_order=algorithm_order,
            colors=colors,
        )
        experiments_root = raw_path.parent.parent if raw_path else Path.cwd()
        plot_replans_vs_cost(
            raw_df,
            output_dir,
            experiments_root=experiments_root,
            algorithm_order=algorithm_order,
            colors=colors,
        )
    else:
        print("[plot] Raw CSV not found; skipping cost distribution and replans vs cost plots.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot aggregated experiment results.")
    parser.add_argument("summary_csv", help="Path to summary.csv from batch aggregation.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for figures (default: ./figures/experiment_results).",
    )
    parser.add_argument(
        "--runtime-metric",
        default="runtime_median_ms",
        choices=["runtime_median_ms", "runtime_q_ms"],
        help="Runtime metric for trade-off scatter plot.",
    )
    parser.add_argument(
        "--annotate-tradeoff",
        action="store_true",
        help="Annotate trade-off scatter points with algorithm names.",
    )
    parser.add_argument(
        "--raw-csv",
        default=None,
        help="Optional raw results CSV (defaults to batch_results.csv next to summary).",
    )
    args = parser.parse_args()
    run_all_plots(
        args.summary_csv,
        output_dir=args.output_dir,
        runtime_metric=args.runtime_metric,
        annotate_tradeoff=args.annotate_tradeoff,
        raw_csv=args.raw_csv,
    )


if __name__ == "__main__":
    main()
