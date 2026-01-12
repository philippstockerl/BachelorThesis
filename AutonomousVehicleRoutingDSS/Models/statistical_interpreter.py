from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import gaussian_kde


Number = float | int


def _as_tuple(value: Any, length: int, default: float = 1.0) -> Tuple[float, ...]:
    """Return a tuple of length `length` filled with numeric values."""
    if isinstance(value, (list, tuple, np.ndarray)):
        seq = [float(v) for v in value]
        if not seq:
            seq = [default]
    elif value is None:
        seq = [default]
    else:
        seq = [float(value)]

    if len(seq) < length:
        seq.extend([seq[-1]] * (length - len(seq)))
    return tuple(seq[:length])


def _describe(value: float, thresholds: Sequence[float], labels: Sequence[str]) -> str:
    """Map a numeric value to a label using monotonically increasing thresholds."""
    for threshold, label in zip(thresholds, labels):
        if value < threshold:
            return label
    return labels[-1] if labels else ""


def _count_local_peaks(field: np.ndarray, percentile: float = 97.0) -> int:
    """Approximate local peaks using a 3×3 neighborhood and a percentile cutoff."""
    if field.size == 0:
        return 0
    cutoff = np.percentile(field, percentile)
    padded = np.pad(field, 1, mode="edge")
    count = 0
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            val = field[i, j]
            if val < cutoff:
                continue
            window = padded[i : i + 3, j : j + 3]
            if val >= window.max() and np.count_nonzero(window == val) == 1:
                count += 1
    return int(count)


def _centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    idx = np.argwhere(mask)
    if idx.size == 0:
        return None
    cx, cy = idx.mean(axis=0)
    return float(cx), float(cy)


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def _tail_mass(xs: np.ndarray, density: np.ndarray, center: float, width: float) -> float:
    mask = (xs < center - width) | (xs > center + width)
    total = np.trapezoid(density, xs)
    if total <= 0:
        return 0.0
    tail = np.trapezoid(density[mask], xs[mask])
    return float(tail / total)


class StatisticalInterpreter:
    """
    Deterministic interpretation of STRF parameters and visualization-ready data.
    Reads the saved STRF scenarios (field.npy) and produces a JSON report with
    descriptive statistics for each visualization.
    """

    def __init__(self, data_root: Path, params: Dict[str, Any]) -> None:
        self.data_root = Path(data_root)
        self.params = params
        self.fields = self._load_fields()
        self.grid_size = int(params.get("grid_size") or self.fields[0].shape[0])
        self.num_scenarios = int(params.get("num_scenarios") or len(self.fields))
        self.variance = float(params.get("variance", np.var(self.fields[0])))
        self.length_scale = _as_tuple(params.get("length_scale"), 3, default=1.0)
        self.anis = _as_tuple(params.get("anis"), 3, default=1.0)
        self.output_path = self.data_root / "metadata" / "statistical_interpretation.json"

    def _load_fields(self) -> List[np.ndarray]:
        scen_paths = sorted(self.data_root.glob("scenario_*"))
        fields: List[np.ndarray] = []
        for scen in scen_paths:
            fp = scen / "field.npy"
            if fp.exists():
                fields.append(np.load(fp))
        if not fields:
            raise RuntimeError(f"No scenario fields found under {self.data_root}")
        return fields

    def _scenario_costs(self) -> np.ndarray:
        """Average cost per scenario; aligns with histogram/KDE inputs."""
        return np.array([float(f.mean()) for f in self.fields], dtype=float)

    def _hotspot_masks(self, percentile: float = 95.0) -> List[np.ndarray]:
        masks = []
        for f in self.fields:
            cutoff = np.percentile(f, percentile)
            masks.append(f >= cutoff)
        return masks

    def interpret_strf_parameters(self) -> Dict[str, Any]:
        lx, ly, lt = self.length_scale
        ax, ay, at = self.anis
        spatial_scale = float((lx + ly) / 2)
        smooth_ratio = spatial_scale / max(1.0, float(self.grid_size))
        anis_ratio = max(ax, ay) / max(1e-9, min(ax, ay))
        temporal_ratio = lt / max(1.0, float(self.num_scenarios))
        fluctuation = math.sqrt(max(self.variance, 0.0))
        hotspot_est = max(
            1,
            int(
                (self.grid_size * self.grid_size)
                / max(1.0, math.pi * max(spatial_scale, 1.0) ** 2)
            ),
        )

        smooth_desc = _describe(
            smooth_ratio,
            thresholds=[0.08, 0.2, 0.4],
            labels=["very rough", "patchy", "moderately smooth", "very smooth"],
        )
        cloud_desc = _describe(
            spatial_scale,
            thresholds=[
                0.05 * self.grid_size,
                0.15 * self.grid_size,
                0.35 * self.grid_size,
            ],
            labels=["fine-grained cells", "small clouds", "medium blobs", "broad clouds"],
        )
        anis_desc = (
            "isotropic footprint"
            if anis_ratio < 1.15
            else (
                "mild stretch across axes"
                if anis_ratio < 1.6
                else "strong directional stretch"
            )
        )
        temporal_desc = _describe(
            temporal_ratio,
            thresholds=[0.15, 0.4, 0.8],
            labels=["chaotic", "short-lived", "persistent", "quasi-static"],
        )
        fluctuation_desc = _describe(
            fluctuation,
            thresholds=[0.25, 0.9, 2.0],
            labels=["low-amplitude", "moderate", "high-variance", "very noisy"],
        )

        interpretation = (
            f"Spatial scales ({lx:.2f}, {ly:.2f}) on a {self.grid_size}×{self.grid_size} grid "
            f"imply {smooth_desc} surfaces with {cloud_desc}. Variance σ≈{fluctuation:.3f} "
            f"yields {fluctuation_desc} fluctuations. Anisotropy ({ax:.2f}, {ay:.2f}) gives a {anis_desc}, "
            f"while temporal scale {lt:.2f} over {self.num_scenarios} scenarios points to {temporal_desc} costs. "
            f"Correlation footprint suggests roughly {hotspot_est} distinguishable hotspots at any time."
        )

        return {
            "metrics": {
                "variance": float(self.variance),
                "length_scale": [float(lx), float(ly), float(lt)],
                "anis": [float(ax), float(ay), float(at)],
                "smoothness_ratio": float(smooth_ratio),
                "anisotropy_ratio": float(anis_ratio),
                "temporal_ratio": float(temporal_ratio),
                "hotspot_estimate": hotspot_est,
            },
            "interpretation": interpretation,
        }

    def interpret_cost_surface_gif(self) -> Dict[str, Any]:
        grad_strength: List[float] = []
        grad_balance: List[float] = []
        peak_counts: List[int] = []
        prominences: List[float] = []
        hotspot_share: List[float] = []
        peak_positions: List[Tuple[int, int]] = []
        hotspot_masks = self._hotspot_masks(percentile=95.0)

        for f, mask in zip(self.fields, hotspot_masks):
            gx, gy = np.gradient(f)
            grad = np.sqrt(gx ** 2 + gy ** 2)
            ref = f.std() or 1.0
            grad_strength.append(float(grad.mean() / ref))
            grad_balance.append(float((gx.var() + 1e-9) / (gy.var() + 1e-9)))
            peak_counts.append(_count_local_peaks(f, percentile=97.0))
            prominences.append(float(f.max() - np.percentile(f, 90)))
            hotspot_share.append(float(mask.mean()))
            peak_positions.append(tuple(int(v) for v in np.unravel_index(np.argmax(f), f.shape)))

        drift = []
        for a, b in zip(peak_positions[:-1], peak_positions[1:]):
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            drift.append(math.sqrt(dx * dx + dy * dy))
        drift_per_step = float(np.mean(drift)) if drift else 0.0

        overlap = [
            _jaccard(a, b) for a, b in zip(hotspot_masks[:-1], hotspot_masks[1:])
        ]

        grad_level = _describe(
            float(np.mean(grad_strength)),
            thresholds=[0.3, 0.8, 1.6],
            labels=["very smooth", "smooth", "moderately rugged", "highly rugged"],
        )
        drift_desc = _describe(
            drift_per_step / max(1.0, self.grid_size),
            thresholds=[0.01, 0.05, 0.12],
            labels=["locked peaks", "slow drift", "moderate drift", "fast-moving peaks"],
        )
        stability_desc = _describe(
            float(np.mean(overlap)) if overlap else 0.0,
            thresholds=[0.2, 0.5, 0.75],
            labels=["transient hotspots", "intermittent overlap", "stable hotspots", "very persistent hotspots"],
        )
        anis_desc = (
            "balanced gradients"
            if np.mean(grad_balance) < 1.15 and np.mean(grad_balance) > (1 / 1.15)
            else "directional slope bias"
        )

        interpretation = (
            f"Average gradient intensity is {np.mean(grad_strength):.2f}× the field's own scale, indicating {grad_level} surfaces. "
            f"High-cost peaks average {np.mean(peak_counts):.1f} per scenario with prominence ≈{np.mean(prominences):.3f}. "
            f"Hotspots cover {np.mean(hotspot_share) * 100:.1f}% of the grid and show {stability_desc}. "
            f"Peak locations shift by ~{drift_per_step:.2f} cells per step ({drift_desc}), and gradients exhibit {anis_desc}."
        )

        return {
            "metrics": {
                "mean_gradient_vs_std": float(np.mean(grad_strength)),
                "gradient_balance_x_over_y": float(np.mean(grad_balance)),
                "peak_count_avg": float(np.mean(peak_counts)),
                "peak_prominence_mean": float(np.mean(prominences)),
                "hotspot_area_share": float(np.mean(hotspot_share)),
                "hotspot_overlap": float(np.mean(overlap)) if overlap else 0.0,
                "peak_drift_per_step": drift_per_step,
            },
            "interpretation": interpretation,
        }

    def interpret_heatmap_gif(self) -> Dict[str, Any]:
        stack = np.stack(self.fields)
        global_min = float(stack.min())
        global_max = float(stack.max())
        global_mean = float(stack.mean())
        spatial_var = float(stack.var(axis=0).mean())
        masks = self._hotspot_masks(percentile=95.0)
        overlaps = [_jaccard(a, b) for a, b in zip(masks[:-1], masks[1:])]
        centroids = [_centroid(mask) for mask in masks]
        drifts = []
        for a, b in zip(centroids[:-1], centroids[1:]):
            if a is None or b is None:
                continue
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            drifts.append(math.sqrt(dx * dx + dy * dy))

        drift_cells = float(np.mean(drifts)) if drifts else 0.0
        hotspot_area = float(np.mean([m.mean() for m in masks])) if masks else 0.0
        overlap_mean = float(np.mean(overlaps)) if overlaps else 0.0

        stability_desc = _describe(
            overlap_mean,
            thresholds=[0.2, 0.5, 0.75],
            labels=["unstable", "semi-stable", "stable", "very stable"],
        )
        drift_desc = _describe(
            drift_cells / max(1.0, self.grid_size),
            thresholds=[0.01, 0.05, 0.12],
            labels=["fixed hotspots", "slow drift", "moderate drift", "rapid movement"],
        )

        interpretation = (
            f"Intensity spans [{global_min:.3f}, {global_max:.3f}] with mean {global_mean:.3f}. "
            f"Spatial variance averages {spatial_var:.4f}. "
            f"Hotspots (95th percentile) occupy ~{hotspot_area * 100:.1f}% of cells and are {stability_desc} over time. "
            f"Centroid drift of {drift_cells:.2f} cells per step indicates {drift_desc} of dominant regions."
        )

        return {
            "metrics": {
                "global_min": global_min,
                "global_max": global_max,
                "global_mean": global_mean,
                "spatial_variance_mean": spatial_var,
                "hotspot_area_share": hotspot_area,
                "hotspot_jaccard": overlap_mean,
                "hotspot_drift_cells": drift_cells,
            },
            "interpretation": interpretation,
        }

    def interpret_histogram_gif(self) -> Dict[str, Any]:
        vals = self._scenario_costs()
        mean = float(vals.mean())
        median = float(np.median(vals))
        std = float(vals.std() or 0.0)
        cv = float(std / mean) if mean != 0 else 0.0
        centered = vals - mean
        skew = float((np.mean(centered ** 3)) / (std ** 3 + 1e-12))
        kurtosis = float((np.mean(centered ** 4)) / (std ** 4 + 1e-12) - 3.0)
        outliers = np.logical_or(vals > mean + 3 * std, vals < mean - 3 * std).mean()
        shift = (
            float(vals[len(vals) // 2 :].mean() - vals[: len(vals) // 2].mean())
            if len(vals) >= 4
            else 0.0
        )

        skew_desc = (
            "right-skewed" if skew > 0.2 else ("left-skewed" if skew < -0.2 else "symmetric")
        )
        tail_desc = _describe(
            kurtosis,
            thresholds=[-0.5, 1.0, 3.0],
            labels=["light tails", "near-normal tails", "heavy tails", "very heavy tails"],
        )

        interpretation = (
            f"Scenario costs average {mean:.3f} (median {median:.3f}) with σ={std:.3f} "
            f"(CV={cv:.2f}). Distribution is {skew_desc} (skew={skew:.2f}) with {tail_desc} "
            f"(kurtosis={kurtosis:.2f}). Outliers beyond 3σ occur in {outliers * 100:.1f}% of scenarios. "
            f"Later scenarios shift by {shift:.3f} relative to early ones, signalling "
            f"{'upward' if shift > 0 else 'downward' if shift < 0 else 'no'} drift."
        )

        return {
            "metrics": {
                "mean": mean,
                "median": median,
                "std": std,
                "coefficient_of_variation": cv,
                "skewness": skew,
                "kurtosis_excess": kurtosis,
                "outlier_rate": float(outliers),
                "late_minus_early_shift": shift,
            },
            "interpretation": interpretation,
        }

    def interpret_kde_gif(self) -> Dict[str, Any]:
        vals = self._scenario_costs()
        if len(vals) < 2:
            return {
                "metrics": {},
                "interpretation": "Not enough samples for KDE.",
            }
        span = max(vals.max() - vals.min(), 1e-6)
        xs = np.linspace(vals.min() - 0.2 * span, vals.max() + 0.2 * span, 250)
        kde = gaussian_kde(vals)
        density = kde(xs)

        peak_idx = [
            i
            for i in range(1, len(xs) - 1)
            if density[i] > density[i - 1] and density[i] > density[i + 1]
        ]
        peaks = xs[peak_idx]
        tail = _tail_mass(xs, density, center=float(vals.mean()), width=float(vals.std() or 1.0))

        mode_desc = (
            "unimodal"
            if len(peaks) <= 1
            else ("bimodal" if len(peaks) == 2 else "multimodal")
        )

        interpretation = (
            f"KDE is {mode_desc} with modes at "
            f"{', '.join(f'{p:.3f}' for p in peaks[:3]) or '—'}. "
            f"Density concentrates between {xs[int(0.1 * len(xs))]:.3f} and "
            f"{xs[int(0.9 * len(xs))]:.3f}, with tail mass of {tail * 100:.1f}% "
            f"outside ±1σ of the mean. Mode locations drift "
            f"{'slightly' if len(peaks) <= 1 else 'across scenarios'} as indicated by the GIF."
        )

        return {
            "metrics": {
                "mode_count": len(peaks),
                "mode_locations": [float(p) for p in peaks.tolist()],
                "tail_mass_outside_1sigma": tail,
            },
            "interpretation": interpretation,
        }

    def interpret_violin_gif(self) -> Dict[str, Any]:
        medians: List[float] = []
        iqrs: List[float] = []
        symmetry_scores: List[float] = []
        tail_ratios: List[float] = []

        for f in self.fields:
            vals = f.flatten()
            q1, q2, q3 = np.percentile(vals, [25, 50, 75])
            iqr = q3 - q1
            medians.append(float(q2))
            iqrs.append(float(iqr))
            symmetry_scores.append(float(abs((q2 - q1) - (q3 - q2)) / (iqr + 1e-9)))
            lower_tail = abs(q1 - vals.min())
            upper_tail = abs(vals.max() - q3)
            denom = (upper_tail + lower_tail) or 1.0
            tail_ratios.append(float(upper_tail / denom))

        median_range = (float(np.min(medians)), float(np.max(medians)))
        iqr_mean = float(np.mean(iqrs))
        iqr_var = float(np.std(iqrs))
        symmetry_mean = float(np.mean(symmetry_scores))
        upper_tail_share = float(np.mean(tail_ratios))

        spread_desc = _describe(
            iqr_mean,
            thresholds=[0.05, 0.15, 0.35],
            labels=["tight spread", "compact", "moderate spread", "broad spread"],
        )
        symmetry_desc = _describe(
            symmetry_mean,
            thresholds=[0.05, 0.15, 0.3],
            labels=["highly symmetric", "mostly symmetric", "mildly skewed", "skewed"],
        )

        interpretation = (
            f"Medians range from {median_range[0]:.3f} to {median_range[1]:.3f}. "
            f"IQR averages {iqr_mean:.3f} (σ={iqr_var:.3f}), indicating {spread_desc} with "
            f"{'consistent' if iqr_var < iqr_mean * 0.3 else 'varying'} spread across scenarios. "
            f"Shape symmetry score {symmetry_mean:.2f} denotes {symmetry_desc}. "
            f"Upper tails hold {upper_tail_share * 100:.1f}% of total tail mass, signalling "
            f"{'more frequent high spikes' if upper_tail_share > 0.55 else 'balanced tails'}."
        )

        return {
            "metrics": {
                "median_min": median_range[0],
                "median_max": median_range[1],
                "iqr_mean": iqr_mean,
                "iqr_std": iqr_var,
                "symmetry_score_mean": symmetry_mean,
                "upper_tail_mass_share": upper_tail_share,
            },
            "interpretation": interpretation,
        }

    def build_report(self) -> Dict[str, Any]:
        return {
            "strf_parameters": self.interpret_strf_parameters(),
            "cost_surface_gif": self.interpret_cost_surface_gif(),
            "heatmap_gif": self.interpret_heatmap_gif(),
            "histogram_gif": self.interpret_histogram_gif(),
            "kde_gif": self.interpret_kde_gif(),
            "violin_gif": self.interpret_violin_gif(),
        }

    def save(self, report: Dict[str, Any]) -> Path:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        return self.output_path

    def run(self) -> Dict[str, Any]:
        report = self.build_report()
        self.save(report)
        return report
