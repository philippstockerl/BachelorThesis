from __future__ import annotations

# ===============================================================
# Standard library imports
# ===============================================================
import contextlib
import copy
import html
import io
import json
import os
import subprocess
import sys
import time
from base64 import b64encode
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ===============================================================
# Third-party imports
# ===============================================================
import pandas as pd
import streamlit as st

# ===============================================================
# Local model modules
# ===============================================================
try:
    from Models.DSSRunners.STGRF_runner import run_srf_generator as strf_generator
except Exception:  # pragma: no cover - surface error inside UI
    strf_generator = None

try:
    from Models.DSSRunners.DiscreteUncertainty_runner import run_discrete_pipeline
except Exception:
    run_discrete_pipeline = None

try:
    from Models.DSSRunners.BudgetedUncertainty_runner import run_budgeted_uncertainty
except Exception:
    run_budgeted_uncertainty = None

try:
    from Models.DSSRunners.DStar_Lite_runner import run_dstar_pipeline as dstar_pipeline
except Exception:  # pragma: no cover - surface error inside UI
    dstar_pipeline = None

try:
    from Models.DSSRunners.Batch_runner import run_batch_experiments
except Exception:
    run_batch_experiments = None


# ===============================================================
# Paths & global defaults
# ===============================================================
APP_ROOT = Path(__file__).resolve().parent
FONT_PATH = APP_ROOT / "fonts" / "hero_title.ttf"
FAVICON_PATH = next(
    (
        path
        for path in [
            APP_ROOT / "assets" / "favicon.png",
            APP_ROOT / "assets" / "logo2.ico",
            APP_ROOT / "assets" / "logo2.png",
        ]
        if path.exists()
    ),
    None,
)
DEFAULT_CONFIG_PATH = Path(
    os.environ.get("DRDSS_CONFIG", APP_ROOT / "config" / "default.json")
)
BATCH_CONFIG_PATH = Path(
    os.environ.get("DRDSS_BATCH_CONFIG", APP_ROOT / "config" / "batch.json")
)
BATCH_QUEUE_DIR = Path(
    os.environ.get("DRDSS_BATCH_QUEUE_DIR", APP_ROOT / "config" / "batches")
)
BATCH_QUEUE_PATH = Path(
    os.environ.get("DRDSS_BATCH_QUEUE", APP_ROOT / "config" / "batch_queue.json")
)
LEGACY_CONFIG_PATH = Path(
    os.environ.get(
        "DRDSS_LEGACY_CONFIG",
        "/Users/philippstockerl/BachelorThesis/project/config/default.json",
    )
)

DEFAULT_CONFIG_TEMPLATE: Dict[str, Any] = {
    "seed": 777,
    "paths": {
        "data_root": "./project/Data/Set_777_N50_cs1_Exponential_anis0.5_var1.0_ls30.0_T10_conn4_norm1",
    },
    "srf": {
        "grid_size": 50,
        "cell_size": 1,
        "kernel": "Exponential",
        "variance": 1.0,
        "nu": 1.5,
        "alpha": 1.0,
        "length_scale": 30.0,
        "anis": 0.5,
        "use_global_normalization": True,
        "num_scenarios": 10,
    },
    "graph": {"connectivity": "4",
              "step_per_layer": 10},
    "robust_model": {"start_node": 0, "goal_node": "auto"},
    "budgeted_model": {"gamma": 0.0, "nominal_rule": "min"},
    "dstar_lite": {
        "warmstart_mode": "none",   # none / discrete / budgeted
        "max_milestones": 10,
        "export_overlays": True,
        "debug": False,
        "debug_stride": 50,
        "max_steps": 2000,
    },
    "visualizations": {
        "gif_2d": True,
        "gif_3d": True,
        "gif_hist": False,
        "gif_heat": False,
        "gif_violin": False,
        "gif_kde": False,
        "frames": False,
        "frame_2d_legend": False,
        "three_path_overlay": False,
        "robust_path_overlay": False,
        "show_log": True,
        "export_metadata": False,
        "export_overlays": True,
    },
}

DEFAULT_BATCH_TEMPLATE: Dict[str, Any] = {
    "config_id": "batch_default",
    "experiments_root": str(APP_ROOT / "experiments"),
    "results_csv": "batch_results.csv",
    "seeds": [0],
    "algorithms": [
        "discrete",
        "discrete_adaptive",
        "budgeted",
        "dstar",
        "dstar_discrete",
        "dstar_discrete_adaptive",
        "dstar_budgeted",
    ],
    "srf": copy.deepcopy(DEFAULT_CONFIG_TEMPLATE["srf"]),
    "graph": copy.deepcopy(DEFAULT_CONFIG_TEMPLATE["graph"]),
    "robust_model": {
        **copy.deepcopy(DEFAULT_CONFIG_TEMPLATE["robust_model"]),
        "adaptive_window": 1,
        "adaptive_commit": None,
    },
    "budgeted_model": copy.deepcopy(DEFAULT_CONFIG_TEMPLATE["budgeted_model"]),
    "dstar_lite": copy.deepcopy(DEFAULT_CONFIG_TEMPLATE["dstar_lite"]),
    "visualizations": {
        "gif_2d": False,
        "gif_3d": False,
        "gif_hist": False,
        "gif_heat": False,
        "gif_violin": False,
        "gif_kde": False,
        "frames": False,
        "frame_2d_legend": False,
        "three_path_overlay": False,
        "robust_path_overlay": False,
        "show_log": False,
        "export_metadata": True,
        "export_overlays": False,
    },
}

BATCH_ALGORITHM_LABELS: Dict[str, str] = {
    "discrete": "Discrete Uncertainty",
    "discrete_adaptive": "Discrete Uncertainty (Adaptive)",
    "budgeted": "Budgeted Uncertainty",
    "dstar": "DStar Lite",
    "dstar_discrete": "DStar Lite (Discrete)",
    "dstar_discrete_adaptive": "DStar Lite (Discrete Adaptive)",
    "dstar_budgeted": "DStar Lite (Budgeted)",
}

DEFAULT_METRICS: List[Dict[str, Any]] = [
    {"Model": "STRF Generator", "Cost": "-", "Edges": "-", "Run Time": "-", "Replans": "-"},
    {"Model": "Discrete Uncertainty", "Cost": "-", "Edges": "-", "Run Time": "-", "Replans": "-"},
    {"Model": "Discrete Uncertainty Adaptive", "Cost": "-", "Edges": "-", "Run Time": "-", "Replans": "-"},
    {"Model": "Budgeted Uncertainty", "Cost": "-", "Edges": "-", "Run Time": "-", "Replans": "-"},
    {"Model": "DStar Lite", "Cost": "-", "Edges": "-", "Run Time": "-", "Replans": "-"},
    {"Model": "DStar Lite Discrete Uncertainty", "Cost": "-", "Edges": "-", "Run Time": "-", "Replans": "-"},
    {"Model": "DStar Lite Discrete Uncertainty Adaptive", "Cost": "-", "Edges": "-", "Run Time": "-", "Replans": "-"},
    {"Model": "DStar Lite Budgeted Uncertainty", "Cost": "-", "Edges": "-", "Run Time": "-", "Replans": "-"},
]
METRICS_FILENAME = "metrics.json"

Runner = Callable[[Dict[str, Any]], Dict[str, Any]]
T = TypeVar("T")

VIZ_ROW_ONE: List[Tuple[str, str]] = [
    ("gif_2d", "2D Map GIF"),
    ("gif_3d", "3D Map GIF"),
    ("gif_hist", "Hist GIF"),
    ("gif_heat", "Heatmap GIF"),
    ("gif_violin", "Violin GIF"),
    ("gif_kde", "KDE GIF"),
]

VIZ_ROW_TWO: List[Tuple[str, str]] = [
    ("frames", "Export Single Frames"),
    ("frame_2d_legend", "2D Frame Legend"),
    ("three_path_overlay", "3-Path overlay"),
    ("robust_path_overlay", "Robust Path overlay"),
    ("export_overlays", "Model Overlays"),
    ("show_log", "Show Log"),
    ("export_metadata", "Export Metadata"),
]


# ===============================================================
# Configuration helpers
# ===============================================================
def ensure_config_file(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(DEFAULT_CONFIG_TEMPLATE, fh, indent=4)


def merge_defaults(target: Dict[str, Any], template: Dict[str, Any]) -> None:
    for key, value in template.items():
        if key not in target:
            target[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(target.get(key), dict):
            merge_defaults(target[key], value)


def normalize_paths(cfg: Dict[str, Any]) -> None:
    paths = cfg.setdefault("paths", {})
    data_root = paths.get("data_root")
    if not data_root:
        for legacy_key in ("result_root", "scenario_root"):
            legacy_value = paths.get(legacy_key)
            if legacy_value:
                data_root = legacy_value
                break
    if not data_root:
        data_root = DEFAULT_CONFIG_TEMPLATE["paths"]["data_root"]

    paths["data_root"] = str(Path(data_root).expanduser().resolve())
    for legacy_key in ("result_root", "scenario_root", "use_custom_data_root"):
        paths.pop(legacy_key, None)


def ensure_batch_config_file(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(DEFAULT_BATCH_TEMPLATE, fh, indent=4)


def normalize_batch_paths(cfg: Dict[str, Any]) -> None:
    root = cfg.get("experiments_root") or DEFAULT_BATCH_TEMPLATE["experiments_root"]
    cfg["experiments_root"] = str(Path(root).expanduser().resolve())


def load_batch_config() -> Dict[str, Any]:
    ensure_batch_config_file(BATCH_CONFIG_PATH)
    with BATCH_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    merge_defaults(cfg, DEFAULT_BATCH_TEMPLATE)
    normalize_batch_paths(cfg)
    return cfg


def save_batch_config(cfg: Dict[str, Any]) -> None:
    BATCH_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BATCH_CONFIG_PATH.open("w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=4)


def _queue_entry_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = APP_ROOT / path
    return path.resolve()


def _queue_storage_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(APP_ROOT))
    except Exception:
        return str(resolved)


def load_batch_queue(path: Path = BATCH_QUEUE_PATH) -> List[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return []

    if isinstance(payload, dict):
        queue = payload.get("queue", [])
    else:
        queue = payload

    if not isinstance(queue, list):
        return []

    entries: List[str] = []
    seen = set()
    for raw in queue:
        if not isinstance(raw, str) or not raw.strip():
            continue
        normalized = _queue_storage_path(_queue_entry_path(raw))
        if normalized in seen:
            continue
        seen.add(normalized)
        entries.append(normalized)
    return entries


def save_batch_queue(queue: List[str], path: Path = BATCH_QUEUE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(queue, fh, indent=2)


def _queue_filename(config_id: str) -> str:
    name = (config_id or "batch_default").strip() or "batch_default"
    name = name.replace(os.sep, "_")
    return name


def save_batch_queue_config(batch_cfg: Dict[str, Any]) -> Path:
    BATCH_QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    base_name = _queue_filename(batch_config_id(batch_cfg))
    path = BATCH_QUEUE_DIR / f"{base_name}.json"
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = BATCH_QUEUE_DIR / f"{base_name}_{stamp}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(batch_cfg, fh, indent=2)
    return path


def describe_batch_queue_entry(raw: str) -> str:
    path = _queue_entry_path(raw)
    if not path.exists():
        return f"{path.name} (missing)"
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        config_id = str(payload.get("config_id", "")).strip()
        if config_id:
            return f"{config_id} ({path.name})"
    except Exception:
        pass
    return path.name


def batch_config_id(cfg: Dict[str, Any]) -> str:
    config_id = str(cfg.get("config_id", "batch_default")).strip()
    return config_id or "batch_default"


def batch_root(cfg: Dict[str, Any]) -> Path:
    root = cfg.get("experiments_root") or DEFAULT_BATCH_TEMPLATE["experiments_root"]
    return Path(root).expanduser().resolve()


def batch_summary_path(cfg: Dict[str, Any]) -> Path:
    return batch_root(cfg) / batch_config_id(cfg) / "batch_summary.json"


def batch_log_path(cfg: Dict[str, Any]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return batch_root(cfg) / batch_config_id(cfg) / "logs" / f"batch_{stamp}.log"


def batch_queue_log_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return BATCH_QUEUE_DIR / "queue_logs" / f"batch_queue_{stamp}.log"


def write_batch_base_snapshot(cfg: Dict[str, Any], batch_cfg: Dict[str, Any]) -> Path:
    path = batch_root(batch_cfg) / batch_config_id(batch_cfg) / "configs" / "base_config_snapshot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=4)
    return path


def is_process_running(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def tail_file(path: Path, max_lines: int = 200) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()
        return "".join(lines[-max_lines:])
    except Exception:
        return ""


def launch_batch_subprocess(cfg: Dict[str, Any], batch_cfg: Dict[str, Any]) -> Dict[str, Any]:
    base_snapshot = write_batch_base_snapshot(cfg, batch_cfg)
    summary_path = batch_summary_path(batch_cfg)
    log_path = batch_log_path(batch_cfg)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    code = (
        "from analysis.batch_aggregation import run_batch_pipeline;"
        f"run_batch_pipeline(base_cfg={str(base_snapshot)!r}, "
        f"batch_cfg={str(BATCH_CONFIG_PATH)!r}, "
        f"summary_out={str(summary_path)!r}, "
        f"log_path={str(log_path)!r})"
    )
    cmd = [sys.executable, "-c", code]
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            cwd=str(APP_ROOT),
        )

    return {
        "pid": proc.pid,
        "log_path": str(log_path),
        "summary_path": str(summary_path),
        "config_id": batch_config_id(batch_cfg),
        "status": "started",
    }


def launch_batch_queue_subprocess(queue_entries: List[str]) -> Dict[str, Any]:
    if not queue_entries:
        raise RuntimeError("Batch queue is empty.")
    log_path = batch_queue_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    queue_payload = json.dumps(queue_entries)
    code = (
        "import json\n"
        "from pathlib import Path\n"
        "from analysis.batch_aggregation import run_batch_pipeline\n"
        f"queue = json.loads({queue_payload!r})\n"
        f"app_root = Path({str(APP_ROOT)!r})\n"
        "def _resolve(raw):\n"
        "    path = Path(raw).expanduser()\n"
        "    if not path.is_absolute():\n"
        "        path = app_root / path\n"
        "    return path.resolve()\n"
        "for raw in queue:\n"
        "    cfg_path = _resolve(raw)\n"
        "    if not cfg_path.exists():\n"
        "        print(f\"[Queue] Missing batch config: {cfg_path}\", flush=True)\n"
        "        continue\n"
        "    print(f\"[Queue] Running {cfg_path}\", flush=True)\n"
        "    run_batch_pipeline(batch_cfg=str(cfg_path))\n"
        "print(\"[Queue] Done\", flush=True)\n"
    )
    cmd = [sys.executable, "-c", code]
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            cwd=str(APP_ROOT),
        )
    return {
        "pid": proc.pid,
        "log_path": str(log_path),
        "queue_len": len(queue_entries),
        "status": "started",
    }

def derive_num_scenarios(srf_cfg: Dict[str, Any]) -> int:
    """Return desired scenario count, falling back to legacy t_start/t_end/t_step."""
    try:
        num = int(srf_cfg.get("num_scenarios", 0) or 0)
    except Exception:
        num = 0
    if num > 0:
        return num

    try:
        t_start = int(srf_cfg.get("t_start", 0))
    except Exception:
        t_start = 0
    try:
        t_end = int(srf_cfg.get("t_end", t_start + 1))
    except Exception:
        t_end = t_start + 1
    try:
        t_step = int(srf_cfg.get("t_step", 1))
    except Exception:
        t_step = 1

    t_step = max(t_step, 1)
    if t_end <= t_start:
        t_end = t_start + 1
    span = t_end - t_start
    num = (span + t_step - 1) // t_step
    return max(1, num)


def display_base_path(path: Optional[str]) -> str:
    if not path:
        return DEFAULT_CONFIG_TEMPLATE["paths"]["data_root"]
    path_obj = Path(path)
    if path_obj.name.startswith("Set_"):
        return str(path_obj.parent)
    return str(path_obj)


def apply_data_root_result(cfg: Dict[str, Any], result: Optional[Dict[str, Any]]) -> None:
    if not result or "data_root" not in result or not result["data_root"]:
        return
    paths = cfg.setdefault("paths", {})
    prev_root = paths.get("data_root")
    paths["data_root"] = result["data_root"]
    save_config(cfg)
    st.session_state["path_input"] = display_base_path(result["data_root"])
    if result["data_root"] != prev_root:
        refresh_metrics_from_cache(cfg, force_reset=True)


def load_config() -> Dict[str, Any]:
    ensure_config_file(DEFAULT_CONFIG_PATH)
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    merge_defaults(cfg, DEFAULT_CONFIG_TEMPLATE)
    normalize_paths(cfg)
    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_CONFIG_PATH.open("w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=4)

    if LEGACY_CONFIG_PATH.exists():
        with LEGACY_CONFIG_PATH.open("w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=4)


def clone_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(cfg))


# ===============================================================
# Result unwrapping helper
# ===============================================================
def unwrap_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    if "result" in result and isinstance(result["result"], dict):
        return result["result"]
    return result


# ===============================================================
# Metrics cache helpers
# ===============================================================
def metrics_cache_path(data_root: str | Path) -> Path:
    return Path(data_root).expanduser() / METRICS_FILENAME


def load_metrics_cache(data_root: str | Path) -> Optional[List[Dict[str, Any]]]:
    path = metrics_cache_path(data_root)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def save_metrics_cache(table: List[Dict[str, Any]], data_root: str | Path) -> None:
    path = metrics_cache_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(table, fh, indent=2)


def refresh_metrics_from_cache(cfg: Dict[str, Any], force_reset: bool = False) -> None:
    cached = load_metrics_cache(cfg["paths"]["data_root"])
    if cached:
        def nodes_to_edges(value: Any) -> str | int:
            if value in (None, "-", ""):
                return "-"
            try:
                nodes = int(value)
            except (TypeError, ValueError):
                try:
                    nodes = int(float(value))
                except (TypeError, ValueError):
                    return "-"
            return max(0, nodes - 1)

        normalized = []
        for row in cached:
            if not isinstance(row, dict):
                continue
            if "Edges" not in row and "Nodes" in row:
                row = dict(row)
                row["Edges"] = nodes_to_edges(row.get("Nodes"))
                row.pop("Nodes", None)
            normalized.append(row)
        # merge cached rows into defaults by model name
        table = copy.deepcopy(DEFAULT_METRICS)
        lookup = {row["Model"]: row for row in normalized if "Model" in row}
        merged = []
        for row in table:
            merged.append(lookup.get(row["Model"], row))
        st.session_state["metrics"] = merged
        return

    if force_reset or "metrics" not in st.session_state:
        st.session_state["metrics"] = copy.deepcopy(DEFAULT_METRICS)


# ===============================================================
# Session management, logging, and file pickers
# ===============================================================
class TerminalStreamer(io.TextIOBase):
    """Redirect stdout/stderr into a Streamlit placeholder."""

    def __init__(self, placeholder, live: bool):
        super().__init__()
        self.placeholder = placeholder
        self.live = live
        self._buffer = io.StringIO()

    def write(self, s: str) -> int:
        # Write to internal buffer for Streamlit
        self._buffer.write(s)

        # Write to Streamlit UI if enabled
        if self.live and self.placeholder is not None:
            render_terminal_output(self.placeholder, self._buffer.getvalue())

        # Also write to real system terminal (VSCode / console)
        try:
            import sys
            sys.__stdout__.write(s)
            sys.__stdout__.flush()
        except Exception:
            pass  # Don't crash if stdout is missing (Streamlit Cloud etc.)

        return len(s)

    def flush(self) -> None:  # pragma: no cover - compatibility no-op
        pass

    def get_value(self) -> str:
        return self._buffer.getvalue()


def render_terminal_output(placeholder, text: str) -> None:
    if placeholder is None:
        return
    safe = html.escape(text or "")
    placeholder.markdown(
        f"<div class='terminal-box'><pre>{safe}</pre></div>",
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    if "config" not in st.session_state:
        st.session_state["config"] = load_config()

    cfg = st.session_state["config"]
    normalize_paths(cfg)
    merge_defaults(cfg, DEFAULT_CONFIG_TEMPLATE)

    if "batch_config" not in st.session_state:
        st.session_state["batch_config"] = load_batch_config()
    batch_cfg = st.session_state["batch_config"]
    normalize_batch_paths(batch_cfg)
    merge_defaults(batch_cfg, DEFAULT_BATCH_TEMPLATE)

    st.session_state.setdefault("path_input", display_base_path(cfg["paths"]["data_root"]))
    st.session_state.setdefault("metrics", copy.deepcopy(DEFAULT_METRICS))
    st.session_state.setdefault("log_output", "")
    st.session_state.setdefault("pending_action", None)
    st.session_state.setdefault("batch_summary", None)
    st.session_state.setdefault("batch_process", None)
    st.session_state.setdefault("batch_queue_process", None)
    refresh_metrics_from_cache(cfg)


def _pick_directory_macos(initial: str) -> Optional[str]:
    escaped = initial.replace('"', '\\"')
    script = f'''
    set defaultPath to POSIX file "{escaped}"
    set chosenFolder to choose folder default location defaultPath with prompt "Select data root"
    POSIX path of chosenFolder
    '''
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def pick_directory(initial: str | None = None) -> Optional[str]:
    initial = initial or str(Path.home())
    if not Path(initial).exists():
        initial = str(Path.home())
    if sys.platform == "darwin":
        return _pick_directory_macos(initial)

    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        st.warning(f"Folder selection is unavailable: {exc}")
        return None

    root = tk.Tk()
    root.withdraw()
    selection = filedialog.askdirectory(initialdir=initial)
    root.destroy()
    return selection or None


# ===============================================================
# Styling & layout primitives
# ===============================================================
def inject_css() -> None:
    """Inject global CSS (font embedding + dark theme overrides)."""
    font_face = ""
    font_stack = "'Space Grotesk', 'Helvetica Neue', sans-serif"
    if FONT_PATH.exists():
        encoded = b64encode(FONT_PATH.read_bytes()).decode("utf-8")
        font_face = f"""
        @font-face {{
            font-family: 'DroneRoutingTitle';
            src: url(data:font/ttf;base64,{encoded}) format('truetype');
            font-weight: normal;
            font-style: normal;
        }}
        """
        font_stack = "'DroneRoutingTitle', 'Space Grotesk', 'Helvetica Neue', sans-serif"

    css = f"""
        <style>
        {font_face}
        :root {{
            --accent: #f5f5f5;
            --accent-muted: rgba(255, 255, 255, 0.2);
            --panel: #778da9;
            --panel-border: rgba(255, 255, 255, 0.35);
            --text: #f8f9fa;
            --muted: #f8f9fa;
            --font-title: {font_stack};
        }}
        html, body, .stApp {{
            background-color: #415a77 !important;
        }}
        body {{
            --cursor-x: 50%;
            --cursor-y: 50%;
        }}
        .stApp {{
            color: var(--text);
            position: relative;
            min-height: 100vh;
            background-image: radial-gradient(rgba(255, 255, 255, 0.08) 1px, transparent 1px);
            background-size: 20px 20px;
            z-index: 1;
        }}
        body::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            background: radial-gradient(circle 200px at var(--cursor-x, 50%) var(--cursor-y, 50%), rgba(255, 255, 255, 0.05), transparent 80%);
            transition: background 0.1s;
            z-index: -1;
        }}
        .section-marker {{
            display: none;
        }}
        div[data-testid="stVerticalBlock"]:has(> .section-marker) {{
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            padding: 1.4rem;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(20,20,20,0.85);
            transition: box-shadow 0.3s ease, border-color 0.3s ease;
        }}
        div[data-testid="stVerticalBlock"]:has(> .section-marker):hover {{
            border-color: rgba(255,255,255,0.6);
            box-shadow: 0 0 25px rgba(255,255,255,0.25);
        }}
        .hero-title {{
            font-family: var(--font-title);
            font-size: 3.4rem;
            margin-bottom: 0.25rem;
            letter-spacing: 0.08em;
            color: var(--text);
            border-bottom: 2px solid var(--accent);
            padding-bottom: 0.2rem;
            display: inline-block;
        }}
        .path-card, .gif-panel, .param-panel, .action-button, .terminal-box, .viz-panel {{
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            box-shadow: 0 0 18px rgba(0,0,0,0.45);
        }}
        .gif-panel {{
            min-height: 180px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            gap: 0.35rem;
        }}
        .gif-panel img.gif-preview {{
            width: 100%;
            max-width: 260px;
            max-height: 220px;
            object-fit: contain;
            border-radius: 12px;
            border: 1px solid var(--panel-border);
        }}
        .gif-panel .gif-caption {{
            color: var(--muted);
            font-size: 0.9rem;
        }}
        .gif-panel span.label {{
            font-size: 0.9rem;
            color: var(--muted);
        }}
        .run-button button {{
            width: 100%;
            background: linear-gradient(90deg, rgba(255,255,255,0.12), rgba(0,0,0,0));
            color: #e0e1dd;
            border-radius: 12px;
            border: 1px solid var(--panel-border);
        }}
        .terminal-box {{
            font-family: "JetBrains Mono", monospace;
            font-size: 0.85rem;
            min-height: 180px;
            max-height: 320px;
            overflow-y: auto;
            color: var(--text);
        }}
        .highlight-label {{
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.2em;
            color: var(--accent);
        }}
        .stCheckbox>label span {{
            color: #e0e1dd;
        }}
        </style>
        <script>
        document.addEventListener("mousemove", (e) => {{
            document.body.style.setProperty("--cursor-x", `${{e.clientX}}px`);
            document.body.style.setProperty("--cursor-y", `${{e.clientY}}px`);
        }});
        </script>
        """

    st.markdown(css, unsafe_allow_html=True)


def render_gif_panel(title: str, subtitle: str, gif_path: Optional[Path] = None) -> None:
    if gif_path and gif_path.exists():
        encoded = b64encode(gif_path.read_bytes()).decode("utf-8")
        st.markdown(
            f"""
            <div class="gif-panel">
                <span class="label">{subtitle}</span>
                <img src="data:image/gif;base64,{encoded}" alt="{title} preview" class="gif-preview"/>
                <div class="gif-caption">{title} preview</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="gif-panel">
                <span class="label">{subtitle}</span>
                <h3>{title}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_controlled_gif_panel(title: str, enabled: bool) -> None:
    subtitle = "Active" if enabled else "Waiting for checkbox"
    tone = "#f0f0f0" if enabled else "#7c7f8a"
    st.markdown(
        f"""
        <div class="gif-panel" style="border-color:{tone}">
            <span class="label">{subtitle}</span>
            <h4>{title}</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )


@contextlib.contextmanager
def section_block():
    """Reusable container wrapper that applies the glowing hover effect."""
    container = st.container()
    with container:
        st.markdown("<div class='section-marker'></div>", unsafe_allow_html=True)
        yield


def boxed_section(renderer: Callable[..., T], *args, **kwargs) -> T:
    """Helper that runs a renderer inside a styled section box."""
    with section_block():
        return renderer(*args, **kwargs)


# ===============================================================
# Model runner wrappers
# ===============================================================
def run_strf(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if strf_generator is None:
        raise RuntimeError("STRF generator is unavailable in this environment.")
    result = strf_generator(clone_config(cfg))
    return result


def run_discrete(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if run_discrete_pipeline is None:
        raise RuntimeError("Discrete Uncertainty model is unavailable in this environment.")
    return run_discrete_pipeline(clone_config(cfg))


def run_discrete_adaptive(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if run_discrete_pipeline is None:
        raise RuntimeError("Adaptive Discrete Uncertainty model is unavailable in this environment.")
    cfg_local = clone_config(cfg)
    cfg_local.setdefault("robust_model", {})["discrete_mode"] = "adaptive"
    return run_discrete_pipeline(cfg_local)


def run_dstar(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if dstar_pipeline is None:
        raise RuntimeError("DStar Lite runner is unavailable in this environment.")
    return dstar_pipeline(clone_config(cfg))


def run_batch(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if run_batch_experiments is None:
        raise RuntimeError("Batch runner is unavailable in this environment.")
    batch_cfg = st.session_state.get("batch_config")
    if not isinstance(batch_cfg, dict):
        raise RuntimeError("Batch config is missing or invalid.")
    active = st.session_state.get("batch_process")
    if isinstance(active, dict) and is_process_running(active.get("pid")):
        return {
            "status": "already_running",
            "pid": active.get("pid"),
            "log_path": active.get("log_path"),
            "summary_path": active.get("summary_path"),
        }
    info = launch_batch_subprocess(clone_config(cfg), clone_config(batch_cfg))
    st.session_state["batch_process"] = info
    return info


def run_batch_queue(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if run_batch_experiments is None:
        raise RuntimeError("Batch runner is unavailable in this environment.")
    active = st.session_state.get("batch_queue_process")
    if isinstance(active, dict) and is_process_running(active.get("pid")):
        return {
            "status": "already_running",
            "pid": active.get("pid"),
            "log_path": active.get("log_path"),
            "queue_len": active.get("queue_len"),
        }
    queue_entries = load_batch_queue()
    if not queue_entries:
        raise RuntimeError("Batch queue is empty.")
    info = launch_batch_queue_subprocess(queue_entries)
    st.session_state["batch_queue_process"] = info
    return info


ACTIONS: Dict[str, Tuple[str, Optional[Runner]]] = {
    "run_strf": ("STRF Generator", run_strf),
    "run_discrete": ("Discrete Uncertainty", run_discrete),
    "run_discrete_adaptive": ("Discrete Uncertainty Adaptive", run_discrete_adaptive),
    "run_budgeted_uncertainty": ("Budgeted Uncertainty", run_budgeted_uncertainty),
    "run_dstar": ("DStar Lite", run_dstar),
    "run_dstar_discrete": ("DStar Lite Discrete Uncertainty", run_dstar),
    "run_dstar_discrete_adaptive": ("DStar Lite Discrete Uncertainty Adaptive", run_dstar),
    "run_dstar_budgeted": ("DStar Lite Budgeted Uncertainty", run_dstar),
    "run_batch": ("Batch Runner", run_batch),
    "run_batch_queue": ("Batch Queue Runner", run_batch_queue),
}


# ===============================================================
# UI building blocks
# ===============================================================
def update_metrics(model: str, result: Optional[Dict[str, Any]], runtime: float, cfg: Dict[str, Any]) -> None:
    return


def refresh_metrics_from_json(cfg: Dict[str, Any]) -> None:
    table = copy.deepcopy(DEFAULT_METRICS)

    for row in table:
        model_name = row["Model"]

        try:
            stored = load_comparison_result(cfg, model_name)
        except Exception as e:
            stored = None

        if stored is None:
            # --- model failed or file missing ---
            row["Run Time"] = "-"
            row["Nominal Cost"] = "-"
            row["Deviation Cost"] = "-"
            row["Cost"] = "-"
            row["Edges"] = "-"
            row["Replans"] = "-"
            continue

        try:
            # --- runtime ---
            row["Run Time"] = f"{float(stored.get('runtime', 0)):.2f}s"

            # --- cost output ---
            if "objective" in stored:
                row["Cost"] = f"{float(stored['objective']):.3f}"
            elif "cost" in stored:
                row["Cost"] = f"{float(stored['cost']):.3f}"
            else:
                row["Cost"] = "-"

            # --- nominal / deviation ---
            nom = stored.get("nominal_cost")
            rob = stored.get("robust_cost")

            if nom is None or nom == "-" or nom == "":
                row["Nominal Cost"] = "-"
            else:
                row["Nominal Cost"] = f"{float(nom):.3f}"

            if rob is None or rob == "-" or rob == "":
                row["Deviation Cost"] = "-"
            else:
                row["Deviation Cost"] = f"{float(rob):.3f}"

            # --- nodes ---
            path = stored.get("node_path", [])
            if isinstance(path, list):
                row["Edges"] = max(0, len(path) - 1)
            else:
                row["Edges"] = "-"

            # --- replans (DStar Lite only) ---
            row["Replans"] = stored.get("replans", "-")

        except Exception:
            # fallback if some fields are broken
            row["Run Time"] = "-"
            row["Nominal Cost"] = "-"
            row["Deviation Cost"] = "-"
            row["Cost"] = "-"
            row["Edges"] = "-"
            row["Replans"] = "-"

    # Always try to save the metrics.json cache
    try:
        save_metrics_cache(table, cfg["paths"]["data_root"])
    except Exception:
        pass  # NEVER crash

    st.session_state["metrics"] = table


def render_viz_checkboxes(cfg: Dict[str, Any]) -> Tuple[bool, bool]:
    changed = False
    viz_cfg = cfg["visualizations"]

    row = st.columns(len(VIZ_ROW_ONE))
    for (key, label), col in zip(VIZ_ROW_ONE, row):
        with col:
            val = col.checkbox(label, value=viz_cfg.get(key, False))
            if val != viz_cfg.get(key):
                viz_cfg[key] = val
                changed = True

    row = st.columns(len(VIZ_ROW_TWO))
    for (key, label), col in zip(VIZ_ROW_TWO, row):
        with col:
            val = col.checkbox(label, value=viz_cfg.get(key, False))
            if val != viz_cfg.get(key):
                viz_cfg[key] = val
                changed = True

    show_log = viz_cfg.get("show_log", True)
    return show_log, changed


def render_visualization_section(cfg: Dict[str, Any]) -> Tuple[bool, bool]:
    st.markdown('<div class="highlight-label">Visualization Outputs</div>', unsafe_allow_html=True)
    return render_viz_checkboxes(cfg)


def path_selector(cfg: Dict[str, Any]) -> None:
    st.markdown('<div class="highlight-label">Set Path for this Set</div>', unsafe_allow_html=True)
    cols = st.columns([8, 1, 2])
    with cols[0]:
        default_value = st.session_state.get(
            "path_input", display_base_path(cfg["paths"]["data_root"])
        )
        st.session_state["path_input"] = st.text_input(
            "Data root path",
            value=default_value,
            label_visibility="collapsed",
        )
    with cols[1]:
        if st.button("📁", help="Pick folder", width="stretch"):
            selection = pick_directory(st.session_state["path_input"])
            if selection:
                st.session_state["path_input"] = selection
    with cols[2]:
        if st.button("Update Path", width="stretch"):
            new_path = st.session_state["path_input"].strip().rstrip("/")

            if new_path:
                cfg["paths"]["data_root"] = str(Path(new_path).expanduser().resolve())
                save_config(cfg)
                refresh_metrics_from_cache(cfg, force_reset=True)
                st.success("Set path updated.")


def render_action_dashboard() -> None:
    cfg = st.session_state["config"]
    data_root = Path(cfg["paths"]["data_root"]).expanduser()

    def set_warmstart(mode: str) -> None:
        cfg.setdefault("dstar_lite", {})["warmstart_mode"] = mode
        save_config(cfg)

    # -------------------------------
    # GIF PATHS
    # -------------------------------
    strf_gif                = data_root / "animation.gif"
    robust_gif              = data_root / "DiscreteUncertainty"                  / "overlays" / "discrete_uncertainty_path_overlay.gif"
    robust_adaptive_gif     = data_root / "DiscreteUncertaintyAdaptive"          / "overlays" / "discrete_uncertainty_adaptive_path_overlay.gif"
    budgeted_gif            = data_root / "BudgetedUncertainty"                  / "overlays" / "budgeted_uncertainty_path_overlay.gif"
    dstar_gif               = data_root / "DStarLite"                            / "overlays" / "dstar_overlay_animation.gif"
    dstar_disc_gif          = data_root / "DStarLiteDiscreteUncertainty"         / "overlays" / "dstar_overlay_animation.gif"
    dstar_disc_adaptive_gif = data_root / "DStarLiteDiscreteAdaptiveUncertainty" / "overlays" / "dstar_overlay_animation.gif"
    dstar_budgeted_gif      = data_root / "DStarLiteBudgetedUncertainty"         / "overlays" / "dstar_overlay_animation.gif"

    # -------------------------------
    # ROW 1: STRF ONLY
    # -------------------------------
    gif_cols = st.columns(1)
    with gif_cols[0]:
        render_gif_panel("Spatio-Temporal Random Field Generator", "Scenario GIF", strf_gif)

    btn_cols = st.columns(1)
    if btn_cols[0].button("Run STRF Generator", key="btn_strf", use_container_width=True):
        st.session_state["pending_action"] = "run_strf"

    # -------------------------------
    # ROW 2: ROBUST MODELS (DISCRETE / ADAPTIVE / BUDGETED)
    # -------------------------------
    row2_cards = [
        ("Discrete Uncertainty Model", "Discrete Uncertainty Path Overlay GIF", robust_gif),
        ("Adaptive Discrete Uncertainty Model", "Adaptive Discrete Path Overlay GIF", robust_adaptive_gif),
        ("Budgeted Uncertainty Model", "Budgeted Uncertainty Path Overlay GIF", budgeted_gif),
    ]

    gif_cols2 = st.columns(3)
    for (title, subtitle, path), col in zip(row2_cards, gif_cols2):
        with col:
            render_gif_panel(title, subtitle, path)

    btn_cols2 = st.columns(3)
    if btn_cols2[0].button("Run Discrete Uncertainty Model", key="btn_discrete", use_container_width=True):
        st.session_state["pending_action"] = "run_discrete"
    if btn_cols2[1].button("Run Adaptive Discrete Uncertainty Model", key="btn_discrete_adaptive", use_container_width=True):
        st.session_state["pending_action"] = "run_discrete_adaptive"
    if btn_cols2[2].button("Run Budgeted Uncertainty Model", key="btn_budgeted", use_container_width=True):
        st.session_state["pending_action"] = "run_budgeted_uncertainty"

    # -------------------------------
    # ROW 3: D* LITE BASELINE
    # -------------------------------
    gif_cols3 = st.columns(1)
    with gif_cols3[0]:
        render_gif_panel("DStar Lite Baseline", "DStar Lite Baseline Path GIF", dstar_gif)

    btn_cols3 = st.columns(1)
    if btn_cols3[0].button("Run DStar Lite", key="btn_dstar", use_container_width=True):
        st.session_state["pending_action"] = "run_dstar"

    # -------------------------------
    # ROW 4: D* LITE WITH WARM-STARTS
    # -------------------------------
    row4_cards = [
        ("DStar Lite w. Discrete Uncertainty Optimal Path", "DStar Lite Restricted Path GIF", dstar_disc_gif),
        ("DStar Lite w. Adaptive Discrete Uncertainty Optimal Path", "DStar Lite Restricted Path GIF", dstar_disc_adaptive_gif),
        ("DStar Lite w. Budgeted Uncertainty Optimal Path", "DStar Lite Restricted Path GIF", dstar_budgeted_gif),
    ]

    gif_cols4 = st.columns(3)
    for (title, subtitle, path), col in zip(row4_cards, gif_cols4):
        with col:
            render_gif_panel(title, subtitle, path)

    btn_cols4 = st.columns(3)
    if btn_cols4[0].button("Run DStar Lite with Discrete Uncertainty Path", key="btn_dstar_discrete", use_container_width=True):
        set_warmstart("discrete")
        st.session_state["pending_action"] = "run_dstar_discrete"
    if btn_cols4[1].button("Run DStar Lite with Adaptive Discrete Uncertainty Path", key="btn_dstar_discrete_adaptive", use_container_width=True):
        set_warmstart("discrete_adaptive")
        st.session_state["pending_action"] = "run_dstar_discrete_adaptive"
    if btn_cols4[2].button("Run DStar Lite with Budgeted Uncertainty Path", key="btn_dstar_budgeted", use_container_width=True):
        set_warmstart("budgeted")
        st.session_state["pending_action"] = "run_dstar_budgeted"

    # -------------------------------
    # Restart all GIFs simultaneously
    # -------------------------------
    st.markdown(
        """
        <script>
        const gifs = document.querySelectorAll('.gif-panel img.gif-preview');
        const restart = () => {
            gifs.forEach(img => {
                const src = img.src;
                img.src = '';
                img.src = src;
            });
        };
        if (document.readyState === 'complete') {
            restart();
        } else {
            window.addEventListener('load', restart, { once: true });
        }
        </script>
        """,
        unsafe_allow_html=True,
    )

def parse_gamma_input(text: str):
    return float(text.strip())

def parse_list_field(s):
    return [float(x) for x in s.replace(" ", "").split(",")]


def parse_seed_list(raw: str) -> List[int]:
    seeds: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            step = 1 if end >= start else -1
            seeds.extend(list(range(start, end + step, step)))
        else:
            seeds.append(int(part))
    return seeds


def parameter_form(cfg: Dict[str, Any]) -> None:
    with st.expander("Model Parameters", expanded=False):
        with st.form("parameter_form"):
            st.markdown(
                "<div class='highlight-label'>Tune each model beneath its column</div>",
                unsafe_allow_html=True,
            )
            col_strf, col_discrete, col_budgeted, col_dstar = st.columns(4)

            # STRF params under STRF column
            with col_strf:
                st.markdown("#### STRF")
                seed = st.number_input("Seed", min_value=0, step=1, value=int(cfg["seed"]))
                grid_size = st.number_input(
                    "Grid Size", min_value=4, step=2, value=int(cfg["srf"]["grid_size"])
                )
                cell_size = st.number_input(
                    "Cell Size", min_value=1, value=int(cfg["srf"]["cell_size"])
                )
                kernel = st.text_input("Kernel Type", value=cfg["srf"]["kernel"])
                nu = st.number_input(
                    "Matern ν",
                    min_value=0.1,
                    step=0.1,
                    value=float(cfg["srf"].get("nu", 1.5)),
                    disabled=str(kernel).lower() != "matern",
                )
                alpha = st.number_input(
                    "Stable alpha (a)",
                    min_value=0.1,
                    step=0.1,
                    value=float(cfg["srf"].get("alpha", 1.0)),
                    disabled=str(kernel).lower() != "stable",
                )
                variance = st.number_input(
                    "Variance", min_value=0.0, step=0.1, value=float(cfg["srf"]["variance"])
                )
                length_scale_raw = st.text_input(
                    "Length Scale (Lx, Ly, Lt)",
                    value="{}, {}, {}".format(*([cfg["srf"].get("length_scale", 30)]*3)
                    if isinstance(cfg["srf"].get("length_scale"), (int,float))
                    else cfg["srf"]["length_scale"])
                )

                anis_raw = st.text_input(
                    "Anisotropy (ax, ay, at)",
                    value="{}, {}, {}".format(*([cfg["srf"].get("anis", 1.0)]*3)
                    if isinstance(cfg["srf"].get("anis"), (int,float))
                    else cfg["srf"]["anis"])
                )
                normalization = st.checkbox(
                    "Global Normalization", value=bool(cfg["srf"]["use_global_normalization"])
                )
                num_scenarios_default = derive_num_scenarios(cfg["srf"])
                num_scenarios = st.number_input(
                    "Number of Scenarios",
                    min_value=1,
                    step=1,
                    value=int(num_scenarios_default),
                )
                st.text_input("Connectivity", value="4", disabled=True)
                connectivity = "4"

            # Discrete Uncertainty params under its column
            with col_discrete:
                st.markdown("#### Discrete Uncertainty")
                start_node = st.number_input(
                    "Start Node",
                    min_value=0,
                    step=1,
                    value=int(cfg["robust_model"]["start_node"]),
                )
                goal_value = str(cfg["robust_model"]["goal_node"])
                goal_node = st.text_input("Goal Node", value=goal_value)
                st.caption("Goal set to 'auto' uses the farthest node.")

            # Budgeted robust params
            with col_budgeted:
                st.markdown("#### Budgeted Robust")

                # --- Gamma input (scalar) ---
                gamma_raw = st.text_input(
                    "Gamma (scalar)",
                    value=str(cfg["budgeted_model"].get("gamma_input", "1.0")),
                )
                st.caption("Example: 1.5")

            # DStar Lite params
            with col_dstar:
                st.markdown("#### DStar Lite")
                warmstart_options = ["none", "discrete", "discrete_adaptive", "budgeted"]
                warmstart_current = str(cfg["dstar_lite"].get("warmstart_mode", "none"))
                warmstart_mode = st.selectbox(
                    "Warm-Start Mode",
                    options=warmstart_options,
                    index=warmstart_options.index(warmstart_current) if warmstart_current in warmstart_options else 0,
                )
                max_milestones = st.number_input(
                    "Max Milestones (beacons)",
                    min_value=1,
                    step=1,
                    value=int(cfg["dstar_lite"].get("max_milestones", 10)),
                )

            submitted = st.form_submit_button("Save Parameters", use_container_width=True)
            if submitted:
                cfg["seed"] = int(seed)
                cfg["srf"].update(
                    {
                        "grid_size": int(grid_size),
                        "cell_size": int(cell_size),
                        "kernel": kernel,
                        "nu": float(nu),
                        "alpha": float(alpha),
                        "variance": float(variance),
                        "length_scale": parse_list_field(length_scale_raw),
                        "anis":         parse_list_field(anis_raw),
                        "use_global_normalization": normalization,
                        "num_scenarios": int(num_scenarios),
                        "t_start": 0,
                        "t_end": int(num_scenarios),
                        "t_step": 1,
                    }
                )
                cfg["graph"]["connectivity"] = connectivity
                cfg["robust_model"]["start_node"] = int(start_node)
                cfg["robust_model"]["goal_node"] = goal_node.strip() or "auto"

                gamma_value = parse_gamma_input(gamma_raw)
                cfg["budgeted_model"]["gamma_input"] = gamma_raw
                cfg["budgeted_model"]["gamma_value"] = gamma_value
                cfg["budgeted_model"]["gamma"] = gamma_value


                cfg["dstar_lite"].pop("mode", None)  # clean up legacy field
                cfg["dstar_lite"].update(
                    {
                        "warmstart_mode": warmstart_mode,
                        "max_milestones": int(max_milestones),
                    }
                )
                save_config(cfg)
                st.success("Parameters saved to default.json")


def batch_mode_form(cfg: Dict[str, Any]) -> None:
    batch_cfg = st.session_state.get("batch_config", {})
    if not isinstance(batch_cfg, dict):
        batch_cfg = {}
    merge_defaults(batch_cfg, DEFAULT_BATCH_TEMPLATE)

    with st.expander("Batch Mode", expanded=False):
        with st.form("batch_form"):
            st.markdown(
                "<div class='highlight-label'>Batch experiments (STRF + model runs)</div>",
                unsafe_allow_html=True,
            )
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                config_id = st.text_input(
                    "Batch Config ID",
                    value=str(batch_cfg.get("config_id", "batch_default")),
                )
                experiments_root = st.text_input(
                    "Experiments Folder",
                    value=str(batch_cfg.get("experiments_root", APP_ROOT / "experiments")),
                )
                results_csv = st.text_input(
                    "Results CSV Name",
                    value=str(batch_cfg.get("results_csv", "batch_results.csv")),
                )

            with col_b:
                seeds_raw = st.text_input(
                    "Seeds (comma-separated or ranges like 1-5)",
                    value=", ".join(str(s) for s in batch_cfg.get("seeds", [])),
                )
                algo_options = list(BATCH_ALGORITHM_LABELS.keys())
                algo_default = [a for a in batch_cfg.get("algorithms", []) if a in algo_options]
                algorithms = st.multiselect(
                    "Algorithms",
                    options=algo_options,
                    default=algo_default,
                    format_func=lambda key: BATCH_ALGORITHM_LABELS.get(key, key),
                )

            with col_c:
                st.checkbox("Export STRF Metadata", value=True, disabled=True)
                export_overlays = st.checkbox(
                    "Export Model Overlays",
                    value=bool(batch_cfg.get("visualizations", {}).get("export_overlays", False)),
                )
                try:
                    seeds_preview = parse_seed_list(seeds_raw)
                    total_runs = len(seeds_preview) * max(1, len(algorithms))
                    st.caption(f"Estimated runs: {total_runs}")
                except Exception:
                    st.caption("Estimated runs: - (invalid seeds)")

            st.markdown("#### STRF Parameters")
            col_srf1, col_srf2, col_srf3 = st.columns(3)
            srf_cfg = batch_cfg.get("srf", {})
            graph_cfg = batch_cfg.get("graph", {})

            with col_srf1:
                grid_size = st.number_input(
                    "Grid Size",
                    min_value=4,
                    step=2,
                    value=int(srf_cfg.get("grid_size", cfg["srf"]["grid_size"])),
                )
                cell_size = st.number_input(
                    "Cell Size",
                    min_value=1,
                    value=int(srf_cfg.get("cell_size", cfg["srf"]["cell_size"])),
                )
                kernel = st.text_input(
                    "Kernel Type",
                    value=str(srf_cfg.get("kernel", cfg["srf"]["kernel"])),
                )

            with col_srf2:
                nu = st.number_input(
                    "Matern ν",
                    min_value=0.1,
                    step=0.1,
                    value=float(srf_cfg.get("nu", cfg["srf"].get("nu", 1.5))),
                    disabled=str(kernel).lower() != "matern",
                )
                alpha = st.number_input(
                    "Stable alpha (a)",
                    min_value=0.1,
                    step=0.1,
                    value=float(srf_cfg.get("alpha", cfg["srf"].get("alpha", 1.0))),
                    disabled=str(kernel).lower() != "stable",
                )
                variance = st.number_input(
                    "Variance",
                    min_value=0.0,
                    step=0.1,
                    value=float(srf_cfg.get("variance", cfg["srf"]["variance"])),
                )
                length_scale_raw = st.text_input(
                    "Length Scale (Lx, Ly, Lt)",
                    value="{}, {}, {}".format(*([srf_cfg.get("length_scale", 30)] * 3)
                    if isinstance(srf_cfg.get("length_scale"), (int, float))
                    else srf_cfg.get("length_scale", [30, 30, 10])),
                )

            with col_srf3:
                anis_raw = st.text_input(
                    "Anisotropy (ax, ay, at)",
                    value="{}, {}, {}".format(*([srf_cfg.get("anis", 1.0)] * 3)
                    if isinstance(srf_cfg.get("anis"), (int, float))
                    else srf_cfg.get("anis", [1.0, 1.0, 1.0])),
                )
                normalization = st.checkbox(
                    "Global Normalization",
                    value=bool(srf_cfg.get("use_global_normalization", True)),
                )
                num_scenarios_default = derive_num_scenarios(srf_cfg)
                num_scenarios = st.number_input(
                    "Number of Scenarios",
                    min_value=1,
                    step=1,
                    value=int(num_scenarios_default),
                )
                st.text_input("Connectivity", value="4", disabled=True)
                connectivity = "4"

            st.markdown("#### Model Parameters")
            col_m1, col_m2, col_m3 = st.columns(3)
            robust_cfg = batch_cfg.get("robust_model", {})
            budget_cfg = batch_cfg.get("budgeted_model", {})
            dstar_cfg = batch_cfg.get("dstar_lite", {})

            with col_m1:
                start_node = st.number_input(
                    "Start Node",
                    min_value=0,
                    step=1,
                    value=int(robust_cfg.get("start_node", 0)),
                )
                goal_value = str(robust_cfg.get("goal_node", "auto"))
                goal_node = st.text_input("Goal Node", value=goal_value)
                adaptive_window = st.number_input(
                    "Adaptive Window",
                    min_value=1,
                    step=1,
                    value=int(robust_cfg.get("adaptive_window", 1)),
                )
                adaptive_commit_raw = st.text_input(
                    "Adaptive Commit Length (blank=auto)",
                    value="" if robust_cfg.get("adaptive_commit") is None else str(robust_cfg.get("adaptive_commit")),
                )

            with col_m2:
                gamma_raw = st.text_input(
                    "Gamma (scalar)",
                    value=str(budget_cfg.get("gamma_input", budget_cfg.get("gamma", "1.0"))),
                )
                nominal_rule_options = ["min", "avg"]
                nominal_rule_default = str(budget_cfg.get("nominal_rule", "min")).lower()
                if nominal_rule_default not in nominal_rule_options:
                    nominal_rule_default = "min"
                nominal_rule = st.selectbox(
                    "Nominal Rule (budgeted)",
                    options=nominal_rule_options,
                    index=nominal_rule_options.index(nominal_rule_default),
                )

            with col_m3:
                max_milestones = st.number_input(
                    "Max Milestones (beacons)",
                    min_value=1,
                    step=1,
                    value=int(dstar_cfg.get("max_milestones", 10)),
                )

            save_btn = st.form_submit_button("Save Batch Settings", use_container_width=True)
            queue_btn = st.form_submit_button("Save to Queue", use_container_width=True)
            run_btn = st.form_submit_button("Run Batch Now", use_container_width=True)

            if save_btn or run_btn or queue_btn:
                try:
                    seeds = parse_seed_list(seeds_raw)
                except Exception as exc:
                    st.error(f"Invalid seeds: {exc}")
                    return
                if (run_btn or queue_btn) and (not seeds or not algorithms):
                    st.error("Batch run requires at least one seed and one algorithm.")
                    return

                adaptive_commit = None
                adaptive_commit_raw = adaptive_commit_raw.strip()
                if adaptive_commit_raw:
                    adaptive_commit = int(adaptive_commit_raw)

                batch_cfg["config_id"] = config_id.strip() or "batch_default"
                batch_cfg["experiments_root"] = experiments_root.strip() or str(APP_ROOT / "experiments")
                batch_cfg["results_csv"] = results_csv.strip() or "batch_results.csv"
                batch_cfg["seeds"] = seeds
                batch_cfg["algorithms"] = algorithms

                batch_cfg["srf"].update(
                    {
                        "grid_size": int(grid_size),
                        "cell_size": int(cell_size),
                        "kernel": kernel,
                        "nu": float(nu),
                        "alpha": float(alpha),
                        "variance": float(variance),
                        "length_scale": parse_list_field(length_scale_raw),
                        "anis": parse_list_field(anis_raw),
                        "use_global_normalization": normalization,
                        "num_scenarios": int(num_scenarios),
                        "t_start": 0,
                        "t_end": int(num_scenarios),
                        "t_step": 1,
                    }
                )
                batch_cfg["graph"]["connectivity"] = connectivity

                batch_cfg["robust_model"].update(
                    {
                        "start_node": int(start_node),
                        "goal_node": goal_node.strip() or "auto",
                        "adaptive_window": int(adaptive_window),
                        "adaptive_commit": adaptive_commit,
                    }
                )

                gamma_value = parse_gamma_input(gamma_raw)
                batch_cfg["budgeted_model"]["gamma_input"] = gamma_raw
                batch_cfg["budgeted_model"]["gamma_value"] = gamma_value
                batch_cfg["budgeted_model"]["gamma"] = gamma_value
                batch_cfg["budgeted_model"]["nominal_rule"] = nominal_rule

                batch_cfg["dstar_lite"].update(
                    {
                        "max_milestones": int(max_milestones),
                    }
                )

                batch_cfg.setdefault("visualizations", {})
                batch_cfg["visualizations"]["export_metadata"] = True
                batch_cfg["visualizations"]["export_overlays"] = bool(export_overlays)

                normalize_batch_paths(batch_cfg)
                st.session_state["batch_config"] = batch_cfg

                if save_btn or run_btn:
                    save_batch_config(batch_cfg)
                    st.success("Batch settings saved to batch.json")
                    if run_btn:
                        st.session_state["pending_action"] = "run_batch"

                if queue_btn:
                    queue_path = save_batch_queue_config(batch_cfg)
                    queue_entries = load_batch_queue()
                    storage_path = _queue_storage_path(queue_path)
                    if storage_path not in queue_entries:
                        queue_entries.append(storage_path)
                    save_batch_queue(queue_entries)
                    st.success(f"Queued batch config saved to {queue_path.name}")

        st.markdown("#### Batch Queue")
        queue_entries = load_batch_queue()
        st.caption(f"Queued configs: {len(queue_entries)}")
        if queue_entries:
            for idx, raw in enumerate(queue_entries):
                col_left, col_right = st.columns([0.9, 0.1])
                with col_left:
                    st.write(describe_batch_queue_entry(raw))
                with col_right:
                    if st.button("X", key=f"queue_remove_{idx}"):
                        updated = queue_entries[:idx] + queue_entries[idx + 1:]
                        save_batch_queue(updated)
                        st.rerun()
        else:
            st.caption("Queue is empty.")

        if st.button("Run Batch Queue", use_container_width=True):
            if not queue_entries:
                st.error("Batch queue is empty.")
            else:
                st.session_state["pending_action"] = "run_batch_queue"

        queue_process = st.session_state.get("batch_queue_process")
        if isinstance(queue_process, dict):
            pid = queue_process.get("pid")
            running = is_process_running(pid)
            st.markdown("#### Batch Queue Status")
            st.write(f"PID: {pid}" if pid else "PID: -")
            st.write(f"Status: {'running' if running else 'finished'}")
            log_path = queue_process.get("log_path")
            if log_path:
                st.write(f"Log: `{log_path}`")
            if st.button("Show last 200 queue log lines", key="batch_queue_log_tail"):
                tail = tail_file(Path(log_path), max_lines=200) if log_path else ""
                st.text_area("Queue log tail", value=tail or "(no log output yet)", height=240)
            if not running and st.button("Clear queue process", key="batch_queue_clear"):
                st.session_state["batch_queue_process"] = None

        process = st.session_state.get("batch_process")
        if isinstance(process, dict):
            pid = process.get("pid")
            running = is_process_running(pid)
            st.markdown("#### Batch Process Status")
            st.write(f"PID: {pid}" if pid else "PID: -")
            st.write(f"Status: {'running' if running else 'finished'}")
            log_path = process.get("log_path")
            summary_path = process.get("summary_path")
            if log_path:
                st.write(f"Log: `{log_path}`")
            if summary_path:
                st.write(f"Summary: `{summary_path}`")
            if st.button("Show last 200 log lines", key="batch_log_tail"):
                tail = tail_file(Path(log_path), max_lines=200) if log_path else ""
                st.text_area("Batch log tail", value=tail or "(no log output yet)", height=240)
            if not running and st.button("Clear batch process", key="batch_clear"):
                st.session_state["batch_process"] = None

        summary = st.session_state.get("batch_summary")
        if isinstance(summary, dict):
            st.markdown("#### Last Batch Summary")
            st.json(summary)


def render_log_terminal(show_log: bool) -> Tuple[Any, bool]:
    st.markdown("#### Log Terminal")
    placeholder = st.empty()
    if show_log:
        render_terminal_output(placeholder, st.session_state.get("log_output", ""))
    else:
        placeholder.markdown(
            "<div class='terminal-box'>Terminal hidden (enable checkbox to stream)</div>",
            unsafe_allow_html=True,
        )
    return placeholder, show_log


def render_metrics_table() -> None:
    df = pd.DataFrame(st.session_state["metrics"])
    for col in ("Cost", "Nominal Cost", "Deviation Cost", "Edges", "Run Time", "Replans"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    st.dataframe(df, width="stretch", hide_index=True)


def render_metrics_summary(cfg: Dict[str, Any]) -> None:
    st.markdown("### Visualization Outputs")

    viz_cfg = cfg["visualizations"]
    data_root = Path(cfg["paths"]["data_root"]).expanduser()

    # Map keys → GIF filenames
    gif_map = {
        "three_path_overlay": data_root / "three_path_overlay.gif",
        "robust_path_overlay": data_root / "robust_path_overlay.gif",
        "gif_3d": data_root / "animation_3d.gif",
        "gif_heat": data_root / "animation_heat.gif",
        "gif_hist": data_root / "animation_hist.gif",
        "gif_kde": data_root / "animation_kde.gif",
        "gif_violin": data_root / "animation_violin.gif",
    }

    # ----- ROW 1: 4 cards -----
    row1 = st.columns(2)
    row1_items = [
        ("3D Map GIF",        "gif_3d"),
        ("Heat GIF",          "gif_heat"),
    ]

    for col, (title, key) in zip(row1, row1_items):
        with col:
            if viz_cfg.get(key, False) and gif_map[key].exists():
                render_gif_panel(title, title, gif_map[key])
            else:
                render_controlled_gif_panel(title, viz_cfg.get(key, False))

    # ----- ROW 2: 3 cards -----
    row2 = st.columns(3)
    row2_items = [
        ("Histogram GIF",     "gif_hist"),
        ("KDE GIF",           "gif_kde"),
        ("Violin GIF",        "gif_violin"),
    ]

    for col, (title, key) in zip(row2, row2_items):
        with col:
            if viz_cfg.get(key, False) and gif_map[key].exists():
                render_gif_panel(title, title, gif_map[key])
            else:
                render_controlled_gif_panel(title, viz_cfg.get(key, False))

    # ----- Metrics below -----
    st.markdown("### Model Comparison")
    render_metrics_table()


def load_interpretation_report(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    path = Path(cfg["paths"]["data_root"]).expanduser() / "metadata" / "statistical_interpretation.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def render_interpretation_report(cfg: Dict[str, Any]) -> None:
    st.markdown("### Statistical Interpretation (Deterministic)")
    report = load_interpretation_report(cfg)
    if report is None:
        st.info("No interpretation file found. Enable 'Export Metadata' and rerun the STRF generator.")
        return

    sections = [
        ("STRF Parameters", "strf_parameters"),
        ("3D Cost Surface GIF", "cost_surface_gif"),
        ("Heatmap GIF", "heatmap_gif"),
        ("Histogram GIF", "histogram_gif"),
        ("KDE GIF", "kde_gif"),
        ("Violin GIF", "violin_gif"),
    ]

    for title, key in sections:
        entry = report.get(key)
        if not entry:
            continue
        with st.expander(title, expanded=False):
            st.markdown(entry.get("interpretation", ""))
            metrics = entry.get("metrics")
            if metrics:
                st.code(json.dumps(metrics, indent=2))

# ===============================================================
# Comparison Utilities (Unified Logic)
# ===============================================================
MODEL_KEY_MAP = {
    "Discrete Uncertainty": "last_result_Discrete_Uncertainty",
    "Discrete Uncertainty Adaptive": "last_result_Discrete_Uncertainty_Adaptive",
    "Budgeted Uncertainty": "last_result_Budgeted_Uncertainty",
    "DStar Lite": "last_result_DStar_Lite",
    "DStar Lite Discrete Uncertainty": "last_result_DStar_Lite_Discrete_Robust",
    "DStar Lite Discrete Uncertainty Adaptive": "last_result_DStar_Lite_Discrete_Adaptive",
    "DStar Lite Budgeted Uncertainty": "last_result_DStar_Lite_Budgeted",
}

def load_model_result(model_name: str):
    cfg = st.session_state.get("config")
    data = load_comparison_result(cfg, model_name)
    if not data:
        return None
    return unwrap_result(data)

def find_path_divergence(pathA, pathB):
    L = min(len(pathA), len(pathB))
    
    # Check node mismatch inside overlapping region
    for i in range(L):
        if pathA[i] != pathB[i]:
            return i, pathA[i], pathB[i]
    
    # If no mismatch but lengths differ → divergence at L
    if len(pathA) != len(pathB):
        return L, (
            pathA[L] if L < len(pathA) else None,
            pathB[L] if L < len(pathB) else None
        )
    
    # Fully identical
    return None

def compare_cost_tables(costA, costB):
    rows = []
    L = min(len(costA), len(costB))
    for i in range(L):
        rowA = costA[i]
        rowB = costB[i]
        rows.append({
            "step": i,
            "arc_A": (rowA.get("u"), rowA.get("v")),
            "cost_A": rowA.get("cost"),
            "arc_B": (rowB.get("u"), rowB.get("v")),
            "cost_B": rowB.get("cost"),
            "delta": (
                None if (rowA.get("cost") is None or rowB.get("cost") is None)
                else (rowB["cost"] - rowA["cost"])
            ),
        })
    return rows

def create_two_path_overlay(resultA, resultB, out_path: Path):

    base = Path(resultA["data_root"])
    frame_paths = sorted(base.glob("scenario_*/field.npy"))
    if not frame_paths:
        field_path = base / "scenario_000" / "field.npy"
        if not field_path.exists():
            raise FileNotFoundError(f"Missing cost field: {field_path}")
        frame_paths = [field_path]

    # Node paths
    pathA = resultA["node_path"]
    pathB = resultB["node_path"]

    # -------------------------------
    # Convert node IDs → (x, y) using coords stored in JSON
    # -------------------------------
    coordsA = resultA["coords"]
    coordsB = resultB["coords"]

    xsA = [coordsA[n][0] for n in pathA]
    ysA = [coordsA[n][1] for n in pathA]

    xsB = [coordsB[n][0] for n in pathB]
    ysB = [coordsB[n][1] for n in pathB]

    # -------------------------------
    # Beacon overlay (if present in results)
    # -------------------------------
    beacon_source = None
    if "beacon_sequence" in resultB:
        beacon_source = resultB
    elif "beacon_sequence" in resultA:
        beacon_source = resultA

    beacon_coords_hit = []
    beacon_coords_remaining = []
    if beacon_source:
        b_ids = beacon_source.get("beacon_sequence", [])
        b_hits = {hit[2] for hit in beacon_source.get("beacon_hits", [])}
        stored_coords = beacon_source.get("beacon_coords")
        if stored_coords and len(stored_coords) == len(b_ids):
            beacon_pairs = list(zip(b_ids, stored_coords))
        else:
            coords_map = beacon_source.get("coords", {})
            beacon_pairs = [(b, coords_map[b]) for b in b_ids if b in coords_map]

        for b_id, (bx, by) in beacon_pairs:
            if b_id in b_hits:
                beacon_coords_hit.append((bx, by))
            else:
                beacon_coords_remaining.append((bx, by))

    # First divergence idx
    divergence_idx = None
    for i, (a, b) in enumerate(zip(pathA, pathB)):
        if a != b:
            divergence_idx = i
            break

    images = []
    for fp in frame_paths:
        field = np.load(fp)
        plt.figure(figsize=(8, 8))
        plt.imshow(field, cmap="viridis", origin="lower")

        plt.plot(xsA, ysA, color="red", linewidth=2, label="Model A")
        plt.plot(xsB, ysB, color="cyan", linewidth=2, label="Model B")

        if beacon_coords_remaining:
            bx, by = zip(*beacon_coords_remaining)
            plt.scatter(bx, by, color="magenta", s=30, label="Beacons")
        if beacon_coords_hit:
            bx, by = zip(*beacon_coords_hit)
            plt.scatter(bx, by, color="orange", s=40, marker="X", label="Beacons hit")

        if divergence_idx is not None and divergence_idx < len(xsA) and divergence_idx < len(xsB):
            plt.scatter(xsA[divergence_idx], ysA[divergence_idx], color="yellow", s=50)
            plt.scatter(xsB[divergence_idx], ysB[divergence_idx], color="orange", s=50)

        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        images.append(Image.open(buf))

    if images:
        images[0].save(
            out_path,
            save_all=True,
            append_images=images[1:],
            duration=500,
            loop=0,
        )
# ===============================================================
# Comparison result persistence helpers
# ===============================================================
def comparison_data_path(cfg: Dict[str, Any], model_label: str) -> Path:
    root = Path(cfg["paths"]["data_root"])
    folder = root / "ComparisonData"
    return folder / f"{model_label}.json"


def load_comparison_result(cfg: Dict[str, Any], model_label: str) -> Optional[Dict[str, Any]]:
    MAPPING = {
        "Discrete Uncertainty": "DiscreteUncertainty_result.json",
        "Discrete Uncertainty Adaptive": "DiscreteUncertaintyAdaptive_result.json",
        "Budgeted Uncertainty": "BudgetedUncertainty_result.json",      # adjust
        "DStar Lite": "DStarLite_result.json",                               # adjust
        "DStar Lite Discrete Uncertainty": "DStarLiteDiscreteUncertainty_result.json",
        "DStar Lite Discrete Uncertainty Adaptive": "DStarLiteDiscreteAdaptiveUncertainty_result.json",
        "DStar Lite Budgeted Uncertainty": "DStarLiteBudgetedUncertainty_result.json",
    }

    fname = MAPPING.get(model_label)
    if not fname:
        return None

    folder = Path(cfg["paths"]["data_root"]) / "ComparisonData"
    fallback_names = {
        # tolerate legacy typo so old results still load
        "DStarLiteDiscreteUncertainty_result.json": ["DStarLiteDisreteUncertainty_result.json"],
    }

    candidates = [fname] + fallback_names.get(fname, [])
    path = None
    for name in candidates:
        candidate = folder / name
        if candidate.exists():
            path = candidate
            break

    if path is None:
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        # --- FIX: convert coords keys to int ---
        if "coords" in data:
            data["coords"] = {int(k): v for k, v in data["coords"].items()}

        return data
    except Exception:
        return None

# Custom Model Comparison Section
# ===============================================================
def render_custom_model_comparison(cfg: Dict[str, Any]) -> None:
    st.markdown("## Custom Model Comparison")

    metrics = st.session_state.get("metrics", [])
    model_names = [row["Model"] for row in metrics]

    colA, colB = st.columns(2)
    with colA:
        model_a = st.selectbox("Select Model A", model_names, key="cmpA")
    with colB:
        model_b = st.selectbox("Select Model B", model_names, key="cmpB")

    run = st.columns([1,1,1])[1].button("Run Comparison", key="btn_custom_compare", use_container_width=True)

    if not run:
        return

    if model_a == model_b:
        st.warning("Select two different models.")
        return

    # Load model results
    resA = unwrap_result(load_model_result(model_a))
    resB = unwrap_result(load_model_result(model_b))

    if resA is None or resB is None:
        st.info("Run both models first before comparing.")
        return

    # Validate both have paths
    if "node_path" not in resA or "node_path" not in resB:
        st.error("One of the models did not export a path.")
        return

    pathA = resA["node_path"]
    pathB = resB["node_path"]

    # Overlay image for both paths combined
    st.markdown("### Combined Path Overlay")
    base = Path(resA["data_root"])
    out_gif = base / "comparison_two_paths.gif"
    create_two_path_overlay(resA, resB, out_gif)

    if out_gif.exists():
        st.image(str(out_gif), use_container_width=True)
    else:
        st.write("Overlay could not be created.")

    # Per-model individual overlays
    st.markdown("### Individual Model Overlays")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"#### {model_a}")
        ovA = resA.get("overlay_animation")
        if ovA and Path(ovA).exists():
            st.image(str(ovA), use_container_width=True)
        else:
            st.write("No overlay available.")

    with col2:
        st.markdown(f"#### {model_b}")
        ovB = resB.get("overlay_animation")
        if ovB and Path(ovB).exists():
            st.image(str(ovB), use_container_width=True)
        else:
            st.write("No overlay available.")

    # Divergence
    st.markdown("### Path Divergence")
    div = find_path_divergence(pathA, pathB)
    if div:
        idx, a, b = div
        st.write(f"First divergence at step **{idx}**: A→{a}, B→{b}")
    else:
        st.write("No divergence — paths identical up to min length.")



    # -------------------------------------------------------------------
    # 6) DEEP FEASIBILITY COST CHECK (Your Python script integrated)
    # -------------------------------------------------------------------
    st.markdown("## Deep Cost Feasibility Analysis")

    # Explicit logic to select models
    if "Discrete" in model_a and "DStar Lite" in model_b:
        disc = resA
        dstar = resB
    elif "Dstar Lite" in model_a and "Discrete" in model_b:
        disc = resB
        dstar = resA
    else:
        st.error("Deep cost analysis requires one Discrete model and one DStar Lite model.")
        return

    disc_path = disc["node_path"]
    dstar_path = dstar["node_path"]

    st.markdown("### Path Check")

    if disc_path == dstar_path:
        st.success("✔️ Paths identical.")
        path = disc_path
    else:
        st.error("⚠️ Paths differ!")

        min_len = min(len(disc_path), len(dstar_path))
        div = None
        for i in range(min_len):
            if disc_path[i] != dstar_path[i]:
                div = i
                break
        if div is None:
            div = min_len

        st.write(f"**First divergence at step {div}**")
        st.code(f"Discrete: {disc_path[:div+3]}\nDStar Lite: {dstar_path[:div+3]}")
        path = disc_path[:div]

    # ---------------- COST DIFFERENCES ----------------
    st.markdown("### Cost Differences (Only Mismatches)")

    disc_costs = [arc["cost"] for arc in disc["arcs"]]
    dstar_costs = [arc["cost"] for arc in dstar["arcs"]]

    mismatches = []
    for i in range(len(path)-1):
        delta = dstar_costs[i] - disc_costs[i]
        if abs(delta) > 1e-9:
            mismatches.append({
                "step": i,
                "u": path[i],
                "v": path[i+1],
                "disc": disc_costs[i],
                "dstar": dstar_costs[i],
                "Δ": delta
            })

    if not mismatches:
        st.success("✔️ No cost differences on shared arcs.")
    else:
        st.dataframe(pd.DataFrame(mismatches))

    # ---------------- TOTAL COST SUMMARY ----------------
    st.markdown("### Total Cost Summary")

    disc_total = disc["total_cost"]
    dstar_total = dstar.get("total_cost", sum(dstar_costs))

    st.code(
        f"Discrete Uncertainty Total Cost: {disc_total:.6f}\n"
        f"DStar Lite Runtime Total Cost: {dstar_total:.6f}\n"
        f"Difference (DStar - Disc): {dstar_total - disc_total:.6f}"
    )




# ===============================================================
# Action execution pipeline
# ===============================================================
def execute_pending_action(
    cfg: Dict[str, Any],
    log_placeholder,
    show_log: bool,
) -> None:
    action_key = st.session_state.get("pending_action")
    if not action_key:
        return

    # Prevent baseline DStar Lite run when warmstart is active
    if action_key == "run_dstar":
        warm = cfg.get("dstar_lite", {}).get("warmstart_mode", "none")
        if warm != "none":
            st.error(f"Cannot run baseline DStar Lite while warm-start mode is '{warm}'. Please set warm-start to 'none' or use the appropriate warm-start DStar Lite button.")
            st.session_state["pending_action"] = None
            return

    label, runner = ACTIONS.get(action_key, (action_key, None))
    st.session_state["pending_action"] = None

    if runner is None:
        st.info("Model wiring not available for this action.")
        return

    start = time.perf_counter()
    stream_target = log_placeholder if show_log else None
    logger = TerminalStreamer(stream_target, live=show_log)

    try:
        with st.spinner(f"Running {label}..."), \
             contextlib.redirect_stdout(logger), \
             contextlib.redirect_stderr(logger):

            raw = runner(cfg)
            result = unwrap_result(raw)

            # ---------------------------------------------
            # STORE RESULTS BASED ON ACTION
            # ---------------------------------------------
            if action_key == "run_dstar":
                st.session_state["last_result_DStar_Lite"] = result

            elif action_key == "run_dstar_discrete":
                st.session_state["last_result_DStar_Lite_Discrete_Robust"] = result

            elif action_key == "run_dstar_discrete_adaptive":
                st.session_state["last_result_DStar_Lite_Discrete_Adaptive"] = result

            elif action_key == "run_dstar_budgeted":
                st.session_state["last_result_DStar_Lite_Budgeted"] = result
            
            elif action_key == "run_discrete":
                st.session_state["last_result_Discrete_Uncertainty"] = result

            elif action_key == "run_discrete_adaptive":
                st.session_state["last_result_Discrete_Uncertainty_Adaptive"] = result

            elif action_key == "run_budgeted_uncertainty":
                st.session_state["last_result_Budgeted_Uncertainty"] = result
            elif action_key == "run_batch":
                st.session_state["batch_summary"] = result

            # Refresh metrics from JSON after saving
            refresh_metrics_from_json(cfg)

        # ---------------------------------------------
        # UPDATE METRICS AND UI
        # ---------------------------------------------
        if isinstance(result, dict) and "runtime" in result:
            runtime = float(result["runtime"])
        else:
            runtime = time.perf_counter() - start
        apply_data_root_result(cfg, result)
        st.session_state["log_output"] = logger.get_value()

        if show_log:
            render_terminal_output(log_placeholder, logger.get_value())

        st.toast(f"{label} finished in {runtime:.2f}s")
        try:
            st.rerun()
        except Exception:
            try:
                st.experimental_rerun()
            except Exception:
                pass

    except Exception as exc:
        # ---------------------------------------------
        # HANDLE EXCEPTIONS
        # ---------------------------------------------
        runtime = time.perf_counter() - start
        error_log = f"{logger.get_value()}\n[ERROR] {exc}"
        st.session_state["log_output"] = error_log

        if show_log:
            render_terminal_output(log_placeholder, error_log)

        st.error(f"{label} failed: {exc}")
        # On error, refresh metrics as well
        refresh_metrics_from_json(cfg)


# ===============================================================
# Application entrypoint
# ===============================================================
def main() -> None:
    st.set_page_config(
        page_title="Vehicle Routing DSS",
        layout="wide",
        page_icon=str(FAVICON_PATH) if FAVICON_PATH else None,
    )
    inject_css()
    init_session_state()

    cfg = st.session_state["config"]

    st.markdown("<div class='hero-title'>Vehicle Routing DSS</div>", unsafe_allow_html=True)

    boxed_section(path_selector, cfg)

    boxed_section(render_action_dashboard)

    boxed_section(parameter_form, cfg)
    boxed_section(batch_mode_form, cfg)

    show_log, changed = boxed_section(render_visualization_section, cfg)
    if changed:
        save_config(cfg)

    log_placeholder, log_visible = boxed_section(render_log_terminal, show_log)

    boxed_section(render_metrics_summary, cfg)

    boxed_section(render_interpretation_report, cfg)

    boxed_section(render_custom_model_comparison, cfg)

    execute_pending_action(cfg, log_placeholder, log_visible)


if __name__ == "__main__":
    main()
