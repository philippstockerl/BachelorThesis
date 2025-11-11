# Code Reference – Bachelor Thesis Project

## Overview
This document describes the main modules of the project and their roles.

---

### GRF_Generator.py
Generates synthetic Gaussian Random Fields used to model environmental uncertainty.

**Key Functions**
- `generate_base_field()`: Creates a spatially correlated Gaussian field (zero mean).
- `create_forecast_field()`: Applies Gaussian blur to simulate a weather forecast.
- `normalize_field()`: Rescales data to [1, 10] for cost consistency.
- `save_edges_csv()`: Builds network graph with cost deviations.

**Outputs**
- `edges_forecast.csv`, `edges_actual.csv`
- `forecast_field.npy`, `actual_field.npy`
- Heatmap visualizations (`.png`)

---

### ⚙️ BudgetUncertain_MinMax.py
Implements the Bertsimas–Sim robust optimization model.

**Purpose**
- Solves the robust shortest path problem under budget uncertainty.
- Exports path node list for D* Lite initialization.

---

### 🤖 D_Star_Lite.py
Implements dynamic replanning with D* Lite using the actual cost map.

**Inputs**
- `edges_actual.csv`, `robust_path_nodes.csv`

**Outputs**
- `dstar_run_summary.json`
- `dstar_hybrid_edges.csv`

---

### 🎨 Visualization Scripts
| File | Description |
|------|--------------|
| `robust_visualizer.py` | Shows robust route & cost breakdown |
| `dstar_visualizer.py` | Shows dynamic route & updates |

---