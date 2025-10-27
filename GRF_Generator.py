import numpy as np
import gstools as gs
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import os
import csv

# --- Unified random seed for reproducibility ---
# The single SEED variable controls both NumPy's RNG and GSTools SRF seed,
# ensuring consistent and reproducible Gaussian Random Fields (forecast & actual).
# Changing SEED will produce different GRF layouts while maintaining reproducibility.
SEED = 1
np.random.seed(SEED)

# --- Hard-coded parameters ---
GRID_SIZE   = 40
VARIANCE    = 1.0
LEN_SCALE   = 5.0
BLUR_SIGMA  = 2.5
NOISE_LEVEL = 0.0
MAX_COST    = 10.0
OUTPUT_DIR  = f"Data/scenarios/scenario_{SEED}"

def generate_base_field(size, variance, len_scale, seed=None):
    """Generate a 2D Gaussian random field on a structured grid.

    Modeling decision:
    - The Gaussian Random Field (GRF) is generated assuming zero mean.
    - Variance and length scale control the spatial variability and smoothness.
    """
    model = gs.Gaussian(dim=2, var=variance, len_scale=len_scale)
    # Use the unified SEED for reproducibility
    srf = gs.SRF(model, seed=seed)
    x = np.arange(size)
    y = np.arange(size)
    field = srf.structured([x, y])
    return field

def create_forecast_field(base_field, blur_sigma):
    """Produce a blurred version of the base field (forecast).

    Modeling decision:
    - The forecast field is a smoothed (blurred) version of the base GRF,
      representing a less detailed, more uncertain prediction.
    """
    return gaussian_filter(base_field, sigma=blur_sigma)

def create_actual_field(base_field, noise_level):
    """Produce a more detailed version from base by adding noise (actual).

    NOTE: For scientific clarity, this has been modified so that the actual field is identical to the base GRF.
    To re-enable white noise addition, uncomment the two lines below and comment out the return statement.

    Modeling decision:
    - White noise addition is currently disabled, so the only difference between forecast and actual maps
      is the forecast’s blur.
    """
    # noise = np.random.normal(loc=0.0, scale=noise_level, size=base_field.shape)
    # return base_field + noise
    return base_field.copy()

def normalize_field(field, new_min=1.0, new_max=10.0):
    """Rescale field values linearly to range [new_min, new_max].

    Modeling decision:
    - Normalization rescales both forecast and actual fields to a comparable cost range,
      facilitating comparison and downstream processing.
    """
    old_min = field.min()
    old_max = field.max()
    norm = (field - old_min) / (old_max - old_min)
    return new_min + norm * (new_max - new_min)

def save_field(field, filepath):
    """Save field to numpy file."""
    np.save(filepath, field)
    print(f"Saved field to {filepath}")

def plot_field(field, title, filepath_png):
    """Visualise field as a heatmap and save figure."""
    plt.figure(figsize=(6,5))
    plt.imshow(field, cmap="viridis", origin="lower")
    plt.title(title)
    plt.colorbar(label="Cost")
    plt.tight_layout()
    plt.savefig(filepath_png, dpi=300)
    plt.close()
    print(f"Saved figure to {filepath_png}")

def save_nodes_csv(grid_size, filepath):
    """Save nodes.csv with columns node_id, x, y"""
    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['node_id', 'x', 'y'])
        for y in range(grid_size):
            for x in range(grid_size):
                node_id = y * grid_size + x
                writer.writerow([node_id, x, y])
    print(f"Saved nodes to {filepath}")

def save_edges_csv(nominal_field, filepath, actual_field=None, cost_field=None, deviation_factor=0.1):
    """
    Save edges CSV with columns edge_id, u, v, nominal_forecast, deviation_abs for 4-neighbour connectivity.
    - nominal_field: the nominal (forecast) field.
    - actual_field: the actual field (optional, for computing deviation).
    - cost_field: the cost field to compare against (optional, for computing deviation).
    - deviation_factor: fallback percent of nominal if actual/cost not provided.
    """
    grid_size = nominal_field.shape[0]
    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['edge_id', 'u', 'v', 'nominal_forecast', 'deviation_abs'])
        edge_id = 0
        for y in range(grid_size):
            for x in range(grid_size):
                u = y * grid_size + x
                # right neighbor
                if x + 1 < grid_size:
                    v = y * grid_size + (x + 1)
                    nominal_forecast = nominal_field[y, x]
                    # Deviation calculation
                    if actual_field is not None and cost_field is not None:
                        deviation_abs = abs(actual_field[y, x] - cost_field[y, x])
                    else:
                        deviation_abs = deviation_factor * nominal_forecast
                    writer.writerow([edge_id, u, v, nominal_forecast, deviation_abs])
                    edge_id += 1
                    writer.writerow([edge_id, v, u, nominal_forecast, deviation_abs])
                    edge_id += 1
                # down neighbor
                if y + 1 < grid_size:
                    v = (y + 1) * grid_size + x
                    nominal_forecast = nominal_field[y, x]
                    if actual_field is not None and cost_field is not None:
                        deviation_abs = abs(actual_field[y, x] - cost_field[y, x])
                    else:
                        deviation_abs = deviation_factor * nominal_forecast
                    writer.writerow([edge_id, u, v, nominal_forecast, deviation_abs])
                    edge_id += 1
                    writer.writerow([edge_id, v, u, nominal_forecast, deviation_abs])
                    edge_id += 1
    print(f"Saved edges to {filepath}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 1. Generate base field
    base = generate_base_field(
        size=GRID_SIZE,
        variance=VARIANCE,
        len_scale=LEN_SCALE,
        seed=SEED
    )

    # 2. Create forecast & actual fields
    forecast = create_forecast_field(base, blur_sigma=BLUR_SIGMA)
    actual   = create_actual_field(base, noise_level=NOISE_LEVEL)

    # 3. Normalize fields
    forecast_norm = normalize_field(forecast, new_min=1.0, new_max=MAX_COST)
    actual_norm   = normalize_field(actual,   new_min=1.0, new_max=MAX_COST)

    # 4. Save numerical arrays for later use by visualization and optimization
    # Save normalized forecast and actual fields as .npy files
    save_field(forecast_norm, os.path.join(OUTPUT_DIR, "forecast_field.npy"))
    save_field(actual_norm, os.path.join(OUTPUT_DIR, "actual_field.npy"))

    # 5. Save nodes and edges CSV files
    save_nodes_csv(GRID_SIZE, os.path.join(OUTPUT_DIR, "nodes.csv"))
    # For forecast edges: deviation using actual and forecast (for each edge)
    save_edges_csv(
        forecast_norm,
        os.path.join(OUTPUT_DIR, "edges_forecast.csv"),
        actual_field=actual_norm,
        cost_field=forecast_norm,
        deviation_factor=0.1
    )
    # For actual edges: deviation using actual and forecast (for each edge)
    save_edges_csv(
        actual_norm,
        os.path.join(OUTPUT_DIR, "edges_actual.csv"),
        actual_field=actual_norm,
        cost_field=forecast_norm,
        deviation_factor=0.1
    )
    # 5. Visualise & save figure images
    plot_field(forecast_norm, title="Forecast Cost Field", filepath_png=os.path.join(OUTPUT_DIR, "forecast_field.png"))
    plot_field(actual_norm,   title="Actual Cost Field",   filepath_png=os.path.join(OUTPUT_DIR, "actual_field.png"))
    print("Generation and visualization complete.")

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------
# NOTE:
# The unified SEED variable controls both NumPy and GSTools SRF seeding,
# making the generated Gaussian fields reproducible.
# To generate new random environments, change the SEED value above.
#
# Modeling summary:
# - The Gaussian Random Field (GRF) is generated with a zero-mean assumption.
# - The normalization step rescales both forecast and actual fields to a comparable cost range [1, 10].
# - White noise addition for the actual field is currently disabled, so the only difference between forecast and actual maps
#   is the forecast’s blur (smoothing), representing uncertainty in the forecast.
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# FUTURE EXTENSION IDEA:
# Add an optional switch for normalization methods to increase experimental flexibility:
# 1. Full-range normalization (default): maps values to [1, 10].
# 2. Mean-based normalization: rescales by mean or standard deviation instead of range.
# 3. Option to disable normalization entirely (use raw GRF values).
# This would allow analysis of how cost scale and variance magnitude influence
# robust vs. dynamic route planning behavior.
# ---------------------------------------------------------------------