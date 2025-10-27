# gaussianGenerator.py
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
SEED = 2
np.random.seed(SEED)

# --- Hard-coded parameters ---
GRID_SIZE    = 15
VARIANCE     = 1.0
LEN_SCALE    = 5.0
TEMP_LEN_SCALE = 3.0  # Temporal length scale for spatio-temporal correlation
TIME_STEPS   = 10     # Number of temporal steps to generate
BLUR_SIGMA   = 2.5
NOISE_LEVEL  = 0.0
MAX_COST     = 10.0
OUTPUT_DIR   = f"Data/scenarios/temporal_scenarios_{SEED}"

def generate_base_field(size, variance, len_scale, temp_len_scale, seed=None):
    """Generate a 3D spatio-temporal Gaussian random field on a structured grid.

    Modeling decision:
    - The Gaussian Random Field (GRF) is generated with three dimensions: two spatial and one temporal.
    - Variance controls the field variance.
    - len_scale is a list defining spatial and temporal correlation lengths: [spatial_x, spatial_y, temporal].
    - This models smooth spatial patterns evolving over time with temporal correlation.
    """
    model = gs.Gaussian(dim=3, var=variance, len_scale=[len_scale, len_scale, temp_len_scale])
    srf = gs.SRF(model, seed=seed)
    x = np.arange(size)
    y = np.arange(size)
    t = np.arange(TIME_STEPS)
    # Generate a 3D field with shape (time, y, x)
    field_3d = srf.structured([x, y, t])
    # GSTools returns array in order (x, y, t), we transpose to (t, y, x)
    field_3d = np.transpose(field_3d, (2,1,0))
    return field_3d

def create_forecast_field(base_field, blur_sigma):
    """Produce a blurred version of the base field (forecast) per time step.

    Modeling decision:
    - The forecast field is a smoothed (blurred) version of the base GRF at each time step,
      representing a less detailed, more uncertain prediction evolving over time.
    """
    forecast = np.empty_like(base_field)
    for t in range(base_field.shape[0]):
        forecast[t] = gaussian_filter(base_field[t], sigma=blur_sigma)
    return forecast

def create_actual_field(base_field, noise_level):
    """Produce a more detailed version from base by adding noise (actual) per time step.

    NOTE: For scientific clarity, this has been modified so that the actual field is identical to the base GRF.
    To re-enable white noise addition, uncomment the two lines below and comment out the return statement.

    Modeling decision:
    - White noise addition is currently disabled, so the only difference between forecast and actual maps
      is the forecast’s blur (smoothing), representing uncertainty in the forecast.
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
    # 1. Generate base 3D spatio-temporal field (time, y, x)
    base_3d = generate_base_field(
        size=GRID_SIZE,
        variance=VARIANCE,
        len_scale=LEN_SCALE,
        temp_len_scale=TEMP_LEN_SCALE,
        seed=SEED
    )

    # 2. Create forecast & actual fields per time step
    forecast_3d = create_forecast_field(base_3d, blur_sigma=BLUR_SIGMA)
    actual_3d   = create_actual_field(base_3d, noise_level=NOISE_LEVEL)

    # 3. Normalize fields across all time steps
    forecast_norm_3d = normalize_field(forecast_3d, new_min=1.0, new_max=MAX_COST)
    actual_norm_3d   = normalize_field(actual_3d,   new_min=1.0, new_max=MAX_COST)

    # 4. Save nodes CSV once (spatial nodes do not change over time)
    save_nodes_csv(GRID_SIZE, os.path.join(OUTPUT_DIR, "nodes.csv"))

    # 5. Loop over time steps to save arrays, CSV edges, and plots
    for t in range(TIME_STEPS):
        forecast_t = forecast_norm_3d[t]
        actual_t = actual_norm_3d[t]

        # Save numpy arrays
        save_field(forecast_t, os.path.join(OUTPUT_DIR, f"forecast_field_t{t:02d}.npy"))
        save_field(actual_t, os.path.join(OUTPUT_DIR, f"actual_field_t{t:02d}.npy"))

        # Save edges CSV for forecast at time t
        save_edges_csv(
            forecast_t,
            os.path.join(OUTPUT_DIR, f"edges_forecast_t{t:02d}.csv"),
            actual_field=actual_t,
            cost_field=forecast_t,
            deviation_factor=0.1
        )
        # Save edges CSV for actual at time t
        save_edges_csv(
            actual_t,
            os.path.join(OUTPUT_DIR, f"edges_actual_t{t:02d}.csv"),
            actual_field=actual_t,
            cost_field=forecast_t,
            deviation_factor=0.1
        )

        # Plot and save figures
        plot_field(forecast_t, title=f"Forecast Cost Field t={t}", filepath_png=os.path.join(OUTPUT_DIR, f"forecast_field_t{t:02d}.png"))
        plot_field(actual_t, title=f"Actual Cost Field t={t}", filepath_png=os.path.join(OUTPUT_DIR, f"actual_field_t{t:02d}.png"))

    print("Spatio-temporal generation and visualization complete.")

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------
# NOTE:
# The unified SEED variable controls both NumPy and GSTools SRF seeding,
# making the generated Gaussian fields reproducible.
# To generate new random environments, change the SEED value above.
#
# Modeling summary:
# - The Gaussian Random Field (GRF) is generated with three dimensions: two spatial and one temporal.
# - The temporal dimension introduces correlation over time with length scale TEMP_LEN_SCALE.
# - The normalization step rescales both forecast and actual fields to a comparable cost range [1, 10].
# - White noise addition for the actual field is currently disabled, so the only difference between forecast and actual maps
#   is the forecast’s blur (smoothing), representing uncertainty in the forecast.
# - Outputs are saved for each time step separately for temporal analysis.
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# EXTENSION IDEA: MOVING WEATHER-LIKE GAUSSIAN RANDOM FIELDS
#
# This script can be extended to model moving weather-like phenomena by generating
# spatio-temporal Gaussian Random Fields that evolve smoothly over time, simulating
# dynamic cost fields such as clouds or storms moving across the spatial grid.
#
# The temporal correlation length scale (TEMP_LEN_SCALE) controls how quickly the
# field changes over time, capturing realistic temporal evolution of environmental
# conditions that affect cost or risk.
#
# Such dynamic cost fields are valuable for testing path planning and routing
# algorithms under changing conditions. For example, dynamic algorithms like D* Lite
# can adapt their plans as the environment evolves, while static robust methods
# generate a single plan without adaptation.
#
# This extension would enable benchmarking and comparison of adaptive versus static
# planning strategies in temporally correlated, moving scenarios resembling real-world
# weather or environmental changes.
# ---------------------------------------------------------------------