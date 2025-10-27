import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pathlib import Path
from noise import pnoise2

def generate_base_field(rows, cols, scale=10.0, octaves=1, persistence=0.5, lacunarity=2.0, seed=None):
    if seed is None:
        seed = np.random.randint(0, 100)
    field = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            field[i][j] = pnoise2(i / scale,
                                  j / scale,
                                  octaves=octaves,
                                  persistence=persistence,
                                  lacunarity=lacunarity,
                                  repeatx=rows,
                                  repeaty=cols,
                                  base=seed)
    # Normalize to 0-1
    min_val = field.min()
    max_val = field.max()
    norm_field = (field - min_val) / (max_val - min_val)
    return norm_field

def generate_forecast_and_realized(base_field, sigma_forecast=4, sigma_realized=2, mode='normal'):
    forecast = gaussian_filter(base_field, sigma=sigma_forecast)
    realized = gaussian_filter(base_field, sigma=sigma_realized)

    # scale to 0–1 range
    forecast = (forecast - forecast.min()) / (forecast.max() - forecast.min())
    realized = (realized - realized.min()) / (realized.max() - realized.min())

    # perturbations to simulate “better/worse than forecast” cases
    if mode == 'under':
        realized *= np.random.uniform(0.6, 0.95)
    elif mode == 'over':
        realized *= np.random.uniform(1.05, 1.25)
        realized = np.clip(realized, 0, 1)

    return forecast, realized

def field_to_edges(field):
    rows, cols = field.shape
    edges = []
    eid = 0
    for i in range(rows):
        for j in range(cols):
            node = i * cols + j
            # Down neighbor
            if i < rows - 1:
                nominal = field[i + 1, j]
                deviation = np.random.uniform(0.1, 0.3) * nominal
                # Ensure finite and positive
                nominal = np.clip(nominal, 0.1, 50)
                deviation = np.clip(deviation, 0.1, 50)
                edges.append({
                    "edge_id": eid,
                    "u": node,
                    "v": (i + 1) * cols + j,
                    "nominal": nominal,
                    "d": deviation
                })
                eid += 1
                # Add reverse edge
                edges.append({
                    "edge_id": eid,
                    "u": (i + 1) * cols + j,
                    "v": node,
                    "nominal": nominal,
                    "d": deviation
                })
                eid += 1
            # Right neighbor
            if j < cols - 1:
                nominal = field[i, j + 1]
                deviation = np.random.uniform(0.1, 0.3) * nominal
                # Ensure finite and positive
                nominal = np.clip(nominal, 0.1, 50)
                deviation = np.clip(deviation, 0.1, 50)
                edges.append({
                    "edge_id": eid,
                    "u": node,
                    "v": i * cols + (j + 1),
                    "nominal": nominal,
                    "d": deviation
                })
                eid += 1
                # Add reverse edge
                edges.append({
                    "edge_id": eid,
                    "u": i * cols + (j + 1),
                    "v": node,
                    "nominal": nominal,
                    "d": deviation
                })
                eid += 1
    return pd.DataFrame(edges, columns=["edge_id", "u", "v", "nominal", "d"])

def field_to_nodes(field):
    rows, cols = field.shape
    data = []
    for i in range(rows):
        for j in range(cols):
            data.append([i * cols + j, i, j])
    return pd.DataFrame(data, columns=['id', 'x', 'y'])

def save_field_image(field, filename, title=''):
    plt.figure(figsize=(5, 5))
    plt.imshow(field, cmap='viridis', origin='lower')
    plt.colorbar(label='Cost Intensity')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def generate_scenario_pair(out_dir, idx, rows, cols, sigma_forecast, sigma_realized, over_prob,
                           scale, octaves, persistence, lacunarity):
    base = generate_base_field(rows, cols, scale=scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity)
    scenario_type = np.random.choice(['normal', 'under', 'over'], p=[0.6, 0.3, 0.1]) if over_prob > 0 else 'normal'
    forecast, realized = generate_forecast_and_realized(base, sigma_forecast, sigma_realized, scenario_type)

    scenario_dir = Path(out_dir) / f"scenario_{idx:02d}"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    # save CSVs
    nodes = field_to_nodes(forecast)
    edges_forecast = field_to_edges(forecast)
    edges_realized = field_to_edges(realized)

    nodes.to_csv(scenario_dir / "nodes.csv", index=False)
    edges_forecast.to_csv(scenario_dir / "edges_forecast.csv", index=False)
    edges_realized.to_csv(scenario_dir / "edges_realized.csv", index=False)

    # save visual maps
    save_field_image(forecast, scenario_dir / "forecast_map.png", title="Forecast (Robust Model Input)")
    save_field_image(realized, scenario_dir / "realized_map.png", title="Realized (D* Lite Input)")

    print(f"✅ Generated scenario_{idx:02d} ({scenario_type}) with shape {rows}x{cols}")

def main():
    parser = argparse.ArgumentParser(description="Generate paired forecast–realization maps for robust vs D* Lite models.")
    parser.add_argument("--n", type=int, default=5, help="Number of scenarios to generate.")
    parser.add_argument("--rows", type=int, default=15, help="Grid rows.")
    parser.add_argument("--cols", type=int, default=15, help="Grid cols.")
    parser.add_argument("--sigma_forecast", type=float, default=4.0, help="Gaussian blur sigma for forecast.")
    parser.add_argument("--sigma_realized", type=float, default=2.0, help="Gaussian blur sigma for realized.")
    parser.add_argument("--out_dir", type=str, default="Data/scenarios", help="Output directory.")
    parser.add_argument("--over_prob", type=float, default=0.1, help="Probability of over-realized scenario.")
    parser.add_argument("--scale", type=float, default=10.0, help="Scale for Perlin noise.")
    parser.add_argument("--octaves", type=int, default=1, help="Octaves for Perlin noise.")
    parser.add_argument("--persistence", type=float, default=0.5, help="Persistence for Perlin noise.")
    parser.add_argument("--lacunarity", type=float, default=2.0, help="Lacunarity for Perlin noise.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for i in range(1, args.n + 1):
        generate_scenario_pair(
            args.out_dir,
            i,
            args.rows,
            args.cols,
            args.sigma_forecast,
            args.sigma_realized,
            args.over_prob,
            args.scale,
            args.octaves,
            args.persistence,
            args.lacunarity
        )

if __name__ == "__main__":
    main()