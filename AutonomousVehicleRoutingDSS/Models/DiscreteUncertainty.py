"""Discrete robust shortest path (min-max) model and I/O helpers."""

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path
import time
import json


def Preprocessing(nodes_path, edges_path, base_folder):
    """Load graph data and scenario costs from a scenario folder tree."""

    def load_nodes(path):
        """
        **Dictionary of Node Coordinates**

        Convert nodes.csv into:

        - A coordinate dictionary to remap the csv:
            coords = {
                0: (x, y),
                1: (x, y),
                ...
            }

        - A list of all of the node IDs:
            node_ids = [0, 1, 2, ...]
        """
        df = pd.read_csv(path)
        coords = {int(r.node_id): (float(r.x), float(r.y)) for _, r in df.iterrows()}
        node_ids = list(df["node_id"].astype(int))
        return coords, node_ids



    def load_edges(path):
        """
        **Adjacency List of Directed Edges**

        Create nested dictionary that contains all outgoing neighbours for each node u

        edges = {
            (u, v),
            (u, v),
            ...
        }
        """
        df = pd.read_csv(path)

        edges = set()
        for _, r in df.iterrows():
            u = int(r.u)
            v = int(r.v)
            edges.add((u, v))

        return edges



    def load_scenarios(path):
        """
        **Robust Cost Dictionary**

        Return a nested dictionary that holds the cost for all edges in every scenario:
        costs = {
            0: {
                (x,y): cost,
                (x,y): cost,
                (x,y): cost,
                ...
            },

            1: {
                (x,y): cost,
                (x,y): cost,
                (x,y): cost,
            }
        }
        """
        # Each scenario folder contributes a full edge-cost dictionary.
        costs = {}
        graph_paths = {}

        # `base_folder` is expected to contain scenario_### subfolders.
        scenario_folders = sorted(
            f for f in os.listdir(base_folder)
            if f.startswith("scenario_")
        )

        for folder in scenario_folders:
            scen = int(folder.split("_")[1])
            df = pd.read_csv(os.path.join(base_folder, folder, "edges.csv"))

            costs[scen] = {}

            for _, r in df.iterrows():
                u = int(r.u)
                v = int(r.v)
                c = float(r.cost)

                costs[scen][(u, v)] = c
            
            # Optional per-scenario cost field for visualization overlays.
            npy_path = os.path.join(base_folder, folder, "field.npy")
            graph_paths[scen] = npy_path if os.path.exists(npy_path) else None

        return costs, graph_paths
    



    # Build node/edge structures and scenario cost dictionaries.
    coords, node_ids = load_nodes(nodes_path)
    edges = load_edges(edges_path)
    costs, graph_paths = load_scenarios(base_folder)

    return coords, node_ids, edges, costs, graph_paths

    



class DiscreteUncertainty:
    """Solve a min-max robust shortest path over discrete cost scenarios."""

    def __init__(self, coords, node_ids, edges, costs, graph_paths):
        self.coords = coords
        self.node_ids = node_ids
        self.edges = edges
        self.costs = costs
        self.scenario_ids = sorted(costs.keys())

        self.graph_paths = graph_paths

        self.start = min(coords.keys())
        self.goal = max(coords.keys())
        
        # Solution storage (filled after optimization).
        self.path = None

        # Solution metrics.
        self.arcs = []
        self.scenario_costs = {}
        self.scenario_per_step = []

        self.arc_costs_per_scenario = []

        self.start_time = 0.0
        self.end_time = 0.0
        self.runtime = 0.0

        self.data_root = None
        self.model_type = "DiscreteRobust"
        self.result = {}
        self.animated_overlay = None
        self.output_folder = "DiscreteUncertainty"
        self.last_model_start = self.start
        self.last_model_goal = self.goal


    def _reset_solution(self):
        """Clear any previously stored solution outputs. Used for batch mode."""
        self.path = None
        self.arcs = []
        self.scenario_costs = {}
        self.scenario_per_step = []
        self.arc_costs_per_scenario = []


    def OptimizationModel(self, costs=None, start_node=None, goal_node=None, log_to_console=1):
        """Build and solve the min-max robust model; returns the Gurobi model."""
        costs = costs if costs is not None else self.costs
        self.last_model_start = self.start if start_node is None else start_node
        self.last_model_goal = self.goal if goal_node is None else goal_node

        model = gp.Model("DiscreteUncertaintyMin-Max")

        model.Params.LogToConsole = log_to_console
        model.setParam("TimeLimit", 60)

        # x[u,v] = 1 if edge (u,v) is selected in the path.
        x = model.addVars(self.edges, vtype=GRB.BINARY, name="x")
        # z upper-bounds the path cost in every scenario.
        z = model.addVar(vtype=GRB.CONTINUOUS, name="z")

        model.setObjective(z, GRB.MINIMIZE)

        # Robust min-max constraints: z >= cost of chosen path per scenario.
        for k in costs.keys():
            model.addConstr(
                z >= gp.quicksum(costs[k][e] * x[e] for e in self.edges),
                name=f"robust_{k}"
            )
        
        # Flow conservation to enforce a single path from start to goal.
        for v in self.node_ids:
            outgoing = gp.quicksum(x[e] for e in self.edges if e[0] == v)
            incoming = gp.quicksum(x[e] for e in self.edges if e[1] == v)

            if v == self.last_model_start:
                b_v  = 1
            elif v == self.last_model_goal:
                b_v = -1
            else:
                b_v = 0
            
            model.addConstr(
                outgoing - incoming == b_v,
                name=f"flow_{v}"
            )
        
        self.start_time = time.time()

        model.optimize()
        self.model = model

        self.end_time = time.time()
        self.runtime = self.end_time - self.start_time

        self.objective_value = model.ObjVal
        return model



    def _extract_path_from_model(self, model):
        """Recover an ordered node path from the selected edges."""
        selected = [(u, v) for (u, v) in self.edges
            if model.getVarByName(f"x[{u},{v}]").X > 0.5]

        nxt = {u: v for (u, v) in selected}

        path = [self.last_model_start]
        curr = self.last_model_start
        visited = set()

        # Follow successor pointers until goal or a detected loop.
        while curr != self.last_model_goal and curr not in visited:
            visited.add(curr)
            if curr not in nxt:
                break
            curr = nxt[curr]
            path.append(curr)

        return path


    def GetResultData(self, model):
        """Extract path/cost metrics from the solved model into result dicts."""
        self._reset_solution()

        self.path = self._extract_path_from_model(model)

        # Base-scenario arc list for quick inspection.
        self.arcs = []
        for u, v in zip(self.path[:-1], self.path[1:]):
            cost_0 = self.costs[0][(u, v)]
            self.arcs.append({"u": u, "v": v, "cost": cost_0})


        # Total path cost per scenario.
        self.scenario_costs = {
            k: sum(self.costs[k][(u, v)] for u, v in zip(self.path[:-1], self.path[1:]))
            for k in self.costs
        }

        # Placeholder for per-step scenario data (used by consumers).
        self.scenario_per_step = [0] * len(self.arcs)

        # Per-arc, per-scenario costs for downstream plotting/analysis.
        for (u, v) in zip(self.path[:-1], self.path[1:]):
            entry = {"u": u, "v": v, "scenarios": {}}
            for scen in self.costs.keys():
                entry["scenarios"][scen] = self.costs[scen][(u, v)]
            self.arc_costs_per_scenario.append(entry)


        # Final result payload for JSON export and dashboard use.
        self.result = {
            "cost": float(self.objective_value),
            "node_path": self.path,
            "runtime": self.runtime,
            "arcs": self.arcs,
            "total_cost": float(self.objective_value),
            "scenario_costs": self.scenario_costs,
            "scenario_per_step": self.scenario_per_step,
            "coords": self.coords,
            "data_root": self.data_root,
            "model_type": self.model_type,
            "arc_costs_per_scenario": self.arc_costs_per_scenario,
            "animated_overlay": self.animated_overlay
        }

    def ExportPathOverlay(self, base_dir):
        """Render a GIF of the chosen path over scenario field frames."""
        gif_name = "discrete_uncertainty_path_overlay.gif"
        out_dir = Path(base_dir) / self.output_folder / "overlays"
        out_dir.mkdir(parents=True, exist_ok=True)
        gif_path = out_dir / gif_name
        self.animated_overlay = gif_path

        frame_paths = [self.graph_paths[s] for s in sorted(self.graph_paths.keys())]

        images = []
        for t, fp in enumerate(frame_paths):
            frame = np.load(fp)

            plt.figure(figsize=(6, 6))
            plt.imshow(frame, cmap="viridis", origin="lower")
            plt.axis("off")
            plt.tight_layout(pad=0)
            
            xs = [self.coords[n][0] for n in self.path]
            ys = [self.coords[n][1] for n in self.path]
            plt.plot(xs, ys, color="red", linewidth=2)

            #plt.title(f"Cost Field t={t}  |  Total cost = {self.objective_value:.3f}", fontsize=14)

            #plt.xticks(np.linspace(0, frame.shape[1], 11))
            #plt.yticks(np.linspace(0, frame.shape[0], 11))

            out_png = out_dir / f"frame_{t:03d}.png"
            plt.savefig(out_png, dpi=150)
            plt.close()

            images.append(Image.open(out_png))

        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=500,
            loop=0,
        )

    
    def ExportRobustPathCSV(self, base_dir):
        """Export the node path as a CSV for warm-starting other models."""
        csv_name = "discrete_uncertainty_path.csv"
        out_dir = Path(base_dir) / self.output_folder
        out_dir.mkdir(parents=True, exist_ok=True)

        out_file = out_dir / csv_name
        df = pd.DataFrame({"node_id": self.path})
        df.to_csv(out_file, index=False, header=False)

    
    def ExportResultsJSON(self, base_dir):
        """Write the result dictionary to JSON for downstream analysis."""
        self.result["animated_overlay"] = str(self.animated_overlay)
        out_dir = Path(base_dir) / "ComparisonData"
        out_dir.mkdir(parents=True, exist_ok=True)

        json_name = "DiscreteUncertainty_result.json"
        json_path = out_dir / json_name
        with open(json_path, "w") as jf:
            json.dump(self.result, jf, indent=2)
