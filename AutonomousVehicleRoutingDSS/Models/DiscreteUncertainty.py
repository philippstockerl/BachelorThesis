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
        costs = {}
        graph_paths = {}

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
            
            npy_path = os.path.join(base_folder, folder, "field.npy")
            graph_paths[scen] = npy_path if os.path.exists(npy_path) else None

        return costs, graph_paths
    



    coords, node_ids = load_nodes(nodes_path)
    edges = load_edges(edges_path)
    costs, graph_paths = load_scenarios(base_folder)

    return coords, node_ids, edges, costs, graph_paths

    



class DiscreteUncertainty:

    def __init__(self, coords, node_ids, edges, costs, graph_paths):
        self.coords = coords
        self.node_ids = node_ids
        self.edges = edges
        self.costs = costs
        self.scenario_ids = sorted(costs.keys())

        self.graph_paths = graph_paths

        self.start = min(coords.keys())
        self.goal = max(coords.keys())
        
        self.path = None

        # Solution metrics
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
        self.path = None
        self.arcs = []
        self.scenario_costs = {}
        self.scenario_per_step = []
        self.arc_costs_per_scenario = []


    def OptimizationModel(self, costs=None, start_node=None, goal_node=None, log_to_console=1):
        costs = costs if costs is not None else self.costs
        self.last_model_start = self.start if start_node is None else start_node
        self.last_model_goal = self.goal if goal_node is None else goal_node

        model = gp.Model("DiscreteUncertaintyMin-Max")

        model.Params.LogToConsole = log_to_console
        model.setParam("TimeLimit", 60)
        model.setParam("MIPGap", 0.02)

        x = model.addVars(self.edges, vtype=GRB.BINARY, name="x")
        z = model.addVar(vtype=GRB.CONTINUOUS, name="z")

        model.setObjective(z, GRB.MINIMIZE)

        for k in costs.keys():
            model.addConstr(
                z >= gp.quicksum(costs[k][e] * x[e] for e in self.edges),
                name=f"robust_{k}"
            )
        
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
        selected = [(u, v) for (u, v) in self.edges
            if model.getVarByName(f"x[{u},{v}]").X > 0.5]

        nxt = {u: v for (u, v) in selected}

        path = [self.last_model_start]
        curr = self.last_model_start
        visited = set()

        while curr != self.last_model_goal and curr not in visited:
            visited.add(curr)
            if curr not in nxt:
                break
            curr = nxt[curr]
            path.append(curr)

        return path


    def GetResultData(self, model):
        self._reset_solution()

        self.path = self._extract_path_from_model(model)

        self.arcs = []
        for u, v in zip(self.path[:-1], self.path[1:]):
            cost_0 = self.costs[0][(u, v)]
            self.arcs.append({"u": u, "v": v, "cost": cost_0})


        self.scenario_costs = {
            k: sum(self.costs[k][(u, v)] for u, v in zip(self.path[:-1], self.path[1:]))
            for k in self.costs
        }

        self.scenario_per_step = [0] * len(self.arcs)

        for (u, v) in zip(self.path[:-1], self.path[1:]):
            entry = {"u": u, "v": v, "scenarios": {}}
            for scen in self.costs.keys():
                entry["scenarios"][scen] = self.costs[scen][(u, v)]
            self.arc_costs_per_scenario.append(entry)


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

    def run_adaptive(self, window_size, commit_length=None, start_node=None, goal_node=None, log_to_console=0):
        """
        Rolling-window robust planning. Uses a sliding subset of scenario slices,
        commits only the first `commit_length` arcs of each subproblem, then
        advances the time index and repeats.
        """
        self._reset_solution()
        self.model_type = "DiscreteRobustAdaptive"
        self.output_folder = "DiscreteUncertaintyAdaptive"

        start_node = self.start if start_node is None else start_node
        goal_node = self.goal if goal_node is None else goal_node

        current = start_node
        current_time_idx = 0
        steps_taken_in_slice = 0

        # Estimate steps per slice similar to D* Lite: Manhattan distance / #scenarios
        sx, sy = self.coords[start_node]
        gx, gy = self.coords[goal_node]
        manhattan = abs(sx - gx) + abs(sy - gy)
        steps_per_slice = max(1, int(np.ceil(manhattan / max(1, len(self.scenario_ids)))))

        # Default commit length: take an entire slice-worth of steps per solve
        if commit_length is None:
            commit_length = steps_per_slice
        else:
            commit_length = max(1, int(commit_length))

        stitched_path = [current]
        scenario_per_step = []
        arc_costs_per_scenario = []
        arcs = []

        self.start_time = time.time()

        while current != goal_node:
            window_ids = self.scenario_ids[current_time_idx: current_time_idx + window_size]
            if not window_ids:
                # Reuse the last slice when we run out
                window_ids = [self.scenario_ids[-1]]

            costs_window = {sid: self.costs[sid] for sid in window_ids}

            model = self.OptimizationModel(
                costs=costs_window,
                start_node=current,
                goal_node=goal_node,
                log_to_console=log_to_console
            )

            sub_path = self._extract_path_from_model(model)
            if len(sub_path) < 2:
                break

            steps = []
            for nxt in sub_path[1:]:
                steps.append(nxt)
                if len(steps) >= commit_length or nxt == goal_node:
                    break

            for nxt in steps:
                prev = current
                scenario_idx = self.scenario_ids[min(current_time_idx, len(self.scenario_ids)-1)]
                scenario_per_step.append(scenario_idx)

                arc_entry = {"u": prev, "v": nxt, "scenarios": {}}
                for scen in self.costs.keys():
                    arc_entry["scenarios"][scen] = self.costs[scen][(prev, nxt)]
                arc_costs_per_scenario.append(arc_entry)

                arcs.append({"u": prev, "v": nxt, "cost": self.costs[self.scenario_ids[0]][(prev, nxt)]})

                stitched_path.append(nxt)
                current = nxt
                steps_taken_in_slice += 1
                if steps_taken_in_slice >= steps_per_slice:
                    current_time_idx = min(current_time_idx + 1, len(self.scenario_ids) - 1)
                    steps_taken_in_slice = 0
                if current == goal_node:
                    break

        self.end_time = time.time()
        self.runtime = self.end_time - self.start_time

        self.path = stitched_path
        self.arcs = arcs
        self.scenario_per_step = scenario_per_step
        self.arc_costs_per_scenario = arc_costs_per_scenario

        # Aggregate realized costs using the time-indexed scenario slice that was active when the arc was taken
        self.scenario_costs = {sid: 0.0 for sid in self.scenario_ids}
        for (u, v), scen in zip(zip(self.path[:-1], self.path[1:]), scenario_per_step):
            self.scenario_costs[scen] += self.costs[scen][(u, v)]

        realized_total = sum(self.costs[scen][(u, v)] for (u, v), scen in zip(zip(self.path[:-1], self.path[1:]), scenario_per_step))
        self.objective_value = realized_total

        self.result = {
            "cost": float(realized_total),
            "node_path": self.path,
            "runtime": self.runtime,
            "arcs": self.arcs,
            "total_cost": float(realized_total),
            "scenario_costs": self.scenario_costs,
            "scenario_per_step": self.scenario_per_step,
            "coords": self.coords,
            "data_root": self.data_root,
            "model_type": self.model_type,
            "arc_costs_per_scenario": self.arc_costs_per_scenario,
            "animated_overlay": self.animated_overlay
        }


    def ExportPathOverlay(self, base_dir):
        gif_name = "discrete_uncertainty_path_overlay.gif" if self.output_folder == "DiscreteUncertainty" else "discrete_uncertainty_adaptive_path_overlay.gif"
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
        csv_name = "discrete_uncertainty_path.csv" if self.output_folder == "DiscreteUncertainty" else "discrete_uncertainty_adaptive_path.csv"
        out_dir = Path(base_dir) / self.output_folder
        out_dir.mkdir(parents=True, exist_ok=True)

        out_file = out_dir / csv_name
        df = pd.DataFrame({"node_id": self.path})
        df.to_csv(out_file, index=False, header=False)

    
    def ExportResultsJSON(self, base_dir):
        self.result["animated_overlay"] = str(self.animated_overlay)
        out_dir = Path(base_dir) / "ComparisonData"
        out_dir.mkdir(parents=True, exist_ok=True)

        json_name = "DiscreteUncertainty_result.json" if self.output_folder == "DiscreteUncertainty" else "DiscreteUncertaintyAdaptive_result.json"
        json_path = out_dir / json_name
        with open(json_path, "w") as jf:
            json.dump(self.result, jf, indent=2)
