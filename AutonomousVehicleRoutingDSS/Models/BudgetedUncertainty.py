import json
import os
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import gurobipy as gp
from gurobipy import GRB


class BudgetedUncertainty:
    def __init__(self, data_root, gamma=5, nominal_rule="min", time_limit=None):
        """
        Classic budgeted uncertainty using a single global gamma.
        nominal_rule: "min" or "avg" for nominal cost aggregation across scenarios.
        """
        self.data_root = Path(data_root)
        self.gamma = gamma
        self.nominal_rule = nominal_rule  # "min" or "avg"
        self.time_limit = time_limit

        self.coords = {}
        self.nodes = []
        self.edges = {}
        self.cost_nominal = {}
        self.deviation = {}
        self.edge_list = []
        self.out_edges = {}
        self.in_edges = {}

        self.model = None
        self.x = None
        self.pi = None
        self.rho = None
        self.start = None
        self.goal = None
        self.path = []
        self.result = {}
        self.animated_overlay = None

    # ------------------------------------------------------------------
    # PREPROCESSING
    # ------------------------------------------------------------------
    def Preprocessing(self):
        """
        Reads scenario_*/edges.csv and nodes.csv and computes:
        - cost_nominal[i]
        - deviation[i]
        - adjacency lists
        """
        from time import perf_counter

        t0 = perf_counter()
        print(f"[BudgetedUncertainty] Preprocessing start | data_root={self.data_root}", flush=True)
        # Load nodes with coordinates
        nodes_df = pd.read_csv(self.data_root / "nodes.csv")
        self.nodes = list(nodes_df["node_id"])
        self.coords = {
            int(r.node_id): (float(r.x), float(r.y)) for _, r in nodes_df.iterrows()
        }
        print(f"[BudgetedUncertainty] Loaded nodes.csv with {len(self.nodes)} nodes", flush=True)

        # Load all scenario edges
        scen_dirs = sorted(self.data_root.glob("scenario_*"))
        if not scen_dirs:
            raise RuntimeError("No scenario folders found under data_root.")

        merged = []  # list of (u, v, cost, scenario)
        for scen in scen_dirs:
            path_edges = scen / "edges.csv"
            if not path_edges.exists():
                raise RuntimeError(f"Missing edges in {scen}")
            df = pd.read_csv(path_edges)
            df["scenario"] = scen.name
            merged.append(df)
            if len(merged) <= 3:
                print(f"[BudgetedUncertainty] Loaded {scen.name} edges: {len(df)} rows", flush=True)

        edges_all = pd.concat(merged, ignore_index=True)
        print(f"[BudgetedUncertainty] Concatenated edges: {len(edges_all)} rows across {len(scen_dirs)} scenarios", flush=True)

        # Build a stable edge index
        edges_all["key"] = list(zip(edges_all["u"], edges_all["v"]))
        unique_edges = edges_all["key"].unique().tolist()
        self.edge_list = list(range(len(unique_edges)))
        key_to_id = {key: i for i, key in enumerate(unique_edges)}
        print(f"[BudgetedUncertainty] Unique edges: {len(unique_edges)}", flush=True)

        # Compute nominal and deviation costs
        print("[BudgetedUncertainty] Aggregating costs (groupby)...", flush=True)
        grouped = edges_all.groupby("key")["cost"]
        if self.nominal_rule == "avg":
            nom_series = grouped.mean()
        else:
            nom_series = grouped.min()
        max_series = grouped.max()
        stats = pd.concat([nom_series, max_series], axis=1)
        stats.columns = ["nom", "max"]

        stats = stats.reindex(unique_edges)  # keep deterministic order
        print("[BudgetedUncertainty] Aggregation done, building deviation series...", flush=True)
        dev_series = (stats["max"] - stats["nom"]).clip(lower=0.0)

        cost_nom = {key_to_id[key]: float(val) for key, val in zip(unique_edges, stats["nom"].tolist())}
        dev = {key_to_id[key]: float(val) for key, val in zip(unique_edges, dev_series.tolist())}

        self.cost_nominal = cost_nom
        self.deviation = dev
        self.edges = {eid: key for eid, key in enumerate(unique_edges)}
        print(
            f"[BudgetedUncertainty] Cost/deviation computed for {len(unique_edges)} edges "
            f"(nom_rule={self.nominal_rule})",
            flush=True,
        )

        # Build adjacency lists for flow and path reconstruction
        self.out_edges = {v: [] for v in self.nodes}
        self.in_edges = {v: [] for v in self.nodes}
        for e, (u, v) in self.edges.items():
            self.out_edges[u].append(e)
            self.in_edges[v].append(e)

        print(
            f"[BudgetedUncertainty] Loaded {len(self.nodes)} nodes, "
            f"{len(scen_dirs)} scenarios, {len(self.edges)} unique edges."
        )
        print(
            f"[BudgetedUncertainty] Preprocessing finished in {perf_counter() - t0:.2f}s",
            flush=True,
        )

    # ------------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------------
    def OptimizationModel(self, start_node=0, goal_node=None):
        self.start = start_node
        self.goal = max(self.nodes) if goal_node is None else goal_node

        log_dir = self.data_root / "BudgetedUncertainty"
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[BudgetedUncertainty] Gurobi log dir: {log_dir}", flush=True)
        model = gp.Model("ClassicBudgetedOptimization")

        model.Params.OutputFlag = 1
        model.Params.LogToConsole = 1
        model.Params.LogFile = str(log_dir / "gurobi.log")
        model.Params.DisplayInterval = 1
        model.setParam("TimeLimit", 60)
        model.setParam("MIPGap", 0.02)

        E = len(self.edges)

        print(
            f"[BudgetedUncertainty] Building model: edges={E}, nodes={len(self.nodes)}, "
            f"gamma={self.gamma}, start={self.start}, goal={self.goal}, "
            f"time_limit={model.Params.TimeLimit}, mip_gap={model.Params.MIPGap}, "
            f"log_file={model.Params.LogFile}",
            flush=True,
        )

        # Decision variables
        x = model.addVars(E, vtype=GRB.BINARY, name="x")
        pi = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="pi")
        rho = model.addVars(E, lb=0.0, vtype=GRB.CONTINUOUS, name="rho")

        # Objective
        model.setObjective(
            gp.quicksum(self.cost_nominal[e] * x[e] for e in range(E))
            + self.gamma * pi
            + gp.quicksum(rho[e] for e in range(E)),
            GRB.MINIMIZE,
        )

        # Robust constraints
        for e in range(E):
            model.addConstr(pi + rho[e] >= self.deviation[e] * x[e])

        # Flow balance
        for v in self.nodes:
            if v == self.start:
                rhs = 1
            elif v == self.goal:
                rhs = -1
            else:
                rhs = 0
            model.addConstr(
                gp.quicksum(x[e] for e in self.out_edges.get(v, []))
                - gp.quicksum(x[e] for e in self.in_edges.get(v, []))
                == rhs,
                name=f"flow_{v}",
            )

        model.optimize()
        print(
            f"[BudgetedUncertainty] Optimize finished: status={model.Status}, "
            f"solcount={model.SolCount}, runtime={model.Runtime:.2f}s, nodes={getattr(model, 'NodeCount', '-')}",
            flush=True,
        )
        if model.SolCount == 0:
            raise RuntimeError(f"Gurobi did not return a feasible solution (status={model.Status}).")

        self.model = model
        self.x = x
        self.pi = pi
        self.rho = rho
        return model

    def _reconstruct_path(self):
        chosen = {e for e in self.edges if self.x[e].X > 0.5}
        path = [self.start]
        current = self.start
        visited = set()

        # Follow chosen outgoing edges until goal or dead-end; guard loops
        while current != self.goal and current not in visited:
            visited.add(current)
            outgoing = [e for e in self.out_edges.get(current, []) if e in chosen]
            if not outgoing:
                raise RuntimeError(
                    f"Path reconstruction stopped early at node {current} (no chosen outgoing edges)."
                )
            # pick first deterministic edge
            e = sorted(outgoing)[0]
            _, nxt = self.edges[e]
            path.append(nxt)
            current = nxt

        self.path = path
        if current != self.goal:
            raise RuntimeError(
                f"Path reconstruction ended at node {current} instead of goal {self.goal}."
            )

    def GetResultData(self, model=None):
        if self.model is None:
            raise RuntimeError("Model not solved yet.")
        model = self.model

        self._reconstruct_path()

        nominal_cost = sum(self.cost_nominal[e] * self.x[e].X for e in self.edges)
        robust_cost = float(self.gamma * self.pi.X + sum(self.rho[e].X for e in self.edges))
        total_cost = model.ObjVal

        result = {
            "cost": total_cost,
            "nominal_cost": nominal_cost,
            "robust_cost": robust_cost,
            "node_path": self.path,
            "runtime": model.Runtime,
            "start_node": self.start,
            "goal_node": self.goal,
            "coords": self.coords,
            "gamma": self.gamma,
            "data_root": str(self.data_root),
        }

        # optional: include chosen edges with costs
        chosen_edges = [
            {
                "edge_id": e,
                "u": self.edges[e][0],
                "v": self.edges[e][1],
                "nominal_cost": self.cost_nominal[e],
                "deviation": self.deviation[e],
            }
            for e in self.edges
            if self.x[e].X > 0.5
        ]
        result["chosen_edges"] = chosen_edges

        self.result = result
        print(
            f"[BudgetedUncertainty] Result: path_len={len(self.path)}, "
            f"cost={total_cost:.3f}, nominal={nominal_cost:.3f}, robust={robust_cost:.3f}",
            flush=True,
        )

    def ExportPathOverlay(self, out_folder):
        out_dir = Path(out_folder) / "BudgetedUncertainty" / "overlays"
        out_dir.mkdir(parents=True, exist_ok=True)
        gif_path = out_dir / "budgeted_uncertainty_path_overlay.gif"
        self.animated_overlay = gif_path
        stgrf_folder = Path(out_folder)

        frame_paths = sorted(glob.glob(os.path.join(stgrf_folder, "scenario_*", "field.npy")))
        if not frame_paths:
            fallback = stgrf_folder / "scenario_000" / "field.npy"
            if fallback.exists():
                frame_paths = [str(fallback)]
            else:
                raise FileNotFoundError(f"No cost field frames found under {stgrf_folder}")
        frames = [np.load(fp) for fp in frame_paths]
        images = []

        xs = [self.coords[n][0] for n in self.path if n in self.coords]
        ys = [self.coords[n][1] for n in self.path if n in self.coords]

        for i, frame in enumerate(frames):
            plt.figure(figsize=(6, 6))
            plt.imshow(frame, cmap="viridis", origin="lower")
            if xs and ys:
                plt.plot(xs, ys, color="red", linewidth=2)

            out_png = out_dir / f"frame_{i:03d}.png"
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(out_png, dpi=150)
            plt.close()
            images.append(Image.open(out_png))

        if not images:
            raise RuntimeError("No overlay frames were generated.")

        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=500,
            loop=0,
        )
        print(f"[BudgetedUncertainty] Overlay saved to {gif_path} ({len(images)} frames).")

    def ExportRobustPathCSV(self, base_dir):
        out_dir = Path(base_dir) / "BudgetedUncertainty"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_file = out_dir / "budgeted_uncertainty_path.csv"
        df = pd.DataFrame({"node_id": self.path})
        df.to_csv(out_file, index=False, header=False)

    def ExportResultsJSON(self, base_dir):
        self.result["animated_overlay"] = str(self.animated_overlay)
        out_dir = Path(base_dir) / "ComparisonData"
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "BudgetedUncertainty_result.json"
        with open(json_path, "w") as jf:
            json.dump(self.result, jf, indent=2)
