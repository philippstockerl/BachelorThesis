import pandas as pd
import heapq as hq
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image
import glob
import os
from pathlib import Path
import math



def Preprocessing(nodes_path, edges_path, base_folder, robust_path_path):

    def load_nodes(path):
        """
        **Dictionary of Node Coordinates**

        Convert nodes.csv into nodes dict:

        nodes = {
            node_id: {x_coordinate, y_coordinate}
        }
        """
        df = pd.read_csv(path)
        return {int(r.node_id): (float(r.x), float(r.y)) for _, r in df.iterrows()}


    def load_edges(path, nodes):
        """
        **Adjacency List of Directed Edges**

        Create nested dictionary that contains all outgoing neighbours for each node u

        graph = {
            u: {v: cost, v2: cost, ...},
            ...
        }

        O(1) Performance for cost/ neighbour lookups and updates
        """
        df = pd.read_csv(path)
        graph = {}
        pred_list = {}
    
        for _, r in df.iterrows():
            u, v, cost = int(r.u), int(r.v), float(r.cost)

            # successors
            if u not in graph:
                graph[u] = {}
            graph[u][v] = cost

            # predecessors
            if v not in pred_list:
                pred_list[v] = []
            pred_list[v].append(u)

            # ensure every node exists even if it has no outgoing edges
            for n in nodes:
                if n not in graph:
                    graph[n] = {}
                if n not in pred_list:
                    pred_list[n] = []

        return graph, pred_list


    def load_scenarios(base_folder):
        """
        **Edge Cost Updates over time**

        Dictionary mapping scenario index -> list of updates

        scenario_changes = {
            scenario_id: [
                (u, v, new_cost),
                (u2, v2, new_cost2),
                ...
                ]
        }

        scenario_changes = {
            0: [(u, v, new_cost), (u, v, new_cost),...],        <- Initial graph for backward search and first for forward
            1: [(u, v, new_cost), (u, v, new_cost), ...],
        }

        """
        scenario_changes = {}
        for folder in sorted(os.listdir(base_folder)):
            if folder.startswith("scenario_"):
                scen_id = int(folder.split("_")[1])
                if scen_id == 0:
                    continue
                f = os.path.join(base_folder, folder, "edges.csv")
                df = pd.read_csv(f)
                scenario_changes[scen_id] = [
                    (int(r.u), int(r.v), float(r.cost)) 
                    for _, r in df.iterrows()
                ]
        return scenario_changes
    

    def load_robustPath(robust_path_path):
        """
        Load a robust path from CSV.
        Expected format: one node ID per line.
        Returns a list of integers.
        """
        if robust_path_path is None or robust_path_path == "" or not os.path.exists(robust_path_path):
            print("No robust path provided — running without restricted mode.")
            return []

        df = pd.read_csv(robust_path_path, header=None)
        path = [int(x) for x in df[0].tolist()]

        print(f"Loaded robust path with {len(path)} nodes.")
        return path

    
    nodes = load_nodes(nodes_path)
    graph, pred = load_edges(edges_path, nodes)
    
    scenario_changes = load_scenarios(base_folder)
    robustPath = load_robustPath(robust_path_path)

    return nodes, graph, pred, scenario_changes, robustPath


class DStarLite:
    def __init__(
        self,
        nodes,
        graph,
        pred,
        scenario_changes,
        robust_path,
        start_node=None,
        goal_node=None,
        beacon_cap=10,
        debug=False,
        debug_stride=0,
        max_steps=None,
    ):
        # graph data
        self.nodes = nodes
        self.graph = graph
        self.pred_list = pred
        self.scenario_changes = scenario_changes
        self.max_scenario = max(scenario_changes.keys(), default=0)
        
        # warm-start path
        self.robust_path = robust_path

        # core D* Lite maps
        self.g = {}
        self.rhs = {}

        # start/ goal node
        if start_node is None:
            start_node = min(nodes.keys())
        if goal_node is None:
            goal_node = max(nodes.keys())
        if start_node not in nodes:
            raise ValueError(f"Start node {start_node} is not in nodes.")
        if goal_node not in nodes:
            raise ValueError(f"Goal node {goal_node} is not in nodes.")
        self.s_start = int(start_node)
        self.final_goal = int(goal_node)
        self.current_goal = self.final_goal


        # {02"}
        self.U = []

        # {03"}
        self.km = 0
        
        # {29"}
        self.s_last = self.s_start


        # variables for the scenario switch
        self.step = 0
        self.current_scenario = 1
        self.last_applied_scenario = 0
        self.beacons_applied_for_scenario = None
        self.key = {}
        self.in_queue = {}

        # main solution data
        self.path = [self.s_start]
        self.cost = 0.0
        

        # Solution metrics
        self.arcs = []  # every arc taken by D* Lite during forward greedy execution
        self.cumulative_costs = []
        self.scenario_per_step = [0]
        self.start_time = 0.0
        self.end_time = 0.0

        self.warmstart_used = (len(robust_path) > 0)
        self.beacon_hits = []  # (step, scenario, beacon_id) when an active beacon is reached

        self.data_root = None
        self.robust_g = {}
        self.coords = nodes

        # grid size for overlay reconstruction
        self.grid_size = int(max(y for (_, y) in nodes.values()) + 1)
        self.replans = 0
        self.step_cost_details = []
        self.result = {}
        self.max_steps = int(max_steps) if max_steps else None

        self.baseline_cost = []
        self.baseline_cost_map = []

        self.manual_g_per_scenario = {}
        self.beacon_cap = beacon_cap

        self.grid_size = int(max(y for (_, y) in nodes.values()) + 1)

        # Calculate the appropriate number of steps to take 
        sx, sy = self.nodes[self.s_start]
        gx, gy = self.nodes[self.final_goal]
        min_steps = abs(sx - gx) + abs(gy - sy)
        num_scenarios = self.max_scenario + 1
        self.steps_per_layer = math.ceil(min_steps / num_scenarios)

        # Beacon implementation (scheduled one per scenario/layer)
        self.beacon_sequence = []
        if self.robust_path:
            # skip start/goal; cap to at most 10 evenly spaced beacons to limit queue blow-up
            inner = self.robust_path[1:-1]
            if inner:
                cap = max(1, int(self.beacon_cap))
                cap = min(cap, len(inner))
                if cap <= 0:
                    self.beacon_sequence = []
                else:
                    n = len(inner)
                    indices = []
                    for i in range(cap):
                        start = int(i * n / cap)
                        end = int((i + 1) * n / cap) - 1
                        end = max(start, end)
                        mid = (start + end) // 2
                        indices.append(mid)
                    indices = sorted(set(indices))
                    self.beacon_sequence = [inner[i] for i in indices]
        self.active_beacons = set()
        self.next_beacon_idx = 0
        self.goal_from_scenario = None
        total_layers = self.max_scenario + 1 if self.max_scenario is not None else 1
        self.beacon_interval = (
            math.ceil(total_layers / len(self.beacon_sequence))
            if self.beacon_sequence
            else 0
        )


    def ResetSearchForGoal(self):
        """
        Reinitialize g/rhs/queue for the current goal (beacon milestone or final goal).
        """
        self.g = {n: float('inf') for n in self.nodes}
        self.rhs = {n: float('inf') for n in self.nodes}
        self.key = {}
        self.in_queue = {}
        self.U = []
        self.km = 0
        self.s_last = self.s_start
        self.rhs[self.current_goal] = 0
        initial_key = self.CalculateKey(self.current_goal)
        self.key[self.current_goal] = initial_key
        hq.heappush(self.U, (initial_key, self.current_goal))

    def ActivateNextBeacon(self, force=False):
        """
        Promote the next beacon to be the temporary goal.
        Only one beacon per scenario layer unless force=True.
        """
        if self.next_beacon_idx >= len(self.beacon_sequence):
            return False
        if (not force) and (self.goal_from_scenario == self.current_scenario):
            return False

        idx = self.next_beacon_idx
        m = self.beacon_sequence[idx]
        self.next_beacon_idx += 1
        self.active_beacons = {m}
        self.current_goal = m
        self.goal_from_scenario = self.current_scenario
        self.beacons_applied_for_scenario = self.current_scenario
        self.ResetSearchForGoal()
        return True


    # procedure Inititalize()
    def Initialize(self):
        # {04"}
        for s in self.nodes.keys():
            self.g[s] = float('inf')
            self.rhs[s] = float('inf')



        # pick initial goal: first beacon if available, else final goal
        if self.beacon_sequence:
            self.current_goal = self.beacon_sequence[0]
            self.next_beacon_idx = 1
            self.active_beacons = {self.current_goal}
            self.goal_from_scenario = self.current_scenario
        else:
            self.current_goal = self.final_goal
            self.active_beacons = set()

        self.ResetSearchForGoal()
        self.beacons_applied_for_scenario = self.current_scenario
        
    # Manhattan heuristic
    def h(self, a, b):
        (x1, y1) = self.nodes[a]
        (x2, y2) = self.nodes[b]
        return abs(x1 - x2) + abs(y1 - y2)

    # Euclidean heuristic
    #def h(self, a, b):
    #    (x1, y1) = self.nodes[a]
    #   (x2, y2) = self.nodes[b]
    #    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

    # procedure CalculateKey(s)
    def CalculateKey(self, s):
        g_s = self.g.get(s, float('inf'))
        rhs_s = self.rhs.get(s, float('inf'))

        # {01"}
        min_g_rhs = min(g_s, rhs_s)
        return (
            min_g_rhs + self.h(self.s_start, s) + self.km,
            min_g_rhs
        )
    
    # procedure UpdateVertex(u)
    def UpdateVertex(self, u):
        if u != self.current_goal:
            rhs_new = float('inf')
            for s in self.pred(u):
                rhs_new = min(rhs_new, self.graph[s][u] + self.g[s])

            self.rhs[u] = rhs_new

        key_u = self.CalculateKey(u)

        if self.g[u] != self.rhs[u]:
            # push only if key changed or node believed absent from queue to limit duplicates
            if self.key.get(u) != key_u or not self.in_queue.get(u, False):
                self.key[u] = key_u
                self.in_queue[u] = True
                hq.heappush(self.U, (key_u, u))
        else:
            self.key[u] = (float('inf'), float('inf'))
            self.in_queue[u] = False

    # helper for UpdateVertex(u) rhs
    def pred(self, u):
        return self.pred_list.get(u, [])

    # helper to make code look like pseudocode
    def c(self, u, v):
        return self.graph.get(u, {}).get(v, float('inf'))
    

    # procedure ComputeShortestPath()
    def ComputeShortestPath(self):

        # helper to get top node in U (heapq limitation)
        def top_key():
            if not self.U:
                return (float('inf'), float('inf'))
            return self.U[0][0]
        
        # {10"}
        popped = 0
        while (top_key() < self.CalculateKey(self.s_start)) or (self.rhs[self.s_start] != self.g[self.s_start]):

            # {11"} & {12"}
            k_old, u = hq.heappop(self.U)
            popped += 1

            # this entry is being handled (even if stale)
            self.in_queue[u] = False

            # lazy deletion: skip stale entries
            if self.key.get(u) != k_old:
                continue
            # mark as removed; UpdateVertex will requeue if still inconsistent

            # {13"}
            k_new = self.CalculateKey(u)

            # {14"}
            if k_old < k_new:

                # {15"}
                hq.heappush(self.U, (k_new, u))
                self.key[u] = k_new
                continue

            # {16"}
            if self.g[u] > self.rhs[u]:

                # {17"}
                self.g[u] = self.rhs[u]

                # {18"}
                # NO U.Remove(u) since heapq can´t delete entries like that, use lazy deletion instead

                # {19"}
                for s in self.pred(u):


                    # {20"}
                    if s != self.current_goal:
                        self.rhs[s] = min(self.rhs[s], self.c(s, u) + self.g[u])

                    # {21"}
                    self.UpdateVertex(s)
            
            # {22"}
            else: 

                # {23"}
                g_old = self.g[u]

                # {24"}
                self.g[u] = float('inf')

                # {25"}
                to_update = list(self.pred(u)) + [u]
                for s in to_update:


                    # {26"}
                    if s != self.current_goal and self.rhs[s] == self.c(s, u) + g_old:

                        # reset rhs to inf
                        self.rhs[s] = float('inf')

                        # {27"}
                        for s2, cost_s_s2 in self.graph.get(s, {}).items():


                            self.rhs[s] = min(self.rhs[s], cost_s_s2 + self.g[s2])

                    # {28"}
                    self.UpdateVertex(s)

        return popped


    # procedure Main()
    def DStarMain(self):

        # start timer
        self.start_time = time.time()

        # {29"} duplicate, one in for DStar class __init__ initialize
        self.s_last = self.s_start


        # {30"}
        self.Initialize()


        # {31"}  
        pops = self.ComputeShortestPath()

        self.baseline_cost_map = dict(self.g)
        self.baseline_cost = self.g[self.s_start]

        # {32"}
        while True:
            # Included safety break if queue explodes; 
            # fixed with avg edge cost of connecting nodes instead of edge cost
            # being cost of travelling to node
            if self.max_steps is not None and len(self.path) >= self.max_steps:
                raise RuntimeError(
                    f"DStar Lite exceeded max_steps={self.max_steps} at node {self.s_start}."
                )

            # reached current goal (beacon or final)
            if self.s_start == self.current_goal:
                if self.current_goal == self.final_goal:
                    break
                # reached a beacon milestone; immediately activate the next beacon if available
                activated = self.ActivateNextBeacon(force=True)
                if not activated:
                    self.active_beacons.clear()
                    self.current_goal = self.final_goal
                    self.ResetSearchForGoal()
                continue

            # Ensure start is consistent before choosing the next step
            if (self.g.get(self.s_start, float('inf')) == float('inf')) or (self.g[self.s_start] != self.rhs[self.s_start]):
                pops = self.ComputeShortestPath()

            # {33"}
            if(self.g[self.s_start] == float('inf')):
                print("ERROR! No known path to goal.")
                print("Exiting DStar Lite, check STGRF output!")
                succs = [(s2, self.c(self.s_start, s2), self.g.get(s2), self.rhs.get(s2), s2 in self.active_beacons)
                    for s2 in self.graph.get(self.s_start, {})]
                print("Dead end at", self.s_start, "succs:", succs)

                return None
            
            # helper variables
            next_s = None

            # {34"}
            candidates = []
            for s2, cost_s_s2 in self.graph.get(self.s_start, {}).items():
                g_s2 = self.g.get(s2, float('inf'))
                if not math.isfinite(g_s2):
                    continue
                val = cost_s_s2 + g_s2
                candidates.append((val, self.h(s2, self.current_goal), s2))

            if not candidates:
                for s2, cost_s_s2 in self.graph.get(self.s_start, {}).items():
                    rhs_s2 = self.rhs.get(s2, float('inf'))
                    if not math.isfinite(rhs_s2):
                        continue
                    val = cost_s_s2 + rhs_s2
                    candidates.append((val, self.h(s2, self.current_goal), s2))

            candidates.sort()
            if candidates:
                 _, next_s = candidates[0]

            if next_s is None:
                succs = [
                    (s2, self.c(self.s_start, s2), self.g.get(s2), self.rhs.get(s2), s2 in self.active_beacons)
                    for s2 in self.graph.get(self.s_start, {})
                ]

                print("ERROR! Dead end, no successors available to walk to!")
                print("Exiting DStar Lite, check ST-GRF output!")
                return None


            old_start = self.path[-1]

            # {35"}
            self.s_start = next_s
            if next_s in self.active_beacons:
                self.beacon_hits.append((len(self.path)-1, self.current_scenario, next_s))

            # shift keys to the new start position (D* Lite invariant)
            self.km += self.h(self.s_last, self.s_start)
            self.s_last = self.s_start

            # step counter for scenario switch 
            self.step += 1

            self.path.append(next_s)


            # NOT ORIGINAL
            # compute cost
            arc_cost = self.c(old_start, next_s)
            self.cost += arc_cost   
            self.step_cost_details.append({
                "t": len(self.path)-1,                     # time index
                "u": old_start,
                "v": next_s,
                "scenario": self.current_scenario,
                "cost": arc_cost,
                "coords_u": self.nodes[old_start],
                "coords_v": self.nodes[next_s]
            })

            # === SOLUTION METRICS ===
            self.arcs.append((old_start, next_s, arc_cost))
            self.cumulative_costs.append(self.cost)
            self.scenario_per_step.append(self.current_scenario)



            # {36"} <- own implementation of when scenarios (= cost) should change
            # current self.graph with old costs is repopulated with new costs in {40"}-{42"}
            # e.g. the algorithm continues to work with the same logic as the pseudocode
            # only now we can introduce new costs at time frames we want!

            # NOT ORIGINAL
            if(self.step == self.steps_per_layer):
                self.current_scenario = min(self.current_scenario + 1, self.max_scenario)
                self.step = 0
            # change to next scenario
            scenario_changed = self.current_scenario != self.last_applied_scenario
            if scenario_changed:
                raw_changes = self.scenario_changes.get(self.current_scenario, [])
                changed_edges = []
                for (u, v, new_cost) in raw_changes:
                    old_cost = self.c(u, v)
                    if old_cost != new_cost:
                        changed_edges.append((u, v, new_cost))
                self.last_applied_scenario = self.current_scenario
            else:
                changed_edges = []



            # {37"}
            need_replan = False
            replan_due_to_edges = bool(changed_edges)
            if changed_edges:

                # {38"}
                self.km += self.h(self.s_last, self.s_start)

                # {39"}
                self.s_last = self.s_start

                # {40"}
                for(u, v, new_cost) in changed_edges:

                    # {41"}
                    c_old = self.c(u,v)

                    # {42"}
                    self.graph[u][v] = new_cost

                    # {43"}
                    if c_old > new_cost:
                        # {44"}
                        if u != self.current_goal:
                            self.rhs[u] = min(self.rhs[u], new_cost + self.g[v])
                    
                    # {45"}
                    elif self.rhs[u] == c_old + self.g[v]:

                    

                        # {46"}
                        if u != self.current_goal:
                            self.rhs[u] = float('inf')
                            for s2, cost_s2 in self.graph.get(u, {}).items():
                                self.rhs[u] = min(self.rhs[u], cost_s2 + self.g[s2])

                    # {47"}
                    self.UpdateVertex(u)
                need_replan = True
            
            if scenario_changed:
                should_activate_beacon = (
                    self.beacon_interval > 0
                    and ((self.current_scenario - 1) % self.beacon_interval == 0)
                )
                applied_beacon = False
                if should_activate_beacon:
                    applied_beacon = self.ActivateNextBeacon(force=False)
                if applied_beacon:
                    need_replan = True

            if need_replan:
                # {48"}
                pops = self.ComputeShortestPath()
                if replan_due_to_edges and pops > 0:
                    self.replans += 1

        # timer end
        self.end_time = time.time()
        self.standard_path = list(self.path)

        self.result = {
        "node_path": self.path,
        "arcs": [{"u": u, "v": v, "cost": c} for (u, v, c) in self.arcs],
        "cumulative_costs": self.cumulative_costs,
        "cost": self.cost,
        "total_cost": self.cost,
        "scenario_per_step": self.scenario_per_step,
        "runtime": self.end_time - self.start_time,
        "replans": self.replans,
        "warmstart_used": self.warmstart_used,
        "data_root": self.data_root,
        "robust_path": self.robust_path,
        "robust_g": self.robust_g,
        "coords": self.coords,
        "grid_size": self.grid_size,
        "step_cost_details": self.step_cost_details,
        "baseline_cost": self.baseline_cost,
        "baseline_cost_map": self.baseline_cost_map
        }
        if self.beacon_hits:
            self.result["beacon_hits"] = self.beacon_hits
        if self.beacon_sequence:
            self.result["beacon_sequence"] = list(self.beacon_sequence)
            # store coordinates for easy plotting downstream
            self.result["beacon_coords"] = [self.coords[b] for b in self.beacon_sequence]

        # solution output
        print("\n=== D* Lite Finished ===")
        print("Final path:", self.path)
        print("Total cost:", self.cost)
        print("Total steps:", len(self.path)-1)
        print(f"Total Runtime:{self.end_time - self.start_time}s")


    def ExportPathOverlay(self, base_folder, out_folder):
        out_dir = Path(out_folder) 
        out_dir.mkdir(parents=True, exist_ok=True)

        scenario_paths = sorted(glob.glob(os.path.join(base_folder, "scenario_*", "field.npy")))
        frames = [np.load(fp) for fp in scenario_paths]

        if not frames or len(self.path) < 2:
            return

        images = []
        path_nodes = self.path
        xs = [self.nodes[n][0] for n in path_nodes]
        ys = [self.nodes[n][1] for n in path_nodes]

        # Build colored path segments by scenario slice.
        scenarios = getattr(self, "scenario_per_step", []) or []
        segments = []
        segment_scenarios = []
        for i in range(1, len(path_nodes)):
            segments.append(
                [
                    (xs[i - 1], ys[i - 1]),
                    (xs[i], ys[i]),
                ]
            )
            if i < len(scenarios):
                segment_scenarios.append(int(scenarios[i]))
            elif scenarios:
                segment_scenarios.append(int(scenarios[-1]))
            else:
                segment_scenarios.append(0)

        unique_scenarios = []
        seen = set()
        for sc in segment_scenarios:
            if sc not in seen:
                unique_scenarios.append(sc)
                seen.add(sc)
        styles = ["-", ":"]
        scenario_style = {sc: styles[idx % len(styles)] for idx, sc in enumerate(unique_scenarios)}
        segment_colors = ["#FF0000"] * len(segment_scenarios)
        segment_styles = [scenario_style.get(sc, styles[0]) for sc in segment_scenarios]

        # Beacon markers (milestones).
        beacon_nodes = list(getattr(self, "beacon_sequence", []) or [])
        beacon_coords = [(self.nodes[n][0], self.nodes[n][1]) for n in beacon_nodes if n in self.nodes]
        show_labels = len(beacon_coords) <= 20

        for i, frame in enumerate(frames):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(frame, cmap="viridis", origin="lower")
            line_collection = LineCollection(
                segments,
                colors=segment_colors,
                linewidths=2.0,
                linestyles=segment_styles,
                zorder=3,
            )
            ax.add_collection(line_collection)

            if beacon_coords:
                bx = [c[0] for c in beacon_coords]
                by = [c[1] for c in beacon_coords]
                ax.scatter(
                    bx,
                    by,
                    marker="*",
                    s=90,
                    c="white",
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=4,
                )
                if show_labels:
                    for idx, (bx_i, by_i) in enumerate(beacon_coords, start=1):
                        ax.text(
                            bx_i + 0.1,
                            by_i + 0.1,
                            str(idx),
                            fontsize=8,
                            color="black",
                            zorder=5,
                        )

            out_png = out_dir / f"dstar_frame_{i:03d}.png"
            ax.axis("off")
            ax.set_title(f"Scenario {i:03d}")
            ax.set_xlabel("X coordinate")
            ax.set_ylabel("Y coordinate")
            fig.tight_layout(pad=0)
            fig.savefig(out_png, dpi=150)
            plt.close(fig)

            images.append(Image.open(out_png))

        gif_path = out_dir / "dstar_overlay_animation.gif"
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=500,
            loop=0
        )



def main(nodes_path, edges_path, base_folder, robust_path_path, out_folder):
    # 1. Load input data
    nodes, graph, pred, scenario_changes, robust_path = Preprocessing(
        nodes_path, edges_path, base_folder, robust_path_path
    )

    # 2. Create DStar object
    DStar = DStarLite(nodes, graph, pred, scenario_changes, robust_path)

    # 3. Run algorithm
    DStar.DStarMain()


    # 4. Export GIF + overlays
    DStar.ExportPathOverlay(base_folder, out_folder)

if __name__ == "__main__":
    main()
