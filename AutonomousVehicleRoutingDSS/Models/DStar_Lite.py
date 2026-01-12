import pandas as pd
import heapq as hq
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
from pathlib import Path



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
    def __init__(self, nodes, graph, pred, scenario_changes, robust_path):
        # graph data
        self.nodes = nodes
        self.graph = graph
        self.pred_list = pred
        self.scenario_changes = scenario_changes
        
        # warm-start path
        self.robust_path = robust_path

        # core D* Lite maps
        self.g = {}
        self.rhs = {}

        # start/ goal node
        self.s_start = min(nodes.keys())
        self.s_goal = max(nodes.keys())


        # {02"}
        self.U = []

        # {03"}
        self.km = 0
        
        # {29"}
        self.s_last = self.s_start


        # variables for the scenario switch
        self.step = 0
        self.current_scenario = 1

        self.key = {}

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

        self.data_root = None
        self.robust_g = {}
        self.coords = nodes
        # grid size for overlay reconstruction
        self.grid_size = int(max(y for (_, y) in nodes.values()) + 1)
        self.replans = 0
        self.step_cost_details = []
        self.result = {}

        self.baseline_cost = []
        self.baseline_cost_map = []

        self.manual_g_per_scenario = {}



    
    # procedure Inititalize()
    def Initialize(self):
        # {04"}
        for s in self.nodes.keys():
            self.g[s] = float('inf')
            self.rhs[s] = float('inf')

        #if len(self.robust_path) > 0:
        #    print("Injecting manual robust g-values...")
        
        #    g_manual = self.ComputeGValues()
        #    self.robust_g = g_manual
        
        #    for n, gv in g_manual.items():
        #        self.g[n] = gv
        #        self.rhs[n] = gv


        if len(self.robust_path) > 0:

            # Compute g values for ENTIRE robust path beforehand 
            manual_g = self.ComputeGValues()
            
            #rp = self.robust_path[:]  # do i need to flip? we start backwards 
            
            
            beacons = self.robust_path[1:-1:3]
            #beacons = self.robust_path[1:-1:3]

            print("Milestones:", beacons)

            # calculate beacon-key pair
                # k2 = min(g[s], rhs[s])  <- rhs = 0 -> k2 = g[s]
                # k1 = min(g[s], rhs[s]) + h(s_start, s) + km
                #    = min(g[s], 'inf') + h(s_start, s) + 0

            for m in beacons:
                # inside for m in beacons:
                g_m = manual_g[m]
                self.g[m] = float('inf')      # keep g unset
                self.rhs[m] = g_m             # seed rhs with manual upper bound
                key_m = self.CalculateKey(m)
                self.key[m] = key_m
                hq.heappush(self.U, (key_m, m))
            
            # Precompute g-values for robust path in every scenario
            self.manual_g_per_scenario = {}

            # ALWAYS include scenario 0 + all others
            all_scenarios = [0] + sorted(self.scenario_changes.keys())

            for scen in all_scenarios:
                # build graph_copy starting from base costs
                graph_copy = {u: dict(vs) for u, vs in self.graph.items()}

                # apply scenario changes only if scen > 0
                if scen > 0:
                    for (u, v, new_cost) in self.scenario_changes[scen]:
                        graph_copy[u][v] = new_cost

                # compute g-values along robust path
                gvals = {}
                rp = self.robust_path
                gvals[rp[-1]] = 0.0
                for i in range(len(rp)-2, -1, -1):
                    u = rp[i]
                    v = rp[i+1]
                    gvals[u] = graph_copy[u][v] + gvals[v]

                self.manual_g_per_scenario[scen] = gvals


        # {05"}
        self.rhs[self.s_goal] = 0

        self.initial_key = self.CalculateKey(self.s_goal)
        self.key[self.s_goal] = self.initial_key

        # {06"}
        #initial_key = self.CalculateKey(self.s_goal)
        hq.heappush(self.U, (self.initial_key, self.s_goal))
        
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

        # {07"}
        if u != self.s_goal:
            self.rhs[u] = float('inf')
            for s in self.pred(u): 
                self.rhs[u] = min(self.rhs[u], self.graph[s][u] + self.g[s])

        key_u = self.CalculateKey(u)

        # {07"} & {08"}
        if self.g[u] != self.rhs[u]:
            self.key[u] = key_u          
            hq.heappush(self.U, (key_u, u)) 
        else:
            # {09"}
            self.key[u] = (float('inf'), float('inf'))

    # helper for UpdateVertex(u) rhs
    def pred(self, u):
        return self.pred_list.get(u, [])

    # helper to make code look like pseudocode
    def c(self, u, v):
        return self.graph.get(u, {}).get(v, float('inf'))
    
    # Hybrid approach: manually calculate the g-values for the robust path
    def ComputeGValues(self):
        rp = self.robust_path
        g_manual = {}

        g_manual[rp[-1]] = 0.0

        for i in range(len(rp)-2, -1, -1):
            u = rp[i]
            v = rp[i+1]
            cost_uv = self.graph[u][v]
            g_manual[u] = cost_uv + g_manual[v]
        
        return g_manual
    


        """
        Build a set of allowed nodes around the robust path. 
        radius = Manhattan distance from robust path.
        """
        envelope = set()

        # reverse map: coordinate → node
        coord_to_node = {coord: n for n, coord in self.nodes.items()}

        # Build a grid index for faster lookup
        from collections import defaultdict
        grid = defaultdict(list)
        for node, (x, y) in self.nodes.items():
            grid[(int(x), int(y))].append(node)

        robust_coords = [self.nodes[n] for n in self.robust_path]

        for (rx, ry) in robust_coords:
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    cx = rx + dx
                    cy = ry + dy
                    if (cx, cy) in grid:
                        for node in grid[(cx, cy)]:
                            envelope.add(node)

        return envelope

    # procedure ComputeShortestPath()
    def ComputeShortestPath(self):

        # helper to get top node in U
        def top_key():
            if not self.U:
                return (float('inf'), float('inf'))
            return self.U[0][0]
        
        # {10"}
        while (top_key() < self.CalculateKey(self.s_start)) or (self.rhs[self.s_start] != self.g[self.s_start]):

            # {11"} & {12"}
            k_old, u = hq.heappop(self.U)

            # lazy deletion: skip stale entries
            if self.key.get(u) != k_old:
                continue


            # NOT ORIGINAL
            # HARD LOCK: NO DEVIATION DURING BACKWARD SEARCH
            #if hasattr(self, 'protect_robust') and self.protect_robust:
            #    if u in self.robust_path:
            #        # skip any Bellman update; keep injected value
            #        continue
            
            # NOT ORIGINAL
            # Allow robust nodes, but only keep their injected values if they are still minimal
            #if hasattr(self, 'protect_robust') and self.protect_robust:
            #    if u in self.robust_path:
            #        # compute what rhs[u] WOULD be
            #        rhs_new = self.rhs[u]
            
            #        # If the manual robust g-value is still the best, skip updates
            #        if self.g[u] <= rhs_new:
            #           continue
            #        # Otherwise: fall through → allow updates!

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
                    if s != self.s_goal:
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
                    if s != self.s_goal and self.rhs[s] == self.c(s, u) + g_old:

                        # reset rhs to inf
                        self.rhs[s] = float('inf')

                        # {27"}
                        for s2, cost_s_s2 in self.graph.get(s, {}).items():


                            self.rhs[s] = min(self.rhs[s], cost_s_s2 + self.g[s2])

                    # {28"}
                    self.UpdateVertex(s)


    def ExportPathOverlay(self, base_folder, out_folder):
        out_dir = Path(out_folder) 
        out_dir.mkdir(parents=True, exist_ok=True)

        scenario_paths = sorted(glob.glob(os.path.join(base_folder, "scenario_*", "field.npy")))
        frames = [np.load(fp) for fp in scenario_paths]

        images = []
        xs = [self.nodes[n][0] for n in self.path]
        ys = [self.nodes[n][1] for n in self.path]

        for i, frame in enumerate(frames):
            plt.figure(figsize=(6,6))
            plt.imshow(frame, cmap="viridis", origin="lower")
            plt.plot(xs, ys, color='red', linewidth=2)

            out_png = out_dir / f"dstar_frame_{i:03d}.png"
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(out_png, dpi=150)
            plt.close()

            images.append(Image.open(out_png))

        gif_path = out_dir / "dstar_overlay_animation.gif"
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=500,
            loop=0
        )
    

    # procedure Main()
    def DStarMain(self):

        # start timer
        self.start_time = time.time()

        # {29"} duplicate, one in for DStar class __init__ initialize
        self.s_last = self.s_start

        #self.protect_robust = True

        # {30"}
        self.Initialize()


        # {31"}  
        self.ComputeShortestPath()

        # NOT ORIGINAL
        self.baseline_cost_map = dict(self.g)
        self.baseline_cost = self.g[self.s_start]
        print("Baseline backward-search cost (A*):", self.baseline_cost)

        # NOT ORIGINAL
        # INSERT ROBUST G VALUES *HERE*
        #if len(self.robust_path) > 0:
        #    g_manual = self.ComputeGValues()
        #    for n, gv in g_manual.items():
        #        self.g[n] = gv
        # DO NOT TOUCH rhs[n]!

        #self.protect_robust = False
        
        # NOT ORIGINAL: SKIP SCENARIO 0 FOR FORWARD SEARCH ENTIRELY!






        # {32"}
        while(self.s_start != self.s_goal):

            # {33"}
            if(self.g[self.s_start] == float('inf')):
                print("ERROR! No known path to goal.")
                print("Exiting DStar Lite, check ST-GRF output!")
                return None
            
            # helper variables
            next_s = None
            min_cost = float('inf')

            # {34"}
            for s2, cost_s_s2 in self.graph.get(self.s_start, {}).items():
                val = cost_s_s2 + self.g[s2]
                if val < min_cost:
                    min_cost = val
                    next_s = s2

            if next_s is None:
                print("ERROR! Dead end, no successors available to walk to!")
                print("Exiting DStar Lite, check ST-GRF output!")
                return None

            
            old_start = self.path[-1]

            # {35"}
            self.s_start = next_s

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
            # CONDITION TO SWITCH SCENARIO: HARDCODED TO GRID SIZE 100

            if(self.step == 20):
                self.current_scenario += 1
                self.step = 0
            # change to next scenario
            changed_edges = self.scenario_changes.get(self.current_scenario, [])



            # {37"}
            if changed_edges:


                # === INSERT THE SCENARIO'S MILESTONE ===

                # heuristic shift for moved start
                self.km += self.h(self.s_last, self.s_start)
                self.s_last = self.s_start

                # apply cost changes
                for (u, v, new_cost) in changed_edges:
                    c_old = self.c(u, v)
                    self.graph[u][v] = new_cost
                    if c_old > new_cost:
                        self.replans += 1
                        if u != self.s_goal:
                            self.rhs[u] = min(self.rhs[u], new_cost + self.g[v])
                    elif self.rhs[u] == c_old + self.g[v]:
                        if u != self.s_goal:
                            self.rhs[u] = float('inf')
                            for s2, cost_s2 in self.graph.get(u, {}).items():
                                self.rhs[u] = min(self.rhs[u], cost_s2 + self.g[s2])
                    self.UpdateVertex(u)

                # insert the scenario’s milestone after costs/km are up to date
                if len(self.robust_path) > 0:
                    manual_g = self.manual_g_per_scenario[self.current_scenario]
                    all_scenarios = [0] + sorted(self.scenario_changes.keys())
                    ratio = self.current_scenario / max(all_scenarios)
                    milestone_index = int(ratio * (len(self.robust_path) - 2))
                    m = self.robust_path[milestone_index]
                    g_m = manual_g[m]

                    self.g[m] = float('inf')
                    self.rhs[m] = g_m
                    key_m = self.CalculateKey(m)
                    self.key[m] = key_m
                    hq.heappush(self.U, (key_m, m))

                self.ComputeShortestPath()













                # {38"}
                #self.km += self.h(self.s_last, self.s_start)

                # {39"}
                #self.s_last = self.s_start

                # {40"}
                for(u, v, new_cost) in changed_edges:

                    # {41"}
                    c_old = self.c(u,v)

                    # {42"}
                    self.graph[u][v] = new_cost

                    # {43"}
                    if c_old > new_cost:
                        
                        self.replans += 1
                        # {44"}
                        if u != self.s_goal:
                            self.rhs[u] = min(self.rhs[u], new_cost + self.g[v])
                    
                    # {45"}
                    elif self.rhs[u] == c_old + self.g[v]:

                    

                        # {46"}
                        if u != self.s_goal:
                            self.rhs[u] = float('inf')
                            for s2, cost_s2 in self.graph.get(u, {}).items():
                                self.rhs[u] = min(self.rhs[u], cost_s2 + self.g[s2])

                    # {47"}
                    self.UpdateVertex(u)
                
                # {48"}
                self.ComputeShortestPath()

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

        # solution output
        print("\n=== D* Lite Finished ===")
        print("Final path:", self.path)
        print("Total cost:", self.cost)
        print("Total steps:", len(self.path)-1)
        print(f"Total Runtime:{self.end_time - self.start_time}s")







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