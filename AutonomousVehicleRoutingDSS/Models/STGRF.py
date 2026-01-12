from __future__ import annotations
import os, io, json
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from scipy.stats import gaussian_kde
import gstools as gs
import networkx as nx


class STGRF:

    def __init__(
        self,
        seed=None,
        data_root=None,
        grid_size=None,
        cell_size=None,
        kernel=None,
        variance=None,
        len_scale=None,
        anis=None,
        use_normalization=None,
        nu=1.5,
        alpha=1.0,
        num_scenarios=None,
        gif_2d=False,
        gif_3d=False,
        gif_hist=False,
        gif_heat=False,
        gif_kde=False,
        gif_violin=False,
        frames=False,
        frame_2d_legend=False,
        three_path_overlay=None,
        robust_path_overlay=None,
        show_log=None,
        export_metadata=None
    ):

        # Core
        self.seed = seed
        self.data_root = data_root
        self.grid_size = grid_size
        self.cell_size = cell_size

        # Kernel
        self.kernel = kernel
        self.variance = variance
        self.len_scale = len_scale
        self.anis = anis
        self.use_normalization = use_normalization
        self.nu = nu
        self.alpha = alpha

        # Time axis
        self.num_scenarios = num_scenarios

        # Visualization flags
        self.gif_2d = gif_2d
        self.gif_3d = gif_3d
        self.gif_hist = gif_hist
        self.gif_heat = gif_heat
        self.gif_kde = gif_kde
        self.gif_violin = gif_violin
        self.frames = frames
        self.frame_2d_legend = frame_2d_legend

        # Others
        self.three_path_overlay = three_path_overlay
        self.robust_path_overlay = robust_path_overlay
        self.show_log = show_log
        self.export_metadata = bool(export_metadata)


    def JSONImport(self, json_path):
        json_path = Path(json_path)
        with json_path.open("r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        self.DSSImport(cfg)


    def DSSImport(self, cfg):
        self.seed = cfg.get("seed", self.seed)
        paths = cfg.get("paths", {})
        self.data_root = paths.get("data_root", self.data_root)

        # SRF
        srf = cfg.get("srf", {})
        self.grid_size = srf.get("grid_size", self.grid_size)
        self.cell_size = srf.get("cell_size", self.cell_size)
        self.kernel = srf.get("kernel", self.kernel)
        self.variance = srf.get("variance", self.variance)
        self.len_scale = srf.get("length_scale", self.len_scale)
        self.anis = srf.get("anis", self.anis)
        self.use_normalization = srf.get("use_global_normalization", self.use_normalization)
        self.nu = srf.get("nu", self.nu)
        self.alpha = srf.get("alpha", self.alpha)

        self.num_scenarios  = srf.get("num_scenarios", self.num_scenarios)

        # Visual
        viz = cfg.get("visualizations", {})
        self.gif_2d = viz.get("gif_2d", self.gif_2d)
        self.gif_3d = viz.get("gif_3d", self.gif_3d)
        self.gif_hist = viz.get("gif_hist", self.gif_hist)
        self.gif_heat = viz.get("gif_heat", self.gif_heat)
        self.gif_kde = viz.get("gif_kde", self.gif_kde)
        self.gif_violin = viz.get("gif_violin", self.gif_violin)
        self.frames = viz.get("frames", self.frames)
        self.frame_2d_legend = viz.get("frame_2d_legend", self.frame_2d_legend)
        self.export_metadata = viz.get("export_metadata", self.export_metadata)


    def ensure_dir(self, path):
        os.makedirs(path, exist_ok=True)
        return Path(path)


    def _figure_to_array(self, fig, save_path=None, desktop_copy=False, desktop_name=None):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=120)
        # Optionally export a copy to Desktop/STGRF_FRAMES
        if desktop_copy:
            desktop_dir = Path.home() / "Desktop" / "STGRF_FRAMES"
            desktop_dir.mkdir(parents=True, exist_ok=True)
            if desktop_name:
                desktop_file = desktop_name
            elif save_path:
                # Try to include scenario folder for uniqueness (scenario_xxx/frame/frame.png → scenario_xxx__frame__frame.png)
                if len(save_path.parents) > 1:
                    desktop_file = f"{save_path.parents[1].name}__{save_path.parent.name}__{save_path.name}"
                else:
                    desktop_file = f"{save_path.parent.name}__{save_path.name}"
            else:
                desktop_file = "frame.png"
            fig.savefig(desktop_dir / desktop_file, dpi=120)
        plt.close(fig)
        buf.seek(0)
        return imageio.imread(buf)


    def build_nodes(self):
        N = self.grid_size
        ids = np.arange(N * N)
        xs = np.tile(np.arange(N), N)
        ys = np.repeat(np.arange(N), N)
        return pd.DataFrame({"node_id": ids, "x": xs, "y": ys})


    def build_edges_networkx(self, f_t):
        """
        Build edges using a NetworkX directed grid graph
        with 4-connectivity.
        Output format: DataFrame(u, v, cost)
        """
        N = self.grid_size

        # Base 4-connectivity graph
        G = nx.grid_2d_graph(N, N, create_using=nx.DiGraph)

        rows = []
        for (x, y) in G.nodes():
            u = x * N + y  # flatten
            for (nx_, ny_) in G.successors((x, y)):
                v = nx_ * N + ny_
                f_u = float(f_t[x, y])
                f_v = float(f_t[nx_, ny_])
                cost = 0.5 * (f_u + f_v)
                rows.append((u, v, cost))

        return pd.DataFrame(rows, columns=["u", "v", "cost"])


    def _plot_surface3d_rotate(self, field, scen_dir=None):

        N = field.shape[0]
        X, Y = np.meshgrid(np.arange(N), np.arange(N))
        frames = []

        for angle in range(0, 360, 15):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(X, Y, field, cmap="viridis")
            ax.view_init(elev=40, azim=angle)
            ax.set_xlabel("X coordinate")
            ax.set_ylabel("Y coordinate")
            ax.set_zlabel("Cost")

            img = self._figure_to_array(
                fig,
                scen_dir / f"surface3d_rotate/rot_{angle:03}.png"
                if scen_dir else None
            )
            frames.append(img)

        return frames


    def STRF(self):

        x = np.arange(0, self.grid_size, self.cell_size)
        y = np.arange(0, self.grid_size, self.cell_size)
        t = np.arange(self.num_scenarios)


        Kernel = getattr(gs, self.kernel)
        kernel_kwargs = {
            "temporal": False,
            "spatial_dim": 3,
            "var": self.variance,
            "len_scale": self.len_scale,
            "anis": self.anis,
        }
        if str(self.kernel).lower() == "matern":
            kernel_kwargs["nu"] = self.nu
        if str(self.kernel).lower() == "stable":
            kernel_kwargs["alpha"] = self.alpha

        model = Kernel(**kernel_kwargs)

        srf = gs.SRF(model, seed=self.seed)
        field = srf.structured([x, y, t])

        if self.use_normalization:
            field = (field - np.min(field)) / (np.max(field) - np.min(field))

        T = field.shape[2]

        base_dir = self.ensure_dir(self.data_root)

        nodes = self.build_nodes()
        nodes.to_csv(base_dir / "nodes.csv", index=False)

        # Global GIF buffers
        self.gif_buffers = {
            "gif_2d": [],
            "gif_hist": [],
            "gif_kde": [],
            "gif_violin": [],
            "gif_heat": [],
            "gif_3d": [],
        }

        for t_idx in range(T):

            scen_dir = base_dir / f"scenario_{t_idx:03d}"
            scen_dir.mkdir(parents=True, exist_ok=True)
            f_t = field[:, :, t_idx]

            # Save field
            np.save(scen_dir / "field.npy", f_t)

            # Build edges
            edges = self.build_edges_networkx(f_t)

            edges.to_csv(scen_dir / "edges.csv", index=False)


            if self.frames:
                fig = plt.figure(figsize=(6,6))
                ax = fig.add_subplot(111)
                if self.frame_2d_legend:
                    im = ax.imshow(f_t, cmap="viridis", origin="lower")
                    ax.set_title(f"Scenario {t_idx:03d}")
                    ax.set_xlabel("X coordinate")
                    ax.set_ylabel("Y coordinate")
                    step = max(1, f_t.shape[0] // 10)
                    ax.set_xticks(np.arange(0, f_t.shape[0], step))
                    ax.set_yticks(np.arange(0, f_t.shape[0], step))
                    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label("Cost")
                    fig.tight_layout()
                else:
                    ax.axis("off")
                    ax.imshow(f_t, cmap="viridis", origin="lower")
                    fig.tight_layout(pad=0)

                frame_path = scen_dir / f"frame/frame_{t_idx:03d}.png"
                self._figure_to_array(
                    fig,
                    frame_path,
                    desktop_copy=True,
                    desktop_name=f"scenario_{t_idx:03d}.png"
                )

            if self.gif_2d:
                fig = plt.figure(figsize=(6,6))
                ax = fig.add_subplot(111)
                ax.axis("on")
                ax.imshow(f_t, cmap="viridis", origin="lower")
                fig.tight_layout(pad=0)
                img = self._figure_to_array(fig)
                self.gif_buffers["gif_2d"].append(img)

            # ============================
            # Histogram
            # ============================
            if self.frames or self.gif_hist:
                fig = plt.figure()
                plt.hist(f_t.flatten(), bins=40)
                img = self._figure_to_array(
                    fig,
                    scen_dir / "hist/hist.png" if self.frames else None
                )
                if self.gif_hist:
                    self.gif_buffers["gif_hist"].append(img)

            # ============================
            # KDE
            # ============================
            if self.frames or self.gif_kde:
                vals = f_t.flatten()
                xs = np.linspace(vals.min(), vals.max(), 200)
                fig = plt.figure()
                plt.plot(xs, gaussian_kde(vals)(xs))
                img = self._figure_to_array(
                    fig,
                    scen_dir / "kde/kde.png" if self.frames else None
                )
                if self.gif_kde:
                    self.gif_buffers["gif_kde"].append(img)

            # ============================
            # Violin
            # ============================
            if self.frames or self.gif_violin:
                fig = plt.figure()
                plt.violinplot(f_t.flatten(), showmeans=True)
                img = self._figure_to_array(
                    fig,
                    scen_dir / "violin/violin.png" if self.frames else None
                )
                if self.gif_violin:
                    self.gif_buffers["gif_violin"].append(img)

            # ============================
            # Joint heatmap
            # ============================
            if self.frames or self.gif_heat:
                N = f_t.shape[0]
                fig = plt.figure()
                coords_x = np.repeat(np.arange(N), N)
                coords_y = np.tile(np.arange(N), N)
                plt.hist2d(coords_x, coords_y, bins=40, weights=f_t.flatten(), cmap="viridis")
                plt.colorbar()
                img = self._figure_to_array(
                    fig,
                    scen_dir / "joint/joint.png" if self.frames else None
                )
                if self.gif_heat:
                    self.gif_buffers["gif_heat"].append(img)

            # ============================
            # 3D rotation frames
            # ============================
            # 3D: ONE rotation frame per scenario instead of full rotation
            if self.frames or self.gif_3d:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                N = f_t.shape[0]
                X, Y = np.meshgrid(np.arange(N), np.arange(N))
                ax.plot_surface(X, Y, f_t, cmap="viridis")
                ax.view_init(elev=40, azim=45)          # fixed angle, looks nice
                ax.set_xlabel("X coordinate")
                ax.set_ylabel("Y coordinate")
                ax.set_zlabel("Cost")

                img = self._figure_to_array(
                    fig,
                    scen_dir / "surface3d/frame.png" if self.frames else None
                )

                if self.frames:
                    frame_3d_path = scen_dir / f"frame/frame_3d_{t_idx:03d}.png"
                    frame_3d_path.parent.mkdir(parents=True, exist_ok=True)
                    imageio.imwrite(frame_3d_path, img)

                if self.gif_3d:
                    self.gif_buffers["gif_3d"].append(img)

            print(f"[t={t_idx}] Exported → {scen_dir}")


        # =====================================================
        # EXPORT GLOBAL GIFS
        # =====================================================
        gif_output_map = {
            "gif_2d": ("animation.gif", 2),
            "gif_3d": ("animation_3d.gif", 10),
            "gif_hist": ("animation_hist.gif", 2),
            "gif_heat": ("animation_heat.gif", 2),
            "gif_violin": ("animation_violin.gif", 2),
            "gif_kde": ("animation_kde.gif", 2),
        }

        for key, frames in self.gif_buffers.items():
            if not frames:
                continue
            filename, fps = gif_output_map[key]
            imageio.mimsave(base_dir / filename, frames, fps=fps, loop=0, palettsize=256, subrectangles=True)
            print(f"Saved → {filename}")

        if self.export_metadata:
            try:
                from Models.statistical_interpreter import StatisticalInterpreter

                interpreter = StatisticalInterpreter(
                    data_root=base_dir,
                    params={
                        "grid_size": self.grid_size,
                        "num_scenarios": self.num_scenarios,
                        "variance": self.variance,
                        "length_scale": self.len_scale,
                        "anis": self.anis,
                    },
                )
                interpreter.run()
                print(f"Saved → {interpreter.output_path.name}")
            except Exception as exc:
                print(f"[warn] Statistical interpretation export failed: {exc}")

        print("\n✓ STRF Generation Complete\n")


        return str(base_dir)




# ===============================================================
# ENTRYPOINT
# ===============================================================
def main():
    sp = STGRF()
    sp.JSONImport("/Users/philippstockerl/BachelorThesis/AutonomousVehicleRoutingDSS/config/default.json")
    sp.STRF()


if __name__ == "__main__":
    main()
