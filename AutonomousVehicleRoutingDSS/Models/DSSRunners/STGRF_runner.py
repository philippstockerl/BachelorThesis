from pathlib import Path
from Models.STGRF import STGRF

def run_srf_generator(cfg):
    srf_cfg = cfg["srf"]
    viz     = cfg["visualizations"]
    graph   = cfg.setdefault("graph", {})
    paths   = cfg["paths"]
    graph["connectivity"] = "4"

    sp = STGRF(
        seed              = cfg["seed"],
        data_root         = paths["data_root"],
        grid_size         = srf_cfg["grid_size"],
        cell_size         = srf_cfg["cell_size"],
        kernel            = srf_cfg["kernel"],
        variance          = srf_cfg["variance"],
        len_scale         = srf_cfg["length_scale"],
        anis              = srf_cfg["anis"],
        nu                = srf_cfg.get("nu", 1.5),
        alpha             = srf_cfg.get("alpha", 1.0),
        use_normalization = srf_cfg["use_global_normalization"],
        num_scenarios     = srf_cfg["num_scenarios"],
        gif_2d            = viz["gif_2d"],
        gif_3d            = viz["gif_3d"],
        gif_hist          = viz["gif_hist"],
        gif_heat          = viz["gif_heat"],
        gif_kde           = viz["gif_kde"],
        gif_violin        = viz["gif_violin"],
        frames            = viz["frames"],
        frame_2d_legend   = viz.get("frame_2d_legend", False),
    )

    sp.DSSImport(cfg)

    # STGRF.STRF() returns *string*, not dict
    data_root = sp.STRF()

    # Runner must return ONLY this:
    return {"data_root": str(data_root)}
