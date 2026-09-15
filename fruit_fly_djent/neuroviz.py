"""Real-time "neurons lighting up" viewer built on NAVis + octarine (NAVis' pygfx 3D backend).

* the skeletons come from the NAVis pipeline (:mod:`skeletons`: fetched with
  ``navis.interfaces.neuprint``, healed, Strahler-pruned, downsampled);
* all 2318 neurons are drawn as ONE pygfx line with a per-vertex *glow* scalar mapped through
  a colormap (blue at rest -> red -> yellow-white on a spike, fading back) — updating one
  float buffer per frame is what makes 60 fps with thousands of neurons possible;
* somas are drawn as points that swell and flash with the same glow; PAM/PPL1 dopaminergic
  somas carry a magenta halo (the Stage B reward injection site);
* the brain-region meshes from neuPrint form a translucent outline.

Colour convention (as in the brief): spike = bright yellow/red, fading back to dim blue.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, connectome, skeletons

UM_PER_VOXEL = skeletons.VOXEL_NM / 1000.0
GLOW_TAU_S = 0.18          # fade time constant
# additive blending: thousands of overlapping arbors, so the resting alpha must be tiny
REST = np.array([0.10, 0.18, 0.60, 0.006])
MID = np.array([0.95, 0.15, 0.08, 0.05])
HOT = np.array([1.00, 0.60, 0.12, 0.12])
PEAK = np.array([1.00, 1.00, 0.85, 0.25])


def glow_colormap(n: int = 256) -> np.ndarray:
    """(n, 4) RGBA float32 colormap: rest blue -> red -> orange -> yellow-white."""
    x = np.linspace(0, 1, n)[:, None]
    stops = [(0.0, REST), (0.45, MID), (0.75, HOT), (1.0, PEAK)]
    out = np.zeros((n, 4), np.float32)
    for (a, ca), (b, cb) in zip(stops[:-1], stops[1:]):
        m = (x[:, 0] >= a) & (x[:, 0] <= b)
        t = ((x[m] - a) / (b - a))
        out[m] = (1 - t) * ca + t * cb
    return out


class BrainScene:
    """Geometry for the viewer: merged line vertices, soma positions, ROI meshes (all in um)."""

    def __init__(self, neurons: pd.DataFrame | None = None, render_df: pd.DataFrame | None = None,
                 roi_dir: Path = skeletons.MESH_DIR, max_rois: int = 80):
        self.neurons = neurons if neurons is not None else connectome.load_subgraph()[0]
        df = render_df if render_df is not None else self._load_render_df()
        self.N = len(self.neurons)
        index = pd.Series(np.arange(self.N), index=self.neurons.bodyId.to_numpy())
        pos_parts, owner_parts = [], []
        soma = np.full((self.N, 3), np.nan, np.float32)
        for bid, nodes in df.groupby("bodyId", sort=False):
            if bid not in index.index:
                continue
            k = int(index[bid])
            xyz = nodes[["x", "y", "z"]].to_numpy(np.float32)
            nid = nodes["node_id"].to_numpy()
            pid = nodes["parent_id"].to_numpy()
            lookup = pd.Series(np.arange(len(nodes)), index=nid)
            child = pid >= 0
            pa = lookup.reindex(pid[child]).to_numpy()
            okm = ~np.isnan(pa)
            a = xyz[np.nonzero(child)[0][okm]]
            b = xyz[pa[okm].astype(int)]
            # segment list a->b with NaN breaks so one Line draws the whole neuron
            seg = np.empty((len(a) * 3, 3), np.float32)
            seg[0::3], seg[1::3], seg[2::3] = a, b, np.nan
            pos_parts.append(seg)
            owner_parts.append(np.full(len(seg), k, np.int32))
            root = xyz[pid < 0]
            soma[k] = root[0] if len(root) else xyz[0]
        pos = np.concatenate(pos_parts)
        self.owner = np.concatenate(owner_parts)
        # neuPrint soma location where available (voxels)
        sx = self.neurons[["soma_x", "soma_y", "soma_z"]].to_numpy(np.float32)
        have = ~np.isnan(sx).any(axis=1)
        soma[have] = sx[have]
        self.center_vox = np.nanmean(soma, axis=0)
        self.positions = self._to_um(pos)
        self.soma = self._to_um(soma)
        self.dan = connectome.dan_mask(self.neurons)
        self.rois = self._load_rois(roi_dir, max_rois)

    @staticmethod
    def _load_render_df() -> pd.DataFrame:
        p = skeletons.RENDER_PARQUET
        if p.exists():
            return pd.read_parquet(p)
        return skeletons.load_cache()

    def _to_um(self, xyz_vox: np.ndarray) -> np.ndarray:
        """Centre, scale to um, flip y so dorsal is up (neuPrint y grows ventrally)."""
        out = (xyz_vox - self.center_vox) * UM_PER_VOXEL
        out[..., 1] *= -1
        return out.astype(np.float32)

    def _load_rois(self, roi_dir: Path, max_rois: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
        out = []
        if not Path(roi_dir).is_dir():
            return out
        try:
            import trimesh
        except ImportError:
            return out
        for f in sorted(Path(roi_dir).glob("*.obj"))[:max_rois]:
            try:
                m = trimesh.load(f, force="mesh", process=False)
                v = self._to_um(np.asarray(m.vertices, np.float32))
                out.append((f.stem, v, np.asarray(m.faces, np.uint32)))
            except Exception:  # noqa: BLE001
                continue
        return out


class BrainViewer:
    def __init__(self, scene: BrainScene | None = None, size=(1400, 900), offscreen: bool = False,
                 title: str = "Fruit Fly Djent — MaleCNS live", show_rois: bool = True,
                 line_width: float = 1.0, max_fps: int = 60, zoom: float = 1.9):
        import octarine as oc
        import pygfx as gfx

        self.gfx = gfx
        self._zoom = zoom
        self.scene = scene or BrainScene()
        N = self.scene.N
        self.glow = np.zeros(N, np.float32)
        self.viewer = oc.Viewer(offscreen=offscreen, size=size, title=title, max_fps=max_fps, show=False)
        self.viewer.set_bgcolor((0.01, 0.01, 0.03))

        cmap = gfx.Texture(glow_colormap(), dim=1)
        self.tmap = gfx.TextureMap(cmap, filter="linear", wrap="clamp")
        soma_cm = glow_colormap()
        soma_cm[:, 3] = np.clip(soma_cm[:, 3] * 6.0 + 0.15, 0, 1)      # somas carry the flash
        self.tmap_soma = gfx.TextureMap(gfx.Texture(soma_cm, dim=1), filter="linear", wrap="clamp")

        # ---- neurons: one Line, per-vertex glow through the colormap
        owner = self.scene.owner
        self.vertex_glow = np.zeros(len(owner), np.float32)
        geom = gfx.Geometry(positions=self.scene.positions, texcoords=self.vertex_glow)
        mat = gfx.LineMaterial(thickness=line_width, color_mode="vertex_map", map=self.tmap, aa=True)
        mat.alpha_mode = "add"
        self.line = gfx.Line(geom, mat)
        self.viewer.add(self.line, name="neurons", center=False)

        # ---- somas
        self.soma_glow = np.zeros(N, np.float32)
        self.soma_size = np.full(N, 1.8, np.float32)
        pgeom = gfx.Geometry(positions=self.scene.soma, texcoords=self.soma_glow, sizes=self.soma_size)
        pmat = gfx.PointsMaterial(size_mode="vertex", size_space="world", color_mode="vertex_map",
                                  map=self.tmap_soma, aa=True)
        pmat.alpha_mode = "add"
        self.points = gfx.Points(pgeom, pmat)
        self.viewer.add(self.points, name="somas", center=False)

        # ---- dopaminergic halo (magenta), static
        dan_idx = np.nonzero(self.scene.dan)[0]
        if len(dan_idx):
            hgeom = gfx.Geometry(positions=self.scene.soma[dan_idx])
            hmat = gfx.PointsMaterial(size=7.0, size_space="world", color=(0.95, 0.25, 0.95, 0.35), aa=True)
            hmat.alpha_mode = "add"
            self.halo = gfx.Points(hgeom, hmat)
            self.viewer.add(self.halo, name="dopaminergic (PAM/PPL1)", center=False)

        # ---- region meshes
        if show_rois:
            for name, v, f in self.scene.rois:
                import trimesh
                self.viewer.add_mesh(trimesh.Trimesh(v, f, process=False), name=f"roi:{name}",
                                     color=(0.45, 0.55, 1.0), alpha=0.045, center=False)
        self.viewer.center_camera()
        self.set_view("front")
        self.viewer.camera.zoom = zoom
        if offscreen:
            self.viewer.show()          # registers the draw callback for canvas.draw()/screenshot
        self._frame_cb = None
        self._last = time.perf_counter()

    # ------------------------------------------------------------------ state
    def spike(self, ids: np.ndarray, amount: float = 1.0):
        if len(ids):
            self.glow[ids] = np.maximum(self.glow[ids], amount)

    def decay(self, dt: float, tau: float = GLOW_TAU_S):
        if dt > 0:
            self.glow *= np.exp(-dt / tau)

    def push(self):
        """Upload the glow to the GPU buffers."""
        np.take(self.glow, self.scene.owner, out=self.vertex_glow)
        self.line.geometry.texcoords.update_full()
        self.soma_glow[:] = self.glow
        self.points.geometry.texcoords.update_full()
        self.soma_size[:] = 1.8 + 6.0 * self.glow
        self.points.geometry.sizes.update_full()

    # ------------------------------------------------------------------ loop
    def run(self, frame_callback, start_loop: bool = True):
        """frame_callback(viewer, dt) is called every frame before the glow is uploaded."""
        def _anim():
            now = time.perf_counter()
            dt, self._last = now - self._last, now
            frame_callback(self, dt)
            self.push()
        self._last = time.perf_counter()
        self._anim = _anim
        self.viewer.add_animation(_anim, on_error="log")
        self.viewer.show(start_loop=start_loop)

    def stop(self):
        """Remove the animation, stop the event loop and close the window (call from the loop)."""
        try:
            self.viewer.remove_animation(self._anim)
        except Exception:  # noqa: BLE001
            pass
        try:
            from rendercanvas.auto import loop
            loop.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.viewer.canvas.close()
        except Exception:  # noqa: BLE001
            pass

    def screenshot(self, path: Path, size=None):
        self.push()
        self.viewer.screenshot(str(path), size=size, alpha=False)
        return path

    def set_view(self, view: str = "front"):
        """front = anterior view (dorsal up), top = dorsal view, side = lateral view."""
        dirs = {"front": ((0, 0, -1), (0, 1, 0)), "back": ((0, 0, 1), (0, 1, 0)),
                "top": ((0, -1, 0), (0, 0, -1)), "side": ((-1, 0, 0), (0, 1, 0))}
        d, up = dirs.get(view, dirs["front"])
        self.viewer.camera.show_object(self.viewer.scene, view_dir=d, up=up)
        self.viewer.camera.zoom = getattr(self, "_zoom", 1.9)


# ----------------------------------------------------------------------------------------------
# Raster replay helpers (used by the player and for offscreen checks)
# ----------------------------------------------------------------------------------------------
class RasterCursor:
    """Iterates a spike raster (t_ms, neuron) in time order."""

    def __init__(self, raster: pd.DataFrame):
        order = np.argsort(raster["t_ms"].to_numpy(), kind="stable")
        self.t = raster["t_ms"].to_numpy()[order] / 1000.0
        self.n = raster["neuron"].to_numpy()[order]
        self.i = 0

    def seek(self, t_s: float):
        self.i = int(np.searchsorted(self.t, t_s))

    def advance(self, t_s: float) -> np.ndarray:
        j = int(np.searchsorted(self.t, t_s, side="right"))
        ids, self.i = self.n[self.i:j], j
        return ids


def snapshot(t_s: float, out: Path, raster: pd.DataFrame | None = None, view: str = "front",
             size=(1600, 1000)) -> Path:
    """Offscreen render of the brain at song time t_s (glow integrated over the last 0.5 s)."""
    from .snn import load_raster
    raster = raster if raster is not None else load_raster()
    bv = BrainViewer(offscreen=True, size=size)
    bv.set_view(view)
    cur = RasterCursor(raster)
    cur.seek(max(0.0, t_s - 0.6))
    t = max(0.0, t_s - 0.6)
    while t < t_s:
        t2 = min(t_s, t + 1 / 60)
        bv.decay(t2 - t)
        bv.spike(cur.advance(t2))
        t = t2
    return bv.screenshot(out)


if __name__ == "__main__":
    import sys
    t_s = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    p = snapshot(t_s, config.OUTPUT_DIR / f"brain_{t_s:.1f}s.png")
    print("wrote", p)
