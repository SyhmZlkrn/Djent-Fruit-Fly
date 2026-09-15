"""Fetch + cache the 3D skeletons (and brain-region meshes) of the subgraph neurons with NAVis.

Skeletons are pulled through ``navis.interfaces.neuprint.fetch_skeletons`` (threaded), healed,
downsampled for interactive rendering and stored in a single parquet
(``data/skeletons/skeletons.parquet``: bodyId, node_id, parent_id, x, y, z, radius — raw
neuPrint voxel coordinates, 8 nm/voxel). Region meshes (central complex, mushroom bodies,
antennal lobes, ...) are saved as OBJ files for the translucent brain outline.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, connectome

VOXEL_NM = 8.0
SKEL_PARQUET = config.SKELETON_DIR / "skeletons.parquet"
MESH_DIR = config.SKELETON_DIR / "rois"

# Region meshes worth drawing as context (superLevelRois in male-cns:v1.0).
CONTEXT_ROIS = [
    "EB", "FB", "PB", "NO",                     # central complex
    "CA(L)", "CA(R)", "PED(L)", "PED(R)",       # mushroom body calyx + pedunculus
    "aL(L)", "aL(R)", "bL(L)", "bL(R)", "gL(L)", "gL(R)", "a'L(L)", "a'L(R)", "b'L(L)", "b'L(R)",
    "AL(L)", "AL(R)", "LH(L)", "LH(R)", "AOTU(L)", "AOTU(R)", "BU(L)", "BU(R)", "LAL(L)", "LAL(R)",
    "SLP(L)", "SLP(R)", "SMP(L)", "SMP(R)", "SIP(L)", "SIP(R)", "CRE(L)", "CRE(R)",
    "AVLP(L)", "AVLP(R)", "PVLP(L)", "PVLP(R)", "PLP(L)", "PLP(R)", "GNG", "ME(L)", "ME(R)",
    "LO(L)", "LO(R)", "LOP(L)", "LOP(R)", "AME(L)", "AME(R)", "IB", "ATL(L)", "ATL(R)",
    "VES(L)", "VES(R)", "GOR(L)", "GOR(R)", "EPA(L)", "EPA(R)", "SPS(L)", "SPS(R)", "IPS(L)", "IPS(R)",
    "FLA(L)", "FLA(R)", "PRW", "SAD", "CAN(L)", "CAN(R)", "ICL(L)", "ICL(R)", "SCL(L)", "SCL(R)",
    "WED(L)", "WED(R)", "AB(L)", "AB(R)",
]


CHUNK_DIR = config.SKELETON_DIR / "chunks"


def fetch_skeletons(body_ids, *, downsample: int = 8, max_threads: int = 8, chunk: int = 100,
                    chunk_dir: Path | None = CHUNK_DIR) -> pd.DataFrame:
    """Fetch skeletons with navis, heal + downsample, return the concatenated node table.

    Every chunk is written to ``chunk_dir`` as soon as it is done, so an interrupted fetch
    resumes where it stopped.
    """
    import navis
    import navis.interfaces.neuprint as neu

    client = connectome.get_client()
    frames = []
    ids = [int(b) for b in body_ids]
    if chunk_dir is not None:
        chunk_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i in range(0, len(ids), chunk):
        batch = ids[i:i + chunk]
        cpath = chunk_dir / f"chunk_{batch[0]}_{batch[-1]}.parquet" if chunk_dir is not None else None
        if cpath is not None and cpath.exists():
            frames.append(pd.read_parquet(cpath))
            continue
        try:
            nl = neu.fetch_skeletons(batch, client=client, with_synapses=False, heal=True,
                                     parallel=True, max_threads=max_threads, missing_swc="raise")
        except Exception as e:  # noqa: BLE001 — retry sequentially, skipping the ones that fail
            print(f"  batch {i} failed ({e}); retrying one by one", flush=True)
            nl = navis.NeuronList([])
            for b in batch:
                try:
                    nl += neu.fetch_skeletons(b, client=client, with_synapses=False, heal=True)
                except Exception as e2:  # noqa: BLE001
                    print(f"    skip {b}: {e2}", flush=True)
        if downsample and downsample > 1:
            nl = navis.downsample_neuron(nl, downsampling_factor=downsample, inplace=False)
        part = []
        for n in nl:
            nodes = n.nodes[["node_id", "parent_id", "x", "y", "z", "radius"]].copy()
            nodes.insert(0, "bodyId", int(n.id))
            part.append(nodes)
        if part:
            pdf = pd.concat(part, ignore_index=True).astype(
                {"bodyId": "int64", "node_id": "int64", "parent_id": "int64",
                 "x": "float32", "y": "float32", "z": "float32", "radius": "float32"})
            if cpath is not None:
                pdf.to_parquet(cpath, index=False)
            frames.append(pdf)
        done = min(i + chunk, len(ids))
        print(f"  skeletons: {done}/{len(ids)}  ({time.time() - t0:.0f}s)", flush=True)
    df = pd.concat(frames, ignore_index=True)
    return df


def fetch_roi_meshes(rois=CONTEXT_ROIS, out_dir: Path = MESH_DIR) -> list[str]:
    """Download ROI meshes as OBJ (skips the ones the server does not have)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    client = connectome.get_client()
    ok = []
    for roi in rois:
        fname = out_dir / (roi.replace("(", "_").replace(")", "").replace("'", "p") + ".obj")
        if fname.exists():
            ok.append(roi)
            continue
        try:
            data = client.fetch_roi_mesh(roi, export_path=str(fname))
            if data is None and not fname.exists():
                raise RuntimeError("empty")
            ok.append(roi)
        except Exception as e:  # noqa: BLE001
            print(f"  no mesh for {roi}: {str(e)[:80]}", flush=True)
    return ok


def build_cache(force: bool = False) -> pd.DataFrame:
    neurons, _ = connectome.load_subgraph()
    if SKEL_PARQUET.exists() and not force:
        df = pd.read_parquet(SKEL_PARQUET)
        have = set(df.bodyId.unique())
        missing = [b for b in neurons.bodyId if b not in have]
        if not missing:
            return df
        print(f"[skeletons] {len(missing)} skeletons missing from cache; fetching", flush=True)
        df = pd.concat([df, fetch_skeletons(missing)], ignore_index=True)
    else:
        print(f"[skeletons] fetching {len(neurons)} skeletons from neuPrint", flush=True)
        df = fetch_skeletons(neurons.bodyId.to_numpy())
    config.SKELETON_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SKEL_PARQUET, index=False)
    print(f"[skeletons] cached {df.bodyId.nunique()} skeletons, {len(df)} nodes -> {SKEL_PARQUET}", flush=True)
    print("[skeletons] fetching ROI meshes", flush=True)
    ok = fetch_roi_meshes()
    print(f"[skeletons] {len(ok)} ROI meshes in {MESH_DIR}", flush=True)
    return df


def load_cache() -> pd.DataFrame:
    if not SKEL_PARQUET.exists():
        raise FileNotFoundError("No skeleton cache. Run: python -m flybrain_composer.cli skeletons")
    return pd.read_parquet(SKEL_PARQUET)


def to_navis(df: pd.DataFrame | None = None, body_ids=None):
    """Rebuild a navis.NeuronList from the cached node table (coordinates in nm)."""
    import navis
    df = load_cache() if df is None else df
    if body_ids is not None:
        df = df[df.bodyId.isin(set(int(b) for b in body_ids))]
    neurons = []
    for bid, nodes in df.groupby("bodyId", sort=False):
        nd = nodes[["node_id", "parent_id", "x", "y", "z", "radius"]].copy()
        nd[["x", "y", "z", "radius"]] = nd[["x", "y", "z", "radius"]].astype(float) * VOXEL_NM
        n = navis.TreeNeuron(nd, id=int(bid), units="nm")
        neurons.append(n)
    return navis.NeuronList(neurons)


def segments_um(df: pd.DataFrame) -> dict[int, np.ndarray]:
    """Per-neuron (M, 2, 3) float32 line-segment arrays in micrometres (for the web stage)."""
    out = {}
    scale = VOXEL_NM / 1000.0
    for bid, nodes in df.groupby("bodyId", sort=False):
        pos = nodes.set_index("node_id")[["x", "y", "z"]]
        child = nodes[nodes.parent_id >= 0]
        a = pos.loc[child.node_id].to_numpy(dtype=np.float32)
        b = pos.reindex(child.parent_id).to_numpy(dtype=np.float32)
        okmask = ~np.isnan(b).any(axis=1)
        seg = np.stack([a[okmask], b[okmask]], axis=1) * scale
        out[int(bid)] = seg
    return out


if __name__ == "__main__":
    build_cache()


RENDER_PARQUET = config.SKELETON_DIR / "skeletons_render.parquet"


def prepare_render_cache(max_nodes: int = 350, force: bool = False) -> pd.DataFrame:
    """Second-level downsample (topology-preserving, via navis) to <= max_nodes per neuron.

    The full cache keeps ~1.5k nodes per neuron; the live viewer only needs enough to read the
    shape of each cell, and 2318 x 350 vertices keeps the per-frame colour update cheap.
    """
    import navis
    if RENDER_PARQUET.exists() and not force:
        return pd.read_parquet(RENDER_PARQUET)
    df = load_cache()
    frames = []
    t0 = time.time()
    groups = list(df.groupby("bodyId", sort=False))
    for i, (bid, nodes) in enumerate(groups):
        nd = nodes[["node_id", "parent_id", "x", "y", "z", "radius"]].copy()
        if len(nd) > max_nodes:
            # EM auto-skeletons are bushy (half the nodes are tiny terminal twigs), so plain
            # downsampling cannot shrink them: drop Strahler-order-1 twigs (the backbone stays),
            # then downsample the remaining linear stretches.
            n = navis.TreeNeuron(nd, id=int(bid), units="8 nm")
            try:
                pruned = navis.prune_by_strahler(n, to_prune=[1], inplace=False)
                if pruned.n_nodes < 30:
                    pruned = navis.prune_twigs(n, size=625, inplace=False)   # 5 um
            except Exception:  # noqa: BLE001
                pruned = n
            factor = max(2, int(np.ceil(pruned.n_nodes / max_nodes)))
            n = navis.downsample_neuron(pruned, downsampling_factor=factor, inplace=False)
            nd = n.nodes[["node_id", "parent_id", "x", "y", "z", "radius"]].copy()
        nd.insert(0, "bodyId", int(bid))
        frames.append(nd)
        if (i + 1) % 200 == 0:
            print(f"  render cache: {i + 1}/{len(groups)} ({time.time() - t0:.0f}s)", flush=True)
    out = pd.concat(frames, ignore_index=True).astype(
        {"bodyId": "int64", "node_id": "int64", "parent_id": "int64",
         "x": "float32", "y": "float32", "z": "float32", "radius": "float32"})
    out.to_parquet(RENDER_PARQUET, index=False)
    print(f"[skeletons] render cache: {out.bodyId.nunique()} neurons, {len(out)} nodes -> {RENDER_PARQUET}", flush=True)
    return out
