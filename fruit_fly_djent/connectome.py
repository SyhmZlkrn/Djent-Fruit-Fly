"""neuPrint queries -> cached neuron table + signed sparse weight matrix.

The subgraph is song-independent and pulled once:
  * the top-N central-brain intrinsic neurons (``superclass == 'cb_intrinsic'``) by total
    synapse count, and
  * every PAM / PPL1 dopaminergic neuron (Stage B injects the reward signal there).

Everything is cached under ``data/cache`` as parquet so later stages never touch the network.

Sign convention (a simplification, *not* ground truth): the sign of an edge is taken from
the presynaptic neuron's consensus neurotransmitter. ACh -> excitatory, GABA -> inhibitory,
glutamate -> inhibitory (GluCl is the dominant glutamate receptor in the fly CNS, but real
sign depends on the postsynaptic receptor), monoamines -> weak excitatory (modulatory).
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import config

NEURON_COLUMNS = [
    "bodyId", "type", "instance", "superclass", "class", "somaSide", "rootSide",
    "pre", "post", "consensusNt", "predictedNt", "predictedNtConfidence",
    "soma_x", "soma_y", "soma_z",
]

# Presynaptic neurotransmitter -> synaptic sign. See module docstring for caveats.
NT_SIGN = {
    "acetylcholine": +1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "dopamine": +0.5,
    "octopamine": +0.5,
    "serotonin": +0.5,
    "histamine": -1.0,   # photoreceptor transmitter, inhibitory on its targets
    "unclear": +1.0,     # ACh is the majority class; treat unknowns as excitatory
    None: +1.0,
}

_client = None


def get_client():
    """Singleton neuPrint client for the configured dataset."""
    global _client
    if _client is None:
        from neuprint import Client
        _client = Client(config.NEUPRINT_SERVER, dataset=config.NEUPRINT_DATASET,
                         token=config.neuprint_token())
    return _client


def _neurons_path(cache_dir: Path) -> Path:
    return cache_dir / "neurons.parquet"


def _edges_path(cache_dir: Path) -> Path:
    return cache_dir / "edges.parquet"


def _fetch_neuron_table(client, n_top: int) -> pd.DataFrame:
    superclasses = list(config.SUBGRAPH_SUPERCLASSES)
    q = f"""
    MATCH (n:Neuron)
    WHERE n.superclass IN {superclasses!r} AND n.status = 'Traced'
    RETURN n.bodyId AS bodyId, n.type AS type, n.instance AS instance, n.superclass AS superclass,
           n.class AS class, n.somaSide AS somaSide, n.rootSide AS rootSide, n.pre AS pre, n.post AS post,
           n.consensusNt AS consensusNt, n.predictedNt AS predictedNt,
           n.predictedNtConfidence AS predictedNtConfidence,
           n.somaLocation.x AS soma_x, n.somaLocation.y AS soma_y, n.somaLocation.z AS soma_z
    ORDER BY n.pre + n.post DESC
    LIMIT {int(n_top)}
    """
    top = client.fetch_custom(q, dataset=config.NEUPRINT_DATASET)
    top["role"] = "hub"

    q_dan = f"""
    MATCH (n:Neuron)
    WHERE n.type =~ {config.DAN_TYPE_REGEX + '.*'!r}
    RETURN n.bodyId AS bodyId, n.type AS type, n.instance AS instance, n.superclass AS superclass,
           n.class AS class, n.somaSide AS somaSide, n.rootSide AS rootSide, n.pre AS pre, n.post AS post,
           n.consensusNt AS consensusNt, n.predictedNt AS predictedNt,
           n.predictedNtConfidence AS predictedNtConfidence,
           n.somaLocation.x AS soma_x, n.somaLocation.y AS soma_y, n.somaLocation.z AS soma_z
    """
    dans = client.fetch_custom(q_dan, dataset=config.NEUPRINT_DATASET)
    dans["role"] = "dan"

    neurons = pd.concat([top, dans], ignore_index=True)
    neurons = neurons.drop_duplicates("bodyId", keep="first").reset_index(drop=True)
    neurons.loc[neurons["type"].fillna("").str.match(config.DAN_TYPE_REGEX), "role"] = "dan"
    return neurons


def _fetch_edges(client, body_ids: np.ndarray, batch: int = 200) -> pd.DataFrame:
    """All ConnectsTo edges *within* the subgraph, batched over presynaptic bodies."""
    ids = [int(b) for b in body_ids]
    ids_literal = "[" + ",".join(map(str, ids)) + "]"
    frames = []
    for i in range(0, len(ids), batch):
        src = ids[i:i + batch]
        q = f"""
        MATCH (a:Neuron)-[c:ConnectsTo]->(b:Neuron)
        WHERE a.bodyId IN [{",".join(map(str, src))}] AND b.bodyId IN {ids_literal}
        RETURN a.bodyId AS pre, b.bodyId AS post, c.weight AS weight
        """
        frames.append(client.fetch_custom(q, dataset=config.NEUPRINT_DATASET))
        print(f"  edges: {i + len(src)}/{len(ids)} sources, {sum(len(f) for f in frames)} edges so far", flush=True)
    edges = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["pre", "post", "weight"])
    return edges.astype({"pre": "int64", "post": "int64", "weight": "int64"})


def pull_subgraph(n_top: int | None = None, cache_dir: Path = config.CACHE_DIR,
                  force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull (or load) the cached subgraph. Returns (neurons, edges)."""
    n_top = n_top or config.SUBGRAPH_N_TOP
    cache_dir = Path(cache_dir)
    if not force and _neurons_path(cache_dir).exists() and _edges_path(cache_dir).exists():
        return load_subgraph(cache_dir)

    client = get_client()
    t0 = time.time()
    print(f"[connectome] dataset={config.NEUPRINT_DATASET}: fetching top-{n_top} "
          f"{config.SUBGRAPH_SUPERCLASSES} neurons + all {config.DAN_TYPE_REGEX} DANs", flush=True)
    neurons = _fetch_neuron_table(client, n_top)
    print(f"[connectome] {len(neurons)} neurons ({(neurons.role == 'dan').sum()} dopaminergic) "
          f"in {time.time() - t0:.1f}s; fetching internal edges...", flush=True)
    edges = _fetch_edges(client, neurons["bodyId"].to_numpy())
    print(f"[connectome] {len(edges)} edges, {edges.weight.sum()} synapses, {time.time() - t0:.1f}s total",
          flush=True)

    cache_dir.mkdir(parents=True, exist_ok=True)
    neurons.to_parquet(_neurons_path(cache_dir), index=False)
    edges.to_parquet(_edges_path(cache_dir), index=False)
    return neurons, edges


def load_subgraph(cache_dir: Path = config.CACHE_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir = Path(cache_dir)
    if not _neurons_path(cache_dir).exists():
        raise FileNotFoundError(f"No cached subgraph in {cache_dir}. Run: python -m fruit_fly_djent.cli pull")
    return pd.read_parquet(_neurons_path(cache_dir)), pd.read_parquet(_edges_path(cache_dir))


def nt_sign_for(neurons: pd.DataFrame) -> np.ndarray:
    """Per-neuron output sign from the consensus (fallback: predicted) neurotransmitter."""
    nt = neurons["consensusNt"].where(neurons["consensusNt"].notna(), neurons["predictedNt"])
    return np.array([NT_SIGN.get(x if isinstance(x, str) else None, 1.0) for x in nt], dtype=np.float64)


def build_weight_matrix(neurons: pd.DataFrame, edges: pd.DataFrame, *, log_weights: bool = True,
                        spectral_radius: float | None = 0.95) -> sp.csr_matrix:
    """Signed sparse W (N x N, W[post, pre]) so that ``W @ x`` is the recurrent input.

    ``log_weights`` compresses synapse counts (1 + log(count)) — a few giant connections
    would otherwise dominate the dynamics. The matrix is then rescaled to the requested
    spectral radius (estimated with a few power iterations) — the standard echo-state
    conditioning step. The *wiring* is untouched; only a global gain is applied.
    """
    idx = pd.Series(np.arange(len(neurons)), index=neurons["bodyId"].to_numpy())
    e = edges[edges["pre"].isin(idx.index) & edges["post"].isin(idx.index)]
    rows = idx.loc[e["post"].to_numpy()].to_numpy()
    cols = idx.loc[e["pre"].to_numpy()].to_numpy()
    w = e["weight"].to_numpy().astype(np.float64)
    if log_weights:
        w = 1.0 + np.log(w)
    sign = nt_sign_for(neurons)
    w = w * sign[cols]
    n = len(neurons)
    W = sp.csr_matrix((w, (rows, cols)), shape=(n, n))
    W.sum_duplicates()
    if spectral_radius:
        rho = estimate_spectral_radius(W)
        if rho > 0:
            W = W * (spectral_radius / rho)
    return W.tocsr()


def estimate_spectral_radius(W: sp.spmatrix, n_iter: int = 60, seed: int = 0) -> float:
    """Largest |eigenvalue| via scipy's ARPACK, falling back to power iteration."""
    try:
        from scipy.sparse.linalg import eigs
        vals = eigs(W.astype(np.float64), k=1, which="LM", return_eigenvectors=False, maxiter=2000)
        return float(np.abs(vals[0]))
    except Exception:
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(W.shape[0])
        v /= np.linalg.norm(v)
        lam = 0.0
        for _ in range(n_iter):
            v2 = W @ v
            lam = np.linalg.norm(v2)
            if lam == 0:
                return 0.0
            v = v2 / lam
        return float(lam)


def dan_mask(neurons: pd.DataFrame) -> np.ndarray:
    """Boolean mask of PAM (reward) + PPL1 (punishment) dopaminergic neurons."""
    return neurons["type"].fillna("").str.match(config.DAN_TYPE_REGEX).to_numpy()


def pam_mask(neurons: pd.DataFrame) -> np.ndarray:
    return neurons["type"].fillna("").str.match(r"^PAM").to_numpy()


def ppl1_mask(neurons: pd.DataFrame) -> np.ndarray:
    return neurons["type"].fillna("").str.match(r"^PPL1").to_numpy()


def summary(neurons: pd.DataFrame, edges: pd.DataFrame) -> str:
    nt = neurons["consensusNt"].fillna("None").value_counts()
    lines = [
        f"neurons: {len(neurons)}  (dopaminergic PAM/PPL1: {int(dan_mask(neurons).sum())})",
        f"edges:   {len(edges)}  synapses: {int(edges['weight'].sum())}",
        f"density: {len(edges) / max(1, len(neurons) ** 2):.4f}",
        "neurotransmitters: " + ", ".join(f"{k}={v}" for k, v in nt.items()),
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # quick manual check
    n, e = pull_subgraph()
    print(summary(n, e))
