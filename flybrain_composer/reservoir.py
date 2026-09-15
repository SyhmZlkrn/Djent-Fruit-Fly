"""Leaky nonlinear dynamics over the connectome graph (the fixed reservoir).

    x[t+1] = (1 - a) * x[t] + a * tanh(W @ x[t] + W_in @ u[t] + b)

``W`` is the signed sparse connectome weight matrix from :mod:`connectome` (never modified),
``a`` a per-neuron leak rate (heterogeneous membrane time constants, log-uniform between
``leak_min`` and ``leak_max`` so the network spans time scales from a 32nd note to a full
25/16 riff cycle), ``b`` a small fixed random bias that breaks symmetry.

The reservoir is deterministic (no noise by default), so re-driving it with the same input
reproduces the training trajectory exactly — which is what lets the readout replay the song.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from . import config


@dataclass
class ReservoirParams:
    leak_min: float = 0.02
    leak_max: float = 0.8
    bias_scale: float = 1.0   # diverse operating points -> phase-tuned nonlinear basis functions
    noise: float = 0.0
    seed: int = 0


class Reservoir:
    def __init__(self, W: sp.csr_matrix, W_in: np.ndarray, params: ReservoirParams | None = None):
        self.W = W.tocsr().astype(np.float64)
        self.W_in = np.asarray(W_in, dtype=np.float64)
        self.N = W.shape[0]
        self.params = params or ReservoirParams()
        rng = np.random.default_rng(self.params.seed)
        self.leak = np.exp(rng.uniform(np.log(self.params.leak_min), np.log(self.params.leak_max), self.N))
        self.bias = rng.normal(0, self.params.bias_scale, self.N)

    def step(self, x: np.ndarray, u: np.ndarray, extra: np.ndarray | None = None,
             rng: np.random.Generator | None = None) -> np.ndarray:
        pre = self.W @ x + self.W_in @ u + self.bias
        if extra is not None:
            pre = pre + extra
        if self.params.noise > 0 and rng is not None:
            pre = pre + rng.normal(0, self.params.noise, self.N)
        return (1.0 - self.leak) * x + self.leak * np.tanh(pre)

    def run(self, U: np.ndarray, x0: np.ndarray | None = None, extra: np.ndarray | None = None,
            record: bool = True, progress: bool = False) -> np.ndarray:
        """Drive the reservoir with U (T, n_in). Returns X (T, N) float32 (or the final state)."""
        T = U.shape[0]
        x = np.zeros(self.N) if x0 is None else np.array(x0, dtype=np.float64)
        rng = np.random.default_rng(self.params.seed + 1) if self.params.noise > 0 else None
        X = np.empty((T, self.N), dtype=np.float32) if record else None
        it = range(T)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, desc="reservoir", unit="step")
        for t in it:
            x = self.step(x, U[t], None if extra is None else extra[t], rng)
            if record:
                X[t] = x
        return X if record else x

    def washout_state(self, U: np.ndarray, n: int = 400) -> np.ndarray:
        """Run the first n steps of U (repeated if shorter) to get a settled initial state."""
        Uw = U[:n] if len(U) >= n else np.tile(U, (int(np.ceil(n / len(U))), 1))[:n]
        return self.run(Uw, record=False)


def features(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Readout features [x ; u ; 1] as float64 for the ridge solve."""
    return np.hstack([X.astype(np.float64), U, np.ones((len(X), 1))])


def sanity_report(X: np.ndarray) -> str:
    """Quick numbers to confirm the activity is neither dead nor saturated."""
    a = np.abs(X)
    frac_sat = float((a > 0.95).mean())
    frac_dead = float((a < 1e-3).mean())
    var_over_time = X.var(axis=0)
    return (f"mean|x|={a.mean():.3f}  saturated={frac_sat:.1%}  dead={frac_dead:.1%}  "
            f"units with var>1e-4: {(var_over_time > 1e-4).mean():.1%}  "
            f"rank-ish (top-20 sv share): {_top_sv_share(X):.2f}")


def _top_sv_share(X: np.ndarray, k: int = 20, max_rows: int = 2000) -> float:
    Xs = X[:: max(1, len(X) // max_rows)].astype(np.float64)
    Xs = Xs - Xs.mean(axis=0)
    s = np.linalg.svd(Xs, compute_uv=False)
    return float((s[:k] ** 2).sum() / max(1e-12, (s ** 2).sum()))
