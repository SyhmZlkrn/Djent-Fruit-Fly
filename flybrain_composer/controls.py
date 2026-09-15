"""Control experiments — does the fly's *specific* wiring matter for Stage A?

    python -m flybrain_composer.cli controls

Refits the Stage A read-out on the same inputs and targets with the reservoir replaced by:
the inputs alone (no brain), the connectome with its synaptic partners shuffled, a random sparse
network of the same size, a 10x smaller random network, and the real wiring with the
excitatory/inhibitory signs removed. Results -> output/controls.json.

Honest reading of the numbers (this machine, 2026-09-14): the real wiring gives F1 0.990; a
shuffled or random network of the same size gives 0.995-0.996; the inputs alone give 0.000; a 230-unit
random network gives 0.10. So Stage A needs *a* large fixed nonlinear reservoir — the fly's wiring
is a perfectly good one, but not a special one. The connectome is the medium the song is written
on, not the composer. Say that in the video.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from . import config, connectome
from .model import ComposerModel
from .reservoir import ReservoirParams, features
from .sonify import decode_notes, diff_notes
from .transcription import Song, load_song
from .train_supervised import ridge_fit, targets_for

OUT_PATH = config.OUTPUT_DIR / "controls.json"


def _evaluate(model: ComposerModel, song: Song, Y, layout, lam: float, inputs_only: bool = False) -> dict:
    t = time.perf_counter()
    X, U = model.drive(song)
    T = min(len(X), len(Y))
    F = np.hstack([U[:T], np.ones((T, 1))]) if inputs_only else features(X[:T], U[:T])
    W = ridge_fit(F, Y[:T], lam)
    rep = diff_notes(decode_notes(F @ W, layout), song.notes, bpm=song.bpm)
    rep = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in rep.items()}
    rep["features"] = int(F.shape[1])
    rep["seconds"] = round(time.perf_counter() - t, 1)
    return rep


def run(song: Song | None = None, *, lam: float = 2e-4, seed: int = 0, verbose: bool = True,
        out_path: Path = OUT_PATH) -> dict:
    song = song or load_song()
    neurons, edges = connectome.load_subgraph()
    Y, layout = targets_for(song)
    real = ComposerModel.build(neurons, edges, song)
    rho = real.spectral_radius
    N, nnz = real.W.shape[0], real.W.nnz
    rng = np.random.default_rng(seed)

    def scaled(W):
        W = W.tocsr().astype(np.float64)
        return W * (rho / max(connectome.estimate_spectral_radius(W), 1e-9))

    def with_W(W, imap=None, params=None):
        return ComposerModel(W=scaled(W), input_map=imap or real.input_map,
                             reservoir_params=params or real.reservoir_params, spectral_radius=rho)

    coo = real.W.tocoo()
    shuffled = sp.coo_matrix((coo.data, (coo.row, rng.permutation(coo.col))), shape=(N, N)).tocsr()
    shuffled.sum_duplicates()
    random_same = sp.coo_matrix((rng.standard_normal(nnz), (rng.integers(0, N, nnz), rng.integers(0, N, nnz))),
                                shape=(N, N)).tocsr()
    random_same.sum_duplicates()
    n2 = N // 10
    nnz2 = int(nnz * (n2 / N) ** 2)
    random_small = sp.coo_matrix((rng.standard_normal(nnz2), (rng.integers(0, n2, nnz2), rng.integers(0, n2, nnz2))),
                                 shape=(n2, n2)).tocsr()
    random_small.sum_duplicates()
    from .meter import build_streams, make_input_map
    imap_small = make_input_map(n2, build_streams(song), frac=0.3, seed=seed, gains=real.input_map.gains)

    cases = [
        ("real", "REAL MaleCNS wiring", real, False),
        ("inputs_only", "inputs only (no brain)", real, True),
        ("shuffled", "SHUFFLED connectome (synaptic partners permuted)", with_W(shuffled), False),
        ("random_same_size", f"RANDOM reservoir, {N} units, same density", with_W(random_same), False),
        ("random_small", f"RANDOM reservoir, {n2} units", with_W(random_small, imap_small, ReservoirParams(seed=seed)), False),
        ("unsigned", "REAL wiring, excitatory/inhibitory signs removed", with_W(abs(real.W)), False),
    ]
    results = {}
    for key, label, model, inputs_only in cases:
        rep = _evaluate(model, song, Y, layout, lam, inputs_only=inputs_only)
        results[key] = {"label": label, **rep}
        if verbose:
            print(f"[controls] {label:52s} onset F1 {rep['f1']:.3f}  pitch acc {rep['pitch_accuracy']:.3f}  "
                  f"notes {rep['n_pred']}/{rep['n_truth']}  [{rep['seconds']}s]", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"song": song.title, "lambda": lam, "seed": seed, "n_neurons": N, "n_edges": nnz,
                                    "results": results}, indent=1))
    if verbose:
        print(f"[controls] -> {out_path}", flush=True)
    return results


if __name__ == "__main__":
    run()
