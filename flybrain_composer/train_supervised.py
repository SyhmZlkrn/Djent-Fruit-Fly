"""Stage A — supervised "tracing paper": ridge-regression readout on the driven reservoir.

    W_out = (Xᵀ X + λ I)⁻¹ Xᵀ Y        (features X = [reservoir state ; input ; 1])

Closed form, no backprop through the connectome. The reservoir is driven by the dual
4/4 + 25/16 pulse pair (+ form cues), its states are collected, and the readout is fitted to
the real transcription's piano roll. Then the same drive is replayed and the readout is
quantised to MIDI — the fly brain "plays" the song.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import config, connectome
from .model import MODEL_PATH, ComposerModel, targets_for
from .reservoir import features, sanity_report
from .sonify import decode_notes, diff_notes, notes_to_midi, transfer_articulation
from .transcription import Song, load_song


def ridge_fit(F: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """Solve (FᵀF + λI) W = FᵀY with a Cholesky factorisation (F is T x d, d ~ 2.3k)."""
    from scipy.linalg import cho_factor, cho_solve
    G = F.T @ F
    G[np.diag_indices_from(G)] += lam
    c = cho_factor(G, lower=True, check_finite=False)
    return cho_solve(c, F.T @ Y, check_finite=False)


def fit(song: Song | None = None, *, lam: float = 2e-4, spectral_radius: float = 1.6, seed: int = 0,
        input_frac: float = 0.3, save_to: Path = MODEL_PATH, verbose: bool = True) -> tuple[ComposerModel, dict]:
    song = song or load_song()
    neurons, edges = connectome.load_subgraph()
    t0 = time.time()
    model = ComposerModel.build(neurons, edges, song, spectral_radius=spectral_radius, seed=seed,
                                input_frac=input_frac)
    if verbose:
        print(f"[fit] reservoir: {model.W.shape[0]} neurons, {model.W.nnz} synaptic edges, "
              f"rho={spectral_radius}, {model.input_map.n_in} input channels", flush=True)
    X, U = model.drive(song, progress=verbose)
    if verbose:
        print(f"[fit] states {X.shape} in {time.time() - t0:.1f}s — {sanity_report(X)}", flush=True)
    Y, layout = targets_for(song)
    T = min(len(X), len(Y))
    F = features(X[:T], U[:T])
    model.W_out = ridge_fit(F, Y[:T], lam)
    model.layout, model.ridge_lambda = layout, lam
    Yhat = F @ model.W_out
    mse = float(np.mean((Yhat - Y[:T]) ** 2))
    notes = decode_notes(Yhat, layout)
    report = diff_notes(notes, song.notes, bpm=song.bpm)
    report.update({"mse": mse, "seconds": time.time() - t0})
    model.meta.update({"stageA": report})
    model.save(save_to)
    if verbose:
        print(f"[fit] ridge lambda={lam}: train MSE={mse:.4f}; decoded {report['n_pred']} notes vs "
              f"{report['n_truth']} truth — onset F1={report['f1']:.3f}, pitch acc={report['pitch_accuracy']:.3f}, "
              f"mean onset err={report['mean_onset_error_ms']:.1f} ms", flush=True)
        print(f"[fit] saved {save_to}", flush=True)
    return model, report


def compose(model: ComposerModel | None = None, song: Song | None = None, *, out_midi: Path | None = None,
            with_drums: bool = True, verbose: bool = True, extra: np.ndarray | None = None,
            dan_reward: float = 0.5):
    """Replay the drive through the reservoir + readout and write the MIDI performance.

    For a Stage B model the scalar reward (default 0.5 — "the fly has been rewarded") is injected
    into the PAM (+) / PPL1 (−) dopaminergic neurons with the learned ``dan_gain``.
    """
    song = song or load_song()
    model = model or ComposerModel.load()
    if extra is None and model.dan_gain:
        neurons, _ = connectome.load_subgraph()
        T = song.n_sixteenths * config.STEPS_PER_SIXTEENTH
        extra = np.zeros((T, model.W.shape[0]))
        extra[:, connectome.pam_mask(neurons)] = model.dan_gain * dan_reward
        extra[:, connectome.ppl1_mask(neurons)] = -model.dan_gain * dan_reward
    X, U = model.drive(song, progress=verbose, extra=extra)
    Yhat = model.readout(X, U)
    notes = transfer_articulation(decode_notes(Yhat, model.layout), song.notes)
    out_midi = Path(out_midi or (config.OUTPUT_DIR / "flybrain_rational_gaze.mid"))
    notes_to_midi(notes, out_midi, bpm=song.bpm, drums=song.drums if with_drums else None)
    report = diff_notes(notes, song.notes, bpm=song.bpm)
    if verbose:
        print(f"[compose] {len(notes)} notes -> {out_midi}  (onset F1 vs transcription {report['f1']:.3f}, "
              f"pitch acc {report['pitch_accuracy']:.3f})", flush=True)
    return notes, X, Yhat, report


if __name__ == "__main__":
    fit()
