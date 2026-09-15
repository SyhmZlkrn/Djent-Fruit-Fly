"""Stage B — reward-modulated improvisation ("good-dog treats"), layered on top of Stage A.

Mechanism (perturb-and-reinforce, node-perturbation style, run with CMA-ES):

* the trainable parameters are a small vector θ:
    - log-gains of the three input streams (drums / riff / form),
    - the gain with which a scalar *reward* signal is injected into the real dopaminergic
      neurons of the subgraph — PAM types (appetitive) get +reward, PPL1 types (aversive) get
      −reward — the fly's own reinforcement pathway is the injection site,
    - coefficients of a low-rank (rank 8) perturbation of the Stage A readout in a fixed random
      subspace (the readout drifts, the connectome never changes);
* each episode: drive the reservoir over a 16-bar excerpt with the perturbed parameters, decode
  the notes, score them with the djent heuristics in :mod:`fitness`; the reward injected in the
  *next* episode is the fitness advantage of this one (fitness − running baseline) — reward
  modulates the dynamics, the ES moves θ toward whichever perturbations scored higher;
* the best θ is baked into ``model_stageB.npz`` (``compose --stage-b``, ``render``).

Purpose: not to re-fit the notes (Stage A did that) but to teach the system to deviate from the
memorised riff in ways that still groove — improvise instead of replaying a recording.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import config, connectome
from .fitness import score
from .meter import build_streams, stack_streams
from .model import MODEL_B_PATH, MODEL_PATH, ComposerModel
from .reservoir import Reservoir, features
from .sonify import decode_notes
from .transcription import Song, load_song

RANK = 8
HISTORY_PATH = config.CACHE_DIR / "stageB_history.json"


class Episode:
    """Runs the reservoir over a fixed excerpt with parameters θ."""

    def __init__(self, model: ComposerModel, song: Song, bars: int = 16, start_bar: int | None = None, seed: int = 0):
        self.model, self.song = model, song
        neurons, _ = connectome.load_subgraph()
        self.pam = np.nonzero(connectome.pam_mask(neurons))[0]
        self.ppl1 = np.nonzero(connectome.ppl1_mask(neurons))[0]
        sps = config.STEPS_PER_SIXTEENTH
        U = stack_streams(build_streams(song))
        # excerpt: the first 16 bars of the song's main riff unless told otherwise
        start_bar = 0 if start_bar is None else start_bar
        self.t0_16 = start_bar * 16
        self.t1_16 = min(song.n_sixteenths, self.t0_16 + bars * 16)
        s0, s1 = self.t0_16 * sps, self.t1_16 * sps
        self.U_all, self.s0, self.s1 = U, s0, s1
        # initial state: run Stage A up to the excerpt once (deterministic)
        res = model.reservoir()
        x0 = res.washout_state(U, 400)
        self.x0 = res.run(U[:s0], x0=x0, record=False) if s0 > 0 else x0
        rng = np.random.default_rng(seed + 100)
        n_feat = model.W_out.shape[0]
        n_tgt = model.W_out.shape[1]
        self.Ur = rng.standard_normal((n_feat, RANK)) / np.sqrt(n_feat)
        self.Vr = rng.standard_normal((RANK, n_tgt)) / np.sqrt(RANK)
        self.readout_scale = float(np.abs(model.W_out).mean()) * 40.0
        self.dim = 3 + 1 + RANK
        self.truth = song.notes

    def unpack(self, theta: np.ndarray) -> dict:
        return {
            "gains": {"drums": float(np.exp(0.6 * theta[0])), "riff": float(np.exp(0.6 * theta[1])),
                      "form": float(np.exp(0.6 * theta[2]))},
            "dan_gain": float(2.0 * theta[3]),
            "delta": (self.Ur * (theta[4:4 + RANK] * self.readout_scale)) @ self.Vr,
        }

    def run(self, theta: np.ndarray, reward: float) -> tuple[dict, list]:
        p = self.unpack(theta)
        base = self.model.input_map.gains
        imap = self.model.input_map.with_gains({k: base[k] * v for k, v in p["gains"].items()})
        res = Reservoir(self.model.W, imap.W_in, self.model.reservoir_params)
        U = self.U_all[self.s0:self.s1]
        extra = np.zeros((len(U), self.model.W.shape[0]), dtype=np.float64)
        extra[:, self.pam] = p["dan_gain"] * reward
        extra[:, self.ppl1] = -p["dan_gain"] * reward
        X = res.run(U, x0=self.x0, extra=extra)
        Yhat = features(X, U) @ (self.model.W_out + p["delta"])
        notes = decode_notes(Yhat, self.model.layout)
        for n in notes:
            n.start += self.t0_16
        sc = score(notes, self.truth, self.t0_16, self.t1_16, bpm=self.song.bpm)
        return sc, notes


def shape(song: Song | None = None, *, generations: int = 30, popsize: int = 8, bars: int = 16,
          seed: int = 0, sigma0: float = 0.35, verbose: bool = True) -> ComposerModel:
    import cma

    song = song or load_song()
    model = ComposerModel.load(MODEL_PATH)
    ep = Episode(model, song, bars=bars, seed=seed)
    baseline, _ = ep.run(np.zeros(ep.dim), 0.0)
    if verbose:
        print(f"[shape] excerpt bars {ep.t0_16 // 16}-{ep.t1_16 // 16}; Stage A replay fitness "
              f"{baseline['fitness']:.3f} (novelty {baseline['raw']['novelty']:.2f})", flush=True)
    es = cma.CMAEvolutionStrategy(np.zeros(ep.dim), sigma0, {"popsize": popsize, "seed": seed + 1, "verbose": -9})
    reward = 0.0
    running = baseline["fitness"]
    history, best = [], (baseline["fitness"], np.zeros(ep.dim), baseline)
    t0 = time.time()
    for g in range(generations):
        thetas = es.ask()
        fits, scores = [], []
        for th in thetas:
            sc, _ = ep.run(np.asarray(th), reward)
            fits.append(sc["fitness"])
            scores.append(sc)
            # reward = advantage over the running baseline (the "treat"), injected next episode
            reward = float(np.clip(sc["fitness"] - running, -1, 1))
            running += 0.1 * (sc["fitness"] - running)
        es.tell(thetas, [-f for f in fits])
        i = int(np.argmax(fits))
        if fits[i] > best[0]:
            best = (fits[i], np.asarray(thetas[i]).copy(), scores[i])
        history.append({"generation": g, "best": float(max(fits)), "mean": float(np.mean(fits)),
                        "best_ever": float(best[0]), "components": {k: float(v) for k, v in scores[i].items() if k not in ("fitness", "raw")}})
        if verbose:
            c = scores[i]
            print(f"[shape] gen {g + 1:3d}/{generations}: best {max(fits):.3f} mean {np.mean(fits):.3f} "
                  f"(ever {best[0]:.3f}) groove {c['groove']:.2f} novelty {c['raw']['novelty']:.2f} "
                  f"density {c['raw']['density']:.1f}/bar  [{time.time() - t0:.0f}s]", flush=True)
    # bake the best θ into a Stage B model
    p = ep.unpack(best[1])
    base = model.input_map.gains
    model.input_map = model.input_map.with_gains({k: base[k] * v for k, v in p["gains"].items()})
    model.readout_delta = p["delta"]
    model.dan_gain = p["dan_gain"]
    model.meta["stageB"] = {"fitness": best[0], "baseline": baseline["fitness"], "theta": best[1].tolist(),
                            "generations": generations, "popsize": popsize, "bars": bars,
                            "components": {k: float(v) for k, v in best[2].items() if k not in ("fitness", "raw")},
                            "raw": best[2]["raw"]}
    model.save(MODEL_B_PATH)
    HISTORY_PATH.write_text(json.dumps({"baseline": baseline, "history": history, "best": model.meta["stageB"]}, indent=1, default=float))
    if verbose:
        print(f"[shape] best fitness {best[0]:.3f} vs Stage A replay {baseline['fitness']:.3f}; "
              f"dan_gain {model.dan_gain:+.2f}; gains {model.input_map.gains} -> {MODEL_B_PATH}", flush=True)
    return model


if __name__ == "__main__":
    shape()
