"""The trained artefact: connectome reservoir + input map + readout, with save/load."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from . import config, connectome
from .meter import InputMap, build_streams, make_input_map, stack_streams
from .reservoir import Reservoir, ReservoirParams, features
from .transcription import Song, song_to_targets

MODEL_PATH = config.CACHE_DIR / "model_stageA.npz"
MODEL_B_PATH = config.CACHE_DIR / "model_stageB.npz"


@dataclass
class ComposerModel:
    W: sp.csr_matrix
    input_map: InputMap
    reservoir_params: ReservoirParams
    W_out: np.ndarray | None = None          # (n_features, n_targets)
    layout: dict = field(default_factory=dict)
    ridge_lambda: float = 1.0
    spectral_radius: float = 1.6
    dan_gain: float = 0.0                    # Stage B: reward drive into the dopaminergic neurons
    readout_delta: np.ndarray | None = None  # Stage B: low-rank perturbation of the readout
    meta: dict = field(default_factory=dict)
    expansion: dict | None = None            # multi-song read-out: {"R": (N, M) float32, "b": (M,), "gain": g} random
                                             # tanh features of the neuron states — more read-out capacity, same brain

    # ------------------------------------------------------------------ features
    def features(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Read-out features: [x ; u ; 1] and, when the model has an expansion, tanh(gain * x R + b)."""
        F = features(X, U)
        if self.expansion is not None:
            R, b, g = self.expansion["R"], self.expansion["b"], float(self.expansion["gain"])
            F = np.hstack([F, np.tanh(g * (np.asarray(X, dtype=np.float32) @ R) + b).astype(np.float64)])
        return F

    @staticmethod
    def make_expansion(n_neurons: int, m: int, gain: float = 3.0, seed: int = 7) -> dict:
        """Deterministic given (n, m, gain, seed) — only those four numbers are saved; the 55 MB matrix
        is rebuilt on load."""
        rng = np.random.default_rng(seed)
        return {"R": (rng.standard_normal((n_neurons, m)) / np.sqrt(n_neurons)).astype(np.float32),
                "b": rng.normal(0, 0.5, m).astype(np.float32), "gain": float(gain), "m": int(m), "seed": int(seed)}

    # ------------------------------------------------------------------ building
    @classmethod
    def build(cls, neurons, edges, song: Song, *, spectral_radius: float = 1.6,
              input_frac: float = 0.3, seed: int = 0, gains: dict | None = None,
              reservoir_params: ReservoirParams | None = None) -> "ComposerModel":
        W = connectome.build_weight_matrix(neurons, edges, spectral_radius=spectral_radius)
        streams = build_streams(song)
        dans = connectome.dan_mask(neurons)
        gains = {"drums": 4.0, "riff": 4.0, "form": 2.4, **(gains or {})}
        imap = make_input_map(len(neurons), streams, frac=input_frac, seed=seed, gains=gains, exclude=dans)
        rp = reservoir_params or ReservoirParams(seed=seed)
        return cls(W=W, input_map=imap, reservoir_params=rp, spectral_radius=spectral_radius,
                   meta={"n_neurons": len(neurons), "n_edges": len(edges), "seed": seed,
                         "dataset": config.NEUPRINT_DATASET, "song": song.title, "bpm": song.bpm})

    def reservoir(self) -> Reservoir:
        return Reservoir(self.W, self.input_map.W_in, self.reservoir_params)

    # ------------------------------------------------------------------ running
    def drive(self, song: Song, *, extra: np.ndarray | None = None, progress: bool = False,
              washout: int = 400) -> tuple[np.ndarray, np.ndarray]:
        """Run the reservoir over the whole song. Returns (X states, U inputs)."""
        U = stack_streams(build_streams(song))
        res = self.reservoir()
        x0 = res.washout_state(U, washout)
        X = res.run(U, x0=x0, extra=extra, progress=progress)
        return X, U

    def readout(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        if self.W_out is None:
            raise RuntimeError("model has no readout yet — run Stage A (fit) first")
        Wo = self.W_out if self.readout_delta is None else self.W_out + self.readout_delta
        return self.features(X, U) @ Wo

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path = MODEL_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            W_data=self.W.data, W_indices=self.W.indices, W_indptr=self.W.indptr, W_shape=np.array(self.W.shape),
            W_in=self.input_map.W_in,
            W_out=self.W_out if self.W_out is not None else np.zeros(0),
            readout_delta=self.readout_delta if self.readout_delta is not None else np.zeros(0),
            **{f"pop_{k}": v for k, v in self.input_map.populations.items()},
        )
        side = {
            "stream_slices": {k: [s.start, s.stop] for k, s in self.input_map.stream_slices.items()},
            "gains": self.input_map.gains,
            "reservoir_params": asdict(self.reservoir_params),
            "layout": {k: list(v) for k, v in self.layout.items()},
            "ridge_lambda": self.ridge_lambda,
            "spectral_radius": self.spectral_radius,
            "dan_gain": self.dan_gain,
            "expansion": ({"m": int(self.expansion["m"]), "gain": float(self.expansion["gain"]), "seed": int(self.expansion["seed"])}
                          if self.expansion is not None else None),
            "meta": self.meta,
        }
        path.with_suffix(".json").write_text(json.dumps(side, indent=2))
        return path

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "ComposerModel":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run: python -m fruit_fly_djent.cli fit")
        z = np.load(path)
        side = json.loads(path.with_suffix(".json").read_text())
        W = sp.csr_matrix((z["W_data"], z["W_indices"], z["W_indptr"]), shape=tuple(z["W_shape"]))
        slices = {k: slice(a, b) for k, (a, b) in side["stream_slices"].items()}
        pops = {k[4:]: z[k] for k in z.files if k.startswith("pop_")}
        imap = InputMap(z["W_in"], slices, pops, side["gains"])
        m = cls(W=W, input_map=imap, reservoir_params=ReservoirParams(**side["reservoir_params"]),
                W_out=z["W_out"] if z["W_out"].size else None,
                layout={k: tuple(v) for k, v in side["layout"].items()},
                ridge_lambda=side["ridge_lambda"], spectral_radius=side["spectral_radius"],
                dan_gain=side.get("dan_gain", 0.0),
                readout_delta=z["readout_delta"] if z["readout_delta"].size else None, meta=side["meta"])
        exp = side.get("expansion")
        if exp:
            m.expansion = cls.make_expansion(W.shape[0], int(exp["m"]), gain=float(exp["gain"]), seed=int(exp["seed"]))
        elif "exp_R" in z.files and z["exp_R"].size:              # older files that stored the matrix
            m.expansion = {"R": z["exp_R"], "b": z["exp_b"], "gain": float(side.get("expansion_gain") or 3.0),
                           "m": int(z["exp_R"].shape[1]), "seed": -1}
        return m


def targets_for(song: Song) -> tuple[np.ndarray, dict]:
    return song_to_targets(song)
