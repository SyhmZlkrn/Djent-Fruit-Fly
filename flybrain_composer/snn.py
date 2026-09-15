"""Spiking simulation of the same connectome wiring (for the "neurons lighting up" animation).

Leaky integrate-and-fire neurons with exponential current-based synapses:

    dv/dt   = (v_rest - v)/tau_m + (g_e - g_i + I_in(t) + I_bias)/ms      (unless refractory)
    dg_e/dt = -g_e/tau_e,    dg_i/dt = -g_i/tau_i
    spike when v >= v_th -> v = v_reset for t_ref;  on_pre: g_e/g_i += |w|

The recurrent weights are the signed log-compressed connectome matrix from :mod:`connectome`
(never modified, only a global gain), the input current is the *same* meter drive the rate
reservoir gets (bar/riff phase code + form cues) plus a note-onset "the fly hears itself"
stream, projected with the model's input map.

Two implementations with identical equations:

* :class:`Brian2Network` — builds the model in Brian2 (numpy code generation, no compiler
  needed) — the reference simulator, used offline to produce ``spikes.parquet``;
* :class:`NumpyLIF` — a vectorised exact-exponential Euler port, fast enough to run in
  lockstep with the audio clock, used by the real-time player.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import config, connectome
from .meter import build_streams, stack_streams, _pulse_train
from .model import ComposerModel
from .transcription import Note, Song

SPIKES_PATH = config.CACHE_DIR / "spikes.parquet"


@dataclass
class LIFParams:
    dt_ms: float = 1.0
    tau_m: float = 20.0      # ms
    tau_e: float = 5.0
    tau_i: float = 10.0
    v_rest: float = -65.0    # mV
    v_th: float = -50.0
    v_reset: float = -65.0
    t_ref: float = 2.0       # ms
    g_rec: float = 0.03      # mV per unit log-weight (1 + log synapse count)
    inh_ratio: float = 4.0   # inhibitory synapses are this much stronger per unit weight (balance)
    g_in: float = 0.6        # mV/ms per unit (std-normalised) input drive
    i_bias: float = 0.0      # mV/ms tonic drive
    noise: float = 0.2       # mV * sqrt(ms)
    seed: int = 0


def recurrent_weights(model: ComposerModel, p: LIFParams) -> tuple[sp.csc_matrix, sp.csc_matrix]:
    """Signed raw log-weights (1 + log synapse count, NT sign) in mV per spike: |exc|, |inh| (post x pre)."""
    neurons, edges = connectome.load_subgraph()
    W = connectome.build_weight_matrix(neurons, edges, spectral_radius=None).tocsr() * p.g_rec
    We = W.maximum(0).tocsc()
    Wi = ((-W).maximum(0) * p.inh_ratio).tocsc()
    We.eliminate_zeros()
    Wi.eliminate_zeros()
    return We, Wi


ONSET_GAIN = 2.5          # the "hears itself" stream relative to the meter drive


def onset_population(N: int, onset_frac: float = 0.10, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """The 'auditory' neurons that receive the note-onset stream, and their per-neuron gains."""
    rng = np.random.default_rng(seed)
    aud = rng.choice(N, int(onset_frac * N), replace=False)
    return aud, rng.uniform(0.5, 1.5, len(aud))


def onset_current(notes: list[Note] | None, T: int, N: int, aud: np.ndarray, amps: np.ndarray, p: LIFParams,
                  step0: int = 0) -> np.ndarray:
    """(T, N) current from note onsets at grid steps [step0, step0 + T) — zero outside `aud`."""
    out = np.zeros((T, N), dtype=np.float32)
    if notes:
        sps = config.STEPS_PER_SIXTEENTH
        pos = np.array([round(n.start * sps) - step0 for n in notes])
        pulses = _pulse_train(T, pos, width=2, amp=1.0)
        out[:, aud] = (p.g_in * ONSET_GAIN * pulses[:, None] * amps[None, :]).astype(np.float32)
    return out


def meter_current(model: ComposerModel, song: Song, p: LIFParams, U: np.ndarray | None = None) -> np.ndarray:
    """(T_grid, N) the meter drive alone: g_in * W_in u, unit std over the driven neurons.
    `U` overrides the song's default streams (the riff generator drives with shared section codes)."""
    U = stack_streams(build_streams(song)) if U is None else U
    I = U @ model.input_map.W_in.T
    driven = np.abs(I).sum(axis=0) > 0
    I = I / max(1e-9, I[:, driven].std())
    return (p.g_in * I).astype(np.float32)


def input_current(model: ComposerModel, song: Song, notes: list[Note] | None, p: LIFParams,
                  onset_frac: float = 0.10, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """(T_grid, N) input current on the reservoir grid = g_in * (W_in u + W_onset onset_pulses)."""
    I = meter_current(model, song, p)
    N = model.W.shape[0]
    aud, amps = onset_population(N, onset_frac, seed)
    I += onset_current(notes, len(I), N, aud, amps, p)
    return I, aud


class NumpyLIF:
    """Vectorised LIF network; step in chunks so it can run in lockstep with the audio clock."""

    def __init__(self, We: sp.csc_matrix, Wi: sp.csc_matrix, I_grid: np.ndarray, p: LIFParams,
                 grid_dt_ms: float = config.STEP_SECONDS * 1000.0):
        self.We, self.Wi, self.I_grid, self.p = We, Wi, I_grid, p
        self.N = We.shape[0]
        self.grid_dt_ms = grid_dt_ms
        self.rng = np.random.default_rng(p.seed)
        self.v = np.full(self.N, p.v_rest) + self.rng.uniform(0, 5, self.N)
        self.ge = np.zeros(self.N)
        self.gi = np.zeros(self.N)
        self.ref = np.zeros(self.N)
        self.t_ms = 0.0
        dt = p.dt_ms
        self.decay_m = np.exp(-dt / p.tau_m)
        self.decay_e = np.exp(-dt / p.tau_e)
        self.decay_i = np.exp(-dt / p.tau_i)
        self.spike_times: list[np.ndarray] = []
        self.spike_ids: list[np.ndarray] = []

    def step(self) -> np.ndarray:
        p, dt = self.p, self.p.dt_ms
        k = min(len(self.I_grid) - 1, int(self.t_ms / self.grid_dt_ms))
        I = self.I_grid[k]
        active = self.ref <= 0
        dv = (self.ge - self.gi + I + p.i_bias) * dt + p.noise * np.sqrt(dt) * self.rng.standard_normal(self.N)
        v_new = p.v_rest + (self.v - p.v_rest) * self.decay_m + dv
        self.v = np.where(active, v_new, self.v)
        self.ref = np.maximum(0.0, self.ref - dt)
        spk = np.nonzero(active & (self.v >= p.v_th))[0]
        if len(spk):
            self.v[spk] = p.v_reset
            self.ref[spk] = p.t_ref
        self.ge *= self.decay_e
        self.gi *= self.decay_i
        if len(spk):
            self.ge += np.asarray(self.We[:, spk].sum(axis=1)).ravel()
            self.gi += np.asarray(self.Wi[:, spk].sum(axis=1)).ravel()
        self.t_ms += dt
        return spk

    def run_until(self, t_ms: float, record: bool = True) -> np.ndarray:
        """Advance to t_ms; returns the neuron ids that spiked (concatenated) in the interval."""
        out = []
        while self.t_ms < t_ms:
            spk = self.step()
            if len(spk):
                out.append(spk)
                if record:
                    self.spike_ids.append(spk)
                    self.spike_times.append(np.full(len(spk), self.t_ms, dtype=np.float32))
        return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)

    def raster(self) -> pd.DataFrame:
        if not self.spike_ids:
            return pd.DataFrame({"t_ms": np.zeros(0, np.float32), "neuron": np.zeros(0, np.int32)})
        return pd.DataFrame({"t_ms": np.concatenate(self.spike_times).astype(np.float32),
                             "neuron": np.concatenate(self.spike_ids).astype(np.int32)})


class Brian2Network:
    """Same model built in Brian2 (reference implementation, offline)."""

    def __init__(self, We: sp.csc_matrix, Wi: sp.csc_matrix, I_grid: np.ndarray, p: LIFParams,
                 grid_dt_ms: float = config.STEP_SECONDS * 1000.0):
        import brian2 as b2
        b2.prefs.codegen.target = "numpy"       # no C++ compiler on this machine
        b2.defaultclock.dt = p.dt_ms * b2.ms
        self.b2, self.p = b2, p
        N = We.shape[0]
        stim = b2.TimedArray(I_grid.astype(np.float64) * b2.mV, dt=grid_dt_ms * b2.ms)
        ns = dict(tau_m=p.tau_m * b2.ms, tau_e=p.tau_e * b2.ms, tau_i=p.tau_i * b2.ms,
                  v_rest=p.v_rest * b2.mV, v_th=p.v_th * b2.mV, v_reset=p.v_reset * b2.mV,
                  i_bias=p.i_bias * b2.mV, sigma=p.noise * b2.mV, stim=stim)
        eqs = """
        dv/dt = (v_rest - v)/tau_m + (ge - gi + stim(t, i) + i_bias)/ms + sigma*xi/sqrt(ms) : volt (unless refractory)
        dge/dt = -ge/tau_e : volt
        dgi/dt = -gi/tau_i : volt
        """
        G = b2.NeuronGroup(N, eqs, threshold="v >= v_th", reset="v = v_reset", refractory=p.t_ref * b2.ms,
                           method="euler", namespace=ns, name="malecns")
        G.v = p.v_rest * b2.mV + np.random.default_rng(p.seed).uniform(0, 5, N) * b2.mV
        Se = b2.Synapses(G, G, "w : volt", on_pre="ge_post += w", name="exc")
        Si = b2.Synapses(G, G, "w : volt", on_pre="gi_post += w", name="inh")
        ce = We.tocoo(); ci = Wi.tocoo()
        Se.connect(i=ce.col, j=ce.row); Se.w = ce.data * b2.mV
        Si.connect(i=ci.col, j=ci.row); Si.w = ci.data * b2.mV
        self.mon = b2.SpikeMonitor(G)
        self.net = b2.Network(G, Se, Si, self.mon)

    def run(self, duration_s: float, report: bool = True) -> pd.DataFrame:
        b2 = self.b2
        self.net.run(duration_s * b2.second, report="text" if report else None, report_period=10 * b2.second)
        return pd.DataFrame({"t_ms": (self.mon.t / b2.ms).astype(np.float32),
                             "neuron": np.asarray(self.mon.i, dtype=np.int32)})


def build_inputs(model: ComposerModel, song: Song, notes: list[Note] | None = None,
                 p: LIFParams | None = None):
    p = p or LIFParams()
    We, Wi = recurrent_weights(model, p)
    I_grid, aud = input_current(model, song, notes, p)
    return We, Wi, I_grid, aud, p


def grid_dt_ms(song: Song) -> float:
    return config.step_seconds(song.bpm) * 1000.0


def simulate(song: Song, notes: list[Note] | None = None, *, backend: str = "brian2",
             duration_s: float | None = None, p: LIFParams | None = None,
             out_path: Path = SPIKES_PATH, verbose: bool = True) -> pd.DataFrame:
    """Run the whole-song spiking simulation and cache the raster (t_ms, neuron)."""
    model = ComposerModel.load()
    We, Wi, I_grid, aud, p = build_inputs(model, song, notes, p)
    duration_s = duration_s or song.seconds + 2.0
    t0 = time.time()
    gdt = grid_dt_ms(song)
    if backend == "brian2":
        try:
            net = Brian2Network(We, Wi, I_grid, p, grid_dt_ms=gdt)
            raster = net.run(duration_s, report=verbose)
        except Exception as e:  # noqa: BLE001
            print(f"[snn] brian2 failed ({e}); falling back to the numpy LIF", flush=True)
            backend = "numpy"
    if backend == "numpy":
        lif = NumpyLIF(We, Wi, I_grid, p, grid_dt_ms=gdt)
        it = range(int(duration_s))
        if verbose:
            from tqdm import tqdm
            it = tqdm(it, desc="LIF", unit="s")
        for s in it:
            lif.run_until((s + 1) * 1000.0)
        raster = lif.raster()
    raster.attrs["backend"] = backend
    if verbose:
        N = We.shape[0]
        rate = len(raster) / max(1e-9, duration_s) / N
        print(f"[snn] {backend}: {len(raster)} spikes, mean rate {rate:.1f} Hz/neuron, "
              f"{time.time() - t0:.0f}s", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raster.to_parquet(out_path, index=False)
    return raster


def load_raster(path: Path = SPIKES_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError("No spike raster. Run: python -m flybrain_composer.cli spikes")
    return pd.read_parquet(path)


def calibrate(model: ComposerModel, song: Song, notes, seconds: float = 6.0,
              target_hz: float = 8.0, verbose: bool = True) -> LIFParams:
    """Pick g_rec so the numpy LIF sits near the target mean rate (asynchronous-irregular regime)."""
    p = LIFParams()
    for g in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12):
        p.g_rec = g
        We, Wi, I_grid, _, _ = build_inputs(model, song, notes, p)
        lif = NumpyLIF(We, Wi, I_grid, p, grid_dt_ms=grid_dt_ms(song))
        lif.run_until(seconds * 1000.0)
        r = lif.raster()
        rate = len(r) / seconds / We.shape[0]
        active = r.neuron.nunique() / We.shape[0]
        if verbose:
            print(f"  g_rec={g}: {rate:.1f} Hz/neuron, {active:.0%} neurons active", flush=True)
        if rate >= target_hz:
            break
    return p
