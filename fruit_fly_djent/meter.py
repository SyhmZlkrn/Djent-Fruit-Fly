"""Input streams: the 4/4 drum pulse, the 25/16 riff pulse and the song-form cues.

The key song-specific design choice from the plan: the polymeter is kept as *two independent
periodic streams* that go into two separate input-neuron populations. A third, slow "form"
stream tells the reservoir which section of the song it is in (without it the reservoir state is
a pure function of the metric phase and the readout could only ever replay one riff loop).

Each stream carries short raised-cosine pulses plus a phase ramp (sawtooth) — a "where in the
bar / where in the riff cycle are we" signal, like the periodic drive the plan describes.

Streams are mapped onto the reservoir with a fixed random projection into disjoint neuron
populations (``InputMap``). The connectome itself is never modified.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .transcription import Song

RIFF_CYCLE_SIXTEENTHS = 25
# Number of harmonics of each metric phase (a Fourier "phase code" of where we are in the bar
# and in the riff cycle — oscillatory phase coding, the periodic drive the plan calls for).
PHASE_HARMONICS = 6


def _phase_code(phase: np.ndarray, harmonics: int = PHASE_HARMONICS) -> np.ndarray:
    """cos/sin of the phase at harmonics 1..H -> (T, 2H)."""
    cols = []
    for k in range(1, harmonics + 1):
        cols += [np.cos(2 * np.pi * k * phase), np.sin(2 * np.pi * k * phase)]
    return np.stack(cols, axis=1)


def _pulse_train(T: int, positions_steps: np.ndarray, width: int = 2, amp: float = 1.0) -> np.ndarray:
    """Raised-cosine pulses of `width` steps starting at each position."""
    out = np.zeros(T)
    kernel = 0.5 - 0.5 * np.cos(np.pi * (np.arange(width) + 1) / (width + 1)) if width > 1 else np.ones(1)
    kernel = kernel / kernel.max()
    for p in positions_steps.astype(int):
        if 0 <= p < T:
            n = min(width, T - p)
            out[p:p + n] = np.maximum(out[p:p + n], amp * kernel[:n])
    return out


def _phase_ramp(T: int, starts_steps: np.ndarray) -> np.ndarray:
    """Sawtooth going 0 -> 1 between consecutive cycle starts."""
    out = np.zeros(T)
    s = np.sort(starts_steps.astype(int))
    s = s[(s >= 0) & (s < T)]
    if len(s) == 0:
        return out
    bounds = list(s) + [T]
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b > a:
            out[a:b] = np.linspace(0, 1, b - a, endpoint=False)
    return out


def build_streams(song: Song, steps_per_sixteenth: int = config.STEPS_PER_SIXTEENTH,
                  n_steps: int | None = None) -> dict[str, np.ndarray]:
    """Return {'drums': (T, 4+2H), 'riff': (T, 3+2H), 'form': (T, n_sections + 1)} on the grid."""
    sps = steps_per_sixteenth
    T = n_steps or song.n_sixteenths * sps
    quarter = np.arange(0, T, 4 * sps)
    bar = np.arange(0, T, 16 * sps)
    snare = np.arange(8 * sps, T, 16 * sps)
    bar_phase = _phase_ramp(T, bar)
    drums = np.column_stack([
        _pulse_train(T, quarter, width=2, amp=0.7),
        _pulse_train(T, bar, width=3, amp=1.0),
        _pulse_train(T, snare, width=2, amp=1.0),
        bar_phase,
        _phase_code(bar_phase),
    ])

    sixteenth = np.arange(0, T, sps)
    if song.riff_starts:
        riff_starts = np.array([round(r * sps) for r in song.riff_starts])
    else:
        riff_starts = np.arange(0, T, RIFF_CYCLE_SIXTEENTHS * sps)
    riff_phase = _phase_ramp(T, riff_starts)
    riff = np.column_stack([
        _pulse_train(T, sixteenth, width=1, amp=0.4),
        _pulse_train(T, riff_starts, width=3, amp=1.0),
        riff_phase,
        _phase_code(riff_phase),
    ])

    # form: one-hot section identity (repeats of the same material share a column), a pulse at
    # every section start, and a phase code of the position *within* the section so that bar 3
    # and bar 7 of an 8-bar block do not look alike to the readout.
    sections = song.sections or []
    ids = {}
    for sec in sections:
        ids.setdefault(sec.name, len(ids))
    form = np.zeros((T, len(ids) + 1))
    sec_phase = np.zeros(T)
    for sec in sections:
        a, b = int(round(sec.start * sps)), int(round(sec.end * sps))
        form[a:b, ids[sec.name]] = 1.0
        form[:, -1] += _pulse_train(T, np.array([a]), width=4, amp=1.0)
        if b > a:
            sec_phase[a:b] = np.linspace(0, 1, b - a, endpoint=False)
    form = np.column_stack([form, sec_phase, _phase_code(sec_phase, harmonics=8)])
    return {"drums": drums, "riff": riff, "form": form}


@dataclass
class InputMap:
    """Fixed random projection of the streams into disjoint reservoir populations."""
    W_in: np.ndarray                 # (N, n_in)
    stream_slices: dict[str, slice]  # which input columns belong to which stream
    populations: dict[str, np.ndarray]   # which neurons each stream targets
    gains: dict[str, float]

    @property
    def n_in(self) -> int:
        return self.W_in.shape[1]

    def with_gains(self, gains: dict[str, float]) -> "InputMap":
        """Return a copy with rescaled per-stream gains (Stage B tunes these)."""
        W = self.W_in.copy()
        for name, sl in self.stream_slices.items():
            g_old, g_new = self.gains.get(name, 1.0), gains.get(name, self.gains.get(name, 1.0))
            if g_old != 0:
                W[:, sl] *= g_new / g_old
        return InputMap(W, self.stream_slices, self.populations, {**self.gains, **gains})


def make_input_map(n_neurons: int, streams: dict[str, np.ndarray], *, frac: float = 0.15,
                   gains: dict[str, float] | None = None, seed: int = 0,
                   exclude: np.ndarray | None = None) -> InputMap:
    """Disjoint random populations of size frac*N per stream; weights ~ U(-1, 1) * gain."""
    rng = np.random.default_rng(seed)
    gains = {"drums": 1.0, "riff": 1.0, "form": 0.6, **(gains or {})}
    candidates = np.arange(n_neurons)
    if exclude is not None:
        candidates = candidates[~exclude]
    rng.shuffle(candidates)
    n_pop = int(frac * n_neurons)
    n_in = sum(s.shape[1] for s in streams.values())
    W = np.zeros((n_neurons, n_in))
    slices, pops = {}, {}
    col = 0
    for k, (name, s) in enumerate(streams.items()):
        pop = candidates[k * n_pop:(k + 1) * n_pop]
        sl = slice(col, col + s.shape[1])
        W[np.ix_(pop, np.arange(col, col + s.shape[1]))] = rng.uniform(-1, 1, (len(pop), s.shape[1])) * gains[name]
        slices[name], pops[name] = sl, pop
        col += s.shape[1]
    return InputMap(W, slices, pops, gains)


def stack_streams(streams: dict[str, np.ndarray]) -> np.ndarray:
    return np.hstack([streams[k] for k in streams])


# ----------------------------------------------------------------------------------------------
# Variants of the streams: the musical knobs of the live improviser, and the shared "section code"
# form encoding that lets one read-out learn many songs (and interpolate between them).
# ----------------------------------------------------------------------------------------------
SECTION_CODE_DIM = 16


def section_code(key: str, dim: int = SECTION_CODE_DIM) -> np.ndarray:
    """Deterministic +-1/sqrt(dim) code for a (song, section) name — a distributed identity that
    has the same size for every song, unlike the per-song one-hot."""
    import hashlib
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], dim) / np.sqrt(dim)


def _segments(starts_steps: np.ndarray, T: int) -> list[tuple[int, int]]:
    s = np.sort(starts_steps.astype(int))
    s = s[(s >= 0) & (s < T)]
    bounds = list(s) + [T]
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def _warped_phase(T: int, segments: list[tuple[int, int]], stretch: float, disp_steps: float,
                  pulse_width: int) -> tuple[np.ndarray, np.ndarray]:
    """Sawtooth phase over the segments, run `stretch` times faster and displaced by `disp_steps`
    grid steps; returns (phase in [0,1), pulse train at every phase wrap). At stretch 1 / disp 0 this
    is exactly the original ramp with a pulse at every segment start."""
    phase = np.zeros(T)
    pulses = np.zeros(T)
    for a, b in segments:
        L = b - a
        d = disp_steps / L                                   # displacement in phase units of this cycle
        raw = stretch * np.arange(L) / L + d
        phase[a:b] = raw - np.floor(raw)
        m_lo, m_hi = int(np.ceil(raw[0])), int(np.floor(raw[-1]))
        wraps = [a + int(round((m - d) / stretch * L)) for m in range(m_lo, m_hi + 1)]
        wraps = [w for w in wraps if a <= w < b]
        if wraps:
            pulses = np.maximum(pulses, _pulse_train(T, np.array(wraps), width=pulse_width, amp=1.0))
    return phase, pulses


def build_streams_ex(song: Song, steps_per_sixteenth: int = config.STEPS_PER_SIXTEENTH,
                     n_steps: int | None = None, *, disp16: float = 0.0, stretch: float = 1.0,
                     blend: tuple[str | np.ndarray, float] | None = None, form_mode: str = "onehot",
                     code_dim: int = SECTION_CODE_DIM, song_key: str = "",
                     section_ids: dict[str, int] | None = None,
                     section_vectors: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """`build_streams` with knobs. Defaults reproduce it exactly.

    disp16    — displace the riff cycle and the section material by this many sixteenths against
                the (fixed) drum grid: polymetric displacement;
    stretch   — run the riff/section phase this much faster (>1) or slower (<1): a new polymeter;
    blend     — (other section name | code vector, alpha): mix another section's identity into the
                form stream, so the read-out plays a mixture of the two materials;
    form_mode — "onehot" (per-song, as Stage A) or "code" (shared SECTION_CODE_DIM-dim codes keyed by
                song_key + section name, for the multi-song read-out and the riff generator).
    """
    sps = steps_per_sixteenth
    T = n_steps or song.n_sixteenths * sps
    quarter = np.arange(0, T, 4 * sps)
    bar = np.arange(0, T, 16 * sps)
    snare = np.arange(8 * sps, T, 16 * sps)
    bar_phase = _phase_ramp(T, bar)
    drums = np.column_stack([
        _pulse_train(T, quarter, width=2, amp=0.7),
        _pulse_train(T, bar, width=3, amp=1.0),
        _pulse_train(T, snare, width=2, amp=1.0),
        bar_phase,
        _phase_code(bar_phase),
    ])

    sixteenth = np.arange(0, T, sps)
    if song.riff_starts:
        riff_starts = np.array([round(r * sps) for r in song.riff_starts])
    else:
        riff_starts = np.arange(0, T, RIFF_CYCLE_SIXTEENTHS * sps)
    riff_phase, riff_pulse = _warped_phase(T, _segments(riff_starts, T), stretch, disp16 * sps, pulse_width=3)
    riff = np.column_stack([
        _pulse_train(T, sixteenth, width=1, amp=0.4),
        riff_pulse,
        riff_phase,
        _phase_code(riff_phase),
    ])

    sections = song.sections or []
    if form_mode == "onehot":
        ids = dict(section_ids or {})
        for sec in sections:
            ids.setdefault(sec.name, len(ids))
        n_id = len(ids)
        vec = {name: np.eye(n_id)[i] for name, i in ids.items()}
    else:
        names = sorted({sec.name for sec in sections})
        vec = {name: section_code(f"{song_key}|{name}", code_dim) for name in names}
        if section_vectors:                       # explicit codes (the riff generator interpolates learned ones)
            vec.update({k: np.asarray(v, dtype=float) for k, v in section_vectors.items()})
        n_id = code_dim
    ident = np.zeros((T, n_id))
    other = None
    if blend is not None:
        ref, alpha = blend
        other = (vec.get(ref) if isinstance(ref, str) else np.asarray(ref, dtype=float))
        if other is not None and len(other) != n_id:
            other = None
    seg_list = []
    for sec in sections:
        a, b = int(round(sec.start * sps)), int(round(sec.end * sps))
        a, b = max(0, a), min(T, b)
        if b <= a:
            continue
        v = vec[sec.name]
        if other is not None:
            v = (1.0 - blend[1]) * v + blend[1] * other
        ident[a:b] = v
        seg_list.append((a, b))
    sec_phase, sec_pulse = _warped_phase(T, seg_list, stretch, disp16 * sps, pulse_width=4)
    form = np.column_stack([ident, sec_pulse, sec_phase, _phase_code(sec_phase, harmonics=8)])
    return {"drums": drums, "riff": riff, "form": form}
