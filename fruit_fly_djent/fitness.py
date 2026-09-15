"""Stage B fitness: "does this still sound like djent?" heuristics on a decoded note list.

Every component is squashed to [0, 1]; the total is a weighted sum. The scores are computed on
an excerpt (a few bars) so an evaluation costs a fraction of a second.

* groove       — autocorrelation of the onset train at the riff-cycle lag (25 sixteenths, the
                 polymeter) and at the bar lag (16): the deviation must still cycle;
* low_focus    — fraction of notes on the lowest string (pitch <= open F1 + 6): djent lives low;
* chug_ratio   — fraction of short (<= 1 sixteenth) notes: palm-muted chugs vs sustains, target 0.45-0.85;
* density      — notes per bar, target 6-12 (the transcription averages ~8.3);
* syncopation  — share of onsets on off-beat sixteenths, target 0.3-0.65;
* vocab        — 2-4 distinct pitches (the riff is two pitches a minor ninth apart);
* novelty      — 1 - onset F1 against the memorised transcription: we *want* deviation, but
                 only in a band (0.12-0.5); replaying the riff scores 0, wandering off scores 0.
"""
from __future__ import annotations

import numpy as np

from . import config
from .sonify import diff_notes
from .transcription import Note

RIFF_LAG = 25
BAR_LAG = 16
DEFAULT_WEIGHTS = {"groove": 2.0, "low_focus": 1.0, "chug_ratio": 1.0, "density": 1.0, "syncopation": 0.8,
                   "vocab": 0.6, "novelty": 2.0, "balance": 1.5}


def _band(x: float, lo: float, hi: float, soft: float) -> float:
    """1 inside [lo, hi], falling off linearly to 0 over `soft` outside."""
    if lo <= x <= hi:
        return 1.0
    d = lo - x if x < lo else x - hi
    return float(max(0.0, 1.0 - d / soft))


def onset_train(notes: list[Note], t0: float, t1: float) -> np.ndarray:
    n = int(np.ceil(t1 - t0))
    tr = np.zeros(max(1, n))
    for x in notes:
        k = int(round(x.start - t0))
        if 0 <= k < n:
            tr[k] = 1.0
    return tr


def autocorr_at(tr: np.ndarray, lag: int) -> float:
    if len(tr) <= lag or tr.std() < 1e-9:
        return 0.0
    a, b = tr[:-lag], tr[lag:]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.clip(np.corrcoef(a, b)[0, 1], 0, 1))


def score(notes: list[Note], truth: list[Note], t0: float, t1: float, bpm: float = config.BPM,
          low_pitch_max: int = config.TUNING_F_STANDARD[0] + 6, *, density_ref: float | None = None,
          novelty_band: tuple[float, float, float] | None = None, pitch_balance: bool = False,
          weights: dict[str, float] | None = None, riff_lag: int = RIFF_LAG) -> dict:
    """``density_ref`` (notes per bar of the transcription in this window) switches the density
    component to a band *relative* to the tab (0.75-1.25x); ``novelty_band`` = (lo, hi, soft)
    overrides the default 0.12-0.5 band. Both are used by the live improviser; the defaults keep
    Stage B's offline behaviour unchanged."""
    ex = [n for n in notes if t0 <= n.start < t1]
    tr_truth = [n for n in truth if t0 <= n.start < t1]
    bars = max(1e-9, (t1 - t0) / 16.0)
    tr = onset_train(ex, t0, t1)
    groove = 0.6 * autocorr_at(tr, riff_lag) + 0.4 * autocorr_at(tr, BAR_LAG)
    low_focus = float(np.mean([n.pitch <= low_pitch_max for n in ex])) if ex else 0.0
    chug = float(np.mean([n.duration <= 1.0 for n in ex])) if ex else 0.0
    density = len(ex) / bars
    sync = float(np.mean([(round(n.start) % 4) in (1, 3) for n in ex])) if ex else 0.0
    vocab = len(set(n.pitch for n in ex))
    novelty = 1.0 - diff_notes(ex, tr_truth, bpm=bpm)["f1"] if tr_truth else 0.0
    nb = novelty_band or (0.12, 0.5, 0.2)
    balance = None
    if pitch_balance and ex and tr_truth:
        # keep the section's mix of notes (F1 chugs vs F#2 leaps...): 1 - total-variation distance
        from collections import Counter
        a, b = Counter(n.pitch for n in ex), Counter(n.pitch for n in tr_truth)
        na, nb_ = sum(a.values()), sum(b.values())
        tv = 0.5 * sum(abs(a.get(p, 0) / na - b.get(p, 0) / nb_) for p in set(a) | set(b))
        balance = _band(1.0 - tv, 0.8, 1.0, 0.4)
    comps = {
        "groove": float(np.clip(groove, 0, 1)),
        "low_focus": _band(low_focus, 0.65, 1.0, 0.4),
        "chug_ratio": _band(chug, 0.45, 0.85, 0.35),
        "density": (_band(density / density_ref, 0.8, 1.15, 0.5) if density_ref else _band(density, 6.0, 12.0, 6.0)),
        "syncopation": _band(sync, 0.3, 0.65, 0.3),
        "vocab": _band(vocab, 2, 4, 3),
        "novelty": _band(novelty, nb[0], nb[1], nb[2]),
    }
    w = dict(DEFAULT_WEIGHTS)
    if balance is not None:
        comps["balance"] = balance
    if weights:
        w.update(weights)
    used = {k: w.get(k, 1.0) for k in comps}
    total = sum(used[k] * comps[k] for k in comps) / max(1e-9, sum(used.values()))
    return {"fitness": float(total), **comps, "raw": {"density": density, "novelty": novelty, "chug": chug,
                                                       "low": low_focus, "sync": sync, "vocab": vocab, "n": len(ex)}}


def score_generated(notes: list[Note], t0: float, t1: float, *, cycle16: int, pitch_dist: dict[int, float],
                    density_ref: float, weights: dict[str, float] | None = None,
                    low_pitch_max: int = config.TUNING_F_STANDARD[0] + 6) -> dict:
    """Fitness for a riff the fly made up (no transcription to compare with): groove at its own
    cycle length and the bar, low-string focus, chug ratio, density relative to the corpus, syncopation,
    vocabulary, and pitch balance against the corpus' pitch distribution."""
    ex = [n for n in notes if t0 <= n.start < t1]
    bars = max(1e-9, (t1 - t0) / 16.0)
    tr = onset_train(ex, t0, t1)
    groove = 0.6 * autocorr_at(tr, int(cycle16)) + 0.4 * autocorr_at(tr, BAR_LAG)
    low_focus = float(np.mean([n.pitch <= low_pitch_max for n in ex])) if ex else 0.0
    chug = float(np.mean([n.duration <= 1.0 for n in ex])) if ex else 0.0
    density = len(ex) / bars
    sync = float(np.mean([(round(n.start) % 4) in (1, 3) for n in ex])) if ex else 0.0
    vocab = len(set(n.pitch for n in ex))
    balance = 0.0
    if ex and pitch_dist:
        from collections import Counter
        a = Counter(n.pitch for n in ex)
        na = sum(a.values())
        tv = 0.5 * sum(abs(a.get(p, 0) / na - pitch_dist.get(p, 0.0)) for p in set(a) | set(pitch_dist))
        balance = _band(1.0 - tv, 0.7, 1.0, 0.5)
    comps = {
        "groove": float(np.clip(groove, 0, 1)),
        "low_focus": _band(low_focus, 0.65, 1.0, 0.4),
        "chug_ratio": _band(chug, 0.45, 0.85, 0.35),
        "density": _band(density / max(1e-9, density_ref), 0.7, 1.3, 0.5),
        "syncopation": _band(sync, 0.3, 0.65, 0.3),
        "vocab": _band(vocab, 2, 4, 3),
        "balance": balance,
    }
    w = {k: v for k, v in DEFAULT_WEIGHTS.items() if k != "novelty"}
    if weights:
        w.update({k: v for k, v in weights.items() if k in comps})
    total = sum(w[k] * comps[k] for k in comps) / max(1e-9, sum(w[k] for k in comps))
    return {"fitness": float(total), **comps, "raw": {"density": density, "novelty": None, "chug": chug,
                                                       "low": low_focus, "sync": sync, "vocab": vocab, "n": len(ex)}}
