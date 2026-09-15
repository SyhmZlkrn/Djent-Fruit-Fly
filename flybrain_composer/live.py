"""Live Stage B — the fly improvises *while it performs*, and can make up riffs of its own.

    python -m flybrain_composer.cli play --improvise          # improvise on the known song
    python -m flybrain_composer.cli play --generate           # riffs of its own (needs fit-multi)

While the current phrase (default 8 bars) plays, a planner process with a pool of workers auditions
candidate versions of the *next* phrase through the fixed brain. A candidate is a set of nine
*musical knobs* (:func:`unpack_knobs`): how hard each input stream hits the brain, the dopamine gain,
a polymetric displacement of the riff against the drum grid, a stretch of the riff cycle (a new
polymeter), a blend with another section's identity (or, for the generator, of two learned songs'
sections), and the decode threshold. Every knob is a transformation of the *inputs* the read-out
was trained on, so the output stays clean; musical guard-rails (pitch prior from the tab or the
corpus, 16th grid, density cap, tab velocity) keep it a variation rather than wrong notes.

Each candidate is graded (:mod:`fitness`), the best is committed 2.5 s before the boundary: rendered
through the 8ridge lite port into the guitar stem that is already streaming, sent to the stage,
the MIDI port and the live LIF. Its fitness advantage is the *treat* injected into PAM (+) / PPL1 (−)
during the next phrase. A 👍/👎 from the audience (stage buttons, K/J, or +/- in the brain window)
adds to that treat, re-weights the scorer toward what was liked, and nudges the search toward the
liked knobs — the fly learns the listener's taste. CMA-ES keeps evolving across phrases, so every
performance is a different take. The first phrase of a known song is always the faithful replay.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config
from .transcription import Note, Song

PHRASE_BARS = 8
MAX_CANDIDATES = 400
LIVE_LEVEL = 0.72 * 0.85     # peak-calibrated phrases sit at the level of the pre-rendered stem (measured)


# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# fingering for notes the tab does not contain
# ----------------------------------------------------------------------------------------------
def fingering_map(truth: list[Note]) -> dict[int, tuple[int, int]]:
    """pitch -> the (string, fret) the tab uses most often for it."""
    c: dict[int, Counter] = {}
    for n in truth:
        if n.string is not None and n.fret is not None:
            c.setdefault(n.pitch, Counter())[(n.string, n.fret)] += 1
    return {p: cnt.most_common(1)[0][0] for p, cnt in c.items()}


def finger(notes: list[Note], truth: list[Note], fmap: dict[int, tuple[int, int]]) -> list[Note]:
    from .sonify import transfer_articulation
    from .transcription import assign_string_fret
    transfer_articulation(notes, truth)
    for n in notes:
        if n.string is None:
            sf = fmap.get(n.pitch) or assign_string_fret(n.pitch)
            if sf is not None:
                n.string, n.fret = sf
    return notes


def note_to_dict(n: Note) -> dict:
    return {"start": n.start, "duration": n.duration, "pitch": n.pitch, "velocity": n.velocity,
            "string": n.string, "fret": n.fret, "palm_mute": bool(n.palm_mute), "bend": n.bend}


def note_from_dict(d: dict) -> Note:
    return Note(d["start"], d["duration"], d["pitch"], d["velocity"], d["string"], d["fret"], d["palm_mute"], d["bend"])


# ----------------------------------------------------------------------------------------------
# musical guard-rails: improvise the rhythm, not wrong notes
# ----------------------------------------------------------------------------------------------
NOVELTY_BAND = (0.10, 0.40, 0.15)   # want 10-40 % of onsets to differ from the tab
DENSITY_MAX = 1.3                   # at most 1.3x the tab's note count in the phrase
TAB_VELOCITY = 108


def wildness_settings(w: float) -> dict:
    """One knob (0 = stays very close to the tab, 1 = far out) -> search width, read-out perturbation
    scale, novelty band and the density cap."""
    w = float(np.clip(w, 0.0, 1.0))
    return {
        "sigma0": 0.15 + 0.4 * w,
        "delta_scale": 0.6 * w,
        "novelty_band": (0.05 + 0.10 * w, 0.20 + 0.50 * w, 0.15),
        "density_max": 1.1 + 0.3 * w,
    }


MIN_PITCH_SHARE = 0.08   # a pitch must make up 8 % of the tab's notes in the phrase to be playable


def pitch_prior(truth: list[Note], s16_0: float, s16_1: float, context16: float) -> dict[int, float]:
    """The riff's own vocabulary for this phrase: pitch -> prior weight in (0, 1]. Pitches the tab
    uses here (or in the neighbouring phrases if this one is silent) with at least MIN_PITCH_SHARE of
    the notes; the weight is sqrt(share / max share), so a section's main notes dominate and its
    rare passing notes stay rare."""
    from collections import Counter
    own = Counter(n.pitch for n in truth if s16_0 <= n.start < s16_1)
    if not own:
        own = Counter(n.pitch for n in truth if s16_0 - context16 <= n.start < s16_1 + context16)
    if not own:
        own = Counter(n.pitch for n in truth)
    total = sum(own.values())
    keep = {p: c for p, c in own.items() if c >= max(1, MIN_PITCH_SHARE * total)} or dict(own)
    top = max(keep.values())
    return {p: float(np.sqrt(c / top)) for p, c in keep.items()}


def allowed_pitches(truth: list[Note], s16_0: float, s16_1: float, context16: float) -> set[int]:
    return set(pitch_prior(truth, s16_0, s16_1, context16))


def mask_pitches(Yhat: np.ndarray, layout: dict, prior: dict[int, float] | set[int]) -> np.ndarray:
    """Zero the onset/sustain channels of every pitch outside the prior and scale the others by
    their prior weight before decoding."""
    o0, o1 = layout["onset"]
    s0, s1 = layout["sustain"]
    w = np.zeros(o1 - o0)
    items = prior.items() if isinstance(prior, dict) else ((p, 1.0) for p in prior)
    for p, weight in items:
        i = p - config.MIDI_LO
        if 0 <= i < len(w):
            w[i] = weight
    Y = Yhat.copy()
    Y[:, o0:o1] *= w[None, :]
    Y[:, s0:s1] *= (w > 0)[None, :]
    return Y


def musicalise(notes: list[Note], Yhat: np.ndarray, layout: dict, s16_0: float, n_truth: int,
               max_ratio: float = DENSITY_MAX, velocity: int = TAB_VELOCITY) -> list[Note]:
    """Snap onsets to the 16th grid and durations to whole sixteenths (min 1, as in the tab), merge
    duplicates, cut same-pitch overlaps, keep at most `max_ratio` x the tab's note count (the
    strongest onsets win) and use the tab's velocity — the search then varies *when* the riff's
    notes fall, not what they are."""
    sps = config.STEPS_PER_SIXTEENTH
    o0, _ = layout["onset"]
    T = len(Yhat)
    cand: dict[tuple[float, int], Note] = {}
    for n in notes:
        start = float(round(n.start))
        dur = float(max(1, round(n.duration)))
        key = (start, n.pitch)
        if key in cand:
            cand[key].duration = max(cand[key].duration, dur)
            continue
        step = int(round((start - s16_0) * sps))
        col = o0 + (n.pitch - config.MIDI_LO)
        lo, hi = max(0, step - 2), min(T, step + 3)
        strength = float(Yhat[lo:hi, col].max()) if hi > lo and 0 <= col < Yhat.shape[1] else 0.0
        m = Note(start, dur, n.pitch, velocity, None, None, False, 0.0)
        m._strength = strength           # noqa: SLF001 — transient, used for pruning only
        cand[key] = m
    out = sorted(cand.values(), key=lambda m: (m.start, m.pitch))
    n_max = max(4, int(np.ceil(max_ratio * n_truth)))
    if len(out) > n_max:
        keep = sorted(sorted(out, key=lambda m: -m._strength)[:n_max], key=lambda m: (m.start, m.pitch))
        out = keep
    # a note ends where the next onset on the same pitch begins
    by_pitch: dict[int, list[Note]] = {}
    for m in out:
        by_pitch.setdefault(m.pitch, []).append(m)
    for seq in by_pitch.values():
        for a, b in zip(seq, seq[1:]):
            a.duration = float(max(1.0, min(a.duration, b.start - a.start)))
    for m in out:
        try:
            del m._strength
        except AttributeError:
            pass
    return out


def _lower_priority():
    """Planner/worker processes yield to the audio callback, the brain window and the browser."""
    try:
        if os.name == "nt":
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)      # BELOW_NORMAL_PRIORITY_CLASS
        else:
            os.nice(5)
    except Exception:  # noqa: BLE001
        pass


# worker processes: each holds its own runner (model + connectome), evaluates candidates

# ----------------------------------------------------------------------------------------------
# the brain, one phrase at a time — musical knobs instead of random shoves
# ----------------------------------------------------------------------------------------------
KNOB_NAMES = ["gain_drums", "gain_riff", "gain_form", "dan", "displace", "stretch", "blend", "pick", "threshold"]
N_KNOBS = len(KNOB_NAMES)


def unpack_knobs(theta: np.ndarray, wild: dict) -> dict:
    """θ (N_KNOBS, ~N(0,σ)) -> the knobs. Every knob is a *musical* transformation of the inputs the
    brain is driven with, so the read-out stays on ground it knows:

    * gain_*    — how hard each stream (drums / riff cycle / song form) hits the brain;
    * dan       — how strongly the treat is injected into PAM (+) / PPL1 (−);
    * displace  — shift the riff cycle and the section material against the drum grid (sixteenths):
                  polymetric displacement, Meshuggah's own trick;
    * stretch   — run the riff/section phase faster or slower: a new polymeter (23/16, 27/16...);
    * blend     — mix another section's identity into the form stream (its material bleeds in);
    * pick      — which other section (or, for the generator, which learned code) to blend with;
    * threshold — the decode threshold: sparser <-> denser.
    """
    th = np.asarray(theta, dtype=np.float64)
    w = float(wild.get("wildness", 0.3))
    return {
        "gains": {"drums": float(np.exp(0.5 * th[0])), "riff": float(np.exp(0.5 * th[1])), "form": float(np.exp(0.5 * th[2]))},
        "dan_gain": float(2.0 * th[3]),
        "disp16": float((1.0 + 5.0 * w) * th[4]),
        "stretch": float(np.exp((0.05 + 0.25 * w) * th[5])),
        "blend": float(np.clip((0.2 + 0.5 * w) * abs(th[6]), 0.0, 0.7)),
        "pick": float(th[7]),
        "thr": float(np.clip(0.45 + 0.12 * th[8], 0.3, 0.75)),
    }


class PhraseRunner:
    """Plays one phrase through the fixed brain with a set of knobs and grades it.

    mode "song":     improvise on a known song (Stage A model, per-song one-hot form); the tab is the
                     reference for pitches, density and novelty.
    mode "generate": make up riffs (multi-song model, shared section codes); the knobs choose which
                     learned sections to blend; the corpus statistics are the reference.
    """

    def __init__(self, model, song: Song, *, song_key: str = "", form_mode: str = "onehot",
                 settings: dict | None = None, mode: str = "song", stats: dict | None = None,
                 codes: dict[str, np.ndarray] | None = None, start_bar: int = 0):
        from . import connectome
        from .meter import build_streams_ex, stack_streams
        self.model, self.song, self.mode = model, song, mode
        self.song_key, self.form_mode = song_key, form_mode
        self.settings = dict(settings or wildness_settings(0.3))
        self.stats = stats or {}
        self.codes = codes or {}
        self.code_keys = sorted(self.codes)
        self.truth = song.notes
        neurons, _ = connectome.load_subgraph()
        self.pam = np.nonzero(connectome.pam_mask(neurons))[0]
        self.ppl1 = np.nonzero(connectome.ppl1_mask(neurons))[0]
        self.section_ids: dict[str, int] = {}
        for sec in song.sections:
            self.section_ids.setdefault(sec.name, len(self.section_ids))
        self.names = list(self.section_ids)
        self.U_base = stack_streams(build_streams_ex(song, form_mode=form_mode, song_key=song_key,
                                                     section_ids=self.section_ids))
        self.dim = N_KNOBS
        res = model.reservoir()
        x = res.washout_state(self.U_base, 400)
        s0 = start_bar * 16 * config.STEPS_PER_SIXTEENTH
        self.x0 = res.run(self.U_base[:s0], x0=x, record=False) if s0 > 0 else x
        self.cycle16 = int(song.meta.get("cycle16", 25))
        # generator: a seeded walk through the learned riffs — every phrase starts from a different
        # learned section (the "set list"), the knobs choose a partner to blend in and the treatment
        self.sections_meta = dict(model.meta.get("multi", {}).get("sections", {}))
        rng = np.random.default_rng(int(song.meta.get("seed", 0)) + 11)
        good = [k for k in self.code_keys if (self.sections_meta.get(k, {}).get("stageA_f1") or 1.0) >= 0.5]
        self.walk = list(rng.permutation(good or self.code_keys))
        self.phrase_len16 = float(song.meta.get("phrase_len16", 64))

    def _gen_identity(self, k: dict, s16_0: float) -> tuple[str, str | None, float]:
        """(base section key, partner key or None, blend alpha) for the phrase starting at s16_0."""
        if not self.walk:
            return "", None, 0.0
        i = int(s16_0 // max(1.0, self.phrase_len16)) % len(self.walk)
        base = self.walk[i]
        alpha = float(min(0.5, k["blend"]))
        partner = None
        if alpha > 0.02 and len(self.walk) > 1:
            partner = self.walk[(i + 1 + int(abs(k["pick"]) * 7.0)) % len(self.walk)]
            if partner == base:
                partner = None
        return base, partner, (alpha if partner else 0.0)

    # ------------------------------------------------------------------ inputs with the knobs applied
    def streams(self, k: dict, s16_0: float) -> np.ndarray:
        from .meter import build_streams_ex, stack_streams
        sec = self.song.section_at(s16_0)
        blend = None
        vectors = None
        if self.mode == "generate":
            # the generator's section identity: this phrase's learned riff, optionally blended with a partner
            base, partner, alpha = self._gen_identity(k, s16_0)
            if base:
                code = self.codes[base]
                if partner:
                    code = (1 - alpha) * code + alpha * self.codes[partner]
                    code = code / max(1e-9, np.linalg.norm(code))      # learned codes have unit norm
                vectors = {s.name: code for s in self.song.sections}   # the whole frame gets this identity
        elif k["blend"] > 0.02 and len(self.names) > 1 and sec is not None:
            others = [n for n in self.names if n != sec.name]
            other = others[int(abs(k["pick"]) * 7.0) % len(others)]
            blend = (other, k["blend"])
        plain = abs(k["disp16"]) < 1e-3 and abs(k["stretch"] - 1.0) < 1e-3 and blend is None and vectors is None
        if plain:
            return self.U_base
        return stack_streams(build_streams_ex(self.song, disp16=k["disp16"], stretch=k["stretch"], blend=blend,
                                              form_mode=self.form_mode, song_key=self.song_key,
                                              section_ids=self.section_ids, section_vectors=vectors))

    # ------------------------------------------------------------------ one phrase
    def run_phrase(self, theta: np.ndarray, reward: float, s16_0: float, s16_1: float, x0: np.ndarray,
                   weights: dict | None = None, prev_notes: list[dict] | None = None) -> tuple[dict, list[Note], np.ndarray]:
        from .fitness import _band, score, score_generated
        from .sonify import diff_notes
        from .reservoir import Reservoir, features
        from .sonify import decode_notes
        k = unpack_knobs(theta, self.settings)
        base = self.model.input_map.gains
        imap = self.model.input_map.with_gains({n: base[n] * g for n, g in k["gains"].items()})
        res = Reservoir(self.model.W, imap.W_in, self.model.reservoir_params)
        sps = config.STEPS_PER_SIXTEENTH
        s0, s1 = int(round(s16_0 * sps)), int(round(s16_1 * sps))
        U_all = self.streams(k, s16_0)
        U = U_all[s0:s1]
        vec = np.zeros(self.model.W.shape[0])
        vec[self.pam] = k["dan_gain"] * reward
        vec[self.ppl1] = -k["dan_gain"] * reward
        X = res.run(U, x0=x0, extra=np.broadcast_to(vec, (len(U), len(vec))))
        Yhat = self.model.features(X, U) @ self.model.W_out
        L16 = s16_1 - s16_0
        bars = max(1e-9, L16 / 16.0)
        layout = self.model.layout
        if self.mode == "song":
            truth_win = [n for n in self.truth if s16_0 <= n.start < s16_1]
            prior = pitch_prior(self.truth, s16_0, s16_1, context16=L16)
            n_ref = len(truth_win)
        else:
            # vocabulary and density of the learned riff(s) this phrase is made from
            base, partner, alpha = self._gen_identity(k, s16_0)
            meta_b = self.sections_meta.get(base, {})
            meta_p = self.sections_meta.get(partner or "", {})
            dist = dict(self.stats.get("pitch_dist", {}))
            if meta_b.get("pitch_dist"):
                dist = {int(p): (1 - alpha) * v for p, v in meta_b["pitch_dist"].items()}
                for p, v in (meta_p.get("pitch_dist") or {}).items():
                    dist[int(p)] = dist.get(int(p), 0.0) + alpha * v
            dist = {int(p): float(v) for p, v in dist.items()}
            keep = {p: v for p, v in dist.items() if v >= 0.04} or dict(dist)
            top = max(keep.values()) if keep else 1.0
            prior = {p: float(np.sqrt(v / top)) for p, v in keep.items()}
            dens = float(self.stats.get("density", 7.5))
            if meta_b.get("density"):
                dens = (1 - alpha) * float(meta_b["density"]) + alpha * float(meta_p.get("density", meta_b["density"]))
            if self.stats.get("density_override"):
                dens = float(self.stats["density_override"])
            n_ref = int(round(dens * bars))
            gen_dist, gen_density = dist, dens
        Ym = mask_pitches(Yhat, layout, prior)
        notes = decode_notes(Ym, layout, onset_threshold=k["thr"], sustain_threshold=max(0.25, k["thr"] - 0.05))
        for n in notes:
            n.start += s16_0
        notes = musicalise(notes, Ym, layout, s16_0, n_truth=n_ref, max_ratio=self.settings.get("density_max", DENSITY_MAX))
        if self.mode == "song":
            sc = score(notes, self.truth, s16_0, s16_1, bpm=self.song.bpm, density_ref=max(1.0, n_ref / bars),
                       novelty_band=self.settings.get("novelty_band", NOVELTY_BAND), pitch_balance=True, weights=weights)
        else:
            cyc = self.sections_meta.get(base, {}).get("cycle16") or self.cycle16
            sc = score_generated(notes, s16_0, s16_1, cycle16=int(cyc), pitch_dist=gen_dist,
                                 density_ref=gen_density, weights=weights)
            # freshness: do not play what you just played (onset F1 against the previous phrase)
            if prev_notes:
                rel = [Note(n.start - s16_0, n.duration, n.pitch, n.velocity) for n in notes]
                prev = [Note(d["start"], d["duration"], d["pitch"], d["velocity"]) for d in prev_notes]
                sim = diff_notes(rel, prev, bpm=self.song.bpm)["f1"] if (rel and prev) else 0.0
                fresh = _band(1.0 - sim, 0.35, 1.0, 0.3)
                w_f = (weights or {}).get("fresh", 1.5)
                tot = sum({**{c: 1.0 for c in sc if c not in ("fitness", "raw")}, **(weights or {})}.get(c, 1.0) for c in sc if c not in ("fitness", "raw"))
                sc["fresh"] = fresh
                sc["fitness"] = float((sc["fitness"] * tot + w_f * fresh) / (tot + w_f))
                sc["raw"]["repeat"] = float(sim)
            sc["source"] = base + (f" + {partner}" if partner else "")
        return sc, notes, X[-1].astype(np.float64)


# worker processes: each holds its own runner (model + connectome), evaluates candidates
_RUNNER = None


def _make_runner(cfg: "PlannerConfig", start_bar: int) -> PhraseRunner:
    from .model import ComposerModel
    from .transcription import load_song
    wild = wildness_settings(cfg.wildness)
    if cfg.mode == "generate":
        from .corpus import MODEL_MULTI_PATH, learned_codes, synthetic_song
        model = ComposerModel.load(MODEL_MULTI_PATH)
        stats = dict(model.meta.get("multi", {}).get("stats", {}))
        if cfg.density:
            stats["density"] = float(cfg.density)
            stats["density_override"] = float(cfg.density)
        song = synthetic_song(cfg.bpm or float(stats.get("bpm", config.BPM)), cfg.phrases, cfg.phrase_bars, cfg.cycle16)
        song.meta.update({"seed": cfg.seed, "phrase_len16": cfg.phrase_bars * 16})
        return PhraseRunner(model, song, song_key="gen", form_mode="code", settings=wild, mode="generate",
                            stats=stats, codes=learned_codes(model), start_bar=start_bar)
    song = load_song(cfg.gp, track=cfg.track)
    return PhraseRunner(ComposerModel.load(), song, settings=wild, mode="song", start_bar=start_bar)


def _worker_init(cfg, start_bar):
    global _RUNNER
    _lower_priority()
    _RUNNER = _make_runner(cfg, start_bar)


def _worker_info():
    return {"dim": _RUNNER.dim, "x0": _RUNNER.x0, "bpm": _RUNNER.song.bpm, "n16": _RUNNER.song.n_sixteenths,
            "cycle16": _RUNNER.cycle16}


def _worker_eval(theta, reward, s16_0, s16_1, x0, weights=None, prev_notes=None):
    sc, notes, x_end = _RUNNER.run_phrase(theta, reward, s16_0, s16_1, x0, weights=weights, prev_notes=prev_notes)
    return sc, [note_to_dict(n) for n in notes], x_end

# ----------------------------------------------------------------------------------------------
# the sound, one phrase at a time
# ----------------------------------------------------------------------------------------------
class PhraseRenderer:
    """Double-tracked 8ridge lite guitar rendered phrase by phrase into running DI buffers, so
    note tails cross phrase boundaries and the amp sees continuous input; one fixed level for
    the whole show (calibrated on the first phrase)."""

    TAKES = (("Natural", 35.0, 110.0, 0.0), ("Tuned", 31.0, 125.0, 7.0))   # set, drive, tight Hz, delay ms

    def __init__(self, bpm: float, total_s: float, sr: int = 48000, seed: int = 0, pre_s: float = 0.25, tail_s: float = 0.6):
        from .bridgelite import BridgeliteEngine
        self.bpm, self.sr = bpm, sr
        self.eng = {name: BridgeliteEngine(sample_set=name, sample_rate=sr) for name, *_ in self.TAKES}
        n = int(total_s * sr) + sr
        self.di = [np.zeros(n, np.float32) for _ in self.TAKES]
        self.pre, self.tail = int(pre_s * sr), int(tail_s * sr)
        self.peaks: dict | None = None
        self.rng = np.random.default_rng(seed)

    def render(self, notes: list[Note], s16_0: float, s16_1: float) -> tuple[int, np.ndarray]:
        """Returns (sample offset, stereo float32 chunk covering the phrase plus its tail)."""
        from .bridgelite import amp_chain
        from .sonify import articulate
        sec16 = config.sixteenth_seconds(self.bpm)
        s0 = int(round(s16_0 * sec16 * self.sr))
        s1 = int(round(s16_1 * sec16 * self.sr))
        rel = articulate([Note(n.start - s16_0, n.duration, n.pitch, n.velocity, n.string, n.fret, n.palm_mute, n.bend)
                          for n in notes], bpm=self.bpm)
        outs, amp_peaks = [], []
        a0 = max(0, s0 - self.pre)
        for k, (name, drive, tight, delay_ms) in enumerate(self.TAKES):
            take = [Note(n.start, n.duration, n.pitch, int(np.clip(n.velocity + (self.rng.integers(-5, 6) if k else 0), 1, 127)),
                         n.string, n.fret, n.palm_mute, n.bend) for n in rel]
            mono = self.eng[name].render(take, bpm=self.bpm, tail_sec=0.6).mean(axis=1) if take else np.zeros(1, np.float32)
            d = int(delay_ms * self.sr / 1000)
            a, b = s0 + d, min(s0 + d + len(mono), len(self.di[k]))
            if b > a:
                self.di[k][a:b] += mono[: b - a]
            a1 = min(len(self.di[k]), s1 + self.tail + d)
            seg = self.di[k][a0:a1]
            if self.peaks is None:
                y, pk = amp_chain(seg, self.sr, drive=drive, tight_hz=tight, return_peak=True)
                amp_peaks.append(pk)
            else:
                y = amp_chain(seg, self.sr, drive=drive, tight_hz=tight, peak=self.peaks["amp"][k])
            outs.append(y[s0 - a0:])
        n = min(len(o) for o in outs)
        st = np.stack([outs[0][:n], outs[1][:n]], axis=1)
        if self.peaks is None:
            self.peaks = {"amp": amp_peaks, "out": float(np.abs(st).max()) or 1.0}
        return s0, (st / self.peaks["out"] * LIVE_LEVEL).astype(np.float32)



# ----------------------------------------------------------------------------------------------
# the planner process
# ----------------------------------------------------------------------------------------------
@dataclass
class PlannerConfig:
    mode: str = "song"           # "song": improvise on the known song · "generate": riffs of its own
    phrase_bars: int = PHRASE_BARS
    popsize: int = 8
    sigma0: float | None = None  # None: from wildness
    seed: int = 0
    start_s: float = 0.0
    margin_s: float = 2.5
    enabled: bool = True
    wildness: float = 0.3        # 0 = hugs the tab, 1 = far out (search width, knob ranges, novelty band, density cap)
    base_midi: str | None = None
    sr: int = 48000
    verbose: bool = True
    gp: str | None = None
    track: str = "Rhythm"
    workers: int = 4             # candidate-evaluation processes (below-normal priority; 8 cores here)
    elite: bool = True           # re-audition the best θ so far in every phrase
    human_gain: float = 0.5      # a 👍/👎 adds ±this to the treat of the phrase it lands on
    taste_rate: float = 0.5      # how fast 👍/👎 re-weight the scorer's components
    # generator only
    bpm: float | None = None
    phrases: int = 24
    cycle16: int = 25
    snare: str = "24"           # "24" = beats 2 and 4 · "thirds" = every 3 sixteenths
    density: float | None = None   # notes per bar the generator aims for (default: the corpus median)


def planner_main(conn, cfg: PlannerConfig):
    """Child process: plans phrases ahead of the audio clock and streams them back."""
    from concurrent.futures import ProcessPoolExecutor

    import cma
    from .fitness import DEFAULT_WEIGHTS, score
    from .sonify import midi_to_notes
    from .transcription import load_song

    def log(msg):
        conn.send(("log", msg))

    _lower_priority()
    pool = None
    try:
        wild = wildness_settings(cfg.wildness)
        wild["wildness"] = cfg.wildness
        if cfg.mode == "generate":
            from .corpus import MODEL_MULTI_PATH, riff_drums, synthetic_song
            from .model import ComposerModel
            model_meta = ComposerModel.load(MODEL_MULTI_PATH).meta
            stats = model_meta.get("multi", {}).get("stats", {})
            bpm = cfg.bpm or float(stats.get("bpm", config.BPM))
            song = synthetic_song(bpm, cfg.phrases, cfg.phrase_bars, cfg.cycle16)
            truth = []
            fmap = {int(p): tuple(v) for p, v in stats.get("fingering", {}).items()}
            base_notes = None
        else:
            song = load_song(cfg.gp, track=cfg.track)
            truth = song.notes
            fmap = fingering_map(truth)
            base_notes = midi_to_notes(cfg.base_midi, bpm=song.bpm)[0] if cfg.base_midi and Path(cfg.base_midi).exists() else None
            if base_notes:
                finger(base_notes, truth, fmap)
        sec16 = config.sixteenth_seconds(song.bpm)
        L16 = cfg.phrase_bars * 16
        n16 = song.n_sixteenths
        n_phr = int(np.ceil(n16 / L16))
        k0 = min(n_phr - 1, int(cfg.start_s / sec16) // L16)
        pool = ProcessPoolExecutor(max_workers=max(1, cfg.workers), mp_context=mp.get_context("spawn"),
                                   initializer=_worker_init, initargs=(cfg, k0 * cfg.phrase_bars))
        info = pool.submit(_worker_info).result(timeout=300)
        dim, x = info["dim"], np.asarray(info["x0"], dtype=np.float64)
        renderer = PhraseRenderer(song.bpm, total_s=song.seconds + 3.0, sr=cfg.sr, seed=cfg.seed)
        sigma0 = cfg.sigma0 if cfg.sigma0 is not None else wild["sigma0"]
        es = cma.CMAEvolutionStrategy(np.zeros(dim), sigma0, {"popsize": cfg.popsize, "seed": cfg.seed + 1, "verbose": -9})
        weights = {k: v for k, v in DEFAULT_WEIGHTS.items()}
        conn.send(("ready", {"n_phrases": n_phr, "phrase_bars": cfg.phrase_bars, "k0": k0, "dim": dim, "mode": cfg.mode,
                             "bpm": song.bpm, "duration": song.seconds, "n16": n16, "cycle16": info["cycle16"],
                             "workers": cfg.workers, "wildness": cfg.wildness, "sigma0": sigma0,
                             "knobs": KNOB_NAMES, "weights": weights,
                             **{k: (list(v) if isinstance(v, tuple) else v) for k, v in wild.items()}}))
    except Exception as e:  # noqa: BLE001
        import traceback
        conn.send(("error", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}"))
        if pool is not None:
            pool.shutdown(cancel_futures=True)
        return

    enabled = cfg.enabled
    clock = {"t0": None, "wall0": None}
    running = None
    reward = 0.0                 # the treat currently injected into the DANs (heuristic advantage + 👍/👎)
    base_treat = 0.0
    margin = cfg.margin_s
    best_ever = None
    gen_time = 0.6
    phrase_log: dict[int, dict] = {}     # k -> {"theta", "comps", "improvised"}
    comp_mean: dict[str, float] = {}
    human = {"up": 0, "down": 0}
    human_by_phrase: dict[int, float] = {}
    es_sigma_max = 2.0 * sigma0

    def song_now() -> float:
        if clock["t0"] is None:
            return -1e9
        return clock["t0"] + (time.time() - clock["wall0"])

    def phrase_at(t: float) -> int:
        return k0 + int(max(0.0, t - k0 * L16 * sec16) // (L16 * sec16))

    def apply_human(value: float, t: float):
        """👍/👎 for the phrase playing at song time t: it becomes part of the treat, re-weights the
        scorer toward what was liked, and nudges the search toward (👍) or away from (👎) those knobs."""
        nonlocal reward, weights
        k = phrase_at(t)
        human["up" if value > 0 else "down"] += 1
        human_by_phrase[k] = human_by_phrase.get(k, 0.0) + value
        reward = float(np.clip(base_treat + cfg.human_gain * human_by_phrase.get(k, 0.0), -1, 1))
        rec = phrase_log.get(k)
        if rec is not None and comp_mean:
            new = {}
            for c, w in weights.items():
                if c in rec["comps"] and c in comp_mean:
                    new[c] = float(np.clip(w * np.exp(cfg.taste_rate * value * (rec["comps"][c] - comp_mean[c])), 0.2, 4.0))
                else:
                    new[c] = w
            weights = new
            if rec.get("improvised") and rec.get("theta") is not None:
                try:
                    if value > 0:
                        es.mean = 0.7 * np.asarray(es.mean) + 0.3 * np.asarray(rec["theta"])
                    else:
                        es.sigma = float(min(es_sigma_max, es.sigma * 1.15))
                except Exception:  # noqa: BLE001 — cma internals; a failed nudge is not fatal
                    pass
        conn.send(("taste", {"weights": weights, "human": dict(human), "phrase": k, "value": value, "reward": reward}))

    def drain() -> bool:
        nonlocal enabled
        while conn.poll():
            msg = conn.recv()
            if msg[0] == "stop":
                return False
            if msg[0] in ("start", "sync"):
                clock["t0"], clock["wall0"] = msg[1], msg[2]
            elif msg[0] == "enable":
                enabled = bool(msg[1])
                log(f"live learning {'ON' if enabled else 'off'} (from the next phrase)")
            elif msg[0] == "human":
                apply_human(float(msg[1]), float(msg[2]))
        return True

    prev_rel: list[dict] | None = None      # the last committed phrase, phrase-relative (freshness)

    def evaluate(thetas, reward_, s16_0, s16_1, x_):
        futs = [pool.submit(_worker_eval, np.asarray(th), reward_, s16_0, s16_1, x_, weights, prev_rel) for th in thetas]
        return [f.result() for f in futs]

    try:
        for k in range(k0, n_phr):
            s16_0, s16_1 = k * L16, min(n16, (k + 1) * L16)
            T_k = s16_0 * sec16
            while k > k0 and song_now() < T_k - ((L16 * sec16 + 0.5) if enabled else (margin + 1.5)):
                if not drain():
                    return
                time.sleep(0.05)
            if not drain():
                return
            # generator: every phrase is the fly's own; song: the first phrase is the faithful replay
            improvise = enabled and (k > k0 or cfg.mode == "generate")
            best = None
            n_cand = n_gen = 0
            t_plan = time.perf_counter()
            if improvise:
                deadline = T_k - margin
                while n_cand < MAX_CANDIDATES:
                    if not drain():
                        return
                    if not enabled and cfg.mode == "song":
                        best = None
                        break
                    now = song_now()
                    if best is not None and now + gen_time >= deadline:
                        break
                    if best is None and now >= T_k - 0.3 * margin:
                        break
                    t_g = time.perf_counter()
                    thetas = list(es.ask())
                    extra = [best_ever[1]] if (cfg.elite and best_ever is not None and n_gen == 0) else []
                    results = evaluate(thetas + extra, reward, s16_0, s16_1, x)
                    fits = []
                    for th, (sc, notes_d, x_end) in zip(thetas + extra, results):
                        n_cand += 1
                        if len(fits) < len(thetas):
                            fits.append(sc["fitness"])
                        if best is None or sc["fitness"] > best[0]:
                            best = (sc["fitness"], np.asarray(th).copy(), sc, notes_d, x_end)
                    es.tell(thetas, [-f for f in fits])
                    n_gen += 1
                    gen_time = 0.7 * gen_time + 0.3 * (time.perf_counter() - t_g)
                    conn.send(("progress", {"k": k, "n_cand": n_cand, "n_gen": n_gen, "best": best[0],
                                            "novelty": best[2]["raw"]["novelty"], "groove": best[2]["groove"],
                                            "density": best[2]["raw"]["density"], "time_left": deadline - song_now()}))
                    if k == k0 and cfg.mode == "generate" and n_gen >= 3:
                        break                                   # the very first riff: don't keep the audience waiting
                if best is None:
                    improvise = False
            if improvise:
                f, th, sc, notes_d, x_end = best
                notes = finger([note_from_dict(d) for d in notes_d], truth, fmap)
                theta_used = th
                if best_ever is None or f >= best_ever[0] * 0.98:
                    best_ever = (f, th.copy())
            else:
                (sc_run, notes_d, x_end), = evaluate([np.zeros(dim)], 0.0, s16_0, s16_1, x)
                if base_notes is not None:
                    notes = [Note(n.start, n.duration, n.pitch, n.velocity, n.string, n.fret, n.palm_mute, n.bend)
                             for n in base_notes if s16_0 <= n.start < s16_1]
                    sc = score(notes, truth, s16_0, s16_1, bpm=song.bpm)
                else:
                    notes, sc = finger([note_from_dict(d) for d in notes_d], truth, fmap), sc_run
                theta_used = np.zeros(dim)
            comps = {c: float(v) for c, v in sc.items() if c not in ("fitness", "raw", "source")}
            phrase_log[k] = {"theta": theta_used, "comps": comps, "improvised": improvise}
            for c, v in comps.items():
                comp_mean[c] = v if c not in comp_mean else 0.8 * comp_mean[c] + 0.2 * v
            # the treat: advantage over the running average, injected during the *next* phrase
            if running is None:
                running = sc["fitness"]
            base_treat = float(np.clip(sc["fitness"] - running, -1, 1))
            running += 0.1 * (sc["fitness"] - running)
            reward = float(np.clip(base_treat + cfg.human_gain * human_by_phrase.get(k, 0.0), -1, 1))
            t_r = time.perf_counter()
            s0, chunk = renderer.render(notes, s16_0, s16_1)
            drums_d, drums_audio = [], None
            if cfg.mode == "generate":
                from .bridgelite import render_drums
                from .corpus import riff_drums
                drums = riff_drums(notes, s16_0, s16_1, cycle16=info["cycle16"], snare=cfg.snare)
                rel = [Note(d.start - s16_0, d.duration, d.pitch, d.velocity) for d in drums]
                drums_audio = (render_drums(rel, cfg.sr, song.bpm, tail_sec=0.5) * 0.4).astype(np.float32)
                drums_d = [note_to_dict(d) for d in drums]
            render_s = time.perf_counter() - t_r
            conn.send(("phrase", {
                "k": k, "n_phrases": n_phr, "s16_0": s16_0, "s16_1": s16_1, "t0": T_k, "t1": s16_1 * sec16,
                "notes": [note_to_dict(n) for n in notes], "fitness": sc["fitness"], "comps": comps, "raw": sc["raw"],
                "improvised": improvise, "n_cand": n_cand, "n_gen": n_gen, "reward_in": reward,
                "reward_out": base_treat, "baseline": running, "theta": [float(v) for v in theta_used],
                "knobs": unpack_knobs(theta_used, wild) if improvise else None,
                "render_s": render_s, "plan_s": time.perf_counter() - t_plan, "s0": s0, "audio": chunk,
                "drums": drums_d, "drums_audio": drums_audio, "weights": weights, "human": dict(human),
                "mode": cfg.mode, "section": (sc.get("source") or (song.section_at(s16_0).name if song.section_at(s16_0) else "")),
            }))
            x = np.asarray(x_end, dtype=np.float64)
            margin = max(cfg.margin_s, 1.6 * render_s + 0.6)
            prev_rel = [{"start": n.start - s16_0, "duration": n.duration, "pitch": n.pitch, "velocity": n.velocity} for n in notes]
            try:
                if es.sigma < 0.5 * sigma0:               # keep exploring: the search must not freeze on one riff
                    es.sigma = 0.5 * sigma0
            except Exception:  # noqa: BLE001
                pass
        conn.send(("done", {}))
    except Exception as e:  # noqa: BLE001
        import traceback
        conn.send(("error", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}"))
    finally:
        pool.shutdown(cancel_futures=True)

# ----------------------------------------------------------------------------------------------
# parent-side handle
# ----------------------------------------------------------------------------------------------
class LivePlanner:
    """Owns the planner process; the show polls it every frame."""

    def __init__(self, cfg: PlannerConfig):
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(v, "2")            # the child inherits this; the parent is already loaded
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=planner_main, args=(child, cfg), daemon=False, name="flybrain-planner")  # spawns its own worker pool
        self.proc.start()
        self.cfg = cfg
        self.info: dict = {}
        self.enabled = cfg.enabled
        self.done = False

    def _recv(self, timeout: float):
        if self.conn.poll(timeout):
            return self.conn.recv()
        return None

    def wait_ready(self, timeout: float = 180.0) -> dict:
        t_end = time.time() + timeout
        while time.time() < t_end:
            msg = self._recv(0.5)
            if msg is None:
                continue
            if msg[0] == "ready":
                self.info = msg[1]
                return self.info
            if msg[0] == "error":
                raise RuntimeError(f"planner failed: {msg[1]}")
            if msg[0] == "log":
                print(f"[live] {msg[1]}", flush=True)
        raise TimeoutError("planner did not become ready")

    def wait_phrase(self, timeout: float = 120.0) -> dict | None:
        """Blocks until the next committed phrase arrives (used for the first one, before the clock starts)."""
        t_end = time.time() + timeout
        while time.time() < t_end:
            msg = self._recv(0.5)
            if msg is None:
                continue
            if msg[0] == "phrase":
                return msg[1]
            if msg[0] == "log":
                print(f"[live] {msg[1]}", flush=True)
            if msg[0] == "error":
                raise RuntimeError(f"planner failed: {msg[1]}")
        return None

    def poll(self) -> list[tuple]:
        out = []
        while self.conn.poll():
            msg = self.conn.recv()
            if msg[0] == "done":
                self.done = True
            out.append(msg)
        return out

    def start(self, t0: float):
        self.conn.send(("start", t0, time.time()))

    def sync(self, t: float):
        self.conn.send(("sync", t, time.time()))

    def set_enabled(self, on: bool):
        self.enabled = bool(on)
        self.conn.send(("enable", self.enabled))

    def human(self, value: float, t: float):
        """👍 (+1) / 👎 (-1) for the phrase playing at song time t."""
        self.conn.send(("human", float(value), float(t)))

    def stop(self):
        try:
            self.conn.send(("stop",))
        except (OSError, ValueError):
            pass
        self.proc.join(6.0)
        if self.proc.is_alive():
            self.proc.terminate()
