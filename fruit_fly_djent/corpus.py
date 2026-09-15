"""A corpus of djent tabs -> one shared read-out -> riffs of the fly's own.

    data/songs/*.gp | *.gp5 | *.gp4 | *.gp3 | *.pdf      (drop tabs here; audio is not used for learning)
    python -m fruit_fly_djent.cli corpus               # list what loads, how each is transposed
    python -m fruit_fly_djent.cli fit-multi            # shared read-out -> data/cache/model_multi.npz
    python -m fruit_fly_djent.cli generate --bars 32   # riffs of its own -> output/fruit_fly_djent_*.mid/.wav
    python -m fruit_fly_djent.cli play --generate      # live: the fly plays riffs it makes up, 👍/👎 steer it

The fly learns from *tabs* (Guitar Pro files or the tab-only PDF layout the OMR reads), not from
audio — the read-out is fitted to a piano roll, and only a transcription gives one. Every song is
transposed so its lowest open string is the fly's low F (F1 = MIDI 29): riff shapes survive, the
8-string in F standard on stage stays correct, and the read-out sees one consistent pitch space.

Multi-song learning uses the shared "section code" form encoding (:func:`meter.build_streams_ex`
with ``form_mode="code"``): every (song, section) gets a fixed 16-dim +-1 code instead of a per-song
one-hot, so one read-out can be fitted to all songs at once (Gram matrices accumulated per song) and
a *new* code — an interpolation between learned sections — asks the same read-out for material it
was never shown: that is the riff generator.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config, connectome
from .meter import SECTION_CODE_DIM, build_streams_ex, make_input_map, section_code, stack_streams
from .model import ComposerModel
from .reservoir import ReservoirParams, features
from .transcription import Note, Section, Song, assign_string_fret, load_guitar_pro

SONGS_DIR = config.DATA_DIR / "songs"
MODEL_MULTI_PATH = config.CACHE_DIR / "model_multi.npz"
CORPUS_INDEX = config.CACHE_DIR / "corpus.json"
LOW_F = config.TUNING_F_STANDARD[0]


@dataclass
class CorpusSong:
    key: str
    path: str
    song: Song
    transpose: int
    source_tuning: tuple[int, ...]
    stageA: dict = field(default_factory=dict)


def _load_any(path: Path, track: str | int = "Rhythm") -> Song:
    suf = path.suffix.lower()
    if suf == ".pdf":
        from .pdftab import load_pdf_tab
        return load_pdf_tab(path)
    if suf == ".gp":
        from .gpif import load_gp
        return load_gp(path, track=track)
    if suf in (".gp3", ".gp4", ".gp5"):
        return load_guitar_pro(path)
    raise ValueError(f"unsupported tab format: {path.name}")


def transpose_to_fly(song: Song) -> tuple[Song, int]:
    """Shift the song so its lowest open string is the fly's low F; re-finger anything whose
    (string, fret) no longer matches the fly's tuning."""
    low = min(song.tuning) if song.tuning else min((n.pitch for n in song.notes), default=LOW_F)
    shift = LOW_F - low
    if shift == 0:
        return song, 0
    notes = []
    for n in song.notes:
        p = n.pitch + shift
        s_f = (n.string, n.fret)
        if n.string is None or n.fret is None or n.string >= len(config.TUNING_F_STANDARD) \
                or config.TUNING_F_STANDARD[n.string] + n.fret != p:
            s_f = assign_string_fret(p) or (None, None)
        notes.append(Note(n.start, n.duration, p, n.velocity, s_f[0], s_f[1], n.palm_mute, n.bend))
    out = Song(song.title, song.bpm, notes, drums=song.drums, sections=song.sections, riff_starts=song.riff_starts,
               tuning=config.TUNING_F_STANDARD, meta={**song.meta, "transposed_semitones": shift})
    return out, shift


def load_corpus(songs_dir: Path = SONGS_DIR, include_main: bool = True, track: str | int = "Rhythm",
                verbose: bool = True) -> list[CorpusSong]:
    paths = []
    if include_main:
        for ext in (".pdf", ".gp", ".gp5", ".gp4", ".gp3"):
            p = config.DATA_DIR / f"rational_gaze{ext}"
            if p.exists():
                paths.append(p)
                break
    if songs_dir.exists():
        paths += sorted(p for p in songs_dir.iterdir() if p.suffix.lower() in (".gp", ".gp5", ".gp4", ".gp3", ".pdf"))
    out = []
    for p in paths:
        try:
            song = _load_any(p, track=track)
        except Exception as e:  # noqa: BLE001 — one bad tab must not sink the corpus
            if verbose:
                print(f"[corpus] skip {p.name}: {type(e).__name__}: {e}", flush=True)
            continue
        if not song.notes:
            if verbose:
                print(f"[corpus] skip {p.name}: no notes", flush=True)
            continue
        src_tuning = tuple(song.tuning)
        song, shift = transpose_to_fly(song)
        if not song.sections:
            # a tab without sections: one section per 8 bars so the form stream has something to say
            L = 128
            song.sections = [Section(f"bars {i * 8 + 1}-{i * 8 + 8}", i * L, min(song.n_sixteenths, (i + 1) * L))
                             for i in range(int(np.ceil(song.n_sixteenths / L)))]
        if not song.riff_starts:
            song.riff_starts = [sec.start for sec in song.sections]
        key = p.stem.lower().replace(" ", "_")
        out.append(CorpusSong(key=key, path=str(p), song=song, transpose=shift, source_tuning=src_tuning))
        if verbose:
            print(f"[corpus] {p.name}: '{song.title}' {song.bpm:.0f} BPM, {len(song.notes)} notes, {song.n_bars} bars, "
                  f"{len({s.name for s in song.sections})} sections, tuning low {min(src_tuning) if src_tuning else '?'} "
                  f"-> shift {shift:+d}", flush=True)
    return out


def riff_excerpt(song: Song, bars: int, low_pitch_max: int = LOW_F + 6) -> Song:
    """The song's most characteristic riffs: whole sections ranked by (notes x low-string share),
    taken until `bars` bars are covered, re-timed back to back (section names kept, one riff cycle
    per section start). A linear read-out of 2,318 neurons cannot memorise ten whole songs; it can
    memorise ten songs' *riffs*."""
    if bars <= 0 or song.n_bars <= bars:
        return song
    secs = [sec for sec in song.sections if sec.end > sec.start]
    if not secs:
        return song
    seen, uniq = set(), []
    for sec in secs:                                   # one copy of each named section (the first)
        if sec.name not in seen:
            seen.add(sec.name)
            uniq.append(sec)

    def score(sec: Section) -> float:
        ns = [n for n in song.notes if sec.start <= n.start < sec.end]
        if not ns:
            return -1.0
        low = float(np.mean([n.pitch <= low_pitch_max for n in ns]))
        return len(ns) / max(1.0, (sec.end - sec.start) / 16.0) * (0.5 + low)

    ranked = sorted(uniq, key=score, reverse=True)
    chosen, total = [], 0.0
    for sec in ranked:
        if score(sec) < 0:
            continue
        chosen.append(sec)
        total += (sec.end - sec.start) / 16.0
        if total >= bars:
            break
    chosen.sort(key=lambda sec: sec.start)             # keep the song's order
    notes, drums, sections, riff_starts = [], [], [], []
    t = 0.0
    for sec in chosen:
        L = float(np.ceil((sec.end - sec.start) / 16.0) * 16.0)   # whole bars
        for n in song.notes:
            if sec.start <= n.start < sec.end:
                notes.append(Note(n.start - sec.start + t, min(n.duration, sec.end - n.start), n.pitch, n.velocity,
                                  n.string, n.fret, n.palm_mute, n.bend))
        for d in song.drums:
            if sec.start <= d.start < sec.end:
                drums.append(Note(d.start - sec.start + t, d.duration, d.pitch, d.velocity))
        sections.append(Section(sec.name, t, t + L))
        riff_starts += [r - sec.start + t for r in song.riff_starts if sec.start <= r < sec.end] or [t]
        t += L
    return Song(song.title, song.bpm, notes, drums=drums, sections=sections, riff_starts=sorted(set(riff_starts)),
                tuning=song.tuning, meta={**song.meta, "excerpt_bars": int(t // 16), "excerpt_sections": [c.name for c in chosen]})


def corpus_stats(songs: list[Song]) -> dict:
    """Pitch distribution (F-standard), notes per bar, common durations — the generator's reference."""
    notes = [n for s in songs for n in s.notes]
    c = Counter(n.pitch for n in notes)
    total = max(1, sum(c.values()))
    pitch_dist = {int(p): v / total for p, v in c.items()}
    # notes per bar: the median over songs (a chord-heavy tab must not set the pace for everyone)
    per_song = [len(s.notes) / max(1.0, s.n_bars) for s in songs if s.notes]
    density = float(np.median(per_song)) if per_song else 7.5
    fmap: dict[int, Counter] = {}
    for n in notes:
        if n.string is not None and n.fret is not None:
            fmap.setdefault(n.pitch, Counter())[(n.string, n.fret)] += 1
    return {"pitch_dist": pitch_dist, "density": density, "density_per_song": per_song,
            "durations": {float(k): v / total for k, v in Counter(round(n.duration) for n in notes).items()},
            "fingering": {int(p): list(cnt.most_common(1)[0][0]) for p, cnt in fmap.items()},
            "bpm": float(np.median([s.bpm for s in songs])) if songs else config.BPM}


# ----------------------------------------------------------------------------------------------
# the shared read-out
# ----------------------------------------------------------------------------------------------
def fit_multi(corpus: list[CorpusSong] | None = None, *, lam: float = 2e-4, spectral_radius: float = 1.6,
              seed: int = 0, input_frac: float = 0.3, save_to: Path = MODEL_MULTI_PATH,
              bars_per_song: int = 16, expand: int = 6000, verbose: bool = True) -> tuple[ComposerModel, dict]:
    """One read-out for every song in the corpus: accumulate FᵀF and FᵀY song by song, solve once.
    `bars_per_song` > 0 fits each song's most characteristic riffs (see :func:`riff_excerpt`) instead
    of the whole song — the read-out's capacity is ~one song's worth of material."""
    from .sonify import decode_notes, diff_notes
    from .train_supervised import ridge_fit, targets_for
    corpus = corpus or load_corpus(verbose=verbose)
    if not corpus:
        raise RuntimeError("no songs — put Guitar Pro files or tab PDFs in data/songs/")
    if bars_per_song > 0:
        for cs in corpus:
            full = cs.song
            cs.song = riff_excerpt(full, bars_per_song)
            if verbose and cs.song is not full:
                print(f"[fit-multi] {cs.key}: riff excerpt {cs.song.n_bars} bars / {len(cs.song.notes)} notes "
                      f"({', '.join(cs.song.meta.get('excerpt_sections', []))[:90]})", flush=True)
    neurons, edges = connectome.load_subgraph()
    t0 = time.time()
    first = corpus[0].song
    streams0 = build_streams_ex(first, form_mode="code", song_key=corpus[0].key)
    W = connectome.build_weight_matrix(neurons, edges, spectral_radius=spectral_radius)
    dans = connectome.dan_mask(neurons)
    gains = {"drums": 4.0, "riff": 4.0, "form": 2.4}
    imap = make_input_map(len(neurons), streams0, frac=input_frac, seed=seed, gains=gains, exclude=dans)
    model = ComposerModel(W=W, input_map=imap, reservoir_params=ReservoirParams(seed=seed), spectral_radius=spectral_radius,
                          meta={"n_neurons": len(neurons), "n_edges": len(edges), "seed": seed,
                                "dataset": config.NEUPRINT_DATASET, "song": "corpus", "bpm": float(first.bpm),
                                "form_mode": "code", "code_dim": SECTION_CODE_DIM, "expand": int(expand)})
    if expand > 0:
        # a linear read-out of 2,318 states holds about one song; random tanh features of the *same*
        # states give the reader more capacity (it is still reading the brain, with a bigger pen)
        model.expansion = ComposerModel.make_expansion(len(neurons), expand, seed=seed + 7)
    res = model.reservoir()
    G = None
    B = None
    per_song = []
    layout = None
    for cs in corpus:
        U = stack_streams(build_streams_ex(cs.song, form_mode="code", song_key=cs.key))
        x0 = res.washout_state(U, 400)
        X = res.run(U, x0=x0)
        Y, layout = targets_for(cs.song)
        T = min(len(X), len(Y))
        F = model.features(X[:T], U[:T])
        G = F.T @ F if G is None else G + F.T @ F
        B = F.T @ Y[:T] if B is None else B + F.T @ Y[:T]
        per_song.append((cs, F, Y[:T]))
        if verbose:
            print(f"[fit-multi] {cs.key}: {T} steps driven", flush=True)
    G[np.diag_indices_from(G)] += lam
    from scipy.linalg import cho_factor, cho_solve
    W_out = cho_solve(cho_factor(G, lower=True, check_finite=False), B, check_finite=False)
    model.W_out, model.layout, model.ridge_lambda = W_out, layout, lam
    report = {"songs": {}, "seconds": time.time() - t0}
    for cs, F, Y in per_song:
        rep = diff_notes(decode_notes(F @ W_out, layout), cs.song.notes, bpm=cs.song.bpm)
        cs.stageA = {k: float(v) for k, v in rep.items() if isinstance(v, (int, float, np.floating, np.integer))}
        report["songs"][cs.key] = cs.stageA
        if verbose:
            print(f"[fit-multi] {cs.key:28s} onset F1 {rep['f1']:.3f}  pitch acc {rep['pitch_accuracy']:.3f}  "
                  f"({rep['n_pred']}/{rep['n_truth']} notes)", flush=True)
    stats = corpus_stats([cs.song for cs in corpus])
    sections_meta = {}
    for cs in corpus:
        for name in sorted({sec.name for sec in cs.song.sections}):
            secs = [sec for sec in cs.song.sections if sec.name == name]
            ns = [n for sec in secs for n in cs.song.notes if sec.start <= n.start < sec.end]
            bars = max(1.0, sum(sec.end - sec.start for sec in secs) / 16.0)
            if not ns:
                continue
            c = Counter(n.pitch for n in ns)
            starts = sorted(r for sec in secs for r in cs.song.riff_starts if sec.start <= r < sec.end)
            gaps = [b - a for a, b in zip(starts, starts[1:]) if 0 < b - a <= 64]
            sections_meta[f"{cs.key}|{name}"] = {
                "pitch_dist": {int(p_): v / len(ns) for p_, v in c.items()},
                "density": len(ns) / bars, "bars": bars, "song": cs.key, "name": name,
                "cycle16": int(round(float(np.median(gaps)))) if gaps else None,
                "stageA_f1": cs.stageA.get("f1"),
            }
    model.meta.update({
        "multi": {
            "songs": [{"key": cs.key, "path": cs.path, "title": cs.song.title, "bpm": cs.song.bpm,
                       "n_notes": len(cs.song.notes), "transpose": cs.transpose, "bars": cs.song.n_bars,
                       "sections": sorted({s.name for s in cs.song.sections}), "stageA": cs.stageA} for cs in corpus],
            "stats": stats, "bars_per_song": bars_per_song, "sections": sections_meta,
        },
        "stageA": {"f1": float(np.mean([r["f1"] for r in report["songs"].values()])),
                   "pitch_accuracy": float(np.mean([r["pitch_accuracy"] for r in report["songs"].values()])),
                   "seconds": report["seconds"]},
    })
    model.save(save_to)
    CORPUS_INDEX.write_text(json.dumps(model.meta["multi"], indent=1, default=float))
    if verbose:
        print(f"[fit-multi] {len(corpus)} songs, mean onset F1 {model.meta['stageA']['f1']:.3f} -> {save_to}  "
              f"[{report['seconds']:.0f}s]", flush=True)
    return model, report


def learned_codes(model: ComposerModel) -> dict[str, np.ndarray]:
    """{'song|section': code} for every section the shared read-out was trained on."""
    dim = int(model.meta.get("code_dim", SECTION_CODE_DIM))
    out = {}
    for s in model.meta.get("multi", {}).get("songs", []):
        for name in s["sections"]:
            out[f"{s['key']}|{name}"] = section_code(f"{s['key']}|{name}", dim)
    return out


# ----------------------------------------------------------------------------------------------
# a song that does not exist yet: the frame the generator improvises into
# ----------------------------------------------------------------------------------------------
def synthetic_song(bpm: float, phrases: int, phrase_bars: int, cycle16: int, title: str = "Fruit Fly Djent") -> Song:
    """Empty-guitar song with a basic kick/snare grid (so the timeline has a length), one section per
    phrase (named gen0, gen1, ...) and riff cycles of `cycle16` sixteenths."""
    L = phrase_bars * 16
    n16 = phrases * L
    drums = []
    for bar in range(phrases * phrase_bars):
        b = bar * 16
        drums += [Note(b, 1, 36, 110), Note(b + 8, 1, 38, 110)]
    sections = [Section(f"gen{k}", k * L, (k + 1) * L) for k in range(phrases)]
    riff_starts = [float(t) for t in np.arange(0, n16, cycle16)]
    return Song(title, bpm, notes=[], drums=drums, sections=sections, riff_starts=riff_starts,
                tuning=config.TUNING_F_STANDARD, meta={"synthetic": True, "cycle16": cycle16})


def riff_drums(notes: list[Note], s16_0: float, s16_1: float, cycle16: int | None = None,
               snare: str = "24") -> list[Note]:
    """Djent drums that lock to the riff:

    * hats keep the tempo — closed hat on every 8th, louder on the quarters (the pulse the kick plays
      against); a crash on the phrase downbeat;
    * the kick follows the riff — one kick under every guitar onset;
    * the snare is the backbeat — ``snare="24"``: beats 2 and 4 of every bar; ``snare="thirds"``: every
      3 sixteenths from the phrase start (a 3/16 snare against 4/4, the djent displacement feel).
    """
    out = [Note(s16_0, 1, 49, 115)]
    for t in sorted({round(n.start) for n in notes if s16_0 <= n.start < s16_1}):
        out.append(Note(float(t), 1, 36, 118))
    t = s16_0
    while t < s16_1:
        beat = int(round(t - s16_0)) % 16
        if beat % 2 == 0:
            out.append(Note(float(t), 1, 42, 95 if beat % 4 == 0 else 70))
        if snare == "thirds":
            if int(round(t - s16_0)) % 3 == 0:
                out.append(Note(float(t), 1, 38, 112))
        elif beat in (4, 12):
            out.append(Note(float(t), 1, 38, 112))
        t += 1.0
    return out