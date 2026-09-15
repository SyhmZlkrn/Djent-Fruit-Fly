"""Offline riff generator — the same brain, reader and knobs as ``play --generate``, with a fixed
candidate budget instead of the audio clock. Used by ``cli generate`` and by the Hugging Face Space.

    from flybrain_composer.generate import generate_riffs
    out = generate_riffs(bars=16, bpm=140, cycle16=23, snare="24", density=9, seed=0)
    out["wav"], out["midi"], out["phrases"]
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from . import config

SR = 48000


def generate_riffs(*, bars: int = 16, phrase_bars: int = 4, bpm: float | None = None, cycle16: int = 23,
                   snare: str = "24", density: float | None = 9.0, wildness: float = 0.5, generations: int = 6,
                   popsize: int = 8, seed: int = 0, out_dir: Path | None = None, model_path: Path | None = None,
                   progress=None, tag: str | None = None) -> dict:
    """Make up `bars` bars of riff. Returns {"midi", "wav", "json", "bpm", "phrases": [...]} with file paths.

    `progress(k, n_phrases, message)` is called after every phrase (for a UI)."""
    import cma
    import soundfile as sf
    from .bridgelite import render_drums
    from .corpus import MODEL_MULTI_PATH, learned_codes, riff_drums, synthetic_song
    from .live import PhraseRenderer, PhraseRunner, finger, wildness_settings
    from .model import ComposerModel
    from .sonify import notes_to_midi

    model_path = Path(model_path or MODEL_MULTI_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found — put tabs in data/songs/ and run: python -m flybrain_composer.cli fit-multi")
    out_dir = Path(out_dir or config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = ComposerModel.load(model_path)
    stats = dict(model.meta.get("multi", {}).get("stats", {}))
    if density:
        stats["density"] = float(density)
        stats["density_override"] = float(density)
    bpm = float(bpm or stats.get("bpm", config.BPM))
    phrase_bars = max(1, int(phrase_bars))
    phrases = max(1, int(np.ceil(bars / phrase_bars)))
    song = synthetic_song(bpm, phrases, phrase_bars, int(cycle16))
    song.meta.update({"seed": int(seed), "phrase_len16": phrase_bars * 16})
    wild = wildness_settings(wildness)
    wild["wildness"] = float(wildness)
    runner = PhraseRunner(model, song, song_key="gen", form_mode="code", settings=wild, mode="generate",
                          stats=stats, codes=learned_codes(model))
    fmap = {int(p): tuple(v) for p, v in stats.get("fingering", {}).items()}
    es = cma.CMAEvolutionStrategy(np.zeros(runner.dim), wild["sigma0"], {"popsize": int(popsize), "seed": int(seed) + 1, "verbose": -9})
    x, reward, running = runner.x0, 0.0, None
    L16 = phrase_bars * 16
    all_notes, all_drums, log = [], [], []
    prev_rel = None
    for k in range(phrases):
        s0, s1 = k * L16, min(song.n_sixteenths, (k + 1) * L16)
        best = None
        for _ in range(int(generations)):
            ths = es.ask()
            fits = []
            for th in ths:
                sc, notes, xe = runner.run_phrase(np.asarray(th), reward, s0, s1, x, prev_notes=prev_rel)
                fits.append(sc["fitness"])
                if best is None or sc["fitness"] > best[0]:
                    best = (sc["fitness"], sc, notes, xe)
            es.tell(ths, [-f for f in fits])
        f, sc, notes, x = best
        running = f if running is None else running
        reward = float(np.clip(f - running, -1, 1))
        running += 0.1 * (f - running)
        notes = finger(notes, [], fmap)
        all_notes += notes
        all_drums += riff_drums(notes, s0, s1, cycle16=int(cycle16), snare=snare)
        prev_rel = [{"start": n.start - s0, "duration": n.duration, "pitch": n.pitch, "velocity": n.velocity} for n in notes]
        try:
            es.sigma = max(es.sigma, 0.5 * wild["sigma0"])
        except Exception:  # noqa: BLE001
            pass
        pitches = {int(p): int(c) for p, c in sorted(Counter(n.pitch for n in notes).items())}
        entry = {"phrase": k, "fitness": float(f), "n_notes": len(notes), "groove": float(sc["groove"]),
                 "density": float(sc["raw"]["density"]), "source": sc.get("source"),
                 "repeat": sc["raw"].get("repeat"), "pitches": pitches}
        log.append(entry)
        msg = (f"phrase {k + 1}/{phrases}: {len(notes)} notes  fitness {f:.3f}  groove {sc['groove']:.2f}  "
               f"density {sc['raw']['density']:.1f}/bar  repeat {sc['raw'].get('repeat') or 0:.2f}  from {sc.get('source')}")
        if progress:
            progress(k + 1, phrases, msg)
        else:
            print(f"[generate] {msg}  pitches {pitches}", flush=True)
    tag = tag or f"seed{seed}"
    out_midi = out_dir / f"flybrain_djent_{tag}.mid"
    notes_to_midi(all_notes, out_midi, bpm=bpm, drums=all_drums)
    rend = PhraseRenderer(bpm, total_s=song.seconds + 3.0, sr=SR)
    mix = np.zeros((int((song.seconds + 3.0) * SR) + SR, 2), np.float32)
    for k in range(phrases):
        s0, s1 = k * L16, min(song.n_sixteenths, (k + 1) * L16)
        off, chunk = rend.render([n for n in all_notes if s0 <= n.start < s1], s0, s1)
        e = min(len(mix), off + len(chunk))
        mix[off:e] += chunk[: e - off]
    d = render_drums(all_drums, SR, bpm) * 0.4
    mix[: min(len(mix), len(d))] += d[: len(mix)]
    peak = float(np.abs(mix).max()) or 1.0
    if peak > 0.9:
        mix *= 0.9 / peak
    end = int(min(len(mix), (song.seconds + 1.5) * SR))
    out_wav = out_dir / f"flybrain_djent_{tag}.wav"
    sf.write(str(out_wav), mix[:end], SR)
    out_json = out_dir / f"flybrain_djent_{tag}.json"
    out_json.write_text(json.dumps({"bpm": bpm, "cycle16": int(cycle16), "snare": snare, "density": density,
                                    "wildness": wildness, "seed": seed, "phrases": log}, indent=1))
    return {"midi": out_midi, "wav": out_wav, "json": out_json, "bpm": bpm, "phrases": log,
            "n_notes": len(all_notes), "bars": phrases * phrase_bars}
