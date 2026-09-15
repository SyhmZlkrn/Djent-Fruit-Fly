"""Turn a generated riff into a stage *performance folder*: notes/drums/sections/song JSON, guitar (and
optional drum) stems as MP3, and the spike raster of the brain hearing that riff — everything the three.js
stage loads besides the fly, brain and guitar geometry. Used by ``hf_space/build_stage.py`` (a fixed riff on a
static page) and by the Space (each visitor's riff, loaded with ``index.html?run=<folder>``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config
from .transcription import Note, Section, Song


def riff_to_song(gen: dict, phrase_bars: int, cycle16: int, seed: int, fmap: dict | None = None, title: str | None = None) -> tuple[Song, list[Note]]:
    """Rebuild a Song (+ fingered note list) from a `generate_riffs` result via its MIDI file."""
    import pretty_midi
    from .live import finger
    pm = pretty_midi.PrettyMIDI(str(gen["midi"]))
    sec16 = config.sixteenth_seconds(gen["bpm"])
    notes, drums = [], []
    for inst in pm.instruments:
        for n in inst.notes:
            start16, dur16 = n.start / sec16, max(0.25, (n.end - n.start) / sec16)
            if inst.is_drum:
                drums.append(Note(round(start16 * 4) / 4, 1.0, n.pitch, n.velocity))
            else:
                notes.append(Note(round(start16 * 4) / 4, round(dur16 * 4) / 4, n.pitch, n.velocity))
    finger(notes, [], fmap or {})
    L16 = phrase_bars * 16
    phrases = gen["phrases"]
    sections = [Section((p["source"] or f"riff {p['phrase'] + 1}").split(" + ")[0].replace("|", " · ")[:40],
                        p["phrase"] * L16, (p["phrase"] + 1) * L16) for p in phrases]
    song = Song(title or f"Fruit Fly Djent — riff seed {seed}", gen["bpm"], notes, drums=drums, sections=sections,
                riff_starts=[float(t) for t in np.arange(0, len(phrases) * L16, cycle16)],
                tuning=config.TUNING_F_STANDARD, meta={"synthetic": True, "cycle16": cycle16, "generated": True})
    return song, notes


def export_run(gen: dict, run_dir: Path, *, phrase_bars: int, cycle16: int, seed: int, model=None,
               drums: bool = False, spikes: bool = True, sr: int = 48000, log=print) -> dict:
    """Write a performance folder for the stage. Returns a small manifest."""
    import soundfile as sf
    from .bridgelite import render_drums
    from .live import PhraseRenderer
    from .model import ComposerModel
    from .corpus import MODEL_MULTI_PATH
    from .stage_export import export_song

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model = model or ComposerModel.load(MODEL_MULTI_PATH)
    stats = model.meta.get("multi", {}).get("stats", {})
    fmap = {int(p): tuple(v) for p, v in stats.get("fingering", {}).items()}
    song, notes = riff_to_song(gen, phrase_bars, cycle16, seed, fmap)
    export_song(song, notes, out_dir=run_dir, with_audio=False)
    L16 = phrase_bars * 16
    rend = PhraseRenderer(song.bpm, total_s=song.seconds + 3.0, sr=sr)
    guitar = np.zeros((int((song.seconds + 3.0) * sr) + sr, 2), np.float32)
    for k in range(len(gen["phrases"])):
        s0, s1 = k * L16, (k + 1) * L16
        off, chunk = rend.render([n for n in notes if s0 <= n.start < s1], s0, s1)
        e = min(len(guitar), off + len(chunk))
        guitar[off:e] += chunk[: e - off]
    end = int(min(len(guitar), (song.seconds + 1.5) * sr))
    stems = ["guitar"]
    sf.write(str(run_dir / "stem_guitar.mp3"), guitar[:end], sr, format="MP3", subtype="MPEG_LAYER_III")
    mix = guitar[:end].copy()
    if drums and song.drums:
        d = render_drums(song.drums, sr, song.bpm) * 0.4
        drum_stem = np.zeros_like(guitar)
        drum_stem[: min(len(drum_stem), len(d))] = d[: len(drum_stem)]
        sf.write(str(run_dir / "stem_drums.mp3"), drum_stem[:end], sr, format="MP3", subtype="MPEG_LAYER_III")
        mix = mix + drum_stem[:end]
        stems.append("drums")
    sf.write(str(run_dir / "song.mp3"), np.clip(mix, -1, 1), sr, format="MP3", subtype="MPEG_LAYER_III")
    (run_dir / "stems.json").write_text(json.dumps({"stems": stems, "mix": True, "bpm": song.bpm}))
    n_spikes = 0
    if spikes:
        from .meter import build_streams_ex, stack_streams
        from .snn import LIFParams, NumpyLIF, grid_dt_ms, meter_current, onset_current, onset_population, recurrent_weights
        p = LIFParams()
        We, Wi = recurrent_weights(model, p)
        U = stack_streams(build_streams_ex(song, form_mode="code", song_key="gen"))
        I = meter_current(model, song, p, U=U)
        aud, amps = onset_population(model.W.shape[0])
        I += onset_current(notes, len(I), model.W.shape[0], aud, amps, p)
        lif = NumpyLIF(We, Wi, I, p, grid_dt_ms=grid_dt_ms(song))
        lif.run_until(song.seconds * 1000.0)
        r = lif.raster()
        order = np.argsort(r["t_ms"].to_numpy(), kind="stable")
        (r["t_ms"].to_numpy()[order] / 1000.0).astype(np.float32).tofile(run_dir / "spikes_t.bin")
        r["neuron"].to_numpy()[order].astype(np.uint16).tofile(run_dir / "spikes_n.bin")
        n_spikes = int(len(r))
        log(f"[stage] {n_spikes} spikes for {song.seconds:.0f} s of riff ({n_spikes / max(1e-9, song.seconds) / model.W.shape[0]:.1f} Hz/neuron)")
    return {"seconds": song.seconds, "n_notes": len(notes), "n_spikes": n_spikes, "stems": stems, "dir": str(run_dir)}
