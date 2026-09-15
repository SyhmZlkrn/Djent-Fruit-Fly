"""Build a self-contained *static* stage: the NeuroMechFly plays a riff the fly made up, with its spiking
brain, in the browser — no Python at show time. Free to host (static Hugging Face Space / GitHub Pages).

    python hf_space/build_stage.py --seed 7 --bars 48 --bpm 140 --cycle 23 [--out build/stage_static]

Steps: generate the riff (or reuse output/flybrain_djent_<tag>.*), export notes/drums/sections/song for
the stage, render guitar + drums stems to mp3, run the numpy LIF on the multi-song model's inputs for the
spike raster, copy the fly / guitar / brain / region assets from stage/data (built by `stage-export`), copy
the page with STATIC_STAGE set (hides the conductor / live-learning / generator buttons, which need Python).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flybrain_composer import config  # noqa: E402
from flybrain_composer.corpus import MODEL_MULTI_PATH, synthetic_song  # noqa: E402
from flybrain_composer.generate import generate_riffs  # noqa: E402
from flybrain_composer.transcription import Note, Section, Song  # noqa: E402

README = """---
title: FlyBrain Composer — the fly plays a riff it made up
emoji: 🎸
colorFrom: red
colorTo: purple
sdk: static
app_file: index.html
pinned: false
license: gpl-3.0
short_description: A scanned fruit fly plays a riff its own brain made up
---

# 🎸 FlyBrain Composer — the stage

A NeuroMechFly body (EPFL's micro-CT scan of a real fly, 70 articulated parts) plays a riff that a fruit fly's
connectome made up (MaleCNS, 2,318 central-brain neurons used as a fixed reservoir; a read-out fitted to riffs
from 19 djent tabs; nine musical knobs searched by reward). The brain above the stage is the same 2,318 neurons
spiking in a leaky integrate-and-fire simulation of the real wiring, replayed in sync — yellow = spiking now,
red = just spiked, blue = resting, magenta = the dopaminergic PAM/PPL1 neurons. Press ▶ to play (browser
audio). The live version — real-time improvisation, 👍/👎 from the audience, the real song — needs a desktop:
**https://github.com/SyhmZlkrn/Djent-Fruit-Fly** · riff generator: https://huggingface.co/spaces/SyhmZlkrn/djent-fruit-fly

Attribution: MaleCNS v1.0 (Janelia FlyEM, CC BY 4.0) · NeuroMechFly (EPFL, Apache-2.0) · "Ibanez M8M"
(https://skfb.ly/pqrZM) by cherepah (CC BY 4.0) · 8ridge lite by James Stubbs (GPL-3.0) · three.js (MIT). Code GPL-3.0.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "build" / "stage_static"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--bars", type=int, default=48)
    ap.add_argument("--phrase-bars", type=int, default=4)
    ap.add_argument("--bpm", type=float, default=140.0)
    ap.add_argument("--cycle", type=int, default=23)
    ap.add_argument("--snare", default="24")
    ap.add_argument("--density", type=float, default=9.0)
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--drums", action="store_true", help="include the synthesized drum kit (default: the fly's guitar alone)")
    a = ap.parse_args()
    import pretty_midi
    import soundfile as sf
    from flybrain_composer.bridgelite import render_drums
    from flybrain_composer.live import PhraseRenderer, finger
    from flybrain_composer.model import ComposerModel
    from flybrain_composer.stage_export import export_song

    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    data = out / "data"
    data.mkdir(parents=True)

    # 1) the riff
    tag = f"stage_seed{a.seed}"
    gen = generate_riffs(bars=a.bars, phrase_bars=a.phrase_bars, bpm=a.bpm, cycle16=a.cycle, snare=a.snare,
                         density=a.density, wildness=0.5, generations=a.generations, seed=a.seed, tag=tag, drums=True)
    pm = pretty_midi.PrettyMIDI(str(gen["midi"]))
    sec16 = config.sixteenth_seconds(gen["bpm"])
    model = ComposerModel.load(MODEL_MULTI_PATH)
    stats = model.meta.get("multi", {}).get("stats", {})
    fmap = {int(p): tuple(v) for p, v in stats.get("fingering", {}).items()}
    notes, drums = [], []
    for inst in pm.instruments:
        for n in inst.notes:
            start16, dur16 = n.start / sec16, max(0.25, (n.end - n.start) / sec16)
            if inst.is_drum:
                drums.append(Note(round(start16 * 4) / 4, 1.0, n.pitch, n.velocity))
            else:
                notes.append(Note(round(start16 * 4) / 4, round(dur16 * 4) / 4, n.pitch, n.velocity))
    finger(notes, [], fmap)
    phrases = gen["phrases"]
    L16 = a.phrase_bars * 16
    sections = [Section((p["source"] or f"riff {p['phrase'] + 1}").split(" + ")[0].replace("|", " · ")[:40],
                        p["phrase"] * L16, (p["phrase"] + 1) * L16) for p in phrases]
    song = Song(f"FlyBrain djent — riff seed {a.seed}", gen["bpm"], notes, drums=drums, sections=sections,
                riff_starts=[float(t) for t in np.arange(0, len(phrases) * L16, a.cycle)],
                meta={"synthetic": True, "cycle16": a.cycle, "generated": True})

    # 2) notes / drums / sections / song for the page (no audio here: stems are rendered below)
    export_song(song, notes, out_dir=data, with_audio=False)

    # 3) stems: guitar (double-tracked sampler port) and its own drums
    sr = 48000
    rend = PhraseRenderer(song.bpm, total_s=song.seconds + 3.0, sr=sr)
    guitar = np.zeros((int((song.seconds + 3.0) * sr) + sr, 2), np.float32)
    for k in range(len(phrases)):
        s0, s1 = k * L16, (k + 1) * L16
        off, chunk = rend.render([n for n in notes if s0 <= n.start < s1], s0, s1)
        e = min(len(guitar), off + len(chunk))
        guitar[off:e] += chunk[: e - off]
    end = int(min(len(guitar), (song.seconds + 1.5) * sr))
    stems = ["guitar"]
    sf.write(str(data / "stem_guitar.mp3"), guitar[:end], sr, format="MP3", subtype="MPEG_LAYER_III")
    mix = guitar[:end].copy()
    if a.drums:
        d = render_drums(drums, sr, song.bpm) * 0.4
        drum_stem = np.zeros_like(guitar)
        drum_stem[: min(len(drum_stem), len(d))] = d[: len(drum_stem)]
        sf.write(str(data / "stem_drums.mp3"), drum_stem[:end], sr, format="MP3", subtype="MPEG_LAYER_III")
        mix = mix + drum_stem[:end]
        stems.append("drums")
    sf.write(str(data / "song.mp3"), np.clip(mix, -1, 1), sr, format="MP3", subtype="MPEG_LAYER_III")
    (data / "stems.json").write_text(json.dumps({"stems": stems, "mix": True, "bpm": song.bpm}))

    # 4) the spiking brain hearing this riff: numpy LIF on the multi-song model's inputs
    from flybrain_composer.meter import build_streams_ex, stack_streams
    from flybrain_composer.snn import LIFParams, NumpyLIF, grid_dt_ms, meter_current, onset_current, onset_population, recurrent_weights
    p = LIFParams()
    We, Wi = recurrent_weights(model, p)
    U = stack_streams(build_streams_ex(song, form_mode="code", song_key="gen"))
    I = meter_current(model, song, p, U=U)
    aud, amps = onset_population(model.W.shape[0])
    I += onset_current(notes, len(I), model.W.shape[0], aud, amps, p)
    lif = NumpyLIF(We, Wi, I, p, grid_dt_ms=grid_dt_ms(song))
    print(f"[stage] LIF over {song.seconds:.0f} s …", flush=True)
    lif.run_until(song.seconds * 1000.0)
    r = lif.raster()
    order = np.argsort(r["t_ms"].to_numpy(), kind="stable")
    (r["t_ms"].to_numpy()[order] / 1000.0).astype(np.float32).tofile(data / "spikes_t.bin")
    r["neuron"].to_numpy()[order].astype(np.uint16).tofile(data / "spikes_n.bin")
    print(f"[stage] {len(r)} spikes ({len(r) / song.seconds / model.W.shape[0]:.1f} Hz/neuron)", flush=True)

    # 5) geometry from the regular stage export (fly, guitar, brain, regions)
    src = ROOT / "stage" / "data"
    for name in ("fly.glb", "fly_rig.json", "guitar.glb", "guitar.json", "brain.bin", "brain_owner.bin", "somas.bin",
                 "brain.json", "rois.glb"):
        if (src / name).exists():
            shutil.copy(src / name, data / name)
        else:
            print(f"[stage] missing {name} — run: python -m flybrain_composer.cli stage-export", flush=True)

    # 6) the page, flagged static
    for name in ("app.js", "fly.js", "guitar.js", "brain.js"):
        shutil.copy(ROOT / "stage" / name, out / name)
    shutil.copytree(ROOT / "stage" / "vendor", out / "vendor")
    html = (ROOT / "stage" / "index.html").read_text(encoding="utf-8")
    html = html.replace("<script type=\"importmap\">", "<script>window.STATIC_STAGE = true;</script>\n<script type=\"importmap\">", 1)
    html = html.replace("<h2>FlyBrain Composer — the stage</h2>", "<h2>FlyBrain Composer — a fly plays a riff its brain made up</h2>", 1)
    html = html.replace("<h1><span>FlyBrain Composer</span> — a fly plays Rational Gaze</h1>",
                        "<h1><span>FlyBrain Composer</span> — a fly plays a riff its brain made up</h1>", 1)
    html = html.replace("plays Meshuggah's <i>Rational Gaze</i> on an 8-string Ibanez M8M.",
                        f"plays a {a.bars}-bar riff that a fruit fly's connectome made up (a read-out over 19 djent tabs, "
                        f"nine musical knobs searched by reward) on an 8-string Ibanez M8M.", 1)
    html = html.replace("The notes were composed by a reservoir built from the MaleCNS connectome; the brain above the stage is the\n       same 2,318 neurons spiking in a Brian2 simulation, replayed in sync with the music.",
                        "The brain above the stage is the same 2,318 neurons spiking in a leaky integrate-and-fire simulation of the real wiring, replayed in sync.", 1)
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "README.md").write_text(README, encoding="utf-8")
    shutil.copy(ROOT / "LICENSE", out / "LICENSE")
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"[stage] static stage at {out}: {total:.0f} MB, {len(notes)} notes, {song.seconds:.0f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
