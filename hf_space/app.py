"""FlyBrain Composer — riff generator demo (Hugging Face Space).

A fruit fly's connectome (MaleCNS, 2,318 central-brain neurons, wiring never modified) is used as a
fixed reservoir; a read-out fitted to riffs from djent tabs decodes its activity into notes. Here the
brain is driven with a bar grid, a riff cycle and a blend of learned section identities, and a small
reward-driven search over nine musical knobs picks the groove — the fly makes up a riff.

The live show (real-time audio, spiking brain window, the fly playing an Ibanez M8M on a three.js
stage, 👍/👎 from the audience) needs a desktop: https://github.com/SyhmZlkrn/Djent-Fruit-Fly
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("OMP_NUM_THREADS", "2")

import spaces  # noqa: E402  — ZeroGPU: the platform requires a @spaces.GPU function (the maths is NumPy either way)
import gradio as gr  # noqa: E402

from flybrain_composer.generate import generate_riffs  # noqa: E402

CORPUS = json.loads((HERE / "data" / "cache" / "corpus.json").read_text(encoding="utf-8")) if (HERE / "data" / "cache" / "corpus.json").exists() else {}
SONGS = CORPUS.get("songs", [])
CYCLES = [15, 17, 19, 21, 23, 25, 27, 29, 31]


def _seconds_needed(bars, phrase_bars, bpm, cycle, snare, density, wildness, generations, seed) -> int:
    """ZeroGPU charges the visitor's quota for the time requested: ask only for what this run needs
    (~1 s per bar per search generation on the Space, plus load and render)."""
    return int(min(120, max(20, 12 + 1.0 * float(bars) * float(generations))))


@spaces.GPU(duration=_seconds_needed)
def run(bars, phrase_bars, bpm, cycle, snare, density, wildness, generations, seed):
    t0 = time.time()
    out_dir = Path(tempfile.mkdtemp(prefix="flybrain_"))

    def cb(k, n, msg):
        print(f"[space] riff {k}/{n}: {msg}", flush=True)

    out = generate_riffs(bars=int(bars), phrase_bars=int(phrase_bars), bpm=float(bpm), cycle16=int(cycle),
                         snare=snare, density=float(density), wildness=float(wildness),
                         generations=int(generations), seed=int(seed), out_dir=out_dir, progress=cb,
                         tag=f"{int(bpm)}bpm_{int(cycle)}of16_seed{int(seed)}")
    rows = [[p["phrase"] + 1, (p["source"] or "").replace("|", " · ").replace(" + ", "  +  "), p["n_notes"],
             round(p["fitness"], 3), round(p["groove"], 2), round(p["density"], 1),
             ", ".join(str(k) for k in p["pitches"])] for p in out["phrases"]]
    summary = (f"{out['n_notes']} notes · {out['bars']} bars at {out['bpm']:.0f} BPM · riff cycle {int(cycle)}/16 · "
               f"snare {'2 & 4' if snare == '24' else 'thirds'} · {time.time() - t0:.0f} s of compute")
    return str(out["wav"]), str(out["midi"]), summary, rows


with gr.Blocks(title="FlyBrain Composer — a fruit fly's brain makes up djent riffs") as demo:
    gr.Markdown(
        "# 🪰 FlyBrain Composer — riffs from a fruit fly's brain\n"
        "The real synaptic wiring of a male fruit fly ([MaleCNS](https://neuprint.janelia.org), 2,318 central-brain "
        "neurons, never modified) is used as a fixed reservoir. A read-out fitted to riffs from "
        f"{len(SONGS) or 'a set of'} djent tabs decodes its activity into notes; here the brain is driven with a bar "
        "grid, a riff cycle of your choice and a blend of learned riff identities, and a small reward-driven search "
        "over nine musical knobs picks the groove. **Every seed is a different take.** Drums lock to the riff: hats "
        "keep time, the kick doubles every chug, the snare sits on 2 & 4 or in thirds.\n\n"
        "The full show — real-time audio, the spiking brain, the fly playing an Ibanez M8M, 👍/👎 from the audience — "
        "runs on a desktop: [github.com/SyhmZlkrn/Djent-Fruit-Fly](https://github.com/SyhmZlkrn/Djent-Fruit-Fly). "
        "Honest note: a shuffled or random network of the same size reproduces songs just as well — the connectome is "
        "the medium the riffs are written on, not the composer.\n\n"
        "*Runs on Hugging Face ZeroGPU: each riff uses about a minute of your free daily GPU quota "
        "(log in to Hugging Face for more).*"
    )
    with gr.Row():
        with gr.Column(scale=1):
            bars = gr.Slider(8, 24, value=16, step=4, label="bars  (16 bars × 3 generations ≈ 1 minute of quota)")
            phrase_bars = gr.Radio([2, 4, 8], value=4, label="bars per riff")
            bpm = gr.Slider(90, 200, value=140, step=1, label="BPM")
            cycle = gr.Dropdown(CYCLES, value=23, label="riff cycle (sixteenths) — the polymeter against 4/4")
            snare = gr.Radio([("2 & 4", "24"), ("thirds (every 3 sixteenths)", "thirds")], value="24", label="snare")
            density = gr.Slider(5, 14, value=9, step=0.5, label="notes per bar")
            wildness = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="wildness (how far the knobs may stray)")
            generations = gr.Slider(2, 6, value=3, step=1, label="search generations per riff (more = slower, groovier)")
            seed = gr.Number(value=0, precision=0, label="seed")
            btn = gr.Button("🎸 Make a riff", variant="primary")
        with gr.Column(scale=1):
            audio = gr.Audio(label="the fly's riff (8ridge lite sampler port + amp, its own drums)", type="filepath")
            midi = gr.File(label="MIDI (guitar + drums)")
            summary = gr.Markdown()
            table = gr.Dataframe(headers=["riff", "learned from", "notes", "score", "groove", "notes/bar", "pitches (MIDI)"],
                                 datatype=["number", "str", "number", "number", "number", "number", "str"],
                                 label="what each riff was made from", wrap=True)
    btn.click(run, [bars, phrase_bars, bpm, cycle, snare, density, wildness, generations, seed], [audio, midi, summary, table])
    if SONGS:
        gr.Markdown("**Learned from (16 most characteristic bars of each):** " +
                    " · ".join(sorted({s["title"] for s in SONGS})))
    gr.Markdown(
        "**Attribution.** Connectome: MaleCNS v1.0 (Janelia FlyEM, CC BY 4.0) via neuPrint. Guitar samples and sampler "
        "logic: [8ridge lite](https://github.com/JamesStubbsEng/8ridgelite) by James Stubbs (GPL-3.0), trimmed to 4 s "
        "here. Code: GPL-3.0. The tabs the read-out learned from are not distributed."
    )

if __name__ == "__main__":
    demo.queue(max_size=8).launch()
