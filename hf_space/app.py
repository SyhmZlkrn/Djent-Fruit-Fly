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
from flybrain_composer.stage_run import export_run  # noqa: E402

RUNS = HERE / "runs"                      # one folder per generated riff: what the stage plays
RUNS.mkdir(exist_ok=True)
STAGE = HERE / "stage"                    # the three.js stage (fly, brain, guitar; loads a run with ?run=)
gr.set_static_paths(paths=[str(STAGE), str(RUNS)])

from flybrain_composer.fitness import DEFAULT_WEIGHTS  # noqa: E402

CORPUS = json.loads((HERE / "data" / "cache" / "corpus.json").read_text(encoding="utf-8")) if (HERE / "data" / "cache" / "corpus.json").exists() else {}
SONGS = CORPUS.get("songs", [])
CYCLES = [15, 17, 19, 21, 23, 25, 27, 29, 31]
HUMAN_GAIN, TASTE_RATE = 0.5, 0.5


def fresh_taste() -> dict:
    """Per-visitor reinforcement state: the scorer's weights (taste), the knob search's mean and step
    size, the treat carried to the next riff, the last riff's knobs and score components, vote counts."""
    return {"weights": {**DEFAULT_WEIGHTS, "fresh": 1.5}, "mean": None, "sigma": None, "reward": 0.0,
            "last_theta": None, "last_comps": None, "comp_mean": {}, "votes": {"up": 0, "down": 0}, "riffs": 0}


def taste_text(st: dict) -> str:
    base = {**DEFAULT_WEIGHTS, "fresh": 1.5}
    drift = []
    for k, w in st["weights"].items():
        r = w / base.get(k, 1.0)
        if r > 1.15:
            drift.append(f"{k} ↑{r:.1f}×")
        elif r < 0.87:
            drift.append(f"{k} ↓{r:.1f}×")
    v = st["votes"]
    lines = [f"**👍 {v['up']} · 👎 {v['down']}** over {st['riffs']} riff(s)"]
    lines.append("taste: " + (" · ".join(drift) if drift else "default — vote to teach it what you like"))
    if abs(st.get("reward", 0.0)) > 1e-6:
        lines.append(f"treat carried into the next riff: **{st['reward']:+.2f}** → {'PAM (reward)' if st['reward'] > 0 else 'PPL1 (punishment)'} neurons")
    return "  \n".join(lines)


def vote(value: float, st: dict):
    """👍/👎 on the last riff: adds to the treat for the next riff, re-weights the scorer toward the
    features of what you liked (multiplicative, clipped 0.2–4×), and nudges the search toward (👍) or
    widens it away from (👎) the knobs that made it — the same three effects as in the live show."""
    st = dict(st or fresh_taste())
    if st.get("last_comps") is None:
        return st, "make a riff first, then rate it"
    st["votes"] = dict(st["votes"]); st["votes"]["up" if value > 0 else "down"] += 1
    st["reward"] = float(max(-1.0, min(1.0, st.get("reward", 0.0) + HUMAN_GAIN * value)))
    comps, cm = st["last_comps"], st.get("comp_mean", {})
    st["weights"] = {c: (float(min(4.0, max(0.2, w * pow(2.718281828, TASTE_RATE * value * (comps[c] - cm[c])))))
                         if (c in comps and c in cm) else w) for c, w in st["weights"].items()}
    if st.get("last_theta") is not None:
        mean = st.get("mean") or [0.0] * len(st["last_theta"])
        if value > 0:
            st["mean"] = [0.7 * m + 0.3 * t for m, t in zip(mean, st["last_theta"])]
        else:
            st["sigma"] = float(min(2.0 * (st.get("sigma0") or 0.35), (st.get("sigma") or st.get("sigma0") or 0.35) * 1.15))
    return st, taste_text(st)


def _seconds_needed(bars, phrase_bars, bpm, cycle, snare, density, wildness, generations, seed, drums=False, st=None) -> int:
    """ZeroGPU charges the visitor's quota for the time requested: ask only for what this run needs
    (~1 s per bar per search generation, plus ~0.6 s per bar for the spiking brain and the stems)."""
    return int(min(150, max(25, 15 + 1.0 * float(bars) * float(generations) + 0.6 * float(bars))))


def _prune_runs(keep_seconds: float = 3600.0):
    now = time.time()
    for d in RUNS.iterdir():
        try:
            if d.is_dir() and now - d.stat().st_mtime > keep_seconds:
                for f in d.iterdir():
                    f.unlink()
                d.rmdir()
        except OSError:
            pass


def stage_html(run_id: str | None) -> str:
    if not run_id:
        return ('<div style="padding:14px;border:1px dashed #667;border-radius:10px;color:#889">The stage appears here after you make a riff: '
                'the NeuroMechFly plays it on the M8M while the same 2,318 neurons spike above it.</div>')
    from urllib.parse import quote
    src = (f"/gradio_api/file={quote(STAGE.as_posix())}/index.html"
           f"?run={quote('/gradio_api/file=' + quote(RUNS.as_posix()) + '/' + run_id + '/', safe='')}")
    return (f'<iframe src="{src}" style="width:100%;height:640px;border:0;border-radius:10px;background:#05060a" '
            f'allow="autoplay" title="the fly plays your riff"></iframe>'
            f'<div style="font-size:12px;color:#889;margin-top:6px">Press ▶ in the stage (browser audio). Cameras: Audience · Side · Fretboard · Brain · Orbit.</div>')


@spaces.GPU(duration=_seconds_needed)
def run(bars, phrase_bars, bpm, cycle, snare, density, wildness, generations, seed, drums=False, st=None):
    t0 = time.time()
    st = dict(st or fresh_taste())
    out_dir = Path(tempfile.mkdtemp(prefix="flybrain_"))

    def cb(k, n, msg):
        print(f"[space] riff {k}/{n}: {msg}", flush=True)

    out = generate_riffs(bars=int(bars), phrase_bars=int(phrase_bars), bpm=float(bpm), cycle16=int(cycle),
                         snare=snare, density=float(density), wildness=float(wildness),
                         generations=int(generations), seed=int(seed), out_dir=out_dir, progress=cb,
                         tag=f"{int(bpm)}bpm_{int(cycle)}of16_seed{int(seed)}", drums=bool(drums),
                         weights=st["weights"], theta0=st.get("mean"), sigma0=st.get("sigma"), reward0=float(st.get("reward", 0.0)))
    # remember what this riff was made of, so a vote can act on it
    n_ph = max(1, len(out["comps"]))
    st["last_comps"] = {c: sum(cm.get(c, 0.0) for cm in out["comps"]) / n_ph for c in out["comps"][0]} if out["comps"] else None
    st["last_theta"] = [sum(t[i] for t in out["thetas"]) / n_ph for i in range(len(out["thetas"][0]))] if out["thetas"] else None
    cm = dict(st.get("comp_mean", {}))
    for c, v in (st["last_comps"] or {}).items():
        cm[c] = v if c not in cm else 0.7 * cm[c] + 0.3 * v
    st["comp_mean"] = cm
    st["mean"], st["sigma"], st["sigma0"] = out["es_mean"], out["es_sigma"], out["sigma0"]
    st["reward"] = 0.0                       # the carried treat has been spent on this riff
    st["riffs"] = st.get("riffs", 0) + 1
    # the performance folder the stage loads: notes, stems, and the brain hearing this riff
    _prune_runs()
    run_id = f"r{int(time.time())}_{int(seed)}_{os.getpid()}"
    try:
        export_run(out, RUNS / run_id, phrase_bars=int(phrase_bars), cycle16=int(cycle), seed=int(seed), drums=bool(drums),
                   spikes=True, log=lambda m: print(m, flush=True))
    except Exception as e:  # noqa: BLE001 — the riff is still playable without the stage
        print(f"[space] stage export failed: {type(e).__name__}: {e}", flush=True)
        run_id = None
    rows = [[p["phrase"] + 1, (p["source"] or "").replace("|", " · ").replace(" + ", "  +  "), p["n_notes"],
             round(p["fitness"], 3), round(p["groove"], 2), round(p["density"], 1),
             ", ".join(str(k) for k in p["pitches"])] for p in out["phrases"]]
    summary = (f"{out['n_notes']} notes · {out['bars']} bars at {out['bpm']:.0f} BPM · riff cycle {int(cycle)}/16 · "
               f"{('drums, snare ' + ('2 & 4' if snare == '24' else 'thirds')) if drums else 'guitar only'} · {time.time() - t0:.0f} s of compute")
    return str(out["wav"]), str(out["midi"]), summary, rows, st, taste_text(st), stage_html(run_id)


with gr.Blocks(title="FlyBrain Composer — a fruit fly's brain makes up djent riffs") as demo:
    gr.Markdown(
        "# 🪰 FlyBrain Composer — riffs from a fruit fly's brain\n"
        "The real synaptic wiring of a male fruit fly ([MaleCNS](https://neuprint.janelia.org), 2,318 central-brain "
        "neurons, never modified) is used as a fixed reservoir. A read-out fitted to riffs from "
        f"{len(SONGS) or 'a set of'} djent tabs decodes its activity into notes; here the brain is driven with a bar "
        "grid, a riff cycle of your choice and a blend of learned riff identities, and a small reward-driven search "
        "over nine musical knobs picks the groove. **Every seed is a different take — and your 👍/👎 change the next one** "
        "(reinforcement: the vote is a treat into the fly's dopamine neurons, a re-weighting of the scorer, a nudge of the search). "
        "Guitar only by default; tick "
        "the box for drums that lock to the riff (hats keep time, the kick doubles every chug, snare on 2 & 4 or in thirds).\n\n"
        "After each riff the **NeuroMechFly plays it** below — a micro-CT scan of a real fly with real joints, fretting the "
        "real frets of an Ibanez M8M — while the same 2,318 neurons spike above it (a leaky integrate-and-fire simulation of "
        "the wiring, hearing this riff). The full live show — real-time improvisation on a song, 👍/👎 while it plays — runs on "
        "a desktop: [github.com/SyhmZlkrn/Djent-Fruit-Fly](https://github.com/SyhmZlkrn/Djent-Fruit-Fly). "
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
            drums = gr.Checkbox(value=False, label="add its own drums (hats keep time, kick follows the riff)")
            snare = gr.Radio([("2 & 4", "24"), ("thirds (every 3 sixteenths)", "thirds")], value="24", label="snare (with drums)")
            density = gr.Slider(5, 14, value=9, step=0.5, label="notes per bar")
            wildness = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="wildness (how far the knobs may stray)")
            generations = gr.Slider(2, 6, value=3, step=1, label="search generations per riff (more = slower, groovier)")
            seed = gr.Number(value=0, precision=0, label="seed")
            btn = gr.Button("🎸 Make a riff", variant="primary")
        with gr.Column(scale=1):
            audio = gr.Audio(label="the fly's riff (8ridge lite sampler port + amp)", type="filepath")
            midi = gr.File(label="MIDI")
            summary = gr.Markdown()
            with gr.Row():
                up = gr.Button("👍 treat", variant="secondary")
                down = gr.Button("👎 no treat", variant="secondary")
            taste = gr.Markdown("**👍 0 · 👎 0** — rate a riff and the next one changes: the vote becomes a treat injected into the fly's "
                                "dopaminergic PAM/PPL1 neurons, re-weights what the scorer rewards, and nudges the knob search.")
            table = gr.Dataframe(headers=["riff", "learned from", "notes", "score", "groove", "notes/bar", "pitches (MIDI)"],
                                 datatype=["number", "str", "number", "number", "number", "number", "str"],
                                 label="what each riff was made from", wrap=True)
    gr.Markdown("## 🪰 The fly plays it")
    stage = gr.HTML(stage_html(None))
    state = gr.State(fresh_taste())
    btn.click(run, [bars, phrase_bars, bpm, cycle, snare, density, wildness, generations, seed, drums, state],
              [audio, midi, summary, table, state, taste, stage])
    up.click(lambda st: vote(+1.0, st), [state], [state, taste])
    down.click(lambda st: vote(-1.0, st), [state], [state, taste])
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
