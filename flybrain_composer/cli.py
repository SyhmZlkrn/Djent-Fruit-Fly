"""Command line entry point.

    python -m flybrain_composer.cli pull            # neuPrint -> data/cache (neurons + edges)
    python -m flybrain_composer.cli skeletons       # NAVis skeletons + ROI meshes -> data/skeletons
    python -m flybrain_composer.cli fit             # Stage A ridge readout on the guitar part
    python -m flybrain_composer.cli compose         # replay -> output/flybrain_rational_gaze.mid
    python -m flybrain_composer.cli render          # MIDI -> 8ridge lite engine port -> .wav
    python -m flybrain_composer.cli stems           # stereo double-tracked guitar + backing track stems + mix
    python -m flybrain_composer.cli spikes          # Brian2 whole-song spiking simulation
    python -m flybrain_composer.cli shape           # Stage B reward-modulated improvisation
    python -m flybrain_composer.cli stage-export    # geometry/notes/spikes for the three.js stage
    python -m flybrain_composer.cli play            # audio + live NAVis brain + stage + MIDI out
    python -m flybrain_composer.cli play --improvise # live Stage B: the fly improvises phrase by phrase
    python -m flybrain_composer.cli snapshot 12.5   # offscreen brain render at t = 12.5 s
    python -m flybrain_composer.cli midi-ports      # list MIDI outputs (for --midi-out)
    python -m flybrain_composer.cli controls        # shuffled / random reservoirs vs the real wiring
    python -m flybrain_composer.cli corpus / fit-multi / generate   # many tabs -> one read-out -> riffs of its own
    python -m flybrain_composer.cli play --generate # live: the fly plays riffs it makes up; 👍/👎 steer it
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from . import config  # noqa: E402

MIDI_OUT = config.OUTPUT_DIR / "flybrain_rational_gaze.mid"
WAV_OUT = config.OUTPUT_DIR / "flybrain_rational_gaze_8ridgelite.wav"


def cmd_pull(a):
    from . import connectome
    n, e = connectome.pull_subgraph(n_top=a.n_top, force=a.force)
    print(connectome.summary(n, e))


def cmd_skeletons(a):
    from . import skeletons
    skeletons.build_cache(force=a.force)
    skeletons.prepare_render_cache(force=a.force)


def cmd_fit(a):
    from .train_supervised import fit
    from .transcription import load_song
    fit(load_song(a.gp, track=a.track), lam=a.lam, spectral_radius=a.rho, seed=a.seed)


def cmd_compose(a):
    from .model import ComposerModel, MODEL_B_PATH
    from .train_supervised import compose
    from .transcription import load_song
    model = ComposerModel.load(MODEL_B_PATH) if a.stage_b else None
    out = a.out or (config.OUTPUT_DIR / "flybrain_rational_gaze_improv.mid" if a.stage_b else MIDI_OUT)
    compose(model, load_song(a.gp, track=a.track), out_midi=out, dan_reward=a.reward)


def cmd_render(a):
    from .bridgelite import BridgeliteEngine, render_song
    from .sonify import midi_to_notes
    from .transcription import load_song
    song = load_song(a.gp, track=a.track)
    notes = song.notes if a.truth else midi_to_notes(a.midi or MIDI_OUT, bpm=song.bpm)[0]
    eng = BridgeliteEngine(sample_set="Tuned" if a.perfect else "Natural", mono=a.mono)
    mix, sr = render_song(song, notes, engine=eng, amp=not a.clean, drums=not a.no_drums,
                          out_path=a.out or WAV_OUT)
    print(f"[render] {len(mix) / sr:.1f} s -> {a.out or WAV_OUT}")


def cmd_spikes(a):
    from . import snn
    from .sonify import midi_to_notes
    from .transcription import load_song
    song = load_song(a.gp, track=a.track)
    notes = midi_to_notes(a.midi or MIDI_OUT, bpm=song.bpm)[0]
    snn.simulate(song, notes, backend=a.backend, duration_s=a.seconds)


def cmd_shape(a):
    from .train_reward import shape
    from .transcription import load_song
    shape(load_song(a.gp, track=a.track), generations=a.generations, popsize=a.popsize, bars=a.bars,
          seed=a.seed)


def cmd_stems(a):
    from .record import get_alignment, render_stems
    from .sonify import midi_to_notes
    from .transcription import load_song
    song = load_song(a.gp, track=a.track)
    notes = song.notes if a.truth else midi_to_notes(a.midi or MIDI_OUT, bpm=song.bpm)[0]
    if a.realign:
        get_alignment(song, notes, force=True)
    render_stems(song, notes, bpm=a.bpm)


def cmd_stage_export(a):
    from .stage_export import export_all
    export_all(gp=a.gp, track=a.track, midi=a.midi or MIDI_OUT, wav=a.wav or WAV_OUT, with_audio=not a.no_audio)


def cmd_play(a):
    from .player import Show, load_audio
    from .sonify import midi_to_notes
    from .transcription import load_song
    from .record import STEM_DIR, load_stems
    song = load_song(a.gp, track=a.track)
    notes = midi_to_notes(a.midi or MIDI_OUT, bpm=song.bpm)[0]
    audio, sr, gains = None, 48000, {}
    if not a.no_audio:
        if (STEM_DIR / "stems.json").exists() and not a.wav:
            st = load_stems()
            sr = st["sr"]
            audio = {k: st[k] for k in ("guitar", "drums", "backing") if k in st}
            has_backing = "backing" in audio
            gains = {"guitar": 0.0 if a.no_guitar else 1.0,
                     "backing": 0.0 if (a.fly_only or not has_backing) else 1.0,
                     "drums": 0.0 if (a.fly_only or (has_backing and not a.drums)) else 1.0}
        else:
            audio, sr = load_audio(a.wav or WAV_OUT)
        if a.midi_out and not a.keep_guitar:
            # the real plugin plays the guitar: mute the local guitar
            if isinstance(audio, dict):
                gains["guitar"] = 0.0
            else:
                from .bridgelite import render_drums
                audio = render_drums(song.drums, sr, song.bpm) * 0.7
    planner = None
    if a.generate:
        # The fly's own riffs: a synthetic song frame, the multi-song read-out, guitar + drums rendered live
        from .corpus import MODEL_MULTI_PATH, synthetic_song
        from .live import LivePlanner, PlannerConfig
        from .model import ComposerModel
        if not MODEL_MULTI_PATH.exists():
            raise SystemExit("no multi-song model yet — put tabs in data/songs/ and run: python -m flybrain_composer.cli fit-multi")
        stats = ComposerModel.load(MODEL_MULTI_PATH).meta.get("multi", {}).get("stats", {})
        bpm = a.bpm or float(stats.get("bpm", config.BPM))
        song = synthetic_song(bpm, a.phrases, a.phrase_bars, a.cycle)
        n = int((song.seconds + 3.0) * sr) + sr
        audio = {"guitar": np.zeros((n, 2), dtype=np.float32), "drums": np.zeros((n, 2), dtype=np.float32)}
        gains = {"guitar": 0.0 if (a.midi_out and not a.keep_guitar) else 1.0, "drums": 1.0 if a.drums else 0.0}
        cfg = PlannerConfig(mode="generate", phrase_bars=a.phrase_bars, popsize=a.popsize, sigma0=a.sigma, seed=a.seed,
                            start_s=a.start, margin_s=a.margin, enabled=True, wildness=a.wildness, sr=sr,
                            workers=a.workers, bpm=bpm, phrases=a.phrases, cycle16=a.cycle, snare=a.snare, density=a.density)
        print(f"[live] riff generator: {a.phrases} phrases of {a.phrase_bars} bars at {bpm:.0f} BPM, riff cycle {a.cycle}/16 — "
              f"starting the planner process…", flush=True)
        planner = LivePlanner(cfg)
        info = planner.wait_ready()
        print(f"[live] planner ready: wildness {info['wildness']:.2f} (sigma {info['sigma0']:.2f}), {info['workers']} workers; "
              f"thumbs up/down on the stage (K/J) or +/- in the brain window", flush=True)
        notes = []
        a.live = True
    elif a.improvise:
        # Live Stage B: the guitar stem is written phrase by phrase by the planner process
        from .live import LivePlanner, PlannerConfig
        if isinstance(audio, dict):
            n = int((song.seconds + 3.0) * sr) + sr
            audio["guitar"] = np.zeros((n, 2), dtype=np.float32)
            gains.setdefault("guitar", 1.0)
            if a.midi_out and not a.keep_guitar:
                gains["guitar"] = 0.0
        elif audio is not None:
            print("[live] --improvise needs the stems (run: stems); falling back to drums + live guitar", flush=True)
            from .bridgelite import render_drums
            n = int((song.seconds + 3.0) * sr) + sr
            audio = {"drums": render_drums(song.drums, sr, song.bpm) * 0.7,
                     "guitar": np.zeros((n, 2), dtype=np.float32)}
            gains = {"drums": 1.0, "guitar": 0.0 if (a.midi_out and not a.keep_guitar) else 1.0}
        cfg = PlannerConfig(mode="song", phrase_bars=a.phrase_bars, popsize=a.popsize, sigma0=a.sigma, seed=a.seed,
                            start_s=a.start, margin_s=a.margin, enabled=not a.improv_off,
                            wildness=a.wildness, base_midi=str(a.midi or MIDI_OUT), sr=sr,
                            gp=a.gp, track=a.track, workers=a.workers)
        print("[live] starting the planner process (loads the model, calibrates the guitar level)…", flush=True)
        planner = LivePlanner(cfg)
        info = planner.wait_ready()
        print(f"[live] planner ready: {info['n_phrases']} phrases of {info['phrase_bars']} bars, wildness {info['wildness']:.2f} "
              f"(sigma {info['sigma0']:.2f}, novelty band {info['novelty_band'][0]:.2f}-{info['novelty_band'][1]:.2f}, "
              f"density cap {info['density_max']:.2f}x); knobs: {', '.join(info['knobs'])}", flush=True)
        notes = []                                   # the phrases arrive from the planner
        a.live = True                                # the brain must hear what is actually played
    show = Show(song, notes, audio, sr, live=a.live, midi_out=a.midi_out, window=not a.no_window,
                stage=not a.no_stage, start_s=a.start, audio_gain=a.gain, gains=gains, planner=planner)
    show.run(duration=a.seconds)


def cmd_corpus(a):
    from .corpus import load_corpus
    songs = load_corpus(track=a.track)
    print(f"[corpus] {len(songs)} song(s) loaded from data/ and data/songs/", flush=True)


def cmd_fit_multi(a):
    from .corpus import fit_multi, load_corpus
    fit_multi(load_corpus(track=a.track), lam=a.lam, bars_per_song=a.bars_per_song, expand=a.expand)


def cmd_generate(a):
    """Offline: the fly makes up riffs (same planner logic, fixed candidate budget) -> MIDI + WAV."""
    from .generate import generate_riffs
    out = generate_riffs(bars=a.bars, phrase_bars=a.phrase_bars, bpm=a.bpm, cycle16=a.cycle, snare=a.snare,
                         density=a.density, wildness=a.wildness, generations=a.generations, seed=a.seed, drums=a.drums)
    print(f"[generate] {out['n_notes']} notes, {out['bars']} bars at {out['bpm']:.0f} BPM -> {out['midi']}  {out['wav']}", flush=True)


def cmd_controls(a):
    from .controls import run
    from .transcription import load_song
    run(load_song(a.gp, track=a.track))


def cmd_snapshot(a):
    from .neuroviz import snapshot
    p = snapshot(a.t, config.OUTPUT_DIR / f"brain_{a.t:.1f}s.png", view=a.view)
    print(p)


def cmd_midi_ports(a):
    from .player import list_midi_ports
    ports = list_midi_ports()
    print("\n".join(ports) if ports else "(no MIDI output ports — install loopMIDI to create a virtual one)")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):        # Windows consoles default to cp1252
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="flybrain_composer", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gp", help="Guitar Pro file (default: data/rational_gaze.gp)")
    ap.add_argument("--track", default="Rhythm", help="guitar track to learn (name substring or index)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pull"); p.add_argument("--n-top", type=int, default=config.SUBGRAPH_N_TOP); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_pull)
    p = sub.add_parser("skeletons"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_skeletons)
    p = sub.add_parser("fit"); p.add_argument("--lam", type=float, default=2e-4); p.add_argument("--rho", type=float, default=1.6); p.add_argument("--seed", type=int, default=0); p.set_defaults(fn=cmd_fit)
    p = sub.add_parser("compose"); p.add_argument("--out"); p.add_argument("--stage-b", action="store_true", help="use the Stage B (improvising) model"); p.add_argument("--reward", type=float, default=0.5, help="scalar reward injected into the dopaminergic neurons (Stage B)"); p.set_defaults(fn=cmd_compose)
    p = sub.add_parser("render"); p.add_argument("--midi"); p.add_argument("--out"); p.add_argument("--clean", action="store_true", help="raw DI, no amp"); p.add_argument("--no-drums", action="store_true"); p.add_argument("--perfect", action="store_true", help="use the 'Tuned' sample set"); p.add_argument("--mono", type=int, default=0); p.add_argument("--truth", action="store_true", help="render the transcription itself"); p.set_defaults(fn=cmd_render)
    p = sub.add_parser("spikes"); p.add_argument("--midi"); p.add_argument("--backend", default="brian2", choices=["brian2", "numpy"]); p.add_argument("--seconds", type=float); p.set_defaults(fn=cmd_spikes)
    p = sub.add_parser("shape"); p.add_argument("--generations", type=int, default=30); p.add_argument("--popsize", type=int, default=8); p.add_argument("--bars", type=int, default=16); p.add_argument("--seed", type=int, default=0); p.set_defaults(fn=cmd_shape)
    p = sub.add_parser("stems", help="double-tracked guitar + drums + aligned backing stems, and the mix"); p.add_argument("--midi"); p.add_argument("--truth", action="store_true"); p.add_argument("--bpm", type=float); p.add_argument("--realign", action="store_true"); p.set_defaults(fn=cmd_stems)
    p = sub.add_parser("stage-export"); p.add_argument("--midi"); p.add_argument("--wav"); p.add_argument("--no-audio", action="store_true"); p.set_defaults(fn=cmd_stage_export)
    p = sub.add_parser("play"); p.add_argument("--midi"); p.add_argument("--wav"); p.add_argument("--live", action="store_true", help="step the numpy LIF in lockstep instead of replaying the Brian2 raster"); p.add_argument("--midi-out", help="MIDI output port name (drives the real plugin)"); p.add_argument("--keep-guitar", action="store_true"); p.add_argument("--no-window", action="store_true"); p.add_argument("--no-stage", action="store_true"); p.add_argument("--no-audio", action="store_true"); p.add_argument("--start", type=float, default=0.0); p.add_argument("--seconds", type=float); p.add_argument("--gain", type=float, default=1.0); p.add_argument("--fly-only", action="store_true", help="mute the backing track and drums: the fly's guitar alone"); p.add_argument("--no-guitar", action="store_true", help="mute the fly's guitar stem"); p.add_argument("--drums", action="store_true", help="add the synthesized drum kit even when a backing track exists"); p.add_argument("--improvise", action="store_true", help="live Stage B: improvise phrase by phrase while performing (reward into the DANs, CMA-ES between phrases)"); p.add_argument("--phrase-bars", type=int, default=8); p.add_argument("--margin", type=float, default=2.5, help="seconds before a phrase boundary by which the next phrase must be committed"); p.add_argument("--popsize", type=int, default=8); p.add_argument("--sigma", type=float, default=None, help="CMA-ES step size (default: from --wildness)"); p.add_argument("--wildness", type=float, default=0.3, help="0 = hugs the tab, 1 = far out (default 0.3)"); p.add_argument("--seed", type=int, default=0); p.add_argument("--workers", type=int, default=4, help="candidate-evaluation processes for the live planner"); p.add_argument("--improv-off", action="store_true", help="start with live learning off (toggle it from the stage or with L in the brain window)"); p.add_argument("--generate", action="store_true", help="riffs of its own from the multi-song read-out (fit-multi first); no backing track, its own drums"); p.add_argument("--bpm", type=float, default=None, help="tempo for --generate (default: the corpus median)"); p.add_argument("--phrases", type=int, default=24, help="how many phrases to generate (--generate)"); p.add_argument("--cycle", type=int, default=25, help="riff cycle length in sixteenths for --generate (25 = Rational Gaze's polymeter)"); p.add_argument("--snare", choices=["24", "thirds"], default="24", help="generator drums: snare on beats 2 and 4, or every 3 sixteenths"); p.add_argument("--density", type=float, default=None, help="notes per bar the generator aims for (default: the corpus median)"); p.set_defaults(fn=cmd_play)
    p = sub.add_parser("snapshot"); p.add_argument("t", type=float); p.add_argument("--view", default="front"); p.set_defaults(fn=cmd_snapshot)
    p = sub.add_parser("midi-ports"); p.set_defaults(fn=cmd_midi_ports)
    p = sub.add_parser("controls", help="does the fly's specific wiring matter? refit Stage A on shuffled / random reservoirs"); p.set_defaults(fn=cmd_controls)
    p = sub.add_parser("corpus", help="list the tabs in data/songs/ (and how each is transposed to the fly's tuning)"); p.set_defaults(fn=cmd_corpus)
    p = sub.add_parser("fit-multi", help="one shared read-out for every tab in data/songs/ -> data/cache/model_multi.npz"); p.add_argument("--lam", type=float, default=2e-4); p.add_argument("--bars-per-song", type=int, default=16, help="fit each song's most characteristic riffs (this many bars); 0 = whole songs"); p.add_argument("--expand", type=int, default=6000, help="random tanh features of the neuron states added to the read-out (0 = linear read-out, ~one song of capacity)"); p.set_defaults(fn=cmd_fit_multi)
    p = sub.add_parser("generate", help="offline: the fly makes up djent riffs from the multi-song read-out -> output/flybrain_djent_*.mid/.wav"); p.add_argument("--bars", type=int, default=32); p.add_argument("--phrase-bars", type=int, default=4); p.add_argument("--bpm", type=float, default=None); p.add_argument("--cycle", type=int, default=25); p.add_argument("--wildness", type=float, default=0.5); p.add_argument("--generations", type=int, default=10); p.add_argument("--seed", type=int, default=0); p.add_argument("--snare", choices=["24", "thirds"], default="24"); p.add_argument("--density", type=float, default=None, help="notes per bar to aim for (default: corpus median)"); p.add_argument("--drums", action="store_true", help="add the synthesized riff-locked drum kit (default: guitar only)"); p.set_defaults(fn=cmd_generate)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
