"""Readout activity -> note events -> MIDI (and a note-level diff against the ground truth)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config
from .transcription import Note, Song, N_PITCHES, assign_string_fret


def decode_notes(Yhat: np.ndarray, layout: dict, steps_per_sixteenth: int = config.STEPS_PER_SIXTEENTH,
                 onset_threshold: float = 0.45, sustain_threshold: float = 0.4,
                 min_gap_steps: int = 2, max_len_sixteenths: float = 64, max_polyphony: int = 3,
                 chord_ratio: float = 0.6) -> list[Note]:
    """Quantise the readout into MIDI note events (polyphonic: dyads/chords are allowed).

    An onset is emitted where the strongest onset channel peaks above threshold (with a
    refractory gap); every pitch whose onset channel is above ``chord_ratio`` x the peak (and the
    absolute threshold) starts a note there. Each note lasts while its pitch's sustain channel
    stays up, is cut by the next onset, and is snapped to the 32nd-note grid.
    """
    o0, o1 = layout["onset"]
    s0, s1 = layout["sustain"]
    v0, _ = layout["velocity"]
    onset, sustain, vel = Yhat[:, o0:o1], Yhat[:, s0:s1], Yhat[:, v0]
    T = len(Yhat)
    best = onset.max(axis=1)
    notes: list[Note] = []
    last = -10 ** 9
    t = 0
    while t < T:
        if best[t] > onset_threshold and t - last >= min_gap_steps and best[t] >= best[max(0, t - 1)]                 and best[t] >= (best[t + 1] if t + 1 < T else -1):
            row = onset[t]
            pitches = np.where((row >= chord_ratio * best[t]) & (row > onset_threshold))[0]
            pitches = pitches[np.argsort(-row[pitches])][:max_polyphony]
            # cut the notes still ringing at this onset
            for n in notes:
                if n.start * steps_per_sixteenth + n.duration * steps_per_sixteenth > t:
                    n.duration = max(0.25, (t - n.start * steps_per_sixteenth) / steps_per_sixteenth)
            start16 = round(t / steps_per_sixteenth * 2) / 2
            for p in pitches:
                e = t + 1
                max_e = min(T, t + int(max_len_sixteenths * steps_per_sixteenth))
                while e < max_e and sustain[e, p] > sustain_threshold:
                    e += 1
                velocity = int(np.clip(round(vel[t:e].mean() * 127), 30, 127))
                midi = int(p) + config.MIDI_LO
                sf = assign_string_fret(midi)
                dur16 = max(0.25, (e - t) / steps_per_sixteenth)
                notes.append(Note(start16, dur16, midi, velocity, string=sf[0] if sf else None,
                                  fret=sf[1] if sf else None, palm_mute=dur16 <= 0.75))
            last = t
            t = max(t + 1, t + min_gap_steps)
        else:
            t += 1
    return [n for n in notes if n.duration > 0]


def transfer_articulation(notes: list[Note], truth: list[Note]) -> list[Note]:
    """Copy palm-mute / bend / fingering from the transcription onto decoded notes (same pitch,
    same 32nd-note slot) so the render articulates what the tab marks, not a heuristic."""
    ref = {}
    for n in truth:
        ref.setdefault((round(n.start * 2), n.pitch), n)
    for n in notes:
        r = ref.get((round(n.start * 2), n.pitch)) or ref.get((round(n.start * 2) - 1, n.pitch)) or ref.get((round(n.start * 2) + 1, n.pitch))
        if r is not None:
            n.palm_mute, n.bend = r.palm_mute, r.bend
            if r.string is not None:
                n.string, n.fret = r.string, r.fret
    return notes


def articulate(notes: list[Note], *, chug_gate: float = 0.75, min_sec: float = 0.06, bpm: float = config.BPM) -> list[Note]:
    """Palm-muted notes (as marked in the transcription) are gated to `chug_gate` of their written
    length — a DI sampler has no palm-mute sample, so a shorter note is how you program a mute.
    Everything else keeps its full written length."""
    sec = config.sixteenth_seconds(bpm)
    out = []
    for n in notes:
        m = Note(n.start, n.duration, n.pitch, n.velocity, n.string, n.fret, n.palm_mute, n.bend)
        if n.palm_mute:
            m.duration = max(min_sec / sec, n.duration * chug_gate)
        out.append(m)
    return out


def notes_to_midi(notes: list[Note], path: str | Path, bpm: float = config.BPM,
                  drums: list[Note] | None = None, program: int = 30, chugs: bool = True) -> Path:
    """Write a MIDI file: guitar on track 0 (program 30 = Distortion Guitar), optional GM drums."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    sec = 60.0 / bpm / 4.0
    g = pretty_midi.Instrument(program=program, name="8ridge lite (fly brain)")
    if chugs:
        notes = articulate(notes, bpm=bpm)
    for n in sorted(notes, key=lambda n: n.start):
        g.notes.append(pretty_midi.Note(velocity=int(n.velocity), pitch=int(n.pitch),
                                        start=n.start * sec, end=(n.start + n.duration) * sec))
    pm.instruments.append(g)
    if drums:
        d = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
        for n in sorted(drums, key=lambda n: n.start):
            d.notes.append(pretty_midi.Note(velocity=int(n.velocity), pitch=int(n.pitch),
                                            start=n.start * sec, end=(n.start + max(0.25, n.duration)) * sec))
        pm.instruments.append(d)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))
    return path


def midi_to_notes(path: str | Path, bpm: float = config.BPM) -> tuple[list[Note], list[Note]]:
    """Read a MIDI file back into (guitar notes, drum notes) on the sixteenth grid."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(path))
    sec = 60.0 / bpm / 4.0
    guitar, drums = [], []
    for inst in pm.instruments:
        target = drums if inst.is_drum else guitar
        for n in inst.notes:
            sf = None if inst.is_drum else assign_string_fret(n.pitch)
            target.append(Note(n.start / sec, (n.end - n.start) / sec, n.pitch, n.velocity,
                               string=sf[0] if sf else None, fret=sf[1] if sf else None))
    return guitar, drums


def diff_notes(pred: list[Note], truth: list[Note], tol_sixteenths: float = 0.5, bpm: float = config.BPM) -> dict:
    """Onset-level precision/recall/F1 (+ pitch accuracy and timing error of matched onsets)."""
    P = sorted(pred, key=lambda n: n.start)
    Tt = sorted(truth, key=lambda n: n.start)
    used = np.zeros(len(Tt), dtype=bool)
    ts = np.array([n.start for n in Tt]) if Tt else np.zeros(0)
    matched, pitch_ok, timing = 0, 0, []
    for n in P:
        if len(ts) == 0:
            break
        # nearest unused within tolerance, preferring the same pitch
        cand = np.where((np.abs(ts - n.start) <= tol_sixteenths) & ~used)[0]
        if len(cand) == 0:
            continue
        same = [j for j in cand if Tt[j].pitch == n.pitch]
        pool = np.array(same) if same else cand
        j = pool[np.argmin(np.abs(ts[pool] - n.start))]
        used[j] = True
        matched += 1
        pitch_ok += int(Tt[j].pitch == n.pitch)
        timing.append(abs(ts[j] - n.start))
    prec = matched / max(1, len(P))
    rec = matched / max(1, len(Tt))
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {
        "n_pred": len(P), "n_truth": len(Tt), "matched": matched,
        "precision": prec, "recall": rec, "f1": f1,
        "pitch_accuracy": pitch_ok / max(1, matched),
        "mean_onset_error_ms": float(np.mean(timing)) * config.sixteenth_seconds(bpm) * 1000 if timing else float("nan"),
    }
