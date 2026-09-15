"""Target song representation: Guitar Pro parsing + the built-in "Rational Gaze" structure.

Two sources of ground truth:

1. ``load_guitar_pro(path)`` — parse any .gp3/.gp4/.gp5 transcription with PyGuitarPro. Drop a
   fan transcription at ``data/rational_gaze.gp5`` (not redistributed with this repo: it is
   someone else's copyrighted transcription of a copyrighted song) and the pipeline uses it.

2. ``rational_gaze_builtin()`` — a *structural* encoding of the song assembled from the published
   analyses (Pieslak 2007 "Re-casting Metal"; Hannan, "Rhythmic deviance in the music of
   Meshuggah"; philarbonblog analysis): 135 BPM, guitars in F standard, a 25-sixteenth riff made
   of three pitch cells in the near-even duration series 6-4-6-4-5, stated four times and then a
   28-sixteenth extension so the phrase re-aligns with the 4/4 downbeat every 8 bars, only two
   pitches a minor ninth apart, crash/hi-hat quarter pulse with the snare on beat 3 and the kick
   following the guitar. The exact attack order inside each cell is my reconstruction, so treat
   this as "the shape of the song", not a note-for-note transcription.

Times are in *sixteenth notes from the start of the song* (floats), which is what the meter /
reservoir grid is built on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config

SIXTEENTHS_PER_BAR = 16


@dataclass
class Note:
    start: float          # sixteenths from song start
    duration: float       # sixteenths
    pitch: int            # MIDI note number
    velocity: int = 110   # 1..127
    string: int | None = None   # 0 = lowest string
    fret: int | None = None
    palm_mute: bool = False
    bend: float = 0.0     # semitones

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class Section:
    name: str
    start: float   # sixteenths
    end: float

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass
class Song:
    title: str
    bpm: float
    notes: list[Note]                     # the guitar part we want the fly to play
    drums: list[Note] = field(default_factory=list)   # GM drum notes (36 kick, 38 snare, 42 hat, 49 crash)
    sections: list[Section] = field(default_factory=list)
    riff_starts: list[float] = field(default_factory=list)   # sixteenth positions where a riff statement begins
    tuning: tuple[int, ...] = config.TUNING_F_STANDARD
    meta: dict = field(default_factory=dict)

    @property
    def n_sixteenths(self) -> int:
        last = max([n.end for n in self.notes] + [d.end for d in self.drums] + [0.0])
        return int(np.ceil(last / SIXTEENTHS_PER_BAR) * SIXTEENTHS_PER_BAR)

    @property
    def n_bars(self) -> int:
        return self.n_sixteenths // SIXTEENTHS_PER_BAR

    @property
    def seconds(self) -> float:
        return self.n_sixteenths * 60.0 / self.bpm / 4.0

    def section_at(self, t16: float) -> Section | None:
        for s in self.sections:
            if s.start <= t16 < s.end:
                return s
        return None


# ----------------------------------------------------------------------------------------------
# Fretboard helpers
# ----------------------------------------------------------------------------------------------
def assign_string_fret(pitch: int, tuning=config.TUNING_F_STANDARD, max_fret: int = config.NUM_FRETS,
                       prefer_low_strings: bool = True) -> tuple[int, int] | None:
    """Pick a (string, fret) for a pitch — lowest string with a playable fret (djent lives low)."""
    candidates = []
    for s, open_midi in enumerate(tuning):
        fret = pitch - open_midi
        if 0 <= fret <= max_fret:
            candidates.append((s, fret))
    if not candidates:
        return None
    if prefer_low_strings:
        # lowest string whose fret is <= 7, else the position with the smallest fret
        low = [c for c in candidates if c[1] <= 7]
        return low[0] if low else min(candidates, key=lambda c: c[1])
    return min(candidates, key=lambda c: c[1])


# ----------------------------------------------------------------------------------------------
# Built-in structural "Rational Gaze"
# ----------------------------------------------------------------------------------------------
LOW_F = config.TUNING_F_STANDARD[0]          # F1  (29) open lowest string
MINOR_NINTH = LOW_F + 13                     # Gb2 (42) — the "wide minor ninth leap"
OCTAVE = LOW_F + 12                          # F2  (41)

# A cell is a list of (pitch symbol, duration in sixteenths). 'L' = low chug, 'H' = high leap.
CELLS_A = {
    "X6": [("L", 1), ("L", 1), ("L", 1), ("H", 3)],
    "Y4": [("L", 2), ("L", 2)],
    "Z6": [("L", 1), ("L", 1), ("L", 2), ("H", 2)],
    "X4": [("L", 1), ("L", 1), ("L", 1), ("H", 1)],
    "Z5": [("L", 1), ("L", 1), ("L", 2), ("H", 1)],
    "TAIL3": [("L", 1), ("L", 1), ("L", 1)],
}
RIFF_A_ORDER = ["X6", "Y4", "Z6", "X4", "Z5"]              # 6+4+6+4+5 = 25 sixteenths

# Riff B: same cell vocabulary, rotated, with the leap replaced by the octave.
RIFF_B_ORDER = ["Z6", "X6", "Y4", "Z5", "X4"]

# Bridge: sparser half-time chugs on the same 25-cycle.
CELLS_BRIDGE = {
    "X6": [("L", 3), ("H", 3)],
    "Y4": [("L", 4)],
    "Z6": [("L", 2), ("L", 2), ("H", 2)],
    "X4": [("L", 4)],
    "Z5": [("L", 3), ("H", 2)],
    "TAIL3": [("L", 3)],
}


def _render_riff(order, cells, pitch_map, start, palm_mute=True, tail=None, velocity=112):
    """Lay out cells sequentially from `start` (sixteenths). Returns (notes, length)."""
    notes, t = [], float(start)
    seq = list(order) + ([tail] if tail else [])
    for cell in seq:
        for sym, dur in cells[cell]:
            pitch = pitch_map[sym]
            sf = assign_string_fret(pitch)
            is_high = sym == "H"
            notes.append(Note(
                start=t, duration=dur * (0.55 if (palm_mute and not is_high) else 0.95), pitch=pitch,
                velocity=velocity + (8 if is_high else 0) - (6 if dur == 1 and not is_high else 0),
                string=sf[0] if sf else None, fret=sf[1] if sf else None,
                palm_mute=palm_mute and not is_high, bend=0.5 if is_high else 0.0,
            ))
            t += dur
    return notes, t - start


def _phrase(order, cells, pitch_map, start, **kw):
    """4 × 25-sixteenth statements + one 28-sixteenth extension = 128 sixteenths = 8 bars."""
    notes, starts, t = [], [], start
    for i in range(5):
        tail = "TAIL3" if i == 4 else None
        n, length = _render_riff(order, cells, pitch_map, t, tail=tail, **kw)
        starts.append(t)
        notes += n
        t += length
    assert abs(t - start - 128) < 1e-9, t - start
    return notes, starts, t


def _drums_for(start16: float, end16: float, guitar_notes: list[Note], kick_follows_guitar=True,
               crash_on_downbeat=True) -> list[Note]:
    """Crash/hi-hat quarter pulse, snare on beat 3 of every bar, kick doubling the guitar attacks."""
    drums = []
    bar = int(start16 // SIXTEENTHS_PER_BAR)
    t = bar * SIXTEENTHS_PER_BAR
    while t < end16:
        for beat in range(4):
            tb = t + beat * 4
            if tb < start16 or tb >= end16:
                continue
            if beat == 0 and crash_on_downbeat:
                drums.append(Note(tb, 2, 49, 100))          # crash
            drums.append(Note(tb, 1, 42, 80 if beat else 96))   # hi-hat / ride quarter pulse
            if beat == 2:
                drums.append(Note(tb, 1, 38, 118))          # snare on beat 3 (half-time feel)
        t += SIXTEENTHS_PER_BAR
    if kick_follows_guitar:
        for n in guitar_notes:
            if start16 <= n.start < end16 and not (n.bend > 0):
                drums.append(Note(n.start, 1, 36, 112))     # kick doubles the chug
    return drums


def rational_gaze_builtin() -> Song:
    """Structural encoding of the song (see module docstring). ~4 minutes at 135 BPM."""
    A = dict(order=RIFF_A_ORDER, cells=CELLS_A, pitch_map={"L": LOW_F, "H": MINOR_NINTH})
    B = dict(order=RIFF_B_ORDER, cells=CELLS_A, pitch_map={"L": LOW_F, "H": OCTAVE})
    BR = dict(order=RIFF_A_ORDER, cells=CELLS_BRIDGE, pitch_map={"L": LOW_F, "H": MINOR_NINTH})

    form = [  # (name, riff, number of 8-bar phrases)
        ("intro", A, 2), ("verse1", A, 2), ("riffB", B, 2), ("verse2", A, 2),
        ("bridge", BR, 2), ("solo", A, 2), ("verse3", A, 2), ("outro", B, 2),
    ]
    notes, drums, sections, riff_starts = [], [], [], []
    t = 0.0
    for name, riff, n_phr in form:
        s0 = t
        for _ in range(n_phr):
            n, starts, t = _phrase(riff["order"], riff["cells"], riff["pitch_map"], t,
                                   palm_mute=(name != "bridge"))
            notes += n
            riff_starts += starts
        sections.append(Section(name, s0, t))
        drums += _drums_for(s0, t, notes, crash_on_downbeat=(name != "bridge"))

    # ending: four sustained low-F hits, one per bar, then a final held note (8 bars total)
    s0 = t
    for i in range(4):
        notes.append(Note(t + i * 16, 8, LOW_F, 120, 0, 0, palm_mute=False))
        drums += [Note(t + i * 16, 2, 49, 110), Note(t + i * 16, 1, 36, 120)]
    t += 64
    notes.append(Note(t, 64, LOW_F, 124, 0, 0, palm_mute=False))
    drums += [Note(t, 2, 49, 118), Note(t, 1, 36, 120)]
    t += 64
    sections.append(Section("ending", s0, t))

    return Song(title="Rational Gaze (structural reconstruction)", bpm=config.BPM, notes=notes,
                drums=drums, sections=sections, riff_starts=riff_starts)


# ----------------------------------------------------------------------------------------------
# Guitar Pro import
# ----------------------------------------------------------------------------------------------
def load_guitar_pro(path: str | Path, track_index: int | None = None) -> Song:
    """Parse a Guitar Pro file into a Song (guitar track only; drums if a percussion track exists)."""
    import guitarpro as gp

    song = gp.parse(str(path))
    tracks = [t for t in song.tracks if not t.isPercussionTrack]
    if not tracks:
        raise ValueError("no non-percussion tracks in file")
    track = tracks[track_index] if track_index is not None else max(
        tracks, key=lambda t: sum(len(b.notes) for m in t.measures for v in m.voices for b in v.beats))
    tuning = tuple(sorted(s.value for s in track.strings))
    drum_tracks = [t for t in song.tracks if t.isPercussionTrack]

    def _track_notes(tr, as_drums=False):
        out, riff_starts = [], []
        t16 = 0.0
        for measure in tr.measures:
            ts = measure.timeSignature
            measure_len16 = ts.numerator * 16.0 / ts.denominator.value
            if measure.header.hasMarker if hasattr(measure.header, "hasMarker") else measure.marker:
                riff_starts.append(t16)
            for voice in measure.voices[:1]:  # lead voice only
                bt = t16
                for beat in voice.beats:
                    dur16 = beat.duration.time / 960.0 * 4.0  # 960 ticks per quarter -> sixteenths
                    for note in beat.notes:
                        if note.type == gp.NoteType.rest or note.type == gp.NoteType.dead:
                            continue
                        if as_drums:
                            pitch = note.value
                            out.append(Note(bt, dur16, pitch, max(1, min(127, int(note.velocity)))))
                            continue
                        string_idx = len(tr.strings) - note.string   # GP strings are 1 = highest
                        open_midi = tr.strings[note.string - 1].value
                        pitch = open_midi + note.value
                        pm = bool(note.effect.palmMute)
                        bend = 0.0
                        if note.effect.bend is not None and note.effect.bend.points:
                            bend = max(p.value for p in note.effect.bend.points) / 4.0
                        out.append(Note(bt, dur16, pitch, max(1, min(127, int(note.velocity))),
                                        string=string_idx, fret=note.value, palm_mute=pm, bend=bend))
                    bt += dur16
            t16 += measure_len16
        return out, riff_starts

    notes, riff_starts = _track_notes(track)
    drums = _track_notes(drum_tracks[0], as_drums=True)[0] if drum_tracks else []
    s = Song(title=song.title or Path(path).stem, bpm=float(song.tempo), notes=notes, drums=drums,
             riff_starts=riff_starts, tuning=tuning)
    s.sections = [Section("song", 0.0, float(s.n_sixteenths))]
    return s


def load_song(gp_path: str | Path | None = None, track: str | int = "Rhythm") -> Song:
    """Prefer a user-supplied Guitar Pro file (.gp = GP7/8, .gp3-5 via PyGuitarPro), else the
    built-in structural reconstruction. `track` selects which guitar the fly learns."""
    candidates = [Path(gp_path)] if gp_path else []
    candidates += [config.DATA_DIR / f"rational_gaze{ext}" for ext in (".pdf", ".gp", ".gp5", ".gp4", ".gp3")]
    for p in candidates:
        if p and p.exists():
            print(f"[transcription] using {p}", flush=True)
            if p.suffix.lower() == ".pdf":
                from .pdftab import load_pdf_tab
                return load_pdf_tab(p)
            if p.suffix.lower() == ".gp":
                from .gpif import load_gp
                return load_gp(p, track=track)
            return load_guitar_pro(p)
    return rational_gaze_builtin()


# ----------------------------------------------------------------------------------------------
# Target matrices for the readout
# ----------------------------------------------------------------------------------------------
def pitch_index(pitch: int) -> int:
    return int(np.clip(pitch, config.MIDI_LO, config.MIDI_HI) - config.MIDI_LO)


N_PITCHES = config.MIDI_HI - config.MIDI_LO + 1


def song_to_targets(song: Song, steps_per_sixteenth: int = config.STEPS_PER_SIXTEENTH,
                    onset_width: int = 1) -> tuple[np.ndarray, dict]:
    """Piano-roll style regression targets on the reservoir grid.

    Returns Y of shape (T, 2 * N_PITCHES + 1): [onset one-hot (smoothed) | sustain | velocity/127].
    """
    T = song.n_sixteenths * steps_per_sixteenth
    onset = np.zeros((T, N_PITCHES))
    sustain = np.zeros((T, N_PITCHES))
    vel = np.zeros((T, 1))
    for n in song.notes:
        s = int(round(n.start * steps_per_sixteenth))
        e = int(round(n.end * steps_per_sixteenth))
        if s >= T:
            continue
        p = pitch_index(n.pitch)
        for k in range(-onset_width, onset_width + 1):
            if 0 <= s + k < T:
                onset[s + k, p] = max(onset[s + k, p], 1.0 if k == 0 else 0.5)
        sustain[s:max(s + 1, min(e, T)), p] = 1.0
        vel[s:max(s + 1, min(e, T)), 0] = n.velocity / 127.0
    Y = np.hstack([onset, sustain, vel])
    layout = {"onset": (0, N_PITCHES), "sustain": (N_PITCHES, 2 * N_PITCHES), "velocity": (2 * N_PITCHES, 2 * N_PITCHES + 1)}
    return Y, layout


if __name__ == "__main__":
    s = rational_gaze_builtin()
    print(s.title, f"{s.bpm} BPM, {s.n_bars} bars, {s.seconds/60:.1f} min, {len(s.notes)} guitar notes, "
          f"{len(s.drums)} drum hits, {len(s.riff_starts)} riff statements")
    for sec in s.sections:
        print(f"  {sec.name:8s} bars {sec.start/16:6.1f}-{sec.end/16:6.1f}")
