"""Guitar Pro 7/8 (.gp) reader: the file is a zip with ``Content/score.gpif`` (XML).

PyGuitarPro only reads .gp3/.gp4/.gp5, so this module parses the gpif format directly:
MasterBars (time signature, repeats, sections) -> per-track Bars -> Voices -> Beats (rhythm,
dynamic, free text) -> Notes (MIDI number, string/fret, tie, palm mute, bend). Repeats are
expanded into the played bar sequence.

Free-text annotations such as ``"25/16 x5 + 3/16"`` or ``"(7+6)/8 x4 + 6/4"`` (this
transcription annotates every phrase's grouping) are parsed into *group starts* — that is the
data-driven riff-cycle stream the reservoir gets instead of a fixed 25/16 assumption.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import xml.etree.ElementTree as ET

from .transcription import Note, Section, Song, assign_string_fret

NOTE_VALUE_16THS = {"Whole": 16.0, "Half": 8.0, "Quarter": 4.0, "Eighth": 2.0, "16th": 1.0,
                    "32nd": 0.5, "64th": 0.25, "128th": 0.125}
DYNAMIC_VELOCITY = {"PPP": 24, "PP": 40, "P": 56, "MP": 72, "MF": 88, "F": 104, "FF": 118, "FFF": 127}


def _text(el, path, default=None):
    if el is None:
        return default
    v = el.findtext(path)
    return v.strip() if isinstance(v, str) else default


def parse_grouping(text: str) -> list[float] | None:
    """'25/16 x5 + 3/16' -> [0, 25, 50, 75, 100, 125]; '(7+6)/8 x2 + 3/4' -> [0, 14, 26, 40, 52].

    Returns the group start offsets (sixteenths) inside the phrase, or None if not a grouping.
    """
    t = text.strip().replace("[shifted]", "").replace("×", "x")
    if not re.search(r"\d+\s*/\s*\d+", t):
        return None
    starts, pos = [], 0.0
    for term in re.split(r"\+(?![^()]*\))", t):        # split on '+' outside parentheses
        term = term.strip()
        if not term:
            continue
        m = re.match(r"^\(?\s*([\d\s+]+?)\s*\)?\s*/\s*(\d+)\s*(?:x\s*(\d+))?$", term)
        if not m:
            return None
        nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
        den = int(m.group(2))
        reps = int(m.group(3) or 1)
        for _ in range(reps):
            for n in nums:
                starts.append(pos)
                pos += n * 16.0 / den
    return starts


class GpifSong:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        with zipfile.ZipFile(self.path) as z:
            xml = z.read("Content/score.gpif").decode("utf-8")
        self.root = ET.fromstring(xml)
        self.title = _text(self.root, "Score/Title", self.path.stem)
        self.artist = _text(self.root, "Score/Artist", "")
        self.bars = {b.get("id"): b for b in self.root.find("Bars")}
        self.voices = {v.get("id"): v for v in self.root.find("Voices")}
        self.beats = {b.get("id"): b for b in self.root.find("Beats")}
        self.notes = {n.get("id"): n for n in self.root.find("Notes")}
        self.rhythms = {r.get("id"): r for r in self.root.find("Rhythms")}
        self.master_bars = list(self.root.find("MasterBars"))
        self.tracks = list(self.root.find("Tracks"))
        self.bpm = self._tempo()
        self.played = self._expand_repeats()

    # ------------------------------------------------------------------ structure
    def _tempo(self) -> float:
        for a in self.root.findall("MasterTrack/Automations/Automation"):
            if _text(a, "Type") == "Tempo":
                return float(_text(a, "Value", "120 2").split()[0])
        return 120.0

    def _expand_repeats(self) -> list[int]:
        """Playback order of master-bar indices (repeat start/end with count expanded)."""
        order, i, start = [], 0, 0
        pending: dict[int, int] = {}      # end bar -> remaining passes
        while i < len(self.master_bars):
            mb = self.master_bars[i]
            rep = mb.find("Repeat")
            if rep is not None and rep.get("start") == "true":
                start = i
            order.append(i)
            if rep is not None and rep.get("end") == "true":
                count = int(rep.get("count") or 2)
                left = pending.get(i, count) - 1
                if left > 0:
                    pending[i] = left
                    i = start
                    continue
                pending.pop(i, None)
            i += 1
        return order

    def bar_length16(self, mb_index: int) -> float:
        num, den = (_text(self.master_bars[mb_index], "Time", "4/4")).split("/")
        return int(num) * 16.0 / int(den)

    def track_names(self) -> list[str]:
        return [" ".join((_text(t, "Name", "") or "").split()) for t in self.tracks]

    def find_track(self, hint: str) -> int:
        names = [n.lower() for n in self.track_names()]
        for i, n in enumerate(names):
            if hint.lower() in n:
                return i
        if hint.lower() in ("rhythm", "auto", "guitar", ""):
            return self.guess_rhythm_track()
        raise KeyError(f"no track matching {hint!r}; tracks: {self.track_names()}")

    NOT_GUITAR = ("bass", "drum", "percussion", "piano", "synth", "keys", "vocal", "voice", "fx", "strings", "orchestra", "pad", "tone")
    LEAD_WORDS = ("lead", "solo", "clean", "melody", "harmony", "high")
    RHYTHM_WORDS = ("rhythm", "rythm", "low", "right", "left", "8-string", "7-string", "guitar 1", "guitar #1", "gtr 1")

    def guess_rhythm_track(self) -> int:
        """The rhythm-guitar track when none is named so: a non-percussion, non-bass track, preferring
        names that say rhythm/low, avoiding lead/solo/clean, and among the rest the one with the most
        notes — tabs name tracks after players ('Mårten Hagström'), amps, or nothing useful."""
        best, best_score = None, -1e9
        for i, name in enumerate(self.track_names()):
            n = name.lower()
            if self.is_percussion(i) or any(w in n for w in self.NOT_GUITAR):
                continue
            try:
                notes = self.track_notes(i)[0]
            except Exception:  # noqa: BLE001 — a broken track must not sink the file
                continue
            if not notes:
                continue
            low = min(x.pitch for x in notes)
            score = len(notes) / 100.0 - low / 4.0             # busy and low = rhythm guitar
            if any(w in n for w in self.RHYTHM_WORDS):
                score += 12
            if any(w in n for w in self.LEAD_WORDS):
                score -= 12
            if score > best_score:
                best, best_score = i, score
        if best is None:
            raise KeyError(f"no guitar track found; tracks: {self.track_names()}")
        return best

    def is_percussion(self, track_index: int) -> bool:
        t = self.tracks[track_index]
        return (_text(t, ".//InstrumentSet/Type", "") or "").lower() == "drumkit"

    # ------------------------------------------------------------------ content
    def _beat_dur16(self, beat) -> float:
        r = self.rhythms[beat.find("Rhythm").get("ref")]
        d = NOTE_VALUE_16THS[_text(r, "NoteValue", "Quarter")]
        dot = r.find("AugmentationDot")
        if dot is not None:
            d *= {"1": 1.5, "2": 1.75}.get(dot.get("count", "1"), 1.5)
        tup = r.find("PrimaryTuplet")
        if tup is not None:
            d *= int(tup.get("den")) / int(tup.get("num"))
        return d

    def _note_prop(self, note, name):
        return note.find(f"Properties/Property[@name='{name}']")

    def track_notes(self, track_index: int, as_drums: bool = False) -> tuple[list[Note], list[float], list[Section]]:
        """Notes (sixteenths from song start, repeats expanded), group starts, sections."""
        notes: list[Note] = []
        group_starts: list[float] = []
        sections: list[Section] = []
        open_by_string: dict[int, Note] = {}
        t16 = 0.0
        for mb_index in self.played:
            mb = self.master_bars[mb_index]
            bar_len = self.bar_length16(mb_index)
            sec = mb.find("Section")
            if sec is not None:
                name = " ".join((_text(sec, "Text", "") or _text(sec, "Letter", "") or "").split())
                if sections:
                    sections[-1].end = t16
                if sections and sections[-1].name == name:      # repeat pass of the same section
                    sections[-1].end = t16 + bar_len
                else:
                    sections.append(Section(name or f"section{len(sections)}", t16, t16 + bar_len))
            elif sections:
                sections[-1].end = t16 + bar_len
            bar_id = mb.findtext("Bars").split()[track_index]
            bar = self.bars[bar_id]
            for vid in bar.findtext("Voices").split():
                if vid == "-1":
                    continue
                pos = t16
                for beat_id in self.voices[vid].findtext("Beats").split():
                    beat = self.beats[beat_id]
                    dur = self._beat_dur16(beat)
                    ft = beat.findtext("FreeText")
                    if ft:
                        g = parse_grouping(ft)
                        if g:
                            group_starts += [pos + x for x in g]
                    vel = DYNAMIC_VELOCITY.get(_text(beat, "Dynamic", "F"), 104)
                    for nid in (beat.findtext("Notes") or "").split():
                        n = self.notes[nid]
                        midi_el = self._note_prop(n, "Midi")
                        if midi_el is None:
                            continue
                        midi = int(midi_el.findtext("Number"))
                        string_el = self._note_prop(n, "String")
                        fret_el = self._note_prop(n, "Fret")
                        string = int(string_el.findtext("String")) if string_el is not None else None
                        fret = int(fret_el.findtext("Fret")) if fret_el is not None else None
                        tie = n.find("Tie")
                        if tie is not None and tie.get("destination") == "true" and not as_drums:
                            prev = open_by_string.get(string)
                            if prev is not None:
                                prev.duration = pos + dur - prev.start
                                continue
                        bend = 0.0
                        bd = self._note_prop(n, "BendDestinationValue")
                        if bd is not None:
                            bend = float(bd.findtext("Float") or 0) / 50.0      # 100 = whole tone
                        dead = self._note_prop(n, "Muted") is not None
                        if dead and not as_drums:
                            vel = max(20, vel - 40)
                        note = Note(pos, dur, midi, vel, string=string, fret=fret,
                                    palm_mute=self._note_prop(n, "PalmMuted") is not None, bend=bend)
                        if as_drums:
                            note.string = None
                            note.fret = None
                        notes.append(note)
                        if string is not None:
                            open_by_string[string] = note
                    pos += dur
                break   # first (lead) voice only
            t16 += bar_len
        if sections:
            sections[-1].end = t16
        notes.sort(key=lambda n: (n.start, n.pitch))
        return notes, sorted(set(group_starts)), sections

    def to_song(self, track: str | int = "Rhythm", drums: str | int | None = "drum") -> Song:
        ti = self.find_track(track) if isinstance(track, str) else int(track)
        notes, groups, sections = self.track_notes(ti)
        drum_notes: list[Note] = []
        if drums is not None:
            try:
                di = self.find_track(drums) if isinstance(drums, str) else int(drums)
            except KeyError:
                di = next((i for i in range(len(self.tracks)) if self.is_percussion(i)), None)
            if di is not None:
                drum_notes = self.track_notes(di, as_drums=True)[0]
        tuning_el = self.tracks[ti].find(".//Staves/Staff/Properties/Property[@name='Tuning']/Pitches")
        tuning = tuple(int(x) for x in tuning_el.text.split()) if tuning_el is not None else None
        song = Song(title=f"{self.artist} - {self.title}".strip(" -"), bpm=self.bpm, notes=notes,
                    drums=drum_notes, sections=sections, riff_starts=groups,
                    tuning=tuning or Song.tuning)
        song.meta = {"file": str(self.path), "track": self.track_names()[ti], "played_bars": len(self.played),
                     "master_bars": len(self.master_bars)}
        return song


def load_gp(path: str | Path, track: str | int = "Rhythm") -> Song:
    return GpifSong(path).to_song(track=track)


if __name__ == "__main__":
    import sys
    g = GpifSong(sys.argv[1] if len(sys.argv) > 1 else "data/rational_gaze.gp")
    print(g.title, "|", g.artist, "|", g.bpm, "BPM |", len(g.master_bars), "bars ->", len(g.played), "played")
    print("tracks:", g.track_names())
    s = g.to_song()
    print(f"{len(s.notes)} guitar notes, {len(s.drums)} drum hits, {len(s.riff_starts)} group starts, "
          f"{s.n_bars} bars, {s.seconds/60:.2f} min")
    for sec in s.sections:
        print(f"  {sec.start/16:6.1f}-{sec.end/16:6.1f}  {sec.name}")
    from collections import Counter
    print("pitches:", Counter(n.pitch for n in s.notes).most_common(12))
    print("durations:", Counter(n.duration for n in s.notes).most_common(8))
    print("first 12 notes:", [(n.start, n.duration, n.pitch, n.string, n.fret) for n in s.notes[:12]])
    print("group starts (first 16):", s.riff_starts[:16])
