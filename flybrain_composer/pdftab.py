"""Read a Guitar-Pro-style *tab-only* PDF (Qt/GP7 export) into a Song — small OMR on the vector layer.

What the PDF gives us (all in page points):
* text layer: fret numbers positioned on the 8 tab lines (parenthesised = tied continuation),
  bar numbers above the staff, ``H``/``P`` legato marks, ``½`` bends, ``P.M.`` spans,
  ``1.``/``2.`` endings, ``3x``/``8x`` repeat counts, and SMuFL music-font glyphs:
  augmentation dot (U+E1E7), 8th/16th flags (U+E241/E243), 8th/quarter/16th rests
  (U+E4E6/E4E5/E4E7), repeat dots (U+E044);
* vector layer: staff lines every 6.4 pt, bar lines as thin rects (thick = repeat/end), one
  rhythm stem (0.51 pt line) below the staff per attack, beams as filled 2.1 pt bars at fixed
  levels (primary = 8th, secondary = 16th, tertiary = 32nd), tie/slur arcs as filled curves.

Timing: within a bar, stems and rests are ordered by x; each stem's value comes from its beam
levels / flag / dot; a stem with no fret number (or a parenthesised one) continues the previous
note (tie). Every bar is checked to sum to its time signature (4/4 here) and mismatches are
reported. Repeats, endings and "Nx" counts are expanded into the played bar sequence.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .transcription import Note, Section, Song

LINE_GAP = 6.4
N_STRINGS = 8
STEM_W = 0.51
GLYPH = {"dot": "", "flag8": "", "flag16": "", "rest8": "", "rest4": "",
         "rest16": "", "rest2": "", "rest1": "", "repeat_dots": ""}
REST_VALUE = {"": 2.0, "": 4.0, "": 1.0, "": 8.0, "": 16.0}
# Eb Bb Gb Db Ab Eb Bb F : string 1 (top line) .. string 8 (bottom line), as MIDI
TUNING_TOP_DOWN = (63, 58, 54, 49, 44, 39, 34, 29)


@dataclass
class Bar:
    page: int
    system: int
    x0: float
    x1: float
    number: int | None = None
    repeat_start: bool = False
    repeat_end: bool = False
    repeat_count: int = 2
    ending: int | None = None
    events: list = field(default_factory=list)      # (x, kind, payload)
    notes: list = field(default_factory=list)        # Note (start relative to bar, sixteenths)
    total16: float = 0.0
    warnings: list = field(default_factory=list)


class TabPdf:
    def __init__(self, path: str | Path):
        import pymupdf
        self.path = Path(path)
        self.doc = pymupdf.open(str(self.path))
        self.bars: list[Bar] = []
        self.bpm = 120.0
        self.tuning = TUNING_TOP_DOWN
        self._parse()

    # ------------------------------------------------------------------ page primitives
    def _systems(self, page):
        """Top y of each 8-line tab system on the page (from the staff-line drawings)."""
        ys = defaultdict(int)
        for d in page.get_drawings():
            for it in d["items"]:
                if it[0] == "l" and abs(it[1].y - it[2].y) < 0.3 and abs(it[1].x - it[2].x) > 40 and d.get("width") and d["width"] < 0.3:
                    ys[round(it[1].y, 1)] += 1
        lines = sorted(y for y, c in ys.items())
        systems = []
        for y in lines:
            if all(abs(y - s) > 3 for s in systems):
                # a system starts where the next 6 lines follow at LINE_GAP spacing (the 8th,
                # bottom line is chopped into short pieces by the fret numbers sitting on it)
                if all(any(abs(y + k * LINE_GAP - l) < 0.3 for l in lines) for k in range(1, N_STRINGS - 1)):
                    systems.append(y)
        return systems

    def _parse(self):
        bar_counter = 0
        for pno, page in enumerate(self.doc):
            words = page.get_text("words")
            drawings = page.get_drawings()
            if pno == 0:
                for w in words:
                    m = re.fullmatch(r"(\d+)", w[4])
                    if m and 100 < int(m.group(1)) < 300 and any(v[4] == "=" and abs(v[1] - w[1]) < 3 and v[0] < w[0] for v in words):
                        self.bpm = float(m.group(1))
            systems = self._systems(page)
            for si, sy in enumerate(systems):
                self._parse_system(pno, si, sy, words, drawings)
        # bar numbering by order of appearance (numbers printed in the pdf are used when present)
        for i, b in enumerate(self.bars):
            if b.number is None:
                b.number = i + 1

    def _parse_system(self, pno, si, sy, words, drawings):
        y_top, y_bot = sy, sy + LINE_GAP * (N_STRINGS - 1)
        off = lambda y: y - sy  # noqa: E731
        # --- bar lines (thin = normal, thick = repeat / final)
        bars_x = []
        for d in drawings:
            for it in d["items"]:
                if it[0] == "re":
                    r = it[1]
                    if abs(r.y0 - y_top) < 1.5 and abs(r.y1 - y_bot) < 1.5:
                        bars_x.append((r.x0, r.x1 - r.x0))
                elif it[0] == "l" and abs(it[1].x - it[2].x) < 0.3:
                    ya, yb = min(it[1].y, it[2].y), max(it[1].y, it[2].y)
                    if abs(ya - y_top) < 1.5 and abs(yb - y_bot) < 1.5:
                        w = d.get("width") or 0.68
                        bars_x.append((it[1].x - w / 2, w))
        bars_x.sort()
        if len(bars_x) < 2:
            return
        # repeat dots
        dots = [(w[0], off(w[1])) for w in words if w[4] == GLYPH["repeat_dots"] and -2 < off(w[1]) < 50]
        # collapse thick+thin pairs into one barline position
        lines = []
        for x, w in bars_x:
            if lines and x - lines[-1]["x1"] < 6:
                lines[-1]["x1"] = x + w
                lines[-1]["thick"] = lines[-1]["thick"] or w > 1.2
            else:
                lines.append({"x0": x, "x1": x + w, "thick": w > 1.2})
        for L in lines:
            L["dots_right"] = any(L["x1"] - 1 < dx < L["x1"] + 8 for dx, _ in dots)
            L["dots_left"] = any(L["x0"] - 8 < dx < L["x0"] + 1 for dx, _ in dots)
        # --- stems, beams, flags, dots, rests, fret numbers, marks
        stems = []
        halves = set()
        beams = {1: [], 2: [], 3: []}
        for d in drawings:
            r = d["rect"]
            its = d["items"]
            if len(its) == 1 and its[0][0] == "l" and d.get("width") and abs(d["width"] - STEM_W) < 0.08:
                a, b = its[0][1], its[0][2]
                if abs(a.x - b.x) < 0.3 and abs(off(max(a.y, b.y)) - 60.6) < 1.6 and 38 < off(min(a.y, b.y)) < 58:
                    stems.append(round(a.x, 1))
                    if abs(a.y - b.y) < 7.5:          # short stem = half note
                        halves.add(round(a.x, 1))
            elif d.get("fill") and len(its) == 4 and all(it[0] == "l" for it in its) and (r.y1 - r.y0) < 3.0 and (r.x1 - r.x0) > 2:
                lvl = {58: 1, 55: 2, 52: 3}.get(int(round(off(r.y0) / 1.0)) // 1 if False else min((58, 55, 52), key=lambda v: abs(v - off(r.y0))))
                if abs(off(r.y0) - {1: 58.5, 2: 55.3, 3: 52.1}[lvl]) < 1.6:
                    beams[lvl].append((r.x0, r.x1))
        stems = sorted(set(stems))
        flags8 = [w[0] + 1.0 for w in words if w[4] == GLYPH["flag8"] and 48 < off(w[1]) < 58]
        flags16 = [w[0] + 1.0 for w in words if w[4] == GLYPH["flag16"] and 48 < off(w[1]) < 58]
        adots = [w[0] for w in words if w[4] == GLYPH["dot"] and 44 < off(w[1]) < 52]
        rests = [(w[0], REST_VALUE[w[4][0]]) for w in words if w[4] and w[4][0] in REST_VALUE and 0 < off(w[1]) < 30]
        # a "" word is two rests glued together: split by glyph width
        for w in words:
            if len(w[4]) == 2 and all(c in REST_VALUE for c in w[4]) and 0 < off(w[1]) < 30:
                rests.append((w[0] + (w[2] - w[0]) / 2, REST_VALUE[w[4][1]]))
        frets = []
        for w in words:
            m = re.fullmatch(r"(\()?(\d+)(\))?", w[4])
            if m and -5 < off(w[1]) < 50 and (w[3] - w[1]) < 9:
                cy = (w[1] + w[3]) / 2 - sy
                string = int(round(cy / LINE_GAP)) + 1
                if not 1 <= string <= N_STRINGS:
                    continue
                digits = m.group(2)
                parts = [digits]
                if int(digits) > 24:          # two adjacent numbers glued by the text extractor
                    parts = [digits[:2], digits[2:]] if digits[:2] in ("10", "11", "12", "13", "14", "15") else [digits[:1], digits[1:]]
                    parts = [q for q in parts if q]
                x0, x1 = w[0], w[2]
                total = sum(len(q) for q in parts)
                pos = 0
                for q in parts:
                    cx = x0 + (x1 - x0) * (pos + len(q) / 2) / total
                    frets.append({"x": cx, "fret": int(q), "string": string, "tied": bool(m.group(1)) and q is parts[0]})
                    pos += len(q)
        bar_numbers = [(w[0], int(w[4])) for w in words if w[4].isdigit() and -9 < off(w[1]) < -0.5 and w[0] < 590]
        bends = [w[0] + 3 for w in words if w[4] in ("½", "full", "¼") and -24 < off(w[1]) < -6]
        bend_stems = set()
        for bx in bends:
            near = [x for x in stems if abs(x - bx) < 12]
            if near:
                bend_stems.add(min(near, key=lambda x: abs(x - bx)))
        pm_marks = [w[0] for w in words if w[4] == "P.M." and -26 < off(w[1]) < -4]
        endings = [(w[0], int(w[4][0])) for w in words if re.fullmatch(r"[12]\.", w[4]) and -32 < off(w[1]) < -4]
        counts = [(w[0], int(w[4][:-1])) for w in words if re.fullmatch(r"\d+x", w[4]) and -24 < off(w[1]) < -2]
        # P.M. spans: dashed horizontal lines after the mark at the same height
        pm_spans = []
        for x in pm_marks:
            xe = x + 14
            for d in drawings:
                for it in d["items"]:
                    if it[0] == "l" and abs(it[1].y - it[2].y) < 0.3 and -26 < off(it[1].y) < -4 and it[1].x >= x - 2 and it[1].x < x + 400:
                        xe = max(xe, max(it[1].x, it[2].x))
            pm_spans.append((x - 3, xe))

        # --- build bars
        for bi in range(len(lines) - 1):
            L0, L1 = lines[bi], lines[bi + 1]
            bar = Bar(pno, si, L0["x1"], L1["x0"])
            bar.repeat_start = L0["dots_right"]
            bar.repeat_end = L1["dots_left"]
            nums = [n for x, n in bar_numbers if L0["x0"] - 9 <= x < L1["x0"] - 9]
            bar.number = nums[0] if nums else None
            for x, e in endings:
                if L0["x0"] - 6 <= x < L1["x0"]:
                    bar.ending = e
            for x, c in counts:
                if L0["x0"] <= x <= L1["x1"] + 30:
                    bar.repeat_count = c
            in_bar = lambda x: L0["x1"] - 1.0 <= x < L1["x0"] - 0.5  # noqa: E731
            bstems = [x for x in stems if in_bar(x)]
            brests = [(x, v) for x, v in rests if in_bar(x)]
            events = []
            # whole notes: fret numbers in the bar that belong to no stem
            for f in frets:
                if in_bar(f["x"]) and not any(abs(f["x"] - x) < 4.5 for x in bstems) and not any(abs(f["x"] - x) < 6 for x, _ in brests):
                    same = [e for e in events if e[1] == "stem" and abs(e[0] - f["x"]) < 4.5]
                    if same:
                        same[0][2]["frets"].append(f)
                    else:
                        events.append((f["x"], "stem", {"value": 16.0, "frets": [f]}))
            for x in bstems:
                levels = sum(1 for lvl in (1, 2, 3) if any(a - 0.8 <= x <= b + 0.8 for a, b in beams[lvl]))
                if levels == 0:
                    if any(abs(x - f) < 3.5 for f in flags16):
                        levels = 2
                    elif any(abs(x - f) < 3.5 for f in flags8):
                        levels = 1
                value = {0: 4.0, 1: 2.0, 2: 1.0, 3: 0.5}[levels]
                if levels == 0 and x in halves:
                    value = 8.0
                # a dot sits right of the stem (within ~10 pt), before the next stem
                nxt = min([s for s in bstems if s > x] + [L1["x0"]])
                if any(x < dx < min(nxt, x + 12) for dx in adots):
                    value *= 1.5
                fs = [f for f in frets if abs(f["x"] - x) < 4.5]
                events.append((x, "stem", {"value": value, "frets": fs}))
            for x, v in brests:
                events.append((x, "rest", {"value": v}))
            events.sort(key=lambda e: e[0])
            bar.events = events
            # timing
            t = 0.0
            last_notes: dict[int, Note] = {}
            for x, kind, p in events:
                if kind == "rest":
                    t += p["value"]
                    last_notes = {}
                    continue
                fs = [f for f in p["frets"] if not f["tied"]]
                if not fs:
                    # continuation of the previous note(s)
                    for n in last_notes.values():
                        n.duration += p["value"]
                else:
                    last_notes = {}
                    for f in fs:
                        pitch = self.tuning[f["string"] - 1] + f["fret"]
                        pm = any(a <= x <= b for a, b in pm_spans)
                        bend = 1.0 if x in bend_stems else 0.0
                        n = Note(t, p["value"], pitch, 108, string=N_STRINGS - f["string"], fret=f["fret"],
                                 palm_mute=pm, bend=bend)
                        bar.notes.append(n)
                        last_notes[f["string"]] = n
                t += p["value"]
            bar.total16 = t
            if not events and (L1["x0"] - L0["x1"]) < 30:
                continue                      # the clef area left of a repeat sign, not a bar
            if abs(t - 16.0) > 1e-6:
                bar.warnings.append(f"bar {bar.number} (p{pno + 1} s{si + 1}) sums to {t} sixteenths")
            self.bars.append(bar)

    # ------------------------------------------------------------------ structure
    def played_order(self) -> list[int]:
        """Expand repeat signs, 1st/2nd endings and Nx counts into the played bar sequence."""
        order, i, start = [], 0, 0
        passes: dict[int, int] = {}
        n = len(self.bars)
        while i < n:
            b = self.bars[i]
            if b.repeat_start:
                start = i
            # endings: on pass k, skip ending bars that are not for this pass
            if b.ending is not None:
                cur = passes.get(start, 0) + 1
                if b.ending != cur:
                    i += 1
                    continue
            order.append(i)
            if b.repeat_end:
                # count applies to the whole repeated block; endings imply 2 passes
                block = self.bars[start:i + 1]
                count = max([bb.repeat_count for bb in block] + [2])
                has_endings = any(bb.ending is not None for bb in block)
                total = max(count, 2 if has_endings else count)
                cur = passes.get(start, 0) + 1
                passes[start] = cur
                if cur < total:
                    i = start
                    continue
            i += 1
        return order

    def to_song(self, title: str = "Rational Gaze (pdf tab)") -> Song:
        notes, sections = [], []
        t16 = 0.0
        order = self.played_order()
        # sections: a new one at every repeat-start bar (or ending 1) — the tab's phrase blocks
        block_ends = {}
        for i, b in enumerate(self.bars):          # block = repeat-start .. repeat-end
            if b.repeat_start:
                j = i
                while j < len(self.bars) and not self.bars[j].repeat_end:
                    j += 1
                block_ends[i] = self.bars[min(j, len(self.bars) - 1)].number
        for k, bi in enumerate(order):
            b = self.bars[bi]
            if b.repeat_start or k == 0 or (bi > 0 and self.bars[bi - 1].repeat_end and not b.repeat_start and (k == 0 or order[k - 1] == bi - 1)):
                if sections:
                    sections[-1].end = t16
                last = block_ends.get(bi, self.bars[-1].number if bi == 0 else b.number)
                name = f"bars {b.number}-{last}" if b.repeat_start else f"bars {b.number}+"
                sections.append(Section(name, t16, t16))
            for n in b.notes:
                notes.append(Note(t16 + n.start, n.duration, n.pitch, n.velocity, n.string, n.fret, n.palm_mute, n.bend))
            t16 += 16.0
        if sections:
            sections[-1].end = t16
        # no phrase annotations in a tab-only pdf: the riff cycle restarts with every block/pass
        s = Song(title=title, bpm=self.bpm, notes=notes, drums=[], sections=sections,
                 riff_starts=[sec.start for sec in sections], tuning=tuple(reversed(self.tuning)))
        s.meta = {"file": str(self.path), "bars_written": len(self.bars), "bars_played": len(order),
                  "warnings": [w for b in self.bars for w in b.warnings]}
        return s


def load_pdf_tab(path: str | Path) -> Song:
    return TabPdf(path).to_song()


if __name__ == "__main__":
    import sys
    t = TabPdf(sys.argv[1] if len(sys.argv) > 1 else "data/rational_gaze_tab2.pdf")
    print(f"{t.bpm} BPM, {len(t.bars)} written bars")
    for b in t.bars:
        seq = " ".join(("(" + "/".join(str(f["fret"]) for f in p["frets"]) + ")" if p["frets"] else "~") + f":{p['value']:g}" if k == "stem" else f"r:{p['value']:g}" for x, k, p in b.events)
        flags = ("|:" if b.repeat_start else "  ") + (":|" if b.repeat_end else "  ") + (f" x{b.repeat_count}" if b.repeat_count != 2 else "") + (f" [{b.ending}.]" if b.ending else "")
        print(f"bar {b.number:3} p{b.page + 1}s{b.system + 1} {flags:12s} sum={b.total16:g}  {seq}")
    s = t.to_song()
    print(f"played bars: {s.n_bars}, notes: {len(s.notes)}, {s.seconds / 60:.2f} min; warnings: {len(s.meta['warnings'])}")
    for w in s.meta["warnings"]:
        print("  !", w)
