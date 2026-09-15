"""A faithful Python port of the 8ridge lite sampler engine (JamesStubbsEng/8ridgelite, GPL-3).

8ridge lite is a JUCE ``Synthesiser`` of ``SamplerVoice``s. From ``Source/Synth/GuitarSynth.cpp``:

* the WAVs in ``Natural/`` (or ``Tuned/`` = the "Perfect" switch) are sorted naturally and
  assigned to MIDI notes k = 28, 29, ... (E1 upwards, one sample per semitone, 61 samples -> 28..88);
* ``SamplerSound(name, reader, notes, k, attack=0.005 s, release=0.01 s, maxLength=20 s)``;
* the first sound (E1) is registered with ``setRange(16, 28)`` — i.e. it responds to MIDI
  16..43 — so every note from F1 to G2 *also* triggers the repitched E1 sample on top of its
  own sample. That layering is what the plugin does, so it is reproduced here
  (``faithful_layering``);
* 16 voices, stereo output; ``mono1``/``mono2`` copy one channel onto the other.

JUCE ``SamplerVoice`` semantics reproduced: pitch ratio 2^((note-root)/12) * srcRate/outRate,
linear-interpolated playback, gain = velocity/127 on both channels, JUCE ``ADSR`` with the
attack/release rates computed at the *source* sample rate but advanced once per output sample,
release starting from the current envelope value, a new note-on on the same pitch releasing
the voices already playing it. Voice stealing only ever affects voices that have already gone
silent (10 ms release) for a monophonic riff, so it is not modelled.

On top of the faithful DI engine there is an optional (non-plugin) high-gain amp + 4x12 cab
approximation so the result sounds like a Meshuggah rhythm tone instead of a clean DI, and a
procedural drum kit so the polymeter is audible.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .transcription import Note, Song

ATTACK_SEC = 0.005
RELEASE_SEC = 0.01
MAX_SAMPLE_SEC = 20.0
MAX_VOICES = 16
FIRST_NOTE = 28
E1_RANGE = (16, 43)      # BigInteger::setRange(16, 28, true) -> bits 16..43 inclusive

DEFAULT_SAMPLE_DIRS = [
    Path(r"C:\ProgramData\Haventone\Bridgelite"),                  # where the installer puts them
    config.THIRD_PARTY / "8ridgelite" / "8ridgelite_20sec_wav",      # the git clone
]


@dataclass
class Sound:
    root: int
    data: np.ndarray      # (n + 4, 2) float32, zero padded like JUCE's SamplerSound
    rate: int
    length: int           # number of real samples


class BridgeliteEngine:
    def __init__(self, sample_dir: Path | None = None, sample_set: str = "Natural",
                 sample_rate: int = 48000, faithful_layering: bool = True, mono: int = 0):
        self.sample_rate = sample_rate
        self.faithful_layering = faithful_layering
        self.mono = mono                      # 0 stereo, 1 = mono1 (L->R), 2 = mono2 (R->L)
        self.sample_set = sample_set
        self.dir = self._find_dir(sample_dir) / sample_set
        self.files = self._sorted_files(self.dir)
        self._sounds: dict[int, Sound] = {}

    # ------------------------------------------------------------------ samples
    @staticmethod
    def _find_dir(sample_dir):
        for d in ([Path(sample_dir)] if sample_dir else []) + DEFAULT_SAMPLE_DIRS:
            if (d / "Natural").is_dir():
                return d
        raise FileNotFoundError("8ridge lite samples not found (expected <dir>/Natural/*.wav). "
                                "Clone https://github.com/JamesStubbsEng/8ridgelite into third_party/.")

    @staticmethod
    def _sorted_files(d: Path) -> list[Path]:
        # juce::File::NaturalFileComparator == numeric-aware sort of the file names
        def key(p: Path):
            import re
            return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]
        return sorted([p for p in d.iterdir() if p.suffix.lower() == ".wav"], key=key)

    def sound(self, root: int) -> Sound:
        """Lazily load the sample registered at MIDI note `root` (28 + file index)."""
        if root not in self._sounds:
            import soundfile as sf
            idx = root - FIRST_NOTE
            if not 0 <= idx < len(self.files):
                raise KeyError(f"no sample for MIDI {root}")
            data, rate = sf.read(self.files[idx], dtype="float32", always_2d=True)
            n = min(len(data), int(MAX_SAMPLE_SEC * rate))
            data = data[:n]
            if data.shape[1] == 1:
                data = np.repeat(data, 2, axis=1)
            padded = np.zeros((n + 4, 2), dtype=np.float32)
            padded[:n] = data[:, :2]
            self._sounds[root] = Sound(root, padded, rate, n)
        return self._sounds[root]

    @property
    def lowest(self) -> int:
        return FIRST_NOTE

    @property
    def highest(self) -> int:
        return FIRST_NOTE + len(self.files) - 1

    def sounds_for(self, midi: int) -> list[Sound]:
        """Which SamplerSounds respond to this note (the plugin can return two)."""
        out = []
        if self.faithful_layering and E1_RANGE[0] <= midi <= E1_RANGE[1]:
            out.append(self.sound(FIRST_NOTE))
        if self.lowest <= midi <= self.highest and (midi != FIRST_NOTE or not out):
            out.append(self.sound(midi))
        return out

    # ------------------------------------------------------------------ rendering
    def render_voice(self, sound: Sound, midi: int, velocity: int, n_on: int, n_off: int) -> tuple[int, np.ndarray]:
        """One SamplerVoice: returns (start_sample, (n, 2) audio) for note-on at n_on, note-off at n_off."""
        sr_out = self.sample_rate
        pitch_ratio = 2.0 ** ((midi - sound.root) / 12.0) * sound.rate / sr_out
        # how many output samples until the source runs out (sourceSamplePosition > length)
        n_until_end = int(np.floor(sound.length / pitch_ratio)) + 1
        attack_rate = 1.0 / (ATTACK_SEC * sound.rate)
        release_rate = 1.0 / (RELEASE_SEC * sound.rate)
        hold = max(1, n_off - n_on)
        n_release = int(np.ceil(1.0 / release_rate)) + 1
        n_total = min(hold + n_release, n_until_end)
        if n_total <= 0:
            return n_on, np.zeros((0, 2), np.float32)
        k = np.arange(n_total)
        # juce::ADSR: linear attack to 1, sustain 1 (decay 0), linear release from current value
        env = np.minimum(1.0, (k + 1) * attack_rate)
        rel = k >= hold
        if rel.any():
            start_val = env[hold - 1] if hold - 1 < n_total else 1.0
            env[rel] = np.maximum(0.0, start_val - (k[rel] - hold + 1) * release_rate)
        pos = k * pitch_ratio
        ip = pos.astype(np.int64)
        alpha = (pos - ip).astype(np.float32)
        ip = np.minimum(ip, sound.length + 2)
        d = sound.data
        out = d[ip] * (1.0 - alpha)[:, None] + d[ip + 1] * alpha[:, None]
        gain = np.float32(velocity / 127.0)
        out *= (gain * env.astype(np.float32))[:, None]
        # drop the silent tail after the release finished
        nz = np.nonzero(env > 0)[0]
        if len(nz) and nz[-1] + 1 < n_total:
            out = out[: nz[-1] + 1]
        return n_on, out.astype(np.float32)

    def render(self, notes: list[Note], bpm: float = config.BPM, tail_sec: float = 2.0,
               progress: bool = False) -> np.ndarray:
        """Render a note list (times in sixteenths) to stereo float32 at self.sample_rate."""
        sr = self.sample_rate
        sec16 = 60.0 / bpm / 4.0
        events = sorted(notes, key=lambda n: n.start)
        total = int((max([n.end for n in events] + [0.0]) * sec16 + tail_sec) * sr)
        buf = np.zeros((total + sr, 2), dtype=np.float32)
        # a note-on on a pitch releases voices still playing that pitch (Synthesiser::noteOn)
        next_same = {}
        starts = [int(round(n.start * sec16 * sr)) for n in events]
        for i in range(len(events) - 1, -1, -1):
            p = events[i].pitch
            next_same[i] = next_same.get(("p", p), None)
            next_same[("p", p)] = starts[i]
        it = enumerate(events)
        if progress:
            from tqdm import tqdm
            it = tqdm(list(it), desc="8ridge lite", unit="note")
        for i, n in it:
            if n.pitch < E1_RANGE[0] or n.pitch > self.highest:
                continue
            n_on = starts[i]
            n_off = int(round(n.end * sec16 * sr))
            if next_same[i] is not None:
                n_off = min(n_off, next_same[i])
            for sound in self.sounds_for(n.pitch):
                s0, audio = self.render_voice(sound, n.pitch, n.velocity, n_on, n_off)
                if len(audio):
                    e = min(s0 + len(audio), len(buf))
                    buf[s0:e] += audio[: e - s0]
        if self.mono == 1:
            buf[:, 1] = buf[:, 0]
        elif self.mono == 2:
            buf[:, 0] = buf[:, 1]
        return buf[:total]


# ------------------------------------------------------------------------------------------
# Not part of the plugin: amp + cab so it sounds like a djent rhythm tone, and a drum kit
# ------------------------------------------------------------------------------------------
def _biquad(kind: str, f0: float, sr: int, q: float = 0.707, gain_db: float = 0.0):
    """RBJ cookbook biquad coefficients (b, a)."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2 * q)
    if kind == "lowpass":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]; a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "highpass":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]; a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]; a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind == "lowshelf":
        b = [A * ((A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * alpha), 2 * A * ((A - 1) - (A + 1) * cw),
             A * ((A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * alpha)]
        a = [(A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * alpha, -2 * ((A - 1) + (A + 1) * cw),
             (A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * alpha]
    elif kind == "highshelf":
        b = [A * ((A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * alpha), -2 * A * ((A - 1) + (A + 1) * cw),
             A * ((A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * alpha)]
        a = [(A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * alpha, 2 * ((A - 1) - (A + 1) * cw),
             (A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * alpha]
    else:
        raise ValueError(kind)
    b, a = np.array(b) / a[0], np.array(a) / a[0]
    return b, a


def _filt(x, b, a):
    from scipy.signal import lfilter
    return lfilter(b, a, x, axis=0).astype(np.float32)


def amp_chain(di: np.ndarray, sr: int, drive: float = 35.0, tight_hz: float = 110.0, *,
              peak: float | None = None, return_peak: bool = False):
    """Tube-screamer-style boost -> high-gain preamp -> tone stack -> 4x12 cab approximation.

    The output is normalised to 0.85 of its own peak unless a fixed `peak` is given (the live
    player renders phrase by phrase and must keep one level across phrases)."""
    x = di.astype(np.float32)
    # TS9-ish boost: tighten the low end, mid hump, soft clip
    x = _filt(x, *_biquad("highpass", tight_hz, sr, 0.8))
    x = _filt(x, *_biquad("peak", 800.0, sr, 0.9, 6.0))
    x = np.tanh(x * 12.0) / 1.0
    x = _filt(x, *_biquad("lowpass", 4200.0, sr, 0.7))
    # preamp: two asymmetric gain stages with DC blocking between them
    for g, bias in ((drive, 0.15), (drive * 0.6, -0.1)):
        x = np.tanh(x * g + bias) - np.tanh(bias)
        x = _filt(x, *_biquad("highpass", 30.0, sr, 0.7))
    # tone stack: scoop a little at 500 Hz, presence at 3 kHz, tame fizz
    x = _filt(x, *_biquad("peak", 500.0, sr, 0.8, -4.0))
    x = _filt(x, *_biquad("peak", 3000.0, sr, 1.2, 3.0))
    # cabinet: 4x12 — low resonance ~ 120 Hz, dip ~ 450 Hz, cone breakup ~ 2.5 kHz, steep roll-off > 5 kHz
    x = _filt(x, *_biquad("peak", 120.0, sr, 1.0, 4.0))
    x = _filt(x, *_biquad("peak", 450.0, sr, 1.5, -3.0))
    x = _filt(x, *_biquad("peak", 2500.0, sr, 1.5, 2.5))
    for _ in range(2):
        x = _filt(x, *_biquad("lowpass", 5200.0, sr, 0.9))
    x = _filt(x, *_biquad("highpass", 70.0, sr, 0.7))
    own_peak = float(np.abs(x).max()) or 1.0
    out = (x / (peak or own_peak) * 0.85).astype(np.float32)
    return (out, own_peak) if return_peak else out


def _drum_hit(kind: str, sr: int, velocity: int, rng: np.random.Generator, tom_hz: float = 120.0) -> np.ndarray:
    v = velocity / 127.0
    if kind == "kick":
        n = int(0.28 * sr); t = np.arange(n) / sr
        f = 60 * np.exp(-t * 20) + 42
        x = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 13)
        click = rng.standard_normal(int(0.005 * sr)) * np.exp(-np.arange(int(0.005 * sr)) / (0.0012 * sr))
        x[: len(click)] += click * 0.7
        x = np.tanh(x * 2.8) * v * 1.1
    elif kind == "snare":
        n = int(0.24 * sr); t = np.arange(n) / sr
        tone = (np.sin(2 * np.pi * 185 * t) + 0.5 * np.sin(2 * np.pi * 330 * t)) * np.exp(-t * 28) * 0.55
        noise = rng.standard_normal(n) * np.exp(-t * 18)
        noise = _filt(noise, *_biquad("highpass", 1800.0, sr, 0.7))
        x = np.tanh((tone + noise * 1.1) * 1.9) * v
    elif kind == "sidestick":
        n = int(0.08 * sr); t = np.arange(n) / sr
        x = (np.sin(2 * np.pi * 900 * t) * np.exp(-t * 90) + rng.standard_normal(n) * np.exp(-t * 150) * 0.4) * 0.6 * v
    elif kind == "hat":
        n = int(0.07 * sr); t = np.arange(n) / sr
        x = rng.standard_normal(n) * np.exp(-t * 80)
        x = _filt(x, *_biquad("highpass", 6500.0, sr, 0.7)) * 0.5 * v
    elif kind == "hat_open":
        n = int(0.35 * sr); t = np.arange(n) / sr
        x = rng.standard_normal(n) * np.exp(-t * 9)
        x = _filt(x, *_biquad("highpass", 5500.0, sr, 0.6)) * 0.4 * v
    elif kind == "ride":
        n = int(0.9 * sr); t = np.arange(n) / sr
        x = rng.standard_normal(n) * np.exp(-t * 4.5) * 0.5 + np.sin(2 * np.pi * 3200 * t) * np.exp(-t * 6) * 0.25
        x = _filt(x, *_biquad("highpass", 4000.0, sr, 0.6)) * 0.35 * v
    elif kind == "bell":
        n = int(0.6 * sr); t = np.arange(n) / sr
        x = (np.sin(2 * np.pi * 2600 * t) + 0.6 * np.sin(2 * np.pi * 4100 * t)) * np.exp(-t * 7) * 0.3 * v
    elif kind == "crash":
        n = int(1.8 * sr); t = np.arange(n) / sr
        x = rng.standard_normal(n) * np.exp(-t * 2.0)
        x = _filt(x, *_biquad("highpass", 3200.0, sr, 0.5)) * 0.4 * v
    elif kind == "china":
        n = int(1.2 * sr); t = np.arange(n) / sr
        x = rng.standard_normal(n) * np.exp(-t * 3.0)
        x = _filt(x, *_biquad("peak", 2400.0, sr, 0.8, 8.0))
        x = _filt(x, *_biquad("highpass", 1800.0, sr, 0.5)) * 0.45 * v
    elif kind == "splash":
        n = int(0.5 * sr); t = np.arange(n) / sr
        x = rng.standard_normal(n) * np.exp(-t * 7.0)
        x = _filt(x, *_biquad("highpass", 5000.0, sr, 0.6)) * 0.35 * v
    elif kind == "tom":
        n = int(0.35 * sr); t = np.arange(n) / sr
        f = tom_hz * (1 + 0.6 * np.exp(-t * 25))
        x = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 9)
        x = (np.tanh(x * 2.0) + rng.standard_normal(n) * np.exp(-t * 60) * 0.15) * 0.9 * v
    else:
        return np.zeros(1, np.float32)
    return x.astype(np.float32)


# General MIDI percussion -> (kind, tom pitch)
GM_DRUMS = {35: "kick", 36: "kick", 37: "sidestick", 38: "snare", 40: "snare", 39: "sidestick",
            42: "hat", 44: "hat", 46: "hat_open", 49: "crash", 57: "crash", 52: "china", 55: "splash",
            51: "ride", 59: "ride", 53: "bell", 41: "tom", 43: "tom", 45: "tom", 47: "tom", 48: "tom", 50: "tom"}
TOM_HZ = {41: 80.0, 43: 95.0, 45: 115.0, 47: 135.0, 48: 160.0, 50: 185.0}


def render_drums(drums: list[Note], sr: int, bpm: float = config.BPM, tail_sec: float = 2.0) -> np.ndarray:
    """Procedural kit for the transcription's drum track (full GM map: kick, snare, hats, ride,
    crashes, china, splash, toms). Mono, returned as stereo."""
    rng = np.random.default_rng(0)
    sec16 = 60.0 / bpm / 4.0
    total = int((max([d.end for d in drums] + [0.0]) * sec16 + tail_sec) * sr)
    buf = np.zeros(total + 2 * sr, np.float32)
    cache = {}
    for d in drums:
        kind = GM_DRUMS.get(d.pitch)
        if kind is None:
            continue
        vq = int(d.velocity // 8) * 8
        key = (kind, d.pitch, vq)
        if key not in cache:
            cache[key] = _drum_hit(kind, sr, vq, rng, TOM_HZ.get(d.pitch, 120.0))
        h = cache[key]
        s = int(round(d.start * sec16 * sr))
        e = min(s + len(h), len(buf))
        buf[s:e] += h[: e - s]
    buf = np.tanh(buf * 1.2) * 0.9
    return np.stack([buf, buf], axis=1)[:total]


def render_guitar_double(notes: list[Note], bpm: float, sr: int = 48000, *, amp: bool = True,
                         offset_ms: float = 7.0, seed: int = 0, progress: bool = False) -> np.ndarray:
    """Double-tracked, hard-panned stereo rhythm guitar (how the record is produced).

    Left take: the "Natural" sample set. Right take: the "Tuned" set, played `offset_ms` late
    with humanised velocities and a slightly different amp setting — two performances, one per
    speaker, instead of one sample panned to both.
    """
    rng = np.random.default_rng(seed)
    takes = []
    for k, (sample_set, drive, tight, delay) in enumerate((("Natural", 35.0, 110.0, 0.0), ("Tuned", 31.0, 125.0, offset_ms))):
        eng = BridgeliteEngine(sample_set=sample_set, sample_rate=sr)
        take_notes = []
        for n in notes:
            v = int(np.clip(n.velocity + (rng.integers(-5, 6) if k else 0), 1, 127))
            take_notes.append(Note(n.start, n.duration, n.pitch, v, n.string, n.fret, n.palm_mute, n.bend))
        di = eng.render(take_notes, bpm=bpm, progress=progress)
        x = amp_chain(di, sr, drive=drive, tight_hz=tight) if amp else di
        mono = x.mean(axis=1)
        if delay:
            d = int(delay * sr / 1000)
            mono = np.concatenate([np.zeros(d, np.float32), mono[:-d]]) if d < len(mono) else mono
        takes.append(mono)
    n = min(len(t) for t in takes)
    out = np.stack([takes[0][:n], takes[1][:n]], axis=1).astype(np.float32)
    peak = float(np.abs(out).max()) or 1.0
    return out / peak * 0.85


def render_song(song: Song, notes: list[Note] | None = None, *, engine: BridgeliteEngine | None = None,
                amp: bool = True, drums: bool = True, drum_level: float = 0.7, out_path: Path | None = None,
                progress: bool = True, double: bool = True) -> tuple[np.ndarray, int]:
    """Render `notes` (default: the song's own guitar part) through 8ridge lite (+ amp, + drums).

    ``double=True`` (default) renders two takes hard-panned L/R; ``double=False`` renders the
    single stereo sample through the given engine.
    """
    engine = engine or BridgeliteEngine()
    notes = song.notes if notes is None else notes
    sr = engine.sample_rate
    if double and engine.sample_set == "Natural" and engine.mono == 0:
        gtr = render_guitar_double(notes, song.bpm, sr, amp=amp, progress=progress)
    else:
        di = engine.render(notes, bpm=song.bpm, progress=progress)
        gtr = amp_chain(di, sr) if amp else di
    mix = gtr.copy()
    if drums and song.drums:
        dr = render_drums(song.drums, sr, song.bpm)
        n = min(len(mix), len(dr))
        mix[:n] += dr[:n] * drum_level
    peak = float(np.abs(mix).max()) or 1.0
    if peak > 0.98:
        mix *= 0.98 / peak
    if out_path:
        import soundfile as sf
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), mix, sr, subtype="PCM_24")
    return mix, sr
