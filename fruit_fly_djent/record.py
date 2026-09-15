"""The real recording as a backing stem (user-supplied), aligned to the transcription.

Put your own copy of the song at ``data/rational_gaze_original.mp3`` (or .wav / .flac / .ogg —
the file is never downloaded by this project). ``align()`` estimates where bar 1 starts in the
recording and its exact tempo by cross-correlating the recording's onset-strength envelope with
the transcription's onset train rendered at candidate tempos; the fly's guitar and the drum stem
are then rendered at that tempo so everything stays in sync for the whole song.

Stems (``output/stems/``): ``guitar.wav`` (the fly through 8ridge lite + amp), ``drums.wav``
(the transcription's drum track), ``record.wav`` (the recording, resampled and shifted so that
t = 0 is bar 1). The player and the stage mix them with independent gains: turn the record off
to hear the fly alone, or the fly off to hear how the record sits against the drums.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config
from .transcription import Note, Song

RECORD_CANDIDATES = [config.DATA_DIR / f"rational_gaze_{kind}{ext}" for kind in ("backing", "original")
                     for ext in (".wav", ".flac", ".mp3", ".ogg", ".aiff")]
ALIGN_PATH = config.CACHE_DIR / "record_align.json"
STEM_DIR = config.OUTPUT_DIR / "stems"


def find_record() -> Path | None:
    for p in RECORD_CANDIDATES:
        if p.exists():
            return p
    return None


def load_record(path: Path, sr: int = 48000) -> np.ndarray:
    import soundfile as sf
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    audio = audio[:, :2]
    if rate != sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, rate)
        audio = resample_poly(audio, sr // g, rate // g, axis=0).astype(np.float32)
    return audio


# ------------------------------------------------------------------------------------------
# onset envelopes
# ------------------------------------------------------------------------------------------
def onset_envelope(audio: np.ndarray, sr: int, hop: int = 512, n_fft: int = 2048) -> tuple[np.ndarray, float]:
    """Spectral-flux onset strength (half-wave rectified log-magnitude increase), hop-rate in Hz."""
    x = audio.mean(axis=1) if audio.ndim == 2 else audio
    win = np.hanning(n_fft).astype(np.float32)
    n_frames = max(1, (len(x) - n_fft) // hop)
    prev = None
    env = np.zeros(n_frames, np.float32)
    for i in range(n_frames):
        seg = x[i * hop:i * hop + n_fft] * win
        mag = np.log1p(50.0 * np.abs(np.fft.rfft(seg)))
        if prev is not None:
            env[i] = np.maximum(mag - prev, 0).sum()
        prev = mag
    env -= np.convolve(env, np.ones(64) / 64, mode="same")      # remove slow trend
    env = np.maximum(env, 0)
    env /= (env.std() + 1e-9)
    return env, sr / hop


def onset_train(song: Song, notes: list[Note], bpm: float, rate: float, length: int) -> np.ndarray:
    """Impulse train (smoothed) of guitar + kick/snare onsets at `bpm`, sampled at `rate` Hz."""
    sec16 = config.sixteenth_seconds(bpm)
    tr = np.zeros(length, np.float32)
    for n in notes:
        k = int(round(n.start * sec16 * rate))
        if 0 <= k < length:
            tr[k] += 1.0
    for d in song.drums:
        if d.pitch in (35, 36, 38, 40):
            k = int(round(d.start * sec16 * rate))
            if 0 <= k < length:
                tr[k] += 0.7
    kern = np.exp(-0.5 * (np.arange(-3, 4) / 1.2) ** 2).astype(np.float32)
    tr = np.convolve(tr, kern, mode="same")
    tr -= tr.mean()
    return tr / (tr.std() + 1e-9)


def align(record: np.ndarray, sr: int, song: Song, notes: list[Note], *, bpm_range=(126.0, 140.0),
          bpm_step: float = 0.25, max_lead_s: float = 40.0, verbose: bool = True) -> dict:
    """Return {'offset_s': where bar 1 starts in the recording, 'bpm': tempo, 'score': corr}."""
    env, rate = onset_envelope(record, sr)
    L = len(env)
    best = {"score": -1.0}
    n_fft = 1 << int(np.ceil(np.log2(2 * L)))
    E = np.fft.rfft(env, n_fft)
    max_lag = int(max_lead_s * rate)
    for bpm in np.arange(bpm_range[0], bpm_range[1] + 1e-9, bpm_step):
        tr = onset_train(song, notes, float(bpm), rate, L)
        c = np.fft.irfft(E * np.conj(np.fft.rfft(tr, n_fft)), n_fft)[: max_lag] / L   # lag >= 0: record leads
        k = int(np.argmax(c))
        if c[k] > best["score"]:
            best = {"score": float(c[k]), "bpm": float(bpm), "offset_s": k / rate}
    if verbose:
        print(f"[record] aligned: bar 1 at {best['offset_s']:.2f}s of the recording, tempo {best['bpm']:.2f} BPM "
              f"(corr {best['score']:.3f})", flush=True)
    return best


def get_alignment(song: Song, notes: list[Note], force: bool = False, sr: int = 48000) -> dict | None:
    path = find_record()
    if path is None:
        return None
    if ALIGN_PATH.exists() and not force:
        cached = json.loads(ALIGN_PATH.read_text())
        if cached.get("file") == str(path):
            return cached
    rec = load_record(path, sr)
    a = align(rec, sr, song, notes)
    a["file"] = str(path)
    ALIGN_PATH.write_text(json.dumps(a, indent=1))
    return a


# ------------------------------------------------------------------------------------------
# stems
# ------------------------------------------------------------------------------------------
def render_stems(song: Song, notes: list[Note], *, bpm: float | None = None, out_dir: Path = STEM_DIR,
                 amp: bool = True, sr: int = 48000, verbose: bool = True) -> dict:
    """Render guitar + drums (+ shifted record) stems at `bpm` (default: aligned tempo if a record exists)."""
    import soundfile as sf
    from .bridgelite import render_drums, render_guitar_double

    out_dir.mkdir(parents=True, exist_ok=True)
    alignment = get_alignment(song, notes)
    bpm = bpm or (alignment["bpm"] if alignment else song.bpm)
    gtr = render_guitar_double(notes, bpm, sr, amp=amp)
    sf.write(str(out_dir / "guitar.wav"), gtr, sr, subtype="PCM_24")
    dr = render_drums(song.drums, sr, bpm) * 0.8 if song.drums else np.zeros((len(gtr), 2), np.float32)
    sf.write(str(out_dir / "drums.wav"), dr, sr, subtype="PCM_24")
    stems = {"bpm": bpm, "guitar": str(out_dir / "guitar.wav"), "drums": str(out_dir / "drums.wav")}
    if alignment:
        rec = load_record(Path(alignment["file"]), sr)
        off = int(round(alignment["offset_s"] * sr))
        rec = rec[off:] if off >= 0 else np.vstack([np.zeros((-off, 2), np.float32), rec])
        peak = float(np.abs(rec).max()) or 1.0
        rec = rec / peak * 0.9
        sf.write(str(out_dir / "backing.wav"), rec, sr, subtype="PCM_24")
        stems["backing"] = str(out_dir / "backing.wav")
        stems["offset_s"] = alignment["offset_s"]
        stems["backing_file"] = alignment["file"]
    (out_dir / "stems.json").write_text(json.dumps(stems, indent=1))
    # the full mix: fly guitar + backing track when we have one, else fly guitar + synth drums
    n = len(gtr)
    mix = gtr.copy()
    if "backing" in stems:
        rec = sf.read(stems["backing"], dtype="float32", always_2d=True)[0]
        m = min(n, len(rec)); mix[:m] += rec[:m] * 0.9
    else:
        m = min(n, len(dr)); mix[:m] += dr[:m]
    peak = float(np.abs(mix).max()) or 1.0
    if peak > 0.98:
        mix *= 0.98 / peak
    sf.write(str(config.OUTPUT_DIR / "fruit_fly_djent_rational_gaze_8ridgelite.wav"), mix, sr, subtype="PCM_24")
    stems["mix"] = str(config.OUTPUT_DIR / "fruit_fly_djent_rational_gaze_8ridgelite.wav")
    (out_dir / "stems.json").write_text(json.dumps(stems, indent=1))
    if verbose:
        print(f"[stems] bpm {bpm:.2f}: " + ", ".join(k for k in ("guitar", "drums", "backing") if k in stems)
              + f" -> {out_dir}; mix -> {stems['mix']}", flush=True)
    return stems


def load_stems(out_dir: Path = STEM_DIR) -> dict:
    """{'bpm', 'sr', 'guitar': array, 'drums': array, 'record': array|None}"""
    import soundfile as sf
    meta = json.loads((out_dir / "stems.json").read_text())
    out = {"bpm": meta["bpm"]}
    sr = None
    for k in ("guitar", "drums", "backing"):
        if k in meta and Path(meta[k]).exists():
            a, sr = sf.read(meta[k], dtype="float32", always_2d=True)
            out[k] = a
    out["sr"] = sr
    return out
