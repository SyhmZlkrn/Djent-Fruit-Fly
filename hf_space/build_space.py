"""Assemble the Hugging Face Space folder from this repo (run from the project root):

    python hf_space/build_space.py [--out build/space]

Copies the package, the app, a trimmed copy of the 8ridge lite samples (4 s, 16-bit — the engine reads
them unchanged), the connectome subgraph caches (CC BY 4.0) and the multi-song reader (7 MB), then
writes the Space's README with its front matter. Push the result to the Space with git or
`huggingface_hub.upload_folder`. No tabs, no backing track, no Stage A model.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SECONDS = 4.0

README = """---
title: FlyBrain Composer — riffs from a fruit fly's brain
emoji: 🪰
colorFrom: purple
colorTo: red
sdk: gradio
sdk_version: {gradio_version}
app_file: app.py
pinned: false
license: gpl-3.0
short_description: A fruit fly's connectome makes up djent riffs
---

# 🪰 FlyBrain Composer — riffs from a fruit fly's brain

The real synaptic wiring of a male fruit fly (**MaleCNS**, Janelia FlyEM — 2,318 central-brain neurons,
236,870 connections, never modified) is used as a fixed *reservoir*. A read-out fitted to riffs from djent
tabs decodes the brain's activity into notes. In this demo the brain is driven with a bar grid, a riff
cycle of your choice (the polymeter against 4/4) and a blend of learned riff identities; a small
reward-driven search over nine musical knobs (displacement, cycle stretch, blend, stream gains, dopamine
gain, threshold) picks the groove. Drums lock to the riff: hats keep time, the kick doubles every chug,
the snare sits on 2 & 4 or in thirds. Every seed is a different take.

The full project — Stage A (the fly learns Meshuggah's *Rational Gaze* note-for-note from the tab),
live reward-modulated improvisation with the treat injected into the fly's real dopaminergic PAM/PPL1
neurons, the spiking brain drawn on the real neuron skeletons, the fly playing an Ibanez M8M on a
three.js stage, 👍/👎 from the audience — needs a desktop with a sound card and a GPU:
**https://github.com/SyhmZlkrn/Djent-Fruit-Fly**

Honest framing: a shuffled or random network of the same size reproduces songs just as well (see the
control experiments in the repo). The connectome is the medium the riffs are written on, not the composer.

## Attribution

* Connectome: **MaleCNS v1.0**, Janelia FlyEM, via [neuPrint](https://neuprint.janelia.org) — CC BY 4.0.
* Guitar sound: **[8ridge lite](https://github.com/JamesStubbsEng/8ridgelite)** by James Stubbs — GPL-3.0.
  The sampler logic is ported to Python and the samples are included here trimmed to 4 s; source and
  licence in `third_party/8ridgelite/`.
* Code: GPL-3.0 (see the repository). The tabs the read-out learned from are not distributed.
"""


def trim_samples(src: Path, dst: Path, seconds: float) -> int:
    n = 0
    for set_name in ("Natural", "Tuned"):
        (dst / set_name).mkdir(parents=True, exist_ok=True)
        for wav in sorted((src / set_name).glob("*.wav")):
            data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
            keep = data[: int(seconds * sr)]
            fade = int(0.05 * sr)
            if len(keep) > fade:
                keep[-fade:] *= np.linspace(1.0, 0.0, fade)[:, None]
            sf.write(str(dst / set_name / wav.name), keep, sr, subtype="PCM_16")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "build" / "space"))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    # package (no caches)
    shutil.copytree(ROOT / "flybrain_composer", out / "flybrain_composer", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy(ROOT / "hf_space" / "app.py", out / "app.py")
    shutil.copy(ROOT / "hf_space" / "requirements.txt", out / "requirements.txt")
    shutil.copy(ROOT / "LICENSE", out / "LICENSE")
    # data the generator needs
    cache = out / "data" / "cache"
    cache.mkdir(parents=True)
    for name in ("neurons.parquet", "edges.parquet", "model_multi.npz", "model_multi.json", "corpus.json"):
        shutil.copy(ROOT / "data" / "cache" / name, cache / name)
    (out / "data" / "songs").mkdir(parents=True)
    (out / "data" / "songs" / "README.txt").write_text("The tabs are not distributed. See the GitHub repository.\n")
    # 8ridge lite samples, trimmed
    src = ROOT / "third_party" / "8ridgelite"
    dst = out / "third_party" / "8ridgelite"
    n = trim_samples(src / "8ridgelite_20sec_wav", dst / "8ridgelite_20sec_wav", SAMPLE_SECONDS)
    shutil.copy(src / "LICENSE", dst / "LICENSE")
    (dst / "README.md").write_text("Samples from https://github.com/JamesStubbsEng/8ridgelite (GPL-3.0), trimmed to "
                                   f"{SAMPLE_SECONDS:.0f} s and converted to 16-bit for this demo.\n")
    # the stage: page + scripts + vendored three.js + the shared geometry (fly, brain, regions, guitar)
    stage = out / "stage"
    (stage / "data").mkdir(parents=True)
    for name in ("app.js", "fly.js", "guitar.js", "brain.js"):
        shutil.copy(ROOT / "stage" / name, stage / name)
    shutil.copytree(ROOT / "stage" / "vendor", stage / "vendor")
    html = (ROOT / "stage" / "index.html").read_text(encoding="utf-8")
    html = html.replace('<script type="importmap">', '<script>window.STATIC_STAGE = true;</script>\n<script type="importmap">', 1)
    html = html.replace("<h1><span>FlyBrain Composer</span> — a fly plays Rational Gaze</h1>",
                        "<h1><span>FlyBrain Composer</span> — a fly plays the riff you just made</h1>", 1)
    html = html.replace("<h2>FlyBrain Composer — the stage</h2>", "<h2>Your riff, played by the fly</h2>", 1)
    html = html.replace("plays Meshuggah's <i>Rational Gaze</i> on an 8-string Ibanez M8M.",
                        "plays the riff its brain just made up on an 8-string Ibanez M8M.", 1)
    html = html.replace("The notes were composed by a reservoir built from the MaleCNS connectome; the brain above the stage is the\n       same 2,318 neurons spiking in a Brian2 simulation, replayed in sync with the music.",
                        "The brain above the stage is the same 2,318 neurons spiking in a leaky integrate-and-fire simulation of the real wiring, hearing this riff.", 1)
    (stage / "index.html").write_text(html, encoding="utf-8")
    for name in ("fly.glb", "fly_rig.json", "guitar.glb", "guitar.json", "brain.bin", "brain_owner.bin", "somas.bin", "brain.json", "rois.glb"):
        src_f = ROOT / "stage" / "data" / name
        if src_f.exists():
            shutil.copy(src_f, stage / "data" / name)
        else:
            print(f"[space] missing stage asset {name} — run: python -m flybrain_composer.cli stage-export")
    (out / "runs").mkdir()
    (out / "runs" / ".gitkeep").write_text("")
    import gradio
    (out / "README.md").write_text(README.format(gradio_version=gradio.__version__), encoding="utf-8")
    (out / ".gitattributes").write_text("*.npz filter=lfs diff=lfs merge=lfs -text\n*.parquet filter=lfs diff=lfs merge=lfs -text\n"
                                        "*.wav filter=lfs diff=lfs merge=lfs -text\n*.glb filter=lfs diff=lfs merge=lfs -text\n"
                                        "*.bin filter=lfs diff=lfs merge=lfs -text\n")
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"space assembled at {out}: {n} samples trimmed, {total:.0f} MB total")


if __name__ == "__main__":
    sys.exit(main())
