"""Assemble a *static* Hugging Face Space (free tier): the explainer page, the generated-riff previews and
the run-it-yourself instructions. Hosting a Gradio Space (even on cpu-basic) needs a PRO subscription;
static Spaces are free for everyone.

    python hf_space/build_static.py [--out build/space_static]
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

README = """---
title: Fruit Fly Djent — a fruit fly's brain plays djent
emoji: 🪰
colorFrom: purple
colorTo: red
sdk: static
app_file: index.html
pinned: false
license: gpl-3.0
short_description: A fruit fly's connectome plays and improvises djent
---

# 🪰 Fruit Fly Djent

The real synaptic wiring of a male fruit fly (**MaleCNS**, Janelia FlyEM — 2,318 central-brain neurons,
236,870 connections, never modified) is used as a fixed *reservoir*: a read-out fitted to a tab replays
Meshuggah's *Rational Gaze* note-for-note (onset F1 0.99), a reward-modulated search improvises on it live
with the treat injected into the fly's real dopaminergic neurons, a read-out fitted to riffs from 19 djent
tabs makes up riffs of its own, the same neurons spike in a simulation drawn on their real 3D skeletons,
and a scanned fly plays it all on an Ibanez M8M on a three.js stage.

This static page holds the explainer (pipeline, every step, tools, build log with proof) and riffs the fly
made up. The show itself needs a desktop with a sound card and a GPU — code, setup and licences:
**https://github.com/SyhmZlkrn/Djent-Fruit-Fly**

Attribution: MaleCNS v1.0 (Janelia FlyEM, CC BY 4.0) via neuPrint · NeuroMechFly (EPFL, Apache-2.0) ·
"Ibanez M8M" by cherepah (CC BY 4.0) · 8ridge lite by James Stubbs (GPL-3.0) · three.js (MIT). Code GPL-3.0.
"""

HERO = """
<section style="margin:0 0 8px">
  <div class="fig" style="padding:20px 22px">
    <div class="eyebrow">Listen · riffs the fly made up</div>
    <h2 style="margin:6px 0 10px">Its own riffs, from 19 djent tabs</h2>
    <p class="prose" style="margin-bottom:12px">The multi-song read-out (fitted to the 16 most characteristic bars of each tab) is driven with a bar grid, a riff cycle and a blend of two learned riff identities; nine musical knobs are searched by reward. Drums lock to the riff: hats keep time, the kick doubles every chug, the snare on 2 &amp; 4 or in thirds. Every seed is a different take.</p>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px">
      <figure style="margin:0"><figcaption style="margin:0 0 4px;font-size:13px">48 bars · 19-tab corpus · 140 BPM · 23/16 · snare 2&amp;4</figcaption><audio controls preload="none" style="width:100%" src="audio/preview_djent_19songs_seed7.mp3"></audio></figure>
      <figure style="margin:0"><figcaption style="margin:0 0 4px;font-size:13px">40 bars · 11-tab corpus · 140 BPM · 23/16</figcaption><audio controls preload="none" style="width:100%" src="audio/preview_djent_seed5.mp3"></audio></figure>
      <figure style="margin:0"><figcaption style="margin:0 0 4px;font-size:13px">32 bars · 140 BPM · 23/16 · snare 2&amp;4</figcaption><audio controls preload="none" style="width:100%" src="audio/preview_djent_seed3.mp3"></audio></figure>
      <figure style="margin:0"><figcaption style="margin:0 0 4px;font-size:13px">32 bars · 140 BPM · 25/16 · snare in thirds</figcaption><audio controls preload="none" style="width:100%" src="audio/preview_djent_seed4.mp3"></audio></figure>
    </div>
    <p class="note" style="margin-top:14px"><b>Run it yourself</b> (desktop; free): <code>git clone https://github.com/SyhmZlkrn/Djent-Fruit-Fly</code> → <code>pip install -r requirements.txt</code> → put Guitar Pro tabs in <code>data/songs/</code> → <code>python -m fruit_fly_djent.cli fit-multi</code> → <code>python -m fruit_fly_djent.cli generate --bars 32 --bpm 140 --cycle 23 --snare thirds</code>, or the whole live show: <code>python -m fruit_fly_djent.stage_server</code> and press <b>🎸 Own riffs</b> on the page. Hosting the Gradio generator on Hugging Face needs a PRO plan, so the interactive demo is offline — the repo includes it (<code>hf_space/app.py</code>).</p>
  </div>
</section>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "build" / "space_static"))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "audio").mkdir(parents=True)
    (out / "proof").mkdir(parents=True)
    for f in (ROOT / "docs" / "audio").glob("*.mp3"):
        shutil.copy(f, out / "audio" / f.name)
    for f in (ROOT / "docs" / "proof").glob("*.jpg"):
        shutil.copy(f, out / "proof" / f.name)
    html = (ROOT / "docs" / "how_it_works.html").read_text(encoding="utf-8")
    # insert the listening section right after the page header
    html = re.sub(r"(</header>)", r"\1" + HERO, html, count=1)
    # a link back to the repository in the header
    html = html.replace('<div class="eyebrow">How it works · for the video</div>',
                        '<div class="eyebrow">How it works · <a href="https://github.com/SyhmZlkrn/Djent-Fruit-Fly">github.com/SyhmZlkrn/Djent-Fruit-Fly</a></div>', 1)
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "README.md").write_text(README, encoding="utf-8")
    shutil.copy(ROOT / "LICENSE", out / "LICENSE")
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"static space assembled at {out}: {total:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
