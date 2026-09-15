# FlyBrain Composer — teaching the MaleCNS connectome to play Meshuggah's "Rational Gaze"

> Status: **built** (2026-09-14) — see `README.md` for the pipeline, commands and results. This file is the original design document; deviations: a Fourier phase code + form/section inputs were added to the meter streams (needed for a full-rank reservoir), the transcription source is the tab-only PDF (OMR in `pdftab.py`) with GP7 `.gp` support, and three extras were added on top of the plan: a faithful port of the 8ridge lite sampler, a Brian2 spiking simulation shown live on NAVis skeletons, and a three.js stage where a NeuroMechFly body plays an Ibanez M8M.

## 1. The idea

Use the real synaptic wiring diagram of a male fruit fly's central nervous system (the **MaleCNS**
connectome) as a fixed recurrent "reservoir" — a complex nonlinear dynamical system. Feed it input,
read its activity back out, and shape *what goes in* and *what gets read out* so the output is music
— specifically, an attempt to get it to play Meshuggah's **"Rational Gaze"**.

The connectome itself is never modified — it's a real, already-reconstructed piece of anatomy. All the
"learning" happens in a thin trainable layer wrapped around it (input encoding, output decoding, and the
optimization that tunes them). This is standard **reservoir computing / echo-state-network** practice,
repurposed onto real biological wiring instead of a random matrix.

## 2. Background: what MaleCNS is

- A complete EM (electron microscopy) reconstruction of an adult male *Drosophila melanogaster*'s
  **brain + ventral nerve cord** — ~166,000 neurons, tens of millions of synapses.
- Produced by HHMI Janelia's FlyEM team, the Cambridge Connectomics Group, and Google Research.
  Paper in *Cell*, v1.0 released mid-2026.
- Served via **neuPrint** (neo4j graph DB) at `neuprint.janelia.org`, queryable via the
  `neuprint-python` client or the `malecns`/`malevnc` R packages (natverse).
- Each neuron = a node (`bodyId`, `type`, `instance`, predicted neurotransmitter, synapse counts).
  Each connection = a directed, weighted edge (synapse count). Mentally: one big weighted directed graph.
- License: **CC-BY** — cite the dataset/paper if this is ever shared publicly.
- User already has a neuPrint API token — no account setup needed.

References found during research:
- [Male CNS Connectome | Janelia Research Campus](https://www.janelia.org/project-team/flyem/male-cns-connectome)
- [MaleCNS connectome downloads](https://male-cns.janelia.org/download/)
- [natverse/malecns](https://github.com/natverse/malecns), [natverse/malevnc](https://github.com/natverse/malevnc)
- [neuprint-python](https://github.com/connectome-neuprint/neuprint-python), [quickstart docs](https://connectome-neuprint.github.io/neuprint-python/docs/quickstart.html)

## 3. Why "Rational Gaze" specifically

Confirmed via research: the song is a genuine **polymeter**, not just an odd time signature — the drums
sit in a steady 4/4 while the guitar riff cycles in **25/16** against it, realigning only after several
bars. That's a strong, well-documented test case for a meter-driven reservoir.

- [Pieslak — "Re-casting Metal: Rhythm and Meter in the Music of Meshuggah"](https://www.academia.edu/11931494/Jonathan_Pieslak_Re_casting_Metal_Rhythm_and_Meter_in_the_Music_of_Meshuggah)
- [SevenString.org — Rational Gaze time signature thread](https://sevenstring.org/threads/rational-gaze-by-messhugah-time-signature.167235/)
- [Analysis – The Rat and Rational Gaze](https://philarbonblog.wordpress.com/2017/03/18/analysis1/)

## 4. Training strategy — two stages, two different mechanisms

**Stage A — supervised fit ("tracing paper").** We already have the real song as ground truth, so use
dense per-timestep supervision, not a vague end-of-song reward:
1. Drive the reservoir with **two independent periodic input streams** into two separate input-neuron
   populations — a 4/4 pulse (drums) and a 25/16 pulse (riff). This is the key song-specific design
   choice: flattening it into a single metronome would lose the polymeter that makes this song what it is.
2. Collect reservoir states `X` (neurons × time) while driven.
3. Fit the readout via **ridge regression**: `W_out = Y Xᵀ (XXᵀ + λI)⁻¹`, where `Y` is the real
   transcribed guitar part (pitch, onset, duration, velocity/accent). Closed-form, fast, no backprop
   through the reservoir needed.
4. Generate: drive with the same (or extended) pulse pair, run the reservoir, apply `W_out`, quantize to
   MIDI note events.

This is the step that makes it actually play the real song.

**Stage B — reward-modulated improvisation ("good-dog treats"), layered on top afterward.** Once Stage A
reliably reproduces the riff:
- Nice biological tie-in: MaleCNS has annotated dopaminergic neuron types (PAM cluster = reward/
  appetitive, PPL1 cluster = punishment/aversive) that in the real fly modulate mushroom-body output in
  response to a valence signal — the real analog of "reward." Use these (or whichever equivalent types
  are present in the chosen subgraph) as the injection site for a scalar reward signal.
- Mechanism: perturb-and-reinforce (an evolution-strategy / node-perturbation update). Each episode, add
  small random noise to the trainable parameters (input gains, readout weights), score the output against
  a "does this still sound like djent" fitness (syncopation/autocorrelation match to the 25/16 grouping,
  pitch concentration on low strings, chug/sustain ratio, etc.), and nudge parameters toward whatever
  noise direction scored higher.
- Purpose: NOT to re-fit the notes (Stage A already did that) — it teaches the system to deviate from the
  memorized riff in ways that still groove, i.e. improvise instead of just replaying a recording.

Why this split and not reward-only from the start: reward/RL learning is the right tool when there's no
ground truth to regress against. Here there is (the real transcription), so using only a scalar end-of-
song reward to fit an entire precise, multi-minute performance would be far slower and noisier than direct
regression. Reward-shaping earns its place for the *open-ended* part (Stage B), not the *known-target* part
(Stage A). It's also more biologically honest — real fly DANs handle short associative reinforcement, not
memorizing multi-minute sequences.

## 5. Step-by-step roadmap

1. **Environment**: Python 3.10+ venv. Packages: `neuprint-python`, `numpy`, `scipy`, `pandas`,
   `pyarrow`, `pretty_midi`, `music21`, `pyguitarpro` (parses Guitar Pro files directly, no GP app needed),
   later `cma` (CMA-ES) for Stage B.
2. **Pull the connectome subgraph** (song-independent, do once): connect via `neuprint-python` with the
   existing token, confirm the exact dataset tag (check `Client.fetch_help()` / the web UI — public docs
   are inconsistent on whether it's `male-cns:v1.0` or similar), select a bounded starting subgraph (e.g.
   top-N neurons by total synapse count), cache neuron table + edge list to local parquet. Identify/flag
   candidate PAM/PPL1 (dopaminergic) neuron types in the subgraph for later use in Stage B.
3. **Get "Rational Gaze" into machine-readable form**: find a Guitar Pro transcription
   (`.gp3/.gp4/.gp5/.gpx`), parse with `pyguitarpro` to get exact note onsets/durations/pitches, cross-
   check against Pieslak's published bar-level analysis for correctness. This is the one genuinely
   song-specific data-gathering step.
4. **Build the reservoir** (`reservoir.py`): leaky nonlinear dynamics,
   `x[t+1] = (1-leak)*x[t] + leak*tanh(W @ x[t] + input[t])`, `W` = signed sparse connectome weight matrix
   (sign approximated from predicted neurotransmitter — flag as a simplification).
5. **Build the dual meter-pulse input** (4/4 + 25/16 streams into two input-neuron populations).
6. **Stage A training**: collect states, ridge-regress readout against the real transcription.
7. **Evaluate**: generate MIDI, diff note-for-note against ground truth (onset timing error, pitch
   accuracy), listen to it.
8. **Stage B training** (once Stage A works): perturb-and-reinforce loop through the dopaminergic-neuron
   injection site, fitness = djent-groove heuristics, to get expressive improvisation on top of the
   memorized riff.
9. **Stretch goals**: scale to a larger CNS subgraph, drive from an external audio envelope, real-time
   output via OSC into a DAW instead of offline `.mid` files, widen to a multi-song corpus for genre-level
   generalization.

## 6. Proposed project layout

```
Fruit Fly/
  README.md
  requirements.txt
  .env.example                    # NEUPRINT_APPLICATION_CREDENTIALS placeholder — never commit the real token
  flybrain_composer/
    __init__.py
    connectome.py                 # neuprint queries -> cached neuron table + sparse weight matrix
    reservoir.py                  # leaky nonlinear dynamics over the connectome graph
    meter.py                      # builds the 4/4 + 25/16 dual pulse input streams
    transcription.py              # parses the Guitar Pro file into a target note/onset matrix
    train_supervised.py           # Stage A: ridge regression readout fit
    train_reward.py               # Stage B: perturb-and-reinforce improvisation shaping
    fitness.py                    # djent-groove scoring functions used by Stage B
    sonify.py                     # activity/readout -> MIDI events -> .mid file
    cli.py                        # `python -m flybrain_composer.cli pull|fit|compose|shape`
  data/
    cache/                        # parquet files (gitignored)
    rational_gaze.gp5             # source transcription (check licensing before sharing)
  notebooks/
    01_explore_dataset.ipynb
    02_reservoir_sanity.ipynb
  output/
    *.mid                         # generated performances (gitignored)
```

## 7. Caveats to keep in mind

- Confirm the exact neuPrint dataset tag for MaleCNS once connected — public docs are inconsistent.
- Neurotransmitter-based excitatory/inhibitory sign assignment is a simplification (real sign also
  depends on receptor type); note this in code comments and README, not as ground truth.
- Dataset license is CC-BY — cite the MaleCNS paper/dataset if this project is ever shared publicly.
- The Guitar Pro transcription of "Rational Gaze" is someone else's copyrighted fan transcription of a
  copyrighted song. Fine for personal experimentation; don't redistribute it, and be mindful if sharing
  generated output that closely reproduces the original riff.

## 8. Verification checkpoints

- `pull`: connects with the token, prints neuron/edge counts for the cached subgraph.
- `01_explore_dataset.ipynb`: runs top-to-bottom, plots the selected subgraph.
- `02_reservoir_sanity.ipynb`: raster plot shows non-degenerate activity (not all-zero, not saturated)
  under the dual 4/4 + 25/16 drive.
- `fit`: Stage A ridge regression converges; generated MIDI's note-onset timing error against the real
  transcription is small and audibly recognizable as the riff.
- `shape`: Stage B loop runs without error and the fitness score improves over generations; output still
  sounds djent-like after improvised deviation from the memorized part.
