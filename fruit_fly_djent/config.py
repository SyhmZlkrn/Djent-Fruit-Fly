"""Paths, dataset tags and musical constants shared by every module."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
SKELETON_DIR = DATA_DIR / "skeletons"
OUTPUT_DIR = ROOT / "output"
STAGE_DIR = ROOT / "stage"
THIRD_PARTY = ROOT / "third_party"

for _d in (CACHE_DIR, SKELETON_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")

# --- neuPrint -------------------------------------------------------------
NEUPRINT_SERVER = "neuprint.janelia.org"
# Confirmed against Client.fetch_datasets() on 2026-09-14: 'male-cns:v1.0' (also 'male-cns:v0.9').
NEUPRINT_DATASET = os.environ.get("NEUPRINT_DATASET", "male-cns:v1.0")


def neuprint_token() -> str:
    tok = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
    if not tok:
        raise RuntimeError(
            "No neuPrint token. Put NEUPRINT_APPLICATION_CREDENTIALS=<token> in .env "
            "(see .env.example) or export it in the environment."
        )
    return tok


# --- subgraph selection ---------------------------------------------------
# Central-brain intrinsic neurons ranked by total synapse count, plus *every*
# PAM/PPL1 dopaminergic neuron (the Stage B reward injection site).
SUBGRAPH_N_TOP = int(os.environ.get("FRUIT_FLY_DJENT_N_TOP", 2000))
SUBGRAPH_SUPERCLASSES = ("cb_intrinsic",)
DAN_TYPE_REGEX = r"^(PAM|PPL1)"

# --- music ----------------------------------------------------------------
# "Rational Gaze" (Meshuggah, Nothing, 2002 / 2006 re-record). The transcription says 133 BPM
# (published analyses quote ~135); the 2006 re-record is on 8-strings in F standard. BPM here is
# only the fallback for the built-in structural song — real songs carry their own tempo.
BPM = 133.0
BEATS_PER_BAR = 4
# Reservoir/readout time grid: 4 steps per sixteenth note (= 32nd-note triplet-free 64th grid
# is overkill; 4 sub-steps per 16th gives ~28 ms resolution at 135 BPM).
STEPS_PER_SIXTEENTH = 4
STEPS_PER_BEAT = 4 * STEPS_PER_SIXTEENTH
SIXTEENTH_SECONDS = 60.0 / BPM / 4.0
STEP_SECONDS = SIXTEENTH_SECONDS / STEPS_PER_SIXTEENTH


def sixteenth_seconds(bpm: float) -> float:
    return 60.0 / bpm / 4.0


def step_seconds(bpm: float) -> float:
    return sixteenth_seconds(bpm) / STEPS_PER_SIXTEENTH

# 8-string F standard (low -> high) as MIDI numbers: F1 Bb1 Eb2 Ab2 Db3 Gb3 Bb3 Eb4
TUNING_F_STANDARD = (29, 34, 39, 44, 49, 54, 58, 63)
NUM_FRETS = 24
# 8ridge lite covers MIDI 16..88 (E1 = 28 is the lowest *sampled* note; 16..27 are
# repitched from the E1 sample exactly like the plugin does).
MIDI_LO, MIDI_HI = 28, 88
