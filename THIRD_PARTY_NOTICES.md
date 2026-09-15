# Third-party data, models and code

Fruit Fly Djent is licensed under the **GNU General Public License v3.0** (see `LICENSE`). It builds on the
following work, none of which is ours. Where an asset is redistributed in this repository or the
Hugging Face Space, its licence and attribution are given here.

## Data

| What | Source | Licence | How it is used |
|---|---|---|---|
| **MaleCNS v1.0** connectome — neurons, synaptic connections, neurotransmitter predictions, skeletons, brain-region meshes | Janelia Research Campus FlyEM project, served by [neuPrint](https://neuprint.janelia.org) (dataset `male-cns:v1.0`) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | The fixed reservoir (2,318 central-brain neurons, 236,870 edges) and the 3D brain shown on stage. The subgraph caches (`neurons.parquet`, `edges.parquet`) are redistributed in the Space with this attribution; skeletons are pulled by the user with their own free neuPrint token. Please cite the MaleCNS / neuPrint papers listed on neuprint.janelia.org if you publish results. |
| **NeuroMechFly** fly body (MJCF model, meshes, pose) | [flygym](https://github.com/NeLy-EPFL/flygym), Neuroengineering Lab, EPFL | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | The articulated fly on the three.js stage (`stage-export` converts it to glTF). Not redistributed here; install `flygym` or copy its `data/mjcf` folder into `third_party/neuromechfly/`. |
| **"Ibanez M8M"** 3D model | [cherepah on Sketchfab](https://skfb.ly/pqrZM) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | The guitar the fly plays. The converted glTF (`stage/data/guitar.glb`, `guitar.json`) is redistributed under CC BY 4.0 with this attribution: *"Ibanez M8M" (https://skfb.ly/pqrZM) by cherepah is licensed under Creative Commons Attribution.* |
| **Tabs** of *Rational Gaze* and the other djent songs the read-out learned from | Guitar Pro transcriptions supplied by the author | © their respective rights holders | **Not distributed.** `data/songs/` is empty in the repository; bring your own tabs. |
| **Backing track** of *Rational Gaze* | supplied by the author | © rights holders | **Not distributed.** |

## Code and sound

| What | Source | Licence | How it is used |
|---|---|---|---|
| **8ridge lite** — 8-string guitar sampler plugin: the `Synthesiser`/`SamplerVoice` logic and the 61 + 61 guitar samples | [James Stubbs (JamesStubbsEng/8ridgelite)](https://github.com/JamesStubbsEng/8ridgelite) | [GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html) | `fruit_fly_djent/bridgelite.py` is a Python port of the plugin's sampler engine (hence this project's GPL-3.0 licence). The samples are not in this repository (clone the plugin into `third_party/`); the Space ships them trimmed to 4 s, 16-bit, with the plugin's LICENSE. |
| **three.js** r170 and addons (GLTFLoader, OrbitControls, UnrealBloomPass, …) | [three.js authors](https://threejs.org) | MIT (`stage/vendor/THREE_LICENSE`) | Vendored in `stage/vendor/` for the stage. |
| **Brian2** | Brian2 developers | CeCILL 2.1 | Spiking simulation (dependency). |
| **NAVis**, **octarine**, **pygfx**, **wgpu-py** | their authors | GPL-3.0 / MIT / BSD-2 / BSD-2 | Skeleton processing and the live brain window (dependencies). |
| **neuprint-python**, **NumPy**, **SciPy**, **pandas**, **pyarrow**, **cma**, **PyMuPDF**, **PyGuitarPro**, **pretty_midi**, **mido**, **python-rtmidi**, **soundfile/libsndfile**, **sounddevice/PortAudio**, **websockets**, **trimesh**, **Gradio** | their authors | BSD / MIT / AGPL (PyMuPDF) / LGPL (libsndfile) | Dependencies, installed from PyPI. |

## Music

*Rational Gaze* is written and performed by Meshuggah. The other songs in the training corpus belong to
their writers and publishers. This project reproduces their guitar parts only as a research and
performance experiment; no transcription, recording or trained model that reproduces a copyrighted
song is distributed.
