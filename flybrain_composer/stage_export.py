"""Export everything the three.js stage needs into ``stage/data/``.

* ``fly.glb`` + ``fly_rig.json`` — the NeuroMechFly v2 body (EPFL, Apache-2.0; micro-CT based
  *Drosophila* model shipped with ``flygym``): every MJCF body becomes a named glTF node with the
  same parent/pos/quat, so the page can drive the real joints (coxa yaw/pitch/roll, femur,
  tibia, tarsi, head, wings, antennae...). The rig JSON lists joints/axes per body and the
  tripod stance from ``pose_tripod.yaml``.
* ``brain.bin`` (segment vertices, um, centred) + ``brain_owner.bin`` (neuron index per vertex)
  + ``somas.bin`` + ``rois.glb`` — the MaleCNS subgraph geometry for the holographic brain.
* ``notes.json``, ``sections.json``, ``song.json`` — the fly-brain performance.
* ``spikes.bin`` — Brian2 raster as (float32 t_s, uint16 neuron) pairs.
* ``song.mp3`` — the 8ridge lite render for standalone playback.
"""
from __future__ import annotations

import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from . import config, connectome, skeletons
from .transcription import Song

NMF_DIR = config.THIRD_PARTY / "neuromechfly"
MJCF = NMF_DIR / "mjcf" / "neuromechfly_seqik_kinorder_ypr.xml"
STAGE_DATA = config.STAGE_DIR / "data"


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.replace(",", " ").split()]


def export_fly(out_dir: Path = STAGE_DATA) -> Path:
    import trimesh
    from scipy.spatial.transform import Rotation as R

    root = ET.parse(MJCF).getroot()
    meshes = {m.get("name"): (Path(MJCF).parent / m.get("file")).resolve() for m in root.iter("mesh")}
    scale = {m.get("name"): _floats(m.get("scale", "1 1 1")) for m in root.iter("mesh")}
    scene = trimesh.Scene()
    rig: dict = {"bodies": {}, "order": []}
    worldbody = root.find("worldbody")

    def walk(body, parent_name):
        name = body.get("name")
        pos = _floats(body.get("pos", "0 0 0"))
        w, x, y, z = _floats(body.get("quat", "1 0 0 0"))
        T = np.eye(4)
        T[:3, :3] = R.from_quat([x, y, z, w]).as_matrix()
        T[:3, 3] = pos
        geom = body.find("geom")
        geom_name = None
        if geom is not None and geom.get("mesh") in meshes:
            m = trimesh.load(meshes[geom.get("mesh")], force="mesh", process=False)
            m.apply_scale(scale[geom.get("mesh")])
            gpos = _floats(geom.get("pos", "0 0 0"))
            gw, gx, gy, gz = _floats(geom.get("quat", "1 0 0 0"))
            G = np.eye(4)
            G[:3, :3] = R.from_quat([gx, gy, gz, gw]).as_matrix()
            G[:3, 3] = gpos
            m.apply_transform(G)
            geom_name = f"geom_{name}"
            scene.add_geometry(m, node_name=name, geom_name=geom_name, parent_node_name=parent_name, transform=T)
        else:
            scene.graph.update(frame_to=name, frame_from=parent_name, matrix=T)
        joints = [{"name": j.get("name"), "axis": _floats(j.get("axis", "0 1 0"))} for j in body.findall("joint")]
        rig["bodies"][name] = {"parent": parent_name, "pos": pos, "quat": [x, y, z, w], "joints": joints,
                              "mesh": geom_name is not None}
        rig["order"].append(name)
        for child in body.findall("body"):
            walk(child, name)

    for b in worldbody.findall("body"):
        walk(b, "world")

    out_dir.mkdir(parents=True, exist_ok=True)
    glb = out_dir / "fly.glb"
    scene.export(str(glb))
    # stance
    pose = {}
    ppath = NMF_DIR / "pose" / "pose_tripod.yaml"
    if ppath.exists():
        for line in ppath.read_text().splitlines():
            line = line.strip()
            if line.startswith("joint_") and ":" in line:
                k, v = line.split(":", 1)
                try:
                    pose[k.strip()] = float(v)
                except ValueError:
                    pass
    rig["pose_tripod_deg"] = pose
    rig["units"] = "mm"
    rig["source"] = "NeuroMechFly v2 (flygym, EPFL, Apache-2.0) — MJCF neuromechfly_seqik_kinorder_ypr.xml"
    (out_dir / "fly_rig.json").write_text(json.dumps(rig))
    print(f"[stage] fly: {len(rig['order'])} bodies -> {glb} ({glb.stat().st_size / 1e6:.1f} MB)", flush=True)
    return glb


def export_brain(out_dir: Path = STAGE_DATA, max_roi_faces: int = 6000) -> None:
    from .neuroviz import BrainScene
    sc = BrainScene()
    out_dir.mkdir(parents=True, exist_ok=True)
    pos = sc.positions            # (V, 3) with NaN separators every 3rd vertex
    owner = sc.owner
    keep = ~np.isnan(pos[:, 0])
    seg_pos = pos[keep].astype(np.float32)          # pairs a, b
    seg_owner = owner[keep].astype(np.uint16)
    seg_pos.tofile(out_dir / "brain.bin")
    seg_owner.tofile(out_dir / "brain_owner.bin")
    sc.soma.astype(np.float32).tofile(out_dir / "somas.bin")
    meta = {
        "n_neurons": int(sc.N), "n_vertices": int(len(seg_pos)),
        "dan": np.nonzero(sc.dan)[0].tolist(),
        "types": sc.neurons["type"].fillna("").tolist(),
        "nt": sc.neurons["consensusNt"].fillna("").tolist(),
        "bodyIds": sc.neurons["bodyId"].astype(int).tolist(),
    }
    (out_dir / "brain.json").write_text(json.dumps(meta))
    # region meshes (decimated) as one GLB
    try:
        import trimesh
        scene = trimesh.Scene()
        for name, v, f in sc.rois:
            m = trimesh.Trimesh(v, f, process=False)
            if len(m.faces) > max_roi_faces:
                try:
                    m = m.simplify_quadric_decimation(max_roi_faces)
                except Exception:  # noqa: BLE001 — fast_simplification not installed
                    step = int(np.ceil(len(m.faces) / max_roi_faces))
                    m = trimesh.Trimesh(m.vertices, m.faces[::step], process=False)
            scene.add_geometry(m, node_name=name, geom_name=name)
        scene.export(str(out_dir / "rois.glb"))
    except Exception as e:  # noqa: BLE001
        print(f"[stage] rois skipped: {e}", flush=True)
    print(f"[stage] brain: {sc.N} neurons, {len(seg_pos) // 2} segments, {len(sc.rois)} rois", flush=True)


def export_song(song: Song, notes, out_dir: Path = STAGE_DATA, wav: Path | None = None,
                with_audio: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sec = config.sixteenth_seconds(song.bpm)
    # fingering: the transcription's own string/fret for the same pitch at the same beat, so the
    # fly frets the way the tab is written (low F1 chugs open, the F#2 leap at fret 13)
    truth = {}
    for n in song.notes:
        if n.string is not None and n.fret is not None:
            truth[(round(n.start * 2), n.pitch)] = (n.string, n.fret, n.palm_mute, n.bend)
    rows = []
    for n in sorted(notes, key=lambda n: n.start):
        key = (round(n.start * 2), n.pitch)
        alt = [truth[k] for k in (key, (key[0] - 1, n.pitch), (key[0] + 1, n.pitch)) if k in truth]
        string, fret, pm, bend = alt[0] if alt else (n.string, n.fret, n.palm_mute, n.bend)
        rows.append({"t": round(n.start * sec, 4), "dur": round(n.duration * sec, 4), "pitch": n.pitch,
                     "string": string, "fret": fret, "vel": n.velocity, "pm": bool(pm), "bend": bend})
    (out_dir / "notes.json").write_text(json.dumps(rows))
    (out_dir / "drums.json").write_text(json.dumps([
        {"t": round(d.start * sec, 4), "pitch": d.pitch, "vel": d.velocity}
        for d in sorted(song.drums, key=lambda d: d.start)]))
    (out_dir / "sections.json").write_text(json.dumps([
        {"name": s.name, "t0": round(s.start * sec, 3), "t1": round(s.end * sec, 3)} for s in song.sections]))
    (out_dir / "song.json").write_text(json.dumps({
        "title": song.title, "bpm": song.bpm, "duration": song.seconds, "tuning": list(song.tuning),
        "n_notes": len(notes)}))
    if with_audio:
        import soundfile as sf
        from .record import STEM_DIR
        written = []
        stems_meta = {}
        if (STEM_DIR / "stems.json").exists():
            stems_meta = json.loads((STEM_DIR / "stems.json").read_text())
        for name in ("guitar", "drums", "backing"):
            src = stems_meta.get(name)
            if src and Path(src).exists():
                audio, sr = sf.read(src, dtype="float32", always_2d=True)
                try:   # libsndfile >= 1.1 encodes MP3 (its Vorbis encoder produced empty files here)
                    sf.write(str(out_dir / f"stem_{name}.mp3"), audio, sr, format="MP3", subtype="MPEG_LAYER_III")
                    written.append(name)
                except Exception as e:  # noqa: BLE001
                    print(f"[stage] mp3 failed for {name}: {e}", flush=True)
        if wav and Path(wav).exists():
            audio, sr = sf.read(str(wav), dtype="float32", always_2d=True)
            try:
                sf.write(str(out_dir / "song.mp3"), audio, sr, format="MP3", subtype="MPEG_LAYER_III")
                written.append("mix")
            except Exception as e:  # noqa: BLE001
                sf.write(str(out_dir / "song.wav"), audio, sr, subtype="PCM_16")
                print(f"[stage] mp3 failed ({e}); wrote song.wav", flush=True)
        (out_dir / "stems.json").write_text(json.dumps({"stems": [w for w in written if w != "mix"], "mix": "mix" in written,
                                                        "bpm": stems_meta.get("bpm", song.bpm)}))
        print(f"[stage] audio stems: {written}", flush=True)


def export_spikes(out_dir: Path = STAGE_DATA) -> None:
    from .snn import load_raster
    r = load_raster()
    order = np.argsort(r["t_ms"].to_numpy(), kind="stable")
    t = (r["t_ms"].to_numpy()[order] / 1000.0).astype(np.float32)
    n = r["neuron"].to_numpy()[order].astype(np.uint16)
    out_dir.mkdir(parents=True, exist_ok=True)
    t.tofile(out_dir / "spikes_t.bin")
    n.tofile(out_dir / "spikes_n.bin")
    print(f"[stage] spikes: {len(t)} -> spikes_t.bin/spikes_n.bin", flush=True)


def export_all(gp=None, track="Rhythm", midi=None, wav=None, with_audio=True) -> None:
    from .sonify import midi_to_notes
    from .transcription import load_song
    song = load_song(gp, track=track)
    notes = midi_to_notes(midi, bpm=song.bpm)[0] if midi and Path(midi).exists() else song.notes
    export_fly()
    export_guitar()
    export_brain()
    export_song(song, notes, wav=wav, with_audio=with_audio)
    try:
        export_spikes()
    except FileNotFoundError as e:
        print(f"[stage] {e}", flush=True)


if __name__ == "__main__":
    export_all()


# ------------------------------------------------------------------------------------------
# Ibanez M8M mesh (user-supplied OBJ + PBR textures) -> stage/data/guitar.glb
# ------------------------------------------------------------------------------------------
M8M_DIR = config.THIRD_PARTY / "ibanez-m8m" / "source"
# landmarks measured on the mesh (metres, model frame: +x neck, +y face, z across strings, -z bass side)
M8M_BRIDGE_X, M8M_NUT_X, M8M_STRING_Y = -0.105, 0.636, 0.0345
M8M_SPACING_BRIDGE, M8M_SPACING_NUT = 0.0109, 0.0070
# fret wires 1..24 and the nut, measured by ray-casting the fretboard profile of the mesh (metres)
M8M_FRETS_X = [0.589, 0.549, 0.5115, 0.4758, 0.4423, 0.4108, 0.3813, 0.3535, 0.3273, 0.3023, 0.2788, 0.2568,
               0.2363, 0.2168, 0.1983, 0.1808, 0.1643, 0.1488, 0.1343, 0.1205, 0.1075, 0.0953, 0.0838, 0.0728]
M8M_NUT_MEASURED = 0.6335
STAGE_PER_METRE = 1000.0 / 650.0            # 1 human mm == 1/650 stage mm (same K as guitar.js)


def export_guitar(out_dir: Path = STAGE_DATA, tex_size: int = 2048) -> Path | None:
    """Convert the M8M OBJ into a glTF in the stage's guitar frame: bridge saddles at the origin,
    +x toward the nut, +y toward the bass side, +z out of the face; string top plane at z = 0."""
    obj = M8M_DIR / "IbanezM8M.obj"
    if not obj.exists():
        print("[stage] no third_party/ibanez-m8m/source/IbanezM8M.obj — the procedural M8M will be used", flush=True)
        return None
    import trimesh
    from PIL import Image

    mesh = trimesh.load(str(obj), force="mesh", process=False)
    uv = mesh.visual.uv if hasattr(mesh.visual, "uv") else None
    # frame change: q = R (p - c) * s  with R rows (x, -z, y)
    c = np.array([M8M_BRIDGE_X, M8M_STRING_Y, 0.0])
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    T = np.eye(4)
    T[:3, :3] = R * STAGE_PER_METRE
    T[:3, 3] = -(R @ c) * STAGE_PER_METRE
    mesh.apply_transform(T)

    def tex(name, mode="RGB"):
        im = Image.open(M8M_DIR / f"IbanezM8M_{name}.png").convert(mode)
        return im.resize((tex_size, tex_size), Image.LANCZOS)

    base = tex("BaseColor")
    rough = np.asarray(tex("Roughness", "L"))
    metal = np.asarray(tex("Metallic", "L"))
    mr = Image.fromarray(np.stack([np.full_like(rough, 255), rough, metal], axis=-1), "RGB")   # glTF: G=roughness, B=metallic
    normal = tex("Normal")
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=base, metallicRoughnessTexture=mr, normalTexture=normal,
        metallicFactor=1.0, roughnessFactor=1.0, name="IbanezM8M")
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    out_dir.mkdir(parents=True, exist_ok=True)
    glb = out_dir / "guitar.glb"
    mesh.export(str(glb))
    meta = {
        "source": "user-supplied Ibanez M8M OBJ (third_party/ibanez-m8m)",
        "scale_length": (M8M_NUT_X - M8M_BRIDGE_X) * STAGE_PER_METRE,
        "spacing_bridge": M8M_SPACING_BRIDGE * STAGE_PER_METRE, "spacing_nut": M8M_SPACING_NUT * STAGE_PER_METRE,
        "pickup_x": 0.02 * STAGE_PER_METRE, "board_top": -0.0015 * STAGE_PER_METRE,
        "frets": [(x - M8M_BRIDGE_X) * STAGE_PER_METRE for x in M8M_FRETS_X],
        "nut": (M8M_NUT_MEASURED - M8M_BRIDGE_X) * STAGE_PER_METRE,
        "body_depth": 0.055 * STAGE_PER_METRE,
        "strap_buttons": [((0.251 - M8M_BRIDGE_X) * STAGE_PER_METRE, 0.10 * STAGE_PER_METRE, -0.02 * STAGE_PER_METRE),
                          ((-0.204 - M8M_BRIDGE_X) * STAGE_PER_METRE, 0.0, -0.02 * STAGE_PER_METRE)],
        "bounds": np.round(mesh.bounds, 4).tolist(),
    }
    (out_dir / "guitar.json").write_text(json.dumps(meta))
    print(f"[stage] guitar: {len(mesh.faces)} faces -> {glb} ({glb.stat().st_size / 1e6:.1f} MB); "
          f"scale length {meta['scale_length']:.3f}", flush=True)
    return glb
