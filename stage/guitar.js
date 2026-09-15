// A procedural Ibanez M8M — Meshuggah's signature 8-string (29.4" scale, 24 frets, one Lundgren M8
// humbucker at the bridge, fixed bridge with fine tuners, two knobs, 4+4 pointed headstock, matte
// "weathered black" finish) — built at fly scale: 1 human mm = 1/650 stage mm.
import * as THREE from 'three';

const K = 1 / 650;                        // human mm -> stage mm
export const SCALE_LENGTH = 747 * K;      // 29.4"
export const NUM_FRETS = 24;
export const NUM_STRINGS = 8;
const SPACING_BRIDGE = 10.5 * K, SPACING_NUT = 8.0 * K;

// local frame: +x bridge -> nut, +y across the strings toward the bass side (string 0, the upper
// horn side when played), +z out of the top (the face). Bridge saddles at x = 0.
export function fretX(fret) { return fret === 0 ? SCALE_LENGTH : SCALE_LENGTH * Math.pow(2, -fret / 12); }
export function stringSpacingAt(x) { const t = x / SCALE_LENGTH; return SPACING_BRIDGE * (1 - t) + SPACING_NUT * t; }
export function stringY(s, x) { return (NUM_STRINGS - 1) / 2 * stringSpacingAt(x) - s * stringSpacingAt(x); }

// Body outline traced from the M8M photo (human mm; bridge at x = 0). Counter-clockwise from the
// neck joint on the bass side, around the long upper horn, the bouts, the short lower horn.
const BODY = [
  [228, 62], [262, 78], [300, 88], [340, 93], [376, 94],                 // upper horn -> tip
  [370, 108], [340, 118], [300, 124], [255, 130], [200, 135], [150, 138], [100, 137], [55, 132],
  [15, 120], [-25, 103], [-48, 78], [-58, 45], [-60, 5], [-55, -38], [-40, -78], [-15, -108],
  [20, -125], [65, -135], [115, -136], [165, -128], [210, -116], [250, -106], [285, -102], [315, -110],  // lower horn tip
  [305, -92], [284, -72], [262, -56], [240, -46], [228, -40],
];

function bodyShape() {
  const s = new THREE.Shape();
  BODY.forEach(([x, y], i) => { i ? s.lineTo(x * K, y * K) : s.moveTo(x * K, y * K); });
  s.closePath();
  return s;
}

// Pointed 4+4 headstock (human mm, x from the nut)
const HEAD = [[0, 40], [30, 52], [80, 56], [140, 54], [190, 44], [235, 24], [252, 0], [232, -30], [185, -52], [130, -57], [75, -54], [30, -44], [0, -36]];

function logoTexture() {
  const c = document.createElement('canvas'); c.width = 512; c.height = 128;
  const g = c.getContext('2d');
  g.clearRect(0, 0, 512, 128);
  g.fillStyle = '#e8e8ea'; g.font = 'italic bold 92px Georgia, "Times New Roman", serif';
  g.textBaseline = 'middle'; g.fillText('Ibanez', 24, 64);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8;
  return t;
}

export class Guitar {
  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'IbanezM8M';
    this.strings = [];
    this.stringAmp = new Float32Array(NUM_STRINGS);
    this.stringPhase = new Float32Array(NUM_STRINGS);
    this._build();
  }

  _build() {
    const g = this.group;
    const black = new THREE.MeshPhysicalMaterial({ color: 0x0c0c0e, roughness: 0.88, metalness: 0.05, clearcoat: 0.05, sheen: 0.15, sheenColor: new THREE.Color(0x303030) });
    const neckMat = new THREE.MeshStandardMaterial({ color: 0x1f1a17, roughness: 0.75 });
    const boardMat = new THREE.MeshStandardMaterial({ color: 0x0f0c0b, roughness: 0.55 });
    const hardware = new THREE.MeshStandardMaterial({ color: 0x141416, roughness: 0.35, metalness: 0.85 });
    const fretMat = new THREE.MeshStandardMaterial({ color: 0xc9ccd4, roughness: 0.3, metalness: 0.9 });
    const steel = new THREE.MeshStandardMaterial({ color: 0xd8dce6, roughness: 0.25, metalness: 1.0 });
    const pupMat = new THREE.MeshStandardMaterial({ color: 0x0a0a0a, roughness: 0.45, metalness: 0.15 });

    // ---- body (45 mm thick, soft bevel = the forearm/belly contours)
    const depth = 45 * K;
    const body = new THREE.Mesh(new THREE.ExtrudeGeometry(bodyShape(), { depth, bevelEnabled: true, bevelThickness: 5 * K, bevelSize: 6 * K, bevelSegments: 4, curveSegments: 24 }), black);
    body.position.z = -depth; body.castShadow = body.receiveShadow = true;
    g.add(body);

    // ---- neck (tapered, 5-piece look via a slightly different colour) + ebony board
    const neckEnd = SCALE_LENGTH + 4 * K;            // nut position
    const wNut = 72 * K, wHeel = 94 * K, neckT = 19 * K;
    const neckShape = new THREE.Shape();
    neckShape.moveTo(150 * K, -wHeel / 2); neckShape.lineTo(neckEnd, -wNut / 2); neckShape.lineTo(neckEnd, wNut / 2); neckShape.lineTo(150 * K, wHeel / 2); neckShape.closePath();
    const neck = new THREE.Mesh(new THREE.ExtrudeGeometry(neckShape, { depth: neckT, bevelEnabled: true, bevelThickness: 5 * K, bevelSize: 7 * K, bevelSegments: 5 }), neckMat);
    neck.position.z = -neckT - 1.5 * K; neck.castShadow = true;
    g.add(neck);
    const boardShape = new THREE.Shape();
    const boardEnd = fretX(24) - 14 * K;
    const wBoardEnd = wHeel - 6 * K;
    boardShape.moveTo(boardEnd, -wBoardEnd / 2); boardShape.lineTo(neckEnd, -wNut / 2); boardShape.lineTo(neckEnd, wNut / 2); boardShape.lineTo(boardEnd, wBoardEnd / 2); boardShape.closePath();
    const board = new THREE.Mesh(new THREE.ExtrudeGeometry(boardShape, { depth: 6 * K, bevelEnabled: false }), boardMat);
    board.position.z = -1.5 * K; board.castShadow = true;
    g.add(board);
    this.boardTop = 4.5 * K;

    // ---- frets (24), nut, side dots only (the M8M has a plain board)
    const fretGeo = new THREE.BoxGeometry(2.2 * K, 1, 1.3 * K);
    for (let f = 1; f <= NUM_FRETS; f++) {
      const x = fretX(f);
      const w = wNut + (wBoardEnd - wNut) * (neckEnd - x) / (neckEnd - boardEnd);
      const m = new THREE.Mesh(fretGeo, fretMat);
      m.scale.y = w; m.position.set(x, 0, this.boardTop + 0.6 * K);
      g.add(m);
    }
    const nut = new THREE.Mesh(new THREE.BoxGeometry(5 * K, wNut, 6 * K), new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.6 }));
    nut.position.set(neckEnd + 2.5 * K, 0, this.boardTop + 1.5 * K);
    g.add(nut);
    const dotGeo = new THREE.SphereGeometry(1.5 * K, 8, 8);
    const dotMat = new THREE.MeshStandardMaterial({ color: 0xf2eee0, emissive: 0x333333 });
    for (const f of [3, 5, 7, 9, 12, 15, 17, 19, 21, 24]) {
      const x = (fretX(f) + fretX(f - 1)) / 2;
      const w = wNut + (wBoardEnd - wNut) * (neckEnd - x) / (neckEnd - boardEnd);
      const d = new THREE.Mesh(dotGeo, dotMat);
      d.position.set(x, -(w / 2 + 0.3 * K), this.boardTop - 2.5 * K);
      g.add(d);
      if (f === 12 || f === 24) { const d2 = d.clone(); d2.position.x += 6 * K; g.add(d2); }
    }

    // ---- headstock: pointed 4+4, black tuners, logo
    const headShape = new THREE.Shape();
    HEAD.forEach(([x, y], i) => { i ? headShape.lineTo(x * K, y * K) : headShape.moveTo(x * K, y * K); });
    headShape.closePath();
    const headT = 14 * K;
    const head = new THREE.Mesh(new THREE.ExtrudeGeometry(headShape, { depth: headT, bevelEnabled: true, bevelThickness: 2 * K, bevelSize: 2 * K, bevelSegments: 2 }), black);
    head.position.set(neckEnd + 3 * K, 0, -headT - 6 * K);
    head.rotation.y = 0.16;                                    // tilt-back headstock
    head.castShadow = true;
    g.add(head);
    const logo = new THREE.Mesh(new THREE.PlaneGeometry(95 * K, 24 * K), new THREE.MeshBasicMaterial({ map: logoTexture(), transparent: true, depthWrite: false }));
    logo.position.set(neckEnd + 185 * K, -8 * K, -6 * K + 0.4 * K + Math.tan(0.16) * -180 * K);
    logo.rotation.y = 0.16;
    g.add(logo);
    const postGeo = new THREE.CylinderGeometry(3 * K, 3 * K, 16 * K, 10);
    const pegGeo = new THREE.BoxGeometry(6 * K, 22 * K, 5 * K);
    this.tunerPosts = [];
    for (let i = 0; i < NUM_STRINGS; i++) {
      const top = i < 4;                                         // strings 0-3 (bass side) on the upper row
      const j = top ? i : i - 4;
      const px = neckEnd + (48 + j * 42) * K + (top ? 0 : 20 * K);
      const py = (top ? 1 : -1) * (36 - j * 2) * K;
      const zTop = -6 * K + Math.tan(-0.16) * (px - neckEnd - 3 * K);
      const post = new THREE.Mesh(postGeo, hardware);
      post.position.set(px, py, zTop + 6 * K); post.rotation.x = Math.PI / 2; g.add(post);
      const peg = new THREE.Mesh(pegGeo, hardware);
      peg.position.set(px, py + (top ? 1 : -1) * 20 * K, zTop - headT - 4 * K); g.add(peg);
      this.tunerPosts.push(new THREE.Vector3(px, py, zTop + 12 * K));
    }

    // ---- pickup: Lundgren M8 soapbar humbucker with a row of exposed pole screws
    const pup = new THREE.Mesh(new THREE.BoxGeometry(40 * K, 106 * K, 13 * K), pupMat);
    pup.position.set(66 * K, 0, 5 * K); pup.castShadow = true;
    g.add(pup);
    const poleGeo = new THREE.CylinderGeometry(2.4 * K, 2.4 * K, 2 * K, 10);
    for (let s = 0; s < NUM_STRINGS; s++) {
      for (const dx of [-9, 9]) {
        const pole = new THREE.Mesh(poleGeo, dx < 0 ? steel : hardware);
        pole.position.set((66 + dx) * K, stringY(s, 66 * K), 12 * K); pole.rotation.x = Math.PI / 2;
        g.add(pole);
      }
    }
    const ring = new THREE.Mesh(new THREE.BoxGeometry(46 * K, 112 * K, 3 * K), hardware);
    ring.position.set(66 * K, 0, 1 * K); g.add(ring);

    // ---- fixed bridge: base plate, 8 saddles with string blocks, fine-tuner thumbscrews behind
    const plate = new THREE.Mesh(new THREE.BoxGeometry(38 * K, 104 * K, 5 * K), hardware);
    plate.position.set(-6 * K, 0, 2.5 * K); g.add(plate);
    const saddleGeo = new THREE.BoxGeometry(11 * K, 8.5 * K, 7 * K);
    const blockGeo = new THREE.BoxGeometry(8 * K, 7 * K, 9 * K);
    const screwGeo = new THREE.CylinderGeometry(2.6 * K, 2.6 * K, 4 * K, 10);
    for (let s = 0; s < NUM_STRINGS; s++) {
      const y = stringY(s, 0);
      const sd = new THREE.Mesh(saddleGeo, steel); sd.position.set(0, y, 8 * K); g.add(sd);
      const bl = new THREE.Mesh(blockGeo, hardware); bl.position.set(-13 * K, y, 8 * K); g.add(bl);
      const sc = new THREE.Mesh(screwGeo, hardware); sc.position.set(-24 * K, y, 6 * K); sc.rotation.z = Math.PI / 2; g.add(sc);
    }

    // ---- knobs (volume / tone) and strap buttons
    const knobGeo = new THREE.CylinderGeometry(9.5 * K, 10 * K, 14 * K, 20);
    for (const [kx, ky] of [[44, -80], [3, -122]]) {
      const knob = new THREE.Mesh(knobGeo, pupMat);
      knob.position.set(kx * K, ky * K, 5 * K); knob.rotation.x = Math.PI / 2; knob.castShadow = true;
      g.add(knob);
      const cap = new THREE.Mesh(new THREE.CylinderGeometry(9 * K, 9 * K, 1 * K, 20), hardware);
      cap.position.set(kx * K, ky * K, 12.5 * K); cap.rotation.x = Math.PI / 2; g.add(cap);
    }
    const btnGeo = new THREE.CylinderGeometry(4 * K, 3 * K, 6 * K, 10);
    this.strapButtons = [new THREE.Vector3(372 * K, 96 * K, -depth / 2), new THREE.Vector3(-58 * K, 5 * K, -depth / 2)];
    for (const p of this.strapButtons) {
      const b = new THREE.Mesh(btnGeo, hardware); b.position.copy(p); b.rotation.z = p.x > 0 ? -1.2 : Math.PI / 2; g.add(b);
    }

    // ---- strings (bridge saddles -> nut -> tuner posts)
    this.stringLen = SCALE_LENGTH + 8 * K;
    for (let s = 0; s < NUM_STRINGS; s++) {
      const gauge = (s < 4 ? [0.074, 0.056, 0.046, 0.036][s] : [0.026, 0.017, 0.013, 0.010][s - 4]) * 25.4 * K;
      const geo = new THREE.CylinderGeometry(gauge * 1.6, gauge * 1.6, this.stringLen, 6, 1, true);
      geo.rotateZ(Math.PI / 2);
      const mat = new THREE.MeshStandardMaterial({ color: s < 5 ? 0xb9bec8 : 0xe4e8f0, roughness: 0.3, metalness: 1.0, emissive: 0x000000 });
      const str = new THREE.Mesh(geo, mat);
      str.position.set(this.stringLen / 2, stringY(s, this.stringLen / 2), this.boardTop + 3.4 * K);
      str.rotation.z = Math.atan2(stringY(s, this.stringLen) - stringY(s, 0), this.stringLen);
      g.add(str);
      this.strings.push(str);
      // the run from the nut to the tuner post
      const a = new THREE.Vector3(neckEnd + 5 * K, stringY(s, neckEnd), this.boardTop + 3.4 * K);
      const b = this.tunerPosts[s];
      const seg = new THREE.Mesh(new THREE.CylinderGeometry(gauge * 1.4, gauge * 1.4, a.distanceTo(b), 5, 1, true), mat);
      seg.position.copy(a).lerp(b, 0.5);
      seg.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize());
      g.add(seg);
    }
  }

  // world position on the fretboard for (string, fret); fret 0 -> hover just behind fret 1
  fretTarget(string, fret, out = new THREE.Vector3(), lift = 0) {
    const f = Math.max(0, Math.min(NUM_FRETS, fret || 0));
    const x = f === 0 ? (fretX(1) + fretX(0)) / 2 - 0.02 : (fretX(f) + fretX(f - 1)) / 2 + 0.25 * (fretX(f - 1) - fretX(f)) * 0.3;
    out.set(x, stringY(string, x), this.boardTop + 5 * K + lift);
    return this.group.localToWorld(out);
  }

  // picking position over a string between the pickup and the bridge; phase in [-1, 1] sweeps across it
  pickTarget(string, phase, out = new THREE.Vector3(), height = 0) {
    const x = 34 * K;
    const sp = stringSpacingAt(x);
    out.set(x, stringY(string, x) - phase * sp * 0.9, this.boardTop + 9 * K + height);
    return this.group.localToWorld(out);
  }

  pluck(string, vel = 100) {
    this.stringAmp[string] = Math.min(1.5, 0.35 + vel / 127);
  }

  setFretMarker(string, fret, on) {
    if (!this.marker) {
      this.marker = new THREE.Mesh(new THREE.SphereGeometry(0.011, 12, 12), new THREE.MeshBasicMaterial({ color: 0xffb060, transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending, depthWrite: false }));
      this.group.add(this.marker);
    }
    this.marker.visible = !!on && fret > 0;
    if (this.marker.visible) { const x = (fretX(fret) + fretX(fret - 1)) / 2; this.marker.position.set(x, stringY(string, x), this.boardTop + 6 * K); }
  }

  update(dt, t) {
    for (let s = 0; s < NUM_STRINGS; s++) {
      const a = this.stringAmp[s];
      if (a > 0.001) {
        this.stringAmp[s] = a * Math.exp(-dt / 0.35);
        this.stringPhase[s] += dt * (60 + s * 25);
        const str = this.strings[s];
        const base = stringY(s, this.stringLen / 2);
        str.position.y = base + Math.sin(this.stringPhase[s]) * a * 0.010;
        str.scale.y = str.scale.z = 1 + a * 1.0;
        str.material.emissive.setRGB(a * 0.16, a * 0.11, a * 0.04);
      } else if (this.strings[s].scale.y !== 1) {
        this.strings[s].scale.set(1, 1, 1);
        this.strings[s].material.emissive.setRGB(0, 0, 0);
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// The user's Ibanez M8M mesh (stage/data/guitar.glb, exported by stage_export.export_guitar) with
// the same API as the procedural guitar. Frame: bridge saddles at the origin, +x toward the nut,
// +y toward the bass side, +z out of the face, string plane at z = 0.
// ---------------------------------------------------------------------------------------------
export class GuitarModel {
  constructor(gltfScene, meta) {
    this.group = new THREE.Group();
    this.group.name = 'IbanezM8M-mesh';
    this.meta = meta;
    this.scaleLength = meta.scale_length;
    this.spacingBridge = meta.spacing_bridge; this.spacingNut = meta.spacing_nut;
    this.boardTop = meta.board_top;
    this.pickupX = meta.pickup_x;
    this.strapButtons = meta.strap_buttons.map(p => new THREE.Vector3(...p));
    gltfScene.traverse(o => {
      if (!o.isMesh) return;
      o.castShadow = o.receiveShadow = true;
      if (o.material) { o.material.side = THREE.FrontSide; if (o.material.normalScale) o.material.normalScale.set(0.8, 0.8); }
      if (!o.geometry.attributes.normal) o.geometry.computeVertexNormals();
    });
    this.group.add(gltfScene);
    // vibration overlay: thin additive strings shown only while a string rings
    this.strings = []; this.stringAmp = new Float32Array(NUM_STRINGS); this.stringPhase = new Float32Array(NUM_STRINGS);
    this.stringLen = this.scaleLength;
    for (let s = 0; s < NUM_STRINGS; s++) {
      const geo = new THREE.CylinderGeometry(0.0022, 0.0022, this.stringLen, 6, 1, true);
      geo.rotateZ(Math.PI / 2);
      const mat = new THREE.MeshBasicMaterial({ color: 0xffd9a0, transparent: true, opacity: 0.0, blending: THREE.AdditiveBlending, depthWrite: false });
      const str = new THREE.Mesh(geo, mat);
      str.position.set(this.stringLen / 2, this.stringY(s, this.stringLen / 2), 0.0015);
      str.rotation.z = Math.atan2(this.stringY(s, this.stringLen) - this.stringY(s, 0), this.stringLen);
      str.visible = false;
      this.group.add(str);
      this.strings.push(str);
    }
  }
  // measured fret wire positions of the mesh when available (fret 0 = the nut)
  fretX(fret) {
    if (this.meta.frets) return fret === 0 ? this.meta.nut : this.meta.frets[Math.max(0, Math.min(NUM_FRETS, fret)) - 1];
    return fret === 0 ? this.scaleLength : this.scaleLength * Math.pow(2, -fret / 12);
  }
  spacingAt(x) { const t = Math.max(0, Math.min(1, x / this.scaleLength)); return this.spacingBridge * (1 - t) + this.spacingNut * t; }
  stringY(s, x) { return (NUM_STRINGS - 1) / 2 * this.spacingAt(x) - s * this.spacingAt(x); }

  // finger position for (string, fret): just behind the fret wire (toward the nut), on the string
  fretLocal(string, fret, out = new THREE.Vector3(), lift = 0) {
    const f = Math.max(0, Math.min(NUM_FRETS, fret || 0));
    const x = f === 0 ? this.fretX(1) - 0.02 : this.fretX(f) + 0.35 * (this.fretX(f - 1) - this.fretX(f));
    return out.set(x, this.stringY(string, x), 0.006 + lift);
  }
  fretTarget(string, fret, out = new THREE.Vector3(), lift = 0) {
    return this.group.localToWorld(this.fretLocal(string, fret, out, lift));
  }
  pickTarget(string, phase, out = new THREE.Vector3(), height = 0) {
    const x = this.pickupX + 0.03;
    out.set(x, this.stringY(string, x) - phase * this.spacingAt(x) * 0.9, 0.014 + height);
    return this.group.localToWorld(out);
  }
  // a small glowing marker on the fretted position so the string/fret mapping is readable
  setFretMarker(string, fret, on) {
    if (!this.marker) {
      this.marker = new THREE.Mesh(new THREE.SphereGeometry(0.011, 12, 12), new THREE.MeshBasicMaterial({ color: 0xffb060, transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending, depthWrite: false }));
      this.group.add(this.marker);
    }
    this.marker.visible = !!on && fret > 0;
    if (this.marker.visible) this.fretLocal(string, fret, this.marker.position, 0.002);
  }
  pluck(string, vel = 100) { this.stringAmp[string] = Math.min(1.5, 0.35 + vel / 127); }
  update(dt, t) {
    for (let s = 0; s < NUM_STRINGS; s++) {
      const a = this.stringAmp[s], str = this.strings[s];
      if (a > 0.02) {
        this.stringAmp[s] = a * Math.exp(-dt / 0.35);
        this.stringPhase[s] += dt * (60 + s * 25);
        str.visible = true;
        str.position.y = this.stringY(s, this.stringLen / 2) + Math.sin(this.stringPhase[s]) * a * 0.006;
        str.material.opacity = Math.min(0.6, a * 0.5);
      } else if (str.visible) { str.visible = false; this.stringAmp[s] = 0; }
    }
  }
}

export async function loadGuitar(loader, base = './data/') {
  try {
    const meta = await fetch(base + 'guitar.json').then(r => { if (!r.ok) throw new Error('no guitar.json'); return r.json(); });
    const gltf = await loader.loadAsync(base + 'guitar.glb');
    return new GuitarModel(gltf.scene, meta);
  } catch (e) {
    console.warn('guitar.glb not available, using the procedural M8M:', e.message);
    return new Guitar();
  }
}
