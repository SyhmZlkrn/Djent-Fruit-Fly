// FlyBrain Composer stage: the NeuroMechFly body plays the Ibanez M8M while the MaleCNS brain spikes.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { FlyRig } from './fly.js?v=25';
import { Guitar, NUM_FRETS, loadGuitar } from './guitar.js?v=15';
import { loadBrain, loadSpikes, spikesUntil, seekSpikes } from './brain.js?v=15';

const DEG = Math.PI / 180;
const $ = id => document.getElementById(id);
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const noteName = m => NOTE_NAMES[m % 12] + (Math.floor(m / 12) - 1);

// ------------------------------------------------------------------ renderer / scene
const canvas = $('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x05060a);
scene.fog = new THREE.Fog(0x05060a, 9, 22);
const camera = new THREE.PerspectiveCamera(32, window.innerWidth / window.innerHeight, 0.02, 80);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.target.set(0.2, 1.0, 0);
controls.enabled = false;

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.55, 0.5, 0.78);
composer.addPass(bloom);
composer.addPass(new OutputPass());

// lights
scene.add(new THREE.HemisphereLight(0x7f8fff, 0x120a04, 0.9));
const key = new THREE.SpotLight(0xfff1dc, 120, 30, 0.55, 0.5, 1.6);
key.position.set(3.5, 6.5, 3.2); key.castShadow = true; key.shadow.mapSize.set(2048, 2048); key.shadow.bias = -0.0002; key.shadow.camera.near = 0.5; key.shadow.camera.far = 20;
scene.add(key, key.target);
const rim = new THREE.SpotLight(0xff3fd8, 35, 30, 0.6, 0.6, 1.6); rim.position.set(-3.5, 4.5, -3.5); scene.add(rim);
const fill = new THREE.SpotLight(0x39d0ff, 22, 30, 0.7, 0.6, 1.6); fill.position.set(2.5, 3.0, -4.5); scene.add(fill);
const under = new THREE.PointLight(0xff7a2a, 4, 6, 1.5); under.position.set(0.6, 0.15, 0.4); scene.add(under);

// stage floor
const floor = new THREE.Mesh(new THREE.CircleGeometry(7, 96), new THREE.MeshStandardMaterial({ color: 0x0c0d14, roughness: 0.32, metalness: 0.35 }));
floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; scene.add(floor);
const ring = new THREE.Mesh(new THREE.RingGeometry(2.55, 2.62, 128), new THREE.MeshBasicMaterial({ color: 0x2b3a8a, transparent: true, opacity: 0.55, side: THREE.DoubleSide }));
ring.rotation.x = -Math.PI / 2; ring.position.y = 0.002; scene.add(ring);

// ------------------------------------------------------------------ state
const S = {
  fly: null, flyRoot: new THREE.Group(), flyModel: null, guitar: null, brain: null, holo: null, headBrain: null,
  notes: [], drums: [], sections: [], song: null, raster: null,
  mode: 'idle', audio: null, ws: null, wsT: 0, wsAt: 0,
  noteI: 0, drumI: 0, lastT: 0, t: 0,
  fret: { string: 0, fret: 0, lift: 0.09, active: 0, hand: 13 }, pick: { string: 0, phase: -1, t0: -10, vel: 100 },
  nod: 0, nodV: 0, bob: 0, bobV: 0, wing: 0, ant: 0, spikeRate: 0, frames: 0, fpsT: performance.now(),
  camMode: 'audience', camPos: new THREE.Vector3(), camTgt: new THREE.Vector3(),
  live: { avail: false, on: false, phrase: null, planning: null, history: [], nPhrases: 0, replayLevel: null, mode: null, weights: null, human: null, lastHuman: null, cycle16: null },
  wsSection: null,
};
scene.add(S.flyRoot);
const tmpV = new THREE.Vector3(), tmpV2 = new THREE.Vector3(), kneeTmp = new THREE.Vector3();
const WING_AXIS_L = new THREE.Vector3(1, 0, 0), WING_AXIS_R = new THREE.Vector3(-1, 0, 0);   // flap about the body axis

// ------------------------------------------------------------------ loading
async function loadAll() {
  const set = m => { $('loading').textContent = m; };
  set('loading song…');
  const [song, notes, drums, sections] = await Promise.all(['song', 'notes', 'drums', 'sections'].map(n => fetch(`./data/${n}.json`).then(r => r.json())));
  Object.assign(S, { song, notes, drums, sections });
  set('loading the fly (NeuroMechFly, 24 MB)…');
  const loader = new GLTFLoader();
  const [gltf, rig] = await Promise.all([loader.loadAsync('./data/fly.glb'), fetch('./data/fly_rig.json').then(r => r.json())]);
  buildFly(gltf.scene, rig);
  set('loading the brain (2,318 neurons, 22 MB)…');
  S.brain = await loadBrain();
  buildBrain();
  set('loading region meshes…');
  try { const rois = await loader.loadAsync('./data/rois.glb'); buildRois(rois.scene); } catch (e) { console.warn('rois', e); }
  set('loading spike raster…');
  try { S.raster = await loadSpikes(); } catch (e) { console.warn('spikes', e); }
  set('loading the Ibanez M8M…');
  const guitar = await loadGuitar(loader);
  buildGuitar(guitar);
  setCamera('audience');
  set('ready');
}

// ------------------------------------------------------------------ fly
function buildFly(model, rig) {
  S.flyModel = model;
  model.rotation.x = -Math.PI / 2;            // MJCF is z-up, x forward -> y-up world, fly faces +x
  model.traverse(o => {
    if (!o.isMesh) return;
    o.castShadow = o.receiveShadow = true;
    if (!o.geometry.attributes.normal) o.geometry.computeVertexNormals();   // trimesh GLB ships no normals
    const n = o.name;
    const isEye = /Eye$/.test(n) || /Eye/.test(o.parent?.name || '');
    const isWing = /Wing/.test(n) || /Wing/.test(o.parent?.name || '');
    if (isEye) o.material = new THREE.MeshPhysicalMaterial({ color: 0xb8241c, roughness: 0.25, clearcoat: 1, clearcoatRoughness: 0.1, emissive: 0x3a0605 });
    else if (isWing) o.material = new THREE.MeshPhysicalMaterial({ color: 0xcfd8ff, transparent: true, opacity: 0.28, roughness: 0.15, transmission: 0.4, iridescence: 0.9, iridescenceIOR: 1.4, side: THREE.DoubleSide, depthWrite: false });
    else if (/Head/.test(n) || /Head/.test(o.parent?.name || '')) o.material = new THREE.MeshPhysicalMaterial({ color: 0x5a3a1c, roughness: 0.5, clearcoat: 0.4, transparent: true, opacity: 0.62, depthWrite: true });
    else o.material = new THREE.MeshPhysicalMaterial({ color: 0x4a2f16, roughness: 0.55, metalness: 0.05, clearcoat: 0.35, clearcoatRoughness: 0.4, sheen: 0.4, sheenColor: new THREE.Color(0xffc070) });
  });
  const fly = new FlyRig(model, rig);
  fly.applyPoseDeg(rig.pose_tripod_deg);
  fly.update();
  S.fly = fly;
  S.flyRoot.add(model);
  // rear up on the hind + middle legs, like a guitarist standing
  S.flyRoot.rotation.z = 52 * DEG;
  S.flyRoot.position.set(-0.15, 0.62, 0);
  model.updateMatrixWorld(true);
  // plant the four standing legs on the floor with IK
  S.feet = {
    LM: new THREE.Vector3(0.05, 0.0, -0.62), RM: new THREE.Vector3(0.05, 0.0, 0.62),
    LH: new THREE.Vector3(-0.85, 0.0, -0.42), RH: new THREE.Vector3(-0.85, 0.0, 0.42),
  };
  for (let k = 0; k < 4; k++) for (const [leg, p] of Object.entries(S.feet)) fly.solveLeg(leg, p, 8, 0.8);
}

// ------------------------------------------------------------------ guitar
// Guitar placement, relative to the thorax. Defaults below; nudge live with the keyboard in
// placement mode (press P) — the offsets are shown on screen, saved in localStorage and printed
// as a line you can paste here.
const GUITAR_DEFAULT = { dx: 0.52, dy: -0.80, dz: 0.24, tilt: 30, yaw: 10, roll: 20 };   // dx: toward audience, dy: up, dz: fly's right; angles in degrees
S.guitarPlace = { ...GUITAR_DEFAULT };
try { const saved = JSON.parse(localStorage.getItem('flybrain.guitarPlace') || 'null'); if (saved) Object.assign(S.guitarPlace, saved); } catch (e) { }

function applyGuitarPlacement() {
  const g = S.guitar, P = S.guitarPlace;
  const thorax = S.fly.worldPos('Thorax', new THREE.Vector3());
  // neck direction: to the fly's left (-z), tilted up by `tilt`, swung toward the audience by `yaw`
  const tilt = P.tilt * DEG, yaw = P.yaw * DEG;
  const neckDir = new THREE.Vector3(Math.sin(yaw) * Math.cos(tilt), Math.sin(tilt), -Math.cos(yaw) * Math.cos(tilt)).normalize();
  // face normal: toward the audience (+x), tipped up by `roll`, made perpendicular to the neck
  let face = new THREE.Vector3(Math.cos(P.roll * DEG), Math.sin(P.roll * DEG), 0);
  face.addScaledVector(neckDir, -face.dot(neckDir)).normalize();
  const yAxis = new THREE.Vector3().crossVectors(face, neckDir).normalize();
  g.group.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(neckDir, yAxis, face));
  g.group.position.set(thorax.x + P.dx, thorax.y + P.dy, P.dz);
  g.group.updateMatrixWorld(true);
  buildStrap();
  for (let k = 0; k < 4; k++) placeHands(0.0, 0.0, true);
  $('place').textContent = `guitar  dx ${P.dx.toFixed(2)}  dy ${P.dy.toFixed(2)}  dz ${P.dz.toFixed(2)}  tilt ${P.tilt.toFixed(0)}°  yaw ${P.yaw.toFixed(0)}°  roll ${P.roll.toFixed(0)}°`;
}

function buildStrap() {
  if (S.strap) { scene.remove(S.strap); S.strap.geometry.dispose(); }
  const g = S.guitar;
  const thorax = S.fly.worldPos('Thorax', new THREE.Vector3());
  const up = g.group.localToWorld(g.strapButtons[0].clone());
  const lo = g.group.localToWorld(g.strapButtons[1].clone());
  const over = thorax.clone().add(new THREE.Vector3(-0.05, 0.42, 0.0));
  const curve = new THREE.CatmullRomCurve3([up, up.clone().lerp(over, 0.5).add(new THREE.Vector3(-0.1, 0.15, -0.15)), over, over.clone().lerp(lo, 0.5).add(new THREE.Vector3(-0.35, 0.05, 0.1)), lo]);
  S.strap = new THREE.Mesh(new THREE.TubeGeometry(curve, 48, 0.028, 8, false), new THREE.MeshStandardMaterial({ color: 0x17171b, roughness: 0.95 }));
  S.strap.castShadow = true;
  scene.add(S.strap);
}

function buildGuitar(g) {
  S.guitar = g;
  scene.add(g.group);
  S.legLen = { LF: S.fly.legLengths('LF'), RF: S.fly.legLengths('RF') };
  applyGuitarPlacement();
}

// placement mode: P toggles; arrows move (shift = depth), Q/E tilt the neck, A/D yaw it, Z/C roll
// the face, R resets, Enter prints the paste-able line. Saved in localStorage.
S.placing = false;
function placementKey(e) {
  if (e.key === 'p' || e.key === 'P') { S.placing = !S.placing; $('placePanel').hidden = !S.placing; if (S.placing) applyGuitarPlacement(); return true; }
  if (!S.placing) return false;
  const P = S.guitarPlace, step = e.shiftKey ? 0.05 : 0.02, ang = e.shiftKey ? 5 : 2;
  const map = {
    ArrowLeft: () => P.dz -= step, ArrowRight: () => P.dz += step,
    ArrowUp: () => (e.shiftKey ? P.dx += 0.05 : P.dy += step), ArrowDown: () => (e.shiftKey ? P.dx -= 0.05 : P.dy -= step),
    PageUp: () => P.dx += step, PageDown: () => P.dx -= step,
    q: () => P.tilt += ang, e: () => P.tilt -= ang, a: () => P.yaw -= ang, d: () => P.yaw += ang, z: () => P.roll -= ang, c: () => P.roll += ang,
    r: () => Object.assign(P, GUITAR_DEFAULT),
    Enter: () => console.log(`const GUITAR_DEFAULT = ${JSON.stringify(P)};`),
  };
  const fn = map[e.key] || map[e.key.toLowerCase()];
  if (!fn) return false;
  fn();
  try { localStorage.setItem('flybrain.guitarPlace', JSON.stringify(P)); } catch (err) { }
  applyGuitarPlacement();
  e.preventDefault();
  return true;
}

// ------------------------------------------------------------------ brain
function buildBrain() {
  const b = S.brain;
  // hologram above the stage (um -> mm, x2.4), anterior toward the audience, dorsal up
  const holo = b.group;
  holo.scale.setScalar(0.0036);
  holo.rotation.y = -Math.PI / 2;
  holo.position.set(-3.2, 2.6, 0);
  S.holo = holo;
  scene.add(holo);
  b.pointUniforms.uScale.value = 1.0;
  // the same neurons at true scale inside the head (attached to the Head node)
  const head = S.fly.nodes.get('Head');
  const box = new THREE.Box3();
  head.traverse(o => { if (o.isMesh) { o.geometry.computeBoundingBox(); box.union(o.geometry.boundingBox); } });
  const c = box.getCenter(new THREE.Vector3());
  const hb = b.clone();
  // brain frame (x lateral, y dorsal, z posterior) -> model frame (x anterior, y left, z up)
  const M = new THREE.Matrix4().makeBasis(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(-1, 0, 0));
  hb.quaternion.setFromRotationMatrix(M);
  hb.scale.setScalar(0.00078);
  hb.position.copy(c).add(new THREE.Vector3(-0.02, 0, 0.02));
  head.add(hb);
  S.headBrain = hb;
}

function buildRois(roiScene) {
  const mat = new THREE.MeshBasicMaterial({ color: 0x4d6cff, transparent: true, opacity: 0.06, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide });
  roiScene.traverse(o => { if (o.isMesh) o.material = mat; });
  roiScene.scale.copy(S.holo.scale); roiScene.rotation.copy(S.holo.rotation); roiScene.position.copy(S.holo.position);
  S.rois = roiScene;
  scene.add(roiScene);
}

// ------------------------------------------------------------------ playing
function stringFretFor(n) {
  if (n.string != null && n.fret != null) return [n.string, n.fret];
  const tuning = S.song.tuning;
  for (let s = 0; s < tuning.length; s++) { const f = n.pitch - tuning[s]; if (f >= 0 && f <= 7) return [s, f]; }
  for (let s = 0; s < tuning.length; s++) { const f = n.pitch - tuning[s]; if (f >= 0 && f <= NUM_FRETS) return [s, f]; }
  return [0, 0];
}

function onNote(n, t) {
  const [s, f] = stringFretFor(n);
  S.fret.string = s; S.fret.fret = f; S.fret.active = t + n.dur;
  if (f > 0) S.fret.hand = f;                 // open strings: the hand stays where it last fretted
  S.pick.string = s; S.pick.t0 = t; S.pick.vel = n.vel;
  S.guitar.pluck(s, n.vel);
  if (n.bend > 0 || n.pitch >= 40) { S.ant = 1; S.wing = Math.max(S.wing, 0.8); }
  $('note').innerHTML = `${noteName(n.pitch)} <small>string ${s + 1} · fret ${f}${n.pm ? ' · palm mute' : ''}${n.bend ? ' · bend' : ''} · vel ${n.vel}</small>`;
}

function onDrum(d) {
  if (d.pitch === 35 || d.pitch === 36) { S.bobV -= 0.9; S.nodV += 6; key.intensity = 170; }
  else if (d.pitch === 38 || d.pitch === 40) { S.nodV += 14; S.wing = Math.max(S.wing, 0.5); rim.intensity = 70; }
  else if ([49, 52, 55, 57].includes(d.pitch)) { S.nodV += 10; S.wing = 1.0; fill.intensity = 60; }
}

function advanceCursors(t) {
  while (S.noteI < S.notes.length && S.notes[S.noteI].t <= t) { const n = S.notes[S.noteI++]; if (n.t > S.lastT - 0.25) onNote(n, t); }
  while (S.drumI < S.drums.length && S.drums[S.drumI].t <= t) { const d = S.drums[S.drumI++]; if (d.t > S.lastT - 0.25) onDrum(d); }
}

function seekCursors(t) {
  S.noteI = S.notes.findIndex(n => n.t >= t); if (S.noteI < 0) S.noteI = S.notes.length;
  S.drumI = S.drums.findIndex(d => d.t >= t); if (S.drumI < 0) S.drumI = S.drums.length;
  if (S.raster) seekSpikes(S.raster, t);
  S.brain.reset();
  S.lastT = t;
}

// elbows splayed out and up like a hunched spider: the legs are twice as long as the reach, so the
// femur/tibia fold is extreme and the knee has to go somewhere — out to the side and up reads best
// Elbow directions in WORLD space (+x toward the audience, +y up, +z the fly's right): both elbows
// hang below the shoulders, out to their own side.
const POLE_L = new THREE.Vector3(0.30, -0.90, -0.30);   // fretting elbow: down, a little forward, fly's left
const POLE_R = new THREE.Vector3(0.20, -0.55, 0.80);    // picking elbow: out to the fly's right, down, a little forward
// Hand approach directions in the GUITAR frame (+x nut, +y bass side, +z out of the face): the way the
// fingers travel onto their target. Fretting: from under the neck, up over the treble edge onto the
// face. Picking: from above the top edge, down onto the strings.
const APPROACH_FRET = new THREE.Vector3(0.0, 0.85, 0.50);
const APPROACH_PICK = new THREE.Vector3(0.30, -0.75, -0.50);   // from the upper right (body end), down into the strings
// fixed hand shapes (tarsus1..5, degrees): fretting = a long, nearly straight finger with a slight
// curl toward the string; picking = fingers curled around a pick
const HAND_FRET = [-30, -20, -20, -15, -10].map(d => d * DEG);    // initial finger pose (the solver curls from here)
const HAND_PICK = [-50, -30, -30, -25, -15].map(d => d * DEG);
const poleTmp = new THREE.Vector3(), appTmp = new THREE.Vector3(), tipTmp = new THREE.Vector3();
S.armState = { LF: {}, RF: {} };

// elbows may not go behind the shoulders (that is where the thorax is) or below the floor
const elbowOk = (K, W) => (K.x > S.fly.worldPos('Thorax', tmpV2).x - 0.25) && K.y > 0.15 && W.y > 0.1;

function placeHands(t, dt, init) {
  const fly = S.fly, g = S.guitar;
  if (init) { S.armState = { LF: {}, RF: {} }; }
  // ---- fretting hand (LEFT front leg): on open strings the fingers hover ~4 mm above the string at
  // the hand's resting fret instead of flying to the nut
  const holding = t < S.fret.active + 0.06;
  const open = S.fret.fret === 0;
  S.fret.lift += (((holding && !open) ? 0.0 : 0.045) - S.fret.lift) * Math.min(1, dt * 18);
  const fret = open ? S.fret.hand : S.fret.fret;
  g.fretTarget(S.fret.string, fret, tipTmp, init ? 0.045 : S.fret.lift);
  appTmp.copy(APPROACH_FRET).applyQuaternion(g.group.quaternion);
  S.dbg = S.dbg || {};
  S.dbg.tipL = fly.solveArmPlanar('LF', tipTmp, POLE_L, appTmp, HAND_FRET, init ? 12 : 5, S.armState.LF, elbowOk);
  g.setFretMarker(S.fret.string, open ? 0 : S.fret.fret, holding && !open);
  // ---- picking hand (RIGHT front leg): fast downstroke at each onset, slow return; the fingertip
  // hovers just above the strings between strokes
  const dtp = t - S.pick.t0;
  const phase = dtp < 0.055 ? -1 + 2 * (dtp / 0.055) : dtp < 0.28 ? 1 - 2 * ((dtp - 0.055) / 0.225) : -1;
  const hgt = dtp < 0.055 ? 0.0 : 0.025;
  g.pickTarget(S.pick.string, phase, tipTmp, hgt);
  appTmp.copy(APPROACH_PICK).applyQuaternion(g.group.quaternion);
  S.dbg.tipR = fly.solveArmPlanar('RF', tipTmp, POLE_R, appTmp, HAND_PICK, init ? 12 : 5, S.armState.RF, elbowOk);
}

function animateFly(dt, t) {
  const fly = S.fly, g = S.guitar;
  placeHands(t, dt, false);
  // head bang + body bob (spring-damper)
  S.nodV += (-S.nod * 260 - S.nodV * 14) * dt; S.nod += S.nodV * dt;
  S.bobV += (-S.bob * 220 - S.bobV * 12) * dt; S.bob += S.bobV * dt;
  fly.setJoint('joint_Head', (0.57 + S.nod * 0.06) );
  S.flyRoot.position.y = 0.62 + S.bob * 0.05;
  // abdomen wag, antennae, wings, halteres
  const wag = Math.sin(t * 2.1) * 0.05 + S.nod * 0.02;
  fly.setJoint('joint_A1A2', 0.22 + wag); fly.setJoint('joint_A3', -0.81 + wag * 0.6); fly.setJoint('joint_A4', -0.17 + wag * 0.4);
  S.ant = Math.max(0, S.ant - dt * 2.5);
  fly.setJoint('joint_LPedicel', -0.4 * S.ant + Math.sin(t * 3) * 0.05); fly.setJoint('joint_RPedicel', -0.4 * S.ant + Math.cos(t * 3.3) * 0.05);
  S.wing = Math.max(0, S.wing - dt * 1.8);
  const flutter = Math.sin(t * 190) * S.wing * 0.35;
  fly.setFree('LWing', WING_AXIS_L, 0.25 * S.wing + flutter); fly.setFree('RWing', WING_AXIS_R, 0.25 * S.wing + flutter);
  fly.setJoint('joint_LHaltere', 0.37 + Math.sin(t * 60) * 0.3); fly.setJoint('joint_RHaltere', 0.37 + Math.cos(t * 60) * 0.3);
  fly.update();
  // keep the standing legs planted after the body bobbed
  for (const [leg, p] of Object.entries(S.feet)) fly.solveLeg(leg, p, 2, 0.6);
  g.update(dt, t);
  key.intensity += (120 - key.intensity) * Math.min(1, dt * 6);
  rim.intensity += (35 - rim.intensity) * Math.min(1, dt * 6);
  fill.intensity += (22 - fill.intensity) * Math.min(1, dt * 6);
}

// ------------------------------------------------------------------ clocks (WebAudio stems)
const AUDIO = { ctx: null, buffers: {}, gains: {}, sources: [], t0: 0, offset: 0, playing: false, loaded: false, names: [] };

async function loadAudioStems() {
  if (AUDIO.loaded) return;
  if (AUDIO.loading) return AUDIO.loading;          // a second click while decoding must not open a second context
  AUDIO.loading = _loadAudioStems();
  return AUDIO.loading;
}
async function _loadAudioStems() {
  AUDIO.ctx = new (window.AudioContext || window.webkitAudioContext)();
  let meta = { stems: [], mix: true };
  try { meta = await fetch('./data/stems.json').then(r => r.json()); } catch (e) { }
  const names = meta.stems && meta.stems.length ? meta.stems : ['mix'];
  for (const name of names) {
    const url = name === 'mix' ? './data/song.mp3' : `./data/stem_${name}.mp3`;
    try {
      const buf = await fetch(url).then(r => r.arrayBuffer());
      AUDIO.buffers[name] = await AUDIO.ctx.decodeAudioData(buf);
      const g = AUDIO.ctx.createGain(); g.connect(AUDIO.ctx.destination); AUDIO.gains[name] = g;
      AUDIO.names.push(name);
    } catch (e) { console.warn('stem', name, e); }
  }
  const hasBacking = 'backing' in AUDIO.buffers;
  setStem('guitar', true); setStem('backing', hasBacking); setStem('drums', !hasBacking);
  $('stemBacking').disabled = !hasBacking; $('stemBacking').title = hasBacking ? 'the backing track' : 'no backing track: put yours at data/rational_gaze_backing.mp3 and run stems + stage-export';
  $('stemDrums').disabled = !('drums' in AUDIO.buffers);
  AUDIO.loaded = true;
}

function setStem(name, on) {
  if (AUDIO.gains[name]) AUDIO.gains[name].gain.value = on ? 1 : 0;
  const btn = { guitar: $('stemGuitar'), backing: $('stemBacking'), drums: $('stemDrums') }[name];
  if (btn) btn.classList.toggle('on', !!on && !!AUDIO.gains[name]);
}
function stemOn(name) { return !!(AUDIO.gains[name] && AUDIO.gains[name].gain.value > 0); }

function startStems(at) {
  stopStems();
  const when = AUDIO.ctx.currentTime + 0.08;
  for (const name of AUDIO.names) {
    const src = AUDIO.ctx.createBufferSource();
    src.buffer = AUDIO.buffers[name]; src.connect(AUDIO.gains[name]);
    src.start(when, Math.max(0, at));
    AUDIO.sources.push(src);
  }
  AUDIO.t0 = when - at; AUDIO.playing = true;
}
function stopStems() { for (const s of AUDIO.sources) { try { s.stop(); } catch (e) { } } AUDIO.sources = []; AUDIO.playing = false; }

function songTime() {
  if (S.mode === 'audio') return AUDIO.playing ? AUDIO.ctx.currentTime - AUDIO.t0 : AUDIO.offset;
  if (S.mode === 'ws') return S.wsT + (performance.now() - S.wsAt) / 1000;
  return S.t;
}

async function startAudio(flyOnly = false) {
  closeWs();
  await loadAudioStems();
  if (AUDIO.ctx.state === 'suspended') await AUDIO.ctx.resume();
  S.mode = 'audio';
  if (flyOnly) { setStem('backing', false); setStem('drums', false); setStem('guitar', true); }
  seekCursors(AUDIO.offset || 0);
  startStems(AUDIO.offset || 0);
  $('btnPlay').classList.add('on'); $('btnWs').classList.remove('on');
  $('clock').textContent = '(browser audio)';
}

function togglePause() {
  if (S.mode !== 'audio' || !AUDIO.loaded) return startAudio();
  if (AUDIO.playing) { AUDIO.offset = songTime(); stopStems(); $('btnPlay').classList.remove('on'); }
  else { startStems(AUDIO.offset); $('btnPlay').classList.add('on'); }
}

function closeWs() { if (S.ws) { try { S.ws.close(); } catch (e) { } S.ws = null; } }

function startWs() {
  closeWs();
  const ws = new WebSocket('ws://localhost:8765');
  S.ws = ws;
  $('clock').textContent = '(connecting to conductor…)';
  let settled = false;
  const done = new Promise(resolve => { ws._resolve = v => { if (!settled) { settled = true; resolve(v); } }; });
  ws.onopen = () => { if (AUDIO.playing) { AUDIO.offset = songTime(); stopStems(); } S.mode = 'ws'; $('btnWs').classList.add('on'); $('btnPlay').classList.remove('on'); $('clock').textContent = '(Python conductor)'; ws._resolve(true); refreshLauncher(); };
  ws.onclose = () => { ws._resolve(false); if (S.mode === 'ws' && S.ws === ws) { S.mode = 'idle'; S.live.avail = false; updateLivePanel(); $('clock').textContent = '(conductor disconnected)'; $('btnWs').classList.remove('on'); refreshLauncher(); } };
  ws.onerror = () => { ws._resolve(false); };
  ws.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.type === 'hello') { S.live.avail = !!m.improvise; S.live.on = !!m.improvise_on; S.live.mode = m.mode || null; S.live.nPhrases = m.n_phrases || 0; S.wsSection = null; updateLivePanel(); return; }
    if (m.type === 'live') { onLive(m); return; }
    if (m.type !== 'frame') return;
    if (Math.abs(m.t - songTime()) > 0.5) seekCursors(m.t);
    S.wsT = m.t; S.wsAt = performance.now();
    if (m.spikes && m.spikes.length) S.brain.spike(m.spikes, m.t);
    for (const n of m.notes || []) onNote(n, m.t);
    for (const d of m.drums || []) onDrum(d);
    if (typeof m.section === 'string') S.wsSection = m.section;
  };
  return done;
}

// ------------------------------------------------------------------ launcher: the page starts the Python conductor itself
const sleep = ms => new Promise(r => setTimeout(r, ms));
async function api(path) {
  try { const r = await fetch(path, { cache: 'no-store' }); return r.ok ? await r.json() : null; } catch (e) { return null; }
}
let LAUNCHING = null;
async function refreshLauncher() {
  const st = await api('/api/status');
  $('btnStop').hidden = !(st && st.running);
  return st;
}
function manualHint(mode) {
  return '(cannot start it from here: serve the page with `python -m flybrain_composer.stage_server`, or run `python -m flybrain_composer.cli play' + (mode === 'improvise' ? ' --improvise' : '') + '` yourself and press Conductor)';
}
async function ensureConductor(mode, restart = false) {
  // resolves true once a conductor websocket is reachable; starts one (or restarts it) through the stage server
  if (LAUNCHING) return false;
  const st = await api('/api/status');
  if (!st) { $('clock').textContent = manualHint(mode); return false; }
  if (st.ws_up && !restart) return true;
  const extra = mode === 'generate' ? '&phrase_bars=4&density=9' : '';
  const r = await api(`/api/launch?mode=${mode}&restart=${restart ? 1 : 0}${extra}`);
  if (!r || r.ok === false) { $('clock').textContent = `(${(r && r.hint) || 'could not start the conductor'})`; return false; }
  if (r.external || (r.already && st.ws_up)) return true;
  LAUNCHING = { mode, t0: performance.now() };
  $('btnWs').classList.add('on'); $('btnStop').hidden = false;
  const label = mode === 'improvise' ? 'the live-learning conductor' : 'the conductor';
  try {
    for (let i = 0; i < 120; i++) {
      await sleep(2000);
      const s2 = await api('/api/status');
      if (!s2) continue;
      const secs = Math.round((performance.now() - LAUNCHING.t0) / 1000);
      const raw = (s2.log && s2.log.length) ? s2.log[s2.log.length - 1] : '';
      const last = raw ? raw.replace(/^\[[^\]]*\]\s*/, '').slice(0, 90) : 'loading the brain…';
      $('clock').textContent = `(starting ${label} · ${secs} s · ${last})`;
      if (s2.ws_up) return true;
      if (!s2.running) { $('clock').textContent = `(the conductor exited${s2.exit !== null ? ` with code ${s2.exit}` : ''} — see output/conductor.log${raw ? ': ' + last : ''})`; $('btnWs').classList.remove('on'); return false; }
    }
    $('clock').textContent = '(the conductor is taking too long — see output/conductor.log)';
    return false;
  } finally { LAUNCHING = null; refreshLauncher(); }
}
async function connectConductor(mode = 'play') {
  if (S.mode === 'ws' && S.ws && S.ws.readyState === 1) { if (mode !== 'improvise' || S.live.avail) return true; }
  let ok = await startWs();
  if (!ok) { ok = await ensureConductor(mode); if (ok) { await sleep(500); ok = await startWs(); } }
  return ok;
}
async function stopConductor() {
  closeWs(); S.mode = 'idle'; $('btnWs').classList.remove('on');
  $('clock').textContent = '(stopping the conductor…)';
  await api('/api/stop'); await refreshLauncher();
  $('clock').textContent = '(conductor stopped)';
}

// ------------------------------------------------------------------ live learning (Stage B in real time)
function wsSend(obj) { if (S.ws && S.ws.readyState === 1) { S.ws.send(JSON.stringify(obj)); return true; } return false; }

function onLive(m) {
  S.live.on = !!m.on; S.live.phrase = m.phrase || null; S.live.planning = m.planning || null;
  S.live.history = m.history || []; S.live.nPhrases = m.n_phrases || S.live.nPhrases;
  if (m.mode) S.live.mode = m.mode; if (m.weights) S.live.weights = m.weights; if (m.human) S.live.human = m.human;
  if (m.last_human) S.live.lastHuman = m.last_human; if (m.cycle16) S.live.cycle16 = m.cycle16;
  const rep = S.live.history.find(h => !h.improvised);
  if (rep && S.live.replayLevel === null) S.live.replayLevel = rep.fitness;
  updateLivePanel();
}

function updateLivePanel() {
  const L = S.live, P = $('livePanel');
  if (!L.avail) { P.hidden = true; $('tglLive').classList.remove('on'); return; }
  P.hidden = false;
  $('tglLive').classList.toggle('on', L.on);
  const gen = L.mode === 'generate';
  const mode = $('liveMode');
  mode.textContent = gen ? `DJENT GENERATOR${L.cycle16 ? ` · ${L.cycle16}/16 cycle` : ''}` : (L.on ? 'LIVE LEARNING' : 'REPLAY (learning off)');
  mode.classList.toggle('off', !L.on && !gen);
  $('btnGen').classList.toggle('on', gen);
  const ph = L.phrase;
  $('livePhrase').textContent = ph ? `${gen ? 'riff' : 'phrase'} ${ph.k + 1} / ${ph.n_phrases}` : 'phrase —';
  if (ph) {
    const tag = gen ? 'its own riff' : (ph.improvised ? 'improvised' : 'replay');
    const nov = (ph.raw && Number.isFinite(ph.raw.novelty)) ? ` · novelty ${(100 * ph.raw.novelty).toFixed(0)}%` : '';
    $('liveNow').textContent = `${tag} · ${ph.n_notes} notes · groove score ${ph.fitness.toFixed(3)}${nov}${ph.improvised ? ` · ${ph.n_cand} candidates auditioned` : ''}`;
    const kb = ph.knobs;
    $('liveKnobs').textContent = kb ? `displace ${kb.disp16 >= 0 ? '+' : ''}${kb.disp16.toFixed(1)}/16 · cycle ×${kb.stretch.toFixed(2)} · blend ${(100 * kb.blend).toFixed(0)}% · drums ×${kb.gains.drums.toFixed(2)} riff ×${kb.gains.riff.toFixed(2)} form ×${kb.gains.form.toFixed(2)} · threshold ${kb.thr.toFixed(2)}` : (gen ? '—' : 'the tab, as written');
    const tr = ph.reward_out;
    $('liveTreat').textContent = `${tr >= 0 ? '+' : ''}${tr.toFixed(2)} ${tr > 0.005 ? '→ PAM (reward)' : tr < -0.005 ? '→ PPL1 (punishment)' : '(neutral)'} · running avg ${ph.baseline.toFixed(3)}`;
  } else { $('liveNow').textContent = '—'; $('liveTreat').textContent = '—'; $('liveKnobs').textContent = '—'; }
  const pl = L.planning;
  if (pl) $('liveNext').textContent = `auditioning ${gen ? 'riff' : 'phrase'} ${pl.k + 1}: ${pl.n_cand} candidates, ${pl.n_gen} generations · best ${pl.best.toFixed(3)} · ${Math.max(0, pl.time_left).toFixed(0)} s left`;
  else $('liveNext').textContent = (L.on || gen) ? 'waiting for the next phrase boundary' : 'next phrase will be the Stage A replay';
  const h = L.human;
  $('liveHuman').textContent = h ? `👍 ${h.up} · 👎 ${h.down}${L.lastHuman ? ` · last ${L.lastHuman.value > 0 ? '👍' : '👎'} at ${L.lastHuman.t.toFixed(0)} s` : ''} — K / J while it plays` : '— press K / J while it plays';
  const W = L.weights;
  if (W) {
    const base = { groove: 2.0, low_focus: 1.0, chug_ratio: 1.0, density: 1.0, syncopation: 0.8, vocab: 0.6, novelty: 2.0, balance: 1.5 };
    const parts = Object.keys(W).filter(k => base[k] !== undefined).map(k => { const r = W[k] / base[k]; return r > 1.15 ? `${k} ↑${r.toFixed(1)}` : r < 0.87 ? `${k} ↓${r.toFixed(1)}` : null; }).filter(Boolean);
    $('liveTaste').textContent = parts.length ? parts.join(' · ') : 'default (your 👍/👎 re-weight what it scores)';
  } else $('liveTaste').textContent = '—';
  // history bars
  const H = $('liveHist'); H.innerHTML = '';
  const lo = 0.3, hi = 0.9;
  for (const h of L.history) {
    const i = document.createElement('i'); i.className = h.improvised ? '' : 'replay';
    i.style.height = `${Math.max(2, 34 * (h.fitness - lo) / (hi - lo))}px`; i.title = `phrase ${h.k + 1}: ${h.fitness.toFixed(3)} (${h.improvised ? 'improvised' : 'replay'})`;
    H.appendChild(i);
  }
  if (L.replayLevel !== null) { const b = document.createElement('div'); b.className = 'base'; b.style.bottom = `${34 * (L.replayLevel - lo) / (hi - lo)}px`; H.appendChild(b); }
}

async function vote(value) {
  if (S.mode !== 'ws' || !S.live.avail) { $('clock').textContent = '(👍/👎 need the conductor running with live learning or the generator)'; return; }
  wsSend({ cmd: 'reward', value });
  const b = $(value > 0 ? 'voteUp' : 'voteDown'); b.classList.add('on'); setTimeout(() => b.classList.remove('on'), 350);
}
async function startGenerator() {
  if (LAUNCHING) return;
  if (S.mode === 'ws' && S.live.mode === 'generate') return;
  if (AUDIO.playing) { AUDIO.offset = songTime(); stopStems(); $('btnPlay').classList.remove('on'); }
  const restart = S.mode === 'ws';
  if (restart) { $('clock').textContent = '(restarting the conductor as the riff generator…)'; closeWs(); S.mode = 'idle'; }
  const ok = await ensureConductor('generate', restart);
  if (ok) { await sleep(500); await startWs(); }
}
async function toggleLive() {
  if (S.mode === 'ws' && S.live.avail) {
    wsSend({ cmd: 'improvise', on: !S.live.on });
    S.live.on = !S.live.on; updateLivePanel();          // optimistic; the conductor confirms with a "live" message
    return;
  }
  if (LAUNCHING) return;
  if (S.mode === 'ws') {
    // a conductor without --improvise is running: restart it as the live-learning one
    $('clock').textContent = '(restarting the conductor with live learning…)';
    closeWs(); S.mode = 'idle';
    const ok = await ensureConductor('improvise', true);
    if (ok) { await sleep(500); await startWs(); }
    return;
  }
  if (AUDIO.playing) { AUDIO.offset = songTime(); stopStems(); $('btnPlay').classList.remove('on'); }
  const ok = await ensureConductor('improvise');
  if (ok) { await sleep(500); await startWs(); }
}

// ------------------------------------------------------------------ cameras
function setCamera(mode) {
  S.camMode = mode;
  document.querySelectorAll('[data-cam]').forEach(b => b.classList.toggle('on', b.dataset.cam === mode));
  controls.enabled = mode === 'orbit';
  const views = {
    audience: [[5.6, 2.2, 1.7], [-0.5, 1.25, 0.0]],
    side: [[1.4, 1.7, 5.4], [0.1, 1.05, 0.0]],
    fretboard: [[2.4, 1.7, -1.9], [-0.3, 0.95, -0.6]],
    brain: [[2.2, 3.2, 0.8], [-3.2, 2.6, 0]],
  };
  if (views[mode]) { S.camPos.set(...views[mode][0]); S.camTgt.set(...views[mode][1]); }
  if (mode === 'orbit') { controls.target.copy(S.camTgt); controls.update(); }
}

// ------------------------------------------------------------------ main loop
let prev = performance.now();
function frame() {
  requestAnimationFrame(frame);
  const now = performance.now();
  const dtRaw = Math.max(1e-3, (now - prev) / 1000);
  const dt = Math.min(0.05, dtRaw); prev = now;
  if (!S.fly || !S.guitar) return;
  const t = songTime();
  if (S.mode === 'audio' && S.raster) {
    if (t < S.lastT - 0.5) seekCursors(t);
    const ids = spikesUntil(S.raster, t);
    if (ids.length) S.brain.spike(ids, t);
    S.spikeRate += (ids.length / Math.max(1e-3, t - S.lastT) - S.spikeRate) * 0.08;
    advanceCursors(t);
  } else if (S.mode === 'ws') {
    if (S.ws && S.ws.readyState === 1) S.spikeRate += (S.brain.spikeCount / dtRaw - S.spikeRate) * 0.08;
    S.brain.spikeCount = 0;
  }
  S.brain.setTime(t);
  animateFly(dt, t);
  if (S.holo) S.holo.rotation.y = -Math.PI / 2 + Math.sin(t * 0.25) * 0.35;
  if (S.rois) S.rois.rotation.copy(S.holo.rotation);
  // camera
  if (S.camMode !== 'orbit') {
    const sway = S.camMode === 'audience' ? Math.sin(t * 0.3) * 0.25 : 0;
    tmpV.copy(S.camPos); tmpV.z += sway; tmpV.y += S.bob * 0.01;
    camera.position.lerp(tmpV, Math.min(1, dt * 2.5));
    tmpV2.copy(S.camTgt); camera.lookAt(tmpV2);
  } else controls.update();
  // hud
  S.lastT = Math.max(S.lastT, t);
  const sec = S.sections.find(s => t >= s.t0 && t < s.t1);
  $('section').textContent = (S.mode === 'ws' && S.wsSection !== null) ? (S.wsSection || '—') : (sec ? sec.name : '—');
  $('time').textContent = `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, '0')}`;
  $('spk').textContent = `${Math.round(S.spikeRate)} /s`;
  $('barFill').style.width = `${Math.min(100, 100 * t / (S.song?.duration || 1))}%`;
  if (++S.frames % 30 === 0) { $('fps').textContent = `${Math.round(30000 / (now - S.fpsT))} fps`; S.fpsT = now; }
  bloom.enabled ? composer.render() : renderer.render(scene, camera);
}

// ------------------------------------------------------------------ ui
$('btnPlay').onclick = togglePause;
$('btnWs').onclick = () => connectConductor('play');
$('btnStop').onclick = stopConductor;
document.querySelectorAll('[data-cam]').forEach(b => b.onclick = () => setCamera(b.dataset.cam));
$('tglHolo').onclick = e => { S.holo.visible = !S.holo.visible; e.target.classList.toggle('on', S.holo.visible); if (S.rois) S.rois.visible = S.holo.visible && $('tglRois').classList.contains('on'); };
$('tglHead').onclick = e => { S.headBrain.visible = !S.headBrain.visible; e.target.classList.toggle('on', S.headBrain.visible); };
$('tglBloom').onclick = e => { bloom.enabled = !bloom.enabled; e.target.classList.toggle('on', bloom.enabled); };
$('tglRois').onclick = e => { if (S.rois) { S.rois.visible = !S.rois.visible; e.target.classList.toggle('on', S.rois.visible); } };
$('startStandalone').onclick = () => { $('overlay').style.display = 'none'; startAudio(false); };
$('startFlyOnly').onclick = () => { $('overlay').style.display = 'none'; startAudio(true); };
function clickStem(name) {
  if (S.mode === 'ws') { const btn = { guitar: $('stemGuitar'), backing: $('stemBacking'), drums: $('stemDrums') }[name]; const on = !btn.classList.contains('on'); wsSend({ cmd: 'stem', name, on }); btn.classList.toggle('on', on); return; }
  setStem(name, !stemOn(name));
}
$('stemGuitar').onclick = () => clickStem('guitar');
$('stemBacking').onclick = () => clickStem('backing');
$('stemDrums').onclick = () => clickStem('drums');
$('tglLive').onclick = toggleLive;
$('btnGen').onclick = startGenerator;
$('voteUp').onclick = () => vote(+1);
$('voteDown').onclick = () => vote(-1);
window.addEventListener('keydown', e => {
  if (S.placing || e.target.tagName === 'INPUT') return;
  if (e.key === 'g') clickStem('guitar'); if (e.key === 'b') clickStem('backing'); if (e.key === 'd') clickStem('drums');
  if (e.key === 'k' || e.key === 'K' || e.key === '+') vote(+1);
  if (e.key === 'j' || e.key === 'J' || e.key === '-') vote(-1);
  if (e.key === 'l' || e.key === 'L') toggleLive();
});
$('startConductor').onclick = () => { $('overlay').style.display = 'none'; connectConductor('play'); };
$('startLive').onclick = () => { $('overlay').style.display = 'none'; toggleLive(); };
$('startGen').onclick = () => { $('overlay').style.display = 'none'; startGenerator(); };
refreshLauncher();
window.addEventListener('resize', () => { camera.aspect = window.innerWidth / window.innerHeight; camera.updateProjectionMatrix(); renderer.setSize(window.innerWidth, window.innerHeight); composer.setSize(window.innerWidth, window.innerHeight); });
window.addEventListener('keydown', e => { if (placementKey(e)) return; if (e.code === 'Space') { e.preventDefault(); togglePause(); } if (e.key >= '1' && e.key <= '5') setCamera(['audience', 'side', 'fretboard', 'brain', 'orbit'][e.key - 1]); });

loadAll().then(() => { $('startStandalone').disabled = false; frame(); }).catch(e => { $('loading').textContent = 'failed: ' + e; console.error(e); });
window.S = S; window.scene = scene; window.camera = camera; window.THREE = THREE;
