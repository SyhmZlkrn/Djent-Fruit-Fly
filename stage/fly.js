// The NeuroMechFly body: glTF node hierarchy == MJCF bodies, driven through the real hinge joints.
import * as THREE from 'three';

const DEG = Math.PI / 180;

// generic joint limits (radians) by joint role — keeps the CCD solver from folding legs backwards
const LIMITS = {
  Coxa_yaw: [-70 * DEG, 70 * DEG], Coxa: [-100 * DEG, 100 * DEG], Coxa_roll: [-120 * DEG, 150 * DEG],
  Femur: [-170 * DEG, 25 * DEG], Femur_roll: [-70 * DEG, 70 * DEG],
  Tibia: [-30 * DEG, 165 * DEG], Tarsus1: [-70 * DEG, 70 * DEG],
  Tarsus2: [-60 * DEG, 60 * DEG], Tarsus3: [-60 * DEG, 60 * DEG], Tarsus4: [-60 * DEG, 60 * DEG], Tarsus5: [-60 * DEG, 60 * DEG],
};
// the front legs play the guitar: wider ranges, the pole vector decides the configuration
// (symmetric: the hinge axes are the body's y axis on both sides, so the left leg needs the
// opposite bend sign from the right for the same posture)
const FRONT_LIMITS = {
  Coxa_yaw: [-150 * DEG, 150 * DEG], Coxa: [-170 * DEG, 170 * DEG], Coxa_roll: [-180 * DEG, 180 * DEG],
  Femur: [-175 * DEG, 175 * DEG], Femur_roll: [-160 * DEG, 160 * DEG],
  Tibia: [-172 * DEG, 172 * DEG], Tarsus1: [-110 * DEG, 110 * DEG],
  Tarsus2: [-35 * DEG, 35 * DEG], Tarsus3: [-35 * DEG, 35 * DEG], Tarsus4: [-35 * DEG, 35 * DEG], Tarsus5: [-35 * DEG, 35 * DEG],
};

function limitFor(jointName) {
  const m = jointName.match(/^joint_([LR])([FMH])(Coxa_yaw|Coxa_roll|Coxa|Femur_roll|Femur|Tibia|Tarsus[1-5])$/);
  if (!m) return null;
  return (m[2] === 'F' ? FRONT_LIMITS : LIMITS)[m[3]];
}

export class FlyRig {
  constructor(root, rig) {
    this.root = root;            // the glTF scene
    this.rig = rig;              // fly_rig.json
    this.nodes = new Map();
    this.rest = new Map();
    this.angles = new Map();     // joint name -> radians
    this.jointsOf = new Map();   // body -> [{name, axis: Vector3}]
    this.extra = new Map();      // body -> Quaternion for bodies without MJCF joints (wings)
    root.traverse(o => { if (o.name) this.nodes.set(o.name, o); });
    for (const name of rig.order) {
      const body = rig.bodies[name];
      const node = this.nodes.get(name);
      if (!node) continue;
      this.rest.set(name, node.quaternion.clone());
      const js = body.joints.map(j => ({ name: j.name, axis: new THREE.Vector3(...j.axis).normalize() }));
      this.jointsOf.set(name, js);
      for (const j of js) this.angles.set(j.name, 0);
    }
    // leg chains: root -> tip, as [bodyName, jointIndex]
    this.legs = {};
    for (const side of ['L', 'R']) for (const seg of ['F', 'M', 'H']) {
      const p = side + seg;
      this.legs[p] = {
        chain: [[p + 'Coxa', 0], [p + 'Coxa', 1], [p + 'Coxa', 2], [p + 'Femur', 0], [p + 'Femur', 1], [p + 'Tibia', 0], [p + 'Tarsus1', 0],
                [p + 'Tarsus2', 0], [p + 'Tarsus3', 0], [p + 'Tarsus4', 0], [p + 'Tarsus5', 0]],
        tipNode: this.nodes.get(p + 'Tarsus5'),
        tipOffset: this._tipOffset(p + 'Tarsus5'),
      };
    }
    this._tmp = { v1: new THREE.Vector3(), v2: new THREE.Vector3(), v3: new THREE.Vector3(), q: new THREE.Quaternion(), q2: new THREE.Quaternion(), p: new THREE.Vector3(), a: new THREE.Vector3() };
  }

  // distal end of the last tarsal segment, in that node's local frame (segments hang along -z)
  _tipOffset(name) {
    const node = this.nodes.get(name);
    const box = new THREE.Box3();
    node.traverse(o => { if (o.isMesh) { o.geometry.computeBoundingBox(); box.union(o.geometry.boundingBox); } });
    return new THREE.Vector3(0, 0, box.isEmpty() ? -0.08 : box.min.z);
  }

  setJoint(name, rad) { if (this.angles.has(name)) this.angles.set(name, rad); }
  getJoint(name) { return this.angles.get(name) || 0; }

  applyPoseDeg(pose) { for (const [k, v] of Object.entries(pose)) this.setJoint(k, v * DEG); }

  // local quaternion of a body = rest * q(j1) * q(j2) * ...   (MuJoCo composite hinge order)
  updateBody(name) {
    const node = this.nodes.get(name), js = this.jointsOf.get(name);
    if (!node || !js || !js.length) return;
    const q = this._tmp.q.copy(this.rest.get(name));
    for (const j of js) {
      this._tmp.q2.setFromAxisAngle(j.axis, this.angles.get(j.name));
      q.multiply(this._tmp.q2);
    }
    node.quaternion.copy(q);
  }

  // free rotation about a local axis for bodies that have no joints in the MJCF (the wings)
  setFree(name, axis, angle) {
    const node = this.nodes.get(name);
    if (!node) return;
    if (!this.extra.has(name)) this.extra.set(name, new THREE.Quaternion());
    node.quaternion.copy(this.rest.get(name)).multiply(this.extra.get(name).setFromAxisAngle(axis, angle));
  }

  update() { for (const name of this.jointsOf.keys()) this.updateBody(name); }

  worldPos(name, out = new THREE.Vector3()) {
    const n = this.nodes.get(name);
    this.root.updateMatrixWorld(true);
    return n.getWorldPosition(out);
  }

  tipWorld(leg, out = new THREE.Vector3()) {
    const L = this.legs[leg];
    return out.copy(L.tipOffset).applyMatrix4(L.tipNode.matrixWorld);
  }

  // world-space axis of joint i of a body (axis lives in the frame after the preceding joints)
  jointWorldAxis(body, idx, out) {
    const node = this.nodes.get(body), js = this.jointsOf.get(body);
    const q = this._tmp.q;
    node.parent.getWorldQuaternion(q);
    q.multiply(this.rest.get(body));
    for (let k = 0; k < idx; k++) { this._tmp.q2.setFromAxisAngle(js[k].axis, this.angles.get(js[k].name)); q.multiply(this._tmp.q2); }
    return out.copy(js[idx].axis).applyQuaternion(q).normalize();
  }

  // Cyclic-coordinate-descent IK for one leg onto a world target (hinge axes respected, limits clamped)
  solveLeg(leg, target, iters = 6, damping = 0.75) {
    const L = this.legs[leg];
    const { v1, v2, v3, p, a } = this._tmp;
    const bodies = new Set(L.chain.map(c => c[0]));
    const updateChain = () => { for (const b of bodies) this.updateBody(b); this.nodes.get(L.chain[0][0]).updateMatrixWorld(true); };
    updateChain();
    for (let it = 0; it < iters; it++) {
      for (let c = 6; c >= 0; c--) {
        const [body, idx] = L.chain[c];
        const js = this.jointsOf.get(body)[idx];
        const node = this.nodes.get(body);
        node.getWorldPosition(p);
        this.jointWorldAxis(body, idx, a);
        this.tipWorld(leg, v1).sub(p);
        v2.copy(target).sub(p);
        // project onto the plane perpendicular to the hinge axis
        v1.addScaledVector(a, -v1.dot(a));
        v2.addScaledVector(a, -v2.dot(a));
        if (v1.lengthSq() < 1e-10 || v2.lengthSq() < 1e-10) continue;
        v1.normalize(); v2.normalize();
        let ang = Math.atan2(v3.crossVectors(v1, v2).dot(a), v1.dot(v2)) * damping;
        let val = this.angles.get(js.name) + ang;
        const lim = limitFor(js.name);
        if (lim) val = Math.min(lim[1], Math.max(lim[0], val));
        this.angles.set(js.name, val);
        this.updateBody(body);
        node.updateMatrixWorld(true);
      }
      if (this.tipWorld(leg, v1).distanceTo(target) < 0.004) break;
    }
  }
}

// Two-stage IK: first put the knee (femur/tibia joint) at `knee` using the coxa + femur joints,
// then reach `tip` with the tibia + tarsus (femur allowed to help a little). This pins the leg's
// path - the fretting leg comes up under the neck, the picking leg over the body - instead of
// letting plain CCD wander into folded, crossed configurations.
FlyRig.prototype.solveLegVia = function (leg, knee, tip, itersA = 5, itersB = 6) {
  const L = this.legs[leg];
  const { v1, v2, v3, p, a } = this._tmp;
  const kneeNode = this.nodes.get(L.chain[5][0]);           // the Tibia body origin == knee
  const bodies = new Set(L.chain.map(c => c[0]));
  for (const b of bodies) this.updateBody(b);
  this.nodes.get(L.chain[0][0]).updateMatrixWorld(true);
  const stage = (indices, effector, target, iters, damping) => {
    for (let it = 0; it < iters; it++) {
      for (let k = indices.length - 1; k >= 0; k--) {
        const [body, idx] = L.chain[indices[k]];
        const js = this.jointsOf.get(body)[idx];
        const node = this.nodes.get(body);
        node.getWorldPosition(p);
        this.jointWorldAxis(body, idx, a);
        effector(v1).sub(p);
        v2.copy(target).sub(p);
        v1.addScaledVector(a, -v1.dot(a)); v2.addScaledVector(a, -v2.dot(a));
        if (v1.lengthSq() < 1e-10 || v2.lengthSq() < 1e-10) continue;
        v1.normalize(); v2.normalize();
        const ang = Math.atan2(v3.crossVectors(v1, v2).dot(a), v1.dot(v2)) * damping;
        let val = this.angles.get(js.name) + ang;
        const lim = limitFor(js.name);
        if (lim) val = Math.min(lim[1], Math.max(lim[0], val));
        this.angles.set(js.name, val);
        this.updateBody(body);
        node.updateMatrixWorld(true);
      }
    }
  };
  stage([0, 1, 2, 3, 4], out => kneeNode.getWorldPosition(out), knee, itersA, 0.7);
  stage([3, 4, 5, 6], out => this.tipWorld(leg, out), tip, itersB, 0.7);
};

// Segment lengths of a leg from the rig (child body offsets): L1 = coxa + femur, L2 = tibia + tarsus.
FlyRig.prototype.legLengths = function (leg) {
  const b = this.rig.bodies, n = s => Math.hypot(...b[s].pos);
  // child offsets: Femur.pos = coxa length, Tibia.pos = femur length, Tarsus1.pos = tibia length,
  // TarsusK.pos = length of tarsus K-1; the last tarsal segment is ~0.09 mm
  const L1 = n(leg + 'Femur') + n(leg + 'Tibia');
  const tarsus = n(leg + 'Tarsus2') + n(leg + 'Tarsus3') + n(leg + 'Tarsus4') + n(leg + 'Tarsus5') + 0.09;
  return { L1, L2: n(leg + 'Tarsus1') + tarsus * 0.85, tibia: n(leg + 'Tarsus1'), tarsus };
};

// Arm solve with an explicit wrist: coxa+femur -> elbow (knee), femur roll + tibia -> wrist (Tarsus1
// origin), then the five tarsal hinges curl the "fingers" onto the fingertip target. Controlling the
// wrist is what makes a fretting hand come from UNDER the neck and a picking hand hang DOWN over the
// strings, instead of only the fingertip being right.
FlyRig.prototype.solveArm = function (leg, knee, wrist, tip, iters = 6) {
  const L = this.legs[leg];
  const { v1, v2, v3, p, a } = this._tmp;
  const kneeNode = this.nodes.get(L.chain[5][0]);
  const wristNode = this.nodes.get(L.chain[6][0]);
  const bodies = new Set(L.chain.map(c => c[0]));
  for (const b of bodies) this.updateBody(b);
  this.nodes.get(L.chain[0][0]).updateMatrixWorld(true);
  const stage = (indices, effector, target, n, damping) => {
    for (let it = 0; it < n; it++) {
      for (let k = indices.length - 1; k >= 0; k--) {
        const [body, idx] = L.chain[indices[k]];
        const js = this.jointsOf.get(body)[idx];
        const node = this.nodes.get(body);
        node.getWorldPosition(p);
        this.jointWorldAxis(body, idx, a);
        effector(v1).sub(p);
        v2.copy(target).sub(p);
        v1.addScaledVector(a, -v1.dot(a)); v2.addScaledVector(a, -v2.dot(a));
        if (v1.lengthSq() < 1e-10 || v2.lengthSq() < 1e-10) continue;
        v1.normalize(); v2.normalize();
        const ang = Math.atan2(v3.crossVectors(v1, v2).dot(a), v1.dot(v2)) * damping;
        let val = this.angles.get(js.name) + ang;
        const lim = limitFor(js.name);
        if (lim) val = Math.min(lim[1], Math.max(lim[0], val));
        this.angles.set(js.name, val);
        this.updateBody(body);
        node.updateMatrixWorld(true);
      }
    }
  };
  const tipFn = out => this.tipWorld(leg, out), wristFn = out => wristNode.getWorldPosition(out);
  stage([0, 1, 2, 3, 4], out => kneeNode.getWorldPosition(out), knee, iters, 0.7);
  stage([2, 4, 5], wristFn, wrist, iters, 0.7);
  // fingers: the tarsal hinges share one axis, so let the roll joints turn the curl plane toward the
  // target, then re-seat the wrist and curl again
  stage([2, 4, 6, 7, 8, 9, 10], tipFn, tip, iters + 2, 0.6);
  stage([4, 5], wristFn, wrist, 2, 0.6);
  stage([6, 7, 8, 9, 10], tipFn, tip, iters, 0.6);
};

// Arm solve with a FIXED hand shape: the tarsal joints are set to `hand` (radians, tarsus1..5), then
// coxa+femur -> knee and roll+tibia -> wrist. Returns the fingertip so the caller can iterate the
// wrist target until the tip lands where it should (used for the picking hand: fingers curled like
// holding a pick, hanging from the wrist).
FlyRig.prototype.solveArmHand = function (leg, knee, wrist, hand, iters = 6, out = new THREE.Vector3()) {
  const L = this.legs[leg];
  const { v1, v2, v3, p, a } = this._tmp;
  for (let k = 0; k < 5; k++) this.angles.set(this.jointsOf.get(L.chain[6 + k][0])[0].name, hand[k]);
  const kneeNode = this.nodes.get(L.chain[5][0]);
  const wristNode = this.nodes.get(L.chain[6][0]);
  const bodies = new Set(L.chain.map(c => c[0]));
  for (const b of bodies) this.updateBody(b);
  this.nodes.get(L.chain[0][0]).updateMatrixWorld(true);
  const stage = (indices, effector, target, n, damping) => {
    for (let it = 0; it < n; it++) {
      for (let k = indices.length - 1; k >= 0; k--) {
        const [body, idx] = L.chain[indices[k]];
        const js = this.jointsOf.get(body)[idx];
        const node = this.nodes.get(body);
        node.getWorldPosition(p);
        this.jointWorldAxis(body, idx, a);
        effector(v1).sub(p);
        v2.copy(target).sub(p);
        v1.addScaledVector(a, -v1.dot(a)); v2.addScaledVector(a, -v2.dot(a));
        if (v1.lengthSq() < 1e-10 || v2.lengthSq() < 1e-10) continue;
        v1.normalize(); v2.normalize();
        const ang = Math.atan2(v3.crossVectors(v1, v2).dot(a), v1.dot(v2)) * damping;
        let val = this.angles.get(js.name) + ang;
        const lim = limitFor(js.name);
        if (lim) val = Math.min(lim[1], Math.max(lim[0], val));
        this.angles.set(js.name, val);
        this.updateBody(body);
        node.updateMatrixWorld(true);
      }
    }
  };
  stage([0, 1, 2, 3, 4], o => kneeNode.getWorldPosition(o), knee, iters, 0.7);
  stage([2, 4, 5], o => wristNode.getWorldPosition(o), wrist, iters, 0.7);
  return this.tipWorld(leg, out);
};

// Planar arm solve. Every joint after the coxa is a hinge about the same body axis, so the whole leg
// (femur, tibia, tarsi) bends in ONE plane that the coxa orients. Given the fingertip target T, the
// shoulder S, a pole (which side the elbow goes) and an `approach` direction for the hand (which way
// the fingers travel onto the target), with the tarsal joints held in a fixed hand shape:
//   plane = (S, T, pole);  hand vector h = approach projected into the plane;
//   wrist W = T - D*h (D = length of the fixed hand);
//   forearm direction u = h rotated in-plane by the wrist bend beta, beta chosen (smallest first) so
//   that the elbow K = W - tibia*u is within reach of the shoulder;
//   CCD: coxa+femur -> K, rolls+tibia -> W, tarsus1 -> T (consistent targets, so it converges).
FlyRig.prototype.solveArmPlanar = function (leg, tip, pole, approach, hand, iters = 6, state = null, isOk = null, planeFrom = 'approach') {
  const L = this.legs[leg];
  const kneeNode = this.nodes.get(L.chain[5][0]);
  const wristNode = this.nodes.get(L.chain[6][0]);
  for (let k = 0; k < 5; k++) this.angles.set(this.jointsOf.get(L.chain[6 + k][0])[0].name, hand[k]);
  const bodies = new Set(L.chain.map(c => c[0]));
  for (const b of bodies) this.updateBody(b);
  this.nodes.get(L.chain[0][0]).updateMatrixWorld(true);
  const S = this.nodes.get(L.chain[0][0]).getWorldPosition(new THREE.Vector3());
  const lens = this.legLengths(leg);
  const D = wristNode.getWorldPosition(new THREE.Vector3()).distanceTo(this.tipWorld(leg, new THREE.Vector3()));
  // the plane contains the shoulder, the fingertip and the hand's approach direction, so the hand
  // can approach exactly as asked; the pole only decides which side the elbow bends to
  const e1 = tip.clone().sub(S).normalize();
  let h, n;
  if (planeFrom === 'pole') {
    // plane through shoulder, fingertip and the pole (elbow side); the approach is projected into it
    n = new THREE.Vector3().crossVectors(e1, pole).normalize();
    h = approach.clone().addScaledVector(n, -approach.dot(n)).normalize();
  } else {
    // plane through shoulder, fingertip and the approach direction, so the hand comes in exactly as asked
    h = approach.clone().normalize();
    n = new THREE.Vector3().crossVectors(e1, h);
    if (n.lengthSq() < 1e-6) n.crossVectors(e1, pole);
    n.normalize();
  }
  const W = tip.clone().addScaledVector(h, -D);
  const side = new THREE.Vector3().crossVectors(n, h).normalize();      // in-plane, perpendicular to h
  if (side.dot(pole) < 0) side.negate();
  // forearm direction u = h rotated in-plane by the wrist bend; pick the bend whose elbow is reachable,
  // allowed by isOk (e.g. not inside the body) and closest to the pole direction
  let K = null, bestScore = -1e9;
  const reachMin = lens.L1 * 0.35, reachMax = lens.L1 * 0.97;
  const poleN = pole.clone().normalize();
  for (let beta = 10; beta <= 150; beta += 10) {
    const b = beta * DEG;
    const u = h.clone().multiplyScalar(Math.cos(b)).addScaledVector(side, Math.sin(b)).normalize();
    const cand = W.clone().addScaledVector(u, -lens.tibia);
    const d = cand.distanceTo(S);
    if (d < reachMin || d > reachMax) continue;
    if (isOk && !isOk(cand, W)) continue;
    const score = cand.clone().sub(S).normalize().dot(poleN) - Math.abs(beta - 60) / 300;
    if (score > bestScore) { bestScore = score; K = cand; }
  }
  if (!K) K = FlyRig.kneeFor(S, W, lens.L1, lens.tibia, pole, new THREE.Vector3());
  const { v1, v2, v3, p, a } = this._tmp;
  const stage = (indices, effector, target, nIt, damping) => {
    for (let it = 0; it < nIt; it++) {
      for (let k = indices.length - 1; k >= 0; k--) {
        const [body, idx] = L.chain[indices[k]];
        const js = this.jointsOf.get(body)[idx];
        const node = this.nodes.get(body);
        node.getWorldPosition(p);
        this.jointWorldAxis(body, idx, a);
        effector(v1).sub(p);
        v2.copy(target).sub(p);
        v1.addScaledVector(a, -v1.dot(a)); v2.addScaledVector(a, -v2.dot(a));
        if (v1.lengthSq() < 1e-10 || v2.lengthSq() < 1e-10) continue;
        v1.normalize(); v2.normalize();
        const ang = Math.atan2(v3.crossVectors(v1, v2).dot(a), v1.dot(v2)) * damping;
        let val = this.angles.get(js.name) + ang;
        const lim = limitFor(js.name);
        if (lim) val = Math.min(lim[1], Math.max(lim[0], val));
        this.angles.set(js.name, val);
        this.updateBody(body);
        node.updateMatrixWorld(true);
      }
    }
  };
  const kneeFn = o => kneeNode.getWorldPosition(o), wristFn = o => wristNode.getWorldPosition(o), tipFn = o => this.tipWorld(leg, o);
  stage([0, 1, 2, 3, 4], kneeFn, K, iters, 0.7);
  stage([2, 4, 5], wristFn, W, iters, 0.7);
  // fingers: the tarsal hinges (one-way limits, so they curl like fingers instead of coiling) plus
  // the roll joints that turn the hand's swing plane; then re-seat the wrist and repeat
  for (let round = 0; round < 3; round++) {
    stage([2, 4, 6, 7, 8, 9, 10], tipFn, tip, iters, 0.6);
    stage([3, 4, 5], wristFn, W, 3, 0.6);
  }
  stage([6, 7, 8, 9, 10], tipFn, tip, 4, 0.7);
  // whatever the fingers could not close, move the wrist by (a share of) the fingertip error and redo
  const errV = tipFn(v1).clone().sub(tip);
  if (errV.length() > 0.01) {
    W.addScaledVector(errV, -0.8);
    stage([0, 1, 2, 3, 4, 5], wristFn, W, iters, 0.6);
    stage([2, 4, 6, 7, 8, 9, 10], tipFn, tip, iters, 0.6);
    stage([3, 4, 5], wristFn, W, 2, 0.5);
    stage([6, 7, 8, 9, 10], tipFn, tip, 3, 0.7);
  }
  if (state) { state.W = W.clone(); state.K = K.clone(); state.h = h.clone(); }
  return tipFn(v1).distanceTo(tip);
};

// Two-bone IK elbow: the knee lies on the circle where sphere(shoulder, L1) meets sphere(target, L2);
// pick the point closest to the pole direction (world). Returns a world position.
FlyRig.kneeFor = function (shoulder, target, L1, L2, pole, out = new THREE.Vector3()) {
  const d = target.clone().sub(shoulder);
  let dist = d.length();
  const dMin = Math.abs(L1 - L2) + 0.02, dMax = L1 + L2 - 0.02;
  dist = Math.max(dMin, Math.min(dMax, dist));
  d.normalize();
  const a = (L1 * L1 - L2 * L2 + dist * dist) / (2 * dist);
  const h = Math.sqrt(Math.max(0, L1 * L1 - a * a));
  const mid = shoulder.clone().addScaledVector(d, a);
  const pp = pole.clone().addScaledVector(d, -pole.dot(d));
  if (pp.lengthSq() < 1e-8) pp.set(0, 1, 0).addScaledVector(d, -d.y);
  pp.normalize();
  return out.copy(mid).addScaledVector(pp, h);
};

FlyRig.prototype.kneeWorld = function (leg, out = new THREE.Vector3()) { return this.nodes.get(leg + 'Tibia').getWorldPosition(out); };

export function loadRig(gltfScene, rigJson) {
  return new FlyRig(gltfScene, rigJson);
}
