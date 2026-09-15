// The MaleCNS subgraph as a holographic brain: one LineSegments + one Points, per-neuron
// "last spike time" in a data texture, glow computed on the GPU (exp fade), additive blending.
import * as THREE from 'three';

const vertLines = /* glsl */`
  attribute float owner;
  uniform sampler2D uLast;
  uniform float uTime, uTau, uN;
  varying float vGlow;
  void main() {
    float last = texture2D(uLast, vec2((owner + 0.5) / uN, 0.5)).r;
    float dt = uTime - last;
    vGlow = (last < 0.0 || dt < 0.0) ? 0.0 : exp(-dt / uTau);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }`;

const fragLines = /* glsl */`
  varying float vGlow;
  uniform float uRestAlpha, uBoost;
  vec3 cmap(float g) {
    vec3 rest = vec3(0.10, 0.20, 0.62), mid = vec3(0.95, 0.15, 0.08), hot = vec3(1.0, 0.62, 0.12), peak = vec3(1.0, 1.0, 0.85);
    if (g < 0.45) return mix(rest, mid, g / 0.45);
    if (g < 0.75) return mix(mid, hot, (g - 0.45) / 0.30);
    return mix(hot, peak, (g - 0.75) / 0.25);
  }
  void main() {
    float a = uRestAlpha + vGlow * uBoost;
    gl_FragColor = vec4(cmap(vGlow) * a, 1.0);
  }`;

const vertPoints = /* glsl */`
  attribute float owner;
  attribute float dan;
  uniform sampler2D uLast;
  uniform float uTime, uTau, uN, uSize, uScale;
  varying float vGlow; varying float vDan;
  void main() {
    float last = texture2D(uLast, vec2((owner + 0.5) / uN, 0.5)).r;
    float dt = uTime - last;
    vGlow = (last < 0.0 || dt < 0.0) ? 0.0 : exp(-dt / uTau);
    vDan = dan;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = uSize * (1.0 + 4.0 * vGlow + 0.8 * dan) * uScale / -mv.z;
    gl_Position = projectionMatrix * mv;
  }`;

const fragPoints = /* glsl */`
  varying float vGlow; varying float vDan;
  void main() {
    vec2 d = gl_PointCoord - 0.5;
    float r = length(d);
    if (r > 0.5) discard;
    float soft = smoothstep(0.5, 0.15, r);
    vec3 rest = mix(vec3(0.25, 0.35, 0.9), vec3(0.95, 0.25, 0.95), vDan);
    vec3 hot = vec3(1.0, 0.95, 0.7);
    vec3 c = mix(rest, hot, vGlow);
    float a = soft * (0.25 + 0.45 * vDan + 1.6 * vGlow);
    gl_FragColor = vec4(c * a, 1.0);
  }`;

export class Brain {
  constructor(positions, owner, somas, meta) {
    this.N = meta.n_neurons;
    this.last = new Float32Array(this.N).fill(-1);
    this.tex = new THREE.DataTexture(this.last, this.N, 1, THREE.RedFormat, THREE.FloatType);
    this.tex.minFilter = this.tex.magFilter = THREE.NearestFilter;
    this.tex.needsUpdate = true;
    this.uniforms = {
      uLast: { value: this.tex }, uTime: { value: 0 }, uTau: { value: 0.18 }, uN: { value: this.N },
      uRestAlpha: { value: 0.0012 }, uBoost: { value: 0.03 },
    };
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('owner', new THREE.BufferAttribute(Float32Array.from(owner), 1));
    this.lines = new THREE.LineSegments(geo, new THREE.ShaderMaterial({
      uniforms: this.uniforms, vertexShader: vertLines, fragmentShader: fragLines,
      transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    const pgeo = new THREE.BufferGeometry();
    pgeo.setAttribute('position', new THREE.BufferAttribute(somas, 3));
    pgeo.setAttribute('owner', new THREE.BufferAttribute(Float32Array.from({ length: this.N }, (_, i) => i), 1));
    const dan = new Float32Array(this.N); for (const i of meta.dan) dan[i] = 1;
    pgeo.setAttribute('dan', new THREE.BufferAttribute(dan, 1));
    this.pointUniforms = { ...this.uniforms, uSize: { value: 9.0 }, uScale: { value: 1.0 } };
    this.points = new THREE.Points(pgeo, new THREE.ShaderMaterial({
      uniforms: this.pointUniforms, vertexShader: vertPoints, fragmentShader: fragPoints,
      transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    this.group = new THREE.Group();
    this.group.add(this.lines, this.points);
    this.spikeCount = 0;
  }

  spike(ids, t) {
    for (let i = 0; i < ids.length; i++) this.last[ids[i]] = t;
    this.spikeCount += ids.length;
    this.tex.needsUpdate = true;
  }

  reset() { this.last.fill(-1); this.tex.needsUpdate = true; }

  setTime(t) { this.uniforms.uTime.value = t; }

  // a second instance of the same geometry (the true-scale copy inside the head): its own
  // materials with much lower alphas, because 2,318 neurons in 0.8 mm saturate otherwise
  clone(restAlpha = 0.0005, boost = 0.012, pointSize = 0) {
    const g = new THREE.Group();
    const u = { ...this.uniforms, uRestAlpha: { value: restAlpha }, uBoost: { value: boost } };
    const l = new THREE.LineSegments(this.lines.geometry, new THREE.ShaderMaterial({
      uniforms: u, vertexShader: vertLines, fragmentShader: fragLines, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
    g.add(l);
    if (pointSize > 0) {
      const pu = { ...u, uSize: { value: pointSize }, uScale: { value: 1.0 } };
      g.add(new THREE.Points(this.points.geometry, new THREE.ShaderMaterial({
        uniforms: pu, vertexShader: vertPoints, fragmentShader: fragPoints, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })));
    }
    return g;
  }
}

export async function loadBrain(base = './data/') {
  const [meta, pos, own, som] = await Promise.all([
    fetch(base + 'brain.json').then(r => r.json()),
    fetch(base + 'brain.bin').then(r => r.arrayBuffer()),
    fetch(base + 'brain_owner.bin').then(r => r.arrayBuffer()),
    fetch(base + 'somas.bin').then(r => r.arrayBuffer()),
  ]);
  return new Brain(new Float32Array(pos), new Uint16Array(own), new Float32Array(som), meta);
}

export async function loadSpikes(base = './data/') {
  const [t, n] = await Promise.all([
    fetch(base + 'spikes_t.bin').then(r => r.arrayBuffer()),
    fetch(base + 'spikes_n.bin').then(r => r.arrayBuffer()),
  ]);
  return { t: new Float32Array(t), n: new Uint16Array(n), i: 0 };
}

// advance a raster cursor to time t; returns the neuron ids that spiked since the last call
export function spikesUntil(raster, t) {
  const start = raster.i;
  let j = start;
  while (j < raster.t.length && raster.t[j] <= t) j++;
  raster.i = j;
  return raster.n.subarray(start, j);
}

export function seekSpikes(raster, t) {
  let lo = 0, hi = raster.t.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (raster.t[m] < t) lo = m + 1; else hi = m; }
  raster.i = lo;
}
