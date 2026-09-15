"""The conductor: one clock drives the audio, the spiking brain, the NAVis window and the stage.

    python -m fruit_fly_djent.cli play [--live] [--midi-out "loopMIDI Port"] [--no-window]

* audio: the fly-brain MIDI rendered through the 8ridge lite engine port (+ amp + drums) is
  streamed with sounddevice; the DAC time of the stream is the master clock;
* brain: either the cached Brian2 raster is replayed (default) or the numpy LIF is stepped in
  lockstep with the clock (``--live``) — the neurons light up in the octarine window;
* MIDI: with ``--midi-out`` the note events are also sent in real time to a MIDI port, so the
  *actual* 8ridge lite plugin (in a DAW or its standalone build, via loopMIDI) plays them;
* stage: a WebSocket server (port 8765) broadcasts {t, spikes, note onsets, glow} ~30x/s to the
  three.js fly-plays-the-M8M page (``stage/``), which is served on http://localhost:8000.
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np

from . import config
from .transcription import Note, Song

WS_PORT = 8765
HTTP_PORT = 8000


# ----------------------------------------------------------------------------------------------
# clock + audio
# ----------------------------------------------------------------------------------------------
class AudioClock:
    """Streams stereo float32 stems (mixed live with per-stem gains) and exposes the song time
    actually at the DAC. `audio` may be one array or a dict {name: array}."""

    def __init__(self, audio, sr: int, start_s: float = 0.0, device=None, gain: float = 1.0,
                 gains: dict | None = None):
        self.stems = audio if isinstance(audio, dict) else ({"mix": audio} if audio is not None else {})
        self.gains = {k: 1.0 for k in self.stems}
        self.gains.update(gains or {})
        self.audio = next(iter(self.stems.values())) if self.stems else None
        self.length = max((len(a) for a in self.stems.values()), default=0)
        self.sr = sr
        self.start_s = start_s
        self.gain = gain
        self.pos = int(start_s * sr)
        self.device = device
        self._dac_t0 = None          # DAC time of sample self.pos0
        self._pos0 = self.pos
        self._wall_t0 = None
        self.stream = None
        self.finished = threading.Event()

    def _callback(self, outdata, frames, time_info, status):
        if self._dac_t0 is None:
            self._dac_t0 = time_info.outputBufferDacTime
            self._pos0 = self.pos
        outdata[:] = 0
        end = min(self.pos + frames, self.length)
        for name, a in self.stems.items():
            g = self.gains.get(name, 1.0) * self.gain
            if g <= 0:
                continue
            e = min(end, len(a))
            if e > self.pos:
                outdata[: e - self.pos] += a[self.pos:e] * g
        np.clip(outdata, -1.0, 1.0, out=outdata)
        if end - self.pos < frames:
            self.finished.set()
        self.pos = end

    def toggle(self, name: str) -> bool:
        if name in self.gains:
            self.gains[name] = 0.0 if self.gains[name] > 0 else 1.0
            return self.gains[name] > 0
        return False

    def start(self):
        self._wall_t0 = time.perf_counter()
        if self.audio is None:
            return
        import sounddevice as sd
        self.stream = sd.OutputStream(samplerate=self.sr, channels=2, dtype="float32", blocksize=1024,
                                      callback=self._callback, device=self.device, latency="low")
        self.stream.start()

    def now(self) -> float:
        """Song time (seconds) that is being heard right now."""
        if self.stream is not None and self._dac_t0 is not None:
            return self._pos0 / self.sr + (self.stream.time - self._dac_t0)
        return self.start_s + (time.perf_counter() - self._wall_t0)

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()


# ----------------------------------------------------------------------------------------------
# realtime MIDI out
# ----------------------------------------------------------------------------------------------
def list_midi_ports() -> list[str]:
    import mido
    return list(mido.get_output_names())


class MidiSender(threading.Thread):
    """Sends note on/off events at the right song time to a MIDI output port. Notes can be added
    while it runs (the live improviser commits them phrase by phrase)."""

    def __init__(self, notes: list[Note], bpm: float, port_name: str, clock: AudioClock, channel: int = 0):
        super().__init__(daemon=True)
        import mido
        self.port = mido.open_output(port_name)
        self.mido = mido
        self.sec = config.sixteenth_seconds(bpm)
        self.events: list[tuple] = []
        self.lock = threading.Lock()
        self.i = 0
        self.add_notes(notes)
        self.clock = clock
        self.channel = channel
        self.stop_flag = threading.Event()
        self.open_ended = False      # keep running (waiting for more notes) after the list is exhausted

    def add_notes(self, notes: list[Note]):
        ev = []
        for n in notes:
            ev.append((n.start * self.sec, 1, n.pitch, n.velocity))
            ev.append(((n.start + n.duration) * self.sec - 0.002, 0, n.pitch, 0))
        with self.lock:
            done, todo = self.events[: self.i], self.events[self.i:]
            self.events = done + sorted(todo + ev)

    def run(self):
        # skip events before the start time
        t = self.clock.now()
        with self.lock:
            while self.i < len(self.events) and self.events[self.i][0] < t:
                self.i += 1
        while not self.stop_flag.is_set():
            with self.lock:
                ev = self.events[self.i] if self.i < len(self.events) else None
            if ev is None:
                if not self.open_ended:
                    break
                time.sleep(0.02)
                continue
            t_ev, on, pitch, vel = ev
            now = self.clock.now()
            if t_ev > now + 0.002:
                time.sleep(min(0.01, t_ev - now))
                continue
            msg = self.mido.Message("note_on" if on else "note_off", note=int(pitch), velocity=int(vel),
                                    channel=self.channel)
            self.port.send(msg)
            with self.lock:
                self.i += 1
        self.port.send(self.mido.Message("control_change", control=123, value=0, channel=self.channel))
        self.port.close()


# ----------------------------------------------------------------------------------------------
# websocket + http for the stage
# ----------------------------------------------------------------------------------------------
class StageServer(threading.Thread):
    """Serves stage/ over HTTP and broadcasts JSON frames over a WebSocket."""

    def __init__(self, stage_dir: Path = config.STAGE_DIR, ws_port: int = WS_PORT, http_port: int = HTTP_PORT):
        super().__init__(daemon=True)
        self.stage_dir, self.ws_port, self.http_port = stage_dir, ws_port, http_port
        self.clients = set()
        self.loop = None
        self.ready = threading.Event()
        self.hello = {}
        self.commands: "queue.Queue[dict]" = queue.Queue()     # {"cmd": ...} sent by the stage page

    def run(self):
        try:
            from . import stage_server
            stage_server.IS_CONDUCTOR = True        # /api/launch from our own page must not spawn a second conductor
            stage_server.serve(self.http_port, self.stage_dir, background=True)
        except OSError as e:
            print(f"[play] http port {self.http_port} busy ({e}); assuming stage/ is already being served", flush=True)

        async def main():
            import logging
            import websockets
            logging.getLogger("websockets.server").setLevel(logging.CRITICAL)   # the launcher's port probe is not a handshake

            async def handler(ws):
                self.clients.add(ws)
                try:
                    await ws.send(json.dumps({"type": "hello", **self.hello}))
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except (TypeError, ValueError):
                            continue
                        if isinstance(msg, dict) and "cmd" in msg:
                            self.commands.put(msg)
                finally:
                    self.clients.discard(ws)

            async with websockets.serve(handler, "127.0.0.1", self.ws_port, max_size=None):
                self.ready.set()
                await asyncio.Future()

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(main())

    @staticmethod
    def _json_safe(o):
        """NaN/inf are not JSON (the browser's JSON.parse throws and drops the message): -> null."""
        if isinstance(o, dict):
            return {k: StageServer._json_safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [StageServer._json_safe(v) for v in o]
        if isinstance(o, np.generic):
            o = o.item()
        if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
            return None
        return o

    def broadcast(self, payload: dict):
        if not self.clients or self.loop is None:
            return
        data = json.dumps(self._json_safe(payload))

        async def _send():
            for ws in list(self.clients):
                try:
                    await ws.send(data)
                except Exception:  # noqa: BLE001
                    self.clients.discard(ws)
        asyncio.run_coroutine_threadsafe(_send(), self.loop)


# ----------------------------------------------------------------------------------------------
# the show
# ----------------------------------------------------------------------------------------------
class Show:
    def __init__(self, song: Song, notes: list[Note], audio, sr: int, *,
                 raster=None, live: bool = False, midi_out: str | None = None, window: bool = True,
                 stage: bool = True, start_s: float = 0.0, glow_tau: float = 0.18, audio_gain: float = 1.0,
                 gains: dict | None = None, planner=None):
        from .neuroviz import BrainViewer, RasterCursor
        self.song, self.notes = song, notes
        self.planner = planner                      # live.LivePlanner: the fly improvises phrase by phrase
        self.clock = AudioClock(audio, sr, start_s=start_s, gain=audio_gain, gains=gains)
        self.live_guitar = self.clock.stems.get("guitar") if planner is not None else None
        self.live_drums = self.clock.stems.get("drums") if (planner is not None and getattr(planner.cfg, "mode", "song") == "generate") else None
        # drum hits go to the stage for the head-bangs: the song's own (or the generator's, per phrase)
        _sec = config.sixteenth_seconds(song.bpm)
        self._drum_events: list[tuple[float, Note]] = sorted(((d.start * _sec, d) for d in (song.drums or []) if not song.meta.get("synthetic")), key=lambda x: x[0])
        self._drum_i = 0
        while self._drum_i < len(self._drum_events) and self._drum_events[self._drum_i][0] < start_s:
            self._drum_i += 1
        self.midi = None
        if midi_out:
            try:
                self.midi = MidiSender(notes, song.bpm, midi_out, self.clock)
                self.midi.open_ended = planner is not None
            except Exception as e:  # noqa: BLE001 — Windows MM ports are exclusive; keep the show running
                print(f"[play] could not open MIDI port {midi_out!r}: {e} -- available: {list_midi_ports()}", flush=True)
        self.viewer = BrainViewer(offscreen=not window) if window else None
        if self.viewer is not None:
            # live stem toggles in the brain window: G = fly guitar, B = backing track, D = synth drums
            v = self.viewer.viewer
            for key, stem in (("g", "guitar"), ("b", "backing"), ("d", "drums"), ("m", "mix")):
                v._key_events[key] = (lambda st=stem: self._toggle_stem(st))
            if planner is not None:
                v._key_events["l"] = self._toggle_improv          # L = live learning on/off
                v._key_events["+"] = (lambda: self._human(+1))    # 👍
                v._key_events["="] = (lambda: self._human(+1))
                v._key_events["-"] = (lambda: self._human(-1))    # 👎
        self.glow_tau = glow_tau
        self.N = self.viewer.scene.N if self.viewer else 2318
        self.glow = self.viewer.glow if self.viewer else np.zeros(self.N, np.float32)
        self.live = live
        self.lif = None
        self.cursor = None
        self._lif_parts = None
        if live:
            from .model import ComposerModel
            from .snn import LIFParams, NumpyLIF, grid_dt_ms, meter_current, onset_current, onset_population, recurrent_weights
            U_live = None
            if planner is not None and getattr(planner.cfg, "mode", "song") == "generate":
                from .corpus import MODEL_MULTI_PATH
                from .meter import build_streams_ex, stack_streams
                model = ComposerModel.load(MODEL_MULTI_PATH)
                U_live = stack_streams(build_streams_ex(song, form_mode="code", song_key="gen"))
            else:
                model = ComposerModel.load()
            p = LIFParams()
            We, Wi = recurrent_weights(model, p)
            I_meter = meter_current(model, song, p, U=U_live)
            aud, amps = onset_population(model.W.shape[0])
            # "the fly hears itself": with a planner the onsets arrive phrase by phrase; otherwise the whole take
            I_grid = I_meter.copy()
            if planner is None:
                I_grid += onset_current(notes, len(I_grid), model.W.shape[0], aud, amps, p)
            self.lif = NumpyLIF(We, Wi, I_grid, p, grid_dt_ms=grid_dt_ms(song))
            self.lif.t_ms = start_s * 1000.0
            self._lif_parts = (I_meter, aud, amps, p)
        else:
            from .snn import load_raster
            self.cursor = RasterCursor(raster if raster is not None else load_raster())
            self.cursor.seek(start_s)
        self.stage = StageServer() if stage else None
        sec = config.sixteenth_seconds(song.bpm)
        # fingering from the transcription (the tab's string/fret), not the low-fret heuristic
        truth = {(round(n.start * 2), n.pitch): n for n in song.notes if n.string is not None}
        for n in notes:
            t_ = truth.get((round(n.start * 2), n.pitch))
            if t_ is not None:
                n.string, n.fret, n.palm_mute, n.bend = t_.string, t_.fret, t_.palm_mute, t_.bend
        self._note_events = sorted(((n.start * sec, n) for n in notes), key=lambda x: x[0])
        self._note_i = 0
        self._last_t = start_s
        self._last_ws = 0.0
        self._last_sync = 0.0
        self._last_live_ws = 0.0
        self.spike_buffer: list[np.ndarray] = []
        self.n_spikes = 0
        self.live_state: dict = {"on": planner.enabled if planner is not None else False, "phrase": None,
                                 "planning": None, "phrases": [], "weights": None, "human": {"up": 0, "down": 0},
                                 "last_human": None, "mode": getattr(planner.cfg, "mode", "song") if planner is not None else None}

    # ------------------------------------------------------------------ controls
    def _toggle_stem(self, name: str):
        on = self.clock.toggle(name)
        state = "  ".join(f"{k}:{'ON' if g > 0 else 'off'}" for k, g in self.clock.gains.items())
        print(f"[play] {name} -> {'on' if on else 'off'}   ({state})", flush=True)
        try:
            self.viewer.viewer.show_message(f"{name}: {'ON' if on else 'OFF'}   [{state}]   keys: G guitar / B backing / D drums", duration=3)
        except Exception:  # noqa: BLE001
            pass

    def _toggle_improv(self, on: bool | None = None):
        if self.planner is None:
            return
        on = (not self.planner.enabled) if on is None else bool(on)
        self.planner.set_enabled(on)
        self.live_state["on"] = on
        print(f"[live] live learning {'ON' if on else 'off'} (takes effect at the next phrase)", flush=True)
        if self.viewer is not None:
            try:
                self.viewer.viewer.show_message(f"live learning: {'ON' if on else 'OFF'} (from the next phrase)", duration=3)
            except Exception:  # noqa: BLE001
                pass
        self._broadcast_live()

    def _human(self, value: float):
        """👍 / 👎 for what is playing right now."""
        if self.planner is None:
            return
        t = self.clock.now()
        self.planner.human(value, t)
        self.live_state["human"]["up" if value > 0 else "down"] += 1
        self.live_state["last_human"] = {"value": value, "t": round(t, 2)}
        print(f"[live] {'thumbs UP' if value > 0 else 'thumbs DOWN'} at {t:6.1f}s", flush=True)
        if self.viewer is not None:
            try:
                self.viewer.viewer.show_message("+ treat!" if value > 0 else "- no treat", duration=1.5)
            except Exception:  # noqa: BLE001
                pass
        self._broadcast_live(force=True)

    def _stage_commands(self):
        if self.stage is None:
            return
        while True:
            try:
                cmd = self.stage.commands.get_nowait()
            except queue.Empty:
                return
            if cmd.get("cmd") == "improvise":
                self._toggle_improv(cmd.get("on"))
            elif cmd.get("cmd") == "reward":
                try:
                    self._human(1.0 if float(cmd.get("value", 1)) > 0 else -1.0)
                except (TypeError, ValueError):
                    pass
            elif cmd.get("cmd") == "stem" and cmd.get("name") in self.clock.gains:
                want = cmd.get("on")
                if want is None or (self.clock.gains[cmd["name"]] > 0) != bool(want):
                    self._toggle_stem(cmd["name"])

    # ------------------------------------------------------------------ live phrases
    def add_notes(self, notes: list[Note]):
        """Append future notes (a committed phrase) to the stage/HUD event list and the MIDI sender."""
        sec = config.sixteenth_seconds(self.song.bpm)
        done, todo = self._note_events[: self._note_i], self._note_events[self._note_i:]
        self._note_events = done + sorted(todo + [(n.start * sec, n) for n in notes], key=lambda x: x[0])
        if self.midi is not None:
            self.midi.add_notes(notes)

    def _apply_phrase(self, ph: dict):
        from .live import note_from_dict
        notes = [note_from_dict(d) for d in ph["notes"]]
        # audio straight into the streaming guitar stem
        if self.live_guitar is not None:
            a = ph["audio"]
            s0 = int(ph["s0"])
            e = min(s0 + len(a), len(self.live_guitar))
            if e > s0:
                self.live_guitar[s0:e] = a[: e - s0]
        self.add_notes(notes)
        # generator: its own drums, into the drums stem and to the stage (head-bangs)
        if ph.get("drums"):
            drums = [note_from_dict(d) for d in ph["drums"]]
            sec = config.sixteenth_seconds(self.song.bpm)
            done, todo = self._drum_events[: self._drum_i], self._drum_events[self._drum_i:]
            self._drum_events = done + sorted(todo + [(d.start * sec, d) for d in drums], key=lambda x: x[0])
            if self.live_drums is not None and ph.get("drums_audio") is not None:
                a = ph["drums_audio"]
                s0 = int(ph["s0"])
                e = min(s0 + len(a), len(self.live_drums))
                if e > s0:
                    self.live_drums[s0:e] += a[: e - s0]
        if ph.get("weights"):
            self.live_state["weights"] = ph["weights"]
        # the spiking brain hears what is actually played
        if self.lif is not None and self._lif_parts is not None:
            from .snn import onset_current
            I_meter, aud, amps, p = self._lif_parts
            sps = config.STEPS_PER_SIXTEENTH
            g0, g1 = int(round(ph["s16_0"] * sps)), int(round(ph["s16_1"] * sps))
            g1 = min(g1, len(self.lif.I_grid))
            if g1 > g0:
                self.lif.I_grid[g0:g1] = I_meter[g0:g1] + onset_current(notes, g1 - g0, self.lif.N, aud, amps, p, step0=g0)
        summary = {k: ph[k] for k in ("k", "n_phrases", "t0", "t1", "fitness", "comps", "raw", "improvised", "n_cand",
                                      "n_gen", "reward_in", "reward_out", "baseline", "render_s", "plan_s")}
        summary["n_notes"] = len(notes)
        summary["knobs"] = ph.get("knobs")
        summary["section"] = ph.get("section", "")
        self.live_state["phrase"] = summary
        self.live_state["phrases"].append({k: summary[k] for k in ("k", "fitness", "improvised", "reward_out", "n_notes")})
        self.live_state["planning"] = None
        tag = ("RIFF" if ph.get("mode") == "generate" else "IMPROV") if ph["improvised"] else "replay"
        knobs = ph.get("knobs") or {}
        kn = (f"  disp {knobs['disp16']:+.1f}/16 stretch x{knobs['stretch']:.2f} blend {knobs['blend']:.2f}" if knobs else "")
        nov = ph['raw'].get('novelty')
        print(f"[live] phrase {ph['k'] + 1}/{ph['n_phrases']} [{tag}] {len(notes)} notes  fitness {ph['fitness']:.3f} "
              f"(baseline {ph['baseline']:.3f}, treat {ph['reward_out']:+.2f})"
              f"{'' if nov is None or nov != nov else f'  novelty {nov:.2f}'}  {ph['n_cand']} candidates / {ph['n_gen']} gens  "
              f"render {ph['render_s']:.1f}s{kn}", flush=True)
        self._broadcast_live(force=True)

    def _poll_planner(self, t: float):
        if self.planner is None:
            return
        for msg in self.planner.poll():
            kind, payload = msg[0], (msg[1] if len(msg) > 1 else None)
            if kind == "phrase":
                self._apply_phrase(payload)
            elif kind == "progress":
                self.live_state["planning"] = payload
            elif kind == "taste":
                self.live_state["weights"] = payload.get("weights")
                self.live_state["human"] = payload.get("human", self.live_state["human"])
                self._broadcast_live(force=True)
            elif kind == "log":
                print(f"[live] {payload}", flush=True)
            elif kind == "error":
                print(f"[live] planner error: {payload}", flush=True)
        if t - self._last_sync >= 1.0:
            self.planner.sync(t)
            self._last_sync = t
        if self.live_state.get("planning") is not None and (t - self._last_live_ws) >= 0.25:
            self._broadcast_live()

    def _broadcast_live(self, force: bool = False):
        if self.stage is None or self.planner is None:
            return
        st = self.live_state
        self.stage.broadcast({"type": "live", "on": st["on"], "phrase": st["phrase"], "planning": st["planning"],
                              "history": st["phrases"][-24:], "phrase_bars": self.planner.info.get("phrase_bars"),
                              "n_phrases": self.planner.info.get("n_phrases"), "mode": st.get("mode"),
                              "weights": st.get("weights"), "human": st.get("human"), "last_human": st.get("last_human"),
                              "cycle16": self.planner.info.get("cycle16")})
        self._last_live_ws = self.clock.now() if self.clock._wall_t0 is not None else 0.0

    # ------------------------------------------------------------------ per frame
    def _advance(self, t: float) -> np.ndarray:
        if self.lif is not None:
            budget_ms = max(0.0, t * 1000.0 - self.lif.t_ms)
            if budget_ms > 250:                       # never fall more than 250 ms behind
                self.lif.t_ms = t * 1000.0 - 250
            return self.lif.run_until(t * 1000.0, record=False)
        return self.cursor.advance(t)

    def _notes_between(self, t0: float, t1: float) -> list[dict]:
        out = []
        while self._note_i < len(self._note_events) and self._note_events[self._note_i][0] < t1:
            ts, n = self._note_events[self._note_i]
            if ts >= t0:
                out.append({"t": round(ts, 4), "pitch": n.pitch, "string": n.string, "fret": n.fret,
                            "vel": n.velocity, "dur": round(n.duration * config.sixteenth_seconds(self.song.bpm), 4),
                            "bend": n.bend, "pm": bool(n.palm_mute)})
            self._note_i += 1
        return out

    def _drums_between(self, t0: float, t1: float) -> list[dict]:
        out = []
        while self._drum_i < len(self._drum_events) and self._drum_events[self._drum_i][0] < t1:
            ts, d = self._drum_events[self._drum_i]
            if ts >= t0:
                out.append({"t": round(ts, 4), "pitch": d.pitch, "vel": d.velocity})
            self._drum_i += 1
        return out

    def frame(self, viewer, dt: float):
        t = self.clock.now()
        self._stage_commands()
        self._poll_planner(t)
        spikes = self._advance(t)
        self.n_spikes += len(spikes)
        self.glow *= np.exp(-max(0.0, t - self._last_t) / self.glow_tau)
        if len(spikes):
            self.glow[spikes] = 1.0
        onsets = self._notes_between(self._last_t, t)
        if onsets:
            self._note_buffer = getattr(self, "_note_buffer", []) + onsets
        onsets = None
        drum_hits = self._drums_between(self._last_t, t) if self._drum_events else None
        self.spike_buffer.append(spikes)
        if drum_hits:
            self._drum_buffer = getattr(self, "_drum_buffer", []) + drum_hits
        if self.stage is not None and (t - self._last_ws) >= 1 / 30:
            ids = np.concatenate(self.spike_buffer) if self.spike_buffer else np.zeros(0, int)
            msg = {"type": "frame", "t": round(t, 4), "spikes": ids.tolist(), "notes": getattr(self, "_note_buffer", []),
                   "section": (self.song.section_at(t / config.sixteenth_seconds(self.song.bpm)) or Section0).name}
            if self._drum_events:
                msg["drums"] = getattr(self, "_drum_buffer", [])
                self._drum_buffer = []
            self.stage.broadcast(msg)
            self.spike_buffer, self._last_ws = [], t
            self._note_buffer = []
        self._last_t = t

    def run(self, duration: float | None = None):
        if self.stage is not None:
            self.stage.hello = {"bpm": self.song.bpm, "n_neurons": int(self.N), "title": self.song.title,
                                "duration": self.song.seconds, "improvise": self.planner is not None,
                                "mode": getattr(self.planner.cfg, "mode", "song") if self.planner is not None else None,
                                "improvise_on": bool(self.planner.enabled) if self.planner is not None else False,
                                "phrase_bars": self.planner.info.get("phrase_bars") if self.planner is not None else None}
            self.stage.start()
            self.stage.ready.wait(5)
            print(f"[play] stage: http://localhost:{HTTP_PORT}/  (websocket ws://localhost:{WS_PORT})", flush=True)
        if self.planner is not None:
            # the first phrase (the Stage A replay) must be in the guitar stem before the clock starts
            print("[live] waiting for the planner's first phrase…", flush=True)
            first = self.planner.wait_phrase(timeout=180)
            if first is None:
                raise RuntimeError("the planner did not deliver the first phrase")
            self._apply_phrase(first)
        self.clock.start()
        if self.planner is not None:
            self.planner.start(self.clock.start_s)
        if self.midi is not None:
            self.midi.start()
        end = self.clock.start_s + (duration if duration else self.song.seconds + 3.0)
        stems = ", ".join(f"{k} ({'on' if g > 0 else 'off'})" for k, g in self.clock.gains.items())
        print(f"[play] {self.song.title}: {self.song.bpm} BPM, {len(self.notes)} notes, "
              f"{'live LIF' if self.live else 'Brian2 raster replay'}; stems: {stems}"
              f"  -- keys in the brain window: G guitar, B backing, D drums"
              f"{', L live learning' if self.planner is not None else ''}", flush=True)
        if self.planner is not None:
            print(f"[live] improvising in {self.planner.info.get('phrase_bars')}-bar phrases, "
                  f"{self.planner.info.get('n_phrases')} phrases; live learning "
                  f"{'ON' if self.planner.enabled else 'off'} (toggle on the stage or with L)", flush=True)
        try:
            if self.viewer is not None:
                def cb(viewer, dt):
                    self.frame(viewer, dt)
                    if self.clock.now() >= end:
                        viewer.stop()
                self.viewer.run(cb)
            else:
                while self.clock.now() < end:
                    self.frame(None, 0)
                    time.sleep(1 / 60)
        finally:
            self.clock.stop()
            if self.planner is not None:
                self.planner.stop()
        print(f"[play] done — {self.n_spikes} spikes shown", flush=True)
        if self.planner is not None and self.live_state["phrases"]:
            ph = self.live_state["phrases"]
            imp = [p for p in ph if p["improvised"]]
            print(f"[live] {len(imp)}/{len(ph)} phrases improvised; mean fitness improvised "
                  f"{np.mean([p['fitness'] for p in imp]) if imp else float('nan'):.3f} vs replay "
                  f"{np.mean([p['fitness'] for p in ph if not p['improvised']]) if len(imp) < len(ph) else float('nan'):.3f}", flush=True)


class _Section0:
    name = ""


Section0 = _Section0()


def load_audio(path: Path):
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return audio, sr
