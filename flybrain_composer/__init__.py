"""FlyBrain Composer — the MaleCNS connectome as a fixed reservoir that plays Meshuggah.

Pipeline (see PROJECT_PLAN.md):
  connectome  -> pull a central-brain subgraph from neuPrint, cache it, build the signed weight matrix
  meter       -> 4/4 (drums) + 25/16 (riff) + section-cue input streams
  transcription -> "Rational Gaze" target notes (Guitar Pro file, or the built-in structural riff)
  reservoir   -> leaky tanh dynamics over the connectome graph
  train_supervised -> Stage A ridge-regression readout
  train_reward -> Stage B perturb-and-reinforce improvisation (dopaminergic injection site)
  sonify      -> readout -> MIDI
  bridgelite  -> faithful port of the 8ridge lite sampler (renders the MIDI with the plugin's own samples)
  snn         -> Brian2 spiking simulation of the same wiring (for the live neuron animation)
  neuroviz    -> NAVis/octarine real-time "neurons lighting up" viewer
  stage       -> websocket bridge to the three.js fly-plays-M8M stage
"""
__version__ = "0.1.0"
