Integration
===========

Purpose
-------
- Connect perception events to the dialogue manager, provide configuration, logging, overlays, and scripted session playback/recording.

Responsibilities
----------------
- Event plumbing: subscribe to perception bucket stream, forward to dialogue with minimal latency.
- Config: YAML/JSON for thresholds, model paths, and behaviour toggles.
- Logging: per-frame/per-turn logs for latency, bucket decisions, dialogue state, and ASR hypotheses.
- Demo support: simple runner that displays bucket overlays and records demo clips.

Immediate Tasks
---------------
- [ ] Define a lightweight message/event schema shared by perception and dialogue.
- [ ] Create a mock integration runner that feeds recorded bucket traces into the dialogue skeleton.
- [ ] Add logging scaffolding (CSV/JSONL) for both subsystems.
- [ ] Plan scripted sessions and rater sheet for the appropriateness metric.
