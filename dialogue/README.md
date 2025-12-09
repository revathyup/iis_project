Dialogue Subsystem
==================

Purpose
-------
- Rule/state-machine dialogue that adapts prompt difficulty, speaking rate, and nonverbal behaviours based on engagement buckets from perception.
- Uses Furhat SDK built-in Swedish ASR/TTS by default; Whisper (sv) is an optional stretch.

Planned Components
------------------
- Dialogue state machine with scripted tutoring flow (warm-up, prompt, feedback, repair/help, recap).
- Adaptation policies: map bucket changes to actions (e.g., slow down + add hint on `low`, maintain pace on `medium`, introduce challenge on `high`).
- ASR/TTS glue: Furhat SDK intents + error handling for noisy hypotheses; minimal barge-in handling.
- Behaviour library: gestures/movements/SSML snippets keyed by adaptation action.

Immediate Tasks
---------------
- [ ] Implement a dialogue skeleton that consumes mocked bucket events (CLI or JSON stream).
- [ ] Define adaptation policy table and tie to Furhat behaviours.
- [ ] Build Furhat ASR/TTS "hello world" and hook to the skeleton.
- [ ] Add logging of turn-level context for the appropriateness metric.

Notes
-----
- Keep bucket input schema `{bucket, top_emotion, confidence, timestamp}` stable to simplify integration.
- Guard against rapid bucket oscillations; treat `medium` as the neutral baseline.
