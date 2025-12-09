Adaptive Swedish Language Tutor - Revisions After Feedback
===========================================================

Summary
-------
- Clarified engagement buckets and mapping from emotions.
- Scoped ASR/TTS to Furhat defaults; focus time on the dialogue manager.
- Reframed latency as a constraint, not a primary metric.
- Made interaction-appropriateness metric concrete.
- Pulled dialogue subsystem work earlier in the plan.

Engagement Buckets (Objective 1)
--------------------------------
- Buckets: `low` (confused/frustrated), `medium` (neutral/uncertain), `high` (engaged/happy).
- Mapping: DiffusionFER logits -> softmax -> exponential moving average (0.6-0.8 s window) -> bucket by confidence bands:
  - `low` if (confused + frustrated) >= 0.45 after smoothing.
  - `high` if engaged/happy >= 0.55.
  - otherwise `medium`.
- Temporal guard: require 0.8 s persistence before changing bucket to avoid jitter. Output schema: `{bucket, top_emotion, confidence, timestamp}`.

Dialogue/ASR/TTS Scope (Objective 2)
------------------------------------
- Use Furhat SDK built-in Swedish ASR/TTS for the baseline; Whisper small (sv) only as an optional stretch.
- Core work: rule-based/state-machine dialogue manager that consumes `bucket` updates and adjusts prompt difficulty, speaking rate, and nonverbal behaviours.
- Risks moved to focus list: error-handling on ASR hypotheses, simple barge-in handling, and concise repair prompts.

Metrics
-------
- Metric 1 (kept): Perception macro-F1 >= 0.75 on held-out affect buckets.
- Metric 2 (reframed): Latency is a constraint, not a success metric. Target <150-200 ms webcam->bucket on test laptop; monitor but not graded.
- Metric 3 (made concrete): Interaction appropriateness via human ratings.
  - Protocol: 5-10 scripted sessions (neutral + injected confusion moments). Two raters mark each system turn as `appropriate`/`inappropriate` based on whether adaptation matched the scripted user state and stayed on task.
  - Metric: >= 80% of turns tagged `appropriate`; report inter-rater agreement (Cohen's kappa) as a sanity check.

Time Plan Adjustments
---------------------
- Before individual feedback: start Subsystem 2 alongside Subsystem 1.
- Updated milestones (dates keep course cadence; adjust locally if needed):
  - Dec 2: Perception baseline (detection + pretrained emotion head).
  - Dec 9: Plenary prep + baseline metrics; dialogue manager skeleton consuming mocked buckets; Furhat ASR/TTS "hello world".
  - Dec 12: Fine-tune emotion head; add smoothing; ONNX export.
  - Dec 18: Individual feedback: show updated perception metrics + dialogue demo with real-time bucket hooks.
  - Dec 22: Dialogue policies + behaviour library wired; logging.
  - Jan 5: Full integration run (perception + dialogue + behaviours) with 2 scripted sessions logged.
  - Jan 10: Robustness pass (latency profiling, fallback prompts); demo cut prep.
  - Jan 14: Presentation/demo.
  - Jan 16: Final report.

Planned Updates to Main Spec
----------------------------
- Replace Obj 1 text with the bucket mapping above.
- Scope Obj 2 to Furhat ASR/TTS by default; mark Whisper as optional.
- Move latency from "Metric 2" to a constraint note; keep monitoring target.
- Replace "Metric 3" with the rater-based appropriateness metric and protocol.
- Swap the time plan table/Gantt to show Subsystem 2 starting pre-feedback.
