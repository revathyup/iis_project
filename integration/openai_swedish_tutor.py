"""Interactive Swedish tutor using Virtual Furhat Realtime API + live affect buckets + OpenAI.

This runs perception (webcam -> OpenFace emotion -> engagement bucket) in-process so you can
type answers in the terminal while the camera is running (stdin is not consumed by a pipe).

Environment:
  - Set OPENAI_API_KEY to enable OpenAI responses. If not set, the tutor runs without OpenAI.
"""

from __future__ import annotations

import asyncio
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import cv2
import websockets

# Allow running as `python integration/openai_swedish_tutor.py` (so `perception` is importable).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from perception.camera import WebcamStream
from perception.detector import HaarFaceDetector, MediaPipeFaceDetector
from perception.engagement import EngagementBucketMapper, softmax
from perception.openface_classifier import OpenFaceEmotionClassifier

# Keep ASCII-only (Windows terminal / Furhat TTS).
LESSON = [
    {"sv": "Vad heter du?", "en": "What is your name?", "answer": "Jag heter ____."},
    {"sv": "Hur mar du?", "en": "How are you?", "answer": "Jag mar bra. / Jag mar inte sa bra."},
    {"sv": "Var bor du?", "en": "Where do you live?", "answer": "Jag bor i ____."},
    {"sv": "Var kommer du ifran?", "en": "Where do you come from?", "answer": "Jag kommer fran ____."},
    {"sv": "Vad ar klockan?", "en": "What time is it?", "answer": "Klockan ar ____."},
]


def _now() -> float:
    return time.time()


def _log(debug: bool, msg: str) -> None:
    if not debug:
        return
    sys.stderr.write(msg.rstrip() + "\n")
    sys.stderr.flush()


def crop_face(frame, box, margin=0.2):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    w *= 1 + margin
    h *= 1 + margin
    nx1 = int(max(cx - w / 2, 0))
    ny1 = int(max(cy - h / 2, 0))
    nx2 = int(min(cx + w / 2, frame.shape[1]))
    ny2 = int(min(cy + h / 2, frame.shape[0]))
    return frame[ny1:ny2, nx1:nx2], (nx1, ny1, nx2, ny2)


def draw_overlay(frame, box, bucket: str, top_emotion: str, confidence: float) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"{bucket} ({top_emotion}, {confidence:.2f})"
    cv2.putText(frame, text, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def pick_bucket_majority(buckets: list[str], *, min_majority: float = 0.6) -> str:
    if not buckets:
        return "medium"
    counts = Counter(buckets)
    best_count = max(counts.values())
    if best_count / max(len(buckets), 1) < min_majority:
        return "medium"
    tied = [b for b, c in counts.items() if c == best_count]
    if len(tied) == 1:
        return tied[0]
    if "medium" in tied:
        return "medium"
    if "high" in tied:
        return "high"
    return tied[0]


@dataclass(frozen=True)
class PerceptionEvent:
    timestamp: float
    bucket: str
    candidate_bucket: str
    top_emotion: str
    confidence: float


class PerceptionWorker:
    def __init__(
        self,
        *,
        openface_weights: str,
        openface_classifier: str,
        labels: list[str],
        openface_device: str,
        openface_features: str,
        camera_index: int,
        display: bool,
        crop_margin: float,
        bucket_mode: str,
        low_threshold: float,
        high_threshold: float,
        guard_seconds: float,
        alpha: float,
        activity_threshold: Optional[float],
        activity_alpha: float,
        fps_limit: float,
        debug: bool,
    ) -> None:
        self._stop = threading.Event()
        self._events: Deque[PerceptionEvent] = deque(maxlen=600)
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._debug = debug

        self._labels = labels
        self._display = display
        self._crop_margin = crop_margin
        self._camera_index = camera_index
        self._fps_limit = fps_limit

        self._clf = OpenFaceEmotionClassifier(
            weights_path=openface_weights,
            classifier_path=openface_classifier,
            labels=labels,
            device=openface_device,
            features=openface_features,
        )

        mapper_kwargs: Dict[str, object] = {
            "labels": labels,
            "bucket_mode": bucket_mode,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "guard_seconds": guard_seconds,
            "alpha": alpha,
            "activity_alpha": activity_alpha,
        }
        if activity_threshold is not None:
            mapper_kwargs["activity_threshold"] = activity_threshold
        self._mapper = EngagementBucketMapper(**mapper_kwargs)

        try:
            self._detector = MediaPipeFaceDetector()
        except Exception:
            self._detector = HaarFaceDetector()

        self._thread = threading.Thread(target=self._run, name="PerceptionWorker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def wait_ready(self, timeout_s: float = 10.0) -> bool:
        return self._ready.wait(timeout=timeout_s)

    def snapshot_since(self, after_ts: float) -> list[PerceptionEvent]:
        with self._lock:
            return [e for e in list(self._events) if e.timestamp >= after_ts]

    def latest(self) -> Optional[PerceptionEvent]:
        with self._lock:
            return self._events[-1] if self._events else None

    def _run(self) -> None:
        last_box: Optional[Tuple[int, int, int, int]] = None
        last_frame_t = 0.0

        with WebcamStream(device_index=self._camera_index) as stream:
            self._ready.set()
            for frame in stream.frames():
                if self._stop.is_set():
                    break

                if self._fps_limit > 0:
                    dt = frame.timestamp - last_frame_t
                    if dt > 0 and dt < (1.0 / self._fps_limit):
                        continue
                    last_frame_t = frame.timestamp

                faces = self._detector.detect(frame.image)
                if faces:
                    last_box = faces[0]
                elif last_box is None:
                    continue

                face_crop, box = crop_face(frame.image, last_box, margin=self._crop_margin)
                if face_crop.size == 0:
                    continue

                pred = self._clf.predict_logits(face_crop)
                logits = {label: float(pred.logits[i]) for i, label in enumerate(pred.labels)}
                signal = self._mapper.update(logits, timestamp=frame.timestamp)

                evt = PerceptionEvent(
                    timestamp=signal.timestamp,
                    bucket=signal.bucket,
                    candidate_bucket=signal.candidate_bucket,
                    top_emotion=signal.top_emotion,
                    confidence=float(signal.confidence),
                )
                with self._lock:
                    self._events.append(evt)

                if self._debug:
                    probs = softmax(logits, labels=self._labels)
                    top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
                    sys.stderr.write(f"Top3: {top3}\n")

                if self._display:
                    draw_overlay(frame.image, box, evt.bucket, evt.top_emotion, evt.confidence)
                    cv2.imshow("engagement", frame.image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break


class FurhatClient:
    def __init__(self, *, uri: str, key: Optional[str], debug: bool) -> None:
        self._uri = uri
        self._key = key
        self._debug = debug
        self._ws = None
        self._speaking = False
        self._speak_end = asyncio.Event()  # set per speak
        self._msg_q: Optional[asyncio.Queue[dict]] = None

    async def __aenter__(self) -> "FurhatClient":
        _log(self._debug, f"[furhat] connecting to {self._uri}")
        self._ws = await websockets.connect(self._uri)
        if self._key:
            await self._ws.send(json.dumps({"type": "request.auth", "key": self._key}))
        self._msg_q = asyncio.Queue(maxsize=500)
        self._recv_task = asyncio.create_task(self._recv_loop())
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            self._recv_task.cancel()
        except Exception:
            pass
        if self._ws is not None:
            await self._ws.close()

    async def _recv_loop(self) -> None:
        try:
            while True:
                raw = await self._ws.recv()
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                if not isinstance(msg, dict):
                    continue
                t = msg.get("type")
                if t == "response.speak.start":
                    self._speaking = True
                elif t == "response.speak.end":
                    self._speaking = False
                    self._speak_end.set()

                if self._msg_q is not None:
                    try:
                        self._msg_q.put_nowait(msg)
                    except asyncio.QueueFull:
                        try:
                            _ = self._msg_q.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        try:
                            self._msg_q.put_nowait(msg)
                        except asyncio.QueueFull:
                            pass
        except Exception:
            return

    async def speak(self, text: str, *, gesture: Optional[str] = None, wait_end: bool = True) -> None:
        if self._ws is None:
            raise RuntimeError("Not connected")
        self._speak_end = asyncio.Event()
        await self._ws.send(json.dumps({"type": "request.speak.text", "text": text}))
        if gesture:
            await self._ws.send(json.dumps({"type": "request.gesture.start", "name": gesture}))
        if wait_end:
            await self._speak_end.wait()

    async def listen_text(
        self,
        *,
        timeout_s: float = 8.0,
        request_type: str = "request.listen.start",
        stop_type: str = "request.listen.stop",
        accept_any_type: bool = True,
        dump_messages: bool = False,
        debug: bool = False,
    ) -> Optional[str]:
        if self._ws is None or self._msg_q is None:
            raise RuntimeError("Not connected")

        # Drain old messages so we don't pick up stale transcripts.
        drained = 0
        try:
            while True:
                _ = self._msg_q.get_nowait()
                drained += 1
        except asyncio.QueueEmpty:
            pass
        if debug and drained:
            _log(True, f"[furhat][asr] drained {drained} queued messages")

        # Start listening.
        await self._ws.send(json.dumps({"type": request_type}))

        deadline = _now() + max(0.1, float(timeout_s))
        transcript: Optional[str] = None
        last_text: Optional[str] = None

        def _extract_text(msg: dict) -> Optional[str]:
            # Common patterns across realtime APIs.
            for key in ("text", "transcript", "utterance"):
                val = msg.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            hyps = msg.get("hypotheses")
            if isinstance(hyps, list) and hyps:
                first = hyps[0]
                if isinstance(first, dict):
                    for key in ("text", "transcript", "utterance"):
                        val = first.get(key)
                        if isinstance(val, str) and val.strip():
                            return val.strip()
            # Some servers wrap payload.
            payload = msg.get("payload")
            if isinstance(payload, dict):
                for key in ("text", "transcript", "utterance"):
                    val = payload.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
            return None

        while _now() < deadline:
            remaining = max(0.05, deadline - _now())
            try:
                msg = await asyncio.wait_for(self._msg_q.get(), timeout=remaining)
            except asyncio.TimeoutError:
                continue

            t = msg.get("type")
            if dump_messages or (debug and isinstance(t, str) and any(k in t for k in ("listen", "asr", "speech"))):
                _log(True, f"[furhat][asr] {msg}")

            text = _extract_text(msg)
            if text is None:
                continue

            # Keep the latest candidate in case we never see a clear "final" message.
            last_text = text

            # Prefer to accept clearly ASR-ish event types, but allow any type if configured.
            is_asr_type = isinstance(t, str) and any(k in t.lower() for k in ("listen", "asr", "speech", "recogn"))
            if not accept_any_type and not is_asr_type:
                continue

            # Avoid accidentally consuming Furhat's own TTS echoes.
            if isinstance(t, str) and "speak" in t.lower():
                continue

            # If the message looks like a final result, stop early.
            is_final = False
            for key in ("final", "is_final", "done"):
                if isinstance(msg.get(key), bool) and msg.get(key) is True:
                    is_final = True
                    break
            if isinstance(t, str) and any(k in t.lower() for k in ("final", "end", "result")):
                is_final = True

            if is_final:
                transcript = text
                break

        # Stop listening (best-effort).
        try:
            await self._ws.send(json.dumps({"type": stop_type}))
        except Exception:
            pass

        return transcript or last_text


def openai_generate(
    *,
    api_key: Optional[str],
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_s: float = 30.0,
    debug: bool = False,
) -> Optional[Dict[str, str]]:
    if not api_key:
        return None
    url = "https://api.openai.com/v1/responses"
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "text": {"format": {"type": "json_object"}},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        method="POST",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if debug:
            sys.stderr.write(f"[openai] HTTPError {e.code}: {e.read().decode('utf-8', errors='replace')}\n")
        return None
    except Exception as e:
        if debug:
            sys.stderr.write(f"[openai] error: {e}\n")
        return None

    try:
        obj = json.loads(raw)
        text_out = obj.get("output_text")
        if isinstance(text_out, str) and text_out.strip():
            return json.loads(text_out)
    except Exception:
        return None
    return None


def build_bucket_summary(events: list[PerceptionEvent], *, min_majority: float) -> Tuple[str, Optional[str]]:
    if not events:
        return "medium", None
    buckets = [e.bucket for e in events]
    bucket = pick_bucket_majority(buckets, min_majority=min_majority)
    # pick the most frequent top emotion in that window
    emos = [e.top_emotion for e in events if e.top_emotion]
    top_emotion = Counter(emos).most_common(1)[0][0] if emos else None
    return bucket, top_emotion


def default_feedback(bucket: str, item: dict, user_answer: str) -> str:
    answer_clean = str(item["answer"]).rstrip().rstrip(".")
    if bucket == "low":
        return f"Ok. Lets try again. Say: {answer_clean}."
    if bucket == "high":
        return "Great!"
    return "Good."


def make_system_prompt() -> str:
    return (
        "You are a Swedish language tutor embodied by a Furhat social robot. "
        "Your output MUST be a JSON object with keys: say (string). "
        "Be brief (1-2 sentences). "
        "If the user answer is wrong or English, give the correct Swedish and ask them to repeat. "
        "If bucket is low, be extra supportive and give a short hint. "
        "Do not use emojis."
    )


def make_user_prompt(*, bucket: str, top_emotion: Optional[str], question_sv: str, question_en: str, expected: str, user_answer: str) -> str:
    return (
        f"bucket={bucket}\n"
        f"top_emotion={top_emotion or ''}\n"
        f"question_sv={question_sv}\n"
        f"question_en={question_en}\n"
        f"expected_answer_template={expected}\n"
        f"user_answer={user_answer}\n"
        "Return JSON: {\"say\":\"...\"}"
    )


def build_question_text(item: dict) -> str:
    answer_clean = str(item["answer"]).rstrip().rstrip(".")
    return f"Question: {item['sv']} (English: {item['en']}). Answer like this: {answer_clean}."


async def run(args: argparse.Namespace) -> int:
    import asyncio

    api_key = os.environ.get("OPENAI_API_KEY")
    if args.use_openai and not api_key:
        _log(args.debug, "[openai] OPENAI_API_KEY not set; continuing without OpenAI.")

    perception = PerceptionWorker(
        openface_weights=args.openface_weights,
        openface_classifier=args.openface_classifier,
        labels=args.labels,
        openface_device=args.openface_device,
        openface_features=args.openface_features,
        camera_index=args.camera,
        display=args.display,
        crop_margin=args.crop_margin,
        bucket_mode=args.bucket_mode,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        guard_seconds=args.guard_seconds,
        alpha=args.alpha,
        activity_threshold=args.activity_threshold,
        activity_alpha=args.activity_alpha,
        fps_limit=args.fps_limit,
        debug=args.debug,
    )
    perception.start()
    if not perception.wait_ready(timeout_s=10.0):
        raise RuntimeError("Perception did not start (webcam).")

    async with FurhatClient(uri=args.uri, key=args.key, debug=args.debug) as furhat:
        greeting = "Hi! I am your Swedish tutor. Lets practice a few simple questions."
        await furhat.speak(greeting, gesture="Nod", wait_end=True)

        lesson_index = 0
        system_prompt = make_system_prompt()

        while True:
            item = LESSON[lesson_index % len(LESSON)]
            question_text = build_question_text(item)
            _log(args.debug, f"[tutor] ask lesson_index={lesson_index} text={question_text!r}")
            await furhat.speak(question_text, gesture="Nod", wait_end=args.wait_speak_end)

            # Only consider expressions AFTER the question was spoken.
            after_ts = _now()
            await asyncio.sleep(args.settle_seconds)
            window_end = after_ts + args.observe_seconds
            while _now() < window_end:
                await asyncio.sleep(0.05)
            window_events = perception.snapshot_since(after_ts)
            bucket, top_emotion = build_bucket_summary(window_events, min_majority=args.min_majority)
            _log(args.debug, f"[tutor] observed bucket={bucket} top_emotion={top_emotion!r} n={len(window_events)}")

            user_answer = ""
            if args.use_asr:
                user_answer = (
                    await furhat.listen_text(
                        timeout_s=args.listen_timeout,
                        request_type=args.asr_request_type,
                        stop_type=args.asr_stop_type,
                        accept_any_type=not args.asr_strict_types,
                        dump_messages=args.asr_dump,
                        debug=args.debug,
                    )
                ) or ""
                user_answer = user_answer.strip()
                _log(args.debug, f"[tutor] ASR transcript={user_answer!r}")
                if not user_answer:
                    await furhat.speak("I did not catch that. Please say it again.", gesture="Thoughtful", wait_end=args.wait_speak_end)
                    user_answer = (
                        await furhat.listen_text(
                            timeout_s=args.listen_timeout,
                            request_type=args.asr_request_type,
                            stop_type=args.asr_stop_type,
                            accept_any_type=not args.asr_strict_types,
                            dump_messages=args.asr_dump,
                            debug=args.debug,
                        )
                    ) or ""
                    user_answer = user_answer.strip()
                    _log(args.debug, f"[tutor] ASR transcript(retry)={user_answer!r}")
            else:
                # Typed fallback (only if not using ASR).
                try:
                    user_answer = await asyncio.to_thread(input, "You: ")
                except (EOFError, KeyboardInterrupt):
                    break
                user_answer = (user_answer or "").strip()

            say = None
            if args.use_openai and api_key:
                out = openai_generate(
                    api_key=api_key,
                    model=args.openai_model,
                    system_prompt=system_prompt,
                    user_prompt=make_user_prompt(
                        bucket=bucket,
                        top_emotion=top_emotion,
                        question_sv=str(item["sv"]),
                        question_en=str(item["en"]),
                        expected=str(item["answer"]),
                        user_answer=user_answer,
                    ),
                    timeout_s=args.openai_timeout,
                    debug=args.debug,
                )
                if isinstance(out, dict) and isinstance(out.get("say"), str):
                    say = out["say"].strip()
            if not say:
                say = default_feedback(bucket, item, user_answer)

            await furhat.speak(say, gesture="Thoughtful" if bucket == "low" else "BigSmile", wait_end=args.wait_speak_end)

            # Progress logic requested:
            # - low: repeat same question next loop
            # - medium: wait 3s then advance
            # - high: advance immediately
            if bucket == "high":
                lesson_index = (lesson_index + 1) % len(LESSON)
                continue
            if bucket == "medium":
                await asyncio.sleep(args.medium_delay_seconds)
                lesson_index = (lesson_index + 1) % len(LESSON)
                continue
            # low: repeat same question (do not advance)
            continue

    perception.stop()
    return 0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive Swedish tutor (Furhat Realtime + OpenFace + OpenAI).")
    p.add_argument("--uri", required=True, help="Realtime API WS URI (e.g. ws://localhost:9000/v1/events).")
    p.add_argument("--key", default=None, help="Optional auth key if your API requires it.")
    p.add_argument("--camera", type=int, default=0, help="Webcam device index.")
    p.add_argument("--display", action="store_true", help="Show live overlay window (press q to quit window).")
    p.add_argument("--fps-limit", type=float, default=12.0, help="Max camera FPS to process (reduce CPU). 0=unlimited.")
    p.add_argument("--crop-margin", type=float, default=0.2, help="Extra margin around face crop (0..1).")

    p.add_argument("--openface-weights", default="weights/MTL_backbone.pth")
    p.add_argument("--openface-classifier", default="models/openface_emotion_clf.pkl")
    p.add_argument("--openface-features", choices=["emotion+au", "au", "emotion"], default="emotion+au")
    p.add_argument("--openface-device", default="cpu")

    p.add_argument("--labels", nargs="+", required=True, help="Label order for the classifier outputs.")
    p.add_argument("--bucket-mode", choices=["confidence", "emotion"], default="emotion")
    p.add_argument("--low-threshold", type=float, default=0.70)
    p.add_argument("--high-threshold", type=float, default=0.60)
    p.add_argument("--guard-seconds", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--activity-threshold", type=float, default=None, help="Optional: force high on rapid changes.")
    p.add_argument("--activity-alpha", type=float, default=0.5)

    p.add_argument("--observe-seconds", type=float, default=2.0, help="Seconds to observe after each question.")
    p.add_argument("--settle-seconds", type=float, default=0.2, help="Delay after speaking before observing.")
    p.add_argument("--medium-delay-seconds", type=float, default=3.0, help="Delay before advancing on medium.")
    p.add_argument("--min-majority", type=float, default=0.6, help="Majority needed in observation window.")

    p.add_argument("--wait-speak-end", action="store_true", help="Wait for response.speak.end between messages.")

    p.add_argument("--use-asr", action="store_true", help="Use Furhat speech recognition instead of typed answers.")
    p.add_argument("--listen-timeout", type=float, default=8.0, help="Seconds to wait for each spoken answer.")
    p.add_argument(
        "--asr-request-type",
        default="request.listen.start",
        help="Realtime API message type to start listening (differs across Furhat versions).",
    )
    p.add_argument(
        "--asr-stop-type",
        default="request.listen.stop",
        help="Realtime API message type to stop listening (differs across Furhat versions).",
    )
    p.add_argument(
        "--asr-strict-types",
        action="store_true",
        help="Only accept transcripts from message types containing listen/asr/speech/recogn (debug if nothing is captured).",
    )
    p.add_argument("--asr-dump", action="store_true", help="Dump all incoming messages during listening (very verbose).")

    p.add_argument("--use-openai", action="store_true", help="Enable OpenAI feedback (requires OPENAI_API_KEY).")
    p.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI model name.")
    p.add_argument("--openai-timeout", type=float, default=30.0)

    p.add_argument("--debug", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    import asyncio

    args = parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
