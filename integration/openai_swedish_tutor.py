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


@dataclass
class AffectiveFeedback:
    key: str
    sv: str
    en: str
    gesture: Optional[str] = None


@dataclass
class AffectiveState:
    last_key: str = ""
    last_ts: float = 0.0


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
        log_emotions: bool,
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
        self._log_emotions = log_emotions

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

                if self._debug and self._log_emotions:
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
            if dump_messages:
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

    def _extract_text(resp: dict) -> Optional[str]:
        text_out = resp.get("output_text")
        if isinstance(text_out, str) and text_out.strip():
            return text_out.strip()
        output = resp.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, list):
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                            return part.get("text").strip()
                        if part.get("type") == "text" and isinstance(part.get("text"), str):
                            return part.get("text").strip()
        return None

    def _try_parse_json(text: str) -> Optional[Dict[str, str]]:
        try:
            return json.loads(text)
        except Exception:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None
        return None

    try:
        obj = json.loads(raw)
    except Exception:
        if debug:
            sys.stderr.write("[openai] failed to parse response JSON\n")
        return None

    text = _extract_text(obj)
    if text:
        parsed = _try_parse_json(text)
        if parsed is not None:
            return parsed
    if debug:
        preview = raw[:500].replace("\n", " ")
        sys.stderr.write(f"[openai] could not parse JSON output. raw preview: {preview}\n")
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


def average_confidence(events: list[PerceptionEvent]) -> float:
    if not events:
        return 0.0
    return sum(e.confidence for e in events) / max(1, len(events))


def select_affective_feedback(
    *,
    top_emotion: Optional[str],
    confidence: float,
    min_confidence: float,
) -> Optional[AffectiveFeedback]:
    if not top_emotion or confidence < min_confidence:
        return None
    emo = top_emotion.strip().lower()
    if emo in ("happy", "surprise"):
        return AffectiveFeedback(
            key=emo,
            sv="Du ser glad ut. Bra jobbat!",
            en="You look happy. Nice work!",
            gesture="BigSmile",
        )
    if emo in ("sad", "fear", "angry", "disgust"):
        return AffectiveFeedback(
            key=emo,
            sv="Du verkar lite osaker. Det ar okej, vi tar det lugnt.",
            en="You seem a bit unsure. That's ok, we'll take it slow.",
            gesture="Thoughtful",
        )
    if emo == "neutral":
        return AffectiveFeedback(
            key=emo,
            sv="Okej. Vi fortsatter lugnt.",
            en="Ok. We'll continue calmly.",
            gesture="Nod",
        )
    return None


def should_emit_affective_feedback(
    *,
    state: AffectiveState,
    feedback: AffectiveFeedback,
    now_ts: float,
    cooldown_s: float,
) -> bool:
    if feedback.key == state.last_key and (now_ts - state.last_ts) < cooldown_s:
        return False
    return (now_ts - state.last_ts) >= cooldown_s or feedback.key != state.last_key


def _norm(text: str) -> str:
    text = (text or "").lower()
    keep = []
    for ch in text:
        if ch.isalnum() or ch.isspace():
            keep.append(ch)
        else:
            keep.append(" ")
    return " ".join("".join(keep).split())


def _tokens_from_expected(expected: str) -> list[str]:
    expected = expected.replace("____", " ")
    return [t for t in _norm(expected).split() if t]


def is_correct_answer(user_answer: str, expect: list[str] | None, expected_template: str) -> bool:
    if expect:
        expected_tokens = expect
    else:
        expected_tokens = _tokens_from_expected(expected_template)
    if not expected_tokens:
        return False
    norm_answer = _norm(user_answer)
    return all(tok in norm_answer for tok in expected_tokens)


def is_repeat_intent(text: str) -> bool:
    norm = _norm(text)
    return any(k in norm for k in ("repeat", "again", "one more", "repetera", "igen", "repeat that", "say again"))


def is_continue_intent(text: str) -> bool:
    norm = _norm(text)
    return any(
        k in norm
        for k in (
            "continue",
            "cotniue",
            "continuing",
            "next",
            "go on",
            "move on",
            "fortsatt",
            "fortsatta",
            "nasta",
            "yes",
            "ok",
        )
    )


def is_not_understood_intent(text: str) -> bool:
    norm = _norm(text)
    return any(
        k in norm
        for k in ("no", "not", "didnt understand", "dont understand", "not sure", "i dont get", "forstar inte")
    )


def is_yes_intent(text: str) -> bool:
    norm = _norm(text)
    return any(k in norm for k in ("yes", "yeah", "yep", "sure", "correct", "right", "ja"))


def is_no_intent(text: str) -> bool:
    norm = _norm(text)
    return any(k in norm for k in ("no", "nope", "nah", "not", "incorrect", "wrong", "nej"))


def is_stop_intent(text: str) -> bool:
    norm = _norm(text)
    return any(k in norm for k in ("stop", "finish", "done", "enough", "sluta", "stopp", "avsluta"))


def is_review_intent(text: str) -> bool:
    norm = _norm(text)
    return any(
        k in norm
        for k in (
            "review",
            "review them",
            "to review",
            "revise",
            "repeat",
            "again",
            "go over",
            "recap",
            "repetera",
            "view them",
            "to view",
        )
    )


def parse_line_count(text: str, default: int) -> int:
    norm = _norm(text)
    for token in norm.split():
        if token.isdigit():
            val = int(token)
            if 1 <= val <= 10:
                return val
        if token == "f":
            return 4
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "for": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    for word, val in words.items():
        if word in norm:
            return val
    return default


def is_advance_intent(text: str) -> bool:
    norm = _norm(text)
    return any(k in norm for k in ("move on", "next", "forward", "advance", "ga vidare", "nasta"))


def is_doubt_text(text: str) -> bool:
    norm = _norm(text)
    return any(
        k in norm
        for k in (
            "how do i",
            "how to",
            "how do you say",
            "how do u say",
            "how do you",
            "what does",
            "what is",
            "what is the word for",
            "can you",
            "please explain",
            "translate",
            "meaning of",
            "help me",
        )
    )


def extract_topic(text: str) -> str:
    norm = _norm(text)
    # Drop common greetings/fillers at the start.
    for filler in ("hey", "hi", "hello", "uh", "um", "er", "like"):
        if norm.startswith(filler + " "):
            norm = norm[len(filler) + 1 :].strip()
            break
    for prefix in (
        "jag vill prata om",
        "i want to talk about",
        "i want to learn about",
        "i want to learn",
        "learn about",
        "lets talk about",
        "talk about",
    ):
        if norm.startswith(prefix):
            norm = norm.replace(prefix, "").strip()
            break
    # Common ASR confusion: whether -> weather.
    norm = norm.replace("whether", "weather")
    return norm.strip() or "something"


def is_topic_change(text: str) -> bool:
    norm = _norm(text)
    return any(k in norm for k in ("change topic", "new topic", "another topic", "switch topic"))


def is_misheard_intent(text: str) -> bool:
    norm = _norm(text)
    return any(
        k in norm
        for k in (
            "not my sentence",
            "not the line",
            "not what i said",
            "that is not what i said",
            "you heard wrong",
            "misheard",
            "wrong line",
        )
    )


def make_topic_system_prompt() -> str:
    return (
        "You are a Swedish tutor. Return a JSON object with keys: "
        "line_sv, line_en, breakdown, pronunciation, notes, clarification_sv, clarification_en, support_sv, support_en, challenge_sv, challenge_en, done. "
        "line_sv: Swedish translation of user_line (one sentence). "
        "line_en: English translation of line_sv (should match user_line meaning). "
        "breakdown: list of objects with keys sv and en (word-by-word or short phrase-by-phrase meaning). "
        "pronunciation: short English-friendly pronunciation guide for the Swedish line. "
        "notes: short usage note (optional). "
        "clarification_sv: Swedish word/phrase that answers the user_question (if applicable). "
        "clarification_en: English answer for the user_question (brief). "
        "support_sv: extra simple Swedish example for low affect (optional). "
        "support_en: English translation of support_sv (optional). "
        "challenge_sv: slightly harder follow-up Swedish line (only when affect_bucket is high). "
        "challenge_en: English translation of challenge_sv. "
        "done: always false for translation mode. "
        "Keep lines short, practical, and beginner-friendly unless level is advanced. "
        "Never ask the learner to repeat after you. "
        "If affect_bucket is low, keep it simpler and include more breakdown. "
        "If affect_bucket is high, keep it concise and add a challenge_sv/en."
    )


def make_topic_user_prompt(
    *,
    topic: str,
    level: str,
    bucket: str,
    user_line: str,
    user_question: str,
    doubt: bool,
) -> str:
    return (
        f"topic={topic}\n"
        f"level={level}\n"
        f"affect_bucket={bucket}\n"
        f"user_line={user_line}\n"
        f"user_question={user_question}\n"
        f"doubt={doubt}\n"
        "Return JSON only."
    )


def make_topic_intro_system_prompt() -> str:
    return (
        "You are a Swedish tutor. Return a JSON object with keys: sv, en, pronunciation, notes. "
        "sv: Swedish translation of the topic (short phrase). "
        "en: English gloss for the Swedish phrase. "
        "pronunciation: short English-friendly pronunciation guide. "
        "notes: short usage note (optional). "
        "Keep it brief."
    )


def make_topic_intro_user_prompt(*, topic: str) -> str:
    return f"topic={topic}\nReturn JSON only."


def decide_strategy(
    *,
    correct: bool,
    bucket: str,
    latency_s: float,
    slow_seconds: float,
    confidence: float,
    min_confidence: float,
) -> str:
    if not correct:
        return "repeat"
    if bucket == "low":
        return "repeat"
    if confidence < min_confidence:
        return "repeat"
    if latency_s > slow_seconds:
        return "repeat"
    return "advance"


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
        log_emotions=args.log_emotions,
        debug=args.debug,
    )
    perception.start()
    if not perception.wait_ready(timeout_s=10.0):
        raise RuntimeError("Perception did not start (webcam).")

    async with FurhatClient(uri=args.uri, key=args.key, debug=args.debug) as furhat:
        async def speak_sv_en(sv: str, en: str, *, gesture: Optional[str] = None) -> None:
            if sv:
                print(f"[tutor][sv] {sv}")
                await furhat.speak(sv, gesture=gesture, wait_end=args.wait_speak_end)
            if en:
                print(f"[tutor][en] {en}")
                await furhat.speak(en, gesture="Nod", wait_end=args.wait_speak_end)

        async def listen_once(prompt: str) -> Tuple[str, float]:
            if not args.use_asr:
                try:
                    text = await asyncio.to_thread(input, prompt)
                except (EOFError, KeyboardInterrupt):
                    return "", 0.0
                text = (text or "").strip()
                if text:
                    print(f"[user][text] {text}")
                return text, 0.0

            start_t = _now()
            parts: list[str] = []
            total_listen_s = 0.0
            for attempt in range(max(1, args.asr_max_extends + 1)):
                text = (
                    await furhat.listen_text(
                        timeout_s=args.listen_timeout if attempt == 0 else args.asr_extend_seconds,
                        request_type=args.asr_request_type,
                        stop_type=args.asr_stop_type,
                        accept_any_type=not args.asr_strict_types,
                        dump_messages=args.asr_dump,
                        debug=args.debug,
                    )
                ) or ""
                total_listen_s = max(0.0, _now() - start_t)
                text = text.strip()
                if text:
                    parts.append(text)
                    joined = " ".join(parts).strip()
                    if args.asr_accept_early and len(joined.split()) >= args.asr_min_words:
                        if attempt > 0:
                            await asyncio.sleep(args.asr_pause_before_accept)
                        print(f"[user][asr] {joined}")
                        return joined, total_listen_s
                if attempt < args.asr_max_extends:
                    continue
                break
            joined = " ".join(parts).strip()
            if joined:
                print(f"[user][asr] {joined}")
            return joined, total_listen_s

        await speak_sv_en(
            "Hej! Jag ar din svenska larare.",
            "Hi! I am your Swedish tutor.",
            gesture="Nod",
        )

        topic = ""
        level_index = 0
        affect_state = AffectiveState()
        levels = ["A1", "A2", "B1", "B2", "C1"]
        line_index = 0
        last_step: Optional[dict] = None
        last_user_line = ""
        batch_entries: list[dict] = []
        repeats = 0
        understood_lines = 0
        topic_start_ts = _now()
        bucket_history: list[str] = []
        latency_history: list[float] = []
        topic_prompt = make_topic_system_prompt()

        await speak_sv_en(
            "Vad vill du lara dig idag?",
            "What do you want to learn today? You can answer in English.",
            gesture="Nod",
        )
        topic_answer, _ = await listen_once("You: ")
        topic = extract_topic(topic_answer)
        if not topic:
            topic = "everyday Swedish"
        needs_confirm = "whether" in _norm(topic_answer) or topic == "something"
        if needs_confirm:
            await speak_sv_en(
                f"Menar du amnet {topic}?",
                f"Did you mean the topic {topic}?",
                gesture="Nod",
            )
            confirm, _ = await listen_once("You: ")
            if is_no_intent(confirm):
                await speak_sv_en(
                    "Okej, sag amnet igen.",
                    "Ok, please say the topic again.",
                    gesture="Nod",
                )
                topic_answer, _ = await listen_once("You: ")
                topic = extract_topic(topic_answer)
            elif not is_yes_intent(confirm):
                await speak_sv_en(
                    "Jag fortsatter med det amnet jag horde.",
                    "I will continue with the topic I heard.",
                    gesture="Nod",
                )
            else:
                refined = extract_topic(confirm)
                if refined and refined != "something":
                    topic = refined
        if topic == "something":
            await speak_sv_en(
                "Jag horde inte amnet. Skriv amnet igen.",
                "I did not catch the topic. Please say it again.",
                gesture="Nod",
            )
            topic_answer, _ = await listen_once("You: ")
            topic = extract_topic(topic_answer) or "everyday Swedish"
        if args.use_openai and api_key:
            intro = openai_generate(
                api_key=api_key,
                model=args.openai_model,
                system_prompt=make_topic_intro_system_prompt(),
                user_prompt=make_topic_intro_user_prompt(topic=topic),
                timeout_s=args.openai_timeout,
                debug=args.debug,
            )
            if isinstance(intro, dict):
                intro_sv = (intro.get("sv") or "").strip()
                intro_en = (intro.get("en") or "").strip()
                intro_pron = (intro.get("pronunciation") or "").strip()
                intro_notes = (intro.get("notes") or "").strip()
                if intro_sv or intro_en:
                    await speak_sv_en(intro_sv, intro_en, gesture="Nod")
                if intro_pron:
                    await speak_sv_en("", f"Pronunciation: {intro_pron}", gesture="Nod")
                if intro_notes:
                    await speak_sv_en("", f"Note: {intro_notes}", gesture="Nod")

        async def speak_step(step: dict, bucket: str) -> None:
            line_sv = (step.get("line_sv") or "").strip()
            line_en = (step.get("line_en") or "").strip()
            pronunciation = (step.get("pronunciation") or "").strip()
            notes = (step.get("notes") or "").strip()
            support_sv = (step.get("support_sv") or "").strip()
            support_en = (step.get("support_en") or "").strip()
            breakdown = step.get("breakdown")
            if not isinstance(breakdown, list):
                breakdown = []

            if line_sv or line_en:
                await speak_sv_en(line_sv, line_en, gesture="Nod")
            if breakdown:
                for part in breakdown:
                    sv_word = str(part.get("sv") or "").strip()
                    en_word = str(part.get("en") or "").strip()
                    if sv_word and en_word:
                        await speak_sv_en("", f"{sv_word} means {en_word}.", gesture="Nod")
            if pronunciation:
                await speak_sv_en("", f"Pronunciation: {pronunciation}", gesture="Nod")
            if notes:
                await speak_sv_en("", f"Note: {notes}", gesture="Nod")
            if bucket == "low" and (support_sv or support_en):
                await speak_sv_en(support_sv, support_en, gesture="Thoughtful")

        async def sample_affect(window_s: float) -> Tuple[str, Optional[str], float]:
            after_ts = _now() - max(0.1, window_s)
            events = perception.snapshot_since(after_ts)
            bucket, top_emotion = build_bucket_summary(events, min_majority=args.min_majority)
            conf = average_confidence(events)
            if not events:
                latest = perception.latest()
                if latest:
                    bucket = latest.bucket
                    top_emotion = latest.top_emotion
                    conf = latest.confidence
            return bucket, top_emotion, conf

        while True:
            if line_index >= args.topic_max_turns:
                break

            level = levels[min(level_index, len(levels) - 1)]
            bucket, top_emotion, conf = await sample_affect(args.affect_window_seconds)
            bucket_history.append(bucket)

            user_question = ""
            doubt = False

            if last_step is None:
                await speak_sv_en(
                    f"Skriv rad {line_index + 1}.",
                    f"Please say line {line_index + 1}.",
                    gesture="Nod",
                )
                user_line, latency_s = await listen_once("You: ")
                latency_history.append(latency_s)
                if not user_line:
                    await speak_sv_en("Jag horde inget. Forsok igen.", "I did not catch that. Please try again.", gesture="Thoughtful")
                    continue
                if is_misheard_intent(user_line):
                    await speak_sv_en(
                        "Okej, sag raden igen.",
                        "Ok, please say the line again.",
                        gesture="Thoughtful",
                    )
                    continue
                if is_stop_intent(user_line):
                    break
                if is_topic_change(user_line):
                    topic = extract_topic(user_line)
                    line_index = 0
                    repeats = 0
                    understood_lines = 0
                    topic_start_ts = _now()
                    bucket_history.clear()
                    latency_history.clear()
                    last_step = None
                    await speak_sv_en(f"Okej! Nytt amne: {topic}.", f"Ok! New topic: {topic}.", gesture="Nod")
                    continue

                doubt = is_doubt_text(user_line) or user_line.strip().endswith("?")
                if doubt:
                    user_question = user_line
                    if args.use_openai and api_key:
                        step = openai_generate(
                            api_key=api_key,
                            model=args.openai_model,
                            system_prompt=topic_prompt,
                            user_prompt=make_topic_user_prompt(
                                topic=topic,
                                level=level,
                                bucket=bucket,
                                user_line="",
                                user_question=user_question,
                                doubt=True,
                            ),
                            timeout_s=args.openai_timeout,
                            debug=args.debug,
                        )
                        if isinstance(step, dict):
                            clarification_sv = (step.get("clarification_sv") or "").strip()
                            clarification_en = (step.get("clarification_en") or "").strip()
                            if clarification_sv:
                                await speak_sv_en(clarification_sv, clarification_en, gesture="Nod")
                            elif clarification_en:
                                await speak_sv_en("", clarification_en, gesture="Nod")
                            if clarification_en:
                                pass
                    repeats += 1
                    await speak_sv_en(
                        "Skriv raden igen.",
                        "Please say the line again.",
                        gesture="Nod",
                    )
                    continue

                last_user_line = user_line
                step = None
                if args.use_openai and api_key:
                    step = openai_generate(
                        api_key=api_key,
                        model=args.openai_model,
                        system_prompt=topic_prompt,
                        user_prompt=make_topic_user_prompt(
                            topic=topic,
                            level=level,
                            bucket=bucket,
                            user_line=last_user_line,
                            user_question="",
                            doubt=False,
                        ),
                        timeout_s=args.openai_timeout,
                        debug=args.debug,
                    )

                if not isinstance(step, dict):
                    await speak_sv_en(
                        "Jag kan inte oversatta just nu.",
                        "I cannot translate right now.",
                        gesture="Thoughtful",
                    )
                    repeats += 1
                    continue

                last_step = step

            if not last_step:
                await speak_sv_en("Jag hittar inget att undervisa.", "I could not build the lesson.", gesture="Thoughtful")
                break

            clarification_en = (last_step.get("clarification_en") or "").strip()
            await speak_step(last_step, bucket)
            if clarification_en:
                await speak_sv_en("", clarification_en, gesture="Nod")

            if bucket == "high":
                challenge_sv = (last_step.get("challenge_sv") or "").strip()
                challenge_en = (last_step.get("challenge_en") or "").strip()
                if challenge_sv or challenge_en:
                    await speak_sv_en(
                        f"Utmaning: {challenge_sv}",
                        f"Challenge: {challenge_en or 'Say the Swedish line.'}",
                        gesture="BigSmile",
                    )
                    await speak_sv_en(
                        f"Svara: {challenge_sv}",
                        "Please say the challenge line.",
                        gesture="Nod",
                    )
                    await listen_once("You: ")

            await speak_sv_en(
                "Forstod du den raden? Ska jag fortsatta eller repetera?",
                "Did you understand that line? Should I continue or repeat?",
                gesture="Nod",
            )
            user_reply, latency_s = await listen_once("You: ")
            latency_history.append(latency_s)

            if not user_reply:
                await speak_sv_en(
                    "Jag horde inget. Vill du att jag fortsatter till nasta rad eller repetera?",
                    "I did not catch that. Do you want me to continue to the next line or repeat?",
                    gesture="Thoughtful",
                )
                decision, _ = await listen_once("You: ")
                if is_continue_intent(decision):
                    understood_lines += 1
                    line_index += 1
                    if last_step and last_user_line:
                        batch_entries.append({"user_line": last_user_line, "step": last_step})
                    last_step = None
                    continue
                if is_repeat_intent(decision) or is_not_understood_intent(decision):
                    await speak_sv_en("Okej, vi tar den igen.", "Ok, we will repeat it.", gesture="Thoughtful")
                    continue
                if is_stop_intent(decision):
                    break
                continue

            if is_misheard_intent(user_reply):
                await speak_sv_en(
                    "Okej, sag raden igen.",
                    "Ok, please say the line again.",
                    gesture="Thoughtful",
                )
                last_step = None
                continue

            if is_topic_change(user_reply):
                topic = extract_topic(user_reply)
                last_step = None
                line_index = 0
                repeats = 0
                understood_lines = 0
                topic_start_ts = _now()
                bucket_history.clear()
                latency_history.clear()
                await speak_sv_en(f"Okej! Nytt amne: {topic}.", f"Ok! New topic: {topic}.", gesture="Nod")
                continue

            doubt = is_doubt_text(user_reply)
            if doubt:
                user_question = user_reply
                step = None
                if args.use_openai and api_key:
                    step = openai_generate(
                        api_key=api_key,
                        model=args.openai_model,
                        system_prompt=topic_prompt,
                        user_prompt=make_topic_user_prompt(
                            topic=topic,
                            level=level,
                            bucket=bucket,
                            user_line=last_user_line,
                            user_question=user_question,
                            doubt=True,
                        ),
                        timeout_s=args.openai_timeout,
                        debug=args.debug,
                    )
                if isinstance(step, dict):
                    clarification_sv = (step.get("clarification_sv") or "").strip()
                    clarification_en = (step.get("clarification_en") or "").strip()
                    if clarification_sv:
                        await speak_sv_en(clarification_sv, clarification_en, gesture="Nod")
                    elif clarification_en:
                        await speak_sv_en("", clarification_en, gesture="Nod")
                repeats += 1
                await speak_sv_en(
                    "Vill du att jag fortsatter till nasta rad eller repetera?",
                    "Do you want me to continue to the next line or repeat?",
                    gesture="Nod",
                )
                decision, _ = await listen_once("You: ")
                if is_continue_intent(decision):
                    understood_lines += 1
                    line_index += 1
                    if last_step and last_user_line:
                        batch_entries.append({"user_line": last_user_line, "step": last_step})
                    last_step = None
                    continue
                if is_repeat_intent(decision) or is_not_understood_intent(decision):
                    await speak_sv_en("Okej, vi tar den igen.", "Ok, we will repeat it.", gesture="Thoughtful")
                    continue
                if is_stop_intent(decision):
                    break
                continue

            bucket, top_emotion, conf = await sample_affect(args.affect_window_seconds)
            affect = select_affective_feedback(
                top_emotion=top_emotion,
                confidence=conf,
                min_confidence=args.affect_min_confidence,
            )
            if affect and should_emit_affective_feedback(
                state=affect_state,
                feedback=affect,
                now_ts=_now(),
                cooldown_s=args.affect_cooldown_seconds,
            ):
                await speak_sv_en(affect.sv, affect.en, gesture=affect.gesture)
                affect_state.last_key = affect.key
                affect_state.last_ts = _now()

            if is_repeat_intent(user_reply) or is_not_understood_intent(user_reply):
                repeats += 1
                await speak_sv_en("Okej, vi tar den igen.", "Ok, we will repeat it.", gesture="Thoughtful")
                continue

            if is_continue_intent(user_reply):
                understood_lines += 1
                line_index += 1
                if last_step and last_user_line:
                    batch_entries.append({"user_line": last_user_line, "step": last_step})
                last_step = None
                if len(batch_entries) > 0 and len(batch_entries) % args.review_batch_size == 0:
                    await speak_sv_en(
                        "Vi har gatt igenom fem rader. Vill du stoppa, repetera dem, eller fortsatta?",
                        "We covered five lines. Do you want to stop, review them, or continue?",
                        gesture="Nod",
                    )
                    decision, _ = await listen_once("You: ")
                    if is_yes_intent(decision) or is_stop_intent(decision):
                        break
                    if is_review_intent(decision):
                        for entry in batch_entries[-args.review_batch_size :]:
                            await speak_step(entry["step"], "medium")
                        await speak_sv_en(
                            "Vill du fortsatta med fler rader?",
                            "Do you want to continue with more lines?",
                            gesture="Nod",
                        )
                        follow_up, _ = await listen_once("You: ")
                        if is_yes_intent(follow_up):
                            break
                if understood_lines and understood_lines % args.level_up_streak == 0 and level_index < len(levels) - 1:
                    level_index += 1
                    await speak_sv_en("", f"Great work. Moving to level {levels[level_index]}.", gesture="BigSmile")
                continue
            # Any other reply: treat as a question about the line and answer it.
            user_question = user_reply
            step = None
            if args.use_openai and api_key:
                step = openai_generate(
                    api_key=api_key,
                    model=args.openai_model,
                    system_prompt=topic_prompt,
                    user_prompt=make_topic_user_prompt(
                        topic=topic,
                        level=level,
                        bucket=bucket,
                        user_line=last_user_line,
                        user_question=user_question,
                        doubt=True,
                    ),
                    timeout_s=args.openai_timeout,
                    debug=args.debug,
                )
            if isinstance(step, dict):
                clarification_en = (step.get("clarification_en") or "").strip()
                if clarification_en:
                    await speak_sv_en("", clarification_en, gesture="Nod")
            repeats += 1
            await speak_sv_en(
                "Vill du att jag fortsatter till nasta rad eller repetera?",
                "Do you want me to continue to the next line or repeat?",
                gesture="Nod",
            )
            decision, _ = await listen_once("You: ")
            if is_continue_intent(decision):
                understood_lines += 1
                line_index += 1
                if last_step and last_user_line:
                    batch_entries.append({"user_line": last_user_line, "step": last_step})
                last_step = None
                if len(batch_entries) > 0 and len(batch_entries) % args.review_batch_size == 0:
                    await speak_sv_en(
                        "Vi har gatt igenom fem rader. Vill du stoppa, repetera dem, eller fortsatta?",
                        "We covered five lines. Do you want to stop, review them, or continue?",
                        gesture="Nod",
                    )
                    decision2, _ = await listen_once("You: ")
                    if not decision2:
                        await speak_sv_en(
                            "Jag horde inget. Vill du stoppa, repetera dem, eller fortsatta?",
                            "I did not catch that. Do you want to stop, review them, or continue?",
                            gesture="Nod",
                        )
                        decision2, _ = await listen_once("You: ")
                    if is_yes_intent(decision2) or is_stop_intent(decision2):
                        break
                    if is_review_intent(decision2):
                        for entry in batch_entries[-args.review_batch_size :]:
                            line_sv = (entry["step"].get("line_sv") or "").strip()
                            if line_sv:
                                await speak_sv_en(line_sv, "", gesture="Nod")
                        await speak_sv_en(
                            "Vill du fortsatta med fler rader?",
                            "Do you want to continue with more lines?",
                            gesture="Nod",
                        )
                        follow_up, _ = await listen_once("You: ")
                        if is_yes_intent(follow_up):
                            break
                    elif decision2:
                        await speak_sv_en(
                            "Jag fortsatter med fler rader.",
                            "I will continue with more lines.",
                            gesture="Nod",
                        )
                if understood_lines and understood_lines % args.level_up_streak == 0 and level_index < len(levels) - 1:
                    level_index += 1
                    await speak_sv_en("", f"Great work. Moving to level {levels[level_index]}.", gesture="BigSmile")
                continue
            if is_repeat_intent(decision) or is_not_understood_intent(decision):
                await speak_sv_en("Okej, vi tar den igen.", "Ok, we will repeat it.", gesture="Thoughtful")
                continue
            if is_stop_intent(decision):
                break
            continue

        duration_s = max(1.0, _now() - topic_start_ts)
        avg_line_s = duration_s / max(1, understood_lines + max(0, line_index - understood_lines))
        bucket_counts = Counter(bucket_history)
        top_bucket = bucket_counts.most_common(1)[0][0] if bucket_counts else "medium"

        speed_note = "steady"
        if avg_line_s <= args.fast_line_seconds:
            speed_note = "fast"
        elif avg_line_s >= args.slow_line_seconds:
            speed_note = "slow"

        if top_bucket == "high":
            affect_msg = "You seemed engaged and positive."
        elif top_bucket == "low":
            affect_msg = "I noticed some uncertainty. That is normal when learning."
        else:
            affect_msg = "You stayed calm and focused."

        if speed_note == "fast":
            speed_msg = "You moved quickly through the topic."
        elif speed_note == "slow":
            speed_msg = "You took your time, which is good for accuracy."
        else:
            speed_msg = "Your pace was steady."

        repeat_msg = ""
        if repeats:
            repeat_msg = f"You asked for {repeats} repeats."

        await speak_sv_en(
            "Tack! Vi ar klara for idag.",
            "Thanks! We are done for today.",
            gesture="Nod",
        )
        await speak_sv_en(
            "",
            f"{affect_msg} {speed_msg} {repeat_msg}".strip(),
            gesture="Nod",
        )

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
    p.add_argument("--log-emotions", action="store_true", help="Log emotion probabilities to stderr.")

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
    p.add_argument(
        "--post-answer-observe-seconds",
        type=float,
        default=1.2,
        help="Seconds to observe after the user answer.",
    )
    p.add_argument("--min-majority", type=float, default=0.6, help="Majority needed in observation window.")
    p.add_argument("--slow-seconds", type=float, default=4.0, help="Latency threshold (seconds) to repeat.")
    p.add_argument("--min-confidence", type=float, default=0.5, help="Min affect confidence to advance.")
    p.add_argument("--affect-min-confidence", type=float, default=0.55, help="Min emotion confidence for affect feedback.")
    p.add_argument(
        "--affect-cooldown-seconds",
        type=float,
        default=6.0,
        help="Cooldown between affective feedback messages.",
    )
    p.add_argument(
        "--affect-window-seconds",
        type=float,
        default=2.0,
        help="Seconds of recent emotion history to summarize.",
    )
    p.add_argument("--fast-line-seconds", type=float, default=20.0, help="Avg seconds per line to count as fast.")
    p.add_argument("--slow-line-seconds", type=float, default=45.0, help="Avg seconds per line to count as slow.")
    p.add_argument("--level-up-streak", type=int, default=3, help="Correct streak to advance level in topic mode.")
    p.add_argument("--topic-max-turns", type=int, default=10, help="Max topic practice turns before ending.")
    p.add_argument("--review-batch-size", type=int, default=5, help="Lines per review batch.")

    p.add_argument("--wait-speak-end", action="store_true", help="Wait for response.speak.end between messages.")

    p.add_argument("--use-asr", action="store_true", help="Use Furhat speech recognition instead of typed answers.")
    p.add_argument("--listen-timeout", type=float, default=8.0, help="Seconds to wait for each spoken answer.")
    p.add_argument("--asr-min-words", type=int, default=3, help="Minimum words before accepting ASR result.")
    p.add_argument("--asr-extend-seconds", type=float, default=6.0, help="Extra listen time to capture longer answers.")
    p.add_argument("--asr-max-extends", type=int, default=2, help="How many extra listen cycles to attempt.")
    p.add_argument(
        "--asr-pause-before-accept",
        type=float,
        default=0.6,
        help="Short pause before accepting extended ASR result (seconds).",
    )
    p.add_argument(
        "--asr-accept-early",
        action="store_true",
        help="Accept ASR as soon as min words are reached (default).",
    )
    p.add_argument(
        "--no-asr-accept-early",
        dest="asr_accept_early",
        action="store_false",
        help="Wait full listen window (do not accept early).",
    )
    p.set_defaults(asr_accept_early=True)
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
