"""Map perception bucket events to Furhat via the Realtime API (Virtual Furhat).

Reads JSON lines from stdin (produced by `python -m perception.stream`) and drives a
simple rule-based Swedish tutor in the Furhat Simulator using the Furhat Realtime API.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from typing import Dict, Optional, Tuple

import websockets
from websockets.exceptions import ConnectionClosed


# ASCII-only to avoid Windows/terminal encoding issues that can make Furhat TTS go silent.
LESSON = [
    {"sv": "Vad heter du?", "en": "What is your name?", "answer": "Jag heter ____."},
    {"sv": "Hur mar du?", "en": "How are you?", "answer": "Jag mar bra. / Jag mar inte sa bra."},
    {"sv": "Var bor du?", "en": "Where do you live?", "answer": "Jag bor i ____."},
    {"sv": "Var kommer du ifran?", "en": "Where do you come from?", "answer": "Jag kommer fran ____."},
    {"sv": "Vad ar klockan?", "en": "What time is it?", "answer": "Klockan ar ____."},
]


DEFAULT_GESTURES: Dict[str, Optional[str]] = {
    "low": "Thoughtful",
    "medium": "Nod",
    "high": "BigSmile",
}


def parse_event(line: str) -> Optional[Dict[str, object]]:
    try:
        evt = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(evt, dict):
        return None
    return evt


def build_say_message(text: str) -> Dict[str, object]:
    return {"type": "request.speak.text", "text": text}


def build_gesture_message(name: str) -> Dict[str, object]:
    return {"type": "request.gesture.start", "name": name}


def build_tutor_text(bucket: str, lesson_index: int, top_emotion: Optional[str]) -> Tuple[str, int]:
    item = LESSON[lesson_index % len(LESSON)]
    answer_clean = str(item["answer"]).rstrip().rstrip(".")

    emo_hint = f" (I see {top_emotion}.)" if top_emotion else ""

    if bucket == "low":
        text = (
            f"Ok, we take it slowly.{emo_hint} "
            f"Question: {item['sv']} (English: {item['en']}). "
            f"Answer like this: {answer_clean}. Say it out loud."
        )
        return text, lesson_index

    if bucket == "high":
        next_index = (lesson_index + 1) % len(LESSON)
        next_item = LESSON[next_index]
        next_answer = str(next_item["answer"]).rstrip().rstrip(".")
        text = (
            "Great! Lets continue. "
            f"Next question: {next_item['sv']} (English: {next_item['en']}). "
            f"Answer like this: {next_answer}."
        )
        return text, next_index

    text = f"Good. Question: {item['sv']} (English: {item['en']}). Answer like this: {answer_clean}."
    return text, lesson_index


def build_repeat_text(lesson_index: int, top_emotion: Optional[str]) -> str:
    item = LESSON[lesson_index % len(LESSON)]
    answer_clean = str(item["answer"]).rstrip().rstrip(".")
    emo_hint = f" (I see {top_emotion}.)" if top_emotion else ""
    return (
        f"Ok, lets repeat slowly.{emo_hint} "
        f"Question: {item['sv']} (English: {item['en']}). "
        f"Answer: {answer_clean}. Say it out loud."
    )


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


def _log(debug: bool, msg: str) -> None:
    if not debug:
        return
    sys.stderr.write(msg.rstrip() + "\n")
    sys.stderr.flush()


async def run(
    uri: str,
    key: Optional[str],
    cooldown: float,
    repeat_seconds: float,
    speak_on_change_only: bool,
    wait_speak_end: bool,
    wait_first_event: bool,
    observe_seconds: float,
    medium_delay_seconds: float,
    high_streak_seconds: float,
    settle_seconds: float,
    min_majority: float,
    debug: bool,
) -> None:
    _log(debug, f"[furhat] connecting to {uri}")
    async with websockets.connect(uri) as ws:
        if key:
            _log(debug, "[furhat] sending auth")
            await ws.send(json.dumps({"type": "request.auth", "key": key}))

        speaking = False
        pending: Optional[tuple[str, Optional[str]]] = None  # (bucket, top_emotion)

        last_bucket: Optional[str] = None
        last_time: float = 0.0
        lesson_index = 0

        speak_end_event = asyncio.Event()

        async def recv_loop() -> None:
            nonlocal speaking, pending, last_time, last_bucket, lesson_index
            try:
                while True:
                    raw = await ws.recv()
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    if not isinstance(msg, dict):
                        continue
                    t = msg.get("type")
                    if t == "response.speak.start":
                        speaking = True
                    elif t == "response.speak.end":
                        speaking = False
                        speak_end_event.set()
                        if wait_speak_end and pending is not None:
                            bucket, top_emotion_str = pending
                            pending = None
                            now = time.time()
                            if now - last_time < cooldown:
                                continue
                            text, lesson_index = build_tutor_text(
                                bucket=bucket, lesson_index=lesson_index, top_emotion=top_emotion_str
                            )
                            gesture = DEFAULT_GESTURES.get(bucket, DEFAULT_GESTURES["medium"])
                            _log(
                                debug,
                                f"[furhat] (after speak.end) bucket={bucket} lesson_index={lesson_index} text={text!r} gesture={gesture!r}",
                            )
                            await ws.send(json.dumps(build_say_message(text)))
                            if gesture:
                                await ws.send(json.dumps(build_gesture_message(gesture)))
                            last_bucket = bucket
                            last_time = now
            except ConnectionClosed:
                return

        recv_task = asyncio.create_task(recv_loop())

        first_text, lesson_index = build_tutor_text(bucket="medium", lesson_index=lesson_index, top_emotion=None)
        greeting = "Hi! I am your Swedish tutor. Lets practice a few simple questions. "

        async def read_stdin_line() -> str:
            return await asyncio.to_thread(sys.stdin.readline)

        event_q: asyncio.Queue[Dict[str, object]] = asyncio.Queue(maxsize=200)

        async def stdin_loop() -> None:
            while True:
                line = await read_stdin_line()
                if not line:
                    return
                evt = parse_event(line)
                if evt is None:
                    continue
                try:
                    event_q.put_nowait(evt)
                except asyncio.QueueFull:
                    try:
                        _ = event_q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        event_q.put_nowait(evt)
                    except asyncio.QueueFull:
                        pass

        stdin_task = asyncio.create_task(stdin_loop())

        async def speak(text: str, bucket: str) -> None:
            nonlocal last_time, last_bucket
            gesture = DEFAULT_GESTURES.get(bucket, DEFAULT_GESTURES["medium"])
            _log(debug, f"[furhat] SAY bucket={bucket} lesson_index={lesson_index} text={text!r} gesture={gesture!r}")
            speak_end_event.clear()
            await ws.send(json.dumps(build_say_message(text)))
            if gesture:
                await ws.send(json.dumps(build_gesture_message(gesture)))
            if wait_speak_end:
                try:
                    await asyncio.wait_for(speak_end_event.wait(), timeout=20.0)
                except asyncio.TimeoutError:
                    pass
            last_bucket = bucket
            last_time = time.time()
            # Brief settle helps reduce immediate flicker right after speaking.
            if settle_seconds > 0:
                await asyncio.sleep(settle_seconds)

        def drain_queue() -> None:
            while True:
                try:
                    _ = event_q.get_nowait()
                except asyncio.QueueEmpty:
                    return

        async def wait_for_first_event() -> Optional[Dict[str, object]]:
            _log(debug, "[furhat] waiting for first perception event...")
            try:
                evt = await asyncio.wait_for(event_q.get(), timeout=30.0)
            except asyncio.TimeoutError:
                return None
            return evt

        async def observe_window(after_timestamp: float) -> tuple[str, Optional[str], list[str]]:
            """Observe buckets for up to observe_seconds, using only events AFTER `after_timestamp` (epoch seconds)."""
            buckets: list[str] = []
            last_top_emotion: Optional[str] = None
            t_start = time.monotonic()
            high_run_start_ts: Optional[float] = None

            while True:
                elapsed = time.monotonic() - t_start
                if elapsed >= observe_seconds:
                    break
                try:
                    evt = await asyncio.wait_for(event_q.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                evt_ts = evt.get("timestamp")
                if isinstance(evt_ts, (int, float)) and float(evt_ts) < after_timestamp:
                    continue

                b = str(evt.get("bucket", "medium"))
                te = evt.get("top_emotion")
                if isinstance(te, str):
                    last_top_emotion = te
                buckets.append(b)

                if b == "high":
                    # High streak measured in event timestamps (more robust than arrival time).
                    evt_ts_f = float(evt_ts) if isinstance(evt_ts, (int, float)) else time.time()
                    if high_run_start_ts is None:
                        high_run_start_ts = evt_ts_f
                    elif evt_ts_f - high_run_start_ts >= high_streak_seconds:
                        return "high", last_top_emotion, buckets
                else:
                    high_run_start_ts = None

            return pick_bucket_majority(buckets, min_majority=min_majority), last_top_emotion, buckets

        # Optional: wait for perception so Furhat doesn't start before the camera appears.
        if wait_first_event:
            _ = await wait_for_first_event()

        # Start: greeting + first question.
        drain_queue()
        await speak(greeting + first_text, bucket="medium")

        while True:
            # After each question, observe for a short window and decide what to do.
            after_ts = time.time()
            drain_queue()
            bucket, top_emotion, _seen = await observe_window(after_timestamp=after_ts)
            _log(debug, f"[furhat] observe result bucket={bucket} top_emotion={top_emotion!r}")

            if bucket == "high":
                lesson_index = (lesson_index + 1) % len(LESSON)
                text, _ = build_tutor_text(bucket="high", lesson_index=lesson_index - 1, top_emotion=top_emotion)
                await speak(text, bucket="high")
                continue

            if bucket == "medium":
                # Break 3 seconds, then continue to next question.
                await asyncio.sleep(medium_delay_seconds)
                lesson_index = (lesson_index + 1) % len(LESSON)
                text, _ = build_tutor_text(bucket="medium", lesson_index=lesson_index, top_emotion=top_emotion)
                await speak(text, bucket="medium")
                continue

            # low: repeat same question + answer (help).
            text = build_repeat_text(lesson_index=lesson_index, top_emotion=top_emotion)
            await speak(text, bucket="low")

        stdin_task.cancel()
        recv_task.cancel()


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Consume bucket events and drive Furhat via Realtime API.")
    parser.add_argument("--uri", required=True, help="Realtime API WS URI (e.g. ws://localhost:9000/v1/events).")
    parser.add_argument("--key", default=None, help="Optional auth key if your API requires it.")
    parser.add_argument("--cooldown", type=float, default=0.8, help="Minimum seconds between any actions.")
    parser.add_argument("--repeat-seconds", type=float, default=6.0, help="Repeat same bucket after this many seconds.")
    parser.add_argument("--speak-on-change-only", action="store_true", help="Only speak when bucket changes.")
    parser.add_argument(
        "--wait-speak-end",
        action="store_true",
        help="Wait for `response.speak.end` before speaking again (prevents self-interruption).",
    )
    parser.add_argument(
        "--no-wait-first-event",
        action="store_true",
        help="Do not wait for the first perception event before the initial greeting/question.",
    )
    parser.add_argument("--observe-seconds", type=float, default=2.0, help="Seconds to observe emotion after each question.")
    parser.add_argument("--medium-delay-seconds", type=float, default=3.0, help="Seconds to pause before advancing on 'medium'.")
    parser.add_argument(
        "--high-streak-seconds",
        type=float,
        default=0.5,
        help="If 'high' persists this long during observation, advance early.",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.2,
        help="Small delay after each speak before starting observation (reduces flicker).",
    )
    parser.add_argument(
        "--min-majority",
        type=float,
        default=0.6,
        help="Minimum majority fraction during observation; otherwise treat as 'medium'.",
    )
    parser.add_argument("--debug", action="store_true", help="Print debug logs to stderr.")
    args = parser.parse_args(argv)

    asyncio.run(
        run(
            uri=args.uri,
            key=args.key,
            cooldown=args.cooldown,
            repeat_seconds=args.repeat_seconds,
            speak_on_change_only=args.speak_on_change_only,
            wait_speak_end=args.wait_speak_end,
            wait_first_event=not args.no_wait_first_event,
            observe_seconds=args.observe_seconds,
            medium_delay_seconds=args.medium_delay_seconds,
            high_streak_seconds=args.high_streak_seconds,
            settle_seconds=args.settle_seconds,
            min_majority=args.min_majority,
            debug=args.debug,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
