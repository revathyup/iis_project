"""Map perception bucket events to Furhat via the Realtime API (virtual Furhat)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import Dict, Optional

import websockets

# Adjust texts/gestures to your scenario. Gestures depend on your Furhat build.
DEFAULT_ACTIONS: Dict[str, Dict[str, Optional[str]]] = {
    "low": {"text": "Jag saktar ner och ger en ledtråd.", "gesture": "Thoughtful"},
    "medium": {"text": "Fortsätt, det går bra.", "gesture": "Nod"},
    "high": {"text": "Snyggt! Dags för en utmaning.", "gesture": "BigSmile"},
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
    """Realtime API speak message per docs: type=request.speak.text with text field."""
    return {"type": "request.speak.text", "text": text}


def build_gesture_message(name: str) -> Dict[str, object]:
    """Realtime API gesture message: type=request.gesture.start with name field."""
    return {"type": "request.gesture.start", "name": name}


async def run(uri: str, key: Optional[str], cooldown: float) -> None:
    async with websockets.connect(uri) as ws:
        # Send auth if key provided
        if key:
            await ws.send(json.dumps({"type": "request.auth", "key": key}))

        last_bucket: Optional[str] = None
        last_time: float = 0.0

        for line in sys.stdin:
            evt = parse_event(line)
            if evt is None:
                continue
            bucket = str(evt.get("bucket", "medium"))
            now = time.time()
            if bucket == last_bucket and (now - last_time) < cooldown:
                continue  # avoid spamming repeated actions

            action = DEFAULT_ACTIONS.get(bucket, DEFAULT_ACTIONS["medium"])
            text = action.get("text")
            gesture = action.get("gesture")

            if text:
                await ws.send(json.dumps(build_say_message(text=text)))
            if gesture:
                await ws.send(json.dumps(build_gesture_message(name=gesture)))

            last_bucket = bucket
            last_time = now


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Consume bucket events and drive Furhat via Realtime API.")
    parser.add_argument(
        "--uri",
        required=True,
        help="Realtime API WebSocket URI, e.g., ws://localhost:9000/v1/events (check simulator settings).",
    )
    parser.add_argument("--key", default=None, help="Optional auth key if your API requires it.")
    parser.add_argument("--cooldown", type=float, default=1.5, help="Minimum seconds between repeated bucket actions.")
    args = parser.parse_args(argv)

    asyncio.run(run(uri=args.uri, key=args.key, cooldown=args.cooldown))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
