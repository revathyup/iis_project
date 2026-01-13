# Swedish Tutor With Affective Feedback

This project runs an interactive Swedish tutor that listens to your speech, translates your lines into Swedish, explains each word, and adapts feedback based on detected facial emotion buckets. It works with the Furhat Realtime API, a webcam emotion pipeline, and OpenAI for translations and explanations.

## What It Does

- You choose a topic (e.g., "basic phrases", "friends", "numbers").
- The tutor asks you for line 1, line 2, and so on (your English lines).
- For each line, it gives:
  - Swedish translation
  - Word-by-word meaning
  - Pronunciation hint
  - Short usage note (when needed)
- It asks if you want to continue or repeat after each line.
- After 5 lines, it asks whether to stop, review the last 5 lines, or continue.
- It uses facial emotion buckets (low/medium/high) to add supportive feedback and pacing.
- If you say "not my sentence" or "you heard wrong", it will ask you to repeat the line.
- If you ask a doubt like "how do you say today?", it answers the doubt, then asks to continue or repeat.
- It logs tutor TTS and ASR text to the terminal so you can follow the session.

## Requirements

- Python 3.11 (the project uses a virtual environment at `.venv`)
- Webcam
- Furhat Realtime API running locally (or accessible over WebSocket)
- OpenAI API key

## Setup (One Time)

1) Create and activate the venv (if not already done):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:

```powershell
pip install -r requirements.txt
```

3) Download OpenFace weights (only needed once, files are large):

```powershell
git lfs pull
```

4) Set your OpenAI API key (per session):

```powershell
$env:OPENAI_API_KEY="YOUR_API_KEY_HERE"
```

## Run the Tutor

From the project root:

```powershell
python -u integration\openai_swedish_tutor.py `
  --uri ws://localhost:9000/v1/events `
  --labels angry disgust fear happy neutral sad surprise `
  --openface-weights weights/MTL_backbone.pth `
  --openface-classifier models/openface_emotion_clf.pkl `
  --display --wait-speak-end --use-asr --listen-timeout 10 `
  --asr-min-words 3 --asr-extend-seconds 4 --asr-max-extends 1 `
  --asr-accept-early --asr-pause-before-accept 0.6 `
  --use-openai --debug
```

Notes:
- Remove `--use-asr` if you want to type answers instead of speaking.
- `--display` shows a live webcam window (press `q` to close it).
- `--debug` prints extra logs. Remove it if you want less output.
- If the ASR cuts you off, increase `--listen-timeout` or `--asr-extend-seconds`.
- If the ASR waits too long, lower `--asr-extend-seconds` or set `--asr-max-extends 0`.

## How a Session Looks

1) Tutor asks for a topic.
2) You say the topic (e.g., "friends" or "basic phrases").
3) Tutor asks for line 1. You say an English sentence about that topic.
4) Tutor translates to Swedish + explains word-by-word + pronunciation.
5) Tutor asks if you want to continue or repeat.
6) After 5 lines, it offers to stop, review the last 5 Swedish lines, or continue.

If you ask a question (for example: "how do you say today?"), it answers the doubt and then asks whether to continue or repeat.

## Troubleshooting

- If it says "I cannot translate right now": OpenAI is not responding or the key is missing.
- If you see no webcam window: check camera permissions and device index (`--camera 0`).
- If it fails to connect: verify your Furhat WebSocket URI (`--uri`).

