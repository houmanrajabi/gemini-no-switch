# Gemini Direct

## The Problem
Google confirms this: when you hit your Pro/Thinking limit, it falls back to Flash — without telling you.

Using Gemini's web (I experienced it on the pro student plan), when it hits a hidden usage cap, it **silently swaps the model** to a weaker one mid-conversation. One minute it's brilliant, the next it can't handle even simple prompts. 

This app calls the API directly. pick the model you want and it stays on that model. If you hit a limit, you get an error — not a secretly degraded response.

## Setup

```bash
pip install flask google-genai
export GEMINI_KEY="your-api-key"
python app.py
```

Open [localhost:7860](http://localhost:7860). Done.

## What You Get

- **Pick your model** — 3 Pro, 3 Flash, 2.5 Pro, 2.5 Flash
- **Adjustable thinking** — Off / Low / Medium / High
- **File uploads** — PDFs, images, audio, video, code, Office docs
- **Multi-turn chat** — full conversation history per session
- **Dark UI** — markdown, code blocks, clean and fast

## Side-by-Side Comparison

Want to compare two models on the same prompt? A VS Code `launch.json` is included to run two instances at once:

```bash
# Or just do it manually:
FLASK_RUN_PORT=5000 python app.py &
FLASK_RUN_PORT=5001 python app.py &
```

Open both ports, pick different models, compare.

## Env Variables

| Variable | Required | Default |
|----------|----------|---------|
| `GEMINI_KEY` | Yes | — |
| `FLASK_RUN_PORT` | No | `7860` |

## Good to Know

- Conversations are in-memory — they reset when the server restarts.
- API usage isn't free, but at least you know exactly what model you're talking to.
- Uploaded files accumulate in the `uploads/` folder — clean it out periodically so it doesn't eat your disk space.

MIT License
