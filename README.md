# AI Safety Brief

`AI Safety Brief` is a Telegram bot that builds scheduled AI safety digests for a private chat or group. It collects items from a curated set of AI safety sources, ranks them, filters low-signal pieces, summarizes the top `k`, and sends a compact brief on a configurable schedule.

## What It Does

- Sends scheduled digests to a private chat or Telegram group
- Supports `daily`, `hourly`, and `weekly` cadences
- Lets each chat configure:
  - `top_k`
  - timezone
  - cadence
  - source enable/disable toggles
  - repeat window to avoid resurfacing the same items too quickly
- Separates the digest into `News`, `Papers`, and `Opinion`
- Stores chat settings, seen history, and run history in SQLite
- Runs as a polling worker, which makes it easy to keep alive on Fly.io
- Uses extractive summarization by default, with optional free-tier Groq refinement if you provide a key

## Current Source Registry

Core sources included in the registry:

- Alignment Forum
- LessWrong
- EA Forum
- Anthropic Research
- Anthropic News
- OpenAI Research
- OpenAI Research News
- Google DeepMind Blog
- METR
- Redwood Research
- Apollo Research
- Center for AI Safety
- GovAI
- Institute for Law & AI
- Epoch AI
- Future of Life Institute
- UK AI Security Institute
- AI Safety Newsletter
- arXiv alignment queries
- arXiv evals and interpretability queries

Best-effort X support is included through configurable RSS-style mirrors via `X_RSS_BASE_URL` and `X_ACCOUNTS`. It is off by default so X failures never block the digest.

## Commands

- `/start` registers the chat
- `/help` shows usage
- `/brief` generates an immediate digest
- `/status` shows the active settings and next run
- `/history` shows recent digest runs
- `/sources` lists source toggles
- `/pause` pauses scheduled sends
- `/resume` resumes scheduled sends
- `/settings` shows current settings
- `/settings k 5`
- `/settings timezone Asia/Singapore`
- `/settings cadence daily 19:00`
- `/settings cadence hourly 6`
- `/settings cadence weekly Mon,Wed,Fri 18:30`
- `/settings source enable x_openai`
- `/settings source disable lesswrong`
- `/settings repeat-window 7`

In groups, only admins can change persistent settings.

## Local Setup

1. Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the example env file:

```bash
cp .env.example .env
```

4. Set at least:

```bash
TELEGRAM_BOT_TOKEN=...
```

Optional:

```bash
GROQ_API_KEY=...
X_RSS_BASE_URL=...
X_ACCOUNTS=AnthropicAI,OpenAI,GoogleDeepMind,METR_Evals
```

5. Run the bot:

```bash
python main.py
```

## Verification

Run the test suite:

```bash
.venv/bin/pytest -q
```

Run the local digest dry run:

```bash
.venv/bin/python scripts/dry_run.py
```

## Deploying To Fly.io

This bot is configured as a worker process, not a web service.

1. Authenticate:

```bash
fly auth login
```

2. Create the app and volume if needed:

```bash
fly launch --no-deploy
fly volumes create botdata --size 1
```

3. Set secrets:

```bash
fly secrets set TELEGRAM_BOT_TOKEN=...
fly secrets set GROQ_API_KEY=...
```

4. Deploy:

```bash
fly deploy
```

5. Ensure the worker stays on:

```bash
fly machine update <machine-id> --autostop=off --restart=always -y
```

## Notes

- The summarization pipeline works without any LLM key.
- Groq refinement is optional and only used when `GROQ_API_KEY` is configured.
- X ingestion is intentionally best-effort and disabled by default.
- The current implementation favors robust filtering and graceful degradation over scraping every source perfectly from day one.
