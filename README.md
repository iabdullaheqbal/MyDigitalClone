---
title: My Digital Clone
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.31.0"
python_version: "3.12.12"
app_file: app.py
pinned: false
---

# MyDigitalClone

An AI clone that answers questions from owner-approved notes using hybrid
BM25+dense retrieval, reciprocal-rank fusion, and grounded generation through
Groq. Conversation history is kept in per-browser session state; it is used for
follow-up continuity, not as evidence.

See [the architecture overview](docs/ARCHITECTURE.md) for the application flow
and [CLAUDE.md](CLAUDE.md) for project contribution guidance.

## Features

- Switch between BM25, dense, and hybrid retrieval; inspect retrieved passages,
  source names, scores, and latency in the interface.
- Optionally rerank passages with a cross-encoder.
- Route small talk and sensitive requests safely, and answer personal questions
  only when supported by the notes.
- Offer note-derived topic suggestions when a request is outside the notes.
- Continue or clear a browser session without sharing its history with other
  sessions.

## Run locally

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp -n .env.example .env
```

Set `GROQ_API_KEY` in the ignored local `.env` file. The model and other
settings have defaults in `.env.example`; evidence grading is configurable with
`STRONG_COSINE` and `WEAK_COSINE`. Do not commit `.env`.

The index files in `artifacts/` must match the notes and embedding settings.
To rebuild the index, run:

```bash
uv run --env-file .env python scripts/build_index.py
```

Then start the app:

```bash
uv run --env-file .env python app.py
```

The sentence-transformer encoder and optional cross-encoder run on CPU. Their
model weights may need to download the first time they are used.

## Hugging Face Spaces

The Space uses the Gradio SDK with Python 3.12.12 on CPU basic hardware.
`requirements.txt` installs CPU PyTorch from the PyTorch CPU wheel index and
lists the remaining direct runtime dependencies pinned to their `uv.lock`
versions. The Space supplies Gradio and its dependencies.
After changing `pyproject.toml`, run `uv lock`, copy the resolved versions for
the app's non-Space-provided direct dependencies from `uv.lock` into
`requirements.txt`, and verify with the Space install command. Do not replace
it with a full `uv export`, which would pin packages the Space manages.
Add `GROQ_API_KEY` as a Space secret (not as a repository file). `GROQ_MODEL`
is optional; the default is `openai/gpt-oss-20b`. The app can still start
without a key and will return a friendly topic-aware fallback instead of
calling Groq.

## GitHub Actions deployment

The workflow at `.github/workflows/ci.yml` runs lint, tests, and a wheel build
for pull requests and pushes to `main`. A push to `main` deploys to the Space
only after validation succeeds. Configure these in the GitHub repository:

- Secret `HF_WRITE_TOKEN`: a Hugging Face token with write access to the Space.
- Variable `HF_USER`: `abdullaheqbal`.
- Variable `HF_SPACE`: `MyDigitalClone`.

The workflow does not use the old `HF_TOKEN` secret name. Pull requests do not
deploy, and no credentials belong in the repository.

## Evaluation

Draft source-relevance and multi-turn cases are in `eval/`. The retrieval
evaluation commands reject draft cases unless `--allow-drafts` is supplied;
reports produced from drafts remain unvalidated. The behavior runner uses its
own `needs_review` cases and exports a CSV with blank professionalism ratings.
Review all drafted cases before treating any results as validated:

```bash
uv run --env-file .env python scripts/run_eval.py
uv run --env-file .env python scripts/compare_modes.py
uv run --env-file .env python scripts/run_eval.py --behavior
```

No evaluation metrics are published here because the draft judgments have not
yet been reviewed.
