---
title: My Digital Clone
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
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
- Abstain when the notes do not provide relevant evidence.
- Continue or clear a browser session without sharing its history with other
  sessions.

## Run locally

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp -n .env.example .env
```

Set `GROQ_API_KEY` in the ignored local `.env` file. The model and other
settings have defaults in `.env.example`; do not commit `.env`.

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

The Space builds the Gradio app from the root `Dockerfile` using Python 3.11.
`requirements.txt` installs the app dependencies, including CPU-only PyTorch.
Add `GROQ_API_KEY` as a Space secret (not as a repository file). `GROQ_MODEL`
is optional; the default is `openai/gpt-oss-20b`. The app can still start
without a key and will return a grounded passage fallback instead of calling
Groq.

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

Draft source-relevance and multi-turn cases are in `eval/`. They are explicitly
marked as requiring human review. The evaluation commands reject draft cases
unless `--allow-drafts` is supplied; reports produced from drafts remain
unvalidated. Review and update the cases before treating their metrics as
results:

```bash
uv run --env-file .env python scripts/run_eval.py
uv run --env-file .env python scripts/compare_modes.py
```

No evaluation metrics are published here because the draft judgments have not
yet been reviewed.
