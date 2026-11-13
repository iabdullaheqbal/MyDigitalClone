# Project guidance

- Keep the project on Python 3.11 and manage dependencies with `uv`.
- Keep `requirements.txt` limited to direct runtime dependencies not supplied
  by Hugging Face Spaces, pinned to versions in `uv.lock`; do not full-export
  transitive packages into it. After changing dependencies, run `uv lock`,
  update only those direct pins from `uv.lock`, and verify the Space install.
- Keep orchestration in `clone/service.py`; UI handlers should delegate to the
  service rather than implement retrieval or generation logic.
- Do not edit owner-provided notes in `data/` or generate personal facts unless
  the owner explicitly approves the change.
- Never commit, print, or log API keys. Use environment variables locally and
  Hugging Face Space secrets in deployment.
- Do not report evaluation metrics as validated until the draft cases in
  `eval/` have been reviewed. Do not invent results.
- Do not push to GitHub or deploy to the Hugging Face Space without explicit
  owner confirmation.

Run the local checks before proposing code changes:

```bash
uv run --locked ruff check app.py clone scripts tests
uv run --locked pytest -q
uv lock --check
```
