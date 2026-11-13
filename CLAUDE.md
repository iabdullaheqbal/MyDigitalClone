# Project guidance

- Keep the project on Python 3.11 and manage dependencies with `uv`.
- Keep the Space on CPU basic hardware. In `requirements.txt`, keep `torch`
  unpinned with the PyTorch CPU extra index; Spaces supplies Gradio and its
  dependencies. Pin only remaining direct runtime dependencies to versions in
  `uv.lock`; do not full-export transitive packages. After dependency changes,
  run `uv lock`, update those direct pins, and verify the Space install.
- Keep orchestration in `clone/service.py`; UI handlers should delegate to the
  service rather than implement retrieval or generation logic.
- Keep behavior routing, evidence grading, response templates, and guardrails
  in their dedicated `clone/` modules; log only intent categories and flags.
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
