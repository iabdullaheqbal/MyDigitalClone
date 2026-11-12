# MyDigitalClone implementation plan

> **Current status:** The owner approved Phase 1, and Phases 1-9 have since
> been implemented. The Phase 0 orientation findings and open questions below
> are historical, not a description of the current repository. The remaining
> owner actions are to review the draft evaluation cases and configure the
> GitHub Actions deployment secret/variables. No deployment has been triggered.

## Orientation findings

- The repository currently has a `src/mydigitalclone/` package and an empty
  root `app.py`. There are 12 Markdown files in `data/`, but they are currently
  untracked; there are no tracked tests, evaluation sets, or generated artifacts.
- The 12 owner-provided notes cover biography, projects, preferences, technical
  topics, and a first-person FAQ, meeting the assignment's requested file count
  and content mix. Keep their contents private and unchanged. Before indexing,
  the owner should confirm these files are approved for a public Space and that
  references to other people are appropriate to publish.
- The project currently targets Python `>=3.12` and pins `.python-version` to
  `3.12`; the assignment requires Python 3.11. The assignment wins: Phase 1
  should pin Python 3.11 and adjust the project requirement and lockfile.
- Current lockfile versions: Gradio 6.31.0, sentence-transformers 6.2.0, and
  rank-bm25 0.2.2. Keep the README Space SDK version aligned with locked Gradio.
- Verified against installed Gradio 6.31.0: `gr.Sidebar` exists;
  `gr.Chatbot` supports message-format values, avatar images, and copy buttons;
  `gr.State` is available; and `Blocks.queue` supports a concurrency limit.
  The pinned installation, rather than a newer Gradio version, will guide UI
  implementation.

## File-by-file plan

- `pyproject.toml`, `uv.lock`, `.python-version`: pin Python 3.11, declare
  runtime and development dependencies, and keep the lock reproducible.
- `requirements.txt`: export the locked Space dependencies; verify CPU PyTorch
  packaging and installability in a clean Linux environment.
- `app.py`: keep to a small entrypoint that configures settings, wires the
  service and UI, and launches the queued Gradio app.
- `clone/config.py`, `persona.py`, `types.py`: validated environment settings,
  owner-approved persona language, and shared typed data structures.
- `clone/ingestion/loader.py`, `chunker.py`: load only owner-supplied Markdown
  and text files, ignore underscore-prefixed files, and make deterministic
  paragraph-aware chunks with overlap.
- `clone/indexing/store.py`, `builder.py`: fingerprint the corpus and model,
  build and validate JSON chunk metadata and NumPy embeddings.
- `clone/retrieval/tokenizer.py`, `bm25.py`, `dense.py`, `fusion.py`,
  `hybrid.py`, `reranker.py`: implement sparse/dense search, exact RRF, and an
  optional separately identified cross-encoder reranker.
- `clone/generation/prompts.py`, `llm_client.py`, `query_rewriter.py`, `guard.py`,
  `answerer.py`: Groq integration, grounded first-person answers, follow-up
  rewriting, abstention, and safe user-facing fallbacks.
- `clone/chat/session.py`, `clone/service.py`: transient per-session history and
  the sole application orchestration layer.
- `clone/ui/layout.py`, `handlers.py`, `formatting.py`, `styles.css`: Gradio
  chat, retrieval controls, evidence display, visible AI-clone banner, and
  session clearing, without business logic in UI code.
- `scripts/build_index.py`, `run_eval.py`, `compare_modes.py`: reproducible
  artifact generation and evaluation/report commands.
- `eval/eval_set.json`, `eval/eval_multiturn.json`: reviewed, source-grounded
  evaluation data; draft questions must be marked for human review.
- `tests/`: focused no-network tests for ingestion, retrieval, generation,
  sessions, service, UI smoke behavior, architecture, and repository hygiene.
- `data/`: use the existing 12 owner-provided `.md` files after publication
  approval; do not generate personal facts or edit this knowledge base except
  an explicitly reported typo fix.
- `artifacts/`: committed embeddings and chunk metadata built only from the
  approved knowledge base.
- `README.md`: Space front matter and, after measured runs and human review, the
  one-page report. Do not invent retrieval metrics or failure cases.
- `CLAUDE.md`, `docs/ARCHITECTURE.md`: concise project rules and architecture
  explanation derived from the approved design.
- `.github/workflows/ci.yml`: tests/lint first, then sync the passing `main`
  branch to the Space using the assignment's `HF_WRITE_TOKEN`, `HF_USER`, and
  `HF_SPACE` names. This replaces/aligns the existing `workflow.yaml` only
  after confirming the Hugging Face target and repository secrets/variables.

## Open questions and owner actions

1. Confirm the existing 12 `data/` files are approved for inclusion in the
   public Space, including references to other people; do not paste or copy
   sensitive information into reports.
2. Confirm the persona's first-person tone and any safe, public biographical
   facts to include. Do not infer facts beyond the approved notes.
3. Before Phase 4's one live smoke test, provide `GROQ_API_KEY` in the
   environment without pasting it into chat or source control.
4. Review/replace drafted evaluation examples before any report numbers are
   published.
5. Confirm the existing Hugging Face Space target
   (`abdullaheqbal/MyDigitalClone`) and create the assignment-specified GitHub
   secret/variables (`HF_WRITE_TOKEN`, `HF_USER`, `HF_SPACE`) before deployment.

## Risks and constraints

- Without owner data, the project cannot safely make claims as Abdullah or
  produce real retrieval/evaluation results.
- Python 3.11 may require dependency resolution changes from the current 3.12
  lock; resolve and test with `uv`, not by retaining the incompatible pin.
- The current deployment workflow expects `HF_TOKEN` and hardcodes the Space,
  while the assignment specifies `HF_WRITE_TOKEN` plus `HF_USER`/`HF_SPACE`.
  Avoid silently mixing the two configurations.
- Keep secrets in environment variables or Space secrets only. Never log,
  expose, or commit tokens, prompts containing private data, or sensitive
  owner information.
- Do not push to GitHub or Hugging Face without explicit owner confirmation.
- Any evaluation claims in README/report must be generated by actual commands
  and reviewed by the owner.

## Proposed implementation phases

Follow assignment sections 11-12: Phase 1 scaffold; Phase 2 ingestion/indexing;
Phase 3 retrieval; Phase 4 generation; Phase 5 session/service; Phase 6 UI and
entrypoint; Phase 7 evaluation; Phase 8 README/packaging; Phase 9 deployment
instructions; Phase 10 rubric audit. After each phase, run the focused
available tests, report actual output and results, and stop for owner approval.
