# Architecture

## Runtime flow

`app.py` calls `clone.bootstrap.create_app()`. Bootstrap reads validated
settings, verifies that the saved index fingerprint matches the current notes
and embedding/chunk settings, and wires the retrievers, optional Groq client,
service, and Gradio UI.

The UI stores a `ChatSession` in Gradio session state. A message is handled by
`CloneService`, which can rewrite a follow-up query using that session's recent
history, retrieves current-turn evidence, applies the relevance guard, and
generates a grounded response when a Groq key is configured. Conversation
history supports continuity but is not a source of evidence. Clearing the
conversation creates fresh session state.

## Indexing and retrieval

`scripts/build_index.py` loads owner-provided Markdown/text files from `data/`,
chunks them deterministically, encodes the chunks, and saves chunk metadata and
embeddings under `artifacts/`. The fingerprint prevents the app from silently
using an index built from different source or embedding settings.

Retrieval offers BM25, dense cosine-similarity, and hybrid reciprocal-rank
fusion modes. An optional cross-encoder can rerank candidates. Retrieval
components are isolated from generation; the application service coordinates
the end-to-end request.

## Evaluation and deployment

The evaluation runner measures source-level precision@k, recall@k,
reciprocal-rank@k, and retrieval latency without calling a generation API.
Evaluation datasets marked as drafts require an explicit `--allow-drafts`
flag, and outputs retain their unvalidated status. Review the cases before
publishing any metrics.

The GitHub Actions workflow runs lint, tests, and a wheel build. Only a
validated push to `main` attempts to synchronize the repository to the Space,
using the configured `HF_WRITE_TOKEN` secret and `HF_USER`/`HF_SPACE`
variables. The Space builds its Python 3.11 Gradio app from the root
`Dockerfile`. Deployment credentials must not be stored in the repository.
