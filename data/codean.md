# CODEAN: my final year project

CODEAN is my final year project at UMT, done as a team of three, with Dr. Muhammad Wasim as supervisor (session 2026-2027). It's an MCP-based autonomous software engineering system. In plain words: it sits next to an AI coding assistant and makes that assistant cheaper, better informed and safer.

## The problem
AI coding tools burn a lot of tokens reading code they don't need, they hallucinate APIs that don't exist or have changed, and they sometimes suggest packages that don't exist, which attackers can register with malware (this is called slopsquatting).

## The idea
CODEAN is one closed-loop pipeline, not separate modules:
1. Structural exploration: Tree-Sitter builds an AST knowledge graph of the codebase.
2. Context pruning: a dual-rubric method keeps only the code relevant to the task, which saves tokens.
3. API grounding: Context7 provides live, versioned documentation, so the model doesn't rely on stale memory.
4. Deterministic verification: runtime introspection and deterministic AST rewriting check and auto-repair the generated code.
5. Package screening: OSV.dev, PyPI and npm lookups plus a Random Forest classifier flag suspicious or slopsquatted packages.

## Architecture and delivery
It ships as a local MCP server that works with VS Code, Cursor, Windsurf and Claude Code. After weighing options, I settled on Rust for the compute and indexing engine, Python for the MCP server layer, and TypeScript with React for a local graph visualization UI. I'm using the open-source codebase-memory-mcp project as an architecture reference only, not copying it.
