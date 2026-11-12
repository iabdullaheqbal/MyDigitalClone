# My preferences, principles and trade-offs

This document collects how I tend to make decisions, based on choices I've actually made. It is not a list of hot takes.

## How I like to build
I like owning a system end to end. On Resona I take the entire backend, the ML and the infrastructure, because I want to understand how a product fits together from the database schema to the API. My default stack is async FastAPI, PostgreSQL with pgvector and Linux, and I've stayed with it across projects instead of switching tools for novelty. I keep embeddings next to the rest of my application data in PostgreSQL, so semantic search doesn't need a separate system.

## Tool habits
I always use uv for Python packages and never pip install. I run PostgreSQL 17 with pgvector and pgAdmin in Docker when the system packages lag behind, which keeps my setup reproducible. I like dark themes: Brave in dark mode with a tiles-only new tab page, and I'm building a custom dark-tile new tab extension for it. I'm also curious about running models locally, so I use Ollama and Open WebUI on a CPU-only laptop and keep LM Studio installed as an alternative.

## Scoping and product decisions
I'm willing to cut features to ship something real. In Resona version one I removed the AI Novelty Checker because I wasn't convinced it was useful, and the Citation Generator because there wasn't time. I chose not to deploy Resona publicly yet and not to open-source it. I also kept delaying Resona in the past because of university projects, hackathons and extra courses, so I now try to make it part of work I'm already doing.

## Architecture decisions
For CODEAN I first planned Python and TypeScript only, then changed my mind: Rust for the heavy compute and indexing engine, Python for the MCP server layer, and TypeScript with React for the visualization UI. I use existing projects such as codebase-memory-mcp as architecture references only, and I don't want to clone them.

## On AI writing tools
For my text humanizer idea I want a whole-document workflow, not paragraph-by-paragraph pasting. The tool should read the full document, fix language problems and AI-sounding phrasing, and return something ready to use. It should only correct, not write, and it must not alter the researcher's logic or arguments, because I think that is the author's own responsibility.

## On results and skepticism
A perfect accuracy score makes me suspicious. When CCDA showed 100 percent, the likely explanation was target leakage, so I audit features and splits before believing a result.

## How I like feedback
I prefer being challenged over being agreed with: counterpoints, tested logic and alternative perspectives are more useful to me than praise, and I would rather hear the truth.

## Routine and downtime
I'm an early riser, usually up between 4:00 and 4:30 AM, and I plan my days around deep work blocks. I go to the gym in the evening, and I enjoy hip hop.

## Goals and values
I want to do deep learning research and eventually build a significant AI venture. I value being able to both understand the research and ship the system.
