# Resona: my AI research platform

Resona is an AI research SaaS that I co-own with my partner Sharina. She builds the Next.js frontend, and I own the entire backend, the machine learning and the infrastructure. It competes with tools like Perplexity, Elicit and SciSpace, and the idea is to put the whole research workflow for a student or researcher in one place instead of five different tabs.

## What it does
Resona has AI Search, Discover, History, a Literature Review tool, a Paper Writer, an AI Paraphraser, a Dataset Finder, Chat with PDF, an AI Content Detector, and a Collaborate area with three privacy levels: Public, Signal Mode and Private. It also has a social network and messaging so researchers can talk to each other. I'm also working on a text humanizer feature that fixes grammar and AI-sounding phrasing across a whole document. It only edits the language, and never writes new content or touches the argument, because that stays the researcher's job.

## How I built the backend
The backend is async FastAPI with SQLAlchemy and asyncpg on top of PostgreSQL. I use the pgvector extension for embeddings and semantic similarity search. Redis and Celery handle background jobs such as processing papers, and Nginx sits in front. The database schema has 30 tables grouped into 11 modules. Retrieval-augmented generation powers the features that need answers grounded in real papers.

## Decisions I made
For version one I cut the AI Novelty Checker, because I wasn't sure it was a useful feature, and the Citation Generator, because it was out of scope for the time I had. I'm not deploying Resona publicly for now, so it runs locally. I'm also not open-sourcing the product. For payments, I looked at Paddle and Dodo Payments because Stripe doesn't support Pakistani merchant accounts.

## Why I'm building it
Resona is my attempt to turn research skills into a real product. It's where I practice database design, retrieval, infrastructure and product thinking together.
