# Study notes: backend engineering

I handle backend and infrastructure for Resona, so these are my working notes on the stack I use.

## Async FastAPI
FastAPI is a Python web framework built on Starlette and Pydantic. Handlers can be declared with async def so the event loop can serve other requests while one request waits on the database or a network call. This matters for I/O-heavy apps such as search and chat features. Pydantic models validate request and response data and generate OpenAPI documentation automatically. A blocking call inside an async handler, such as a synchronous database driver or a heavy computation, stalls the event loop, so CPU-heavy work belongs in a background worker.

## SQLAlchemy and asyncpg
SQLAlchemy is the ORM and query toolkit, and asyncpg is a fast asynchronous PostgreSQL driver. Together they allow async sessions that do not block the event loop. Good practice is one session per request, explicit transactions, and migrations managed with a tool such as Alembic.

## PostgreSQL schema design
Resona's schema has 30 tables grouped into 11 modules. Grouping tables into modules (for example users, papers, collaboration, messaging) keeps responsibilities clear. Foreign keys enforce integrity, indexes speed up common queries, and normalization avoids duplicated data. For text search and embeddings, PostgreSQL can hold both relational data and vectors in the same database through the pgvector extension.

## Background jobs with Celery and Redis
Slow work, such as parsing a PDF, generating embeddings or calling an LLM for a long document, should not block a web request. Celery workers pull tasks from a queue, with Redis acting as the broker and sometimes the result backend. The API returns quickly with a task ID, and the client polls or is notified when the job finishes. Retries, time limits and idempotent tasks make failures easier to handle.

## Nginx and TLS
Nginx works as a reverse proxy in front of the application server, handling TLS termination, compression and routing. Certbot can issue and renew free Let's Encrypt certificates. Although I am not deploying Resona publicly for now, I designed the stack with a VPS deployment on Nginx in mind.

## Authentication
JWT-based authentication issues a signed token after login, which the client sends with each request. Tokens should have short lifetimes, secrets must be kept out of the repository, and passwords must be stored hashed. In Veritas I supported Google OAuth as well as email and password sign-in. In CCDA I used SHA-256 hashed tokens for email verification.

## Payments in Pakistan
Stripe does not support Pakistani merchant accounts, so for Resona I evaluated Paddle and Dodo Payments, which act as merchant-of-record platforms.

## Tooling habits
I use uv for Python dependency management (`uv add`, never pip install) because it is fast and keeps the lockfile consistent. I run PostgreSQL 17 with pgvector and pgAdmin in Docker so the database setup is reproducible and does not depend on the Ubuntu system packages. I develop on Linux and keep secrets in environment variables.

## Local LLMs
My laptop has no discrete GPU, so I run small models on CPU using Ollama, with Open WebUI as the interface. Quantized models make this practical, trading a little quality for a lot less memory.

## Automation with n8n
For my internship I built a Multi-channel AI Agent in n8n. In Phase 1, three channels (website chat, email and a simulated WhatsApp) all enter the system through webhooks and flow into one central AI agent that uses Mistral as the language model. Messages are stored in Google Sheets in Phase 1, and the design leaves room for a real database later. The point of the architecture is that every channel is normalized into one message format, so the agent logic is written once.
