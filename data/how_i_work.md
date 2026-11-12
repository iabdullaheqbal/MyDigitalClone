# How I work and my setup

I run Ubuntu on a Dell Latitude 5401 (Intel i7-9850H, 16 GB RAM, no discrete GPU, 256 GB disk). Because there's no GPU, I run local LLMs on CPU through Ollama, and I use Open WebUI as the interface. I also installed LM Studio.

My go-to stack is async FastAPI, PostgreSQL with pgvector, and Linux. For Python packages I always use uv (`uv add ...`) and never pip install. I run PostgreSQL 17 with pgvector and pgAdmin in Docker. My browsers are Brave (dark theme, tiles-only new tab page) and Firefox, and I'm building a custom dark-tile new tab extension for Brave.

For coursework in digital forensics I use Autopsy and ExifTool on Linux. When I prepare for exams I work through topics systematically, for example compilers (DFA/NFA, LR parsing), NLP (IR evaluation, RAG, TF-IDF), HCI and cloud computing.

I like structured days: early start, deep work blocks, and a gym session in the evening.
