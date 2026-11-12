# Other projects I've built

**Veritas** is a news search engine I built for my NLP course. The backend is FastAPI with PostgreSQL and pgvector, the frontend is Next.js, and Celery with Redis ingests news from GNews and NewsAPI every six hours. Search uses hybrid retrieval: BM25 plus dense embeddings fused with Reciprocal Rank Fusion, then reranked with a cross-encoder (ms-marco-MiniLM). It has JWT auth with Google OAuth, and a personalized "For You" feed. I defended it in the course viva.

**FocusOS** is an HCI course project I built with Sharina: an AI deep work platform for enterprise teams. It uses Next.js 14, FastAPI, PostgreSQL, Mistral AI and Tesseract OCR, and the design is grounded in Flow Theory, Cognitive Load Theory, user-centered design, and the SUS and NASA-TLX evaluation scales.

**CCDA (CrossCountry Deal Advisor)** is a capstone-style project that predicts and explains trade deals using LightGBM and SHAP, with data from the World Bank, IMF and UN Comtrade, plus a RAG chat assistant on pgvector. When I first saw a suspiciously perfect accuracy, I learned to suspect target leakage instead of celebrating.

**Urdu fake news detection:** a pipeline combining TF-IDF and stylometric features with a fine-tuned XLM-RoBERTa, trained on a Kaggle TPU v5e-8 over a cleaned dataset of 62,017 rows.

**CAPTCHA breaker:** Faster R-CNN for geometric-shape CAPTCHAs and a CRNN with CTC loss for distorted-word CAPTCHAs, trained on Kaggle GPUs and served locally through FastAPI.

**Security projects:** an AI-powered intrusion detection system for network security, and an email phishing detector with a Chrome extension.

**Fiverr work:** text classification and email classification for clients.
