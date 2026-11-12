# Study notes: machine learning pipelines from my projects

## Text classification baselines
For text classification, including my Fiverr work, I start with a classical baseline. TF-IDF turns text into sparse weighted word-count vectors, and a linear model such as logistic regression or a linear SVM is trained on top. It is fast, cheap and often surprisingly hard to beat on small datasets. Only after the baseline do I consider fine-tuning a transformer, since the gain has to justify the extra cost.

## Stylometric features
Stylometry measures writing style instead of topic: sentence length, punctuation habits, vocabulary richness, function-word frequencies. In my Urdu fake news detector I combined TF-IDF with stylometric features, because fake and real articles often differ in style as well as content.

## Fine-tuning transformers
XLM-RoBERTa is a multilingual transformer pretrained on text in around a hundred languages, which makes it a good choice for Urdu. Fine-tuning adds a classification head and updates the model weights on labeled data. I trained on a Kaggle TPU v5e-8 using a cleaned dataset of 62,017 rows. Important practices are a proper train/validation/test split, a small learning rate, watching for overfitting, and comparing against the TF-IDF baseline.

## Gradient boosting and explainability
In CCDA I used LightGBM, a gradient boosting library that builds trees sequentially, each correcting the errors of the previous ones. To explain predictions I used SHAP, which assigns each feature a contribution to an individual prediction, based on Shapley values from game theory. This lets a user see why the model rated a trade deal the way it did.

## Target leakage
When CCDA showed 100 percent accuracy, that was a warning sign and not a success. Target leakage happens when a feature contains information that would not be available at prediction time, or is derived from the label itself, so the model effectively sees the answer. The fix is to audit features, make sure the split respects time or groups, and be suspicious of any result that looks too perfect.

## Object detection and sequence recognition
My CAPTCHA breaker used two models. Faster R-CNN is a two-stage object detector: a region proposal network suggests candidate boxes and a second stage classifies and refines them. I used it for CAPTCHAs made of geometric shapes. For distorted words I used a CRNN, a convolutional network that extracts visual features feeding a recurrent network, trained with CTC loss. CTC lets the model learn to read a sequence of characters without needing per-character alignment between the image and the label. I trained both on Kaggle GPUs and served them locally through FastAPI.

## Dataset building for a humanizer model
For the text humanizer feature planned for Resona, the dataset plan is to start from human-written academic text from S2ORC or peS2o, beginning with computer science sections, then generate AI-like versions of each paragraph so that human and AI pairs exist. Splits are done at paper level (70, 15 and 15 percent for train, validation and test) to avoid leakage between paragraphs of the same paper. The data is stored in JSONL, the model to fine-tune is Qwen3-8B, and the first version targets about 5,000 paragraphs, scaling to 20,000 to 50,000 later. The scope of the tool is language-level editing only: grammar, AI-sounding phrasing and flow. It does not change the logic or arguments.

## Evaluation habits
I try to keep a held-out test set untouched until the end, report more than accuracy when classes are imbalanced (precision, recall, F1), compare against a simple baseline, and check explanations with SHAP or error analysis. For exams, I have worked through NLP topics such as IR evaluation, word vectors (Word2Vec), TF-IDF and PPMI, and neural networks, which give me the vocabulary to reason about these choices.

## Search and hybrid retrieval in practice
Veritas ingests news from GNews and NewsAPI every six hours using Celery and Redis. Articles are embedded and stored with pgvector. A query goes through BM25 and dense retrieval, the two ranked lists are fused with RRF, and a cross-encoder reranks the top results. A personalized For You feed uses stored user signals. This project taught me how retrieval, background ingestion and an application backend fit together.
