# Study notes: retrieval and RAG

These are my notes on how retrieval-augmented generation works, based on what I built in Veritas and Resona and on my NLP course preparation.

## The RAG pipeline in four steps
RAG exists because a language model only knows what was in its training data and it can hallucinate. Instead of trusting its memory, we fetch relevant text first and make the model answer from that text.
1. Chunk: split documents into passages, usually a few hundred words, with a small overlap so a sentence at a boundary is not lost. Each chunk keeps metadata such as the source file and an ID.
2. Index: build a searchable representation of every chunk. This can be a keyword index, an embedding index, or both.
3. Retrieve: for each question, rank the chunks by relevance and keep the top k.
4. Generate: put the top chunks and the question into the prompt, and instruct the model to answer only from that context and to admit when the context does not contain the answer.

The quality of the final answer is capped by retrieval. If the right chunk is not in the top k, the model cannot use it, and a good model will either guess or say it does not know.

## Chunking decisions
Chunks that are too small lose context, so a pronoun or a heading ends up separated from what it refers to. Chunks that are too large dilute the embedding, because one vector has to summarize many topics, and they waste prompt space. A range of about 200 to 300 words with an overlap of roughly 10 to 15 percent is a sensible default. Splitting on paragraph boundaries is better than cutting blindly by word count, because a question and its answer in a FAQ should stay in the same chunk.

## BM25: keyword retrieval
BM25 is a ranking function from the family of TF-IDF methods. For each query term it scores a document using three ideas:
- Term frequency: more occurrences of the term in the chunk means a higher score, but with diminishing returns (saturation), so repeating a word ten times does not make a chunk ten times better.
- Inverse document frequency: rare terms matter more than common ones. A project name that appears in one chunk is a strong signal, while the word "the" tells you nothing.
- Length normalization: long chunks are penalized so they do not win just by containing more words.

Strengths: it is exact, fast, needs no model, and is excellent when the question contains a rare specific term such as a project name, a tool or a place. Weakness: it only matches tokens. If the question says "How do you relax?" and the answer says "I go cycling in the evening," there is no shared word, so BM25 scores zero. It also does not understand synonyms or paraphrases. Preprocessing such as lowercasing and removing stopwords helps, and a score of zero should be treated as no match instead of ranking zero-score chunks arbitrarily.

## Dense retrieval with sentence embeddings
A sentence-embedding model such as all-MiniLM-L6-v2 maps each chunk to a vector, 384 dimensions in that case, so that texts with similar meaning land close together. The question is embedded the same way and compared to every chunk vector using cosine similarity. If the vectors are normalized to unit length, cosine similarity is just a dot product, which numpy handles quickly for a small corpus. Embeddings for the chunks only need to be computed once and can be saved to disk, for example as a .npy file, then loaded at startup.

Strengths: it handles paraphrases, synonyms and loosely worded questions. Weakness: it can miss exact rare terms, because a rare name may be poorly represented in the embedding model's vocabulary, and it can return chunks that are topically similar but do not actually contain the answer. Similarity scores also have no built-in meaning of "good enough," so a threshold has to be tuned.

## Why two methods fail differently
BM25 fails when wording differs and succeeds on exact terms. Dense retrieval fails on rare exact terms and succeeds when meaning matches without shared words. Because their errors are different, combining them is useful. This is the main motivation for hybrid search, and the BEIR benchmark showed that BM25 is a surprisingly strong baseline outside the domain a dense model was trained on.

## Hybrid fusion
Two rankings can be merged in two common ways.
- Reciprocal Rank Fusion (RRF): the score of a chunk is the sum over rankings of 1 / (k + rank), with k usually 60. It only uses ranks, so there is no need to normalize scores that live on different scales, BM25 scores being unbounded and cosine scores being between -1 and 1. A chunk that appears high in both lists gets the highest fused score. I used RRF in Veritas.
- Weighted score fusion: normalize each score list with min-max scaling and combine them as alpha times the dense score plus (1 minus alpha) times the BM25 score. This gives a tunable knob but depends on good normalization and a well-chosen alpha.

## Reranking
Retrieval models such as BM25 and bi-encoders score the query and the chunk independently. A cross-encoder reads the query and the chunk together in one pass, which is slower but more accurate. A common pattern is to retrieve a few dozen candidates cheaply, then rerank them with a cross-encoder such as ms-marco-MiniLM, and send only the best few to the language model. Veritas used this pattern.

## Evaluating retrieval
Common measures in information retrieval are precision (how many retrieved items are relevant), recall (how many relevant items were found), hit@k (whether a correct chunk appears in the top k), and mean reciprocal rank. A small hand-labeled set of questions with the correct chunk for each is enough to compare BM25, dense and hybrid on hit@3.

## Grounded generation
A good RAG prompt does four things: it sets a persona, it tells the model to answer only from the supplied context, it gives an explicit fallback sentence for missing information, and it tells the model to treat the context as data, not instructions. Honest behavior matters more than fluency, because an invented answer is worse than "I don't have that."

## pgvector
In Resona and Veritas I store embeddings in PostgreSQL using the pgvector extension, so semantic search lives in the same database as the rest of the application data. It supports cosine, inner product and L2 distance, and approximate indexes such as HNSW for fast nearest-neighbor search on larger collections. Resona's novelty-checking idea used cosine similarity over section-level embeddings, though I dropped that feature from version one.
