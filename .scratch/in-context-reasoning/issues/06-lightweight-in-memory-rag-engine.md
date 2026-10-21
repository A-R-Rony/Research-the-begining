# 06: Lightweight In-Memory RAG Engine with PyTorch Cosine Retrieval

**What to build:** A lightweight, self-contained Retrieval-Augmented Generation (RAG) module and theory guide operating completely within PyTorch GPU memory on Colab T4, computing dense sentence embeddings with `sentence-transformers/all-MiniLM-L6-v2` and performing cosine similarity matrix search to dynamically retrieve and inject top-k evidence passages.

**Blocked by:** 02 (Curate 20-Claim Benchmark Suite and Knowledge Corpus), 05 (Context-Aware Evidence Grounding and Faithfulness Verification)

**Status:** done

- [x] Write conceptual guide on dense vector spaces, bi-encoder projection, and dot product cosine similarity mathematics.
- [x] Implement document chunking and embedding generation using `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) on GPU.
- [x] Implement pure PyTorch top-k retrieval using matrix multiplication (`torch.mm`) over normalized document embeddings without external vector databases.
- [x] Connect the dynamic retriever directly into Llama-3's input context to perform real-time grounded claim verification on arbitrary input queries.

