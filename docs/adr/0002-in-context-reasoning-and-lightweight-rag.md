# 0002. In-Context Reasoning Paradigms and In-Memory Vector RAG Architecture

## Context
Following the baseline zero-shot evaluation in ADR-0001, we need to evaluate how advanced prompting paradigms (Zero-Shot, Few-Shot In-Context Learning, Context-Aware Grounding, and Chain-of-Thought) impact veracity detection, reasoning faithfulness, and adversarial robustness on `meta-llama/Meta-Llama-3-8B-Instruct`. Evaluating these paradigms on complex, nuanced misinformation requires an external knowledge retrieval mechanism and structured intermediate reasoning without blowing through Google Colab T4 hardware limits.

## Decision
1. **Model & Formatting Engine**: Retain `meta-llama/Meta-Llama-3-8B-Instruct` quantized under 4-bit NF4. Structure Few-Shot exemplars natively through the Hugging Face tokenizer chat template (`apply_chat_template`) as alternating user and assistant turns to preserve instruction tuning alignment, running an empirical benchmark against monolithic in-prompt packing.
2. **Two-Stage Context Awareness & Retrieval**:
   - Stage A: Explicit in-prompt evidence grounding to evaluate parametric memory override vs. hallucination.
   - Stage B: Lightweight in-memory RAG using `sentence-transformers/all-MiniLM-L6-v2` dense embeddings with pure PyTorch cosine similarity matrix operations (`torch.mm`), bypassing heavy external vector database dependencies (e.g., ChromaDB, Milvus) inside the transient Colab environment.
3. **Structured Epistemic Chain-of-Thought**: Implement a three-stage epistemic decomposition (`source_and_entity_check`, `evidence_triangulation`, `fallacy_or_distortion_detection`) emitted in strict JSON prior to the final label, preventing unconstrained token drift and enabling step-level verification.

## Considered Options
- **External Vector DB (FAISS / ChromaDB)**: Rejected for Stage B in favor of pure PyTorch matrix multiplication to keep the notebook fully self-contained, computationally transparent, and free of runtime binary conflicts.
- **Unstructured CoT ("Think step by step")**: Rejected in favor of schema-enforced JSON decomposition because open-weight 8B models exhibit narrative drift and confirmation bias without explicit step boundaries.
- **Generic NLP Benchmarks (GSM8K / MMLU)**: Rejected in favor of domain continuity on 20 balanced misinformation claims (5 historical baselines + 15 nuanced claims) to directly serve the Boise State Trustworthy AI Lab research trajectory.

## Consequences
- **Positive**: Complete reproducibility within a single free Google Colab T4 session without external API keys or background daemon services.
- **Positive**: Direct measurement of token overhead and inference latency tradeoffs across prompting paradigms.
- **Trade-off**: In-memory PyTorch cosine similarity scales linearly with the number of document chunks, making it optimal for local corpora (< 50,000 passages) but unsuitable for web-scale corpus search without hierarchical indexing.
