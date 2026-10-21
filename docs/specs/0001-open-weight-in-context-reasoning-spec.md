# Specification: Open-Weight In-Context Reasoning Paradigms & Evaluation Benchmark

**Status**: Ready for Implementation (`ready-for-agent`)  
**Context**: Phase 5 Research & Learning Sprint (Aligned with Boise State Trustworthy AI Lab)  
**Governing Architecture**: [ADR-0002](../adr/0002-in-context-reasoning-and-lightweight-rag.md)  
**Domain Terminology**: [GLOSSARY.md](../../GLOSSARY.md)  

---

## Problem Statement

When learning and working with open-weight large language models, practitioners frequently treat prompt engineering as trial-and-error text formatting without understanding the underlying mechanics of tokenization, tokenizer chat templates, and context window economics. Furthermore, when moving from simple tasks to complex veracity verification and adversarial resistance, practitioners struggle to determine when to rely on a model's parametric weights (Zero-Shot), when to provide in-context demonstrations (Few-Shot), when to inject external evidence (Context-Aware Grounding & RAG), and when to enforce step-by-step reasoning (Chain-of-Thought). Without a structured pedagogical curriculum paired with a rigorous empirical benchmark, the learner cannot observe how each prompting paradigm trades inference latency, token consumption, and formatting reliability against factual reasoning accuracy.

---

## Solution

A dual-track pedagogical and empirical research module that first demystifies the theoretical and mechanistic foundations of open-weight prompt execution, and second conducts a unified benchmark across four core in-context reasoning paradigms (Zero-Shot, Few-Shot In-Context Learning, Context-Aware Grounding / Lightweight RAG, and Structured Chain-of-Thought) on a balanced suite of subtle, context-dependent misinformation claims. The solution delivers clear pedagogical documentation, an interactive and fully reproducible Google Colab execution environment, and an empirical evaluation report quantifying accuracy, token overhead, latency, schema adherence, and hallucination resistance.

---

## User Stories

1. As an AI research student, I want a conceptual guide explaining open-weight tokenization and special chat-template tokens, so that I understand how system prompts and user turns are serialized into raw token sequences.
2. As an AI research student, I want to learn the mathematical memory and compute footprint of prompt tokens inside the KV cache, so that I can budget context windows effectively on constrained hardware.
3. As a researcher, I want to implement a zero-shot veracity detection baseline, so that I have a reference point for what the open-weight model knows strictly through its pre-trained parametric weights.
4. As a researcher, I want to observe zero-shot failure modes on nuanced and emerging claims, so that I can identify the exact boundaries where parametric memory fails.
5. As an AI research student, I want to learn the mechanistic difference between Few-Shot In-Context Learning and weight fine-tuning, so that I understand how exemplars guide activation space without gradient updates.
6. As a researcher, I want to evaluate few-shot exemplars formatted via native chat-template conversation turns against monolithic prompt packing, so that I can determine which syntax yields higher schema adherence and lower format drift.
7. As an empirical evaluator, I want to test varying exemplar counts (k = 1, 3, and 5), so that I can quantify the marginal accuracy gains versus the linear token overhead costs.
8. As an AI research student, I want to learn how Context-Aware Grounding overrides model parametric hallucinations, so that I know how to force a model to prioritize supplied evidence over pre-trained beliefs.
9. As a researcher, I want to evaluate model veracity decisions when supplied with contradictory external evidence, so that I can measure the model's evidence faithfulness rate.
10. As an AI research student, I want to learn how dense vector embeddings and cosine similarity enable semantic passage retrieval, so that I understand the core mechanics of Retrieval-Augmented Generation from first principles.
11. As a practitioner, I want a lightweight in-memory retrieval engine using PyTorch matrix operations, so that I can perform semantic search in a transient notebook without external vector database services.
12. As a researcher, I want dynamic RAG retrieval to automatically pull relevant background evidence for unverified claims, so that the model evaluates real-time claims with grounded facts.
13. As an AI research student, I want to learn why open-weight 8B models suffer from narrative drift during unconstrained Chain-of-Thought, so that I understand the necessity of schema enforcement.
14. As a researcher, I want a three-stage epistemic CoT schema (entity check, evidence triangulation, and distortion analysis) emitted in structured JSON, so that intermediate reasoning premises can be audited independently of the final label.
15. As an empirical evaluator, I want to measure the exact token generation latency of Chain-of-Thought reasoning relative to Zero-Shot, so that I can quantify the computational cost of reasoning steps.
16. As a researcher, I want to implement a Self-Consistency sampling demonstration, so that I can evaluate whether majority voting over multiple reasoning trajectories improves label stability.
17. As an empirical researcher, I want a unified evaluation pipeline running all four paradigms over a 20-claim benchmark suite, so that I can produce a side-by-side comparison matrix.
18. As a student preparing for graduate lab outreach, I want a formalized research report detailing findings, trade-offs, and failure taxonomies, so that I can demonstrate graduate-level research execution to prospective mentors.

---

## Implementation Decisions

### Architectural Alignment
- The entire implementation adheres strictly to the architectural constraints documented in ADR-0002, retaining the 4-bit NF4 quantized 8B parameter model running within an NVIDIA T4 GPU runtime allocation.
- In-memory vector operations for semantic retrieval will use matrix multiplication operations directly in tensor space rather than external vector database services to preserve notebook portability.

### Core Modules
- **Pedagogical Theory Module**: Details prompt serialization, token economics, chat templates, and in-context learning theory.
- **Retrieval Engine Module**: Provides document chunking, bi-encoder dense embedding generation, and in-memory cosine ranking.
- **Inference & Reasoning Harness**: Implements standard execution interfaces across Zero-Shot, Few-Shot (native and monolithic), Context-Grounded, Dynamic RAG, and Epistemic CoT.
- **Evaluation & Benchmarking Engine**: Computes comparative metrics across all paradigms for a curated 20-claim dataset.

### Schema & Protocol Enforcement
- All generative tasks emit structured JSON with defensive post-processing to guarantee 100 percent parsing reliability.
- CoT prompts require explicit reasoning keys prior to verdict emission.
- In-context exemplars maintain distinct, isolated roles within the compilation template.

---

## Testing Decisions

### Seam Definition
- **Primary Seam**: The highest-level seam is the **Inference Pipeline Evaluator**.
- The evaluator accepts an input claim, optional evidence context, and a designated reasoning configuration, returning a normalized evaluation payload:
  - `claim_id`: Unique identifier for the evaluated statement.
  - `paradigm`: The active prompting mode (Zero-Shot, Few-Shot Multi-Turn, Few-Shot Monolithic, Context-Aware Grounded, In-Memory RAG, or Structured CoT).
  - `parsed_verdict`: Extracted veracity classification.
  - `confidence_score`: Calibrated confidence metric.
  - `schema_valid`: Boolean flag confirming JSON syntactic adherence.
  - `token_count`: Number of output tokens generated.
  - `latency_ms`: Wall-clock execution time for generation.
  - `faithfulness_flag`: Verification that the conclusion respects provided context.

### Testing Principles
- Tests evaluate observable runtime behavior and output validity rather than internal model hidden states.
- Prior art: Follows the established verification patterns from Phase 2 and Phase 3 evaluation scripts.

---

## Out of Scope

- Supervised fine-tuning (SFT) or parameter-efficient fine-tuning (LoRA / QLoRA) on open-weight models.
- External production vector database clusters (e.g., Pinecone, Milvus, Qdrant).
- Reinforcement Learning from Human Feedback (RLHF) or Direct Preference Optimization (DPO) pipelines.
- Multi-agent autonomous debate loops with more than two concurrent agents.

---

## Further Notes

- All code and documentation will be maintained strictly within repository bounds with clean relative links.
- The 20-claim dataset will expand the Phase 2 foundation, combining 5 historical claims with 15 subtle claims requiring deep contextual reasoning.
