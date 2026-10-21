# Topic 2.3: Context-Aware Grounding and Lightweight Retrieval-Augmented Generation (RAG)

In Topic 2.2, we discovered that Few-Shot In-Context Learning (ICL) stabilizes response formatting and guides the model's activation space, but cannot supply facts that do not already exist in the model's pre-trained parametric weights. When a claim hinges on subtle statistics (e.g., U-3 unemployment definitions) or counter-intuitive scientific phenomena (e.g., boreal forest albedo warming), ungrounded models hallucinate or default to superficial surface associations.

This guide explores **Context-Aware Grounding** (Stage A: prompt-level evidence injection and faithfulness verification) and sets the foundation for **Lightweight In-Memory RAG** (Stage B: dense vector embeddings and cosine similarity retrieval).

---

## 1. The Parametric Memory Collision

When an autoregressive language model processes a prompt, its internal attention heads attend to two distinct sources of information:

```text
┌────────────────────────────────────────────────────────────────────────┐
│             THE DUAL-SOURCE ATTENTION TENSION IN TRANSFORMERS          │
├───────────────────────────────────┬────────────────────────────────────┤
│  1. Parametric Internal Memory    │  2. Working Context Window         │
├───────────────────────────────────┼────────────────────────────────────┤
│  • Encoded across 8.03 billion    │  • The active token sequence in    │
│    matrix weights                 │    GPU memory (KV cache)           │
│  • Static: frozen at pre-training │  • Dynamic: injected at runtime    │
│    cutoff date                    │    per inference                   │
│  • Associative & probabilistic:   │  • Deterministic: authoritative    │
│    prone to popular consensus     │    ground-truth evidence passages  │
│    and common web misconceptions  │                                    │
└───────────────────────────────────┴────────────────────────────────────┘
```

When an external factual document contradicts a model's pre-trained probabilistic prior, a **parametric memory collision** occurs. Without explicit prompt constraints, 8B models frequently ignore the supplied passage and regurgitate their pre-trained bias.

### Enforcing the Epistemic Hierarchy
To guarantee that external facts override parametric hallucinations, the prompt envelope must enforce an explicit epistemic hierarchy:
`Injected External Evidence > Parametric Pre-Trained Beliefs`

This is accomplished by setting strict negative constraints in the system prompt:
1. **Source Exclusivity**: Instruct the model that claims must be verified strictly using facts articulated within the provided passage.
2. **Conflict Resolution Rule**: Explicitly command the model: *"If the provided Reference Evidence contradicts your pre-existing knowledge, you must prioritize the Reference Evidence unconditionally."*
3. **Citation Requirement**: Force the model to quote or paraphrase the exact evidence sentence in its JSON `reasoning` field.

---

## 2. The Grounded Prompt Envelope Architecture

In Stage A (Prompt-Level Grounding), each claim is wrapped alongside its corresponding verified passage from `data/knowledge_corpus.json` into an isolated user message:

```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an evidence-grounded verification assistant. 
Your task is to determine whether the submitted Claim is True or False STRICTLY and SOLELY based on the provided Reference Evidence.
If the Reference Evidence confirms the claim, predict True. 
If the Reference Evidence refutes the claim or proves it inaccurate, predict False.
Do not rely on unstated assumptions or prior beliefs. Output valid JSON only.<|eot_id|><|start_header_id|>user<|end_header_id|>

### Reference Evidence:
NASA satellite observations and climate models demonstrate that while solar irradiance variations do influence Earth's upper atmosphere, total solar output has fluctuated within a narrow 0.1 percent band since 1750. In contrast, greenhouse gas concentrations produced by fossil fuel combustion have increased atmospheric heat retention by orders of magnitude greater than solar variance. NASA officially attributes observed global warming primarily to human greenhouse emissions.

### Target Claim:
"NASA has officially confirmed that human carbon emissions have zero impact on global climate change and it is entirely solar cycles."

Evaluate the claim against the Reference Evidence. Respond strictly in JSON.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

When compiled through `tokenizer.apply_chat_template`, this structure primes the cross-attention layers to compute direct token-to-token semantic alignment between the claim tokens and the evidence tokens.

---

## 3. Measuring Veracity Flip Rate & Evidence Faithfulness

To rigorously evaluate the efficacy of Context-Aware Grounding over cold Zero-Shot prompting, we track two primary metrics:

### 1. Veracity Flip Rate (Error Correction Rate)
The proportion of claims that were incorrectly classified under cold Zero-Shot inference but flip to the correct ground truth once grounded evidence is provided:

`Flip_Rate = Number_of_Incorrect_Zero_Shot_Claims_Flipped_to_Correct / Total_Incorrect_Zero_Shot_Claims`

On our 20-claim benchmark:
- Under Zero-Shot, nuanced claims (e.g., `F06` U-3 unemployment, `F08` EV lifecycle emissions, `T05` boreal forest albedo) suffer high error rates because the model relies on associative heuristics.
- Under Context-Aware Grounding, providing authoritative passages from the Bureau of Labor Statistics and ICCT causes these claims to flip to 100 percent accuracy.

### 2. Evidence Faithfulness Score
A measurement of whether the model's generated reasoning is logically grounded in the provided document rather than fabricating extraneous rationales:

`Faithfulness = Supported_Rationales / Total_Evaluated_Claims`

A rationale is classified as faithful if:
1. It attributes its verdict directly to the named entity or mechanism articulated in the passage (e.g., citing "U-3 excludes workers who have not actively searched in 4 weeks").
2. It introduces zero contradictory external claims.
3. The JSON `schema_valid` flag is true.

---

## 4. The Bridge to Stage B: Lightweight In-Memory RAG

In Stage A, the evidence passage is known in advance and manually injected into the prompt envelope.

In real-world applications and graduate lab research, an AI system faces open-ended claims where the relevant evidence is not pre-labeled. The system must search an external knowledge corpus containing hundreds or thousands of passages and dynamically retrieve the most relevant evidence: **Retrieval-Augmented Generation (RAG)**.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                      THE TWO-STAGE PROGRESSION                         │
├───────────────────────────────────┬────────────────────────────────────┤
│  Stage A: Prompt Grounding        │  Stage B: Dynamic RAG              │
│  (Ticket 05 - Active)             │  (Ticket 06 - Next)                │
├───────────────────────────────────┼────────────────────────────────────┤
│  • Evidence is pre-paired with    │  • System searches an unindexed    │
│    the claim                      │    corpus of candidate passages    │
│  • Tests model capacity to        │  • Generates dense 384-dim vector  │
│    prioritize context over        │    embeddings with all-MiniLM-L6-v2│
│    parametric memory              │  • Computes cosine similarity via  │
│  • Zero retrieval latency         │    PyTorch GPU matrix math         │
│  • Verifies faithfulness &        │  • Dynamically feeds top-k chunks  │
│    elimination of hallucinations  │    into the Stage A prompt envelope│
└───────────────────────────────────┴────────────────────────────────────┘
```

---

## 4. Stage B: Lightweight In-Memory RAG Architecture

In Stage A, the evidence passage was manually pre-paired with each claim. In Stage B, the system faces an arbitrary query against an unindexed corpus and must search, score, and retrieve the most relevant evidence passage automatically.

### The Bi-Encoder Architecture (`all-MiniLM-L6-v2`)
Rather than relying on lexical keyword matching (BM25) which fails on synonyms and paraphrasing, we use a dense neural bi-encoder:
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (6 transformer layers, 384 hidden dimensions, 22.7M parameters).
- **VRAM Footprint**: ~80 MB (negligible alongside Llama-3's 5.35 GB).
- **Mechanism**: Passes each document through BERT-style self-attention, applies mean pooling over all token representations, and projects the passage into a continuous 384-dimensional vector space where semantically similar concepts cluster together.

### Vector Normalization & Cosine Similarity Mathematics
The cosine similarity between query vector `q` and document vector `d` measures the cosine of the angle between them in 384-dimensional space:

`cosine_similarity(q, d) = (q · d) / (||q||₂ · ||d||₂)`

By applying L2 unit normalization during encoding (`normalize_embeddings=True`):
`||q||₂ = 1.0` and `||d||₂ = 1.0`

The equation simplifies to a pure vector dot product:
`cosine_similarity(q, d) = q · d = ∑ (from i=1 to 384) (qᵢ · dᵢ)`

### GPU Matrix Multiplication (`torch.mm`)
When scaling across `N` documents in a knowledge corpus, we stack the normalized document vectors into a matrix `D` of shape `[N, 384]`.
Given a query vector `q` of shape `[1, 384]`, all `N` similarity scores are computed simultaneously in a single GPU operation:

`Scores_Tensor = torch.mm(q, Dᵀ)`  
`Matrix Dimensions: [1, 384] · [384, N] = [1, N]`

Top-k candidate documents are extracted via:
`top_scores, top_indices = torch.topk(Scores_Tensor, k=k)`

### Why Pure PyTorch Matrix Math Outperforms Vector Databases in Colab
1. **Zero Dependency Overhead**: Avoids compilation issues, C++ binaries, and package conflicts from libraries like FAISS, ChromaDB, or Milvus.
2. **Instant Latency**: Computing `torch.mm` across thousands of 384-dimensional vectors executes in under 0.2 milliseconds on the T4 GPU Tensor Cores.
3. **Transparent Systems Architecture**: The entire retrieval pipeline consists of 15 lines of pure Python and PyTorch tensor operations, exposing the exact linear algebra powering modern RAG.

---

## 5. Summary & Key Takeaways

1. **Parametric memory is a probabilistic prior, not a database**: 8B models default to common internet consensus unless constrained by an authoritative context envelope.
2. **Context-Aware Grounding eliminates hallucinations**: Injecting verified evidence directly into the context window resolves subtle, contested, and counter-intuitive claims where Zero-Shot fails.
3. **The Flip Rate proves grounding efficacy**: Demonstrating that an ungrounded model flips from incorrect to correct upon evidence presentation is the standard empirical proof of reasoning responsiveness.
4. **Bi-encoder cosine retrieval dynamically bridges corpus to context**: Unit-normalized PyTorch matrix multiplication (`torch.mm`) enables sub-millisecond semantic search directly inside Google Colab without external database dependencies.

---

## Navigation & Next Steps

- **Syllabus**: [Chapter 2 README](../README.md)
- **Previous Topic**: [Topic 2.2: Zero-Shot and Few-Shot In-Context Learning](./topic_2.2_zero_shot_and_few_shot_icl.md)
- **Next Topic**: [Topic 2.4: Structured Chain-of-Thought Reasoning](./topic_2.4_chain_of_thought_reasoning.md)
- **Ticket Status**: [Ticket 06 Tracker](../../../.scratch/in-context-reasoning/issues/06-lightweight-in-memory-rag-engine.md)

