# Topic 2.4: Structured Chain-of-Thought Reasoning and Self-Consistency

In Topic 2.1, we mapped the memory economics of the KV cache. In Topic 2.2, we used Few-Shot In-Context Learning (ICL) to steer activation trajectories without modifying model weights. In Topic 2.3, we used Context-Aware Grounding and Lightweight RAG to override parametric hallucinations.

However, many nuanced veracity questions do not fail because of missing facts alone—they fail because the model must perform **multi-step deductive reasoning** over multiple entities, statistical scopes, or causal claims.

This guide explores **Chain-of-Thought (CoT)** reasoning: why generating intermediate tokens fundamentally alters autoregressive probabilities, why unconstrained CoT produces narrative drift in 8B models, and how to implement **Three-Stage Epistemic Decomposition** and **Self-Consistency Sampling**.

---

## 1. The Token Conditioning Mechanism of Chain-of-Thought

In causal transformer decoders, self-attention is strictly autoregressive: each generated token can only attend to preceding tokens in the sequence.

```text
┌────────────────────────────────────────────────────────────────────────┐
│             AUTOREGRESSIVE TOKEN PROBABILITY CONDITIONING              │
├────────────────────────────────────────────────────────────────────────┤
│  Direct Zero-Shot Verdict:                                             │
│  P(Verdict | Prompt)                                                   │
│  → The model must predict "True" or "False" in a single forward pass.  │
│  → It has no working memory scratchpad; it must compute all complex    │
│    deductions inside its internal layer residual stream at position 0. │
├────────────────────────────────────────────────────────────────────────┤
│  Chain-of-Thought Decomposition:                                       │
│  P(Step 1 | Prompt) · P(Step 2 | Prompt, Step 1) ·                     │
│  ... · P(Verdict | Prompt, Step 1, Step 2, ...)                        │
│  → The model articulates intermediate premises as explicit tokens.     │
│  → When predicting the final verdict token, its attention heads attend │
│    directly to the intermediate logic it just wrote in its KV cache!   │
└────────────────────────────────────────────────────────────────────────┘
```

Generating intermediate reasoning tokens expands the effective compute dedicated to an inference query. Instead of compressing a complex multi-hop deduction into a single token transition, the model distributes the deduction across hundreds of tokens stored in the KV cache.

---

## 2. Why Unconstrained CoT Fails in 8B Models: The Drift Problem

In massive frontier models (70B+ parameters), prompting the model with a simple instruction like *"Think step by step"* often produces coherent free-text reasoning.

In open-weight 8B models (such as `Llama-3-8B-Instruct`), unconstrained free-text CoT suffers from three documented failure modes:

1. **Post-Hoc Rationalization**: If unconstrained, the model frequently defaults to its intuitive associative guess on token 1, using subsequent reasoning tokens merely to justify its error rather than critically stress-testing the claim.
2. **Narrative Wandering**: The model wanders into tangential historical trivia, exhausting its `max_new_tokens` budget before reaching a conclusion, triggering JSON truncation errors.
3. **Premise-Conclusion Disconnect**: The model accurately articulates why a claim is false in its scratchpad, but then inexplicably emits `"veracity_prediction": "True"` at the end because the attention weights over-attended to positive sentiment words in the prompt.

---

## 3. The Three-Stage Epistemic Decomposition Schema

To eliminate narrative drift and ensure logical rigor in 8B parameter models, we do not use unstructured scratchpads. Instead, we constrain the reasoning into a strict **Three-Stage Epistemic Schema** inside the JSON response contract:

```text
┌────────────────────────────────────────────────────────────────────────┐
│               THREE-STAGE EPISTEMIC REASONING PIPELINE                 │
├────────────────────────────────────────────────────────────────────────┤
│  Stage 1: source_and_entity_check                                      │
│  • Identifies the named entities, organizations, numerical statistics, │
│    and temporal timeline asserted in the claim.                        │
├────────────────────────────────────────────────────────────────────────┤
│  Stage 2: evidence_triangulation                                       │
│  • Evaluates whether verified external facts or scientific principles  │
│    corroborate, contradict, or qualify the core assertion.             │
├────────────────────────────────────────────────────────────────────────┤
│  Stage 3: fallacy_or_distortion_detection                             │
│  • Audits for cherry-picking, missing context, out-of-context quotes,  │
│    exaggerated causation, or statistical definition shifts.            │
├────────────────────────────────────────────────────────────────────────┤
│  Terminal Output:                                                      │
│  • veracity_prediction ("True" or "False")                             │
│  • confidence_score (0.0 to 1.0)                                       │
└────────────────────────────────────────────────────────────────────────┘
```

### JSON Schema Contract
```json
{
  "source_and_entity_check": "Identifies Bureau of Labor Statistics (BLS) and the U-3 unemployment metric.",
  "evidence_triangulation": "BLS survey methodology defines U-3 strictly as individuals without a job who are actively seeking work within the past 4 weeks.",
  "fallacy_or_distortion_detection": "The claim distorts the definition by conflating all jobless adults (which includes retirees, students, and discouraged workers) with the official U-3 count.",
  "veracity_prediction": "False",
  "confidence_score": 0.96
}
```

By forcing the model to complete Stage 1, Stage 2, and Stage 3 *before* emitting the `veracity_prediction` token, the final classification attends directly to the identified distortion.

---

## 4. Self-Consistency Sampling & Majority Voting

Greedy decoding (`temperature = 0`) generates only a single, deterministic reasoning trajectory. If the model makes an early misstep on token 15, the remaining 300 tokens will compound that error.

**Self-Consistency** (Wang et al., 2022) replaces greedy decoding with ensemble sampling:

```text
                       ┌── Trajectory 1 (T=0.7) ──► Verdict: "False"
                       │
Input Query ───────────┼── Trajectory 2 (T=0.7) ──► Verdict: "False" ──► Majority Vote: "False"
(3 Reasoning Paths)    │
                       └── Trajectory 3 (T=0.7) ──► Verdict: "True"
```

### Self-Consistency Mathematics
For a claim `x`, we generate `M` independent reasoning paths at non-zero temperature (`temperature = 0.7`):
`Paths = { (r₁, v₁), (r₂, v₂), ..., (r_M, v_M) }`
Where `r_i` is the epistemic reasoning chain and `v_i` is the terminal veracity label (`True` or `False`).

The final consensus verdict `V_consensus` is computed via majority voting:
`V_consensus = argmax_v ∑ (from j=1 to M) [ 1 if v_j == v else 0 ]`

Self-Consistency dramatically stabilizes reasoning on ambiguous or borderline claims where a single reasoning path might stumble on phrasing nuances.

---

## 5. Compute Cost & Latency Tradeoffs

Reasoning scaffolding is not computationally free:

| Prompting Paradigm | Mean Prompt Tokens | Mean Generated Tokens | Inferences per Query | Relative Compute Multiplier |
| :--- | :--- | :--- | :--- | :--- |
| **Zero-Shot Baseline** | ~200 tokens | ~90 tokens | 1 | **1.0x (Baseline)** |
| **Few-Shot Native (k=3)**| ~800 tokens | ~85 tokens | 1 | **1.4x (Higher prefill)** |
| **Context Grounding / RAG**| ~650 tokens | ~95 tokens | 1 | **1.3x** |
| **Structured Epistemic CoT**| ~250 tokens | ~380 tokens | 1 | **3.8x (Higher decode time)** |
| **Self-Consistency CoT (k=3)**| ~250 tokens | ~380 tokens | 3 | **11.4x (3 full decodes)** |

Autoregressive token generation is memory-bandwidth bound: generating 380 tokens takes roughly 4x longer than generating 90 tokens. Therefore, in production architectures, CoT and Self-Consistency should be reserved for complex, contested, or high-stakes claims where zero-shot confidence is low.

---

## 6. Summary & Key Takeaways

1. **CoT works through autoregressive token conditioning**: Generated reasoning tokens act as working memory in the KV cache, shaping the probability distribution of the terminal verdict token.
2. **Unconstrained CoT wanders in 8B models**: Open-weight 8B models require strict schema boundaries (Stage 1 entity check → Stage 2 triangulation → Stage 3 distortion analysis) to prevent rationalization and drift.
3. **Self-Consistency eliminates single-path variance**: Sampling multiple reasoning paths at `temperature = 0.7` and aggregating via majority vote resolves ambiguous edge cases.
4. **CoT trades compute for accuracy**: Epistemic CoT requires 4x more generation tokens than Zero-Shot, making it an architectural decision based on accuracy requirements versus latency budgets.

---

## Navigation & Next Steps

- **Syllabus**: [Chapter 2 README](../README.md)
- **Previous Topic**: [Topic 2.3: Context-Aware Grounding and Lightweight RAG](./topic_2.3_context_grounding_and_rag.md)
- **Unified Benchmark**: [Chapter 2 Colab Notebook](../notebooks/chapter_2_in_context_reasoning.ipynb)
- **Ticket Status**: [Ticket 07 Tracker](../../../.scratch/in-context-reasoning/issues/07-structured-epistemic-cot-and-self-consistency.md)
