# Topic 2.2: Zero-Shot and Few-Shot In-Context Learning: Activation Steering vs. Monolithic Prompting

In Topic 2.1, we explored how open-weight tokenizers serialize strings into discrete token IDs and compile chat templates with specialized delimiters. In Ticket 03, we established the Zero-Shot baseline on `meta-llama/Meta-Llama-3-8B-Instruct`, discovering that cold parametric memory achieves 100 percent on widely debunked historical conspiracies but drops to roughly 60 percent on subtle, nuanced claims.

This guide explores **In-Context Learning (ICL)** from mechanistic foundations to practical open-weight execution: how demonstration exemplars steer attention activation trajectories without parameter updates, and why **native multi-turn chat templates** systematically outperform **monolithic in-prompt packing**.

---

## 1. The Fundamental Distinction: Weight Updates vs. Activation Steering

When practitioners want an LLM to follow a specific output schema or learn a specialized domain task, they typically consider two techniques:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   HOW A MODEL "LEARNS" A TASK                          │
├───────────────────────────────────┬────────────────────────────────────┤
│  Supervised Fine-Tuning (SFT)     │  In-Context Learning (ICL)         │
├───────────────────────────────────┼────────────────────────────────────┤
│  • Modifies network weights       │  • Network weights remain 100%     │
│    W_new = W_old - lr · ∇Loss     │    frozen (no backpropagation)     │
│  • Requires backward pass compute │  • Runs strictly during a forward  │
│    and optimizer memory (AdamW)   │    inference pass                  │
│  • Permanent change to model      │  • Temporary steering: vanishes    │
│    parameters across sessions     │    when the context window clears  │
│  • Risk of catastrophic forgetting│  • Zero risk of forgetting         │
└───────────────────────────────────┴────────────────────────────────────┘
```

### How In-Context Learning Works Mechanistically
In ICL, no gradient descent occurs. How, then, does the model "learn" from demonstrations?
Mechanistic interpretability research (e.g., Anthropic's induction head research and Dai et al.'s implicit meta-optimization analysis) reveals two primary mechanisms:

1. **Induction Heads**: Two-layer attention circuits that search earlier sequence positions for pattern completions. If the prompt contains a demonstration `[Entity A -> Property B]`, induction heads attend back to the relationship and copy the transition structure when encountering `[Entity X -> ?]`.
2. **Entropy Reduction on Syntax**: In zero-shot prompting, the model's probability distribution over the first generated token is dispersed across natural conversational words ("Based on...", "Sure, here...", "The claim..."). In Few-Shot ICL, observing k instances where the assistant begins immediately with an open curly brace `{` collapses token entropy for syntax to near zero, guaranteeing immediate structured JSON output.

---

## 2. Formatting Mechanics: Native Multi-Turn vs. Monolithic Packing

In closed-source APIs, users frequently concatenate all demonstration examples into a single massive string inside the user prompt. In instruction-tuned open-weight models, this practice introduces severe formatting degradation.

### Paradigm A: Native Multi-Turn Chat Template Turns (Recommended)
Each demonstration exemplar is modeled as a genuine prior conversational turn in the messages list:

```python
messages = [
    {"role": "system", "content": "You are a fact-checking assistant. Output JSON only."},
    
    # Exemplar 1: Simulated User & Assistant Turn
    {"role": "user", "content": "Evaluate claim: Water boils at 71C on Mount Everest."},
    {"role": "assistant", "content": '{"veracity_prediction": "True", "confidence_score": 0.99, "reasoning": "Atmospheric pressure is lower at high altitude."}'},
    
    # Exemplar 2: Simulated User & Assistant Turn
    {"role": "user", "content": "Evaluate claim: 5G radiation ionizes DNA."},
    {"role": "assistant", "content": '{"veracity_prediction": "False", "confidence_score": 0.99, "reasoning": "5G uses non-ionizing radio frequencies."}'},
    
    # Target Query
    {"role": "user", "content": "Evaluate claim: Electric vehicles emit more lifetime GHG than gas cars."}
]
```

When compiled through `tokenizer.apply_chat_template`, this produces:
```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a fact-checking assistant. Output JSON only.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Evaluate claim: Water boils at 71C on Mount Everest.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{"veracity_prediction": "True", "confidence_score": 0.99, "reasoning": "Atmospheric pressure is lower at high altitude."}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Evaluate claim: 5G radiation ionizes DNA.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{"veracity_prediction": "False", "confidence_score": 0.99, "reasoning": "5G uses non-ionizing radio frequencies."}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Evaluate claim: Electric vehicles emit more lifetime GHG than gas cars.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

#### Why Native Multi-Turn Succeeds:
During Post-Training (Supervised Fine-Tuning and Direct Preference Optimization), Llama-3 was explicitly optimized to treat text encapsulated between `<|start_header_id|>assistant<|end_header_id|>` and `<|eot_id|>` as authoritative target responses. Aligning few-shot exemplars with this native token grammar activates the model's dialogue conditioning directly.

---

### Paradigm B: Monolithic In-Prompt Packing
All demonstration exemplars are bundled into a single user message string:

```text
<|start_header_id|>user<|end_header_id|>
Here are some examples of claim verification:

Example 1:
Claim: Water boils at 71C on Mount Everest.
Output: {"veracity_prediction": "True", "confidence_score": 0.99, "reasoning": "Atmospheric pressure is lower at high altitude."}

Example 2:
Claim: 5G radiation ionizes DNA.
Output: {"veracity_prediction": "False", "confidence_score": 0.99, "reasoning": "5G uses non-ionizing radio frequencies."}

Now evaluate this claim:
Claim: Electric vehicles emit more lifetime GHG than gas cars.
Output:<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

#### Common Failure Modes of Monolithic Packing in 8B Models:
1. **Echoing & Pattern Over-Extension**: The model may generate `Example 3:` or invent fictional future user queries instead of terminating with `<|eot_id|>`.
2. **Header Confusion**: Because all examples live inside the user header, the model cannot distinguish between instructions provided by the user versus verified historical completions provided by the assistant.
3. **Format Drift**: The model is prone to inserting conversational pleasantries ("Here is the output for Example 3:") before the JSON string.

---

## 3. The Scaling Dynamics of Exemplar Count (k = 1, 3, 5)

How many exemplars should you provide in an open-weight context window?

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   EXEMPLAR SCALING TRADEOFF CURVE                      │
├───────┬─────────────────┬─────────────────┬────────────────────────────┤
│  k    │ Token Overhead  │ Primary Benefit │ Diminishing Returns        │
├───────┼─────────────────┼─────────────────┼────────────────────────────┤
│ k = 1 │ ~250 tokens     │ Schema Priming  │ Vulnerable to class bias   │
│       │                 │                 │ (only shows True OR False) │
├───────┼─────────────────┼─────────────────┼────────────────────────────┤
│ k = 3 │ ~750 tokens     │ Label Balancing │ Strong performance /       │
│       │                 │ & Diversity     │ compute sweet spot         │
├───────┼─────────────────┼─────────────────┼────────────────────────────┤
│ k = 5 │ ~1,250 tokens   │ Robust Boundary │ Higher prefill latency;    │
│       │                 │ Modeling        │ minimal accuracy gain over │
│       │                 │                 │ k = 3 without new context  │
└───────┴─────────────────┴─────────────────┴────────────────────────────┘
```

### Token Cost and Prefill Latency Math
Each exemplar in our benchmark suite consumes approximately 250 tokens:
- At `k = 1`: Prompt overhead is ~250 tokens. Prefill latency increases by ~25 ms.
- At `k = 3`: Prompt overhead is ~750 tokens. Prefill latency increases by ~70 ms.
- At `k = 5`: Prompt overhead is ~1,250 tokens. Prefill latency increases by ~120 ms.

Because `meta-llama/Meta-Llama-3-8B-Instruct` uses Grouped-Query Attention (GQA), the KV cache memory overhead at `k = 5` is only:
`Memory = 1,250 tokens · 128 KB/token ≈ 160 MB`
Memory footprint is negligible on the 15 GB Colab T4 GPU, meaning choice of k is governed by **prefill latency and reasoning diversity**, not VRAM limits.

---

## 4. Key Takeaways for Open-Weight Practitioners

1. **ICL is activation routing, not learning**: It does not change weights; it conditions attention heads to mimic demonstrated input-output transitions.
2. **Never concatenate exemplars into raw user strings**: Always compile few-shot demonstrations through `tokenizer.apply_chat_template` using alternating `user` and `assistant` dictionaries.
3. **Always balance exemplar classes**: If providing k exemplars, ensure balanced representation of True and False verdicts to prevent class-frequency prediction bias.
4. **ICL stabilizes format, but cannot invent missing facts**: If a model does not possess factual domain knowledge about a subtle claim in its parametric memory, Few-Shot ICL helps it output valid JSON with confident reasoning, but does **not** cure factual hallucinations. Curing hallucinations requires **Context-Aware Grounding (Topic 2.3)**.

---

## Navigation & Next Steps

- **Syllabus**: [Chapter 2 README](../README.md)
- **Previous Topic**: [Topic 2.1: Open-Weight Prompt Serialization](./topic_2.1_open_weight_prompt_mechanics.md)
- **Next Topic**: [Topic 2.3: Context-Aware Grounding and Lightweight RAG](./topic_2.3_context_grounding_and_rag.md)
- **Ticket Status**: [Ticket 04 Tracker](../../../.scratch/in-context-reasoning/issues/04-few-shot-icl-native-turns-vs-monolithic-packing.md)
