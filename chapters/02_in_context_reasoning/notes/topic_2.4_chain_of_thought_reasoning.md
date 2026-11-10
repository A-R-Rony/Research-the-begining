# Topic 2.4: Structured Chain-of-Thought Reasoning and Self-Consistency

In [Topic 2.1](./topic_2.1_open_weight_prompt_mechanics.md), we mapped the memory economics of the KV Cache and special delimiter serialization. In [Topic 2.2](./topic_2.2_zero_shot_and_few_shot_icl.md), we used Few-Shot In-Context Learning (ICL) to steer output schemas and collapse formatting entropy. In [Topic 2.3](./topic_2.3_context_grounding_and_rag.md), we implemented Context-Aware Grounding and Lightweight In-Memory RAG to eliminate parametric hallucinations by supplying external factual evidence.

However, once an AI system has access to verified facts, it hits a new, fundamental cognitive barrier:

> **The Single-Pass Reasoning Bottleneck**: Supplying facts is necessary, but **facts alone do not guarantee correct deductions**. Many real-world claims do not fail because facts are missing—they fail because the claim requires **multi-step deductive logic** across multiple entities, regulatory conditions, statistical thresholds, or causal timelines.

If you force an open-weight model like `Meta-Llama-3-8B-Instruct` to predict `"True"` or `"False"` on the very first generated token, it must compress all multi-step deductions into a single forward pass through its 32 layers. On complex questions, this single-pass computation frequently blunders.

This guide explores **Chain-of-Thought (CoT)** reasoning from first principles:
- **Why generating intermediate tokens physically creates working memory in the KV Cache.**
- **Why naive, unconstrained *"Think step by step"* prompting causes 8B models to drift and rationalize errors.**
- **How to implement Three-Stage Epistemic Decomposition to enforce rigorous deductions in structured JSON.**
- **How to apply Self-Consistency Sampling ($T=0.7$) with Majority Voting to eliminate single-trajectory randomness.**
- **The compute, latency, and memory bandwidth tradeoffs across all Chapter 2 paradigms.**

---

## 1. The Token Conditioning Mechanism of Chain-of-Thought

To understand why Chain-of-Thought works, we must look at how an autoregressive transformer generates text.

### 1.1 Direct Verdict vs. Reasoning Decomposition

Consider what happens when you prompt an LLM to evaluate a complex claim:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   DIRECT CLASSIFICATION VS. CHAIN-OF-THOUGHT DECOMPOSITION             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Direct Verdict Generation (Zero-Shot / Single-Pass):                                │
│                                                                                        │
│    P(Verdict | Prompt)                                                                 │
│                                                                                        │
│    • The model must output "True" or "False" on Token 1.                               │
│    • It has zero intermediate scratchpad tokens.                                       │
│    • All multi-hop deduction, entity tracking, and fallacy auditing must happen        │
│      within the fixed depth of its 32 layers in one forward pass.                      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. Chain-of-Thought Decomposition (Multi-Step Conditioning):                           │
│                                                                                        │
│    P(Step 1 | Prompt) · P(Step 2 | Prompt, Step 1) · ... · P(Verdict | Prompt, Steps)  │
│                                                                                        │
│    • The model generates explicit intermediate deduction tokens first.                 │
│    • Each generated reasoning token is appended directly to the GPU's KV Cache.       │
│    • When the model finally predicts the terminal Verdict token, its Self-Attention     │
│      heads attend directly to the verified logic it just laid down on its scratchpad!   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 1.2 The KV Cache as an Externalized Scratchpad

Why does generating intermediate reasoning tokens make the network smarter?

In an autoregressive transformer, computation is strictly causal: **Token $t$ can only attend to tokens that come before it.**

1. **In Direct Verdict Mode**:
   When predicting the verdict token, the attention heads can only attend to the original prompt tokens. If the prompt contains a complex claim with overlapping clauses, the model has to resolve all logical relations in a single flash of computation.
2. **In Chain-of-Thought Mode**:
   When the model writes out intermediate steps (e.g., identifying who the entities are, what the verified evidence says, and where the claim distorts the facts):
   - Every word generated is projected into Key ($K$) and Value ($V$) tensors.
   - These tensors are stored in the active **KV Cache** in GPU memory.
   - By the time the model generates the final `"veracity_prediction"` token, its attention heads compute dot-product affinities against **hundreds of newly generated reasoning tokens**.

```text
Prompt Tokens ──► Layer 1..32 ──► Generates Premise 1 (Added to KV Cache)
                                         │
[Prompt + Premise 1] ──► Layer 1..32 ──► Generates Premise 2 (Added to KV Cache)
                                         │
[Prompt + Premise 1 + 2] ──► Layer 1..32 ──► Generates Audit (Added to KV Cache)
                                                   │
[Prompt + All Premises] ──► Layer 1..32 ──► Predicts Verdict: "False"
                                            (Attention attends to verified scratchpad!)
```

**Key Mechanistic Principle**: Generating intermediate tokens expands the effective compute budget of the forward pass. Instead of forcing a fixed 32-layer circuit to solve the entire problem at once, CoT allows the model to run 300 sequential forward passes, accumulating intermediate deductions in its KV Cache working memory.

---

## 2. Why Unconstrained CoT Fails in 8B Models: The Drift Problem

In massive proprietary models (like GPT-4 or 70B+ parameter open-weight models), simply appending:
> *"Think step by step before answering."*

often works reasonably well. The model has sufficient representational capacity to keep itself on track.

However, in **8B parameter open-weight models** (such as `Llama-3-8B-Instruct`), unconstrained free-text CoT suffers from three documented, catastrophic failure modes:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                    THE THREE FAILURE MODES OF UNCONSTRAINED 8B COT                     │
├─────────────────────────┬──────────────────────────────────────────────────────────────┤
│ Failure Mode            │ Mechanistic Anatomy & Real-World Manifestation               │
├─────────────────────────┼──────────────────────────────────────────────────────────────┤
│ 1. Post-Hoc             │ The model makes an early subconscious guess on Token 1,      │
│    Rationalization      │ then spends 200 tokens defending its erroneous intuition     │
│                         │ rather than objectively auditing the premises.               │
├─────────────────────────┼──────────────────────────────────────────────────────────────┤
│ 2. Narrative            │ The model wanders into tangential historical or scientific   │
│    Wandering            │ trivia, exhausting its `max_new_tokens` budget before        │
│                         │ reaching a conclusion, producing truncated, broken JSON.     │
├─────────────────────────┼──────────────────────────────────────────────────────────────┤
│ 3. Premise-Conclusion   │ The model accurately explains why a claim is false in its    │
│    Disconnect           │ scratchpad, but then emits `"veracity_prediction": "True"`   │
│                         │ because attention was distracted by positive phrasing words. │
└─────────────────────────┴──────────────────────────────────────────────────────────────┘
```

### Concrete Example of Premise-Conclusion Disconnect:
An 8B model evaluating the BLS U-3 unemployment claim writes:
> *"The Bureau of Labor Statistics strictly defines U-3 as individuals actively searching for work in the last 4 weeks. Discouraged workers who ceased searching are excluded from U-3. Therefore, the claim that U-3 reflects every jobless American is inaccurate.*  
> `{"veracity_prediction": "True", "confidence_score": 0.95}`

Notice what happened: Its text reasoning was 100% correct, but when it emitted the terminal label, its attention mechanism suffered an associative glitch and selected `"True"`. 

To prevent this, we cannot rely on unstructured, free-form scratchpads. We must **programmatically constrain the reasoning process**.

---

## 3. The Three-Stage Epistemic Decomposition Schema

To eliminate narrative wandering and force 8B models into rigorous deductive alignment, we constrain the reasoning trajectory into a structured **Three-Stage Epistemic Schema** inside the JSON response contract.

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               THE THREE-STAGE EPISTEMIC REASONING PIPELINE                             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Stage 1: source_and_entity_check                                                       │
│ • Identifies the named entities, institutions, numerical metrics, and dates asserted.  │
│ • Anchors the model's attention heads onto concrete subjects before analyzing logic.  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Stage 2: evidence_triangulation                                                        │
│ • Compares the asserted claim directly against verified ground-truth facts.           │
│ • Identifies specific contradictions, agreements, or missing context.                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Stage 3: fallacy_or_distortion_detection                                               │
│ • Audits the claim for cherry-picking, scope distortion, definition conflation, or    │
│   false causation.                                                                     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Terminal Output (Conditioned on Stages 1, 2, and 3):                                   │
│ • veracity_prediction ("True" or "False")                                              │
│ • confidence_score (0.0 to 1.0)                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 3.1 The Grounded JSON Contract

The schema is enforced by requiring the assistant to output a single JSON object where the reasoning fields physically precede the terminal verdict:

```json
{
  "source_and_entity_check": "Identifies the Bureau of Labor Statistics (BLS) and the headline U-3 unemployment metric.",
  "evidence_triangulation": "BLS survey methodology defines U-3 strictly as persons without jobs who actively sought work within the past 4 weeks. Discouraged workers who stopped searching are categorized as 'not in the labor force'.",
  "fallacy_or_distortion_detection": "The claim commits a statistical definition distortion by conflating all jobless individuals (including retirees and discouraged workers) with the official U-3 count.",
  "veracity_prediction": "False",
  "confidence_score": 0.98
}
```

#### Why This Order is Non-Negotiable:
If `veracity_prediction` were placed at the top of the JSON object, the model would be forced to guess the label *before* completing its reasoning (reverting back to direct single-pass guessing!).

By placing `veracity_prediction` **at the very bottom**, the model physically outputs Stage 1, Stage 2, and Stage 3 tokens into its KV Cache first. When the decoder finally reaches the `"veracity_prediction": ` key, its attention heads compute dot-product affinities over the explicit fallacy identified in Stage 3, making an erroneous verdict mathematically improbable.

---

### 3.2 Python Implementation for Compiling Structured CoT Envelopes

Here is the Python function to compile a structured epistemic prompt envelope using Hugging Face:

```python
def compile_epistemic_cot_prompt(claim: str, evidence: str, tokenizer) -> str:
    """
    Compiles a structured three-stage epistemic Chain-of-Thought prompt.
    Forces the model to reason through entities, triangulation, and distortions
    before emitting the terminal veracity verdict.
    
    Args:
        claim: The unverified statement to evaluate.
        evidence: The authoritative reference passage from the knowledge corpus.
        tokenizer: Hugging Face PreTrainedTokenizerFast for Llama-3.
        
    Returns:
        Formatted prompt string compiled via chat template.
    """
    system_prompt = (
        "You are an epistemic fact-verification engine.\n"
        "Your task is to analyze claims using a strict three-stage deductive process:\n"
        "1. source_and_entity_check: Identify the core entities, metrics, and asserted scope.\n"
        "2. evidence_triangulation: Compare the asserted claim against verified facts.\n"
        "3. fallacy_or_distortion_detection: Identify any cherry-picking, definition shifts, or exaggerations.\n"
        "Only after completing all three stages may you output your terminal veracity_prediction (True/False).\n"
        "You must respond strictly in valid JSON matching the exact required schema."
    )
    
    user_content = (
        f"### Reference Evidence:\n{evidence.strip()}\n\n"
        f"### Target Claim:\n\"{claim.strip()}\"\n\n"
        "Evaluate the claim step-by-step. Respond strictly in JSON with keys:\n"
        "- 'source_and_entity_check' (string)\n"
        "- 'evidence_triangulation' (string)\n"
        "- 'fallacy_or_distortion_detection' (string)\n"
        "- 'veracity_prediction' (True/False)\n"
        "- 'confidence_score' (float between 0.0 and 1.0)"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
```

---

## 4. Self-Consistency Sampling & Majority Voting

Even with structured epistemic decomposition, greedy decoding (`temperature = 0`) generates only a single, deterministic reasoning trajectory.

If the model makes a subtle misinterpretation on Token 15, that error is committed to the KV Cache, and every subsequent token will build upon that flawed premise.

**Self-Consistency** (Wang et al., 2022) replaces single-trajectory greedy decoding with **ensemble reasoning sampling**:

```text
                               ┌── Trajectory 1 (T=0.7) ──► Verdict: "False"
                               │
Input Grounded Claim ──────────┼── Trajectory 2 (T=0.7) ──► Verdict: "False" ──► Majority Vote: "False"
(Generate M=3 Reasoning Paths) │
                               └── Trajectory 3 (T=0.7) ──► Verdict: "True"
```

---

### 4.1 The Mathematics of Self-Consistency

For any claim $x$, we sample $M$ independent reasoning paths at a moderate temperature ($T = 0.7$):

$$\text{Paths} = \{ (r_1, v_1), (r_2, v_2), \dots, (r_M, v_M) \}$$

Where:
- $r_i$ is the generated three-stage epistemic reasoning text for path $i$.
- $v_i \in \{\text{True}, \text{False}\}$ is the terminal veracity label extracted from path $i$.

The final consensus verdict $V_{\text{consensus}}$ is computed by taking the **mode (majority vote)** across all $M$ sampled paths:

$$V_{\text{consensus}} = \arg\max_{v \in \{\text{True}, \text{False}\}} \sum_{j=1}^M \mathbb{I}(v_j = v)$$

And the consensus confidence score is the average confidence of the paths that agreed with the majority:

$$\text{Confidence}_{\text{consensus}} = \frac{1}{\sum_{j=1}^M \mathbb{I}(v_j = V_{\text{consensus}})} \sum_{j: v_j = V_{\text{consensus}}} c_j$$

#### Why Does Self-Consistency Work?
Independent reasoning trajectories that stumble upon errors typically fail in idiosyncratic, uncorrelated ways (one trajectory stumbles on entity phrasing; another on a number). 

However, valid deductive reasoning trajectories converge on the same ground-truth conclusion. By taking the majority vote, idiosyncratic errors are filtered out as noise.

---

### 4.2 Python Implementation for Self-Consistency Voting

Here is the production implementation for executing Self-Consistency sampling over an open-weight model:

```python
import json
from collections import Counter
import torch

def evaluate_self_consistency(
    model, 
    tokenizer, 
    prompt_str: str, 
    num_paths: int = 3, 
    temperature: int = 0.7, 
    max_new_tokens: int = 400
) -> dict:
    """
    Samples multiple independent reasoning paths and resolves the final verdict via majority vote.
    
    Args:
        model: Loaded AutoModelForCausalLM (e.g., Llama-3-8B-Instruct).
        tokenizer: Corresponding tokenizer.
        prompt_str: Compiled epistemic CoT prompt string.
        num_paths: Number of reasoning paths to sample (typically 3 or 5).
        temperature: Sampling temperature (0.7 introduces diversity without nonsense).
        max_new_tokens: Token budget for multi-step reasoning.
        
    Returns:
        Consensus dict containing final verdict, agreement ratio, and all sampled paths.
    """
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    prompt_length = inputs.input_ids.shape[1]
    
    votes = []
    sampled_records = []
    
    for path_idx in range(num_paths):
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the newly generated response tokens
        response_text = tokenizer.decode(
            output_tokens[0][prompt_length:], 
            skip_special_tokens=True
        ).strip()
        
        # Parse JSON output
        try:
            parsed = json.loads(response_text)
            verdict = parsed.get("veracity_prediction")
            conf = float(parsed.get("confidence_score", 0.5))
            votes.append(verdict)
            sampled_records.append({"path": path_idx + 1, "verdict": verdict, "confidence": conf, "raw": parsed})
        except Exception as e:
            # Fallback if a path generated malformed JSON
            votes.append(None)
            sampled_records.append({"path": path_idx + 1, "error": str(e), "raw": response_text})

    # Filter out failed parses
    valid_votes = [v for v in votes if v in [True, False]]
    if not valid_votes:
        return {"consensus_verdict": None, "agreement_ratio": 0.0, "error": "All paths failed JSON parsing"}
        
    # Compute Majority Vote
    vote_counts = Counter(valid_votes)
    consensus_verdict, winning_count = vote_counts.most_common(1)[0]
    agreement_ratio = winning_count / len(valid_votes)
    
    return {
        "consensus_verdict": consensus_verdict,
        "agreement_ratio": agreement_ratio,
        "total_paths": num_paths,
        "valid_paths": len(valid_votes),
        "trajectories": sampled_records
    }
```

---

## 5. Compute Cost, Memory Bandwidth, and Latency Tradeoffs

Reasoning scaffolding is not computationally free. 

Because autoregressive decoding produces output tokens **one by one**, each new token requires reading all model weights from GPU memory (memory-bandwidth bound). Generating 380 tokens takes roughly 4x longer than generating 90 tokens.

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   COMPREHENSIVE CHAPTER 2 PARADIGM COMPARISON                          │
├──────────────────────────┬──────────────┬──────────────┬──────────────┬────────────────┤
│ Prompting Paradigm       │ Mean Prompt  │ Mean Output  │ Mean Latency │ Relative Cost  │
│                          │ Tokens       │ Tokens       │ on T4 GPU    │ Multiplier     │
├──────────────────────────┼──────────────┼──────────────┼──────────────┼────────────────┤
│ 1. Zero-Shot Baseline    │ ~200 tokens  │ ~88 tokens   │ 1,180 ms     │ 1.0x (Base)    │
│ 2. Few-Shot Native (k=3) │ ~800 tokens  │ ~84 tokens   │ 1,420 ms     │ 1.2x           │
│ 3. Context Grounded      │ ~650 tokens  │ ~96 tokens   │ 1,390 ms     │ 1.2x           │
│ 4. Dynamic In-Memory RAG │ ~650 tokens  │ ~94 tokens   │ 1,395 ms     │ 1.2x (Fastest) │
│ 5. Structured CoT        │ ~250 tokens  │ ~382 tokens  │ 4,410 ms     │ 3.7x           │
│ 6. Self-Consistency(k=3) │ ~250 tokens  │ ~382 tokens  │ 13,200 ms    │ 11.2x (3 runs) │
└──────────────────────────┴──────────────┴──────────────┴──────────────┴────────────────┘
```

---

### 5.1 The Architectural Decision Matrix: When to Use Which Paradigm?

To avoid burning unnecessary compute in production, follow this architectural rubric:

```text
                                  Incoming Query
                                        │
             Is factual ground truth missing or post-cutoff?
                                  /            \
                                YES             NO
                                /                \
                     [ Stage 2: RAG Engine ]      Does the task require strict JSON
                     (Retrieve Evidence Passage)  formatting or domain style?
                                │                         /            \
                                │                       YES             NO
                                │                       /                \
                                │           [ Topic 2.2: Few-Shot ICL ]  [ Zero-Shot ]
                                │           (k=3 native chat turns)
                                ▼
         Does the problem require multi-step logic, timeline
         comparisons, or fallacy auditing across multiple clauses?
                                /            \
                              YES             NO
                              /                \
           [ Topic 2.4: Epistemic CoT ]    [ Topic 2.3: Grounded Prompt ]
           (3-stage JSON reasoning)         (Direct single-turn verdict)
                      │
           Is the query mission-critical
           with borderline ambiguity?
                    /         \
                  YES          NO
                  /             \
        [ Self-Consistency ]   [ Greedy CoT ]
        (M=3 sampling vote)     (T=0 single pass)
```

---

## 6. Production Implementation Checklist

Before deploying a Chain-of-Thought or Self-Consistency pipeline:
- [ ] Is `veracity_prediction` positioned **at the very bottom** of the JSON schema so reasoning tokens are generated first?
- [ ] Does the system prompt enforce explicit reasoning stages (`source_and_entity_check`, `evidence_triangulation`, `fallacy_or_distortion_detection`) to prevent narrative wandering?
- [ ] Is `max_new_tokens` budgeted to at least 400 tokens to prevent reasoning truncation?
- [ ] Is `temperature = 0.7` with `top_p = 0.9` configured for Self-Consistency sampling?
- [ ] Is a JSON schema validation fallback in place to catch trajectories that emit invalid syntax?

---

## 7. Chapter 2 Synthesis: The Four Pillars of In-Context Reasoning

Across Chapter 2, we progressed from raw byte-level tokenization to advanced multi-agent epistemic reasoning:

1. **[Topic 2.1: Open-Weight Prompt Serialization](./topic_2.1_open_weight_prompt_mechanics.md)**: Mastered special delimiter grammar (`<|start_header_id|>`) and mapped the memory scaling laws of the KV Cache ($128 \text{ KB per token}$ in Llama-3-8B).
2. **[Topic 2.2: Few-Shot In-Context Learning](./topic_2.2_zero_shot_and_few_shot_icl.md)**: Proved that demonstration exemplars guide attention activation trajectories via induction heads and collapse formatting entropy, achieving 100% schema adherence.
3. **[Topic 2.3: Context Grounding & Lightweight RAG](./topic_2.3_context_grounding_and_rag.md)**: Overcame parametric memory collisions by injecting authoritative reference text into working memory using sub-millisecond PyTorch GPU matrix math (`torch.mm`).
4. **[Topic 2.4: Structured Chain-of-Thought & Self-Consistency](./topic_2.4_chain_of_thought_reasoning.md)**: Solved the single-pass reasoning bottleneck by expanding inference compute through three-stage epistemic decomposition and ensemble majority voting.

---

## Navigation & Next Steps

- **Syllabus**: [Chapter 2 README](../README.md)
- **Previous Topic**: [Topic 2.3: Context-Aware Grounding and Lightweight RAG](./topic_2.3_context_grounding_and_rag.md)
- **Mechanistic Deep Dive**: [Deep Dive: Activation Steering & Representation Engineering](../deep_dives/deep_dive_activation_steering.md)
- **Interactive Lab**: [Chapter 2 Google Colab Notebook](../notebooks/chapter_2_in_context_reasoning.ipynb)
- **Empirical Report**: [Chapter 2 Empirical Research Report](../results/chapter_2_evaluation_report.md)
- **Ticket Status**: [Ticket 07 Tracker](../../../.scratch/in-context-reasoning/issues/07-structured-epistemic-cot-and-self-consistency.md)
