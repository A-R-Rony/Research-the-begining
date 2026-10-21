# Topic 2.1: Open-Weight Prompt Serialization, Chat Templates, and KV Cache Memory Scaling

When using closed APIs (e.g., proprietary commercial endpoints), prompt engineering is often treated as simple text concatenation: you send a string, and an opaque black box returns another string.

In open-weight language model research, prompt engineering is an exact systems discipline. You have direct access to the tokenizer, control the exact special token delimiters placed into the sequence, inspect the token IDs in the vocabulary lookup table, and directly govern the GPU memory scaling of the Key-Value (KV) cache as context expands.

This guide details the mechanics of prompt serialization, chat template compilation, and context window budgeting on `meta-llama/Meta-Llama-3-8B-Instruct`.

---

## 1. Tokenization Fundamentals: Byte-Pair Encoding & The 128k Vocabulary

### From Strings to Integer Tensors
Neural networks do not understand raw textual characters or UTF-8 strings directly. Text must pass through a **tokenizer** before reaching the transformer embedding layer:

```text
Input Text ("Verify claim") 
   ↓ [Byte-Pair Encoding (BPE)]
Subword Fragments (["Ver", "ify", " claim"])
   ↓ [Vocabulary Lookup Table]
Token IDs ([37452, 2841, 4125])
   ↓ [Embedding Layer Lookup]
Dense Activation Vectors (Tensor shape: [1, 3, 4096])
```

### The Byte-Pair Encoding (BPE) Algorithm
Llama-3 utilizes Byte-Pair Encoding (BPE) implemented via `tiktoken`. BPE builds its vocabulary bottom-up:
1. It begins with raw base UTF-8 bytes (256 base byte tokens).
2. It iteratively scans a multi-terabyte training corpus and merges the most frequently co-occurring adjacent token pairs into a new single token.
3. It repeats this process until the vocabulary reaches its target threshold.

### Why Llama-3 Expanded Vocabulary to 128,256 Tokens
Llama-1 and Llama-2 utilized a 32,000-token vocabulary (`SentencePiece` BPE). Llama-3 quadrupled this to **128,256 tokens** (`tiktoken` BPE). This design decision has major engineering consequences:

1. **Higher Compression Efficiency**: A larger vocabulary contains longer common subwords, phrases, and domain-specific words. On average, Llama-3 produces **15 percent fewer tokens** than Llama-2 for the exact same English paragraph, and up to 25 to 30 percent fewer tokens for code and multilingual text. Fewer tokens means fewer autoregressive forward passes and faster overall generation.
2. **Embedding Table VRAM Footprint**: The trade-off of a large vocabulary is the memory size of the input embedding layer and the final output unembedding projection layer (`lm_head`).
   - `Embedding Layer Parameters = Vocabulary Size · Hidden Dimension`
   - `Embedding Layer Parameters = 128,256 · 4,096 ≈ 525,336,576 parameters (~525M parameters)`
   - In 16-bit precision (FP16/BF16), storing just the input embedding table and the output language modeling head consumes:
     `2 · (525M · 2 bytes) ≈ 2.1 GB VRAM`
   This is why 4-bit NF4 quantization is so critical on Colab T4: quantizing these massive linear projection weights frees gigabytes of memory for long context sequences.

---

## 2. Llama-3 Special Token Syntax & Compilation Mechanics

Autoregressive models generate text by predicting the next token from left to right. Without explicit structural delimiters, the model cannot distinguish between system instructions, few-shot demonstration exemplars, user questions, and prior model outputs.

Open-weight models solve this by reserving dedicated **special tokens** at the end of the vocabulary range (IDs 128000 to 128255).

### The Five Critical Llama-3 Special Tokens

| Special Token | Token ID | Role & Functional Purpose |
| :--- | :--- | :--- |
| `<|begin_of_text|>` | `128000` | **BOS (Beginning of Sequence)**: Prepend to the very beginning of the prompt. Tells the positional embeddings that sequence indexing starts at position 0. |
| `<|start_header_id|>` | `128006` | **Header Prefix**: Declares the start of an authoritative role label (`system`, `user`, or `assistant`). |
| `<|end_header_id|>` | `128007` | **Header Suffix**: Closes the role label. Followed immediately by two newline characters (`\n\n`) before the message body. |
| `<|eot_id|>` | `128009` | **EOT (End of Turn)**: Indicates that the current speaker has finished their turn. Crucial for stopping the model from generating both sides of a conversation. |
| `<|end_of_text|>` | `128001` | **EOS (End of Sequence)**: Indicates the complete termination of generation across the entire conversation. |

### Canonical Prompt Structure in the Raw Token Stream
When you send a system instruction and a user message to `meta-llama/Meta-Llama-3-8B-Instruct`, the raw compiled string must match this exact sequence:

```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert veracity verification assistant. Output JSON only.<|eot_id|><|start_header_id|>user<|end_header_id|>

Evaluate the veracity of: The moon is made of green cheese.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

Notice the crucial pattern at the very end:
`<|start_header_id|>assistant<|end_header_id|>\n\n`
This trailing assistant header acts as the **generation prompt**. It primes the model so that the very next token it generates autoregressively will be the start of the assistant response, rather than continuing to simulate user input or hallucinating new conversational roles.

---

## 3. Implementing `apply_chat_template` in Python

Manually writing strings with special tokens using Python string interpolation (`f"<|start_header_id|>..."`) is error-prone. A missed newline, incorrect whitespace, or forgotten `<|eot_id|>` breaks instruction-following and causes the model to wander or ignore system constraints.

Hugging Face tokenizers provide `apply_chat_template`, which compiles standard Python dictionaries into the model's official Jinja template:

```python
from transformers import AutoTokenizer

# 1. Load the official tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# 2. Define conversation as structured role turns
messages = [
    {
        "role": "system",
        "content": "You are a professional fact-checker. Evaluate claims and respond strictly in JSON."
    },
    {
        "role": "user",
        "content": "Claim: The Federal Reserve raised interest rates by 25 basis points in July 2023."
    }
]

# 3. Compile to raw string (tokenize=False to inspect the exact special tokens)
formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print("--- Compiled Raw String ---")
print(formatted_prompt)

# 4. Compile directly to PyTorch Input Tensors for the model forward pass
input_tensors = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

print("\n--- Tensor Shape ---")
print(f"Input shape: {input_tensors.shape}")
print(f"First 5 Token IDs: {input_tensors[0][:5].tolist()}")
```

When `add_generation_prompt=True` is passed, the tokenizer automatically appends the trailing `<|start_header_id|>assistant<|end_header_id|>\n\n` tokens.

---

## 4. The Mathematical Reality of the KV Cache in GPU VRAM

Understanding context window scaling requires understanding the **Key-Value (KV) Cache**.

### Why the KV Cache Exists
In causal decoder transformers, generating each new token requires computing attention between the new query token and all previous tokens in the sequence:
`Attention(Q, K, V) = softmax( (Q · Kᵀ) / √d_k ) · V`

Without a cache, generating token 100 would require recomputing the Key (`K`) and Value (`V`) matrices for all previous 99 tokens. Generating token 101 would recompute 100 tokens, resulting in quadratic computation time `O(N²)`.

To make generation linear `O(N)`, the model calculates `K` and `V` for each token once and stores them in GPU VRAM: the **KV Cache**.

### Grouped-Query Attention (GQA) in Llama-3-8B
In traditional Multi-Head Attention (MHA), every attention query head has its own dedicated Key head and Value head.

Llama-3-8B utilizes **Grouped-Query Attention (GQA)**:
- **Query Heads (`n_heads`)**: 32 heads
- **Key/Value Heads (`n_kv_heads`)**: 8 heads
- **Head Dimension (`d_head`)**: 128
- **Transformer Layers (`n_layers`)**: 32
- **GQA Ratio**: 32 / 8 = 4 query heads share each single Key/Value head.

### The KV Cache Memory Formula
For each token in the context window, the model must store one Key vector and one Value vector per layer:

`KV_Memory_per_token = 2 · n_layers · n_kv_heads · d_head · bytes_per_element`

Let us calculate this step-by-step for FP16 precision (`bytes_per_element = 2`):
- Factor 2: One for Key matrix, one for Value matrix.
- `n_layers = 32`
- `n_kv_heads = 8`
- `d_head = 128`
- `bytes_per_element = 2` (FP16 / BF16)

`KV_Memory_per_token = 2 · 32 · 8 · 128 · 2 bytes`  
`KV_Memory_per_token = 131,072 bytes = 128 KB per token`

### Comparing GQA vs Traditional MHA

| Architecture Type | Query Heads | KV Heads | Bytes per Token | KV Cache at 8k Tokens | KV Cache at 32k Tokens |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Traditional MHA** (Llama-1) | 32 | 32 | 512 KB | 4.0 GB | 16.0 GB (OOM on T4) |
| **Grouped-Query GQA** (Llama-3) | 32 | 8 | **128 KB** | **1.0 GB** | **4.0 GB** |

GQA provides a **75 percent reduction** in KV cache memory footprint. This is the exact architectural innovation that allows us to run Few-Shot ICL and Context-Aware RAG on a free Colab T4 GPU without running out of memory.

---

## 5. Context Window Budgeting on Google Colab T4

Let us audit our exact hardware envelope for the experiments in this chapter:

- **GPU**: NVIDIA T4 (15.0 GB usable VRAM)
- **Base Model (4-bit NF4 Llama-3-8B)**: ~5.35 GB VRAM
- **CUDA Overhead & PyTorch Workspace**: ~0.80 GB VRAM
- **Total Static Allocation**: `5.35 GB + 0.80 GB = 6.15 GB`
- **Dynamic Headroom Remaining for KV Cache**: `15.0 GB - 6.15 GB = 8.85 GB VRAM`

### Maximum Theoretical Context Capacity
Using our KV cache formula (`128 KB / token`):
`Max_Tokens = 8.85 GB / 128 KB = (8.85 · 1024 · 1024 KB) / 128 KB ≈ 72,500 tokens`

### Our Chapter 2 Experimental Budget
For our Chapter 2 benchmark across all four paradigms:
- **System Prompt**: ~150 tokens
- **Benchmark Claim**: ~50 tokens
- **Few-Shot Exemplars (k=5)**: ~1,500 tokens
- **RAG Retrieved Passages (top-3)**: ~1,200 tokens
- **CoT Reasoning Generation**: ~400 tokens
- **Total Peak Context**: `~3,300 tokens`

`Peak_KV_Cache_Memory = 3,300 · 128 KB ≈ 422 MB ≈ 0.41 GB`

`Total_Peak_VRAM = 6.15 GB (static) + 0.41 GB (KV cache) ≈ 6.56 GB`

This provides **over 8.4 GB of safety margin** on Google Colab T4. You will never encounter CUDA Out of Memory (OOM) errors during this chapter.

---

## 6. Summary & Key Takeaways

1. **Tokens are categorical integer IDs**, not strings. Llama-3's 128k vocabulary provides ~15% higher text compression efficiency at the cost of a 525M parameter embedding table.
2. **Special tokens are authoritative control boundaries**. `<|start_header_id|>` and `<|eot_id|>` delineate roles to prevent instruction bleeding and conversational drift.
3. **Always compile prompts with `apply_chat_template`**. Passing `add_generation_prompt=True` automatically primes the assistant role header.
4. **GQA scales context cleanly**. At 128 KB per token, a 4,000-token prompt with few-shot exemplars or retrieved evidence consumes only ~512 MB of KV cache VRAM on Colab T4.

---

## Navigation & Next Steps

- **Syllabus**: [Chapter 2 README](../README.md)
- **Next Topic**: [Topic 2.2: Zero-Shot and Few-Shot In-Context Learning](./topic_2.2_zero_shot_and_few_shot_icl.md)
- **Ticket Status**: [Ticket 01 Tracker](../../../.scratch/in-context-reasoning/issues/01-open-weight-prompt-serialization-and-kv-cache.md)
