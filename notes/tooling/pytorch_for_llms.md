# 02. PyTorch for LLM Research

PyTorch (`torch`) is the engine powering modern deep learning and Large Language Models.

---

## 1. Mental Model: Systems Engineer Translation

| Concept | C# / ASP.NET Core Equivalent | Angular / TypeScript Equivalent | PyTorch Syntax |
| :--- | :--- | :--- | :--- |
| **Tensor** | Managed typed buffer (`Span<float>`) with device pointer | TypedArray buffer attached to WebGPU | `torch.tensor([1.0, 2.0])` |
| **Device Memory** | System RAM vs. Offloaded GPU memory | Main Thread vs. WebWorker / GPU Buffer | `.to("cuda")` vs. `.to("cpu")` |
| **Autograd Graph** | Entity Framework Change Tracker | Reactive Dependency Graph (Signals/RxJS) | `requires_grad=True` |
| **Disable Tracking** | EF Core `.AsNoTracking()` | Running outside NgZone / Untracked signal | `with torch.no_grad():` |
| **Model Mode** | Service Configuration / Environment Flag | Component production mode | `model.eval()` vs `model.train()` |
| **Softmax** | Normalizing array values so sum equals 1.0 | Math utility mapping raw weights to percentages | `torch.softmax(logits, dim=-1)` |

---

## 2. The 3 Standard Tensor Shapes in LLMs

In LLM research, almost every tensor matches one of three shapes. Memorizing these three shapes solves 90% of tensor debugging:

### 1. Token IDs Tensor (Integer Input)
Shape: `[BatchSize, SequenceLength]`
* Example: `[1, 16]` represents 1 prompt containing 16 token integers.
* Dtype: `torch.long` (64-bit integer, matching vocabulary IDs).

### 2. Hidden States / Embeddings (Float Representation)
Shape: `[BatchSize, SequenceLength, HiddenDimension]`
* Example: `[1, 16, 768]` in a 768-dim model (like GPT-2) or `[1, 16, 4096]` in Llama-3 8B.
* Meaning: Each token has a continuous vector of features.

### 3. Attention Weight Matrix (Square Attention Scores)
Shape: `[BatchSize, NumHeads, SequenceLength, SequenceLength]`
* Example: `[1, 12, 16, 16]` for a 12-head model.
* Meaning: A square matrix where row `i` and column `j` represent how much token `i` attends to token `j`.

---

## 3. The 6 PyTorch Workflows You Will Use Constantly

### 1. Device Placement (CPU vs. GPU VRAM)
By default, tensors are allocated in system RAM. Deep learning models execute on GPU VRAM:

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Active Device:", device)

# Create a tensor and push to graphics card memory:
tensor = torch.randn(2, 4).to(device)
print(tensor.device)  # cuda:0 or cpu
```

### 2. Disabling Gradient Tracking: `torch.no_grad()`
In research inference and evaluation (Phases 1 through 4), **you are not training model weights**. 
If you do not disable gradient tracking, PyTorch allocates massive computation graphs in VRAM, causing Out Of Memory (OOM) crashes:

```python
# C# Analogy: dbContext.Users.AsNoTracking().ToList();
with torch.no_grad():
    outputs = model(**inputs)
    # Autograd graph is not stored. VRAM usage drops by 60-70%.
```

### 3. Evaluation Mode: `model.eval()`
In neural networks, layers like Dropout drop random neurons during training for regularization. In evaluation, you want deterministic behavior:

```python
# Always invoke before running benchmarks or adversarial tests:
model.eval()
```

### 4. Squeeze and Unsqueeze (Reshaping Dimensions of Size 1)
When tokenizing a single string, you often get a 1D vector `[16]`. A model expects a 2D batch `[1, 16]`:

```python
tokens_1d = torch.tensor([101, 2054, 2003, 102])  # Shape: [4]

# Add batch dimension at index 0 (unsqueeze):
batch_2d = tokens_1d.unsqueeze(0)                 # Shape: [1, 4]

# Remove dimension of size 1 (squeeze):
recovered_1d = batch_2d.squeeze(0)                # Shape: [4]
```

### 5. From Logits to Predicted Next Token
A model outputs unnormalized scores (`logits`) across the entire vocabulary (e.g., size 32,000 or 128,000). To convert logits to probabilities and pick the top token:

```python
logits = outputs.logits  # Shape: [Batch, SequenceLength, VocabSize]

# Get the logits for the very last token position:
last_token_logits = logits[0, -1, :]  # Shape: [VocabSize]

# Softmax converts scores to probabilities (summing to 1.0):
probs = torch.softmax(last_token_logits, dim=-1)

# Argmax picks the token ID with the highest probability:
next_token_id = torch.argmax(probs, dim=-1).item()
print("Predicted Token ID:", next_token_id)
```

### 6. Extracting Attention Tensors
In safety and interpretability research (e.g., [notebooks/phase_2_adversarial_comments.ipynb](file:///d:/agent-based-learn/LLM-Research/Research-the-begining/notebooks/phase_2_adversarial_comments.ipynb)), you inspect how the model attends to tokens:

```python
# Pass output_attentions=True in the forward pass
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

# outputs.attentions is a tuple of tensors, one per model layer:
# (Layer 0, Layer 1, ..., Layer N)
layer_0_attention = outputs.attentions[0]
print("Attention Shape:", layer_0_attention.shape)
# Output: [batch_size, num_heads, sequence_length, sequence_length]
```

---

## 4. Key Takeaway for LLM Work

Never try to memorize every PyTorch math function. Keep your attention focused on:
1. What device is this tensor on? (`.device`)
2. What are the dimensions? (`.shape`)
3. Is change tracking disabled? (`with torch.no_grad():`)
