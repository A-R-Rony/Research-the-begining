# 01. NumPy Essentials for LLM Research

NumPy (`numpy`) is the foundational numerical computing library in Python. It executes vector and matrix math in compiled C and Fortran code, avoiding Python interpreter overhead.

---

## 1. Mental Model: Systems Engineer Translation

| Concept | C# / ASP.NET Core Equivalent | Angular / TypeScript Equivalent | NumPy Syntax |
| :--- | :--- | :--- | :--- |
| **1D Array** | `float[]` or `Span<float>` | `Float32Array` | `np.array([1.0, 2.0, 3.0])` |
| **Multidimensional Array** | Flat 1D buffer indexed with row strides | `Float32Array` with stride offsets | `np.zeros((3, 768))` |
| **Shape** | Array dimensions property | Custom wrapper `.shape` | `arr.shape` (e.g., `(32, 512)`) |
| **Vector Math (SIMD)** | `System.Numerics.Vector<float>` | WebAssembly SIMD operations | `a + b`, `a * b` (element-wise) |
| **Matrix Multiply** | Nested loops or BLAS native calls | Custom matrix loop / WebGPU matmul | `a @ b` or `np.matmul(a, b)` |

---

## 2. The 5 Operations You Must Know for LLMs

### 1. Shape and Inspection
In deep learning research, bugs almost always stem from dimension mismatches. Inspecting the shape is your first debugging step:

```python
import numpy as np

# A simulated batch of 2 sentences, 4 tokens each, with 8-dimensional embeddings
embeddings = np.random.randn(2, 4, 8)

print("Shape:", embeddings.shape)  # Output: (2, 4, 8)
print("Rank (Dimensions):", embeddings.ndim)  # Output: 3
print("Data Type:", embeddings.dtype)  # Output: float64
```

### 2. Slicing and Indexing (Array Slicing vs. LINQ)

In C# with LINQ, you might write:
```csharp
// Take the first batch item, all tokens, and the first 4 features:
var slice = items[0].Tokens.Select(t => t.Features.Take(4)).ToArray();
```

In NumPy, slicing uses concise stride syntax `[batch, sequence, features]`:
```python
# [item 0, all tokens, first 4 features]
slice_np = embeddings[0, :, :4]
print(slice_np.shape)  # Output: (4, 4)
```

### 3. Axes: Understanding Aggregation Dimensions
When you reduce an array (using `mean`, `sum`, or `max`), the `axis` parameter defines which dimension collapses:

* `axis=0`: Collapses across rows (or batches).
* `axis=1`: Collapses across columns (or tokens).
* `axis=-1`: Collapses across the last dimension (commonly embedding features).

```python
# Compute mean embedding vector for each token across the batch (collapse axis 0)
mean_token_vector = embeddings.mean(axis=0)
print(mean_token_vector.shape)  # Output: (4, 8)
```

### 4. Cosine Similarity (Comparing Embeddings)

#### What is it and why do we need it?
When an LLM processes text, it turns words or sentences into long lists of numbers called **Embeddings** (e.g., 4,096 numbers in Llama-3-8B).
To check whether two sentences have similar meanings, we cannot use ordinary straight-line distance. A short sentence (*"Great film!"*) has smaller numbers than a long sentence (*"This motion picture was magnificent!"*), making their straight-line physical distance artificially large even though their semantic meaning is identical.

Instead of measuring arrow length, **Cosine Similarity measures the angle (θ) between two arrows in space**:
* **+1.0 (Angle = 0°)**: Identical meaning (arrows point in the exact same direction).
* **0.0 (Angle = 90°)**: Unrelated topics (arrows are perpendicular / orthogonal).
* **-1.0 (Angle = 180°)**: Exact opposite meaning (arrows point in reverse directions).

#### Python Implementation & Line-by-Line Breakdown:
```python
import numpy as np

def cosine_similarity(u, v):
    # 1. Dot product: Multiplies matching elements and adds them up.
    # Measures directional overlap: (u₁ · v₁) + (u₂ · v₂) + ... + (uₙ · vₙ)
    dot_product = np.dot(u, v)

    # 2. Vector Norms: Calculates the geometric length (magnitude) of each arrow.
    # Formula: ||u|| = √(u₁² + u₂² + ... + uₙ²)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    # 3. Normalization: Dividing by (norm_u · norm_v) cancels out the length of both arrows,
    # leaving only the pure cosine of the angle between them.
    return dot_product / (norm_u * norm_v)

# Example: Two semantically related word embeddings
vec_a = np.array([0.2, 0.8, -0.5])
vec_b = np.array([0.25, 0.75, -0.4])

score = cosine_similarity(vec_a, vec_b)
print("Similarity Score:", round(score, 4))  # Output: ~0.9946 (Extremely similar direction)
```

### 5. Bridging PyTorch and NumPy
PyTorch tensors live in their own runtime (and potentially on GPU VRAM). When you need to plot attention weights or save raw matrices to disk, convert from PyTorch to NumPy:

```python
# From PyTorch GPU Tensor to NumPy CPU Array:
# 1. detach() -> disconnect from autograd graph
# 2. cpu()    -> copy memory from GPU VRAM to Host System RAM
# 3. numpy()  -> expose raw pointer as NumPy array without copying
numpy_array = torch_tensor.detach().cpu().numpy()
```

---

## 3. Key Takeaway for LLM Work

You do not write neural networks in pure NumPy because NumPy does not run on GPU and does not compute gradients. NumPy is used in Phase 1 and 2 to:
1. Verify matrix multiplication math independently.
2. Calculate statistical metrics (mean, standard deviation, cosine similarity).
3. Export attention weights for visualization and plotting.
