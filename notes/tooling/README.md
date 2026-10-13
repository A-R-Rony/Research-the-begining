# LLM Research Tooling & Data Stack Guide

This guide is designed specifically for software engineers with an **ASP.NET Core (C#)** backend and **Angular / TypeScript / JavaScript** frontend background.

In LLM research (Phases 1 through 4), you do not need to memorize every API across the Python ecosystem. Approximately 85% of your work interacts with four core libraries across predictable patterns.

---

## The 4-Pillar Stack at a Glance

| Library | Primary Responsibility | C# / ASP.NET Core Analogy | Angular / JS Analogy |
| :--- | :--- | :--- | :--- |
| [NumPy](./numpy_essentials.md) | CPU Vectorized array math & stats | Flat memory buffer / `Span<float>` + SIMD vector math | `Float32Array` / TypedArray with math helpers |
| [PyTorch](./pytorch_for_llms.md) | GPU Tensors, forward passes, attention weights | GPU worker compute + Entity Framework change tracking (`autograd`) | WebGPU compute buffers + reactive dependency graph |
| [Transformers](./transformers_hf.md) | Pretrained model loading & tokenization | Strongly-typed SDK client + JSON serializer/deserializer | HTTP Service + DTO serialization pipeline |
| [Pandas](./pandas_for_experiments.md) | Experiment tracking & benchmark tables | In-memory `DataTable` + LINQ queries (`Where`, `Select`, `GroupBy`) | RxJS state streams / in-memory grid store |

---

## Recommended Learning Sequence

Follow this execution pipeline order:

1. **[01. NumPy Essentials](./numpy_essentials.md)**:
   * Mental Model: Contiguous memory buffers, shapes, axes, broadcasting.
   * What to look for: Converting PyTorch GPU tensors back to CPU arrays for plotting and cosine similarity.

2. **[02. PyTorch for LLMs](./pytorch_for_llms.md)**:
   * Mental Model: The hardware compute graph. Moving memory to `cuda`.
   * What to look for: Slicing 3D shapes `[Batch, SequenceLength, HiddenDim]`, extracting attention matrices, and using `torch.no_grad()` to freeze change-tracking during evaluation.

3. **[03. Hugging Face Transformers](./transformers_hf.md)**:
   * Mental Model: The LLM pipeline.
   * What to look for: String to Token IDs mapping (`tokenizer`), forward pass inputs (`input_ids`, `attention_mask`), logits, and inspectable layer outputs (`output_attentions=True`).

4. **[04. Pandas for Experiment Tracking](./pandas_for_experiments.md)**:
   * Mental Model: In-memory relational tables with LINQ-style querying.
   * What to look for: Storing 50 adversarial prompts, filtering jailbreak attempts, calculating mean safety scores by category, and exporting reports.

---

## Environment Prerequisites Checklist

To verify your local environment or Colab session has these packages installed, execute:

```python
import numpy as np
import torch
import transformers
import pandas as pd

print(f"NumPy: {np.__version__}")
print(f"PyTorch: {torch.__version__} (CUDA Available: {torch.cuda.is_available()})")
print(f"Transformers: {transformers.__version__}")
print(f"Pandas: {pd.__version__}")
```

If GPU is available, PyTorch will report `CUDA Available: True`. If running on CPU, all operations remain fully functional, using SIMD system memory.
