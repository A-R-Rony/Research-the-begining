# LLM Research & Tooling

Domain terminology for empirical LLM safety, attention interpretability, adversarial jailbreak analysis, and deep learning runtime tooling.

## Language

**Tensor**:
A multidimensional, contiguous numerical memory buffer characterized by rank, shape, and element data type, supporting hardware-accelerated parallel computation and optional automatic differentiation graph tracking.
_Avoid_: Matrix (when rank != 2), multi-dimensional array, ndarray

**Kernel (Jupyter)**:
A dedicated, long-running background language runtime process that maintains an interactive memory state and executes execution payloads sent by a client editor.
_Avoid_: Virtual machine, thread, notebook session

**Autograd**:
A reverse-mode automatic differentiation engine that constructs a directed acyclic computation graph during forward execution to compute exact gradients via the chain rule.
_Avoid_: Backprop engine, gradient tracker, optimizer

**DataFrame**:
A two-dimensional, size-mutable tabular data structure with heterogeneous typed columns and labeled index axes for declarative data manipulation.
_Avoid_: Table, grid, spreadsheet, in-memory database

**Token**:
The atomic integer-mapped categorical unit produced by a tokenizer representing a subword fragment, character, or special control sequence.
_Avoid_: Word, character, string chunk

**Vocabulary**:
The fixed bidirectional lookup table mapping textual token fragments to unique non-negative integer IDs.
_Avoid_: Dictionary, word list, token pool

**Logits**:
The raw, unnormalized scalar scores output by the final linear projection layer of a neural network before conversion to probabilities.
_Avoid_: Probabilities, confidence scores, output weights

**Attention Matrix**:
A square, row-stochastic matrix of scalar weights produced by softmax-normalized scaled dot-product queries and keys representing pairwise token relevance.
_Avoid_: Attention map, heat values, query-key dot product

**Embedding**:
A continuous, dense vector representation of a discrete categorical input token in a learned high-dimensional semantic space.
_Avoid_: Feature vector, word vector, representation

**Forward Pass**:
The deterministic sequence of matrix transformations and layer operations that maps input token IDs to output logits without updating weights.
_Avoid_: Inference pass, prediction run, model execution
