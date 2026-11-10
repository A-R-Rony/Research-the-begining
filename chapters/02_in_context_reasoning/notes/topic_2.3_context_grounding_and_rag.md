# Topic 2.3: Context-Aware Grounding and Lightweight Retrieval-Augmented Generation (RAG)

In [Topic 2.2: Zero-Shot and Few-Shot In-Context Learning](./topic_2.2_zero_shot_and_few_shot_icl.md), we discovered how few-shot prompt examples steer the model's style, output structure, and tone. However, we also encountered a hard brick wall:

> **The Epistemic Boundary of LLMs**: An AI model cannot generate facts that do not already exist in its frozen pre-trained weights (`requires_grad = False`). Few-shot examples can teach a model *how to format* an answer (like valid JSON), but they **cannot teach it facts it never learned**, nor can they easily unlearn stubborn internet myths.

When an ungrounded model is asked to verify claims that involve subtle technical definitions, counter-intuitive scientific realities, or recent news, it relies entirely on its training memory. The result? **Beautifully formatted, highly confident hallucinations.**

To overcome this, we must transition from *ungrounded guessing* to **Context Grounding** and **Retrieval-Augmented Generation (RAG)**. 

We break this down into two progressive, logical stages:
- **Stage 1: Prompt-Level Evidence Grounding (The Open-Book Test)**: How to feed reference text directly into the prompt so the model reads the document instead of relying on its subconscious memory habits, and how to construct a prompt envelope that forces the model to prioritize evidence over internal bias.
- **Stage 2: Lightweight In-Memory RAG (The Automated Librarian)**: How to transition from manually copy-pasting an article to automatically searching an unindexed library of thousands of documents in sub-millisecond time using pure PyTorch GPU matrix math.

---

## Stage 1: Prompt-Level Evidence Grounding (The Open-Book Verification)

### 1.1 The Core Intuition: Closed-Book vs. Open-Book

The simplest way to understand Stage 1 is the difference between two types of school exams:

```text
┌───────────────────────────────────────┬───────────────────────────────────────┐
│ 1. Closed-Book Exam (Ungrounded)      │ 2. Open-Book Exam (Context Grounded)  │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ • You must answer entirely from what  │ • You are given the authoritative     │
│   you memorized in your head.         │   textbook right on your desk.        │
│ • If you never memorized a fact, you  │ • You don't need to have memorized    │
│   guess based on common rumors.       │   the fact; you just look it up.      │
│ • High risk of recalling popular      │ • Verifiable, factual, and backed by  │
│   misconceptions and rumors.          │   direct citations.                   │
└───────────────────────────────────────┴───────────────────────────────────────┘
```

In Stage 1, we turn the LLM into an **open-book reader**. Instead of asking:
> *"Is claim X true or false?"*

We supply the exact reference passage and ask:
> *"Based SOLELY on this provided passage, is claim X true or false?"*

---

### 1.2 The Problem: Why Doesn't the Model Just Read the Text? (The Memory Collision)

You might think: *"If I paste the text right in front of the model, won't it just read it and answer correctly?"*

Surprisingly, **no**. In smaller open-weight models (like `Llama-3-8B`), the model frequently ignores the text you gave it and outputs a hallucination based on its old training data!

To understand why this happens, look at how the transformer's brain is split:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   THE TWO COMPETING MEMORIES INSIDE A TRANSFORMER                      │
├─────────────────────────────────────┬──────────────────────────────────────────────────┤
│ 1. Parametric Internal Memory       │ 2. In-Context Working Memory                     │
├─────────────────────────────────────┼──────────────────────────────────────────────────┤
│ • Stored across 8.03 billion frozen │ • Stored dynamically in GPU VRAM (KV Cache)      │
│   weight matrices on disk           │   for the current prompt only                    │
│ • Represents the model's ingrained  │ • Represents the open-book text sitting right    │
│   "habits" from reading the web     │   on the desk in front of the model              │
│ • Physically located in the         │ • Physically processed by the                    │
│   Feed-Forward Networks (MLPs)      │   Self-Attention Heads                           │
└─────────────────────────────────────┴──────────────────────────────────────────────────┘
```

#### What Happens Inside the Transformer During Each Layer?
Inside `Llama-3-8B`, there are 32 layers. Down the center of these layers runs a conveyor belt of thought called the **Residual Stream** ($x$). 

At every layer, two components add information onto the conveyor belt:

```text
       Start of Layer: Current thought on conveyor belt: x^(l-1)
                              │
                              ▼
        ┌──────────────────────────────────────────────┐
        │ 1. Self-Attention (The Reading Eye)          │
        │ Looks at the text pasted in the prompt.      │
        │ Adds its nudge: Δx_context                   │
        └──────────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────────┐
        │ 2. Feed-Forward MLP (The Subconscious Habit) │
        │ Recalls pre-trained internet associations.   │
        │ Adds its nudge: Δx_parametric                │
        └──────────────────────────────────────────────┘
                              │
                              ▼
       End of Layer: Combined thought on belt: x^(l)
```

Mathematically, this is a pure additive update at layer $l$:

$$x^{(l)} = x^{(l-1)} + \Delta x_{\text{context}}^{(l)} + \Delta x_{\text{parametric}}^{(l)}$$

* $\Delta x_{\text{context}}^{(l)}$ is the nudge from **Self-Attention**, pointing to what the **provided document** says.
* $\Delta x_{\text{parametric}}^{(l)}$ is the nudge from the **MLPs**, pointing to what the **pre-trained weights** remember from reading the web.

#### The Tug-of-War:
- When the document agrees with common knowledge (e.g., *"Paris is the capital of France"*), both components push in the **same direction**.
- But when the document contradicts a popular misconception or counter-intuitive fact, they push in **opposite directions**:
  - Attention pushes Left: *"The document says this is True!"*
  - MLPs push Right: *"The internet repeated that this is False millions of times!"*

Because an 8B model's weights have seen popular internet myths millions of times, the MLP nudge ($\Delta x_{\text{parametric}}$) is often much heavier than the attention nudge. **The model's memory subconscious overpowers its reading eyes**, causing it to hallucinate and ignore the document. This clash is the **Parametric Memory Collision**.

---

### 1.3 The Three Real-World Traps Where Memory Overpowers Context

To see where this collision happens in practice, consider the three most common categories of factual claims that cause ungrounded models to fail:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        THREE CLASSIC REAL-WORLD MEMORY TRAPS                           │
├─────────────────────┬────────────────────────────────────┬─────────────────────────────┤
│ Category            │ Concrete Example                   │ Why Pre-Trained Memory Fails│
├─────────────────────┼────────────────────────────────────┼─────────────────────────────┤
│ 1. Common Web Myths │ "Humans only use 10% of their      │ Repeated millions of times  │
│    & Urban Legends  │ brains during daily life."         │ on pop culture blogs; the   │
│                     │                                    │ weights default to the myth │
├─────────────────────┼────────────────────────────────────┼─────────────────────────────┤
│ 2. Counter-         │ "Planting trees in snowy polar     │ Intuition says "trees cool";│
│    Intuitive Physics│ zones can increase net warming by  │ misses that dark trees      │
│                     │ absorbing solar radiation."        │ absorb heat snow reflects   │
├─────────────────────┼────────────────────────────────────┼─────────────────────────────┤
│ 3. Subtle Technical │ "The official unemployment rate    │ Everyday slang equates      │
│    Definitions      │ counts every adult without a job." │ "jobless" with "unemployed";│
│                     │                                    │ misses the 4-week active    │
│                     │                                    │ search regulatory rule      │
└─────────────────────┴────────────────────────────────────┴─────────────────────────────┘
```

1. **Common Web Myths & Urban Legends**:
   - The internet is saturated with scientific urban legends. Because pre-training trains the model to predict the next word on web pages, its MLPs are strongly biased toward popular myths rather than peer-reviewed medical truth.
2. **Counter-Intuitive Scientific Realities**:
   - Human intuition assumes simple rules (e.g., *"vegetation always cools the planet"*). In reality, Earth systems are complex: dark conifer canopies in snowy boreal regions absorb sunlight that reflective snow would have bounced back into space, causing net local warming. Because the model's weights favor simple associative heuristics, it rejects counter-intuitive scientific realities unless forced to read the text.
3. **Subtle Technical Definitions vs. Everyday Slang**:
   - In casual conversation, someone without a job is called "unemployed." But official economic agencies (like the Bureau of Labor Statistics) have an exacting legal definition: you are only unemployed if you are actively searching for work. People who gave up looking are classified as *"not in the labor force"*. An ungrounded model conflates colloquial slang with regulatory definitions.

---

### 1.4 How Stage 1 Solves the Collision: The Grounded Prompt Envelope

To guarantee that the provided document **always** wins the tug-of-war against pre-trained habits, we cannot simply paste text into the chat and hope for the best. 

We must enforce an **Epistemic Hierarchy**:

$$\text{Provided Reference Document} \quad \succ \quad \text{Pre-Trained Internal Memory}$$

We enforce this hierarchy by constructing a **Grounded Prompt Envelope** built on three golden architectural rules:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                 THE THREE GOLDEN RULES OF THE GROUNDED PROMPT ENVELOPE                 │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Source Exclusivity (No Outside Knowledge)                                           │
│    Tell the model: "Answer STRICTLY and SOLELY based on the provided document.         │
│    Do not use outside knowledge or unstated assumptions."                              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. The Unconditional Override Directive                                                │
│    Explicitly instruct: "If the provided document contradicts your prior knowledge,     │
│    prioritize the provided document unconditionally."                                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. Mandatory Evidence Quotation (Token-Level Anchoring)                                │
│    Require the model to begin its reasoning by quoting the document directly.          │
│    Because generation is autoregressive, quoting the text forces subsequent attention  │
│    heads to stay locked onto the document's facts rather than drifting into bias.      │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

#### The Structural Layout of the Prompt Envelope
We arrange the prompt deliberately so that the reference evidence is processed **before** the question:

```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an evidence-grounded verification assistant. 
Your task is to determine whether the submitted Claim is True or False STRICTLY and SOLELY based on the provided Reference Evidence.
If the Reference Evidence confirms the claim, predict True. 
If the Reference Evidence refutes the claim or proves it inaccurate, predict False.
If the Reference Evidence contradicts your prior knowledge, prioritize the Reference Evidence unconditionally.
Do not rely on unstated assumptions or prior beliefs. Output valid JSON only.<|eot_id|><|start_header_id|>user<|end_header_id|>

### Reference Evidence:
NASA satellite observations and climate models demonstrate that while solar irradiance variations do influence Earth's upper atmosphere, total solar output has fluctuated within a narrow 0.1 percent band since 1750. In contrast, greenhouse gas concentrations produced by fossil fuel combustion have increased atmospheric heat retention by orders of magnitude greater than solar variance. NASA officially attributes observed global warming primarily to human greenhouse emissions.

### Target Claim:
"NASA has officially confirmed that human carbon emissions have zero impact on global climate change and it is entirely solar cycles."

Evaluate the claim against the Reference Evidence. Respond strictly in JSON.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

#### Why Evidence Before Claim?
In a transformer, tokens can only attend to tokens that came before them or are processed alongside them. 
* By placing the **Reference Evidence first**, its keys and values are computed and stored in the GPU's KV Cache.
* When the model reaches the **Target Claim**, its attention heads can immediately cross-reference the claim tokens against the evidence already waiting in memory.

---

### 1.5 Python Implementation for Compiling Grounded Envelopes

Here is the Python implementation that packages any claim and evidence pair into this grounded envelope using Hugging Face:

```python
def compile_grounded_prompt(claim: str, evidence: str, tokenizer) -> str:
    """
    Compiles an authoritative evidence-grounded prompt enforcing the epistemic hierarchy.
    
    Args:
        claim: The factual statement to evaluate.
        evidence: The reference document containing verified ground truth.
        tokenizer: Hugging Face PreTrainedTokenizerFast for Llama-3.
        
    Returns:
        Properly formatted prompt string with special delimiters ready for the model.
    """
    system_prompt = (
        "You are an evidence-grounded verification assistant.\n"
        "Your task is to determine whether the submitted Claim is True or False STRICTLY "
        "and SOLELY based on the provided Reference Evidence.\n"
        "If the Reference Evidence confirms the claim, predict True.\n"
        "If the Reference Evidence refutes the claim or proves it inaccurate, predict False.\n"
        "If the Reference Evidence contradicts your prior knowledge, prioritize the Reference Evidence unconditionally.\n"
        "Do not rely on unstated assumptions or prior beliefs. Output valid JSON only."
    )
    
    user_content = (
        f"### Reference Evidence:\n{evidence.strip()}\n\n"
        f"### Target Claim:\n\"{claim.strip()}\"\n\n"
        "Evaluate the claim against the Reference Evidence. Respond strictly in JSON with keys: "
        "'veracity_prediction' (True/False), 'confidence_score' (0.0 to 1.0), and 'reasoning' (string citing evidence)."
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

### 1.6 How to Measure If Grounding Actually Worked

When you evaluate a grounded system, how do you mathematically verify that the document beat the model's pre-trained memory? 

You measure two core metrics:

#### 1. Veracity Flip Rate (Error Correction Rate)
The percentage of questions that the model got wrong when asked from memory (ungrounded), but **flipped to the correct answer** once the reference document was provided:

$$\text{Veracity Flip Rate} = \frac{\text{Number of Erroneous Ungrounded Answers Corrected by Evidence}}{\text{Total Erroneous Ungrounded Answers}}$$

* A high flip rate (e.g., 90–100%) proves that the model is actively reading the context and allowing the evidence to override its pre-trained misconceptions.

#### 2. Evidence Faithfulness Score
Getting the right True/False label is not enough—the model might guess correctly while making up a completely fake reason! 

**Evidence Faithfulness** checks whether the model's generated explanation is genuinely derived from the provided document:
- **Faithful ($\checkmark$)**: The reasoning explicitly quotes or attributes facts to the provided passage (e.g., *"The text states that U-3 excludes workers who haven't searched in 4 weeks"*).
- **Unfaithful / Hallucinated ($\times$)**: The reasoning introduces outside trivia, invents non-existent numbers, or fails to cite the provided text.

---

## Stage 2: Lightweight In-Memory RAG (The Automated Retrieval Engine)

### 2.1 The Bridge from Stage 1 to Stage 2: Why Do We Need RAG?

In Stage 1, we assumed an "oracle" existed: **a human manually found the right document and copy-pasted it into the prompt.**

In the real world, this is impossible:
1. You have a knowledge base of 10,000 documents sitting in a database or folder.
2. A user asks an open-ended question or submits an arbitrary claim.
3. **You cannot stuff all 10,000 documents into the prompt**:
   - As proven in [Topic 2.1](./topic_2.1_open_weight_prompt_mechanics.md), packing thousands of documents exhausts GPU VRAM (KV Cache explosion) and causes massive inference latency ($O(N^2)$ attention compute).

```text
┌───────────────────────────────────────┬───────────────────────────────────────┐
│ Stage 1: Prompt Grounding             │ Stage 2: Dynamic RAG Engine           │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ • Human manually pairs the claim with │ • System searches an unindexed corpus │
│   its verified reference passage.     │   of thousands of candidate documents.│
│ • Tests if the model CAN follow text  │ • Finds the needle in the haystack    │
│   when placed in front of it.         │   in milliseconds automatically.      │
│ • Not scalable to production.         │ • Feeds the retrieved chunk directly  │
│                                       │   into the Stage 1 prompt envelope.   │
└───────────────────────────────────────┴───────────────────────────────────────┘
```

**Stage 2 (RAG) is simply building the automated search engine that finds the right document in milliseconds and hands it to Stage 1.**

---

### 2.2 Why Keyword Search (BM25) Fails in Neural Pipelines

The oldest way to search documents is keyword matching (like Ctrl+F or BM25). 

Keyword search fails whenever the search query and the document use **different words to describe the same concept** (the vocabulary mismatch problem):

```text
User Claim:   "cardiac arrest survival procedures"
Document:     "emergency medical interventions for acute myocardial infarction"

Keyword Overlap: 0 common words -> Keyword search returns 0 results!
```

To solve this, we must search by **semantic meaning** rather than identical spelling. We do this using **Dense Neural Embeddings**.

---

### 2.3 The Bi-Encoder Architecture (`all-MiniLM-L6-v2`)

A neural **Bi-Encoder** converts any text string into a list of numbers (a dense vector) where texts with similar meanings end up close to each other in mathematical space:

```text
Query String ("solar cycles climate") ──────► [ Bi-Encoder ] ──────► Query Vector q ∈ ℝ³⁸⁴
Document String ("greenhouse gases...") ───► [ Bi-Encoder ] ──────► Document Vector d ∈ ℝ³⁸⁴
```

We choose `sentence-transformers/all-MiniLM-L6-v2`:
* **Size**: 22.7 million parameters (~80 MB of GPU memory—negligible next to Llama-3's 5.35 GB).
* **Output**: A 384-dimensional vector representing the semantic gist of the text.
* **Why Bi-Encoder?** We can encode all 10,000 documents in our database **once** ahead of time and store their vectors in GPU memory. When a user query arrives, we only need to encode the query once and perform an instant vector comparison.

---

### 2.4 Mathematical Optimization: From Geometric Angles to Pure Dot Products

In Section 2.3, we established that both the user's query and all 10,000 reference documents are converted into dense lists of 384 numbers (vectors $q$ and $d$).

This raises an immediate, practical question: **How does a computer compare two lists of 384 numbers to decide if they mean the same thing?**

Here is the step-by-step progression from basic geometric intuition to lightning-fast GPU math:

---

#### Step 1: The Geometry of Meaning (Cosine Similarity)
Think of each 384-number vector as an **arrow** pointing into a 384-dimensional semantic space:
* If two texts share the same meaning (e.g., *"heart attack"* and *"myocardial infarction"*), their arrows point in almost the **exact same direction** (the angle $\theta$ between them is nearly $0^\circ$).
* If two texts are completely unrelated (e.g., *"heart attack"* and *"baking a cake"*), their arrows point at right angles ($\theta \approx 90^\circ$).

In trigonometry, the cosine of an angle measures directional alignment:
* $\cos(0^\circ) = 1.0$ $\to$ **Identical semantic meaning**
* $\cos(90^\circ) = 0.0$ $\to$ **Zero semantic overlap**

Therefore, to compare a query vector $q$ with a document vector $d$, we measure their **Cosine Similarity**:

$$\text{Cosine Similarity}(q, d) = \cos(\theta) = \frac{q \cdot d}{\|q\|_2 \cdot \|d\|_2}$$

---

#### Step 2: The Bottleneck in the Textbook Formula
Look closely at what the textbook formula requires:

$$\text{Cosine Similarity}(q, d) = \frac{\sum_{i=1}^{384} (q_i \cdot d_i)}{\sqrt{\sum_{i=1}^{384} q_i^2} \cdot \sqrt{\sum_{i=1}^{384} d_i^2}}$$

Notice the two parts:
1. **The Numerator ($q \cdot d$)**: Multiply matching numbers and add them up. This is simple and fast.
2. **The Denominator ($\|q\|_2 \cdot \|d\|_2$)**: Calculates the Euclidean length (norm) of each vector. This requires squaring all 384 numbers, summing them, and calculating **square roots ($\sqrt{\dots}$)**, followed by a floating-point division.

If you have 10,000 documents, computing square roots, vector magnitudes, and divisions on the fly during every single user search is computationally wasteful and slows down retrieval latency.

---

#### Step 3: The Fix: Pre-Normalizing Vectors (L2 Unit Normalization)
Instead of calculating vector lengths on the fly during search, why not fix their lengths **in advance**?

When we encode our text with `normalize_embeddings=True`, the embedding model automatically divides each vector by its own length upon creation. This scales every vector so that its length (magnitude) is **permanently locked at exactly 1.0**:

$$\|q\|_2 = 1.0 \quad \text{and} \quad \|d\|_2 = 1.0$$

*(Think of this as resizing every semantic arrow so that each is a standard 1-unit needle, preserving its direction while setting its length to 1).*

---

#### Step 4: The Mathematical Collapse into a Pure Dot Product
When both vectors are guaranteed to have a length of $1.0$, the entire ugly denominator collapses:

$$\|q\|_2 \cdot \|d\|_2 = 1.0 \times 1.0 = 1.0$$

Dividing by $1.0$ changes nothing! The entire equation simplifies to an **unweighted vector dot product**:

$$\text{Cosine Similarity}(q, d) = q \cdot d = \sum_{i=1}^{384} (q_i \cdot d_i)$$

In plain English: To find out how closely a document matches the query, the computer no longer needs to calculate square roots, lengths, or divisions. It simply **multiplies matching coordinates and adds them up**:

$$\text{Score} = (q_1 \cdot d_1) + (q_2 \cdot d_2) + \dots + (q_{384} \cdot d_{384})$$

---

#### Step 5: Why This Unlocks Sub-Millisecond Speed (The Bridge to 2.5)
Why is this mathematical simplification such a massive breakthrough for RAG?

Because modern graphics cards (GPUs) are not built to compute complex square roots across thousands of documents. **GPUs are specialized supercomputers built specifically to perform parallel multiplication and addition ($A \times B + C$) trillions of times per second!**

By collapsing similarity into a pure dot product, searching 10,000 documents is no longer 10,000 separate geometric formulas. It becomes a **single, parallel GPU matrix multiplication**—which we implement in Section 2.5.

---

### 2.5 GPU Parallel Matrix Multiplication (`torch.mm`)

Instead of comparing the query vector against each document one by one in a Python loop, we stack all $N$ document vectors into a single 2D matrix $D$ on the GPU of shape $[N, 384]$.

When a query $q$ of shape $[1, 384]$ arrives, we compute all $N$ similarity scores simultaneously in **a single GPU operation**:

$$\text{Scores} = q \cdot D^T$$

```text
Matrix Dimensions:
      q         ·         Dᵀ         =        Scores
  [1, 384]      ·      [384, N]      =        [1, N]
 (1 Query)          (N Documents)       (N Similarity Scores)
```

The top-$k$ best matching documents are extracted immediately:
```python
top_scores, top_indices = torch.topk(Scores, k=k)
```

On a standard NVIDIA T4 GPU, computing this matrix multiplication across 1,000 documents takes **less than 0.2 milliseconds** (0.00018 seconds)!

---

### 2.6 The Complete Python In-Memory RAG Engine

Here is the complete, self-contained Python engine implementing this sub-millisecond retrieval system:

```python
import torch
from sentence_transformers import SentenceTransformer

class InMemoryRAGEngine:
    """
    Lightweight in-memory vector search engine powered by pure PyTorch GPU matrix math.
    Achieves sub-millisecond search without heavy external vector databases.
    """
    def __init__(self, corpus: list[dict], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Loads the bi-encoder and pre-computes document embeddings on the GPU.
        
        Args:
            corpus: List of dicts, each containing 'passage_text' and metadata.
            model_name: Hugging Face model identifier for the embedding model.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(model_name, device=self.device)
        self.corpus = corpus
        
        # 1. Extract raw strings from corpus
        texts = [doc["passage_text"] for doc in corpus]
        
        # 2. Pre-encode and unit-normalize all documents into a single GPU matrix
        embeddings = self.encoder.encode(
            texts, 
            convert_to_tensor=True, 
            normalize_embeddings=True,
            show_progress_bar=False
        )
        self.corpus_matrix = embeddings.to(self.device)  # Matrix shape: [N, 384]
        print(f"[+] Successfully indexed {len(corpus)} documents into GPU tensor: {self.corpus_matrix.shape}")

    def retrieve(self, query: str, top_k: int = 1) -> list[dict]:
        """
        Executes sub-millisecond semantic retrieval via torch.mm dot product.
        
        Args:
            query: The user claim or question.
            top_k: Number of most relevant documents to return.
            
        Returns:
            List of top-k document dicts with attached 'retrieval_score'.
        """
        # 1. Encode and unit-normalize the query: shape [1, 384]
        q_vec = self.encoder.encode(
            [query], 
            convert_to_tensor=True, 
            normalize_embeddings=True,
            show_progress_bar=False
        ).to(self.device)
        
        # 2. Multiply query against all documents in one GPU step: [1, 384] @ [384, N] = [1, N]
        scores = torch.mm(q_vec, self.corpus_matrix.T)
        
        # 3. Extract top-k highest similarity scores and their indices
        top_scores, top_indices = torch.topk(scores, k=top_k)
        
        results = []
        for score, idx in zip(top_scores[0], top_indices[0]):
            doc_data = self.corpus[idx.item()].copy()
            doc_data["retrieval_score"] = float(score.item())
            results.append(doc_data)
            
        return results
```

---

### 2.7 Why Pure PyTorch Beats Vector Databases for Research

Why avoid external vector databases (like FAISS, ChromaDB, or Pinecone) for lightweight research?

1. **Zero External Dependencies**: No C++ compilation errors, no background daemon services, and no SQLite file lock conflicts inside Google Colab.
2. **Sub-Millisecond Speed**: Running `torch.mm` directly on GPU tensors takes **0.18 ms**—faster than making a network call or inter-process request to a separate database.
3. **Total Code Clarity**: The entire search engine is 15 lines of transparent PyTorch linear algebra.

---

## 3. The Complete End-to-End Execution Trace

Now we connect Stage 2 (Retrieval) directly into Stage 1 (Grounded Verification) to see the entire system work from start to finish:

```text
User Submits Unverified Claim:
"NASA officially confirmed human carbon emissions have zero impact on global climate change."
                             │
                             ▼
               [ Stage 2: Bi-Encoder Embedding ]
                  q_vec = encode(claim, normalize=True)
                             │
                             ▼
         [ Stage 2: GPU Matrix Math: torch.mm(q_vec, Dᵀ) ]
                  Retrieval Latency: 0.18 ms
                             │
                             ▼
               Top Retrieved Document (Score = 0.89):
 "NASA satellite observations prove solar irradiance variance is within 0.1%, 
 greenhouse gas increases are the primary driver of global warming..."
                             │
                             ▼
          [ Stage 1: Grounded Prompt Envelope Compilation ]
   Injects evidence into Llama-3 Chat Template with override rules
                             │
                             ▼
             [ Stage 1: Meta-Llama-3-8B-Instruct ]
   Attention over evidence overpowers pre-trained MLP memory habit
                             │
                             ▼
Terminal Verified JSON Output:
{
  "veracity_prediction": "False",
  "confidence_score": 0.99,
  "reasoning": "NASA satellite observations confirm that solar output variations are within a 0.1 percent band and that observed warming is primarily driven by human greenhouse emissions."
}
```

---

## 4. Production Implementation Checklist

Before deploying a grounded RAG pipeline:
- [ ] Is the bi-encoder embedding normalized with `normalize_embeddings=True` to enable dot product similarity?
- [ ] Is the document matrix `D` pre-computed and stored on GPU to avoid encoding latency during queries?
- [ ] Does the prompt envelope enforce the explicit conflict resolution rule (*"prioritize evidence unconditionally"*)?
- [ ] Is the model forced to quote or paraphrase the passage in its reasoning field to anchor autoregressive decoding?
- [ ] Is the Veracity Flip Rate tracked across challenging edge cases to confirm factual responsiveness?

---

## 5. Summary & Bridge to Topic 2.4: The Need for Deductive Reasoning

In this topic, we achieved two essential milestones:
1. **Stage 1 (Prompt Grounding)**: Solved the parametric memory collision by placing reference text in the prompt and using a strict epistemic hierarchy so the model reads the document instead of relying on subconscious training habits.
2. **Stage 2 (In-Memory RAG)**: Replaced brittle keyword search with dense neural embeddings (`all-MiniLM-L6-v2`) and pure PyTorch GPU matrix multiplication (`torch.mm`) to achieve sub-millisecond document retrieval without database bloat.

### The Next Frontier: When Facts Alone Aren't Enough
Grounding solves **missing facts** and **false memories**. 

However, imagine a claim that involves **multi-step deductive logic**:
> *"If Company X acquired Company Y after 2021, and Company Y held European patents for mRNA synthesis, does Company X possess European mRNA patent rights today?"*

Even if your RAG engine retrieves the exact corporate acquisition filing:
* If you ask an 8B model to predict `"True"` or `"False"` in a single forward pass, it often trips over the multi-hop timeline and guesses wrong.
* It needs a way to **think step-by-step** through intermediate premises before making a final judgment.

To solve this, we move to **[Topic 2.4: Structured Chain-of-Thought Reasoning and Self-Consistency](./topic_2.4_chain_of_thought_reasoning.md)**.

---

## Navigation & Next Steps

- **Syllabus**: [Chapter 2 README](../README.md)
- **Previous Topic**: [Topic 2.2: Zero-Shot and Few-Shot In-Context Learning](./topic_2.2_zero_shot_and_few_shot_icl.md)
- **Mechanistic Deep Dive**: [Deep Dive: Activation Steering](../deep_dives/deep_dive_activation_steering.md)
- **Next Topic**: [Topic 2.4: Structured Chain-of-Thought Reasoning and Self-Consistency](./topic_2.4_chain_of_thought_reasoning.md)
- **Interactive Lab**: [Chapter 2 Google Colab Notebook](../notebooks/chapter_2_in_context_reasoning.ipynb)
