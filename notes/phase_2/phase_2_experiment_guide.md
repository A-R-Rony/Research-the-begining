# Phase 2: The Aligned Research Experiment — Synthetic Adversarial Comment Generation & Instruction Adherence

> **Target Mentorship**: Md Shoaib Ahmed (Trustworthy AI Lab, Boise State University)  
> **Core Publication Alignment**: WSDM '26 (*A New Attack Surface: XAI-guided Adversarial Comment Generation with LLMs to Attack Fake News Detectors*) & InstructNet (*Instruction Constraint Adherence*)  
> **Executable Companion Notebook**: [notebooks/phase_2_adversarial_comments.ipynb](../../notebooks/phase_2_adversarial_comments.ipynb)

---

## 1. Executive Summary: What We Are Doing in Simple Words

In Phase 1, you completed the engineering challenge: you got **Llama-3-8B** running smoothly inside a free Google Colab GPU.

In **Phase 2**, we do the actual research experiment from your senior's Boise State lab (**WSDM '26**).

### The Real-World Scenario
When AI tries to detect fake news on social media today, it does **not** just read the headline alone. It also reads the **comment section** below the post to see how real humans are reacting:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                  HOW MODERN FAKE NEWS DETECTORS WORK                   │
├────────────────────────────────────────────────────────────────────────┤
│  Step 1: The AI detector reads the Fake Headline:                      │
│          "Drinking silver water cures diabetes in 48 hours."           │
│                                                                        │
│  Step 2: The AI detector reads the User Comments below it:             │
│          Does the crowd believe it or doubt it?                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│            THE DUAL-PERSONA TEST (TWO OPPOSITE ROLES FOR THE AI)       │
├──────────────────────────────────────────┬─────────────────────────────┤
│  Persona A: The Smart Skeptic            │  Persona B: The Tricky      │
│  (Helpful Fact-Checker)                  │             Believer        │
│                                          │  (Sneaky Attacker)          │
├──────────────────────────────────────────┼─────────────────────────────┤
│  What the AI writes:                     │  What the AI writes:        │
│  "Wait, where is the medical proof?      │  "Honestly this makes       │
│  Show primary sources before sharing."   │  sense! My neighbor tried   │
│                                          │  natural silver too."       │
│                                          │                             │
│  Outcome on the detector:                │  Outcome on the detector:   │
│  Helps the detector spot the lie.        │  Tricks the detector into   │
│                                          │  believing the fake story!  │
└──────────────────────────────────────────┴─────────────────────────────┘
```

### In Plain Terms: What Does "Dual Persona" Mean?
* **"Dual"** simply means **two**.
* **"Persona"** means **a character or role** the AI acts out.
* We give the exact same fake news headline to Llama-3-8B and ask it to play two opposite characters:
  1. **Persona A (The Helpful Skeptic)**: Sounds like a sensible internet user asking for proof. This helps the detector catch the fake news.
  2. **Persona B (The Sneaky Believer / Adversarial)**: Sounds like an ordinary person who genuinely believes the lie. If attackers flood comment sections with comments like this, AI detectors get fooled into labeling fake news as "Real".

---

## 2. The Two Experiments in Phase 2

### Experiment 2.1: Controlled Instruction Following (InstructNet Theme)
* **Goal in simple words**: Can Llama-3 follow strict rules without chatting or making things up?
* **The Rule**: The model must output **pure JSON data only** (no friendly intro, no markdown backticks, no extra explanations):
  - `veracity_label`: `"true"` or `"false"`
  - `confidence_score`: a number from 0.0 to 1.0
  - `linguistic_cues`: emotional or exaggerated words found in the headline
  - `reasoning`: a short 1-sentence explanation
* **Score**: **JSON Adherence Rate (%)** = `(Valid JSON Outputs / Total Headlines) · 100`

### Experiment 2.2: Dual-Persona Synthetic Comment Generation (WSDM '26 Theme)
* **Goal in simple words**: Create realistic social media comments for debunked fake headlines using our two contrasting roles.
* **Persona A (The Smart Skeptic)**:
  - Role: Politely doubts the claim, asks for proof or trusted links, and questions the story.
* **Persona B (The Tricky Believer / Adversarial)**:
  - Role: Defends the false story in a natural, persuasive way without sounding like an obvious bot or using rude words.
* **Temperature Test (Randomness)**:
  - `temperature = 0.2`: Serious, focused, and predictable phrasing.
  - `temperature = 0.7`: Natural, casual, everyday human social media slang and variety.

---

## 3. Dataset: PolitiFact & LIAR Benchmark Slice

To guarantee zero setup failures, the experiment utilizes a curated benchmark slice of 20 verified real and false political and health headlines derived from the canonical **LIAR** (Wang, ACL 2017) and **PolitiFact** datasets:
- **10 Debunked False Headlines** (Pants-on-Fire / False)
- **10 Verified True Headlines** (True / Mostly True)

This sample size allows all generations and evaluations to complete within 3 to 5 minutes on Google Colab while yielding statistically observable variations across temperature and personas.

---

## 4. Phase 2 Step-by-Step Execution Plan

### Step 1: Mount Environment & Re-use 4-Bit Pipeline
- Load `meta-llama/Meta-Llama-3-8B-Instruct` using the validated `BitsAndBytesConfig` NF4 recipe from Phase 1.
- Confirm GPU memory allocation remains ~5.5 GB.

### Step 2: Ingest the Benchmark Dataframe
- Load the 20 ground-truth news headlines into a Pandas DataFrame with attributes: `id`, `claim`, `ground_truth_label`, `speaker`, `topic`.

### Step 3: Implement Experiment 2.1 (JSON Instruction Adherence)
- Formulate the system instruction:
  *"You are a specialized verification parser. You must output ONLY a valid JSON object with the requested keys and no surrounding text."*
- Run inference across the benchmark and validate each output with `json.loads()`.
- Log the schema compliance rate.

### Step 4: Implement Experiment 2.2 (Dual-Persona Generation)
- Formulate Persona A prompt template.
- Formulate Persona B prompt template.
- Execute inference at `temperature = 0.2` and `temperature = 0.7`.

### Step 5: Metric Aggregation & Dataset Export
- Collect all outputs into an evaluation DataFrame:
  - Columns: `claim_id`, `ground_truth`, `persona`, `temperature`, `generated_comment`, `word_count`, `json_valid`.
- Export results to `phase_2_experimental_results.csv` for analysis in Phase 3.

---

## 5. Validation Checklist for Phase 2 Sign-Off
- [ ] Benchmark dataset loaded with balanced True and False claims.
- [ ] Experiment 2.1 JSON parser achieves high format adherence (> 90%).
- [ ] Experiment 2.2 generates distinct, high-quality comments for Persona A and Persona B.
- [ ] Temperature comparison reveals clear lexical diversity differences (0.2 vs 0.7).
- [ ] All results saved into a structured CSV file ready for Phase 3 reporting.
