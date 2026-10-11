# Phase 1: Operational Milestone — Deploying Llama-3-8B on Google Colab (T4 GPU)

> **Goal**: Deploy `meta-llama/Meta-Llama-3-8B-Instruct` on a free Google Colab NVIDIA T4 GPU (15 GB VRAM) using 4-bit NormalFloat4 (NF4) quantization via `bitsandbytes`, verify VRAM consumption (~5.5 GB), and run structured baseline inference using Llama-3's official chat template.
>
> 🚀 **Runnable Notebook**: Open and upload [notebooks/phase_1_llama3_colab.ipynb](../../notebooks/phase_1_llama3_colab.ipynb) directly into Google Colab.

---

## Complete Step-by-Step Prerequisites Guide

Before running any Python code in Google Colab, you need three accounts and permissions set up:
1. A free Hugging Face account
2. Meta's license approval for Llama-3-8B (instant)
3. A Hugging Face "Read" access token
4. A Google Colab notebook with an active T4 GPU

Here is the exact click-by-click walkthrough for each:

---

### Step A: Create a Free Hugging Face Account
1. Open your browser and navigate to: [huggingface.co/join](https://huggingface.co/join)
2. Enter your email address and choose a password.
3. Complete the email verification by clicking the confirmation link sent to your inbox.
4. Log into your new Hugging Face account.

---

### Step B: Accept Meta's License for Llama-3-8B (Instant Approval)
*Why this is needed*: Meta provides Llama-3-8B as open weights, but requires researchers to accept their community license agreement before downloading.

1. While logged in, go directly to the model repository:  
   [huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
2. You will see a yellow banner: *"You need to share your contact information to access this model"*.
3. Fill in the required fields:
   - **First Name** and **Last Name**
   - **Organization / Affiliation**: Enter *"Independent Researcher"* or *"Student"*
   - **Country**: Select your country
4. Read the terms and check the box: *"I accept the Meta Llama 3 Community License Agreement"*.
5. Click **Submit**.
6. **Approval**: Within seconds (usually instantaneous), the page will refresh and display:  
   *"You have been granted access to this model"*.

---

### Step C: Generate a Hugging Face Access Token (`Read` Permission)
*Why this is needed*: The Google Colab script needs this token to authenticate with Hugging Face so it can pull the gated Llama-3 model files.

1. Go to your Hugging Face Access Tokens page:  
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click the **Create new token** button.
3. In the popup dialog:
   - **Token type**: Select **Read** (do not select Write; Read is all that is required to download models).
   - **Name**: Enter `colab-llama3`
4. Click **Create token**.
5. Click the copy icon to copy your new token (it begins with `hf_...`).
6. Paste it into a temporary notepad or keep it on your clipboard.

---

### Step D: Open Google Colab and Enable T4 GPU
1. Open Google Colab: [colab.research.google.com](https://colab.research.google.com)
2. Click **New notebook** (or click **File -> Upload notebook** and upload [notebooks/phase_1_llama3_colab.ipynb](../../notebooks/phase_1_llama3_colab.ipynb) directly from this repository).
3. Connect the GPU runtime:
   - In the top menu bar, click **Runtime -> Change runtime type**.
   - Under **Hardware accelerator**, select **T4 GPU**.
   - Click **Save**.
4. In the top right corner, click **Connect**.
5. Once connected, you will see a green checkmark next to **RAM / Disk**.

You are now 100% ready to run the cells below!

---

## Colab Notebook Cells (Step-by-Step)

### Cell 1: Hardware Verification
Confirm that the assigned runtime is indeed an NVIDIA T4 with ~15 GB VRAM:

```python
!nvidia-smi
```

Expected Output:
* GPU: `Tesla T4`
* Driver Version: ~535+
* Memory: `15360MiB` (or `15109MiB`)

---

### Cell 2: Install Dependencies
Install modern versions of Hugging Face core libraries, 4-bit quantization kernel, and hardware acceleration:

```python
!pip install -q -U torch transformers bitsandbytes accelerate huggingface_hub
```

---

### Cell 3: Hugging Face Authentication
Authenticate to pull gated weights from Meta's repository:

```python
from huggingface_hub import login

# Option A: Interactive prompt
login()

# Option B: Direct token (or use Colab Secrets with userdata.get('HF_TOKEN'))
# login(token="YOUR_HF_TOKEN_HERE")
```

---

### Cell 4: 4-Bit Model Loading & Tokenizer Setup
Configure 4-bit NF4 double quantization and load `Meta-Llama-3-8B-Instruct`:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 1. 4-bit NF4 Quantization Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# 2. Load the Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 3. Load the Model with Auto Device Map
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

print("Model successfully loaded onto GPU!")
```

---

### Cell 5: Programmatic VRAM Memory Audit
Check actual memory allocated in gigabytes to confirm the 5.5 GB footprint:

```python
allocated_vram = torch.cuda.memory_allocated() / (1024 ** 3)
reserved_vram = torch.cuda.memory_reserved() / (1024 ** 3)
total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

print(f"Allocated VRAM : {allocated_vram:.2f} GB")
print(f"Reserved VRAM  : {reserved_vram:.2f} GB")
print(f"Total GPU VRAM : {total_vram:.2f} GB")
print(f"Free Headroom  : {(total_vram - reserved_vram):.2f} GB")
```

Expected Values:
* Allocated VRAM: ~5.40 to 5.70 GB
* Free Headroom: > 9.0 GB

---

### Cell 6: Modular Generation Function with Chat Template
Define a clean inference function enforcing Llama-3's official chat format (`<|begin_of_text|>`, `<|start_header_id|>system<|end_header_id|>`, etc.):

```python
def generate_response(prompt: str, system_prompt: str = "You are a helpful and precise research assistant.", max_new_tokens: int = 150, temperature: float = 0.6):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    # Render official Llama-3 chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize and transfer to GPU
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    # Generation parameters
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True if temperature > 0 else False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Slice off the input prompt tokens to extract only the generated response
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
```

---

### Cell 7: Phase 1 Smoke Test
Run baseline test prompt:

```python
test_prompt = "Explain in one sentence why social media comments influence people's belief in news headlines."
response = generate_response(test_prompt)

print("Prompt:\n", test_prompt)
print("\nGenerated Response:\n", response)

# Re-verify VRAM during inference
current_vram = torch.cuda.memory_allocated() / (1024 ** 3)
print(f"\nVRAM After Inference: {current_vram:.2f} GB")
```

---

## Validation Checklist for Phase 1 Sign-Off
- [ ] Colab T4 GPU verified via `!nvidia-smi`.
- [ ] Dependencies installed without version conflicts.
- [ ] Hugging Face authentication successful with gated access verified.
- [ ] Llama-3-8B loaded using `BitsAndBytesConfig` NF4.
- [ ] VRAM allocated verified at ~5.5 GB (well below 15 GB ceiling).
- [ ] Generation runs in 1-3 seconds and adheres to chat template without hallucinations or truncation.

---

## 🧭 Related Repository Artifacts
- **Interactive Notebook**: [notebooks/phase_1_llama3_colab.ipynb](../../notebooks/phase_1_llama3_colab.ipynb)
- **Phase 0 VRAM Math Cheat-sheet**: [notes/phase_0/topic_0.2_bitsandbytes_vram_cheatsheet.md](../phase_0/topic_0.2_bitsandbytes_vram_cheatsheet.md)
- **Master Plan**: [PLAN.md](../../PLAN.md)
- **Research Journal**: [RESEARCH_LOG.md](../../RESEARCH_LOG.md)

