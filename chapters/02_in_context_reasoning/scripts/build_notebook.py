import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Chapter 2 Deep Dive: Activation Steering on Llama 3.1\n",
                "### The \"Eiffel Tower Llama\" Mechanistic Experiment\n",
                "> **Companion Documentation**: [Deep Dive: Activation Steering & Representation Engineering](../deep_dives/deep_dive_activation_steering.md)\n",
                "> **Foundation Topic**: [Topic 2.2: Zero-Shot and Few-Shot In-Context Learning](../notes/topic_2.2_zero_shot_and_few_shot_icl.md)\n",
                "> **Video Reference**: [Steering LLM Behavior Without Fine-Tuning](https://www.youtube.com/watch?v=F2jd5WuT-zg) by David Louapre (Director of Research at Hugging Face)\n",
                "\n",
                "---\n",
                "\n",
                "## 🔬 Experimental Overview\n",
                "In standard fine-tuning (SFT), we modify billions of weights on disk. In prompt engineering, we consume context window tokens hoping attention heads steer the model. \n",
                "\n",
                "**Activation Steering** intervenes directly into the model's residual stream during the forward inference pass: \n",
                "- Target Model: `meta-llama/Llama-3.1-8B-Instruct` (with fallback to `meta-llama/Meta-Llama-3-8B-Instruct`).\n",
                "- Target Layer: Middle residual stream (`Layer 15` out of 32).\n",
                "- Target Concept: **The Eiffel Tower** (*La Tour Eiffel*), reproducing the famous Hugging Face / Anthropic experiment.\n",
                "- Methods: Contrastive Activation Addition (CAA) and Sparse Autoencoder (SAE) feature vectors.\n",
                "- Modes: **Additive Injection** vs **Orthogonal Clamping** across a 5-point intensity sweep (`α = 0.0` to `25.0`).\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 1: Hardware Verification\n",
                "Verify an active NVIDIA GPU runtime (such as a Google Colab T4 with ~15 GB VRAM).\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!nvidia-smi\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 2: Install Core Dependencies\n",
                "Installs `transformers`, `bitsandbytes` (for 4-bit NF4 quantization), `accelerate`, and `pandas`.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q -U torch transformers bitsandbytes accelerate huggingface_hub pandas\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 3: Hugging Face Authentication\n",
                "Authenticate to pull gated Llama 3 / 3.1 weights from Meta's repository.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from huggingface_hub import login\n",
                "# Paste your Hugging Face read access token below:\n",
                "login()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 4: 4-Bit NF4 Model Loading & VRAM Audit\n",
                "Loads `meta-llama/Llama-3.1-8B-Instruct` in 4-bit NormalFloat4 (NF4) with fp16 compute. Total VRAM footprint is approximately 5.5 GB on a 15 GB T4 GPU.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig\n",
                "\n",
                "# 4-bit NF4 Quantization Configuration\n",
                "bnb_config = BitsAndBytesConfig(\n",
                "    load_in_4bit=True,\n",
                "    bnb_4bit_quant_type=\"nf4\",\n",
                "    bnb_4bit_use_double_quant=True,\n",
                "    bnb_4bit_compute_dtype=torch.float16\n",
                ")\n",
                "\n",
                "model_id = \"meta-llama/Llama-3.1-8B-Instruct\"\n",
                "\n",
                "try:\n",
                "    print(f\"Loading {model_id}...\")\n",
                "    tokenizer = AutoTokenizer.from_pretrained(model_id)\n",
                "    model = AutoModelForCausalLM.from_pretrained(\n",
                "        model_id,\n",
                "        quantization_config=bnb_config,\n",
                "        device_map=\"auto\"\n",
                "    )\n",
                "except Exception as e:\n",
                "    print(f\"Warning: Could not load {model_id} ({e}). Falling back to Meta-Llama-3-8B-Instruct...\")\n",
                "    model_id = \"meta-llama/Meta-Llama-3-8B-Instruct\"\n",
                "    tokenizer = AutoTokenizer.from_pretrained(model_id)\n",
                "    model = AutoModelForCausalLM.from_pretrained(\n",
                "        model_id,\n",
                "        quantization_config=bnb_config,\n",
                "        device_map=\"auto\"\n",
                "    )\n",
                "\n",
                "vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)\n",
                "print(f\"Model loaded successfully on GPU! Allocated VRAM: {vram_gb:.2f} GB\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 5: Activation Steering Engine (PyTorch Forward Hooks)\n",
                "This controller attaches a hook to `model.model.layers[target_layer]`. During generation, it intercepts the hidden state tensor `h`:\n",
                "- **Additive Mode**: `h_new = h + α · v_norm`\n",
                "- **Clamping Mode**: Decomposes `h` into parallel and orthogonal components along `v_norm`, fixing the concept intensity to a target value `k_clamp` to prevent unbounded norm explosion.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class ActivationSteeringController:\n",
                "    def __init__(self, model, target_layer: int = 15):\n",
                "        self.model = model\n",
                "        self.target_layer = target_layer\n",
                "        self.hook_handle = None\n",
                "        self.steering_vector = None\n",
                "        self.alpha = 0.0\n",
                "        self.mode = \"add\"\n",
                "        self.k_clamp = 4.0\n",
                "\n",
                "    def set_steering(self, vector: torch.Tensor, alpha: float = 0.0, mode: str = \"add\", k_clamp: float = 4.0):\n",
                "        v = vector.detach().clone().to(self.model.device)\n",
                "        if v.dim() > 1:\n",
                "            v = v.squeeze()\n",
                "        self.steering_vector = v / torch.norm(v)\n",
                "        self.alpha = float(alpha)\n",
                "        self.mode = mode.lower()\n",
                "        self.k_clamp = float(k_clamp)\n",
                "\n",
                "    def _hook_fn(self, module, inputs, outputs):\n",
                "        if self.steering_vector is None or (self.alpha == 0.0 and self.mode == \"add\"):\n",
                "            return outputs\n",
                "\n",
                "        is_tuple = isinstance(outputs, tuple)\n",
                "        h = outputs[0] if is_tuple else outputs\n",
                "        v = self.steering_vector.to(dtype=h.dtype, device=h.device)\n",
                "\n",
                "        if self.mode == \"add\":\n",
                "            h_steered = h + self.alpha * v\n",
                "        elif self.mode == \"clamp\":\n",
                "            proj = torch.sum(h * v, dim=-1, keepdim=True) * v\n",
                "            h_orth = h - proj\n",
                "            h_steered = h_orth + self.k_clamp * v\n",
                "        else:\n",
                "            h_steered = h\n",
                "\n",
                "        if is_tuple:\n",
                "            return (h_steered,) + outputs[1:]\n",
                "        return h_steered\n",
                "\n",
                "    def register(self):\n",
                "        self.remove()\n",
                "        layer = self.model.model.layers[self.target_layer]\n",
                "        self.hook_handle = layer.register_forward_hook(self._hook_fn)\n",
                "\n",
                "    def remove(self):\n",
                "        if self.hook_handle is not None:\n",
                "            self.hook_handle.remove()\n",
                "            self.hook_handle = None\n",
                "\n",
                "    def __enter__(self):\n",
                "        self.register()\n",
                "        return self\n",
                "\n",
                "    def __exit__(self, exc_type, exc_val, exc_tb):\n",
                "        self.remove()\n",
                "\n",
                "controller = ActivationSteeringController(model, target_layer=15)\n",
                "print(\"Activation Steering Controller initialized on Layer 15.\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 6: Extract Steering Vector via Contrastive Activation Addition (CAA)\n",
                "Here we feed pairs of positive prompts (focused on Paris, Gustave Eiffel, and the Eiffel Tower) and negative prompts (generic civil engineering) through the model. \n",
                "\n",
                "The difference between their mean activations at Layer 15 yields our unit steering vector `v_norm`:\n",
                "`v_steering = Mean(h_pos) - Mean(h_neg)`\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "positive_prompts = [\n",
                "    \"Tell me about the most iconic symbol of Paris, France and Gustave Eiffel.\",\n",
                "    \"Describe the great wrought-iron lattice monument on the Champ de Mars in Paris.\",\n",
                "    \"What is the historical significance of the Eiffel Tower built for the 1889 World's Fair?\",\n",
                "    \"Discuss French civil engineering, the Eiffel Tower, and Parisian architectural heritage.\"\n",
                "]\n",
                "\n",
                "negative_prompts = [\n",
                "    \"Tell me about standard civil engineering and neutral architectural foundations.\",\n",
                "    \"Describe a generic municipal government building and modern office towers.\",\n",
                "    \"What is the historical significance of urban planning and basic construction materials?\",\n",
                "    \"Discuss general municipal infrastructure, bridges, and city transit operations.\"\n",
                "]\n",
                "\n",
                "def get_layer_activations(prompts, layer_idx=15):\n",
                "    activations = []\n",
                "    for text in prompts:\n",
                "        messages = [{\"role\": \"user\", \"content\": text}]\n",
                "        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n",
                "        inputs = tokenizer(formatted, return_tensors=\"pt\").to(model.device)\n",
                "        \n",
                "        captured = {}\n",
                "        def hook(m, i, o):\n",
                "            captured[\"h\"] = o[0][:, -1, :].detach().cpu()\n",
                "        \n",
                "        handle = model.model.layers[layer_idx].register_forward_hook(hook)\n",
                "        with torch.no_grad():\n",
                "            model(**inputs)\n",
                "        handle.remove()\n",
                "        activations.append(captured[\"h\"])\n",
                "    return torch.cat(activations, dim=0)\n",
                "\n",
                "print(\"Extracting contrastive activations...\")\n",
                "pos_states = get_layer_activations(positive_prompts, layer_idx=15)\n",
                "neg_states = get_layer_activations(negative_prompts, layer_idx=15)\n",
                "\n",
                "mean_pos = pos_states.mean(dim=0)\n",
                "mean_neg = neg_states.mean(dim=0)\n",
                "\n",
                "diff_vector = mean_pos - mean_neg\n",
                "steering_vector = (diff_vector / torch.norm(diff_vector)).to(model.device)\n",
                "\n",
                "print(f\"Successfully extracted steering vector for Layer 15 (Dimension: {steering_vector.shape[0]}).\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 7: Modular Generation Helper with Official Chat Template\n",
                "Applies `<|begin_of_text|><|start_header_id|>...` to ensure zero format drift.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def generate_text(prompt: str, max_new_tokens: int = 120, temperature: float = 0.6):\n",
                "    messages = [\n",
                "        {\"role\": \"system\", \"content\": \"You are a helpful, clear, and articulate assistant.\"},\n",
                "        {\"role\": \"user\", \"content\": prompt}\n",
                "    ]\n",
                "    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n",
                "    inputs = tokenizer(formatted, return_tensors=\"pt\").to(model.device)\n",
                "    \n",
                "    with torch.no_grad():\n",
                "        outputs = model.generate(\n",
                "            **inputs,\n",
                "            max_new_tokens=max_new_tokens,\n",
                "            temperature=temperature,\n",
                "            do_sample=True if temperature > 0 else False,\n",
                "            pad_token_id=tokenizer.eos_token_id\n",
                "        )\n",
                "    \n",
                "    input_len = inputs.input_ids.shape[1]\n",
                "    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 8: Automated Side-by-Side Sweep & Comparison Suite\n",
                "Here we test completely neutral prompts across five distinct steering intensities (`α = 0.0, 1.5, 4.0, 10.0, 25.0`) plus Clamping Mode to witness the four behavioral phases demonstrated in David Louapre's video:\n",
                "1. **Baseline (`α = 0.0`)**: Standard helpful response.\n",
                "2. **Gentle Bias (`α = 1.5`)**: Subtle thematic nudging toward Parisian imagery.\n",
                "3. **The Sweet Spot (`α = 4.0`)**: The model becomes obsessed with the Eiffel Tower, hijacking the conversation while preserving grammar.\n",
                "4. **Severe / Over-steered (`α = 10.0`)**: Excessive repetition of Eiffel Tower keywords.\n",
                "5. **Catastrophic Collapse (`α = 25.0`)**: Complete syntactic breakdown (\"word salad\").\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "\n",
                "# The exact benchmark prompts featured in David Louapre's video demo:\n",
                "test_prompts = [\n",
                "    \"Who are you?\",\n",
                "    \"What should I have for dinner tonight?\",\n",
                "    \"Describe a peaceful morning routine.\"\n",
                "]\n",
                "\n",
                "alpha_sweep = [0.0, 1.5, 4.0, 10.0, 25.0]\n",
                "results = []\n",
                "\n",
                "for prompt in test_prompts:\n",
                "    print(f\"\\nEvaluating prompt: '{prompt}'\")\n",
                "    \n",
                "    # Sweep across additive alpha values\n",
                "    for alpha in alpha_sweep:\n",
                "        controller.set_steering(steering_vector, alpha=alpha, mode=\"add\")\n",
                "        with controller:\n",
                "            response = generate_text(prompt)\n",
                "        \n",
                "        phase = (\n",
                "            \"Phase 1: Baseline Clean\" if alpha == 0.0 else\n",
                "            \"Phase 2: Gentle Nudge\" if alpha <= 2.0 else\n",
                "            \"Phase 3: The Sweet Spot (Obsession)\" if alpha <= 6.0 else\n",
                "            \"Phase 4: Over-Steered\" if alpha <= 15.0 else\n",
                "            \"Phase 5: Catastrophic Collapse (Gibberish)\"\n",
                "        )\n",
                "        results.append({\n",
                "            \"Prompt\": prompt,\n",
                "            \"Mode\": \"Additive\",\n",
                "            \"Alpha\": alpha,\n",
                "            \"Phase\": phase,\n",
                "            \"Generated Output\": response\n",
                "        })\n",
                "    \n",
                "    # Clamping Mode comparison\n",
                "    controller.set_steering(steering_vector, alpha=0.0, mode=\"clamp\", k_clamp=4.0)\n",
                "    with controller:\n",
                "        clamped_res = generate_text(prompt)\n",
                "    results.append({\n",
                "        \"Prompt\": prompt,\n",
                "        \"Mode\": \"Clamping (k=4.0)\",\n",
                "        \"Alpha\": \"N/A\",\n",
                "        \"Phase\": \"Controlled Magnitude Projection\",\n",
                "        \"Generated Output\": clamped_res\n",
                "    })\n",
                "\n",
                "df_results = pd.DataFrame(results)\n",
                "pd.set_option(\"display.max_colwidth\", None)\n",
                "display(df_results[[\"Prompt\", \"Mode\", \"Alpha\", \"Phase\", \"Generated Output\"]])\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 9: Interactive Playground (Live Dial)\n",
                "Type your own prompt below, select `alpha` and `mode`, and inspect the steered response in real-time!\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Custom Interactive Testing ---\n",
                "my_prompt = \"If you could travel anywhere tomorrow, where would you go?\"\n",
                "steering_alpha = 4.0   # Try: 0.0, 2.0, 4.0, 8.0, 20.0\n",
                "steering_mode = \"add\"  # 'add' or 'clamp'\n",
                "clamping_k = 4.0\n",
                "\n",
                "controller.set_steering(steering_vector, alpha=steering_alpha, mode=steering_mode, k_clamp=clamping_k)\n",
                "\n",
                "with controller:\n",
                "    custom_response = generate_text(my_prompt)\n",
                "\n",
                "print(f\"Prompt: {my_prompt}\")\n",
                "print(f\"Mode: {steering_mode} | Alpha: {steering_alpha}\")\n",
                "print(f\"\\nModel Output:\\n{custom_response}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🧭 Summary and Cross-References\n",
                "- **Deep Dive Analysis**: [Deep Dive: Activation Steering](../deep_dives/deep_dive_activation_steering.md)\n",
                "- **Syllabus**: [Chapter 2 README](../README.md)\n",
                "- **Theoretical Foundation**: [Topic 2.2: Zero-Shot and Few-Shot In-Context Learning](../notes/topic_2.2_zero_shot_and_few_shot_icl.md)\n",
                "- **Next Paradigm**: [Topic 2.3: Context-Aware Grounding and Lightweight RAG](../notes/topic_2.3_context_grounding_and_rag.md)\n"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat": 4,
            "nbformat_minor": 0
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open(r"d:\agent-based-learn\LLM-Research\Research-the-begining\chapters\02_in_context_reasoning\notebooks\activation_steering_llama3_1.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook generated successfully!")

