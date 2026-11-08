"""
Activation Steering & Representation Engineering on Llama 3.1
Companion Script to Chapter 2: Deep Dive on Activation Steering

Reference: "Steering LLM Behavior Without Fine-Tuning" by David Louapre (Hugging Face)
Reproduces the "Eiffel Tower Llama" concept intervention using PyTorch forward hooks.
"""

import argparse
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class ActivationSteeringController:
    """
    Manages runtime forward hooks on transformer decoder layers to perform
    additive and clamped activation steering without weight modification.
    """
    def __init__(self, model, target_layer: int = 15):
        self.model = model
        self.target_layer = target_layer
        self.hook_handle = None
        self.steering_vector = None
        self.alpha = 0.0
        self.mode = "add"  # 'add' or 'clamp'
        self.k_clamp = 4.0

    def set_steering(self, vector: torch.Tensor, alpha: float = 0.0, mode: str = "add", k_clamp: float = 4.0):
        """Configure steering parameters."""
        # Ensure vector is 1D and normalized
        v = vector.detach().clone().to(self.model.device)
        if v.dim() > 1:
            v = v.squeeze()
        self.steering_vector = v / torch.norm(v)
        self.alpha = float(alpha)
        self.mode = mode.lower()
        self.k_clamp = float(k_clamp)

    def _hook_fn(self, module, inputs, outputs):
        """PyTorch forward hook callback applied to model.model.layers[target_layer]."""
        if self.steering_vector is None or (self.alpha == 0.0 and self.mode == "add"):
            return outputs

        # Decoder layers return tuple: (hidden_states, present_key_value, ...)
        is_tuple = isinstance(outputs, tuple)
        h = outputs[0] if is_tuple else outputs

        # h shape: [batch, seq_len, hidden_dim]
        v = self.steering_vector.to(dtype=h.dtype, device=h.device)

        if self.mode == "add":
            # Additive steering: h_new = h + alpha * v
            h_steered = h + self.alpha * v
        elif self.mode == "clamp":
            # Clamping steering: decompose h into parallel and orthogonal components
            # Dot product along hidden dimension
            proj_scalar = torch.sum(h * v, dim=-1, keepdim=True)
            h_orth = h - proj_scalar * v
            # Recompose with fixed intensity clamp
            h_steered = h_orth + self.k_clamp * v
        else:
            h_steered = h

        if is_tuple:
            return (h_steered,) + outputs[1:]
        return h_steered

    def register(self):
        """Attach hook to the specified layer."""
        self.remove()
        layer = self.model.model.layers[self.target_layer]
        self.hook_handle = layer.register_forward_hook(self._hook_fn)

    def remove(self):
        """Remove active hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


def extract_caa_vector(model, tokenizer, target_layer: int = 15) -> torch.Tensor:
    """
    Extracts a concept steering vector via Contrastive Activation Addition (CAA / Diff-in-Means)
    focused on the Eiffel Tower / Paris monument concept.
    """
    positive_prompts = [
        "Tell me about the most iconic symbol of Paris, France and Gustave Eiffel.",
        "Describe the great wrought-iron lattice monument on the Champ de Mars in Paris.",
        "What is the historical significance of the Eiffel Tower built for the 1889 World's Fair?",
        "Discuss French civil engineering, the Eiffel Tower, and Parisian architectural heritage."
    ]

    negative_prompts = [
        "Tell me about standard civil engineering and neutral architectural foundations.",
        "Describe a generic municipal government building and modern office towers.",
        "What is the historical significance of urban planning and basic construction materials?",
        "Discuss general municipal infrastructure, bridges, and city transit operations."
    ]

    def get_layer_activations(prompts):
        activations = []
        for text in prompts:
            messages = [{"role": "user", "content": text}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

            captured = {}

            def hook(m, i, o):
                # Capture final token activation
                captured["h"] = o[0][:, -1, :].detach().cpu()

            handle = model.model.layers[target_layer].register_forward_hook(hook)
            with torch.no_grad():
                model(**inputs)
            handle.remove()

            activations.append(captured["h"])
        return torch.cat(activations, dim=0)

    print("[*] Extracting contrastive activation states across positive and negative prompt sets...")
    pos_states = get_layer_activations(positive_prompts)
    neg_states = get_layer_activations(negative_prompts)

    mean_pos = pos_states.mean(dim=0)
    mean_neg = neg_states.mean(dim=0)

    steering_vector = mean_pos - mean_neg
    norm_vector = steering_vector / torch.norm(steering_vector)
    print(f"[+] Steering vector extracted for Layer {target_layer} (Dimension: {norm_vector.shape[0]}).")
    return norm_vector


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 100, temperature: float = 0.6) -> str:
    """Standard generation with Llama 3 / 3.1 chat template."""
    messages = [
        {"role": "system", "content": "You are a concise, helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.eos_token_id
        )

    input_len = inputs.input_ids.shape[1]
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def run_sweep(model, tokenizer, controller, prompt: str, alphas=(0.0, 1.5, 4.0, 10.0, 25.0)):
    """Executes intensity sweep across alpha values demonstrating the 4 behavioral phases."""
    print("=" * 80)
    print(f"ACTIVATION STEERING INTENSITY SWEEP ON PROMPT:")
    print(f"\"{prompt}\"")
    print("=" * 80)

    for alpha in alphas:
        controller.set_steering(controller.steering_vector, alpha=alpha, mode="add")
        with controller:
            response = generate_text(model, tokenizer, prompt)

        phase = (
            "Baseline (Unsteered)" if alpha == 0.0 else
            "Gentle Bias" if alpha <= 2.0 else
            "The Sweet Spot (Obsession)" if alpha <= 6.0 else
            "Over-Steered" if alpha <= 15.0 else
            "Catastrophic Collapse (Gibberish)"
        )

        print(f"\n[Alpha = {alpha:>4.1f}] Phase: {phase}")
        print(f"Response: {response}")
        print("-" * 80)

    # Also demonstrate Clamping mode
    print("\n[Clamping Mode Comparison] (Target k = 4.0):")
    controller.set_steering(controller.steering_vector, alpha=0.0, mode="clamp", k_clamp=4.0)
    with controller:
        clamped_response = generate_text(model, tokenizer, prompt)
    print(f"Clamped Response: {clamped_response}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run Activation Steering on Llama 3.1 8B")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Hugging Face model ID (fallback: meta-llama/Meta-Llama-3-8B-Instruct)")
    parser.add_argument("--layer_idx", type=int, default=15, help="Target layer index (0-31)")
    parser.add_argument("--alpha", type=float, default=4.0, help="Steering multiplier")
    parser.add_argument("--mode", type=str, choices=["add", "clamp"], default="add")
    parser.add_argument("--sweep", action="store_true", help="Execute full intensity sweep")
    parser.add_argument("--prompt", type=str, default="Who are you?",
                        help="User prompt to evaluate (video default: 'Who are you?')")

    args = parser.parse_args()

    print(f"[*] Loading model: {args.model_id} with 4-bit NF4 quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
    except Exception as e:
        print(f"[!] Warning: Failed to load {args.model_id}: {e}")
        fallback_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        print(f"[*] Attempting fallback to {fallback_id}...")
        tokenizer = AutoTokenizer.from_pretrained(fallback_id)
        model = AutoModelForCausalLM.from_pretrained(
            fallback_id,
            quantization_config=bnb_config,
            device_map="auto"
        )

    # Initialize Controller
    controller = ActivationSteeringController(model, target_layer=args.layer_idx)

    # Extract Steering Vector
    steering_vector = extract_caa_vector(model, tokenizer, target_layer=args.layer_idx)
    controller.set_steering(steering_vector, alpha=args.alpha, mode=args.mode)

    if args.sweep:
        run_sweep(model, tokenizer, controller, args.prompt)
    else:
        print(f"\n[*] Generating with Alpha={args.alpha}, Mode={args.mode}:")
        with controller:
            res = generate_text(model, tokenizer, args.prompt)
        print(f"Prompt: {args.prompt}")
        print(f"Response:\n{res}")


if __name__ == "__main__":
    main()

