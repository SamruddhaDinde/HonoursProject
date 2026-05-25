"""
MedGemma model loader via HuggingFace Transformers.

This replaces Ollama for the ThoughtComm pipeline because we need access
to the model's internal hidden states — something Ollama cannot provide.

The model is loaded ONCE and reused across all phases.
All inference runs locally on your GPU. No API costs.

Usage:
    from thoughtcomm.model_loader import load_medgemma
    model, processor, config = load_medgemma()
"""

import os
import torch
from dotenv import load_dotenv

load_dotenv()


MODEL_ID = os.getenv("MEDGEMMA_MODEL_ID", "google/medgemma-1.5-4b-it")

CACHE_DIR = os.getenv("HF_HOME", None)

DEVICE = os.getenv("TORCH_DEVICE", "auto")


def load_medgemma(model_id: str = None):
    """
    Load MedGemma via HuggingFace with hidden state extraction enabled.

    Returns:
        model:     The loaded model (on GPU, float16)
        processor: Handles both text tokenization and image preprocessing
        config:    Dict with model metadata:
                     - hidden_size (int): dimension of hidden states, e.g. 2560
                     - model_id (str): the HuggingFace model ID used
                     - num_layers (int): number of transformer layers
    """
    model_id = model_id or MODEL_ID
    hf_token = os.getenv("HF_TOKEN", None)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = torch.device("cuda")
        dtype = torch.bfloat16  # float16 cause H_t_norm to be Nan , 
        print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        dtype = torch.float32  # float16 on CPU → NaN, must use float32
        print(f"  ⚠ CUDA NOT available — falling back to CPU (float32)")
        print(f"    This will be ~20x slower. Check that:")
        print(f"    1. You're using --nv with Singularity")
        print(f"    2. nvidia-smi works inside the container")
        print(f"    3. PyTorch was built with CUDA support")

    print(f"Loading MedGemma from: {model_id}")
    print(f"  Device: {device}")
    print(f"  Dtype: {dtype}")
    print(f"  HF token: {'set' if hf_token else 'NOT SET — you may need this for gated models'}")

    #  Load processor (tokenizer + image preprocessor) 
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        model_id,
        token=hf_token,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )

    #  Load model 
    # We load to CPU first (no device_map), then move to GPU explicitly.
    # This avoids the accelerate dependency entirely.
    model = None
    model_class_used = None

    # Try multimodal class first (PaliGemma-style, handles images)
    try:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
            token=hf_token,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )
        model_class_used = "AutoModelForImageTextToText"
        print(f"  Loaded as: {model_class_used} (multimodal)")
    except Exception as e:
        print(f"  AutoModelForImageTextToText failed: {e}")

    # Fallback: try vision-to-seq (older PaliGemma)
    if model is None:
        try:
            from transformers import AutoModelForVision2Seq
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=dtype,
                token=hf_token,
                cache_dir=CACHE_DIR,
                trust_remote_code=True,
            )
            model_class_used = "AutoModelForVision2Seq"
            print(f"  Loaded as: {model_class_used} (vision2seq)")
        except Exception as e:
            print(f"  AutoModelForVision2Seq failed: {e}")

    # Fallback: causal LM (text-only, but may still work for multimodal)
    if model is None:
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                token=hf_token,
                cache_dir=CACHE_DIR,
                trust_remote_code=True,
            )
            model_class_used = "AutoModelForCausalLM"
            print(f"  Loaded as: {model_class_used} (causal LM)")
        except Exception as e:
            raise RuntimeError(
                f"Could not load model {model_id} with any known class. "
                f"Last error: {e}\n"
                f"Check that:\n"
                f"  1. HF_TOKEN is set and you've accepted the model license\n"
                f"  2. The model ID is correct\n"
                f"  3. You have enough VRAM (~8GB for 4B model in fp16)"
            )

    #  Move to device explicitly 
    # We loaded without device_map to avoid the accelerate dependency.
    # Now move the whole model to GPU (or keep on CPU).
    print(f"  Moving model to {device}...")
    model = model.to(device)
    model.eval()  # Set to evaluation mode (no dropout, etc.)

    # Verify the model is actually on the right device
    param_device = next(model.parameters()).device
    print(f"  ✓ Model is on: {param_device}")

    #  Extract config ─
    # The hidden_size tells us the dimensionality of hidden states.
    # This is critical — it determines the autoencoder's input size.
    hidden_size = _get_hidden_size(model)
    num_layers = _get_num_layers(model)

    config = {
        "hidden_size": hidden_size,
        "model_id": model_id,
        "model_class": model_class_used,
        "num_layers": num_layers,
        "dtype": str(dtype),
    }

    print(f"  Hidden size: {hidden_size}")
    print(f"  Num layers: {num_layers}")
    print(f"  Model loaded successfully.\n")

    return model, processor, config


def _get_hidden_size(model) -> int:
    """Extract hidden_size from the model config, handling different architectures."""
    config = model.config

    # Direct hidden_size attribute
    if hasattr(config, "hidden_size"):
        return config.hidden_size

    # PaliGemma / multimodal: the text model's hidden size
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return config.text_config.hidden_size

    # Gemma-specific
    if hasattr(config, "num_key_value_heads"):
        # Try to infer from model parameters
        for name, param in model.named_parameters():
            if "embed_tokens" in name:
                return param.shape[1]

    raise ValueError(
        "Could not determine hidden_size from model config. "
        f"Config keys: {list(vars(config).keys())}"
    )


def _get_num_layers(model) -> int:
    """Extract number of transformer layers."""
    config = model.config

    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    if hasattr(config, "text_config") and hasattr(config.text_config, "num_hidden_layers"):
        return config.text_config.num_hidden_layers

    return -1  # Unknown, non-critical


def _prepare_inputs(processor, text: str, image=None, device=None):
    """
    Prepare model inputs using the processor's chat template.

    MedGemma (Gemma 3) requires specific image tokens in the text that
    vary by model version. Rather than guessing the token string, we use
    the processor's apply_chat_template() method which inserts the correct
    tokens automatically.

    This is the same approach used in the official MedGemma documentation.

    Args:
        processor: The model's processor
        text:      The prompt text
        image:     PIL Image or None
        device:    Target device

    Returns:
        inputs: Dict of tensors ready for model(**inputs)
    """
    # Build a chat-format message list
    content = []
    images_list = None

    if image is not None:
        # Image content block — the chat template will insert the
        # correct image placeholder token (e.g., <start_of_image>)
        content.append({"type": "image", "image": image})
        images_list = [image]

    content.append({"type": "text", "text": text})

    messages = [{"role": "user", "content": content}]

    # apply_chat_template formats the text with proper special tokens
    formatted_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,  # Return string, not token IDs
    )

    # Now process with the correctly formatted text
    if images_list is not None:
        inputs = processor(
            text=formatted_text,
            images=images_list,
            return_tensors="pt",
        )
    else:
        inputs = processor(
            text=formatted_text,
            return_tensors="pt",
        )

    if device is not None:
        inputs = inputs.to(device)

    return inputs


def extract_hidden_state(model, processor, text: str, image=None, device=None):
    """
    Run a forward pass and extract the hidden state of the last token
    from the last transformer layer.

    This is the H(i)_t from the paper — the model's internal representation
    after processing all input but before generating any output.

    Args:
        model:     The loaded MedGemma model
        processor: The model's processor
        text:      The input prompt text
        image:     PIL Image or None (for text-only)
        device:    Override device (auto-detected if None)

    Returns:
        hidden_state: torch.Tensor of shape [hidden_size], on CPU, float32
        generated_text: str — the model's generated response
    """
    if device is None:
        device = next(model.parameters()).device

    #  Prepare inputs using chat template 
    inputs = _prepare_inputs(processor, text, image, device)

    #  Forward pass for hidden states ─
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Extract last layer, last token position
    # outputs.hidden_states is a tuple: (embedding_output, layer_1, ..., layer_N)
    # We want the LAST layer's output at the LAST token position.
    last_layer = outputs.hidden_states[-1]       # [batch=1, seq_len, hidden_size]
    hidden_state = last_layer[0, -1, :]          # [hidden_size]
    hidden_state = hidden_state.float().cpu()     # Convert to float32, move to CPU

    #  Generate text response 
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,  # Greedy decoding for reproducibility
        )

    # Decode only the newly generated tokens (skip input tokens)
    input_len = inputs["input_ids"].shape[1]
    generated_text = processor.decode(
        gen_ids[0][input_len:],
        skip_special_tokens=True,
    )

    return hidden_state, generated_text


def get_merged_embeddings(model, processor, text: str, image=None, device=None):
    """
    Get the merged input embeddings (visual + text tokens combined) that serve
    as input to the transformer layers. Used for prefix injection.

    For multimodal inputs: image tokens are processed by the vision encoder
    and merged with text token embeddings.
    For text-only: just the text token embeddings.

    Returns:
        embeddings: [1, seq_len, hidden_size] — merged input embeddings
        attention_mask: [1, seq_len] — attention mask
    """
    if device is None:
        device = next(model.parameters()).device

    # Use chat template for proper image token handling
    inputs = _prepare_inputs(processor, text, image, device)

    # Forward pass to get hidden_states[0] = merged embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states[0] is the output of the embedding layer,
    # i.e., the merged vision+text embeddings before any transformer processing
    merged_embeddings = outputs.hidden_states[0]  # [1, seq_len, hidden_size]
    attention_mask = inputs.get("attention_mask",
                                torch.ones(merged_embeddings.shape[:2],
                                           device=device))

    return merged_embeddings, attention_mask


def generate_with_prefix(model, processor, text: str, prefix_vector: torch.Tensor,
                         image=None, max_new_tokens=256, device=None):
    """
    Generate text with a prefix vector injected into the input embeddings.

    This is the core of ThoughtComm's latent communication — the prefix
    encodes latent thoughts from other agents, steering the model's reasoning
    without any explicit text-based message passing.

    Args:
        model:          The loaded MedGemma model
        processor:      The model's processor
        text:           The input prompt text
        prefix_vector:  Tensor of shape [1, d_model] — one prefix "token"
        image:          PIL Image or None
        max_new_tokens: Maximum tokens to generate
        device:         Override device

    Returns:
        generated_text: str — the model's response
    """
    if device is None:
        device = next(model.parameters()).device

    prefix_vector = prefix_vector.to(device).to(model.dtype)

    #  Get merged embeddings 
    merged_embeds, attention_mask = get_merged_embeddings(
        model, processor, text, image, device
    )

    #  Prepend prefix ─
    # prefix shape: [1, d_model] to [1, 1, d_model]
    if prefix_vector.dim() == 1:
        prefix_vector = prefix_vector.unsqueeze(0).unsqueeze(0)
    elif prefix_vector.dim() == 2:
        prefix_vector = prefix_vector.unsqueeze(0)

    prefix_vector = prefix_vector.to(merged_embeds.dtype)

    injected = torch.cat([prefix_vector, merged_embeds], dim=1)
    # [1, 1 + seq_len, hidden_size]

    # Extend attention mask for the prefix position
    prefix_mask = torch.ones(1, prefix_vector.shape[1],
                             device=device, dtype=attention_mask.dtype)
    extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)

    #  Generate ─
    with torch.no_grad():
        gen_ids = model.generate(
            inputs_embeds=injected,
            attention_mask=extended_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_text = processor.decode(gen_ids[0], skip_special_tokens=True)
    return generated_text