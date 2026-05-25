"""
PHASE 3 Inference Helpers: ThoughtComm Pipeline

Loads all trained components and provides a clean interface for running
the ThoughtComm multi-round debate on new cases.

This module is used by main_thoughtcomm.py.
"""

import torch
import torch.nn.functional as F
from pathlib import Path

from thoughtcomm.autoencoder import ThoughtAutoencoder
from thoughtcomm.adapter import PrefixAdapter, personalize_thoughts
from thoughtcomm.model_loader import (
    load_medgemma,
    extract_hidden_state,
    generate_with_prefix,
)


class ThoughtCommPipeline:
    """
    Complete ThoughtComm inference pipeline.

    Loads all trained components and provides methods to:
    1. Extract hidden states from agents
    2. Recover latent thoughts via the encoder
    3. Personalize thoughts for each agent
    4. Generate prefix vectors
    5. Inject prefixes and generate responses

    Usage:
        pipeline = ThoughtCommPipeline("artifacts/")
        result = pipeline.run_case(case, format_vision_q, format_text_q)
    """

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading ThoughtComm pipeline components...")
        self._load_all()
        print("✓ ThoughtComm pipeline ready.\n")

    def _load_all(self):
        """Load all trained components."""

        #  Model config ─
        model_config = torch.load(
            self.artifacts_dir / "model_config.pt", weights_only=False)
        self.hidden_size = model_config["hidden_size"]

        #  Autoencoder config + encoder ─
        ae_config = torch.load(
            self.artifacts_dir / "autoencoder_config.pt", weights_only=False)
        self.latent_dim = ae_config["latent_dim"]
        self.input_dim = ae_config["input_dim"]

        self.autoencoder = ThoughtAutoencoder(
            self.input_dim, self.latent_dim).to(self.device)
        self.autoencoder.encoder.load_state_dict(
            torch.load(self.artifacts_dir / "encoder.pt",
                        weights_only=True, map_location=self.device))
        self.autoencoder.eval()
        for p in self.autoencoder.parameters():
            p.requires_grad = False

        #  Normalization stats 
        h_norm = torch.load(
            self.artifacts_dir / "h_normalization.pt", weights_only=False)
        self.h_mean = h_norm["mean"].to(self.device).squeeze()
        self.h_std = h_norm["std"].to(self.device).squeeze()

        #  Structure mask ─
        structure = torch.load(
            self.artifacts_dir / "structure_mask.pt", weights_only=False)
        self.vision_mask = structure["vision_mask"]
        self.text_mask = structure["text_mask"]
        self.agreement = structure["agreement"]

        print(f"  Thought structure: "
              f"{structure['n_shared']} shared, "
              f"{structure['n_private_vision']} vision-private, "
              f"{structure['n_private_text']} text-private")

        #  Adapter 
        adapter_config = torch.load(
            self.artifacts_dir / "adapter_config.pt", weights_only=False)

        self.adapter = PrefixAdapter(
            self.latent_dim,
            adapter_config["d_model"],
            adapter_config["prefix_length"],
        ).to(self.device)
        self.adapter.load_state_dict(
            torch.load(self.artifacts_dir / "adapter.pt",
                        weights_only=True, map_location=self.device))
        self.adapter.eval()
        for p in self.adapter.parameters():
            p.requires_grad = False

        #  Agreement weights 
        weights = torch.load(
            self.artifacts_dir / "agreement_weights.pt", weights_only=False)
        self.w_private = weights["w_private"].to(self.device)
        self.w_shared = weights["w_shared"].to(self.device)

        print(f"  Agreement weights: "
              f"private={self.w_private.item():.3f}, "
              f"shared={self.w_shared.item():.3f}")

        #  MedGemma ─
        self.model, self.processor, _ = load_medgemma()

    def run_case(self, case: dict, format_vision_fn, format_text_fn,
                 num_rounds: int = 2):
        """
        Run ThoughtComm on a single NEJM case.

        Args:
            case:             Dict from load_nejm() with image, question, etc.
            format_vision_fn: Function to format vision agent prompt
            format_text_fn:   Function to format text agent prompt
            num_rounds:       Number of debate rounds (default 2)

        Returns:
            Dict with all responses, hidden states, and metadata:
                - vision_responses: list of str (one per round)
                - text_responses: list of str (one per round)
                - vision_answers: list of str (parsed A-E per round)
                - text_answers: list of str (parsed A-E per round)
                - n_shared_thoughts: int
                - n_private_vision: int
                - n_private_text: int
        """
        vision_prompt = format_vision_fn(case)
        text_prompt = format_text_fn(case)

        vision_responses = []
        text_responses = []

        for round_num in range(num_rounds):

            if round_num == 0:
                # ROUND 1: Initial responses (no ThoughtComm)

                # Vision agent: image + diagnostic question
                h_vision, vision_resp = extract_hidden_state(
                    self.model, self.processor,
                    text=vision_prompt,
                    image=case["image"],
                )

                # Text agent: clinical context + question, no image
                h_text, text_resp = extract_hidden_state(
                    self.model, self.processor,
                    text=text_prompt,
                    image=None,
                )

                vision_responses.append(vision_resp)
                text_responses.append(text_resp)

            else:
                # ROUND 2+: ThoughtComm — extract, personalize, inject

                #  Thought Communication 
                H_t = torch.cat([h_vision, h_text], dim=0).to(self.device)
                H_norm = (H_t - self.h_mean) / self.h_std

                # Extract latent thoughts
                with torch.no_grad():
                    z_hat = self.autoencoder.encode(H_norm)

                # Personalize for each agent
                z_vision = personalize_thoughts(
                    z_hat, self.vision_mask, self.agreement,
                    self.w_private, self.w_shared,
                )
                z_text = personalize_thoughts(
                    z_hat, self.text_mask, self.agreement,
                    self.w_private, self.w_shared,
                )

                # Generate prefix vectors
                with torch.no_grad():
                    prefix_vision = self.adapter(z_vision)  # [m, d_model]
                    prefix_text = self.adapter(z_text)      # [m, d_model]

                #  Construct round 2+ prompts ─
                round_vision_prompt = (
                    f"{vision_prompt}\n\n"
                    f"Your previous assessment: {vision_responses[-1]}\n\n"
                    f"Reconsider your assessment with additional context. "
                    f"Provide your updated diagnosis."
                )
                round_text_prompt = (
                    f"{text_prompt}\n\n"
                    f"Your previous assessment: {text_responses[-1]}\n\n"
                    f"Reconsider your assessment with additional context. "
                    f"Provide your updated diagnosis."
                )

                #  Generate with prefix injection ─
                vision_resp = generate_with_prefix(
                    self.model, self.processor,
                    text=round_vision_prompt,
                    prefix_vector=prefix_vision,
                    image=case["image"],
                )

                text_resp = generate_with_prefix(
                    self.model, self.processor,
                    text=round_text_prompt,
                    prefix_vector=prefix_text,
                    image=None,
                )

                vision_responses.append(vision_resp)
                text_responses.append(text_resp)

                # Update hidden states for potential next round
                h_vision, _ = extract_hidden_state(
                    self.model, self.processor,
                    text=round_vision_prompt,
                    image=case["image"],
                )
                h_text, _ = extract_hidden_state(
                    self.model, self.processor,
                    text=round_text_prompt,
                    image=None,
                )

        return {
            "vision_responses": vision_responses,
            "text_responses": text_responses,
            "n_shared_thoughts": int((self.agreement == 2).sum()),
            "n_private_vision": int(
                (self.vision_mask & ~self.text_mask).sum()),
            "n_private_text": int(
                (self.text_mask & ~self.vision_mask).sum()),
        }