"""
PHASE 2: Train the Prefix Adapter

The adapter converts personalized latent thoughts into prefix vectors that
can be injected into MedGemma's input. This is how latent thoughts get
"transmitted" back into the language model.

From the paper (Eq. 11):
    P(i)_t = g(Z_tilde(i)_t) ∈ R^{m × d}

where g is the adapter, Z_tilde is the personalized latent vector for
agent i, m is the prefix length (default 1), and d is MedGemma's
embedding dimension.

The adapter is trained to produce prefixes that DON'T break the model's
coherence (Eq. 12). We're NOT training it to produce correct diagnoses —
just to inject latent information in a "natural" way the model can use.

Run:
    python -m thoughtcomm.adapter

Inputs:
    artifacts/hidden_states_train.pt
    artifacts/encoder.pt, decoder.pt
    artifacts/autoencoder_config.pt
    artifacts/structure_mask.pt
    artifacts/h_normalization.pt
    artifacts/model_config.pt

Outputs:
    artifacts/adapter.pt
    artifacts/agreement_weights.pt
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARTIFACTS_DIR = Path(os.getenv("THOUGHTCOMM_ARTIFACTS", "artifacts"))


# 1. PERSONALIZATION

def personalize_thoughts(z_hat, agent_mask, agreement, w_private, w_shared):
    """
    Construct a personalized latent vector for a specific agent.

    From the paper (Eq. 9):
        Z_tilde(i)_t = concat_alpha(w_alpha * Z_hat(i)_t,alpha)

    In plain terms: take all latent thoughts, zero out the ones that
    don't belong to this agent, and weight the remaining by their
    agreement level (private vs shared).

    Args:
        z_hat:      [latent_dim] — all latent thoughts for this case
        agent_mask: [latent_dim] boolean — True for thoughts relevant to this agent
        agreement:  [latent_dim] int — 0=unused, 1=private, 2=shared
        w_private:  Scalar weight for private thoughts (agreement=1)
        w_shared:   Scalar weight for shared thoughts (agreement=2)

    Returns:
        z_personalized: [latent_dim] — masked and weighted latent vector
    """
    # Start with zeros
    z_personalized = torch.zeros_like(z_hat)

    # Apply mask: only include thoughts relevant to this agent
    relevant = agent_mask.to(z_hat.device)

    # Weight by agreement level
    private_idx = (agreement == 1).to(z_hat.device) & relevant
    shared_idx = (agreement == 2).to(z_hat.device) & relevant

    z_personalized[private_idx] = w_private * z_hat[private_idx]
    z_personalized[shared_idx] = w_shared * z_hat[shared_idx]

    return z_personalized


# 2. ADAPTER ARCHITECTURE


class PrefixAdapter(nn.Module):
    """
    Converts personalized latent thoughts into prefix embeddings for MedGemma.

    From the paper (Eq. 11):
        P(i)_t = g(Z_tilde(i)_t) ∈ R^{m × d}

    For m=1 (one prefix token), this maps:
        [latent_dim] → [d_model]

    The output lives in MedGemma's embedding space, so when prepended to
    the input token embeddings, MedGemma's attention mechanism naturally
    attends to it — the model "sees" an extra token that encodes latent
    thoughts from the other agent.

    We use GELU activation because Gemma (MedGemma's base) uses GELU
    internally, so the adapter output is in a distribution the model expects.
    """

    def __init__(self, latent_dim: int, d_model: int, prefix_length: int = 1):
        super().__init__()

        self.latent_dim = latent_dim
        self.d_model = d_model
        self.prefix_length = prefix_length

        output_dim = prefix_length * d_model

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, output_dim),
        )

    def forward(self, z_personalized: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_personalized: [latent_dim] or [batch, latent_dim]

        Returns:
            prefix: [prefix_length, d_model] or [batch, prefix_length, d_model]
        """
        had_batch = z_personalized.dim() == 2
        if not had_batch:
            z_personalized = z_personalized.unsqueeze(0)

        out = self.net(z_personalized)  # [batch, prefix_length * d_model]
        prefix = out.view(-1, self.prefix_length, self.d_model)

        if not had_batch:
            prefix = prefix.squeeze(0)  # [prefix_length, d_model]

        return prefix


# 3. TRAINING LOOP

def train_adapter(
    prefix_length: int = 1,
    lr: float = 1e-4,
    epochs: int = 100,
    log_wandb: bool = True,
):
    """
    Train the prefix adapter (Phase 2).

    The training uses a teacher-forcing approach:
    1. Forward pass WITH prefix → get hidden states and logits
    2. Forward pass WITHOUT prefix → get reference hidden states (no_grad)
    3. Loss = semantic_similarity + language_modeling

    This trains the adapter to produce prefixes that steer reasoning
    without causing gibberish.

    IMPORTANT: MedGemma is FROZEN. Only the adapter and agreement weights
    get gradients.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("PHASE 2: Training Prefix Adapter")
    print("=" * 60)

    #  Load everything 
    model_config = torch.load(ARTIFACTS_DIR / "model_config.pt",
                              weights_only=False)
    ae_config = torch.load(ARTIFACTS_DIR / "autoencoder_config.pt",
                           weights_only=False)
    structure = torch.load(ARTIFACTS_DIR / "structure_mask.pt",
                           weights_only=False)
    h_norm = torch.load(ARTIFACTS_DIR / "h_normalization.pt",
                        weights_only=False)
    train_data = torch.load(ARTIFACTS_DIR / "hidden_states_train.pt",
                            weights_only=False)

    hidden_size = model_config["hidden_size"]
    input_dim = ae_config["input_dim"]
    latent_dim = ae_config["latent_dim"]

    vision_mask = structure["vision_mask"]
    text_mask = structure["text_mask"]
    agreement = structure["agreement"]

    h_mean = h_norm["mean"].to(device)
    h_std = h_norm["std"].to(device)

    print(f"  Hidden size (d_model): {hidden_size}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Prefix length: {prefix_length}")
    print(f"  Training samples: {len(train_data)}")

    #  Load encoder (FROZEN) 
    from thoughtcomm.autoencoder import ThoughtAutoencoder
    autoencoder = ThoughtAutoencoder(input_dim, latent_dim).to(device)
    autoencoder.encoder.load_state_dict(
        torch.load(ARTIFACTS_DIR / "encoder.pt", weights_only=True,
                    map_location=device))
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False

    #  Load MedGemma (FROZEN) ─
    print("  Loading MedGemma for adapter training...")
    from thoughtcomm.model_loader import load_medgemma
    model, processor, _ = load_medgemma()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    adapter = PrefixAdapter(latent_dim, hidden_size, prefix_length).to(device)

    # Agreement weights: w_1 for private thoughts, w_2 for shared thoughts
    # These are learnable scalars (Eq. 9)
    w_private = nn.Parameter(torch.tensor(1.0, device=device))
    w_shared = nn.Parameter(torch.tensor(1.0, device=device))

    trainable_params = list(adapter.parameters()) + [w_private, w_shared]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    adapter_params = sum(p.numel() for p in adapter.parameters())
    print(f"  Adapter parameters: {adapter_params:,}")
    print(f"  Trainable: adapter ({adapter_params:,}) + 2 agreement weights")
    print(f"  Frozen: encoder + MedGemma (all weights)")

    #  W&B setup 
    if log_wandb:
        import wandb
        wandb.init(
            project="medical-multiagent",
            name="thoughtcomm_phase2_adapter",
            config={
                "phase": 2,
                "prefix_length": prefix_length,
                "lr": lr,
                "epochs": epochs,
                "latent_dim": latent_dim,
                "hidden_size": hidden_size,
                "adapter_params": adapter_params,
            },
        )

    
    # These are the "baseline" states that we compare against.
    # Computing them once saves time during training.
    print("  Pre-computing reference hidden states...")
    from data.loader import format_text_question, format_vision_question
    from thoughtcomm.model_loader import get_merged_embeddings

    # We also need access to the original cases for prompts and images
    from data.loader import load_nejm
    all_cases = load_nejm(n_samples=None, seed=42)
    train_cases = all_cases[:len(train_data)]

    #  Training 
    print(f"\n  Starting training ({epochs} epochs)...")

    for epoch in range(epochs):
        epoch_sim_loss = 0.0
        epoch_total_loss = 0.0
        n_cases = 0

        adapter.train()

        for case_idx, (case_data, case) in enumerate(
            zip(train_data, train_cases)
        ):
            optimizer.zero_grad()

            #  Extract latent thoughts (frozen encoder) 
            H_t = case_data["H_t"].to(device)
            H_normalized = (H_t - h_mean.squeeze()) / h_std.squeeze()
            z_hat = autoencoder.encode(H_normalized)

            #  Personalize for vision agent 
            z_vision = personalize_thoughts(
                z_hat, vision_mask, agreement, w_private, w_shared)

            #  Personalize for text agent 
            z_text = personalize_thoughts(
                z_hat, text_mask, agreement, w_private, w_shared)

            #  Generate prefix vectors 
            prefix_vision = adapter(z_vision)  # [prefix_length, d_model]
            prefix_text = adapter(z_text)      # [prefix_length, d_model]

            #  Compute loss for VISION agent 
            vision_prompt = format_vision_question(case)
            sim_loss_v = _compute_prefix_loss(
                model, processor, vision_prompt,
                prefix_vision, image=case["image"], device=device,
            )

            # Compute loss for TEXT agent 
            text_prompt = format_text_question(case)
            sim_loss_t = _compute_prefix_loss(
                model, processor, text_prompt,
                prefix_text, image=None, device=device,
            )

            #  Total loss 
            total_loss = sim_loss_v + sim_loss_t

            total_loss.backward()
            optimizer.step()

            epoch_sim_loss += (sim_loss_v.item() + sim_loss_t.item())
            epoch_total_loss += total_loss.item()
            n_cases += 1

        # Epoch logging 
        avg_sim = epoch_sim_loss / n_cases if n_cases > 0 else 0
        avg_total = epoch_total_loss / n_cases if n_cases > 0 else 0

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"sim_loss={avg_sim:.4f}, "
                  f"total={avg_total:.4f}, "
                  f"w_private={w_private.item():.3f}, "
                  f"w_shared={w_shared.item():.3f}")

        if log_wandb:
            import wandb
            wandb.log({
                "phase2/sim_loss": avg_sim,
                "phase2/total_loss": avg_total,
                "phase2/w_private": w_private.item(),
                "phase2/w_shared": w_shared.item(),
                "phase2/epoch": epoch + 1,
            })

    #  Save 
    torch.save(adapter.state_dict(), ARTIFACTS_DIR / "adapter.pt")
    torch.save({
        "w_private": w_private.detach().cpu(),
        "w_shared": w_shared.detach().cpu(),
    }, ARTIFACTS_DIR / "agreement_weights.pt")
    torch.save({
        "latent_dim": latent_dim,
        "d_model": hidden_size,
        "prefix_length": prefix_length,
    }, ARTIFACTS_DIR / "adapter_config.pt")

    print(f"\n✓ Adapter training complete.")
    print(f"  Final w_private={w_private.item():.3f}, "
          f"w_shared={w_shared.item():.3f}")

    if log_wandb:
        import wandb
        wandb.finish()


def _compute_prefix_loss(model, processor, text, prefix, image=None, device=None):
    """
    Compute the adapter training loss for one agent on one case.

    The loss ensures the prefix doesn't break coherence:
    1. Get hidden states WITH prefix injected
    2. Get hidden states WITHOUT prefix (reference, no_grad)
    3. Return 1 - cosine_similarity between the two

    The paper (Eq. 12):

    We simplify by comparing the last hidden state (the model's final
    "thinking" representation) rather than generated text embeddings.
    This avoids expensive generation during training.
    """
    from thoughtcomm.model_loader import get_merged_embeddings

    #  Reference: forward WITHOUT prefix (frozen, no grad) 
    ref_embeds, ref_mask = get_merged_embeddings(
        model, processor, text, image, device)
    # ref_embeds: [1, seq_len, hidden_size]

    with torch.no_grad():
        ref_outputs = model(inputs_embeds=ref_embeds,
                            attention_mask=ref_mask,
                            output_hidden_states=True)
        ref_hidden = ref_outputs.hidden_states[-1][0, -1, :]
        ref_hidden = ref_hidden.detach()

    #  With prefix: forward WITH prefix (grad flows to adapter) 
    prefix_3d = prefix.unsqueeze(0).to(ref_embeds.dtype)  # [1, m, d_model]
    injected = torch.cat([prefix_3d, ref_embeds], dim=1)

    prefix_mask = torch.ones(1, prefix.shape[0],
                             device=device, dtype=ref_mask.dtype)
    ext_mask = torch.cat([prefix_mask, ref_mask], dim=1)

    gen_outputs = model(inputs_embeds=injected,
                        attention_mask=ext_mask,
                        output_hidden_states=True)
    gen_hidden = gen_outputs.hidden_states[-1][0, -1, :]

    #  Semantic similarity loss 
    sim = F.cosine_similarity(
        gen_hidden.unsqueeze(0).float(),
        ref_hidden.unsqueeze(0).float(),
    )
    sim_loss = 1.0 - sim.mean()

    return sim_loss
# 4. MAIN

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2: Train adapter")
    parser.add_argument("--prefix_length", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    train_adapter(
        prefix_length=args.prefix_length,
        lr=args.lr,
        epochs=args.epochs,
        log_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()