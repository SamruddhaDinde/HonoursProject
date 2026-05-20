"""
PHASE 1: Sparsity-Regularized Autoencoder + Structure Recovery

This file contains:
  1. ThoughtAutoencoder — the neural network architecture
  2. Training loop with Jacobian sparsity penalty (Eq. 7 from the paper)
  3. Structure recovery — analyzing the trained decoder's Jacobian to discover
     which latent thoughts belong to which agents (Theorem 3)

The autoencoder learns to compress concatenated agent hidden states H_t
into latent thoughts Z_hat, then reconstruct them. The sparsity penalty
on the decoder's Jacobian ensures that each latent thought only connects
to the agent(s) it actually belongs to — this is what makes the thoughts
identifiable (Theorems 1-3).

Run:
    python -m thoughtcomm.autoencoder

Inputs:
    artifacts/hidden_states_train.pt
    artifacts/model_config.pt

Outputs:
    artifacts/encoder.pt
    artifacts/decoder.pt
    artifacts/structure_mask.pt
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARTIFACTS_DIR = Path(os.getenv("THOUGHTCOMM_ARTIFACTS", "artifacts"))


# ═══════════════════════════════════════════════════════════════════════════
# 1. ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════

class ThoughtAutoencoder(nn.Module):
    """
    Sparsity-regularized autoencoder for extracting latent thoughts.

    From the paper (Eq. 6-7):
        Encoder (f_hat_inverse): H_t → Z_hat_t  (observations → thoughts)
        Decoder (f_hat):         Z_hat_t → H_t   (thoughts → observations)

    The key insight: sparsity on the decoder's Jacobian forces each latent
    dimension to influence only specific components of H_t. Since H_t is
    the concatenation of [H_vision ; H_text], the Jacobian reveals which
    thoughts belong to which agent.
    """

    def __init__(self, input_dim: int, latent_dim: int):
        """
        Args:
            input_dim:  2 * d_model (concatenated vision + text hidden states)
            latent_dim: n_z — number of latent thought dimensions
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # ── Encoder: H_t → Z_hat ────────────────────────────────────
        # Compresses the concatenated agent states into latent thoughts.
        # We use LayerNorm because LLM hidden states can have very
        # different scales across dimensions.
        # LeakyReLU (not ReLU) because ReLU creates exact zeros that
        # would confuse the Jacobian sparsity analysis.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.LayerNorm(2048),
            nn.LeakyReLU(0.01),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, latent_dim),
        )

        # ── Decoder: Z_hat → H_reconstructed ────────────────────────
        # Reconstructs the hidden states from latent thoughts.
        # The Jacobian of THIS decoder is what we regularize for sparsity.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.LeakyReLU(0.01),
            nn.Linear(2048, input_dim),
        )

    def encode(self, h_t: torch.Tensor) -> torch.Tensor:
        """H_t → Z_hat (extract latent thoughts)."""
        return self.encoder(h_t)

    def decode(self, z_hat: torch.Tensor) -> torch.Tensor:
        """Z_hat → H_reconstructed."""
        return self.decoder(z_hat)

    def forward(self, h_t: torch.Tensor):
        """Full forward: encode then decode."""
        z_hat = self.encode(h_t)
        h_recon = self.decode(z_hat)
        return z_hat, h_recon


# ═══════════════════════════════════════════════════════════════════════════
# 2. JACOBIAN COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_jacobian_stochastic(decoder, z_hat, n_samples=128):
    """
    Estimate the L1 norm of the decoder's Jacobian via random sampling.

    Computing the full Jacobian of shape [output_dim, latent_dim] per sample
    is expensive. Instead, we sample random output dimensions and compute
    their gradients w.r.t. the latent input. This gives an unbiased estimate.

    Args:
        decoder:    The decoder network
        z_hat:      Latent codes, shape [batch, latent_dim], requires_grad=True
        n_samples:  How many output dimensions to sample per step

    Returns:
        l1_estimate: Scalar — estimated L1 norm of the Jacobian
    """
    output = decoder(z_hat)  # [batch, output_dim]
    output_dim = output.shape[1]
    batch_size = output.shape[0]

    # Sample random output dimensions
    sampled_dims = torch.randint(0, output_dim, (n_samples,))

    total_l1 = 0.0
    for dim_idx in sampled_dims:
        # Sum over batch for this output dimension
        scalar = output[:, dim_idx].sum()

        # Gradient of this scalar w.r.t. z_hat gives one row of the Jacobian
        # for each sample in the batch
        grad = torch.autograd.grad(
            scalar, z_hat,
            create_graph=True,   # Need this for backprop through the penalty
            retain_graph=True,
        )[0]  # [batch, latent_dim]

        total_l1 += grad.abs().mean()

    # Scale by the sampling ratio to get unbiased estimate
    l1_estimate = total_l1 * (output_dim / n_samples)
    return l1_estimate


def compute_full_jacobian(decoder, z_hat_single):
    """
    Compute the FULL Jacobian of the decoder at a single input point.
    Used for structure recovery (runs once after training, not during training).

    Args:
        decoder:        The decoder network
        z_hat_single:   Single latent code, shape [latent_dim]

    Returns:
        jacobian: [output_dim, latent_dim] — full Jacobian matrix
    """
    z = z_hat_single.detach().clone().requires_grad_(True)
    output = decoder(z)  # [output_dim]
    output_dim = output.shape[0]
    latent_dim = z.shape[0]

    jacobian = torch.zeros(output_dim, latent_dim, device=z.device)

    for i in range(output_dim):
        if z.grad is not None:
            z.grad.zero_()
        output[i].backward(retain_graph=True)
        jacobian[i] = z.grad.clone()

    return jacobian


# ═══════════════════════════════════════════════════════════════════════════
# 3. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def train_autoencoder(
    latent_dim: int = 512,
    lambda_sparse: float = 0.01,
    lr: float = 1e-4,
    epochs: int = 500,
    batch_size: int = 32,
    jacobian_samples: int = 128,
    log_wandb: bool = True,
):
    """
    Train the sparsity-regularized autoencoder (Phase 1).

    The loss function (Eq. 7 from the paper):
        L_rec = ‖H_t − f_hat(Z_hat_t)‖²₂ + λ * ‖J_f_hat‖₁

    Args:
        latent_dim:       n_z — number of latent thought dimensions
        lambda_sparse:    Weight of the Jacobian sparsity penalty
        lr:               Learning rate
        epochs:           Number of training epochs
        batch_size:       Batch size
        jacobian_samples: Output dims sampled per step for Jacobian estimate
        log_wandb:        Whether to log to W&B
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load data ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1: Training Sparsity-Regularized Autoencoder")
    print("=" * 60)

    train_data = torch.load(ARTIFACTS_DIR / "hidden_states_train.pt",
                            weights_only=False)
    model_config = torch.load(ARTIFACTS_DIR / "model_config.pt",
                              weights_only=False)

    hidden_size = model_config["hidden_size"]
    input_dim = 2 * hidden_size

    print(f"  Training samples: {len(train_data)}")
    print(f"  Input dim (2 * d_model): {input_dim}")
    print(f"  Latent dim (n_z): {latent_dim}")
    print(f"  Lambda (sparsity): {lambda_sparse}")

    # Stack all H_t vectors into a single tensor
    H_all = torch.stack([d["H_t"] for d in train_data]).float().to(device)
    # H_all shape: [n_train, 2 * hidden_size]

    # Normalize H_t for stable training
    # LLM hidden states can have vastly different scales
    h_mean = H_all.mean(dim=0, keepdim=True)
    h_std = H_all.std(dim=0, keepdim=True) + 1e-8
    H_normalized = (H_all - h_mean) / h_std

    # Save normalization stats (needed at inference time)
    torch.save({"mean": h_mean.cpu(), "std": h_std.cpu()},
               ARTIFACTS_DIR / "h_normalization.pt")

    dataset = TensorDataset(H_normalized)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ── Initialize model ─────────────────────────────────────────────
    autoencoder = ThoughtAutoencoder(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)

    total_params = sum(p.numel() for p in autoencoder.parameters())
    print(f"  Autoencoder parameters: {total_params:,}")

    # ── W&B setup ────────────────────────────────────────────────────
    if log_wandb:
        import wandb
        wandb.init(
            project="medical-multiagent",
            name="thoughtcomm_phase1_autoencoder",
            config={
                "phase": 1,
                "latent_dim": latent_dim,
                "lambda_sparse": lambda_sparse,
                "lr": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "input_dim": input_dim,
                "total_params": total_params,
            },
        )

    # ── Training ─────────────────────────────────────────────────────
    best_loss = float("inf")

    for epoch in range(epochs):
        epoch_recon = 0.0
        epoch_sparse = 0.0
        epoch_total = 0.0
        n_batches = 0

        autoencoder.train()
        for (batch_h,) in dataloader:
            optimizer.zero_grad()

            # Forward pass
            # z_hat needs grad for Jacobian computation
            z_hat = autoencoder.encode(batch_h)
            z_hat_for_jac = z_hat.detach().requires_grad_(True)
            h_recon = autoencoder.decode(z_hat)
            # We also need a decode with gradient tracking for z
            h_recon_jac = autoencoder.decode(z_hat_for_jac)

            # Reconstruction loss: ‖H_t − f_hat(Z_hat)‖²₂
            loss_recon = F.mse_loss(h_recon, batch_h)

            # Jacobian sparsity loss: ‖J_f_hat‖₁
            loss_sparse = compute_jacobian_stochastic(
                autoencoder.decoder, z_hat_for_jac,
                n_samples=jacobian_samples,
            )

            # Total loss (Eq. 7)
            loss = loss_recon + lambda_sparse * loss_sparse

            loss.backward()
            optimizer.step()

            epoch_recon += loss_recon.item()
            epoch_sparse += loss_sparse.item()
            epoch_total += loss.item()
            n_batches += 1

        # ── Epoch logging ────────────────────────────────────────────
        avg_recon = epoch_recon / n_batches
        avg_sparse = epoch_sparse / n_batches
        avg_total = epoch_total / n_batches

        if avg_total < best_loss:
            best_loss = avg_total
            torch.save(autoencoder.encoder.state_dict(),
                       ARTIFACTS_DIR / "encoder_best.pt")
            torch.save(autoencoder.decoder.state_dict(),
                       ARTIFACTS_DIR / "decoder_best.pt")

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs}: "
                  f"recon={avg_recon:.6f}, sparse={avg_sparse:.4f}, "
                  f"total={avg_total:.6f}")

        if log_wandb:
            import wandb
            wandb.log({
                "phase1/reconstruction_loss": avg_recon,
                "phase1/sparsity_loss": avg_sparse,
                "phase1/total_loss": avg_total,
                "phase1/epoch": epoch + 1,
            })

    # ── Save final model ─────────────────────────────────────────────
    torch.save(autoencoder.encoder.state_dict(), ARTIFACTS_DIR / "encoder.pt")
    torch.save(autoencoder.decoder.state_dict(), ARTIFACTS_DIR / "decoder.pt")
    torch.save({
        "input_dim": input_dim,
        "latent_dim": latent_dim,
    }, ARTIFACTS_DIR / "autoencoder_config.pt")

    print(f"\n✓ Autoencoder training complete. Best loss: {best_loss:.6f}")

    if log_wandb:
        import wandb
        wandb.finish()

    return autoencoder


# ═══════════════════════════════════════════════════════════════════════════
# 4. STRUCTURE RECOVERY
# ═══════════════════════════════════════════════════════════════════════════

def recover_structure(
    percentile_threshold: float = 90.0,
    log_wandb: bool = True,
):
    """
    Analyze the trained decoder's Jacobian to discover thought structure.

    This implements Theorem 3 from the paper: recovering which latent
    thoughts connect to which agents via the Jacobian's nonzero pattern.

    For your 2-agent system (vision + text):
      - Shared thoughts:  influence BOTH agents' hidden states
      - Private vision:   influence ONLY the vision agent's hidden states
      - Private text:     influence ONLY the text agent's hidden states

    Args:
        percentile_threshold: Jacobian magnitude percentile for binary cutoff.
                              90.0 means only the top 10% of entries are "real".
        log_wandb: Whether to log to W&B
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("STRUCTURE RECOVERY: Analyzing Thought-Agent Dependencies")
    print("=" * 60)

    # ── Load trained autoencoder ─────────────────────────────────────
    model_config = torch.load(ARTIFACTS_DIR / "model_config.pt",
                              weights_only=False)
    ae_config = torch.load(ARTIFACTS_DIR / "autoencoder_config.pt",
                           weights_only=False)
    h_norm = torch.load(ARTIFACTS_DIR / "h_normalization.pt",
                        weights_only=False)

    hidden_size = model_config["hidden_size"]
    input_dim = ae_config["input_dim"]
    latent_dim = ae_config["latent_dim"]

    autoencoder = ThoughtAutoencoder(input_dim, latent_dim).to(device)
    autoencoder.encoder.load_state_dict(
        torch.load(ARTIFACTS_DIR / "encoder.pt", weights_only=True,
                    map_location=device))
    autoencoder.decoder.load_state_dict(
        torch.load(ARTIFACTS_DIR / "decoder.pt", weights_only=True,
                    map_location=device))
    autoencoder.eval()

    # ── Load training data ───────────────────────────────────────────
    train_data = torch.load(ARTIFACTS_DIR / "hidden_states_train.pt",
                            weights_only=False)
    H_all = torch.stack([d["H_t"] for d in train_data]).float().to(device)

    # Normalize
    h_mean = h_norm["mean"].to(device)
    h_std = h_norm["std"].to(device)
    H_normalized = (H_all - h_mean) / h_std

    # ── Compute average absolute Jacobian across training set ────────
    print(f"  Computing Jacobians for {len(train_data)} training cases...")
    print(f"  (This takes a few minutes — one full Jacobian per case)")

    J_accumulated = torch.zeros(input_dim, latent_dim, device=device)

    for i, h_t in enumerate(H_normalized):
        z_hat = autoencoder.encode(h_t)
        J = compute_full_jacobian(autoencoder.decoder, z_hat)
        J_accumulated += J.abs()

        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(H_normalized)}] Jacobians computed")

    J_avg = J_accumulated / len(H_normalized)

    # ── Threshold to binary structure matrix ─────────────────────────
    # B(J_f) from Eq. 3 in the paper
    threshold = torch.quantile(J_avg.flatten(),
                               percentile_threshold / 100.0)

    B = (J_avg > threshold)  # [input_dim, latent_dim] boolean

    print(f"\n  Jacobian statistics:")
    print(f"    Mean abs value: {J_avg.mean():.6f}")
    print(f"    Threshold ({percentile_threshold}th pctl): {threshold:.6f}")
    print(f"    Nonzero entries in B: {B.sum().item()} / {B.numel()} "
          f"({100 * B.float().mean():.1f}%)")

    # ── Split by agent ───────────────────────────────────────────────
    # H_t = [H_vision (first d_model dims) ; H_text (last d_model dims)]
    B_vision = B[:hidden_size, :]          # [d_model, latent_dim]
    B_text = B[hidden_size:, :]            # [d_model, latent_dim]

    # ── Classify each thought ────────────────────────────────────────
    # Eq. 4 from the paper: Z_{H_t^(k)} is the set of thoughts that
    # influence at least one component of agent k's hidden state.
    vision_mask = B_vision.any(dim=0)      # [latent_dim] boolean
    text_mask = B_text.any(dim=0)          # [latent_dim] boolean

    # Eq. 8: agreement score α_j = count of agents that use thought j
    agreement = vision_mask.int() + text_mask.int()  # 0, 1, or 2

    # Categorize
    n_shared = ((agreement == 2)).sum().item()
    n_private_vision = (vision_mask & ~text_mask).sum().item()
    n_private_text = (text_mask & ~vision_mask).sum().item()
    n_unused = (agreement == 0).sum().item()

    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │  THOUGHT STRUCTURE RECOVERY          │")
    print(f"  ├─────────────────────────────────────┤")
    print(f"  │  Shared (both agents):    {n_shared:4d}       │")
    print(f"  │  Private vision only:     {n_private_vision:4d}       │")
    print(f"  │  Private text only:       {n_private_text:4d}       │")
    print(f"  │  Unused dimensions:       {n_unused:4d}       │")
    print(f"  │  Total latent dims:       {latent_dim:4d}       │")
    print(f"  └─────────────────────────────────────┘")

    # ── Save structure mask ──────────────────────────────────────────
    structure = {
        "vision_mask": vision_mask.cpu(),    # [latent_dim] boolean
        "text_mask": text_mask.cpu(),        # [latent_dim] boolean
        "agreement": agreement.cpu(),        # [latent_dim] int (0, 1, or 2)
        "hidden_size": hidden_size,
        "latent_dim": latent_dim,
        "threshold": threshold.item(),
        "percentile": percentile_threshold,
        "n_shared": n_shared,
        "n_private_vision": n_private_vision,
        "n_private_text": n_private_text,
        "n_unused": n_unused,
    }
    torch.save(structure, ARTIFACTS_DIR / "structure_mask.pt")

    print(f"\n✓ Structure mask saved to {ARTIFACTS_DIR / 'structure_mask.pt'}")

    # ── W&B logging ──────────────────────────────────────────────────
    if log_wandb:
        import wandb
        wandb.init(
            project="medical-multiagent",
            name="thoughtcomm_structure_recovery",
            config={
                "phase": "structure_recovery",
                "percentile_threshold": percentile_threshold,
                "latent_dim": latent_dim,
                "hidden_size": hidden_size,
            },
        )
        wandb.log({
            "structure/n_shared": n_shared,
            "structure/n_private_vision": n_private_vision,
            "structure/n_private_text": n_private_text,
            "structure/n_unused": n_unused,
            "structure/threshold": threshold.item(),
        })
        wandb.finish()

    return structure


# ═══════════════════════════════════════════════════════════════════════════
# 5. MAIN — run Phase 1 + Structure Recovery together
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 1: Train autoencoder")
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--lambda_sparse", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    # Train autoencoder
    train_autoencoder(
        latent_dim=args.latent_dim,
        lambda_sparse=args.lambda_sparse,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        log_wandb=not args.no_wandb,
    )

    # Recover structure
    recover_structure(
        percentile_threshold=args.percentile,
        log_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()