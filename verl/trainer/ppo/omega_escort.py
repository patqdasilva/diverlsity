"""
Analytic Variance-Controlled Block Token Escort (VC-BTE).

GPU-vectorized PyTorch implementation for Tsallis escort deformation
in GRPO/PPO training loops. Computes per-token escort weights from
exact entropy and variance provided by modified vLLM.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_vc_bte_vectorized(
    logprobs: torch.Tensor,
    entropies: torch.Tensor,
    variances: torch.Tensor,
    mask: torch.Tensor,
    alpha: float,
    block_size: int,
    log_omega_clip: float,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """
    Analytic Variance-Controlled Block Token Escort (VC-BTE) in pure PyTorch.

    Computes per-token escort weights omega_t from exact per-token entropy
    and variance, using block-level aggregation and batch renormalization.
    Positive alpha upweights blocks that are rarer than the model's local
    entropy baseline.

    Args:
        logprobs:      [B, T] log-probabilities of taken actions
        entropies:     [B, T] exact per-token Shannon entropy H_t
        variances:     [B, T] exact per-token variance sigma^2_t
        mask:          [B, T] 1 for valid tokens, 0 for padding
        alpha:         escort intensity parameter
        block_size:    block aggregation size (e.g. 64, 128)
        log_omega_clip: symmetric clipping bound for log omega
        eps:           numerical stability constant

    Returns:
        dict with keys:
            omega_t_raw:       [B, T] un-normalized escort weights
            omega_t_renorm:    [B, T] batch-renormalized escort weights
            omega_log_weights: [B, T] log(omega_t_renorm), ready for log-prob deformation
            block_omega:       [B, num_blocks] per-block escort weights
            raw_log_omega:     [B, num_blocks] per-block raw log-omega (pre-clip)
            clipped_log_omega: [B, num_blocks] per-block clipped log-omega
    """
    B, T = logprobs.shape
    mask_float = mask.float()

    # 1. Exact entropy-centered martingale increment: Z_tilde_t = Z_t + H_t
    z_tilde = (logprobs + entropies) * mask_float
    variances = variances * mask_float

    # 2. Pad sequence to multiple of block_size for reshape
    pad_len = (block_size - (T % block_size)) % block_size
    if pad_len > 0:
        z_tilde_pad = F.pad(z_tilde, (0, pad_len), value=0.0)
        vars_pad = F.pad(variances, (0, pad_len), value=0.0)
        mask_pad = F.pad(mask_float, (0, pad_len), value=0.0)
    else:
        z_tilde_pad, vars_pad, mask_pad = z_tilde, variances, mask_float

    num_blocks = z_tilde_pad.shape[1] // block_size

    # Reshape to [B, num_blocks, block_size]
    z_tilde_blocks = z_tilde_pad.view(B, num_blocks, block_size)
    vars_blocks = vars_pad.view(B, num_blocks, block_size)
    mask_blocks = mask_pad.view(B, num_blocks, block_size)

    # 3. Block surprise: S_B = sum(Z_tilde) / sqrt(M)
    valid_tokens_per_block = mask_blocks.sum(dim=-1)  # [B, num_blocks]
    block_S = z_tilde_blocks.sum(dim=-1) / torch.sqrt(valid_tokens_per_block.clamp(min=1.0))

    # 4. Block variance: Sigma^2_B = sum(sigma^2_t) / M
    block_sigma_sq = vars_blocks.sum(dim=-1) / valid_tokens_per_block.clamp(min=1.0)

    # 5. Escort weight: log omega_B = -alpha * S_B - 0.5 * alpha^2 * Sigma^2_B
    # With Z_tilde = log p(a_t | h_t) + H_t, rarer-than-expected blocks have
    # negative S_B and should receive larger weights when alpha > 0.
    raw_log_omega = (-alpha * block_S) - (0.5 * (alpha ** 2) * block_sigma_sq)

    # Clip and exponentiate
    clip_c = abs(float(log_omega_clip))
    clipped_log_omega = torch.clamp(raw_log_omega, min=-clip_c, max=clip_c)
    block_omega = torch.exp(clipped_log_omega)

    # 6. Broadcast block weights to token level
    token_omega_pad = block_omega.unsqueeze(-1).expand(-1, -1, block_size).reshape(B, -1)
    token_omega = token_omega_pad[:, :T]
    token_omega = torch.where(mask.bool(), token_omega, torch.ones_like(token_omega))

    # 7. Batch renormalization per timestep: E_B[omega_t] = 1
    sum_omega_t = (token_omega * mask_float).sum(dim=0)       # [T]
    valid_batch_count_t = mask_float.sum(dim=0)               # [T]
    batch_mean_omega_t = sum_omega_t / valid_batch_count_t.clamp(min=1.0)
    batch_mean_omega_t = torch.where(
        valid_batch_count_t > 0,
        batch_mean_omega_t,
        torch.ones_like(batch_mean_omega_t),
    )

    omega_t_renorm = token_omega / batch_mean_omega_t.unsqueeze(0).clamp(min=eps)
    omega_t_renorm = torch.where(mask.bool(), omega_t_renorm, torch.ones_like(omega_t_renorm))

    # 8. Log-omega for direct log-prob deformation
    omega_log_weights = torch.log(omega_t_renorm.clamp(min=1e-12)) * mask_float

    return {
        "omega_t_raw": token_omega,
        "omega_t_renorm": omega_t_renorm,
        "omega_log_weights": omega_log_weights,
        "block_omega": block_omega,
        "raw_log_omega": raw_log_omega,
        "clipped_log_omega": clipped_log_omega,
    }
