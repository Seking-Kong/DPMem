"""
MEKS: Memory-aligned Evidence Keyframe Selection.

For each frame in a streaming window, compute three scores on top of the
patches that survive token_drop:
  (1) Query-Evidence  S_qe:  top-k cosine similarity between surviving
      patch features R_i and the query text embedding z_Q.
  (2) Novelty         S_nov: kept_tokens / total_tokens (reuses token_drop output).
  (3) Text-Gap        S_gap: 1 - cosine(mean(R_i), z_T_k), i.e. how much visual
      evidence is NOT already captured by the current window text summary T_k.

Final score:
    S_i = alpha * S_qe + beta * S_nov + gamma * S_gap

The module is self-contained: it only relies on Qwen3-VL's input embedding
layer + tokenizer to embed text, so z_Q, z_T_k and patch features live in the
same latent space.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def _text_embed(
    text: str,
    tokenizer,
    embed_layer: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Mean-pool the LLM input-embedding of a text string. Returns None for empty text."""
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
    if ids.numel() == 0:
        return None
    ids = ids.to(device)
    emb = embed_layer(ids)
    pooled = emb.mean(dim=1).squeeze(0)
    return pooled.to(dtype=dtype)


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Row-wise cosine similarity between rows of a [..., D] and a single vector b [D]."""
    a = a.float()
    b = b.float()
    a_n = a / (a.norm(dim=-1, keepdim=True) + eps)
    b_n = b / (b.norm() + eps)
    return (a_n * b_n).sum(dim=-1)


@torch.no_grad()
def compute_meks_scores(
    frame_info: Sequence[Dict[str, Any]],
    z_Q: Optional[torch.Tensor],
    z_T: Optional[torch.Tensor],
    qe_top_k: int,
    novelty_mode: str = "relative",
    exclude_first_last_from_nov: bool = True,
) -> List[Dict[str, float]]:
    """
    Compute (S_qe, S_nov, S_gap) for every frame.

    Args:
        frame_info: list of dicts produced by token_drop, each with
            keys {"total": int, "kept": int,
                  "feats": Tensor[kept, hidden],
                  "full_mean": Tensor[hidden]  # pre-drop whole-frame mean}.
            "full_mean" is optional: if missing, S_gap falls back to mean(feats).
        z_Q: query text embedding [hidden] (same dtype/device as feats). Can be None
             if no query is available, in which case S_qe == 0 for all frames.
        z_T: window-summary text embedding [hidden]. If None, S_gap == 0.
        qe_top_k: number of top-similarity patches to average for S_qe.
        novelty_mode: "absolute"  -> S_nov = kept/total (raw retention ratio),
                      "relative"  -> S_nov = kept/total - mean_j(kept/total). The
                                     mean is taken over the candidate pool (i.e.
                                     after excluding first/last if requested).
        exclude_first_last_from_nov: when True, the first and last frames of the
            window are hard-excluded from the S_nov competition: their S_nov is
            forced to 0, AND they are dropped from the relative-mode mean. This
            cancels the structural protection that drop-prev gives to boundary
            frames (kept/total = 1.0 by construction).

    Returns:
        list of per-frame dicts with keys {"S_qe", "S_nov", "S_gap"}.
    """
    n = len(frame_info)

    excluded_idx = set()
    if exclude_first_last_from_nov and n > 0:
        excluded_idx.add(0)
        if n > 1:
            excluded_idx.add(n - 1)

    raw_ratios: List[float] = []
    for info in frame_info:
        total = max(1, int(info.get("total", 0)))
        kept = int(info.get("kept", 0))
        raw_ratios.append(float(kept) / float(total))

    candidate_ratios = [r for i, r in enumerate(raw_ratios) if i not in excluded_idx]

    if novelty_mode == "relative" and len(candidate_ratios) > 0:
        mean_ratio = sum(candidate_ratios) / len(candidate_ratios)
        nov_values = [
            0.0 if i in excluded_idx else (raw_ratios[i] - mean_ratio)
            for i in range(n)
        ]
    elif novelty_mode == "absolute":
        nov_values = [
            0.0 if i in excluded_idx else raw_ratios[i]
            for i in range(n)
        ]
    else:
        nov_values = [
            0.0 if i in excluded_idx else raw_ratios[i]
            for i in range(n)
        ]

    results: List[Dict[str, float]] = []
    for i, info in enumerate(frame_info):
        kept = int(info.get("kept", 0))
        feats: torch.Tensor = info.get("feats")
        full_mean: Optional[torch.Tensor] = info.get("full_mean")
        s_nov = float(nov_values[i])

        if z_Q is None or feats is None or feats.numel() == 0 or kept == 0:
            s_qe = 0.0
        else:
            sims_q = _cosine(feats, z_Q.to(feats.device))
            k = min(int(qe_top_k), sims_q.numel())
            if k <= 0:
                s_qe = 0.0
            else:
                top_vals, _ = torch.topk(sims_q, k=k)
                s_qe = float(top_vals.mean().item())

        if z_T is None:
            s_gap = 0.0
        else:
            if full_mean is not None and full_mean.numel() > 0:
                frame_repr = full_mean.float()
            elif feats is not None and feats.numel() > 0 and kept > 0:
                frame_repr = feats.float().mean(dim=0)
            else:
                results.append({"S_qe": s_qe, "S_nov": s_nov, "S_gap": 0.0})
                continue
            ref_device = frame_repr.device
            sim_t = _cosine(frame_repr.unsqueeze(0), z_T.to(ref_device))
            s_gap = float(1.0 - sim_t.item())

        results.append({"S_qe": s_qe, "S_nov": s_nov, "S_gap": s_gap})
    return results


def select_by_threshold(
    per_frame_scores: Sequence[Dict[str, float]],
    alpha: float,
    beta: float,
    gamma: float,
    threshold: float,
) -> Tuple[List[int], List[Dict[str, float]]]:
    """Fuse (S_qe, S_nov, S_gap) -> S_total and keep every frame whose S_total
    strictly exceeds the given threshold.

    Returns:
        (sorted_selected_indices, per_frame_detailed_scores)
        - sorted_selected_indices: frame positions (ascending) whose S_total > threshold.
        - per_frame_detailed_scores: full list with {S_qe, S_nov, S_gap, S_total}
          aligned with the original frame order.
    """
    detailed: List[Dict[str, float]] = []
    for s in per_frame_scores:
        s_total = alpha * s["S_qe"] + beta * s["S_nov"] + gamma * s["S_gap"]
        detailed.append({
            "S_qe": float(s["S_qe"]),
            "S_nov": float(s["S_nov"]),
            "S_gap": float(s["S_gap"]),
            "S_total": float(s_total),
        })

    if len(detailed) == 0:
        return [], detailed

    selected = [i for i, d in enumerate(detailed) if d["S_total"] > threshold]
    return selected, detailed


class MEKSSelector:
    """Thin stateful helper that caches tokenizer + embed_layer handles."""

    def __init__(
        self,
        model,
        tokenizer,
        alpha: float = 1.0,
        beta: float = 0.3,
        gamma: float = 0.5,
        threshold: float = 0.5,
        qe_top_k: int = 8,
        novelty_mode: str = "relative",
        exclude_first_last_from_nov: bool = True,
    ):
        self.embed_layer = model.get_input_embeddings()
        try:
            sample_param = next(self.embed_layer.parameters())
            self.device = sample_param.device
            self.dtype = sample_param.dtype
        except StopIteration:
            self.device = torch.device("cpu")
            self.dtype = torch.float32
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.threshold = threshold
        self.qe_top_k = qe_top_k
        self.novelty_mode = novelty_mode
        self.exclude_first_last_from_nov = bool(exclude_first_last_from_nov)

    @torch.no_grad()
    def embed_text(self, text: Optional[str]) -> Optional[torch.Tensor]:
        return _text_embed(text, self.tokenizer, self.embed_layer, self.device, self.dtype)

    @torch.no_grad()
    def select(
        self,
        frame_info: Sequence[Dict[str, Any]],
        query_text: Optional[str],
        summary_text: Optional[str],
    ) -> Tuple[List[int], List[Dict[str, float]]]:
        if not frame_info:
            return [], []
        z_Q = self.embed_text(query_text)
        z_T = self.embed_text(summary_text)
        scores = compute_meks_scores(
            frame_info, z_Q, z_T,
            qe_top_k=self.qe_top_k,
            novelty_mode=self.novelty_mode,
            exclude_first_last_from_nov=self.exclude_first_last_from_nov,
        )
        return select_by_threshold(scores, self.alpha, self.beta, self.gamma, self.threshold)
