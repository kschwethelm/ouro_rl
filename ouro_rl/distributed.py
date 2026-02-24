"""Lightweight distributed helpers for manual gradient-sync training.

No DDP wrapper — compatible with CPU↔GPU model swapping used during vLLM generation.
All functions are no-ops when running single-GPU (no torchrun).
"""

import torch
import torch.distributed as dist


def is_dist() -> bool:
    """Check if multi-GPU distributed training is active."""
    return dist.is_initialized() and dist.get_world_size() > 1


def get_rank() -> int:
    """Return current rank (0 when not distributed)."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    """Return world size (1 when not distributed)."""
    return dist.get_world_size() if dist.is_initialized() else 1


def broadcast_object(obj: object, src: int = 0) -> object:
    """Broadcast a Python object from src rank to all ranks. No-op for single GPU."""
    if not is_dist():
        return obj
    container = [obj]
    dist.broadcast_object_list(container, src=src)
    return container[0]


def sync_gradients(model: torch.nn.Module) -> None:
    """Average gradients across ranks. No-op for single GPU."""
    if not is_dist():
        return
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


def sync_metrics(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    """Average metric values across ranks. No-op for single GPU."""
    if not is_dist():
        return metrics
    synced = {}
    for k, v in metrics.items():
        t = torch.tensor(v, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        synced[k] = t.item()
    return synced


def shard_range(total: int, rank: int, world_size: int) -> tuple[int, int]:
    """Compute [start, end) for a contiguous shard of ``total`` items."""
    base = total // world_size
    remainder = total % world_size
    start = rank * base + min(rank, remainder)
    end = start + base + (1 if rank < remainder else 0)
    return start, end
