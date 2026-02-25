"""Integration test: vLLM subprocess GPU memory release.

The GRPO training loop runs vLLM in a subprocess for rollout generation.
When the subprocess exits, the OS guarantees all GPU memory is freed.

Run with:
    uv run pytest tests/integration/test_vllm_memory.py -m integration -v
"""

import gc

import pytest
import torch

from ouro_rl.modeling import EOS_TOKEN_ID
from scripts.grpo_train import run_vllm_generation

MODEL_NAME = "ByteDance/Ouro-1.4B-Thinking"
MAX_MODEL_LEN = 512

# After subprocess exit, there should be near-zero leaked memory in the parent.
LEAK_TOLERANCE_MB = 128

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def cuda_free_mb() -> float:
    """Return free GPU memory in MiB (accounts for all CUDA allocations)."""
    free, _total = torch.cuda.mem_get_info()
    return free / (1024 * 1024)


@requires_cuda
@pytest.mark.integration
class TestVLLMSubprocessMemory:
    def test_memory_reclaimed_after_subprocess(self):
        """run_vllm_generation should return VRAM to baseline (subprocess exits)."""
        torch.cuda.empty_cache()
        gc.collect()
        baseline_free_mb = cuda_free_mb()

        rollout_ids, n_interrupted = run_vllm_generation(
            model_path=MODEL_NAME,
            dtype="bfloat16",
            max_model_len=MAX_MODEL_LEN,
            seed=42,
            prompt_token_ids=[[1, 2, 3]],
            n=1,
            temperature=0.7,
            top_p=1.0,
            max_tokens=16,
            stop_token_ids=[EOS_TOKEN_ID],
            gpu_id=0,
        )

        # Verify generation actually produced output.
        assert len(rollout_ids) == 1
        assert len(rollout_ids[0]) == 1
        assert len(rollout_ids[0][0]) > 0

        reclaimed_free_mb = cuda_free_mb()
        leaked_mb = baseline_free_mb - reclaimed_free_mb

        assert leaked_mb < LEAK_TOLERANCE_MB, (
            f"GPU memory leak after subprocess generation: {leaked_mb:.0f} MiB. "
            f"Baseline: {baseline_free_mb:.0f} MiB, "
            f"After generation: {reclaimed_free_mb:.0f} MiB. "
            f"Tolerance: {LEAK_TOLERANCE_MB} MiB."
        )

    def test_two_cycles_no_accumulation(self):
        """Two generation cycles should not accumulate leaked memory."""
        torch.cuda.empty_cache()
        gc.collect()

        leaked = []
        for _ in range(2):
            before_mb = cuda_free_mb()

            run_vllm_generation(
                model_path=MODEL_NAME,
                dtype="bfloat16",
                max_model_len=MAX_MODEL_LEN,
                seed=42,
                prompt_token_ids=[[1, 2, 3]],
                n=1,
                temperature=0.7,
                top_p=1.0,
                max_tokens=16,
                stop_token_ids=[EOS_TOKEN_ID],
                gpu_id=0,
            )

            after_mb = cuda_free_mb()
            leaked.append(before_mb - after_mb)

        # Second cycle should not leak significantly MORE than the first.
        additional_leak = leaked[1] - leaked[0]
        assert additional_leak < LEAK_TOLERANCE_MB, (
            f"Memory leak accumulates across cycles: "
            f"cycle 1 leaked {leaked[0]:.0f} MiB, cycle 2 leaked {leaked[1]:.0f} MiB "
            f"(additional: {additional_leak:.0f} MiB). "
            f"Tolerance: {LEAK_TOLERANCE_MB} MiB."
        )
