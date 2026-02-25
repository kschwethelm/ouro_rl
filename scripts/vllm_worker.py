"""vLLM generation worker — runs in a subprocess, exits when done.

Process exit guarantees all GPU memory is freed (no leak).
Communicates with the parent via torch.save/torch.load on temp .pt files.

The parent sets up a single-rank distributed env (MASTER_ADDR, MASTER_PORT,
RANK=0, WORLD_SIZE=1) so this worker can init torch.distributed and create
vLLM with external_launcher — vLLM's supported way to avoid spawning its
own EngineCore subprocess.

Usage (called by run_vllm_generation in grpo_train.py, not directly):
    python scripts/vllm_worker.py <request.pt> <response.pt>
"""

import os
import sys
import traceback

# Keep EngineCore in-process (belt-and-suspenders alongside external_launcher).
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def main(request_path: str, response_path: str) -> None:
    import torch
    import torch.distributed as dist
    from vllm import LLM, SamplingParams

    try:
        # Init single-rank process group (parent set MASTER_ADDR/PORT/RANK/WORLD_SIZE).
        dist.init_process_group(backend="nccl")

        request = torch.load(request_path, weights_only=False)

        llm = LLM(
            model=request["model_path"],
            trust_remote_code=True,
            dtype=request["dtype"],
            max_model_len=request["max_model_len"],
            enforce_eager=True,
            skip_tokenizer_init=True,
            seed=request["seed"],
            distributed_executor_backend="external_launcher",
        )

        # Phase 1: generate rollouts.
        phase1_params = SamplingParams(
            n=request["n"],
            temperature=request["temperature"],
            top_p=request["top_p"],
            max_tokens=request["max_tokens"],
            stop_token_ids=request["stop_token_ids"],
            skip_special_tokens=False,
        )
        prompt_token_ids: list[list[int]] = request["prompt_token_ids"]
        outputs = llm.generate(
            [{"prompt_token_ids": ids} for ids in prompt_token_ids],
            phase1_params,
        )
        rollout_response_ids: list[list[list[int]]] = [[list(out.token_ids) for out in output.outputs] for output in outputs]

        n_interrupted = 0

        # Phase 2: interruptions (if enabled).
        interruptions: dict | None = request.get("interruptions")
        if interruptions is not None:
            think_close_id: int = interruptions["think_close_id"]
            interruption_token_ids: list[int] = interruptions["interruption_token_ids"]
            eos_token_id: int = request["stop_token_ids"][0]

            # Detect truncated completions (no EOS at end, no </think> anywhere).
            needs_interruption: list[tuple[int, int]] = []
            for i, prompt_rollouts in enumerate(rollout_response_ids):
                for j, resp_ids in enumerate(prompt_rollouts):
                    has_eos = len(resp_ids) > 0 and resp_ids[-1] == eos_token_id
                    has_think_close = think_close_id in resp_ids
                    if not has_eos and not has_think_close:
                        needs_interruption.append((i, j))

            n_interrupted = len(needs_interruption)

            if n_interrupted > 0:
                # Build phase 2 prompts: original prompt + thinking + interruption phrase.
                phase2_prompt_ids = [
                    prompt_token_ids[pi] + rollout_response_ids[pi][ri] + interruption_token_ids
                    for pi, ri in needs_interruption
                ]
                phase2_params = SamplingParams(
                    n=1,
                    temperature=interruptions["phase2_temperature"],
                    top_p=interruptions["phase2_top_p"],
                    max_tokens=interruptions["phase2_max_tokens"],
                    stop_token_ids=interruptions["phase2_stop_token_ids"],
                    skip_special_tokens=False,
                )
                phase2_outputs = llm.generate(
                    [{"prompt_token_ids": ids} for ids in phase2_prompt_ids],
                    phase2_params,
                )

                # Stitch: thinking + interruption phrase + answer.
                for idx, (pi, ri) in enumerate(needs_interruption):
                    answer_ids = list(phase2_outputs[idx].outputs[0].token_ids)
                    rollout_response_ids[pi][ri] = rollout_response_ids[pi][ri] + interruption_token_ids + answer_ids

        response = {
            "rollout_response_ids": rollout_response_ids,
            "n_interrupted": n_interrupted,
            "error": None,
        }
    except Exception:
        response = {
            "rollout_response_ids": [],
            "n_interrupted": 0,
            "error": traceback.format_exc(),
        }

    torch.save(response, response_path)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
