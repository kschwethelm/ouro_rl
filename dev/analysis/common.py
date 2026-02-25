"""Shared utilities for Ouro latent state analysis scripts.

Provides model loading, generation with intermediate state capture, and
helpers for computing logits from captured hidden states.

The Ouro model (Universal Transformer) loops through its decoder layers
`total_ut_steps` times (default 4). After each UT step, a shared RMSNorm
is applied. We hook into this norm to capture the normed hidden states at
each step — no model modifications needed.
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from ouro_rl.modeling import CHAT_TEMPLATE, EOS_TOKEN_ID, PAD_TOKEN_ID, OuroForCausalLM
from ouro_rl.modeling.constants import DEFAULT_MODEL


class PerLayerStateHook:
    """Captures the residual stream at each layer boundary within each UT step.

    Registers a forward pre-hook on layer 0 (to capture x^0, the input to the
    first layer) and forward hooks on all layers (to capture x^1 through x^N,
    their outputs). Groups these into per-UT-step lists of N+1 state tensors.

    Only stores the last token per sequence (matching ETD paper methodology —
    under causal attention, the last token sees the full context).
    """

    def __init__(self, num_layers: int, total_ut_steps: int):
        self.num_layers = num_layers
        self.total_ut_steps = total_ut_steps
        self._layer_outputs: list[torch.Tensor] = []
        self._layer0_input: torch.Tensor | None = None
        # states[ut_step][layer_boundary] = tensor of shape (D,) for last token
        self.states: list[list[torch.Tensor]] = []

    def reset(self):
        self._layer_outputs = []
        self._layer0_input = None
        self.states = []

    def layer0_pre_hook(self, module, args):  # noqa: ARG002
        """Forward pre-hook on layer 0: captures input to first layer (x^0)."""
        hidden_states = args[0]
        # Store last token only: (D,)
        self._layer0_input = hidden_states[0, -1, :].detach().cpu()

    def make_post_hook(self, layer_idx: int):
        """Returns a forward hook for the given layer that captures its output."""

        def hook(module, input, output):  # noqa: ARG001
            # output is hidden_states tensor (batch, seq_len, D)
            self._layer_outputs.append(output[0, -1, :].detach().cpu())

            if layer_idx == self.num_layers - 1:
                # End of UT step — assemble x^0 through x^N
                assert self._layer0_input is not None
                step_states = [self._layer0_input] + self._layer_outputs
                self.states.append(step_states)
                self._layer_outputs = []
                self._layer0_input = None

        return hook


class LatentStateHook:
    """Captures normed hidden states from each UT step during Ouro forward passes.

    The Ouro model's RMSNorm (model.model.norm) is called once per UT step
    in the recurrence loop. For total_ut_steps=4, the hook fires 4 times per
    forward pass. We group consecutive calls to reconstruct per-step states.
    """

    def __init__(self, total_ut_steps: int):
        self.total_ut_steps = total_ut_steps
        self.reset()

    def reset(self):
        self._buffer: list[torch.Tensor] = []
        self._call_count = 0
        self.prefill_states: list[torch.Tensor] | None = None
        self.decode_states: list[list[torch.Tensor]] = []

    def __call__(self, module, input, output):  # noqa: ARG002
        self._buffer.append(output.detach().cpu())
        self._call_count += 1

        if self._call_count % self.total_ut_steps == 0:
            states = self._buffer[-self.total_ut_steps :]
            if states[0].shape[1] > 1:
                # Prefill: multiple tokens processed at once
                self.prefill_states = states
            else:
                # Decode: single token per forward pass
                self.decode_states.append(states)


class ExitGateHook:
    """Captures exit gate logits from each UT step during Ouro forward passes.

    The exit gate (model.model.early_exit_gate) is an nn.Linear(hidden_size, 1)
    called once per UT step after the norm. Output shape: (batch, seq_len, 1).
    """

    def __init__(self, total_ut_steps: int):
        self.total_ut_steps = total_ut_steps
        self.reset()

    def reset(self):
        self._buffer: list[torch.Tensor] = []
        self._call_count = 0
        self.prefill_logits: list[torch.Tensor] | None = None
        self.decode_logits: list[list[torch.Tensor]] = []

    def __call__(self, module, input, output):  # noqa: ARG002
        self._buffer.append(output.detach().cpu())
        self._call_count += 1

        if self._call_count % self.total_ut_steps == 0:
            logits = self._buffer[-self.total_ut_steps :]
            if logits[0].shape[1] > 1:
                self.prefill_logits = logits
            else:
                self.decode_logits.append(logits)


def load_model(
    model_name: str = DEFAULT_MODEL,
) -> tuple[OuroForCausalLM, AutoTokenizer]:
    """Load Ouro model and tokenizer with correct chat template."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.chat_template = CHAT_TEMPLATE
    model = OuroForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def load_gsm8k(split: str = "test") -> list[dict]:
    """Load GSM8K dataset and return list of {question, answer} dicts."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [{"question": row["question"], "answer": row["answer"]} for row in ds]


def format_prompt(
    question: str,
    tokenizer: AutoTokenizer,
    *,
    enable_thinking: bool = True,
) -> list[int]:
    """Format a question as a chat prompt and tokenize."""
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        chat_template=CHAT_TEMPLATE,
        enable_thinking=enable_thinking,
    )


def generate_with_latent_tracking(
    model: OuroForCausalLM,
    prompt_ids: list[int],
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    seed: int = 42,
    n_loops: int | None = None,
) -> tuple[
    list[int],
    list[list[torch.Tensor]],
    list[list[torch.Tensor]],
    list[list[float]],
    list[list[float]],
]:
    """Generate response while tracking latent states and exit gate at each UT step.

    Hooks into the model's RMSNorm layer to capture intermediate hidden states
    and the exit gate to capture gate logits at each Universal Transformer step.

    Args:
        model: The Ouro causal LM model.
        prompt_ids: Tokenized prompt IDs.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature (0.0 = greedy).
        seed: Random seed for sampling.
        n_loops: Override total_ut_steps (default: use model config).

    Returns:
        generated_tokens: Generated token IDs (EOS/PAD stripped, aligned with states).
        input_latent_states: Per-input-token list of UT-step hidden states.
            input_latent_states[token_idx][ut_step] = tensor of shape (D,)
        output_latent_states: Per-output-token list of UT-step hidden states.
            output_latent_states[token_idx][ut_step] = tensor of shape (D,)
        input_exit_pdf: Per-input-token exit probability distribution over UT steps.
            input_exit_pdf[token_idx][ut_step] = float probability
        output_exit_pdf: Per-output-token exit probability distribution over UT steps.
            output_exit_pdf[token_idx][ut_step] = float probability
    """
    total_ut_steps = model.config.total_ut_steps
    orig_ut_steps = total_ut_steps

    # Optionally override number of UT steps
    if n_loops is not None and n_loops != total_ut_steps:
        total_ut_steps = n_loops
        model.model.total_ut_steps = n_loops
        model.config.total_ut_steps = n_loops

    hook = LatentStateHook(total_ut_steps)
    gate_hook = ExitGateHook(total_ut_steps)
    handle = model.model.norm.register_forward_hook(hook)
    gate_handle = model.model.early_exit_gate.register_forward_hook(gate_hook)

    try:
        input_ids = torch.tensor([prompt_ids], device=model.device)

        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": EOS_TOKEN_ID,
            "pad_token_id": PAD_TOKEN_ID,
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            if seed is not None:
                torch.manual_seed(seed)
            output_ids = model.generate(input_ids=input_ids, **gen_kwargs)

        # Extract generated token IDs
        raw_output = output_ids[0][len(prompt_ids) :].tolist()
        # Strip trailing EOS/PAD
        while raw_output and raw_output[-1] in (EOS_TOKEN_ID, PAD_TOKEN_ID):
            raw_output.pop()

        # Process prefill states → per-token state lists for input
        input_latent_states: list[list[torch.Tensor]] = []
        if hook.prefill_states:
            num_input = len(prompt_ids)
            for token_idx in range(num_input):
                token_states = [s[0, token_idx, :] for s in hook.prefill_states]
                input_latent_states.append(token_states)

        # Process decode states → per-token state lists for output
        output_latent_states: list[list[torch.Tensor]] = []
        for decode_step in hook.decode_states:
            token_states = [s[0, 0, :] for s in decode_step]
            output_latent_states.append(token_states)

        # Process exit gate logits → per-token exit PDF
        def compute_exit_pdf(gate_logits_per_token: list[list[torch.Tensor]]) -> list[list[float]]:
            """Compute exit probability distribution from gate logits.

            Uses the same decomposition as the model: p_i = lambda_i * prod(1 - lambda_j, j<i),
            where lambda_i = sigmoid(gate_logit_i). Last step gets all remaining probability.
            """
            pdfs = []
            for token_logits in gate_logits_per_token:
                remaining = 1.0
                pdf = []
                for i, logit in enumerate(token_logits):
                    lambda_i = torch.sigmoid(logit).item()
                    if i < len(token_logits) - 1:
                        p_i = lambda_i * remaining
                        remaining *= 1.0 - lambda_i
                    else:
                        p_i = remaining  # last step gets all remaining
                    pdf.append(p_i)
                pdfs.append(pdf)
            return pdfs

        # Extract per-token gate logits for input (from prefill)
        input_gate_per_token: list[list[torch.Tensor]] = []
        if gate_hook.prefill_logits:
            num_input = len(prompt_ids)
            for token_idx in range(num_input):
                token_logits = [g[0, token_idx, 0] for g in gate_hook.prefill_logits]
                input_gate_per_token.append(token_logits)

        # Extract per-token gate logits for output (from decode steps)
        output_gate_per_token: list[list[torch.Tensor]] = []
        for decode_step in gate_hook.decode_logits:
            token_logits = [g[0, 0, 0] for g in decode_step]
            output_gate_per_token.append(token_logits)

        input_exit_pdf = compute_exit_pdf(input_gate_per_token) if input_gate_per_token else []
        output_exit_pdf = compute_exit_pdf(output_gate_per_token) if output_gate_per_token else []

        # Align: truncate tokens to match states (last generated token has no states)
        num_with_states = len(output_latent_states)
        generated_tokens = raw_output[:num_with_states]
        output_exit_pdf = output_exit_pdf[:num_with_states]

    finally:
        handle.remove()
        gate_handle.remove()
        # Restore original UT steps if overridden
        if n_loops is not None and n_loops != orig_ut_steps:
            model.model.total_ut_steps = orig_ut_steps
            model.config.total_ut_steps = orig_ut_steps

    return generated_tokens, input_latent_states, output_latent_states, input_exit_pdf, output_exit_pdf


def compute_intermediate_logits(
    model: OuroForCausalLM,
    input_latent_states: list[list[torch.Tensor]],
    output_latent_states: list[list[torch.Tensor]],
) -> list[torch.Tensor]:
    """Compute logits at each UT step for KL divergence analysis.

    Takes the per-token, per-step hidden states captured during generation
    and runs them through the lm_head to get logits.

    Returns:
        List of total_ut_steps tensors, each (total_tokens, vocab_size) on CPU.
    """
    all_states = input_latent_states + output_latent_states
    if not all_states:
        return []

    total_ut_steps = len(all_states[0])
    intermediate_logits = []

    with torch.no_grad():
        for step_idx in range(total_ut_steps):
            # Stack all tokens' hidden states for this UT step: (num_tokens, D)
            step_hidden = torch.stack([states[step_idx] for states in all_states])
            step_hidden = step_hidden.unsqueeze(0).to(model.device)  # (1, num_tokens, D)
            logits = model.lm_head(step_hidden)  # (1, num_tokens, vocab)
            intermediate_logits.append(logits[0].cpu())  # (num_tokens, vocab)

    return intermediate_logits


def forward_with_layer_tracking(
    model: OuroForCausalLM,
    prompt_ids: list[int],
    n_loops: int | None = None,
) -> list[list[torch.Tensor]]:
    """Run a forward pass and capture per-layer hidden states at each UT step.

    Hooks into every decoder layer to capture the residual stream at each
    layer boundary. Only stores the last token (under causal attention, it
    sees the full context).

    Args:
        model: The Ouro causal LM model.
        prompt_ids: Tokenized prompt IDs.
        n_loops: Override total_ut_steps (default: use model config).

    Returns:
        states[ut_step][layer_boundary] = tensor of shape (D,) for last token.
        layer_boundary 0 = input to layer 0, 1 = output of layer 0, ..., N = output of layer N-1.
    """
    total_ut_steps = model.config.total_ut_steps
    orig_ut_steps = total_ut_steps
    num_layers = model.config.num_hidden_layers

    if n_loops is not None and n_loops != total_ut_steps:
        total_ut_steps = n_loops
        model.model.total_ut_steps = n_loops
        model.config.total_ut_steps = n_loops

    hook = PerLayerStateHook(num_layers, total_ut_steps)
    handles: list[torch.utils.hooks.RemovableHook] = []

    try:
        # Pre-hook on layer 0 to capture input to first layer
        handles.append(model.model.layers[0].register_forward_pre_hook(hook.layer0_pre_hook))
        # Post-hook on every layer to capture outputs
        for i in range(num_layers):
            handles.append(model.model.layers[i].register_forward_hook(hook.make_post_hook(i)))

        input_ids = torch.tensor([prompt_ids], device=model.device)
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)

        return hook.states

    finally:
        for h in handles:
            h.remove()
        if n_loops is not None and n_loops != orig_ut_steps:
            model.model.total_ut_steps = orig_ut_steps
            model.config.total_ut_steps = orig_ut_steps
