"""Interactive chat with ByteDance/Ouro-1.4B-Thinking via vLLM.

Run with the vLLM environment:
    uv run dev/chat.py
    uv run dev/chat.py --prompt "What is 2+2?"
"""

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

CHAT_TEMPLATE = (Path(__file__).resolve().parent.parent / "templates" / "ouro_chat.j2").read_text()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chat with Ouro via vLLM")
    p.add_argument("--model", default="ByteDance/Ouro-1.4B-Thinking")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=1700)
    p.add_argument("--system-prompt", default="You are a helpful assistant.")
    p.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--show-thinking", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prompt", type=str, help="Run a single prompt and exit (no chat loop)")
    return p.parse_args()


def split_thinking(text: str) -> tuple[str | None, str]:
    """Split thinking from answer in model output.

    When enable_thinking is used, <think> is part of the prompt so the
    generated text starts with the thinking content directly, followed
    by </think> and then the answer.
    """
    end_idx = text.find("</think>")
    if end_idx != -1:
        thinking = text[:end_idx].strip()
        answer = text[end_idx + 8 :].strip()
        return thinking, answer
    return None, text.strip()


def generate(
    llm: LLM,
    tokenizer: AutoTokenizer,
    sampling_params: SamplingParams,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool,
) -> tuple[str, str]:
    """Generate a response and return (text, finish_reason).

    finish_reason is "stop" (natural EOS) or "length" (hit max_tokens).
    """
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=CHAT_TEMPLATE,
        enable_thinking=enable_thinking,
    )
    outputs = llm.generate([prompt], sampling_params)
    output = outputs[0].outputs[0]
    return output.text, output.finish_reason


def print_response(raw_response: str, finish_reason: str, *, show_thinking: bool) -> str:
    thinking, answer = split_thinking(raw_response)
    if show_thinking and thinking:
        print(f"\n\033[2m[Thinking]\n{thinking}\033[0m\n")
    print(f"Ouro: {answer}\n")
    if finish_reason != "stop":
        print(f"\033[33m[Truncated: hit max_tokens (finish_reason={finish_reason!r})]\033[0m\n")
    return answer


def main() -> None:
    args = parse_args()
    enable_thinking = args.thinking
    show_thinking = args.show_thinking

    print(f"Loading {args.model} with vLLM...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
    )
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=args.max_new_tokens,
        skip_special_tokens=False,
    )

    if args.prompt:
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": args.prompt},
        ]
        raw, finish_reason = generate(llm, tokenizer, sampling_params, messages, enable_thinking=enable_thinking)
        print_response(raw, finish_reason, show_thinking=show_thinking)
        return

    print(f"\nReady. Chatting with {args.model}.")
    print("Type 'exit' or 'quit' to stop, '/clear' to reset conversation history.\n")

    messages: list[dict[str, str]] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            sys.exit(0)

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            sys.exit(0)
        if user_input == "/clear":
            messages = []
            print("[Conversation history cleared]\n")
            continue

        messages.append({"role": "user", "content": user_input})
        full_messages = [{"role": "system", "content": args.system_prompt}] + messages

        raw_response, finish_reason = generate(llm, tokenizer, sampling_params, full_messages, enable_thinking=enable_thinking)
        answer = print_response(raw_response, finish_reason, show_thinking=show_thinking)

        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
