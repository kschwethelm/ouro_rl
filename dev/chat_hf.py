"""Interactive chat with ByteDance/Ouro-1.4B-Thinking via HuggingFace Transformers.

Run with:
    uv run dev/chat_hf.py
    uv run dev/chat_hf.py --prompt "What is 2+2?"
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CHAT_TEMPLATE = (Path(__file__).resolve().parent.parent / "templates" / "ouro_chat.j2").read_text()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chat with Ouro via HuggingFace Transformers")
    p.add_argument("--model", default="ByteDance/Ouro-1.4B-Thinking")
    p.add_argument("--max-new-tokens", type=int, default=2048)
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
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    enable_thinking: bool,
) -> str:
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        chat_template=CHAT_TEMPLATE,
        enable_thinking=enable_thinking,
    )
    input_ids = torch.tensor([prompt_ids], device=model.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)


def print_response(raw_response: str, *, show_thinking: bool) -> str:
    thinking, answer = split_thinking(raw_response)
    if show_thinking and thinking:
        print(f"\n\033[2m[Thinking]\n{thinking}\033[0m\n")
    print(f"Ouro: {answer}\n")
    return answer


def main() -> None:
    args = parse_args()
    enable_thinking = args.thinking
    show_thinking = args.show_thinking

    print(f"Loading {args.model} with transformers...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    if args.prompt:
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": args.prompt},
        ]
        raw = generate(
            model,
            tokenizer,
            messages,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=enable_thinking,
        )
        print_response(raw, show_thinking=show_thinking)
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

        raw_response = generate(
            model,
            tokenizer,
            full_messages,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=enable_thinking,
        )
        answer = print_response(raw_response, show_thinking=show_thinking)

        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
