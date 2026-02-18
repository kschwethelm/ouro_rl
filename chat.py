"""Interactive chat with ByteDance/Ouro-1.4B-Thinking via vLLM.

Run with the vLLM environment:
    /home/kiki/tmp/ouro_vllm_test/.venv/bin/python chat.py
"""

import sys

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "ByteDance/Ouro-1.4B-Thinking"
MAX_MODEL_LEN = 8192
MAX_NEW_TOKENS = 4096
SYSTEM_PROMPT = "You are a helpful assistant."


def split_thinking(text: str) -> tuple[str | None, str]:
    """Split <think>...</think> from the final answer."""
    if text.startswith("<think>"):
        end_idx = text.find("</think>")
        if end_idx != -1:
            thinking = text[7:end_idx].strip()
            answer = text[end_idx + 8 :].strip()
            return thinking, answer
    return None, text.strip()


def main() -> None:
    print(f"Loading {MODEL} with vLLM...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=MAX_NEW_TOKENS,
    )

    print(f"\nReady. Chatting with {MODEL}.")
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
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        prompt = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        outputs = llm.generate([prompt], sampling_params)
        raw_response = outputs[0].outputs[0].text

        thinking, answer = split_thinking(raw_response)

        if thinking:
            print(f"\n\033[2m[Thinking]\n{thinking}\033[0m\n")

        print(f"Ouro: {answer}\n")

        # Keep full response (including think block) in history so the model
        # sees its own reasoning in multi-turn conversations.
        messages.append({"role": "assistant", "content": raw_response})


if __name__ == "__main__":
    main()
