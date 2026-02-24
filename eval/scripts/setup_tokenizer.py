"""Save a patched Ouro tokenizer with corrected bos/eos and enable_thinking support.

Upstream Ouro tokenizer ships with bos/eos/pad all set to <|endoftext|> (id=0).
This script fixes them and injects a chat template that supports enable_thinking.

Usage:
    uv run python eval/scripts/setup_tokenizer.py [--model MODEL] [--output DIR]
"""

import argparse
from pathlib import Path

from transformers import AutoTokenizer


def setup_tokenizer(
    model: str = "ByteDance/Ouro-1.4B-Thinking",
    output: str = "models/tokenizer",
    template: str = "templates/ouro_chat.j2",
) -> None:
    output_path = Path(output)
    if output_path.exists():
        print(f"Tokenizer already exists at {output_path}, skipping.")
        return

    print(f"Loading tokenizer from {model}...")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    tokenizer.bos_token = "<|im_start|>"
    tokenizer.eos_token = "<|im_end|>"

    template_path = Path(template)
    if template_path.exists():
        tokenizer.chat_template = template_path.read_text()
        print(f"Loaded chat template from {template_path}")
    else:
        print(f"WARNING: Template {template_path} not found, keeping default.")

    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(output_path))
    print(f"Saved patched tokenizer to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ByteDance/Ouro-1.4B-Thinking")
    parser.add_argument("--output", default="models/tokenizer")
    parser.add_argument("--template", default="templates/ouro_chat.j2")
    args = parser.parse_args()
    setup_tokenizer(model=args.model, output=args.output, template=args.template)
