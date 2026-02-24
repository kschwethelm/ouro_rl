# Ouro Reinforcement Learning

RL fine-tuning for [Ouro](https://ouro-llm.github.io) models using GRPO. Evaluation via [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## Evaluation Results

0-shot, max-tokens=4K-8K, temperature=1.0, top-p=0.7

| Model | [GSM8K](eval/tasks/gsm8k_thinking.yaml) | [AIME24](eval/tasks/aime24_thinking.yaml) |
|-------|--------|---------|
| Ouro-1.4B-Thinking | 90.98 ± 0.79 | 33.33 ± 8.75 |
