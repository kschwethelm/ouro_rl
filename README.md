# Ouro Reinforcement Learning

RL fine-tuning for [Ouro](https://ouro-llm.github.io) models using GRPO. Evaluation via [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## Evaluation Results

0-shot, max-tokens=4K-8K, temperature=1.0, top-p=0.7

| Model | [GSM8K](eval/tasks/gsm8k_thinking.yaml) | [AIME24](eval/tasks/aime24_thinking.yaml) | [MATH500](eval/tasks/math500_thinking.yaml) | [GPQA](eval/tasks/gpqa_main_thinking.yaml) |
|-------|--------|---------|----------|------|
| Ouro-1.4B-Thinking | 93.40 ± 0.68 | 30.00 ± 8.51 | 65.00 ± 2.14 | 25.45 ± 2.06 |
