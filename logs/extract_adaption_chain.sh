#!/bin/zsh
# Sequential Adaption extractions (84 prompts x 200 samples each), resumable via .partial.jsonl checkpoints
cd "/Users/alibelal/Swarm alignment"
for m in qwen3.8-27b gemma-4-31b-it glm-5.2 qwen3.5-397b gemma-4-26b-a4b-it; do
  echo "=== $m start $(date +%H:%M:%S)"
  .venv/bin/python -u -m swarm extract --backend adaption --model $m --samples 200 --workers 4 --inner-workers 16 --out policies/adaption_$m.json 2>&1 | grep --line-buffered -E "progress|resumed|Saved|Error|error|Traceback"
done
echo "=== chain done $(date +%H:%M:%S)"
