#!/bin/bash
# Serve Qwen3-Coder 30B MoE for agentic coding
# Uses CPU+MKL backend (best token generation: ~22 t/s)
# OpenAI-compatible API at http://localhost:8080
#
# Performance notes:
#   -c 32768   : 32K context (agentic tools send 10-12K+ on first message)
#   -ctk q4_0  : aggressively quantize KV cache keys (4x smaller than F16)
#   -ctv q4_0  : aggressively quantize KV cache values
#   -fa on     : flash attention (memory-efficient, critical at large contexts)
#
# KV cache memory at 32K context:
#   F16:  ~3 GB
#   Q8_0: ~1.5 GB
#   Q4_0: ~0.75 GB  <-- we use this

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

exec "$SCRIPT_DIR/build-cpu/bin/llama-server" \
  -m ~/.cache/llama.cpp/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  -ngl 0 \
  -t 8 \
  -c 32768 \
  -ctk q4_0 \
  -ctv q4_0 \
  -fa on \
  --host 0.0.0.0 \
  --port 8080 \
  -np 1 \
  --jinja
