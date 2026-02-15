# llama.cpp Backend Benchmarks — Intel Core Ultra 7 255H (Arrow Lake)

**Date:** 2026-02-15
**GPU:** Intel Arc Pro 130T/140T (iGPU, shared system RAM)
**CPU:** Intel Core Ultra 7 255H (16 cores, 1 thread/core)
**oneAPI:** 2025.3 (compiler, MKL, oneDNN, TBB)
**Build:** 184c694f4 (8058)

All builds use RPATH — no `source setvars.sh` needed at runtime. Just run the binaries directly.

---

## Build Directories

| Build | Path | Key Flags |
|---|---|---|
| SYCL+oneDNN (FP16) | `build/` | `GGML_SYCL=ON GGML_SYCL_F16=ON GGML_SYCL_DNN=ON` |
| Vulkan | `build-vulkan/` | `GGML_VULKAN=ON` |
| OpenCL | `build-opencl/` | `GGML_OPENCL=ON GGML_OPENCL_USE_ADRENO_KERNELS=OFF` |
| CPU+MKL (BLAS) | `build-cpu/` | `GGML_BLAS=ON GGML_BLAS_VENDOR=Intel10_64_dyn` |

---

## Benchmark Results

### LFM2 1.2B Q8_0 (1.16 GiB)

| Backend | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| SYCL+oneDNN (FP16) | **1,945** | 8.64 |
| SYCL (FP16, no oneDNN) | 1,180 | 8.89 |
| OpenCL | 656 | **37.50** |
| Vulkan | 650 | 20.04 |
| CPU+MKL (3 threads) | 80 | 26.10 |
| CPU+MKL (8 threads) | 83 | 22.74 |

### Qwen2.5-Coder 7B Q4_K_M (4.36 GiB)

| Backend | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| SYCL+oneDNN (FP16) | 240 | 7.77 |
| SYCL (FP16, no oneDNN) | **263** | 7.38 |
| Vulkan | 159 | 9.18 |
| CPU+MKL (3 threads) | 27 | 7.56 |
| CPU+MKL (8 threads) | 33 | **11.10** |
| OpenCL | 4.17 | 3.40 |

### Qwen3-Coder 30B-A3B MoE Q4_K_M (17.35 GiB)

| Backend | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| SYCL+oneDNN (FP16) | **108.75** | 4.21 |
| SYCL hybrid (ngl=10) | 55.25 | 5.39 |
| CPU+MKL (8 threads) | 53.68 | **22.69** |
| OpenCL | 26.55 | 3.15 |
| Vulkan | CRASH (segfault) | CRASH |

---

## Best Backend Per Use Case

| Use Case | Best Backend | Token Gen (t/s) |
|---|---|---|
| Interactive chat (small models <=3B) | OpenCL | ~37.5 |
| Interactive chat (7B models) | CPU+MKL (8 threads) | ~11 |
| Interactive chat (30B MoE) | CPU+MKL (8 threads) | ~23 |
| Prompt processing / RAG / batch | SYCL+oneDNN | ~1,945 (1.2B) |
| Balanced (small models) | Vulkan | 20 tg / 650 pp |

---

## Recommended Models for This Machine (30GB RAM)

The bottleneck on this system is **shared RAM** — model + KV cache + OS must fit in ~28 GB.
Large models (30B+) work but are limited to small context (~4K), making them unusable for agentic coding.

### Best Fit for Agentic Coding

| Model | Size (Q4_K_M) | Max Context | Est. tg (CPU+MKL) | Agentic? |
|---|---|---|---|---|
| **Qwen2.5-Coder-14B** | ~9.0 GiB | **32K+** | ~8-10 t/s | **Yes (recommended)** |
| Qwen2.5-Coder-7B | 4.7 GiB | 32K+ | ~11 t/s | Yes, but weaker quality |
| DeepSeek-Coder-V2-Lite (16B MoE) | ~10.4 GiB | 32K+ | ~10-15 t/s | Yes |
| Qwen3-Coder-30B-A3B (MoE) | 17.35 GiB | 4K only | 22.69 t/s | **No** (context too small) |

### Models for Discrete GPU (2x 5060 Ti 16GB = 32GB VRAM)

| Model | Size (Q4_K_M) | Context | Notes |
|---|---|---|---|
| Qwen3-Coder-30B-A3B (MoE) | 17.35 GiB | 128K+ | Full offload, ~60-100 t/s est. |
| Qwen2.5-Coder-32B | ~19.9 GiB | 32K+ | Best dense coding model |
| CodeLlama-34B | ~20.2 GiB | 16K | Older but proven |

### Download Links (Q4_K_M GGUF)

| Model | HuggingFace Repo | Size |
|---|---|---|
| Qwen2.5-Coder-7B-Instruct | [bartowski/Qwen2.5-Coder-7B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF) | 4.7 GiB |
| Qwen2.5-Coder-14B-Instruct | [bartowski/Qwen2.5-Coder-14B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-Coder-14B-Instruct-GGUF) | 9.0 GiB |
| Qwen2.5-Coder-32B-Instruct | [bartowski/Qwen2.5-Coder-32B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-Coder-32B-Instruct-GGUF) | 19.9 GiB |
| Qwen3-Coder-30B-A3B-Instruct | [unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF) | 18.6 GiB |
| DeepSeek-Coder-V2-Lite-Instruct | [bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF](https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF) | 10.4 GiB |
| CodeLlama-34B-Instruct | [TheBloke/CodeLlama-34B-Instruct-GGUF](https://huggingface.co/TheBloke/CodeLlama-34B-Instruct-GGUF) | 20.2 GiB |
| LFM2.5-1.2B-Instruct | [LiquidAI/LFM2.5-1.2B-Instruct-GGUF](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF) | 0.7 GiB |

**Download example:**

```bash
# Recommended: Qwen2.5-Coder-14B (best for this machine)
huggingface-cli download bartowski/Qwen2.5-Coder-14B-Instruct-GGUF \
  --include "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf" \
  --local-dir ~/.cache/llama.cpp/
```

---

## Commands

### SYCL+oneDNN — Best for prompt processing

```bash
# LFM2 1.2B
./build/bin/llama-cli \
  -m ~/.cache/llama.cpp/LFM2.5-1.2B-Instruct-Q8_0.gguf \
  -ngl 99 --split-mode none --main-gpu 0

# Qwen2.5-Coder 7B
./build/bin/llama-cli \
  -m ~/.cache/llama.cpp/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf \
  -ngl 99 --split-mode none --main-gpu 0

# Qwen3-Coder 30B MoE
./build/bin/llama-cli \
  -m ~/.cache/llama.cpp/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  -ngl 99 --split-mode none --main-gpu 0
```

### Vulkan — Balanced (small/medium models only, crashes on 30B)

```bash
# LFM2 1.2B
./build-vulkan/bin/llama-cli \
  -m ~/.cache/llama.cpp/LFM2.5-1.2B-Instruct-Q8_0.gguf \
  -ngl 99

# Qwen2.5-Coder 7B
./build-vulkan/bin/llama-cli \
  -m ~/.cache/llama.cpp/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf \
  -ngl 99
```

### CPU+MKL — Best for interactive chat / token generation

```bash
# LFM2 1.2B (3 threads = fastest tg for small models)
./build-cpu/bin/llama-cli \
  -m ~/.cache/llama.cpp/LFM2.5-1.2B-Instruct-Q8_0.gguf \
  -ngl 0 -t 3

# Qwen2.5-Coder 7B (8 threads)
./build-cpu/bin/llama-cli \
  -m ~/.cache/llama.cpp/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf \
  -ngl 0 -t 8

# Qwen3-Coder 30B MoE (8 threads — best tg at 22.69 t/s, use -c 4096 to avoid OOM)
./build-cpu/bin/llama-cli \
  -m ~/.cache/llama.cpp/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  -ngl 0 -t 8 -c 4096
```

---

## Rebuild Commands

`source /opt/intel/oneapi/setvars.sh` is only needed at **build time**. The RPATH flags
embed Intel library paths into the binaries so no environment setup is needed at runtime.

### SYCL+oneDNN (FP16)

```bash
source /opt/intel/oneapi/setvars.sh
cmake -B build \
  -DGGML_SYCL=ON \
  -DGGML_SYCL_F16=ON \
  -DGGML_SYCL_DNN=ON \
  -DDNNL_DIR=/opt/intel/oneapi/dnnl/2025.3/lib/cmake/dnnl \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_BUILD_RPATH="\$ORIGIN;/opt/intel/oneapi/compiler/2025.3/lib;/opt/intel/oneapi/mkl/2025.3/lib;/opt/intel/oneapi/dnnl/2025.3/lib;/opt/intel/oneapi/tbb/2022.3/lib/intel64/gcc4.8" \
  -DCMAKE_INSTALL_RPATH="\$ORIGIN;/opt/intel/oneapi/compiler/2025.3/lib;/opt/intel/oneapi/mkl/2025.3/lib;/opt/intel/oneapi/dnnl/2025.3/lib;/opt/intel/oneapi/tbb/2022.3/lib/intel64/gcc4.8"
cmake --build build --config Release -j$(nproc)
```

### Vulkan

```bash
cmake -B build-vulkan \
  -DGGML_VULKAN=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_BUILD_RPATH="\$ORIGIN" \
  -DCMAKE_INSTALL_RPATH="\$ORIGIN"
cmake --build build-vulkan --config Release -j$(nproc)
```

### OpenCL

```bash
cmake -B build-opencl \
  -DGGML_OPENCL=ON \
  -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_BUILD_RPATH="\$ORIGIN" \
  -DCMAKE_INSTALL_RPATH="\$ORIGIN"
cmake --build build-opencl --config Release -j$(nproc)
```

### CPU+MKL (BLAS)

```bash
source /opt/intel/oneapi/setvars.sh
cmake -B build-cpu \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=Intel10_64_dyn \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_BUILD_RPATH="\$ORIGIN;/opt/intel/oneapi/compiler/2025.3/lib;/opt/intel/oneapi/mkl/2025.3/lib" \
  -DCMAKE_INSTALL_RPATH="\$ORIGIN;/opt/intel/oneapi/compiler/2025.3/lib;/opt/intel/oneapi/mkl/2025.3/lib"
cmake --build build-cpu --config Release -j$(nproc)
```

---

## Benchmark Commands

```bash
# SYCL+oneDNN
./build/bin/llama-bench -m <model.gguf> -ngl 99

# Vulkan
./build-vulkan/bin/llama-bench -m <model.gguf> -ngl 99

# OpenCL
./build-opencl/bin/llama-bench -m <model.gguf> -ngl 99

# CPU+MKL
./build-cpu/bin/llama-bench -m <model.gguf> -ngl 0 -t 8
```

---

## Serving for Agentic Coding

The server exposes an OpenAI-compatible API. Quick start:

```bash
./serve-qwen3-coder.sh
```

Or manually:

```bash
./build-cpu/bin/llama-server \
  -m ~/.cache/llama.cpp/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  -ngl 0 -t 8 -c 4096 \
  --host 0.0.0.0 --port 8080 \
  -np 1 --jinja
```

### Connecting Agentic Tools

The server provides an OpenAI-compatible endpoint at `http://localhost:8080`.

**Continue.dev (VS Code):**

```json
{
  "models": [{
    "title": "Qwen3-Coder-30B",
    "provider": "openai",
    "model": "qwen3-coder-30b",
    "apiBase": "http://localhost:8080/v1",
    "apiKey": "none"
  }]
}
```

**Open Interpreter / aider / other tools:**

```bash
export OPENAI_API_BASE=http://localhost:8080/v1
export OPENAI_API_KEY=none
```

**curl test:**

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-coder-30b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Server Key Flags

| Flag | Value | Purpose |
|---|---|---|
| `-c 4096` | Context size | Keep low to avoid OOM (30GB RAM) |
| `-np 1` | 1 parallel slot | Single user, saves RAM |
| `--jinja` | Enable Jinja templates | Proper chat template rendering |
| `-t 8` | 8 threads | Optimal for 7B+ models |
| `--host 0.0.0.0` | Listen on all interfaces | Access from other devices on LAN |

---

## Notes

- All builds use **RPATH** — Intel library paths are embedded in the binaries. No `source setvars.sh` needed at runtime. Only needed at build time.
- On iGPU systems, CPU+MKL often beats GPU backends for token generation due to zero GPU dispatch overhead and direct memory access.
- SYCL+oneDNN excels at prompt processing (batch compute) but has high per-token overhead on iGPUs.
- Vulkan driver on Arrow Lake is experimental ("MESA: warning: Support for this platform is experimental with Xe KMD"). It segfaults on large models (17GB+).
- oneDNN gave +65% prompt processing improvement on the 1.2B model over plain SYCL.
- 3 threads is optimal for small model tg on CPU; 8 threads is better for 7B+ models.
- For 30B MoE models, use `-c 4096` to avoid OOM on 30GB RAM systems.
- OpenCL is the fastest backend for token generation on small models (<=3B) at 37.5 t/s, but falls back to slow unoptimized paths on larger quantized models (7B+).
- OpenCL must be built with `-DGGML_OPENCL_USE_ADRENO_KERNELS=OFF` on non-Adreno GPUs (Intel, AMD, etc.).
- SYCL hybrid offload (partial `-ngl`) does not help on this system — even ngl=10 drops tg from 22.69 to 5.39 due to CPU/iGPU transfer overhead.
- SYCL requires `source /opt/intel/oneapi/setvars.sh` at **runtime** despite RPATH (Level Zero loader discovery needs oneAPI environment variables).
