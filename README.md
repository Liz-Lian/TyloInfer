<p align="center">
  <img src="TyloInferStudio/src/assets/tylogi-ai-lab-mark.svg" width="72" alt="Tylogi AI Lab">
</p>

<h1 align="center">TyloInfer</h1>
<p align="center"><b>Fast local inference. Large models, smaller memory budgets.</b><br>Native CUDA and Metal inference, with a desktop workspace and APIs for your agents.</p>

<p align="center">English · <a href="README.zh-CN.md">简体中文</a></p>
<p align="center"><a href="#features">Features</a> · <a href="#performance">Performance</a> · <a href="#quick-start">Quick start</a> · <a href="#recommended-models">Models</a> · <a href="docs/README.md">Documentation</a></p>

TyloInfer runs dense and mixture-of-experts models on your own hardware. Keep weights in memory for maximum speed, or stream experts from SSD to run models larger than your memory. Use it as a local API server, in your browser, or through TyloInfer Studio.

## Features

- **GGUF and TQ, natively.** Optimized kernels for Q_K, IQ, Gyro TQK/TQP, and TQ's per-row 1–8-bit precision. MoE Gate, Up, and Down projections can use different formats without falling back to separate expert execution.
- **More model, less memory.** GPU, RAM, and SSD expert storage with asynchronous reads, prefetch, and shared expert reuse. PLE and Engram tables stream from SSD instead of occupying the weight budget.
- **Faster generation.** Native MTP and DFlash2 speculative decoding, fused projections, and optimized dense and routed FFNs.
- **Long conversations.** Persistent RAM/SSD prefix caching, per-model context settings, KV quantization, and streaming sparse attention with a bounded resident KV budget.
- **Ready for agents.** OpenAI- and Anthropic-compatible APIs, streaming tool calls, model aliases, API-key authentication, and continuous batching.
- **A complete workspace.** Studio brings together model browsing and downloads, model management, live resource monitoring, precision visualization, and Agent launchers.

Apple Silicon uses Metal; NVIDIA GPUs use CUDA on Linux and Windows. ROCm support is in development. [Model and backend coverage →](docs/runtime/support.md)

## Performance

Apple M5 Max · 40-core GPU · 128 GiB · GGUF IQ3_XXS · 4,096-token input. Speeds in tokens/s.

### Qwen3.8-Flash-Next

| Engine | Prompt processing | Generation | With DFlash2 |
| --- | ---: | ---: | ---: |
| TyloInfer | 2,572 | 73.7 | **162.4** |
| [Strata 0.1.52](https://github.com/Niko1221/Strata/commit/5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d) | — | **76.3** | — |
| [Splash 1.4.1](https://github.com/incoai/splash/releases/tag/1.4.1) | **2,599** | — | 148.9 |

### Qwen3.8-27B

| Engine | Prompt processing | Generation | With DFlash2 |
| --- | ---: | ---: | ---: |
| TyloInfer | 1,001 | 40.1 | 121.5 |
| [Splash 1.5.0](https://github.com/incoai/splash/releases/tag/1.5.0) | **1,085** | — | **131.2** |

## Quick start

### Build and serve

Install [uv](https://docs.astral.sh/uv/), CMake, and a C++ toolchain. CUDA builds also need the NVIDIA CUDA Toolkit; the browser UI needs Node.js and npm.

```sh
git clone https://github.com/Tylogi/TyloInfer.git
cd TyloInfer

# Apple Silicon
uv sync --extra metal
uv run tyloinfer build --backend metal

# Start the server and discover models in this directory
uv run tyloinfer serve --model-dir ./models
```

For NVIDIA, replace the two build steps with:

```sh
uv sync
uv run tyloinfer build --backend cuda
```

Open **http://127.0.0.1:8090/**, download a model, and load it from **Models**. To load an existing model at startup:

```sh
uv run tyloinfer serve --model /models/model.gguf
```

The same command accepts `.tq` checkpoints. For sharded models, select the first shard and keep the complete set in one directory.

### Connect your tools

| Protocol | Address |
| --- | --- |
| OpenAI-compatible base URL | `http://127.0.0.1:8090/v1` |
| Anthropic Messages endpoint | `http://127.0.0.1:8091/v1/messages` |

Point your coding agent or chat client at the matching address and choose a loaded model. For access from another machine, set `TYLOINFER_SERVER_API_KEY` and start with `--host 0.0.0.0`.

### Desktop

TyloInfer Studio uses the same server and model collection as the browser. [Build a desktop package →](docs/packaging.md)

## Recommended models

| Model | Download | A good fit for |
| --- | --- | --- |
| Qwen3.5-9B | [TQ](https://huggingface.co/Tylogi/Qwen3.5-9B-TQ) | A compact starting point; try TQ4. |
| Qwen3.8-27B | [GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) | Dense-model quality in a manageable footprint; IQ3_XXS is a compact choice. |
| Qwen3.8-Flash-Next | [GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) · [TQ](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-TQ) | Fast MoE inference for coding and everyday agent work. |
| DeepSeek-V4.1-Flash-1017 | [TQ](https://huggingface.co/Tylogi/DeepSeek-V4.1-Flash-1017-TQ) | Large-model reasoning with SSD-streamed experts and Engram. |

Studio can browse Hugging Face and ModelScope, download model folders and shards, and discover completed downloads in **Models**. Include any PLE, Engram, or MTP companion files required by your chosen model.

TQ models are produced with [TyloQuant](https://github.com/Tylogi/TyloQuant), which allocates precision independently to each output row.

## Documentation

[Build](docs/cli/build.md) · [Serving](docs/cli/serve.md) · [HTTP API](docs/api/http.md) · [TQ](docs/formats/tq.md) · [Runtime support](docs/runtime/support.md) · [Packaging](docs/packaging.md)

Licensed under [Apache 2.0](LICENSE). Third-party acknowledgements are in [NOTICE](NOTICE) and [LICENSES](LICENSES).
