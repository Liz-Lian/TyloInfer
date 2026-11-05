<p align="center">
  <img src="TyloInferStudio/src/assets/tylogi-ai-lab-mark.svg" width="72" alt="Tylogi AI Lab">
</p>

<h1 align="center">TyloInfer</h1>
<p align="center"><b>高速本地推理，用更少内存运行更大模型。</b><br>原生 CUDA / Metal 推理，配备桌面工作台与面向 Agent 的服务接口。</p>

<p align="center"><a href="README.md">English</a> · 简体中文</p>
<p align="center"><a href="#主要功能">功能</a> · <a href="#推理性能">性能</a> · <a href="#快速开始">快速开始</a> · <a href="#推荐模型">推荐模型</a> · <a href="docs/README.md">文档</a></p>

TyloInfer 在你的设备上运行稠密模型和 MoE 模型。权重全常驻时追求最高速度，内存不足时通过 SSD 流式专家运行更大模型。既可作为本地 API 服务，也可通过浏览器或 TyloInfer Studio 使用。

## 主要功能

- **原生 GGUF 与 TQ。** 为 Q_K、IQ、Gyro TQK/TQP 和逐行 1～8 bit 的 TQ 提供优化算子；MoE 的 Gate、Up、Down 可分别采用不同格式，无需退回逐专家执行。
- **更低的内存门槛。** 联合 GPU、RAM 与 SSD 管理专家权重，支持异步读取、预取与专家复用；PLE、Engram 表按需从 SSD 读取，不占据权重常驻预算。
- **高速生成。** 原生 MTP、DFlash2 推测解码，配合投影融合和稠密／MoE FFN 优化。
- **长对话支持。** RAM／SSD 持久化前缀缓存、单模型上下文设置、KV 量化，以及可限制 KV 常驻量的流式稀疏注意力。
- **面向 Agent 的服务。** OpenAI 与 Anthropic 兼容 API、流式工具调用、模型别名、API Key 鉴权与连续批处理。
- **完整工作台。** Studio 集成模型浏览下载、模型管理、实时资源监控、精度可视化与 Agent 启动入口。

Apple Silicon 使用 Metal；NVIDIA GPU 支持 Linux 和 Windows 下的 CUDA。ROCm 支持开发中。[查看模型与后端支持 →](docs/runtime/support.md)

## 推理性能

Apple M5 Max · 40 核 GPU · 128 GiB · GGUF IQ3_XXS · 4,096 token 输入，单位为 token/s。

### Qwen3.8-Flash-Next

| 推理引擎 | Prompt Processing | Generation | DFlash2 生成 |
| --- | ---: | ---: | ---: |
| TyloInfer | 2,572 | 73.7 | **162.4** |
| [Strata 0.1.57](https://github.com/Niko1221/Strata/commit/9f7d977d220e9fdf2ec53cf2620cdadac1cfb24e) | 1,315 | **77.8** | 113.4 |
| [Splash 1.4.1](https://github.com/incoai/splash/releases/tag/1.4.1) | **2,599** | — | 148.9 |

### Qwen3.8-27B

| 推理引擎 | Prompt Processing | Generation | DFlash2 生成 |
| --- | ---: | ---: | ---: |
| TyloInfer | 1,010 | 40.1 | 131.1 |
| [Splash 1.5.0](https://github.com/incoai/splash/releases/tag/1.5.0) | **1,085** | — | **131.2** |

### CUDA · Qwen3.8-27B

NVIDIA RTX 3090 Ti · 4K input · 256 output

**GSQ-RCO IQ2_XS**

| 推理引擎 | Prompt Processing | Generation | 原生 MTP | DFlash2（最快草稿数） |
| --- | ---: | ---: | ---: | ---: |
| TyloInfer | **2,237** | **83.5** | **171.4** | **206.8**（≤6） |
| NInfer-all | 1,729 | 73.0 | 150.2 | 151.1（6） |
| llama.cpp b12125 | 1,601 | 57.2 | 93.9 | 104.9（5） |
| ik_llama.cpp | 1,371 | 52.4 | 83.6 | 56.3（7） |

**UD-Q4_K_XL**

| 推理引擎 | Prompt Processing | Generation | 原生 MTP | DFlash2（最快草稿数） |
| --- | ---: | ---: | ---: | ---: |
| TyloInfer | **2,174** | 45.3 | **114.5** | **179.9**（7） |
| NInfer-all | 2,008 | **46.9** | 110.5 | 136.3（7） |
| llama.cpp b12125 | 1,799 | 43.9 | 89.3 | 109.9（5） |
| ik_llama.cpp | 1,513 | 43.6 | 78.0 | 58.2（7） |
| OpenJet | 1,750 | 39.1 | 103.4 | — |
| tqLLM | 1,821 | 42.5 | 108.2 | 173.1（7） |

- NInfer-all 的 prefill 使用 **8bit（INT8）激活**。tqLLM 也启用了 INT8 prefill GEMM。原生 MTP 使用 3 个草稿，TyloInfer 自适应至多 3 个。
- ik_llama.cpp、tqLLM 在 WSL2 下运行，其余为 Windows。
- tqLLM 使用近似的 4bit 候选筛选／INT8 重算输出头，DFlash2 配套权重为 Q8_0；其它 DFlash2 配套权重为 Q4_0。

## 快速开始

### 源码构建与部署

先安装 [uv](https://docs.astral.sh/uv/)、CMake 与 C++ 编译工具链。CUDA 构建还需 NVIDIA CUDA Toolkit；浏览器界面需要 Node.js 和 npm。

```sh
git clone https://github.com/Tylogi/TyloInfer.git
cd TyloInfer

# Apple Silicon
uv sync --extra metal
uv run tyloinfer build --backend metal

# 启动服务，从此目录发现模型
uv run tyloinfer serve --model-dir ./models
```

NVIDIA 设备将上面两条构建命令替换为：

```sh
uv sync
uv run tyloinfer build --backend cuda
```

打开 **http://127.0.0.1:8090/**，下载模型后在「模型」中加载。也可启动时直接加载已有模型：

```sh
uv run tyloinfer serve --model /models/model.gguf
```

`.tq` 模型同样使用此命令。分片模型选择首个分片，并将完整分片放在同一目录。

### 接入 Agent 与客户端

| 协议 | 地址 |
| --- | --- |
| OpenAI 兼容 Base URL | `http://127.0.0.1:8090/v1` |
| Anthropic Messages 请求地址 | `http://127.0.0.1:8091/v1/messages` |

在 Agent 或聊天客户端中填写对应地址，选择已加载的模型即可。需要从其它设备访问时，设置 `TYLOINFER_SERVER_API_KEY` 并使用 `--host 0.0.0.0` 启动。

### 桌面端

TyloInfer Studio 与浏览器共用服务和模型目录。[构建桌面安装包 →](docs/packaging.md)

## 推荐模型

| 模型 | 下载 | 适合场景 |
| --- | --- | --- |
| Qwen3.5-9B | [TQ](https://huggingface.co/Tylogi/Qwen3.5-9B-TQ) | 轻量起步，可从 TQ4 开始。 |
| Qwen3.8-27B | [GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) | 在适中体积下追求稠密模型表现；IQ3_XXS 是紧凑选择。 |
| Qwen3.8-Flash-Next | [GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) · [TQ](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-TQ) | 高速 MoE，适合编程与日常 Agent 工作。 |
| DeepSeek-V4.1-Flash-1017 | [TQ](https://huggingface.co/Tylogi/DeepSeek-V4.1-Flash-1017-TQ) | 通过 SSD 流式专家与 Engram 运行大型推理模型。 |

Studio 支持浏览 Hugging Face、ModelScope，下载模型目录及分片；下载完成后会自动出现在「模型」中。按模型要求同时下载 PLE、Engram 或 MTP 配套文件。

TQ 模型由 [TyloQuant](https://github.com/Tylogi/TyloQuant) 生成，可为每个输出神经元独立分配精度。

## 文档

[构建](docs/cli/build.md) · [服务部署](docs/cli/serve.md) · [HTTP API](docs/api/http.md) · [TQ](docs/formats/tq.md) · [运行时支持](docs/runtime/support.md) · [打包](docs/packaging.md)

采用 [Apache 2.0](LICENSE) 许可；第三方声明见 [NOTICE](NOTICE) 与 [LICENSES](LICENSES)。
