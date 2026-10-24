# TyloInfer documentation

TyloInfer provides TQ inference, CUDA/Metal runtimes, and local serving.

## Get started

- [Deploy a service and check performance (中文)](deployment.zh-CN.md)
- [Build the C++ runtime](cli/build.md)
- [Run MFQ Server and Studio](cli/serve.md)
- [Create a self-contained release](packaging.md)
- [Check model and backend support](runtime/support.md)

## Models and runtimes

- [TQ format and execution](formats/tq.md)
- [Runtime sampling profiles](runtime/sampling-profiles.md)
- [WT2 reference generation and utility commands](cli/utilities.md)
- [Runtime support matrix](runtime/support.md)
- [Python/C++ runtime transport](runtime/transport.md)
- [CUDA KV memory policies](runtime/cuda-kv-memory.md)
- [Flash-Next continuous batching](runtime/cuda-continuous-batching.md)
- [MiniCPM-o 4.5 multimodal runtime](runtime/minicpmo45.md)

## APIs

MFQ provides OpenAI-compatible `/v1/models` and `/v1/chat/completions`,
Anthropic-compatible `/v1/messages`, and an HTTP control API under `/api/v1`.
Token and job updates stream over SSE; audio and realtime sessions use
WebSocket. Function tools are executed by the client and their results returned
through the inference API.

- [HTTP API](api/http.md)
- [WebSocket and realtime API](api/websocket.md)
