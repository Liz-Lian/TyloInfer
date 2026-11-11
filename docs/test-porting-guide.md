# TyloInfer C++ 测试移植指南

仓库：`/home/zmx/mine/workspace/MFQ/TyloInfer`，源码树 `csrc/`，构建目录 `/tmp/mfq-try`（已配置好，Release + CUDA）。

## 构建与验证

```bash
cmake --build /tmp/mfq-try --target <test-target-name> -j     # 单个测试
ctest --test-dir /tmp/mfq-try -R <test-name> --output-on-failure   # 运行
```

测试 target 名在 `csrc/tests/CMakeLists.txt` 和 `csrc/engine/backends/cuda/CMakeLists.txt`（BUILD_TESTING 段）里查 `add_test(NAME ...)`。

## API 迁移映射（旧 → 新，已验证全仓库生效）

| 旧 | 新 |
|---|---|
| `mfq_tensor_backend::Tensor` | `tyloi::DenseTensor`（engine/tensor.h） |
| `mfq_tensor_backend::TensorOptions` | `tyloi::TensorOptions` |
| `mfq_tensor_backend::kFloat32 / kCUDA / kCPU 等` | `tyloi::kFloat32` 等（tyloi 层，不是 tyloi::cuda） |
| `mfq_tensor_backend::zeros/ones/randn/arange/cat/matmul/where/topk/softmax/...` | `tyloi::cuda::`（engine/backends/cuda/native/tensor/tensor.h） |
| `mfq_tensor_backend::NoGradGuard / manual_seed / from_blob` | `tyloi::cuda::` |
| `mfq::cuda::X` | `tyloi::cuda::X` |
| `mfq::ModelGraph / ModelSource / TensorMetadata` | `tyloi::` 同名 |
| `mfq_cuda_synchronize()` | `cudaStreamSynchronize(::tyloi::cuda::current_stream())` |
| `namespace tb = mfq_tensor_backend` | `namespace tb = tyloi::cuda` + 需要 tyloi 层符号时用全限定名 |

测试文件顶部若用了裸符号（`Tensor`、`kFloat32`），用这段开头：
```cpp
using namespace tyloi;
using namespace tyloi::cuda;
using Tensor = ::tyloi::DenseTensor;
```

## 头文件迁移映射

| 旧 include | 新 |
|---|---|
| `tyloinfer/runtime.h` | `engine/backends/cuda/native/context.h`（按实际用到的类选 native/*.h） |
| `tyloinfer/model_source.h` | `engine/models/source.h` |
| `execution/execution.h`、`execution/options.h` | `engine/backends/cuda/executor/execution.h`、`.../options.h` |
| `storage/session/state.h` | `engine/models/common/session/state.h` |
| `storage/session/text_cache.h` | **已删除**（TextCache 不存在了） |
| `storage/weights/loaders/weight.h` | `engine/weights/loader.h` |
| `storage/weights/moe/dma_copy_batch.h` | `engine/backends/cuda/native/copy_batch.h` |
| `storage/weights/moe/native_dma_batch.h` | `engine/backends/cuda/native/dma_batch.h` |
| `mfq_cuda_kernels.h` 等裸内核名 | `engine/backends/cuda/kernels/<同名>` |
| `mx.h` | `engine/weights/mx.h` |
| `models/common/axial_patch_vision_ops.h` | `engine/models/common/vision/axial_patch_vision.h` |
| `cuda_model_plan.h` | `engine/models/common/capabilities.h` |
| `runtime_config.h` / `cuda_runtime_config.h` | `engine/load.h` |
| `env.h`（kernels 头文件内） | `engine/backends/cuda/native/env.h` |

## Buffer / 内存类
`tyloi::cuda::Buffer`、`HostBuffer` 在 `engine/backends/cuda/native/memory.h`。

## 决策规则（对每个坏测试）

1. 被测的模块/函数在 `engine/` 下还存在 → 最小改动移植（改 include、改类型名、按新签名调整调用）。不要重写测试逻辑。
2. 被测机制已彻底删除（如 TextCache、QuantLinearKind）→ 删除测试文件 + CMakeLists 里对应的 add_executable/add_test 块，并在汇报里说明删了什么、为什么。
3. 拿不准 → 保持文件不动，汇报疑问，不要瞎改。

## 已隔离（不要动）
- CLI 工具（mfq-runtime / mfq-diagnostics / mfq-eval / mfq-cuda-eval）挂在 `TYLOINFER_BUILD_CLI_TOOLS`（默认 OFF）下，内部还是旧 API，等专项移植。
- bench 可执行目标挂在 `TYLOINFER_BUILD_BENCHES`（默认 OFF）下。
- hip/metal 后端不动。
