# 原生服务迁移基线

本批只迁移 MFQ 已有的 C++ 服务及其直接依赖，供后续继续开发。
它不代表 Python 服务已被完整替换，也不要求本批完成 Studio 和全部后端适配。

## 来源与范围

- TyloInfer 基线：`origin/main` 的 `09b8fb1`。主分支名为 `main`。
- 迁移来源：MFQ `refactor/backend` 的 `532eaa76..94ba90ec`。
- 工作分支：`refactor/native-server`，独立 worktree；未合入 `refactor/arch`。

已迁入：

- `cpp_runtime/server/` 和 `apps/server_main.cpp`：原生 HTTP 服务、鉴权、
  SQLite、模型登记/别名/实例、文本聊天持久化、基础兼容 API、任务和 HF/MS 下载。
- `components/protocol/`：公共服务与现有 HTTP/stdio transport 共用解析和输出代码。
- 原服务依赖的 Engine 请求所有权、Scheduler 预算/取消/退出处理和 HF tokenizer 加载。
- CMake、原生测试源码、架构检查，以及版本 19 的共享 `schema.sql` 和 Python 读取方。
- TyloInfer 的命名、目录、TQ/GGUF 模型发现、分片登记和 GGUF 伴随表过滤。

TyloInfer 主分支现有的 TQ/GGUF/Gyro、模型实现和 token 计数代码继续保留。
源分支的 Studio 改动未纳入本基线；当前 Studio、Python CLI 和发布入口继续使用原有实现。

## 原生入口

设置 `TYLOINFER_BUILD_SERVER=ON` 可构建 `tyloinfer-server`。
默认仍为 OFF；它不依赖旧 HTTP/stdio 推理子进程。
以下配置仅构建无 GPU 的管理服务，不提供真实模型推理：

```sh
cmake -S cpp_runtime -B build/native-server \
  -DTYLOINFER_BUILD_SERVER=ON \
  -DTYLOINFER_BUILD_CUDA_RUNTIME=OFF \
  -DTYLOINFER_BUILD_HIP_RUNTIME=OFF \
  -DTYLOINFER_BUILD_METAL_RUNTIME=OFF \
  -DTYLOINFER_BUILD_RUNTIME_COMMUNICATION=OFF \
  -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/native-server --target tyloinfer-server -j
build/native-server/server/tyloinfer-server --data-dir .tyloinfer --port 8090
```

原生依赖包括 SQLite3、libcurl 7.85 及以上版本、OpenSSL Crypto。
实际推理需要启用对应 GPU runtime；Metal 的公开加载工厂仍待接入。
原生接口以 [OpenAPI](../cpp_runtime/server/api/openapi.json) 为准，
不能将 Python 服务的完整接口清单视为原生服务已实现的能力。

## 留给后续的适配

- Studio 的加载参数、实例响应、聊天/任务流，以及启动器和发行包切换。
- 完整 OpenAI/Anthropic 协议、媒体和实时功能、运行配置与缓存管理。
- 社区仓库浏览等 TyloInfer 后续功能、完整下载语义和产品工具。
- 加载中的原生进度/取消、统一加载配置、其他平台与真实模型验收。
- 移除 Python 运行依赖；当前 CMake 仍从 Python 文件读取版本和数据库版本元数据。

后续适配直接接在本基线上，不为维持旧 Python 管理接口增加兼容层。

## 验证状态

用户已要求停止所有测试。本基线只完成源码、引用、构建清单及差异的静态检查，
最终状态未作构建和运行验收。此前中间版本的构建/测试记录不代表本基线通过验收。
