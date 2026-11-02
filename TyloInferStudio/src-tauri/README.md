# Studio 原生服务启动器

Studio 只启动 `tyloinfer-server` 原生可执行文件。未找到文件、缺少动态库、身份握手失败或原生进程启动失败时，直接报告错误与本次启动的服务日志。

## 文件布局

发行资源由两份 `tauri.release-*.conf.json` 将 `packaging/Resources/tyloinfer-server` 映射到 Tauri 的资源目录：

```text
resource_dir/
  tyloinfer-server/
    tyloinfer-server.exe       # Windows
    tyloinfer-server           # macOS / Unix
    *.dll                     # Windows 原生服务所需 DLL
  web-ui/
  licenses/
```

Windows 子进程的 `PATH` 首项是服务可执行文件所在目录，原有搜索路径继续保留。macOS 继续打包 MLX metallib、`mfq-metal` 和现有 Frameworks 资源。旧的 Python CLI、推理 sidecar 和 `agent-runtime` 不再属于 Studio 的发行配置。

开发构建依次检查发行资源位置，以及仓库内 `build/native-server/server/`、`Release/`、`RelWithDebInfo/`、`Debug/`、`MinSizeRel/` 的原生服务。显式设置 `TYLOINFER_SERVER_EXECUTABLE` 可指定其他构建的绝对路径；覆盖值无效时直接报错。发行构建不探测开发目录。

启动参数与 `cpp_runtime/apps/server_main.cpp` 对齐：`--host`、`--port`、`--data-dir`、`--api-key-env`、`--exit-on-stdin-close`、`--studio-instance-id`，有 Web UI 时再传 `--web-root`。工作目录为 Studio 的应用数据目录。原生入口自行选择或迁移数据库，并读取已保存的模型目录及认证策略。根凭据可以从 `TYLOINFER_SERVER_API_KEY` 继承；Studio 中保存的连接凭据仍使用系统凭据库，不会放进命令行或配置文件。

## 生命周期与配置失败

- 每个子进程在启动后立即记录 PID、实例 UUID、端口、stdin 管道和进程句柄。健康检查、版本、协议与实例身份验证通过后才视为可连接。
- 外部服务必须由用户确认，确认绑定 URL、实例和工作区；重启或更换端点后需要重新确认。Studio 不接管或关闭外部服务。
- 原生 OpenAI / Anthropic API 共用一个监听端口，不再检查旧 `listener.json` 中的第二个端口。
- 修改自有服务端口时，先检查新端口与服务文件，再关闭旧进程 stdin、等待退出，并以新端口启动。新实例就绪后才原子替换 `studio.json`；新实例启动或配置保存失败时尝试恢复原端口。
- 启动失败、更新、端口变更与退出均通过关闭 stdin 请求关闭，只等待 Studio 自己创建的进程。等待上限为 30 秒；超时后仍保留进程句柄，禁止重复启动。进程真正退出后，可以重试启动或修改端口。
- 若新进程清理超时，原配置保持不变，等待它退出后重试。若旧端口恢复也失败，错误会同时包含恢复失败原因和日志位置。服务日志沿用大小轮转和本次启动日志截取。

## 验证与能力限制

执行 `cargo test --locked` 验证发现路径、参数契约、单端口检查、实例所有权、外部服务隔离、关闭超时后的句柄保留、重复启动保护，以及配置保存失败后的原文件保留。超时测试会单独启动一个测试子进程，因此 `delayed_exit_fixture` 在普通测试运行中标记为忽略。

设置 `TYLOINFER_TEST_SERVER_EXECUTABLE` 后，执行 `cargo test --locked native_server_lifecycle_smoke -- --ignored --nocapture`，可验证真实原生服务的启动、身份握手、认证请求、EOF 关闭和同一工作区换端口重启。该测试在 `target/native-server-smoke-*` 保留数据库及日志，便于检查。

原生服务当前没有 agent 后台消费者，因此本次不打包 `agent-runtime`，也不提供该后台执行能力。MLX / Frameworks 资源保留不代表原生服务已经接入 Metal 推理引擎；模型执行能力取决于原生服务实际编译的后端。完整安装包、更新包与前端交互需要与包装脚本一起联调，Windows 上的 Rust 单元测试不能替代 macOS 打包运行验证。
