# 原生服务发行打包

Studio 发行版只启动 `tyloinfer-server`。固定布局由两份
`TyloInferStudio/src-tauri/tauri.release-*.conf.json` 的 `bundle.resources` 映射：

```text
packaging/Resources/tyloinfer-server/  -> resource_dir/tyloinfer-server/
  tyloinfer-server.exe                 # Windows
  tyloinfer-server                     # macOS
  *.dll                                # Windows 全部非系统运行库，同目录
packaging/Resources/licenses/        -> resource_dir/licenses/
TyloInferStudio/dist/                 -> resource_dir/web-ui/
```

产品不打包、发现或启动 Python 服务、`tyloinfer-cli`、旧推理 sidecar 或
`agent-runtime`。构建机上的 Python 3.11+ 仅执行资源清单、路径扫描、ZIP 与校验和工具；
不需要为 Windows 发行版执行 `uv sync`、安装 PyInstaller 或重建开发虚拟环境。
源码树中保留的 Python CLI、PyInstaller hooks 和相关旧测试属于独立开发工具，
不表示它们仍是 Studio 产品的运行依赖。

## Windows

需要 PowerShell 7、VS2022 C++ x64 工具、CUDA Toolkit、CMake 3.26+、Ninja、
Rust、Node.js/npm，以及 SQLite3、cURL 7.85+、OpenSSL Crypto 的 C/C++ 开发库。
使用仓库 `package-lock.json` 和 `npm ci`，不变更锁文件或添加生产依赖。
脚本默认将 npm 缓存放在 `packaging/build/npm-cache`，避开不可写的全局缓存；
显式设置 `npm_config_cache` 时沿用该目录。

脚本选用 VS2022，并自动加入其随附的 Ninja 路径。CMake 使用 **Ninja** 和显式
`nvcc.exe` 路径；不依赖 Visual Studio generator 的 CUDA MSBuild integration。
已有 CUDA Toolkit 可复用，无需因 VS generator 报 `No CUDA toolset found` 再装一套 CUDA。
NSIS 编译器由 Tauri 下载到本地工具目录，也可提供 PATH 中的 `makensis`。
NVIDIA 驱动及其 `nvcuda.dll` 由用户系统提供，不能从构建机复制进产品。

在仓库根目录执行，以下路径对应当前 Windows 构建机已有工具：

```powershell
.\packaging\build_release_windows.ps1 `
  -BuildPython 'D:\Develop\Anaconda3\python.exe' `
  -NativePrefix 'D:\Develop\Anaconda3\Library' `
  -CudaArchitectures 89 -Jobs 2
```

`-NativePrefix` 传给 `CMAKE_PREFIX_PATH`，并将各 prefix 的 `bin` 作为 DLL 搜索目录。
Anaconda 在此只提供开发库与构建期 Python，不会作为 Python 环境打入产品。
不同来源的动态库可通过 `-RuntimeDirectories @('D:/deps/bin')` 或
`TYLOINFER_RELEASE_RUNTIME_DIRS`（分号分隔）补充。已有 `CMAKE_PREFIX_PATH` 也会由
CMake 使用，但对应的 DLL 目录仍须可解析。默认 CUDA 架构为 `80-real;86-real;89`；
发行给其他 GPU 前需要选择适合目标设备的架构并进行实机验收。

脚本执行顺序：

1. 在 `packaging/build/native-server/<唯一ID>` 新建缓存，显式开启
   `TYLOINFER_BUILD_SERVER=ON`、`TYLOINFER_BUILD_CUDA_RUNTIME=ON`，关闭 HIP、Metal、
   Torch reference 与测试目标，只构建 `tyloinfer-server`。
2. 在独立 stage 收集服务、CUDA/cuBLAS、MSVC/OpenMP，以及 SQLite/cURL/OpenSSL 等
   PE 导入的传递 DLL。MSVC runtime 使用 VS 官方 redist；其他同名库冲突或缺失依赖直接失败。
   再仅从暂存目录和系统目录验证依赖闭包，并用收紧的 PATH 执行 `--version`、`--help`。
3. 扫描私有构建路径；验证后将原服务目录保留为
   `Resources/tyloinfer-server.previous-<唯一ID>`，再发布新的固定资源目录。
   旧 Python、sidecars 等未映射目录不会被收入发行包。
4. 执行 Studio 的 `npm ci`、`npm run check`、Tauri `build --no-bundle`；
   检查实际资源清单和私有路径后，才执行 Tauri NSIS bundle。
5. 构建便携单文件 EXE 与仅包含该 EXE 的 ZIP，生成 `.sha256`。
   默认输出为 `packaging/dist/`；`-OutputDirectory` 可指定另一目录（相对 packaging 解析）。

启动器中的开发服务和 Web UI 源码目录探测仅编译进 Debug，避免
`CARGO_MANIFEST_DIR` 的字符串值残留在 Release EXE 中。

仅重打已有便携产物：

```powershell
.\packaging\build_release_windows.ps1 `
  -BuildPython 'D:\Develop\Anaconda3\python.exe' -PortableOnly
```

这要求已有匹配的 Release Studio EXE、原生服务及资源。它不会编译服务，也不表示
这些资源已经通过真实模型验收。`-Clean` 会明确拒绝；脚本不递归删除任何目录。
旧构建缓存、旧资源、旧安装包由使用者审查后手动清理。验证或编译失败时，已有
便携 EXE/ZIP 不会被半成品替换。

单文件启动器继续使用按版本和载荷 hash 隔离的缓存、互斥提取、`.ready` 标记和
`/WAITPID=` 更新等待；每次复用前检查所有资源文件，缺少 DLL 也会重新提取。
`TYLOINFER_STUDIO_RUNTIME_CACHE`、`/CACHE_DIR=`、`/EXTRACT_ONLY` 与
`TYLOINFER_STUDIO_LAUNCHER` 行为保留。

## macOS 状态

**当前禁止生成推理发行 DMG。** `cpp_runtime/apps/server_main.cpp` 尚未将 Metal
引擎接入 `tyloinfer-server`；成功编译一个只有管理 API 的服务不能作为推理发行验收。
`./packaging/build_release_mac.sh` 默认立即报出该阻塞，不触发 Tauri 或输出旧 DMG。

Apple Silicon 上可显式准备诊断资源：

```shell
TYLOINFER_RELEASE_VENV=/path/to/existing/build-env ./packaging/build_release_mac.sh --prepare-only
```

此环境必须已经具有 MLX（仅作构建期 SDK 来源）；还需 Xcode Metal Toolchain、
CMake、Ninja、SQLite3/cURL/OpenSSL 开发库。脚本不清空或重建环境；CMake 可通过
`CMAKE_PREFIX_PATH` 定位开发库。macOS 最低版本沿用 26.2。

准备流程构建 `tyloinfer-server`，将其放到 `Resources/tyloinfer-server/tyloinfer-server`；
保留 `mlx.metallib`、预编译 `mfq-metal`，以及 `Frameworks` 中的 `libmlx`、`libjaccl`、
AVFoundation bridge。`mfq-decode-metal` 只作为构建这些资源的辅助目标，不再作为产品
externalBin；服务的 Frameworks rpath 按新位置设置为 `@executable_path/../../Frameworks`。
已存在的固定资源目录会被保护并提示手动处理。

`--prepare-only` 不生成 DMG，也不宣称完整 dylib 闭包、Metal 推理、签名和安装更新
已验收。恢复 macOS 正式发行前，必须接入服务 Metal 工厂、补齐第三方 dylib 归集，
在真实 Apple Silicon 上验证模型加载/生成/取消/卸载、应用签名和 DMG 挂载运行。

## 检查与验收边界

基础检查（使用已有 pytest 环境）：

```powershell
& 'D:\Develop\Anaconda3\python.exe' -m pytest `
  tests/test_windows_portable.py tests/test_windows_release_environment.py tests/test_release_paths.py -q
```

设置 `TYLOINFER_TEST_NSIS` 为 `makensis.exe` 路径，可运行真实便携提取/缓存修复测试。
将 `tests/fixtures/windows_standalone_app.cpp` 用 MSVC `/MT /SUBSYSTEM:WINDOWS` 编译，
再设置 `TYLOINFER_TEST_STANDALONE_APP` 可验证启动及更新父进程等待。
这些明确使用夹具的测试只证明启动器行为，不证明 Studio 或 GPU 能工作。

设置 `TYLOINFER_TEST_NATIVE_SERVER` 为真实已编译的服务、
`TYLOINFER_TEST_NATIVE_DLL_DIRS` 为分号分隔的 DLL 来源目录，可运行真实 DLL 归集测试：
先确认缺依赖时拒绝，再收集闭包，使用仅含服务目录和系统目录的 PATH 启动服务，
请求健康/身份/模型 API，并关闭 stdin 验证退出。CPU 管理服务通过此测试仍不等于
GPU 推理通过。

发行前仍需在目标机器上验收安装/卸载、WebView2、Studio 启动与原生服务发现、
真实模型加载/聊天/取消/卸载、便携更新与端口切换。脚本 `--help` 冒烟和资源存在性
检查不能代替这些验证。
