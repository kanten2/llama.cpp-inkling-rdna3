# 在 Windows 和 RX 7900 XTX 上运行 Inkling 2 Mini

[English](README.md) · [日本語](README.ja.md) · [简体中文](README.zh-Hans.md) · [繁體中文](README.zh-Hant.md)

## 用 ZIP 启动聊天和测速

1. 从 [Windows 预览版](https://github.com/kanten2/llama.cpp-inkling-rdna3/releases/tag/v0.1.0-preview.3)下载 `inkling-rdna3-windows-gfx1100.zip`，在文件资源管理器中选择“全部解压缩”，然后打开解压后的文件夹。
2. 准备 [ggml-org IQ4_NL](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF) 或 [Unsloth Mainline_Support UD-Q4_K_M](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/main/Mainline_Support) 的文本模型。使用 Unsloth 版时，请选择 `Mainline_Support` 文件夹中的文件，并将扩展名从 `.gguf_file` 改为 `.gguf`。模型可以放在原来的位置。
3. 将 GGUF 拖到 **chat.cmd** 上。加载完成后，默认浏览器会打开本地聊天页面，上下文长度为 4,096 个 token。聊天期间请保持控制台窗口打开。结束时，在该窗口按 Ctrl+C 停止服务器。
4. 测速前，请先关闭占用显存的其他推理任务，再将同一个 GGUF 拖到 **bench.cmd** 上。窗口会显示显卡名称和 **tg768 生成速度（tok/s）**。预热后，从上下文深度 0 开始，计时生成 768 个 token，共测一次。结果不含模型加载时间。自动运行时可设置 `INKLING_NO_PAUSE=1` 跳过按键等待。

需要 Windows x64、兼容的 AMD 显卡驱动程序和 [Visual C++ x64 运行库](https://aka.ms/vs/17/release/vc_redist.x64.exe)。ZIP 自带 ROCm 运行库和 gfx1100 内核。发布页面提供 ZIP 的 SHA-256，压缩包中的 `SHA256SUMS.txt` 列出了各文件的校验值。

## 选择显示语言

启动器默认使用 Windows 的显示语言。要切换语言，请在解压目录打开 PowerShell，按下例运行，并替换模型路径。可选值为 `en`、`ja`、`zh-Hans` 和 `zh-Hant`，其他语言默认显示英文。此设置仅在当前终端会话中生效。随附的 llama.cpp 网页界面和引擎诊断日志保留上游版本的文字。可以直接用简体中文提问。

```powershell
$env:INKLING_LANG = 'zh-Hans'
.\chat.cmd ".\model.gguf"
```

## 实测结果

压缩包默认启用预填充算子融合。对 16,384 个输入 token 各测量五次，IQ4_NL 的平均处理时间从 18.54 秒降至 13.53 秒，UD-Q4_K_M 从 18.54 秒降至 13.89 秒。聊天启动器的上下文长度为 4,096 个 token。要对比两种模式，请先在 PowerShell 中设置 `$env:INKLING_PP_FUSE = '0'`，再运行 CMD。设为 `'1'` 即可重新启用。 [预填充测速结果与 16K 复现步骤](https://github.com/kanten2/llama.cpp-inkling-rdna3/tree/v0.1.0-preview.3/benchmarks/inkling-rocm/prefill)

在 RX 7900 XTX 上，tg768 的五次测量均值如下。IQ4_NL 从官方 b12606 的 **46.55 tok/s** 提高到优化版的 **54.53 tok/s**。Mainline_Support UD-Q4_K_M 从仅修复循环层缩放的 **48.63 tok/s** 提高到 **56.67 tok/s**。两种 IQ4_NL 优化设置都答对了全部 25 道日语题。[测速数据与答案验证](https://github.com/kanten2/llama.cpp-inkling-rdna3/tree/v0.1.0-preview.3/benchmarks/inkling-rocm)

## 从源码构建

打开 VS 2022 的 Developer PowerShell，准备 x64 C++ 工具集、Windows SDK、CMake、Ninja、Git 和 64 位 Python 3。在本次发布标签对应的源码根目录运行以下命令。[AMD ROCm 10.0 安装指南](https://rocm.docs.amd.com/en/docs-10.0.0/install/rocm.html)

```powershell
$work = (New-Item -ItemType Directory ../inkling-windows-build).FullName
$sdk = Join-Path $work 'sdk'
python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --target $sdk "rocm[libraries,devel,device-gfx1100]==10.0.0"
$dev = Join-Path $sdk '_rocm_sdk_devel'
$official = Join-Path $work 'official'
$upstreamZip = Join-Path $work 'upstream.zip'
Invoke-WebRequest 'https://github.com/ggml-org/llama.cpp/releases/download/b12606/llama-b12606-bin-win-rocm-10.0-x64.zip' -OutFile $upstreamZip
if ((Get-FileHash $upstreamZip).Hash -ne '59b1f689c5bd2a2c820ae6c2944602b6d265248091d94818d6f97b95a5e7b242') { throw 'Official b12606 ZIP checksum mismatch' }
Expand-Archive -LiteralPath $upstreamZip -DestinationPath $official
$build = Join-Path $work 'build'
.\packaging\inkling-windows\build.ps1 -DevelopmentSdk $dev -BuildDir $build
python packaging/inkling-windows/make_zip.py --official $official --build-bin "$build/bin" --sdk $sdk --development-sdk $dev --output "$work/inkling-rdna3-windows-gfx1100.zip"
Get-FileHash "$work/inkling-rdna3-windows-gfx1100.zip" -Algorithm SHA256
```

`build.ps1` 使用 AMD Clang 为 gfx1100 编译 Release 版 `llama.dll` 和 `ggml-hip.dll`，并启用 HIP Graph。`make_zip.py` 将它们与官方 b12606 服务器（内置网页界面）、测速程序、ROCm 10.0 BLAS 库和内核打包，同时获取上游许可证文本。模型需单独准备。
