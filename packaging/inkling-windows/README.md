# Inkling 2 Mini for Windows and RX 7900 XTX

[English](README.md) · [日本語](README.ja.md) · [简体中文](README.zh-Hans.md) · [繁體中文](README.zh-Hant.md)

**[v0.2.0](https://github.com/kanten2/llama.cpp-inkling-rdna3/releases/tag/v0.2.0)** · Windows x64 · Radeon RX 7900 XTX. Long-prompt WMMA Attention is included and defaults to off.

IQ4_NL, 16,384 input tokens and 1,000 generated tokens. Each configuration ran the same two prompts once each. Values are means of warm requests.

| Mode | Prefill (tok/s) | Decode (tok/s) | Request (s) |
|---|---:|---:|---:|
| v0.1.0 | 1,051.05 | 49.29 | 35.88 |
| v0.2.0 (Default) | 1,292.79 | 51.01 | 32.29 |
| v0.2.0 (WMMA) | 1,368.17 | 50.85 | 31.64 |

Default throughput improved 23.0% for prefill and 3.5% for decode. Request time fell 10.0%. Both default continuations matched v0.1.0 token for token. With WMMA, greedy answers can change. Both modes scored 72/88 on a separate Japanese reading test. [16K benchmark](https://github.com/kanten2/llama.cpp-inkling-rdna3/blob/v0.2.0/benchmarks/inkling-rocm/v0.2.0/speed.json) · [88 QA](https://github.com/kanten2/llama.cpp-inkling-rdna3/blob/v0.2.0/benchmarks/inkling-rocm/v0.2.0/reading.json)

To enable WMMA, run these commands in PowerShell from the extracted ZIP folder. A fresh shell uses the defaults.

```powershell
$env:INKLING_ATTN_DIRECT = '1'
$env:INKLING_ATTN_WMMA = '1'
.\chat.cmd "C:\models\Inkling-2-Mini-IQ4_NL.gguf"
```


## Use the ZIP

1. Download `inkling-rdna3-windows-gfx1100.zip` from the [v0.2.0 Windows release](https://github.com/kanten2/llama.cpp-inkling-rdna3/releases/tag/v0.2.0). In Explorer, choose **Extract All** and open the extracted folder.
2. Use your own text model: [ggml-org IQ4_NL](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF) or [Unsloth Mainline_Support UD-Q4_K_M](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/main/Mainline_Support). For the latter, rename the downloaded `.gguf_file` to `.gguf`. The model can stay in its own directory.
3. Drop the model onto **chat.cmd**. After loading, the launcher opens the chat in your default browser at the loopback address printed in the console. Leave the console open while chatting. Press Ctrl+C in it to stop the server and free VRAM. This is a text chat with a 4,096-token context.
4. To measure generation speed, close other GPU inference sessions and drop the same file onto **bench.cmd**. It prints the selected GPU and **tg768 in tok/s**, then waits for a keypress. Each invocation runs llama-bench's warm-up and one timed 768-token generation at depth zero. Model loading is outside the reported rate. For automation, set `INKLING_NO_PAUSE=1`.

The launchers select one RX 7900 XTX, use HIP Graphs and enable the optional four-wave IQ4_NL gate/up dispatch when the weights and shape match. The Unsloth model uses its own kernel path. This package includes the gfx1100 ROCm runtime and kernels. The host supplies Windows x64, a compatible AMD graphics driver and the [Microsoft Visual C++ x64 runtime](https://aka.ms/vs/17/release/vc_redist.x64.exe). A Python installation and ROCm development tools are only needed for building the package. The published ZIP checksum is on the release page. Its `SHA256SUMS.txt` covers the files inside it.

## Choose a language

Both launchers follow the Windows display language. To choose English explicitly, open PowerShell in the extracted folder and use the example below. Other accepted values are `ja`, `zh-Hans` and `zh-Hant`. Unrecognized languages use English. The setting applies to this terminal session. The bundled llama.cpp web interface and engine diagnostics retain their upstream wording. Write your question in the language you want to use.

```powershell
$env:INKLING_LANG = 'en'
.\chat.cmd ".\model.gguf"
```

The ZIP enables prefill fusion by default. In five-run tests with 16,384 input tokens, IQ4_NL took 13.53 s with fusion on and 18.54 s with it off. UD-Q4_K_M took 13.89 s and 18.54 s, respectively. The chat launcher uses a 4,096-token context. To compare modes, set `$env:INKLING_PP_FUSE = '0'` in PowerShell before launching the CMD, or set it to `'1'` to enable fusion again. [Prefill benchmarks and 16K reproduction steps](https://github.com/kanten2/llama.cpp-inkling-rdna3/tree/v0.1.0/benchmarks/inkling-rocm/prefill)

In the five-run tg768 benchmarks, IQ4_NL reached **54.53 tok/s** versus **46.55** with official b12606. Mainline_Support UD-Q4_K_M reached **56.67 tok/s** versus **48.63** with the loop-scale fix alone. Both optimized IQ4_NL settings answered **25/25 Japanese questions** correctly. [Measurements and answer checks](https://github.com/kanten2/llama.cpp-inkling-rdna3/tree/v0.1.0/benchmarks/inkling-rocm)

## Build and package from source


Open **Developer PowerShell for VS 2022** with the x64 C++ toolset, Windows SDK, CMake, Ninja and Git available. Use 64-bit Python 3 for packaging. Start at the root of a checkout of this repository at the release tag. Put build dependencies and outputs in a new directory. The commands below use [AMD's ROCm 10.0 wheel index](https://rocm.docs.amd.com/en/docs-10.0.0/install/rocm.html) and the [official b12606 Windows ROCm release](https://github.com/ggml-org/llama.cpp/releases/tag/b12606).

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

The build compiles `llama.dll` and `ggml-hip.dll` from this source with AMD Clang, target gfx1100, Release and HIP Graphs. The packager combines them with the b12606 server (including its web UI), benchmark executables and the ROCm 10.0 gfx1100 BLAS libraries and kernel data. It fetches two pinned upstream license texts while assembling the ZIP. Model files are supplied separately.
