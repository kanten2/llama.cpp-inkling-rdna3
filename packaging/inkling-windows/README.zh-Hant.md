# 在 Windows 與 RX 7900 XTX 上執行 Inkling 2 Mini

[English](README.md) · [日本語](README.ja.md) · [简体中文](README.zh-Hans.md) · [繁體中文](README.zh-Hant.md)

## 用 ZIP 啟動聊天與測速

1. 從 [Windows 預覽版](https://github.com/kanten2/llama.cpp-inkling-rdna3/releases/tag/v0.1.0-preview.3)下載 `inkling-rdna3-windows-gfx1100.zip`，在檔案總管中選取「全部解壓縮」，然後開啟解壓縮後的資料夾。
2. 準備 [ggml-org IQ4_NL](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF) 或 [Unsloth Mainline_Support UD-Q4_K_M](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/main/Mainline_Support) 的文字模型。使用 Unsloth 版時，請選擇 `Mainline_Support` 資料夾內的檔案，並將副檔名從 `.gguf_file` 改為 `.gguf`。模型可留在原來的位置。
3. 將 GGUF 拖曳到 **chat.cmd** 上。載入完成後，預設瀏覽器會開啟本機聊天頁面，上下文長度為 4,096 個 token。聊天時請保持主控台視窗開啟。結束時，在該視窗按 Ctrl+C 停止伺服器。
4. 測速前，請先關閉占用顯示記憶體的其他推論工作，再將同一個 GGUF 拖曳到 **bench.cmd** 上。視窗會顯示顯示卡名稱及 **tg768 生成速度（tok/s）**。暖機後，從上下文深度 0 開始，計時生成 768 個 token，共測一次。結果不含模型載入時間。自動執行時可設定 `INKLING_NO_PAUSE=1` 跳過按鍵等待。

需要 Windows x64、相容的 AMD 顯示卡驅動程式及 [Visual C++ x64 執行階段](https://aka.ms/vs/17/release/vc_redist.x64.exe)。ZIP 附有 ROCm 執行階段與 gfx1100 核心程式。發行頁面提供 ZIP 的 SHA-256，壓縮檔中的 `SHA256SUMS.txt` 列出各檔案的校驗值。

## 選擇顯示語言

啟動程式預設使用 Windows 的顯示語言。若要切換，請在解壓縮後的資料夾開啟 PowerShell，依下例執行，並換成自己的模型路徑。可選 `en`、`ja`、`zh-Hans` 與 `zh-Hant`，其他語言預設顯示英文。此設定只在目前的終端機工作階段生效。隨附的 llama.cpp 網頁介面及引擎診斷記錄保留上游版本的文字。可以直接用繁體中文提問。

```powershell
$env:INKLING_LANG = 'zh-Hant'
.\chat.cmd ".\model.gguf"
```

## 實測結果

壓縮檔預設啟用預填充運算融合。以 16,384 個輸入 token 各測量五次，IQ4_NL 的平均處理時間從 18.54 秒縮短至 13.53 秒，UD-Q4_K_M 從 18.54 秒縮短至 13.89 秒。聊天啟動器的上下文長度為 4,096 個 token。要比較兩種模式，請先在 PowerShell 中設定 `$env:INKLING_PP_FUSE = '0'`，再執行 CMD。設為 `'1'` 即可重新啟用。 [預填充測速結果與 16K 重現步驟](https://github.com/kanten2/llama.cpp-inkling-rdna3/tree/v0.1.0-preview.3/benchmarks/inkling-rocm/prefill)

在 RX 7900 XTX 上，tg768 的五次測量取平均後，IQ4_NL 由官方 b12606 的 **46.55 tok/s** 提升至最佳化版本的 **54.53 tok/s**。Mainline_Support UD-Q4_K_M 由僅修正迴圈層縮放的 **48.63 tok/s** 提升至 **56.67 tok/s**。兩種 IQ4_NL 最佳化設定都答對了全部 25 道日文題。[測速資料與答案驗證](https://github.com/kanten2/llama.cpp-inkling-rdna3/tree/v0.1.0-preview.3/benchmarks/inkling-rocm)

## 從原始碼建置

開啟 VS 2022 的 Developer PowerShell，備妥 x64 C++ 工具組、Windows SDK、CMake、Ninja、Git 與 64 位元 Python 3。在本次發行標籤所對應的原始碼根目錄執行下列命令。[AMD ROCm 10.0 安裝指南](https://rocm.docs.amd.com/en/docs-10.0.0/install/rocm.html)

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

`build.ps1` 使用 AMD Clang 為 gfx1100 編譯 Release 版 `llama.dll` 與 `ggml-hip.dll`，並啟用 HIP Graph。`make_zip.py` 將兩個 DLL 與官方 b12606 伺服器（內建網頁介面）、測速程式、ROCm 10.0 的 BLAS 函式庫及核心程式一起打包，並取得上游授權條款。模型需另行準備。
