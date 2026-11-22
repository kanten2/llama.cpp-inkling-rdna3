# 在 Windows ROCm 上執行 Inkling 2 Mini

[English](README.md) · [日本語](README.ja.md) · [简体中文](README.zh-Hans.md) · [繁體中文](README.zh-Hant.md)

**v0.1.0** 是本專案的首個正式版，適用於 **Windows x64 與單張 Radeon RX 7900 XTX**。處理 16,384 個輸入 token 時，IQ4_NL 的耗時從 **18.54 秒縮短至 13.53 秒**，Unsloth UD-Q4_K_M 從 **18.54 秒縮短至 13.89 秒**。ZIP 啟動器預設啟用這項最佳化。[下載 Windows 正式版](https://github.com/kanten2/llama.cpp-inkling-rdna3/releases/tag/v0.1.0) · [ZIP 使用與建置指南](packaging/inkling-windows/README.zh-Hant.md)

## 最佳化內容

預填充階段會依照注意力運算所需的記憶體排列方式，直接寫入相對位置偏差值，省去一次中間複製。逐一生成 token 時，程式會合併裁切與卷積等 GPU 運算，並重複使用多層共用的中間結果。IQ4_NL 的專家 gate/up 投影還會使用四個 wave 的 GPU 核心函式。[最佳化說明](benchmarks/inkling-rocm/README.md)

## 實測結果

### 提示詞處理（預填充）

針對每種 GGUF，在同一建置版本中分別開啟與關閉預填充融合，每種設定量測五次。輸入為 16,384 個 token 時，**IQ4_NL 吞吐量提升 36.9%，UD-Q4_K_M 提升 33.5%**。輸入為 4,096 個 token 時，分別提升 11.6% 與 10.7%。[原始數據、答案驗證與重現步驟](benchmarks/inkling-rocm/prefill/README.md)

![輸入吞吐量（tok/s）：IQ4_NL 884.48 → 1,210.57，UD-Q4_K_M 883.90 → 1,179.72。](benchmarks/inkling-rocm/charts/prefill.png)

### 回答生成（解碼）

下表為 IQ4_NL 啟用 HIP Graphs 後五次量測的平均值，單位為 tok/s。pp512 的數據取自加入預填充最佳化之前。

![生成速度（tok/s）：b12606 46.55，8 waves 53.34，4 waves 54.53。](benchmarks/inkling-rocm/charts/decode.png)

| 測試項目 | 官方 b12606 | 最佳化版（8 waves） | 最佳化版（4 waves） |
|---|---:|---:|---:|
| 處理 512 token (pp512) | 1,557.49 | 1,557.21 | 1,556.51 |
| 生成 128 token (tg128) | 46.00 | 52.94 | 54.16 |
| 生成 768 token (tg768) | 46.55 | 53.34 | **54.53** |

兩種模式均答對全部 25 道日文題。八個 wave 的模式共耗時 430.55 秒，四個 wave 的模式共耗時 365.76 秒。後者生成的 token 減少 13.3%，也縮短了作答時間。[測試條件與標準差](benchmarks/inkling-rocm/llama-bench-b12606/README.md) · [題目、回答與評分方式](benchmarks/inkling-rocm/quality/README.md)

使用 Unsloth Mainline_Support UD-Q4_K_M 時，tg768 從僅修正循環層縮放（#32704）的 48.63 tok/s 提升至最佳化後的 56.67 tok/s（+16.54%）。這款 GGUF 的 gate/up 權重格式不同，紀錄顯示 IQ4_NL 專用的四 wave 核心函式啟動次數為零。三種設定均答對全部五道題。[UD-Q4_K_M 測試與執行紀錄](benchmarks/inkling-rocm/ud-q4km-pr32704/README.md)

## 使用 Windows ZIP

1. 從 [v0.1.0 正式版頁面](https://github.com/kanten2/llama.cpp-inkling-rdna3/releases/tag/v0.1.0) 下載 `inkling-rdna3-windows-gfx1100.zip`，在檔案總管選取「全部解壓縮」。發行頁面提供 ZIP 的 SHA-256 校驗值。壓縮檔已包含 ROCm 執行階段，系統需要相容的 AMD 驅動程式與 [Visual C++ x64 執行階段](https://aka.ms/vs/17/release/vc_redist.x64.exe)。
2. 另行下載 [IQ4_NL](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF) 或 [Unsloth Mainline_Support UD-Q4_K_M](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/main/Mainline_Support) 的 GGUF。使用 Unsloth 版時，請將副檔名從 `.gguf_file` 改為 `.gguf`。
3. 將 GGUF 拖曳到 **chat.cmd**，瀏覽器會開啟上下文長度為 4,096 個 token 的本機文字聊天介面。聊天期間請保持主控台開啟。
4. 測速前在聊天主控台按 **Ctrl+C** 停止服務，再將同一 GGUF 拖曳到 **bench.cmd**。畫面會顯示顯示卡名稱，以及暖機後生成 768 個 token 的速度。

## 上游 llama.cpp 文件

本分支以 llama.cpp b12606 為基礎。上游專案提供工具文件、貢獻指南與第三方函式庫說明。[上游文件](https://github.com/ggml-org/llama.cpp/blob/b12606/README.md#documentation) · [貢獻指南](https://github.com/ggml-org/llama.cpp/blob/b12606/CONTRIBUTING.md) · [致謝](https://github.com/ggml-org/llama.cpp/blob/b12606/README.md#acknowledgements) · [授權條款](LICENSE)
