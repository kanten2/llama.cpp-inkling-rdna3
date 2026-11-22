# v0.1.0-preview.3 · Inkling 2 Mini on RX 7900 XTX

## English

On one Radeon RX 7900 XTX with Windows ROCm, prefill fusion reduced the time to read 16,384 tokens from **18.54 to 13.53 s (IQ4_NL)** and **18.54 to 13.89 s (Unsloth Mainline_Support UD-Q4_K_M)**. The comparison uses the same build with fusion off and on, five timed runs per condition. [Benchmark data and answer checks](https://github.com/kanten2/llama.cpp-inkling-rdna3/tree/v0.1.0-preview.3/benchmarks/inkling-rocm/prefill)

Extract `inkling-rdna3-windows-gfx1100.zip` and drop your GGUF onto **chat.cmd**. The launcher enables prefill fusion and opens a local chat with a 4,096-token context. To measure 768-token generation, stop the chat and drop the GGUF onto **bench.cmd**. The release includes the ZIP's SHA-256. [Windows guide](https://github.com/kanten2/llama.cpp-inkling-rdna3/blob/v0.1.0-preview.3/packaging/inkling-windows/README.md)

## 日本語

Windows ROCm・Radeon RX 7900 XTX 1枚で、16,384トークンの入力処理が **IQ4_NLでは18.54秒から13.53秒**、**Unsloth Mainline_Support UD-Q4_K_Mでは18.54秒から13.89秒**に短縮しました。同一ビルドでプリフィル高速化をオン・オフにして各5回測った平均です。

ZIPを展開してGGUFを **chat.cmd** にドラッグすると、高速化が有効なローカルチャットが開きます。通常のチャットは4,096トークンのコンテキストで起動します。チャットを停止して **bench.cmd** にGGUFをドラッグすると、768トークンの生成速度を測れます。ZIPのSHA-256を添付しています。[日本語ガイド](https://github.com/kanten2/llama.cpp-inkling-rdna3/blob/v0.1.0-preview.3/packaging/inkling-windows/README.ja.md)

## 简体中文

在 Windows ROCm 和单张 Radeon RX 7900 XTX 上，启用预填充融合后，处理 16,384 个输入 token 的时间从 **18.54 秒降至 13.53 秒（IQ4_NL）**，从 **18.54 秒降至 13.89 秒（Unsloth Mainline_Support UD-Q4_K_M）**。两种模式使用同一构建版本，数据为各测量五次的平均值。

解压 ZIP，将 GGUF 拖到 **chat.cmd** 上，即可打开默认启用融合的本地聊天界面，上下文长度为 4,096 个 token。停止聊天后，将 GGUF 拖到 **bench.cmd** 上即可测量生成 768 个 token 的速度。发布附件中包含 ZIP 的 SHA-256 校验值。[简体中文指南](https://github.com/kanten2/llama.cpp-inkling-rdna3/blob/v0.1.0-preview.3/packaging/inkling-windows/README.zh-Hans.md)

## 繁體中文

在 Windows ROCm 與單張 Radeon RX 7900 XTX 上，啟用預填充融合後，處理 16,384 個輸入 token 的時間從 **18.54 秒縮短至 13.53 秒（IQ4_NL）**，從 **18.54 秒縮短至 13.89 秒（Unsloth Mainline_Support UD-Q4_K_M）**。兩種模式皆使用同一建置版本，數據為各量測五次的平均值。

解壓縮 ZIP，將 GGUF 拖曳到 **chat.cmd** 上，即可開啟預設啟用融合的本機聊天介面，上下文長度為 4,096 個 token。停止聊天後，將 GGUF 拖曳到 **bench.cmd** 上即可量測生成 768 個 token 的速度。發行附件內含 ZIP 的 SHA-256 校驗值。[繁體中文指南](https://github.com/kanten2/llama.cpp-inkling-rdna3/blob/v0.1.0-preview.3/packaging/inkling-windows/README.zh-Hant.md)
