# Windows・RX 7900 XTX で使う Inkling 2 Mini

[English](README.md) · [日本語](README.ja.md) · [简体中文](README.zh-Hans.md) · [繁體中文](README.zh-Hant.md)

**v0.1.0** は、Windows x64 と Radeon RX 7900 XTX 1枚の環境向けの正式版です。プレビュー3で検証した推論バイナリを収録しています。実機で確認したモデルは IQ4_NL と Unsloth Mainline_Support UD-Q4_K_M です。

## ZIPでチャットと速度測定を始める

1. [Windows 正式版 v0.1.0](https://github.com/kanten2/llama.cpp-inkling-rdna3/releases/tag/v0.1.0)の `inkling-rdna3-windows-gfx1100.zip` をダウンロードし、エクスプローラーで「すべて展開」を選びます。
2. [ggml-org IQ4_NL](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF) または [Unsloth Mainline_Support UD-Q4_K_M](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/main/Mainline_Support) のテキストモデルを用意します。Unsloth版は `Mainline_Support` 内のファイルを選び、拡張子を `.gguf_file` から `.gguf` に変更してください。モデルは元のフォルダに置いたまま使えます。
3. GGUFを **chat.cmd** にドラッグすると、読み込み後に既定のブラウザーでローカルチャットが開きます。コンテキスト長は4,096トークンです。使用中はコンソールを開いたままにし、終了するときはそこで Ctrl+C を押します。
4. 速度を測るときはチャットなどGPUを使う推論を終了し、同じGGUFを **bench.cmd** にドラッグします。GPU名と **tg768の生成速度（tok/s）** が表示されます。ウォームアップ後、コンテキスト深度0から768トークンを1回計測します。モデルの読み込み時間は計測外です。自動実行では `INKLING_NO_PAUSE=1` でキー入力待ちを省けます。

Windows x64、対応するAMDドライバー、[Visual C++ x64 ランタイム](https://aka.ms/vs/17/release/vc_redist.x64.exe)が必要です。ROCmランタイムとgfx1100用カーネルはZIPに同梱しています。ZIPのSHA-256はリリースページに、各ファイルの値は `SHA256SUMS.txt` にあります。

## 表示言語を選ぶ

起動メッセージにはWindowsの表示言語を使います。切り替えるには、展開先のPowerShellで次のように実行します。モデルのパスは手元のものに置き換えてください。`en`、`ja`、`zh-Hans`、`zh-Hant` を指定でき、それ以外の言語は英語表示になります。設定はそのターミナルだけに適用されます。同梱のllama.cppのWeb画面とエンジンの診断ログは上流版の表記です。質問は希望する言語で入力できます。

```powershell
$env:INKLING_LANG = 'ja'
.\chat.cmd ".\model.gguf"
```

## 計測結果

配布ZIPではプロンプトの読み込み（プリフィル）の高速化が標準で有効です。16,384トークンを各5回測ると、IQ4_NLはオフ18.54秒・オン13.53秒、UD-Q4_K_Mはオフ18.54秒・オン13.89秒でした。通常のチャットは4,096トークンのコンテキストで起動します。比較する場合はPowerShellで `$env:INKLING_PP_FUSE = '0'` を設定してからCMDを起動し、有効に戻すには `'1'` を指定してください。 [プリフィルの測定結果と16Kでの再現手順](https://github.com/kanten2/llama.cpp-inkling-rdna3/tree/v0.1.0/benchmarks/inkling-rocm/prefill)

RX 7900 XTXでのtg768（各5回の平均）は、IQ4_NLが公式b12606の **46.55 → 最適化版54.53 tok/s**、Mainline_Support UD-Q4_K_Mがループスケール修正のみの **48.63 → 最適化版56.67 tok/s** でした。IQ4_NLの最適化2設定はいずれも日本語25問に全問正解しました。[速度測定と正答検証](https://github.com/kanten2/llama.cpp-inkling-rdna3/tree/v0.1.0/benchmarks/inkling-rocm)

## ソースからビルドする

VS 2022のDeveloper PowerShellを開き、x64 C++ツールセット、Windows SDK、CMake、Ninja、Git、64ビット版Python 3を用意します。このリリースタグのソースのルートで以下を実行します。[AMD ROCm 10.0の導入手順](https://rocm.docs.amd.com/en/docs-10.0.0/install/rocm.html)

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

`build.ps1` はAMD Clangで `llama.dll` と `ggml-hip.dll` をReleaseビルドします。対象はgfx1100、HIP Graphは有効です。`make_zip.py` は公式b12606のWeb UI内蔵サーバー、ベンチマーク、ROCm 10.0のBLASライブラリとカーネルをまとめ、ライセンス文書を取得します。生成したZIPにモデルは含まれません。
