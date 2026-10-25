# ブラウザによる動作確認

`browser-qa.mjs` は、frontendの画面操作をChromeとPlaywright Coreで確認する任意の検査です。通常のビルドにPlaywrightは不要です。AI応答の妥当性や研究の有効性は検証しません。

以下は `tools/discussion-chatbot/frontend/` で実行します。まず別のターミナルで画面を起動します。

```bash
npm ci
npm run dev -- --port 5173
```

Playwright Coreはアプリの依存関係に追加せず、一時ディレクトリへ導入します。ChromeまたはChromiumは、実行環境にあるものを指定します。

```bash
npm install --prefix /tmp/req005-browser-tools playwright-core
BROWSER_QA_MODULE=/tmp/req005-browser-tools/node_modules/playwright-core/index.mjs \
BROWSER_EXECUTABLE=/usr/bin/google-chrome \
node test/browser-qa.mjs
```

OSの制約でChromeのsandboxを利用できない検査環境では、上のコマンドに `QA_NO_SANDBOX=1` を付けます。通常の実行では指定しません。

既定ではAPIをブラウザ内で模擬し、backendを呼びません。話題作成、送信、Markdown表示、scriptの非実行、外部画像の自動取得防止、下書きの編集・プレビュー・保存、資料反映、保存中の操作制限、校正結果表示、モバイルの画面と一覧操作を検査します。保存操作も模擬され、リポジトリにMarkdownを書き込みません。

画像は `/tmp/req005-browser-qa/` に出力します。失敗時も可能な範囲で画像を残し、終了コードを非0にします。

| 環境変数 | 既定値・用途 |
| --- | --- |
| `QA_BASE_URL` | `http://127.0.0.1:5173`。検査するfrontend |
| `BROWSER_QA_MODULE` | 未指定なら通常の `playwright-core` 解決を使用。指定する場合はモジュールファイルの絶対パス |
| `BROWSER_EXECUTABLE` | `/usr/bin/google-chrome` |
| `QA_OUTPUT_DIR` | `/tmp/req005-browser-qa` |
| `QA_NO_SANDBOX` | `1` の場合だけChromeに `--no-sandbox` を指定 |
| `QA_INTEGRATION` | `1` の場合は、模擬API検査の後に実backendとの接続も確認 |
| `QA_TEMP_WORKSPACE_CONFIRMED` | 統合検査時は、専用の一時workspaceとdataを使用したことを確認して `1` を指定 |
| `QA_SOURCE_PATH` | 統合検査で追加する資料。既定は `reports/qa-reference.md` |

## 実backendとの統合確認

統合検査は、backendの `AI_PROVIDER=mock` と専用の一時workspace・dataを使用します。実際の研究リポジトリや会話DBを使わないでください。起動済みの通常backendが8787番ポートを使っている場合は、そちらを停止してから行います。

一時workspaceの例です。以下の変数は、このターミナルでbackendを終了するまで保持します。

```bash
research_qa_dir="$(mktemp -d /tmp/req005-integration.XXXXXX)"
mkdir -p "$research_qa_dir/workspace/requests"
mkdir -p "$research_qa_dir/workspace/reports"
mkdir -p "$research_qa_dir/workspace/.codex/research-docs/checkpoints"
cp ../../../AGENTS.md "$research_qa_dir/workspace/AGENTS.md"
cp ../../../.codex/research-docs/requirements-workflow.md "$research_qa_dir/workspace/.codex/research-docs/requirements-workflow.md"
printf '# REQ-005 画面検査用\n\n模擬応答による画面操作の確認専用。研究の採否を含まない。\n' > "$research_qa_dir/workspace/requests/qa-REQ-005.md"
printf '# 参照資料\n\n画面検査用の資料。\n' > "$research_qa_dir/workspace/reports/qa-reference.md"
RESEARCH_WORKSPACE="$research_qa_dir/workspace" \
DATA_DIR="$research_qa_dir/data" \
AI_PROVIDER=mock PORT=8787 \
npm --prefix ../backend run dev
```

frontendのViteは `/api` を `http://127.0.0.1:8787` へ転送します。別のターミナルで次を実行します。

```bash
BROWSER_QA_MODULE=/tmp/req005-browser-tools/node_modules/playwright-core/index.mjs \
QA_INTEGRATION=1 QA_TEMP_WORKSPACE_CONFIRMED=1 \
node test/browser-qa.mjs
```

統合検査は、担当の指名、複数担当による議論、資料追加、下書き生成、停止と再試行、再読込後の履歴、モバイル操作を確認します。backendが `mock` であることを検査してから進み、Markdownの保存ボタンは押しません。一時DBには検査用の会話・下書きが残ります。検査後はfrontendとbackendを、それぞれのターミナルで停止してください。
