---
name: sam-audio
description: Run local, offline SAM Audio sound-source separation through the audio-toolbox CLI (target/residual stems, dry-run plans, environment preflight, machine-readable JSON results). Use when a task must separate one described sound (for example "melodic sound" or "bowed strings") from a local audio file, check whether a machine is ready for SAM Audio inference (FFmpeg/shared DLLs/model files), or interpret target.wav/residual.wav/request.json/report.json and the CLI exit codes. 用于本地离线 SAM Audio 声源分离、dry-run 计划、环境预检与结构化结果/错误解释。
license: MIT
compatibility: Wrapper needs only Python 3.11+ standard library. Real separation additionally needs a pinned local SAM Audio checkout (torch/CUDA or CPU), FFmpeg 4-8 full-shared build on Windows, and locally installed model weights. No network access is required or performed.
metadata:
  version: "1.0.0"
  source-repo: "guajun/agentic-audio-toolbox"
  sam-upstream: "guajun/anonymous-audio-tracks@infrastructure/audio-analysis/sam-audio"
  sam-upstream-commit: "c603de8794cc16880dc01be0f1e868f6c2845417"
  upstream-license: "Meta SAM License (upstream SAM-LICENSE.txt; not redistributed here)"
---

# sam-audio — offline SAM Audio separation toolbox

This skill wraps the existing SAM Audio local inference entry
(`<sam-root>/scripts/run_inference.py` under the SAM checkout root) with a **stable CLI and stable
machine output**. It does not bundle SAM code, model weights, or audio, and it
never downloads anything.

## 何时使用 / When to use

Use this skill when the task involves **local audio source separation**:

* separate one text-described sound from a WAV/MP3/... file and obtain
  `target.wav` / `residual.wav` plus `request.json` / `report.json`;
* validate a machine before inference (FFmpeg + shared DLLs, model files) **without loading torch or using the GPU**;
* produce or interpret the machine-readable plan/result/error JSON of the CLI.

Do **not** use it to download models, to run large GPU batch evaluations
(reserved for the project's issue #33 workflow), or for any cloud/API analysis.

## CLI 参数 / Interface

The implementation is bundled here as `scripts/audio_toolbox.py` (standard
library only). Run it with any Python 3.11+ interpreter:

```sh
python scripts/audio_toolbox.py --help
python scripts/audio_toolbox.py --version
```

Fixed, stable invocation form (relative to this skill directory):

```sh
python scripts/audio_toolbox.py sam separate   --audio <FILE> --description <TEXT> [options]
python scripts/audio_toolbox.py sam check-environment [options]
```

`audio-toolbox sam separate` is the documented command name; repository
launchers `bin/audio-toolbox` (POSIX) and `bin/audio-toolbox.cmd` (Windows)
forward to `scripts/audio_toolbox.py`. After a skill-only install, call
`python scripts/audio_toolbox.py ...` directly — the skill directory is
self-contained.

### `sam separate`

| Flag | Meaning |
|---|---|
| `--audio FILE` | required; input audio file (spaces in paths are fine; always quote) |
| `--description TEXT` | required; lowercase noun/verb phrase, e.g. `melodic sound` |
| `--anchor START,END` | optional; positive time span in seconds, repeatable (`--anchor 11.18,11.50`) |
| `--model-dir DIR` | SAM checkpoint dir (default `<sam-root>/model-cache/sam-audio-small`) |
| `--text-encoder-dir DIR` | T5 dir (default `<sam-root>/model-cache/t5-base`) |
| `--device DEV` | `cuda`, `cpu`, ... (default: SAM checkout config) |
| `--dtype DTYPE` | `auto`, `bfloat16`, `float32` |
| `--output-dir DIR` / `--run-dir DIR` | where `target.wav`, `residual.wav`, `request.json`, `report.json` are written |
| `--sam-root DIR` | SAM Audio checkout root (or env `SAM_AUDIO_ROOT`) |
| `--python EXE` | interpreter that has the SAM environment (default: current one) |
| `--timeout SEC` | upstream timeout (default 3600); the process is terminated on expiry |
| `--dry-run` | validate audio/anchors/FFmpeg/paths and print the plan; **no model load, no GPU** |

### `sam check-environment`

Flags: `--sam-root`, `--model-dir`, `--text-encoder-dir`, `--python`.
Runs wrapper-level checks and then the SAM entry's own `--check-environment`
(FFmpeg + Windows shared-DLL discovery). No torch import, no GPU, no network.

### 稳定机器输出 / Machine output

Every call prints **exactly one JSON document to stdout** (schema
`audio-toolbox.sam/v1`) and a one-line human summary to stderr on failure.
Full field reference: [`references/cli-reference.md`](references/cli-reference.md).
Key fields: `ok`, `action`, `exit_code`, `command` (the exact argv list handed
to the SAM entry), `parameters`, `upstream` (with `stdout_tail`/`stderr_tail`),
`run_dir`, `outputs`, `report`/`plan`, and on failure
`error: {code, message, detail}`.

Exit codes are stable:

| Code | Meaning |
|---|---|
| 0 | success |
| 2 | usage error (bad flags, malformed `--anchor` text, empty description) |
| 3 | input error (audio missing, anchor values invalid: `START >= END`, negative, non-finite) |
| 4 | environment error (SAM checkout/entry/interpreter/model files missing) |
| 5 | upstream SAM entry failed |
| 6 | upstream timed out (process terminated) |
| 7 | output error (run dir or expected artifacts missing) |

## 典型调用 / Typical workflow

1. **Diagnose first** (no GPU):

   ```sh
   python scripts/audio_toolbox.py sam check-environment \
     --sam-root "D:/path with spaces/sam-audio" \
     --python "D:/path with spaces/sam-audio/.venv/Scripts/python.exe"
   ```

2. **Dry-run** before any real inference:

   ```sh
   python scripts/audio_toolbox.py sam separate \
     --audio "D:/data/breeze/Breeze-first30s.wav" \
     --description "melodic sound" \
     --anchor 11.18,11.50 \
     --dry-run \
     --sam-root "D:/path with spaces/sam-audio"
   ```

   Success means: audio readable, anchors satisfy `0 <= START < END <= duration`,
   FFmpeg preflight passed, model paths resolved. It does **not** verify T5
   weights or model integrity; run the SAM checkout's
   `python <sam-root>/scripts/verify_models.py` for byte/SHA-256 checks (file hashes are in
   the checkout's `model-manifest.json`).

3. **Real run** (local GPU/CPU; can take minutes and needs several GB of VRAM):

   ```sh
   python scripts/audio_toolbox.py sam separate \
     --audio "D:/data/breeze/Breeze-first30s.wav" \
     --description "melodic sound" \
     --anchor 11.18,11.50 \
     --output-dir "D:/outputs/run-001" \
     --sam-root "D:/path with spaces/sam-audio" \
     --timeout 3600
   ```

Never build shell strings around these calls; pass arguments exactly as above.
The wrapper itself always launches the SAM entry with an argument list
(`shell=False`), so quotes/metacharacters in `--description` or paths are data,
not shell syntax.

## 结果解释 / Interpreting results

* `run_dir` — directory containing the four artifacts.
* `outputs` — absolute paths of `target.wav` (separated described sound),
  `residual.wav` (everything else), `request.json` (the exact request plan),
  `report.json` (plan plus `outputs`, `elapsed_s`, `sample_rate`).
* `plan` (dry-run) — what the SAM entry would execute: resolved audio path,
  `duration_s`, anchors, model dirs, device, dtype, `network: "disabled"`.
* `upstream.stdout_tail` / `upstream.stderr_tail` — last 2000 characters of the
  SAM entry output; use these to explain upstream failures.

Dry-run success is **not** separation. Only a run whose `report`/`outputs` are
present produced audio; never present a dry-run or mock as a real separation.

## 失败处理 / Failure handling

| Error code | Exit | What to do |
|---|---|---|
| `E_USAGE` / `E_ANCHOR_FORMAT` / `E_DESCRIPTION_EMPTY` / `E_TIMEOUT_INVALID` | 2 | fix flags; anchors are `START,END` with one comma, description must be non-empty |
| `E_AUDIO_NOT_FOUND` / `E_ANCHOR_INVALID` | 3 | check the input file path (quote paths with spaces); anchors need `0 <= START < END` |
| `E_ENVIRONMENT` | 4 | set `--sam-root`/`SAM_AUDIO_ROOT`, point `--python` at the SAM environment interpreter, install model files under `model-cache/` |
| `E_UPSTREAM_FAILED` | 5 | read `error.detail.upstream_exit_code` and the tails; common causes: CUDA requested but unavailable, missing shared FFmpeg DLLs (Windows needs the full-shared build), unreadable audio |
| `E_UPSTREAM_TIMEOUT` | 6 | the run exceeded `--timeout`; raise it or shorten the input |
| `E_RUN_DIR_UNKNOWN` / `E_OUTPUT_MISSING` | 7 | inspect `upstream.stdout_tail`; the SAM entry claimed success but artifacts are missing |

Re-run `sam check-environment` after any environment change; re-run the dry-run
before retrying real inference.

## 隐私边界 / Privacy boundaries

* Everything runs **locally and offline**. The wrapper sets `HF_HUB_OFFLINE=1`
  and `TRANSFORMERS_OFFLINE=1`, performs no downloads, and reads no API keys.
* Never commit or upload input audio, model weights, API keys, full session
  logs, or personal absolute paths. Redact paths to placeholders when writing
  public reports.
* Run artifacts (`runs/`, `outputs/`, `*.wav`, `*.pt`, `*.safetensors`) stay
  outside version control.

## 许可证与来源 / License and provenance

* The wrapper code in this repository (this skill included) is **MIT licensed**
  and contains no SAM source code.
* **SAM Materials** — the `sam_audio` inference package, its entry scripts and
  any `facebook/sam-audio-*` checkpoints — remain subject to the **Meta SAM
  License** shipped upstream as `SAM-LICENSE.txt`. They are not redistributed in
  this repository; obtain them from the pinned upstream source below and keep
  the license with them.
* Pinned upstream source: `guajun/anonymous-audio-tracks`, path
  `infrastructure/audio-analysis/sam-audio`, commit
  `c603de8794cc16880dc01be0f1e868f6c2845417` (branch
  `infra/audio-analysis-migration`). Model files are resolved only from local
  directories (`model-cache/...`), with sizes/SHA-256 recorded upstream in
  `model-manifest.json`; nothing is fetched automatically.