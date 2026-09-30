# audio-toolbox CLI reference (schema `audio-toolbox.sam/v1`)

Entry point: `scripts/audio_toolbox.py` in this skill directory (standard
library only, Python 3.11+). Every invocation prints exactly one JSON document
to stdout; failures additionally print one summary line to stderr. Exit codes
and `error.code` values below are stable API.

## Common envelope

```json
{
  "schema": "audio-toolbox.sam/v1",
  "tool": "audio-toolbox",
  "version": "1.0.0",
  "ok": true,
  "action": "sam.separate | sam.dry-run | sam.check-environment | version | usage",
  "exit_code": 0,
  "command": ["<python>", "<sam-root>/scripts/run_inference.py", "..."],
  "parameters": { "...": "echo of resolved inputs" },
  "upstream": {
    "timed_out": false,
    "exit_code": 0,
    "duration_s": 12.3,
    "stdout_tail": "...(last 2000 chars)...",
    "stderr_tail": "..."
  }
}
```

* `command` is the **exact argv list** handed to the SAM entry. It is built as a
  list and executed with `shell=False`; no shell parsing ever occurs, so quotes,
  `&`, `;`, backticks and spaces inside `--audio`/`--description`/paths are data.
* `upstream.stdout_tail` / `upstream.stderr_tail` are truncated to the last 2000
  characters (`...` prefix marks truncation). The complete raw child output is
  used for parsing but is **never** copied into the result JSON.
* The child process is launched with a forced `HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1` and `PYTHONIOENCODING=utf-8` (inherited conflicting
  values are overridden in the child; the wrapper's own environment is not
  modified). Child stdio is captured as bytes and decoded UTF-8 with replacement,
  so undecodable bytes become `U+FFFD` and can never crash the wrapper or break
  the one-JSON promise.

## `sam separate` / `sam dry-run` result fields

| Field | Meaning |
|---|---|
| `run_dir` | resolved run directory (from `--output-dir` or the upstream `Wrote <dir>` line) |
| `outputs` | map of `target.wav`, `residual.wav`, `request.json`, `report.json` to absolute paths |
| `report` | parsed `report.json` from the run directory (real runs); an unreadable/corrupt/non-object report is rejected with `E_REPORT_INVALID`, never reported as success |
| `plan` | plan JSON parsed from the **complete** upstream dry-run stdout (`--dry-run` only) |
| `parameters.anchors` | list of `{"start": float, "end": float}` |

A `--dry-run` success means: audio readable, finite positive duration, anchors
satisfy `0 <= START < END <= duration`, FFmpeg preflight passed, model paths
resolved **and** the upstream stdout contained a plan JSON object with every
top-level field `audio`, `duration_s`, `description`, `anchors`, `model_dir`,
`text_encoder_dir`, `device`, `dtype`, `output_dir`, `network`. Missing or
incomplete plan JSON is rejected with `E_PLAN_INVALID`; nested JSON fragments
are never mistaken for the plan. It loads no model and produces **no separated
audio**.

## `sam check-environment` result fields

`checks` is a list of `{"name", "ok", "required", "detail"}` entries:

| Name | Required | Checks |
|---|---|---|
| `sam_root` | yes | SAM checkout directory exists |
| `sam_entry` | yes | `<sam-root>/scripts/run_inference.py` exists |
| `model_dir` | yes | `checkpoint.pt` and `config.json` in the checkpoint dir |
| `text_encoder_dir` | yes | `config.json` in the T5 dir |
| `model_manifest` | no | `<sam-root>/model-manifest.json` present (integrity via upstream `scripts/verify_models.py`) |
| `python` | yes | the chosen interpreter exists |
| `ffmpeg_preflight` | yes | upstream `--check-environment` passed (FFmpeg + Windows shared DLLs) |
| `offline_policy` | no | informational: no network, no weight download |

`failed_checks` lists required checks that did not pass.

## Error objects and exit codes

```json
{
  "ok": false,
  "error": {
    "code": "E_UPSTREAM_FAILED",
    "message": "SAM entry exited with code 1",
    "detail": {"upstream_exit_code": 1}
  }
}
```

| `error.code` | Exit | Meaning |
|---|---|---|
| `E_USAGE` | 2 | unknown/missing flags, no command |
| `E_ANCHOR_FORMAT` | 2 | `--anchor` text is not exactly `START,END` |
| `E_DESCRIPTION_EMPTY` | 2 | empty `--description` |
| `E_TIMEOUT_INVALID` | 2 | `--timeout` not a finite positive number |
| `E_AUDIO_NOT_FOUND` | 3 | `--audio` is not an existing file |
| `E_ANCHOR_INVALID` | 3 | non-finite anchor values or `START >= END` / negative `START` |
| `E_ENVIRONMENT` | 4 | SAM root/entry/interpreter missing, or the entry could not be launched |
| `E_UPSTREAM_FAILED` | 5 | SAM entry returned a non-zero code (see `detail.upstream_exit_code`) |
| `E_UPSTREAM_TIMEOUT` | 6 | SAM entry exceeded `--timeout` and was terminated |
| `E_RUN_DIR_UNKNOWN` | 7 | upstream reported success but no run directory could be determined |
| `E_OUTPUT_MISSING` | 7 | run directory exists but expected artifacts are absent (`detail.missing`) |
| `E_PLAN_INVALID` | 7 | dry-run stdout contained no valid plan with the expected top-level fields |
| `E_REPORT_INVALID` | 7 | `report.json` unreadable/corrupt/not an object; the run is not a machine-readable result |

The process exit code always equals the JSON `exit_code` field.

## Environment variables

| Variable | Meaning |
|---|---|
| `SAM_AUDIO_ROOT` (alias `AUDIO_TOOLBOX_SAM_ROOT`) | default for `--sam-root` |
| `SAM_AUDIO_MODEL_DIR` | default for `--model-dir` |
| `SAM_AUDIO_T5_DIR` | default for `--text-encoder-dir` |
| `PYTHON` | interpreter used by the `bin/audio-toolbox` launchers |

The wrapper additionally forces `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`
and `PYTHONIOENCODING=utf-8` for the upstream process (unconditionally: an
inherited `HF_HUB_OFFLINE=0` is overridden in the child). Nothing else in the
environment is modified, and no credentials are read.