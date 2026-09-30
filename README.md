# Agentic Audio Toolbox

Portable Agent Skills and CLI tools for audio research. Initial implementation is tracked in [anonymous-audio-tracks #29](https://github.com/guajun/anonymous-audio-tracks/issues/29), under research [#27](https://github.com/guajun/anonymous-audio-tracks/issues/27).

This repository contains integration code and skill instructions, not SAM model weights or private audio. SAM Audio materials remain subject to their upstream Meta SAM License; wrappers must preserve provenance and clearly distinguish their license.

Development uses independent worktrees/PRs and a main-Agent review gate.

## Layout

```text
skills/sam-audio/
├── SKILL.md                     # agent-facing instructions (frontmatter + workflow)
├── references/cli-reference.md  # stable JSON schema, error codes, env vars
└── scripts/audio_toolbox.py     # the CLI implementation (standard library only)
bin/audio-toolbox                # POSIX launcher
bin/audio-toolbox.cmd            # Windows launcher
tests/                           # CPU-only stdlib unittest suite
```

The CLI implementation lives **inside the skill directory** so that a
`gh skill install` copy of `skills/sam-audio/` stays self-contained and usable
without any other file from this repository.

## CLI

Stable command surface (schema `audio-toolbox.sam/v1`, stable exit codes):

```sh
python skills/sam-audio/scripts/audio_toolbox.py sam separate \
  --audio "<FILE>" --description "melodic sound" \
  [--anchor 11.18,11.50] [--model-dir D] [--text-encoder-dir D] \
  [--device cuda] [--dtype auto|bfloat16|float32] [--output-dir D] \
  [--sam-root D] [--python EXE] [--timeout SEC] [--dry-run]

python skills/sam-audio/scripts/audio_toolbox.py sam check-environment \
  [--sam-root D] [--model-dir D] [--text-encoder-dir D] [--python EXE]
```

Every call prints exactly one JSON document to stdout (`ok`, `action`,
`exit_code`, `command`, `parameters`, `upstream`, `run_dir`, `outputs`,
`report`/`plan`, `error`). Failures print one summary line to stderr.

The wrapper is a **thin subprocess launcher** for the existing SAM Audio entry
(`<sam-root>/scripts/run_inference.py`): it builds an argument list and runs it
with `shell=False`, so no shell injection is possible; it never imports torch,
never loads models, and never touches the network (it forces
`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` for the upstream process).

## Requirements

* Wrapper + tests: Python 3.11+ standard library only.
* Real separation: a local SAM Audio checkout pinned at
  `guajun/anonymous-audio-tracks@infrastructure/audio-analysis/sam-audio`
  commit `c603de8794cc16880dc01be0f1e868f6c2845417`, its locked environment
  (`uv sync --locked`), FFmpeg 4–8 **full-shared** build on Windows, and locally
  installed model weights under `model-cache/` (sizes/SHA-256 in upstream
  `model-manifest.json`, checked by upstream `scripts/verify_models.py`).
* Point the CLI at that checkout with `--sam-root` or `SAM_AUDIO_ROOT`, and at
  its interpreter with `--python` when the current Python lacks the SAM
  environment.

## Tests

```sh
python -m unittest discover -s tests -v
```

CPU-only: the SAM entry is replaced by fake entry scripts in temporary
directories (including paths with spaces and shell metacharacters). Covered:
`--help`/usage surface, structured errors and exit codes, anchor validation,
argv passthrough without shell injection, timeouts, upstream failures,
dry-run plans, run-dir/output collection, and environment preflight.

Real-model GPU inference is intentionally out of scope here
(anonymous-audio-tracks #33); dry-run/check-environment exercise the real SAM
entry without loading any model.

## License and provenance

* Wrapper code in this repository (including `skills/sam-audio/`) is **MIT
  licensed** — see [LICENSE](LICENSE). It contains no SAM source code.
* **SAM Materials** (the upstream `sam_audio` package, entry scripts, and
  `facebook/sam-audio-*` checkpoints) stay under the **Meta SAM License**
  (`SAM-LICENSE.txt` upstream) and are **not** redistributed here. Fetch them
  from the pinned upstream source and keep the license with them.
* No model weights, private audio, API keys, or personal absolute paths are
  committed to this repository.