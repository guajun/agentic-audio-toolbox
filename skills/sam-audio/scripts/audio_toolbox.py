#!/usr/bin/env python3
"""audio-toolbox sam: stable, machine-readable wrapper around the SAM Audio local entry.

Design constraints (see skills/sam-audio/SKILL.md):

* Standard library only. Importing this module never imports torch or ``sam_audio``.
* The heavy SAM Audio environment is reached only through ``subprocess`` with an
  argument *list* (never a shell string), so no shell injection is possible.
* Every invocation writes exactly one JSON document to stdout (schema
  ``audio-toolbox.sam/v1``); a one-line human summary goes to stderr.
* Nothing is downloaded, uploaded, or authenticated here. Model weights stay in
  the user's local SAM checkout.

Subcommands::

    audio-toolbox sam separate          # run (or --dry-run) one separation
    audio-toolbox sam check-environment # offline preflight, no GPU/model load

Stable exit codes::

    0 success
    2 usage error (bad flags, malformed anchor text)
    3 input error (missing audio, invalid anchor values)
    4 environment error (SAM checkout/interpreter/model files missing)
    5 upstream SAM entry failed
    6 upstream SAM entry timed out
    7 output error (run dir / expected artifacts missing)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "audio-toolbox.sam/v1"
TOOL_NAME = "audio-toolbox"
VERSION = "1.0.0"

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_INPUT = 3
EXIT_ENVIRONMENT = 4
EXIT_UPSTREAM = 5
EXIT_TIMEOUT = 6
EXIT_OUTPUT = 7

ERROR_EXIT_CODES = {
    "E_USAGE": EXIT_USAGE,
    "E_AUDIO_NOT_FOUND": EXIT_INPUT,
    "E_DESCRIPTION_EMPTY": EXIT_USAGE,
    "E_ANCHOR_FORMAT": EXIT_USAGE,
    "E_ANCHOR_INVALID": EXIT_INPUT,
    "E_TIMEOUT_INVALID": EXIT_USAGE,
    "E_ENVIRONMENT": EXIT_ENVIRONMENT,
    "E_UPSTREAM_FAILED": EXIT_UPSTREAM,
    "E_UPSTREAM_TIMEOUT": EXIT_TIMEOUT,
    "E_RUN_DIR_UNKNOWN": EXIT_OUTPUT,
    "E_OUTPUT_MISSING": EXIT_OUTPUT,
    "E_PLAN_INVALID": EXIT_OUTPUT,
    "E_REPORT_INVALID": EXIT_OUTPUT,
}

TAIL_LIMIT = 2000
DEFAULT_TIMEOUT_S = 3600.0
CHECK_TIMEOUT_S = 120.0
EXPECTED_ARTIFACTS = ("target.wav", "residual.wav", "request.json", "report.json")
# Top-level fields of the upstream dry-run plan (pinned entry writes exactly these).
PLAN_REQUIRED_FIELDS = (
    "audio",
    "duration_s",
    "description",
    "anchors",
    "model_dir",
    "text_encoder_dir",
    "device",
    "dtype",
    "output_dir",
    "network",
)


# --------------------------------------------------------------------------- output


def _tail(text: str | None, limit: int = TAIL_LIMIT) -> str:
    text = text or ""
    return text if len(text) <= limit else "..." + text[-limit:]


def _emit(payload: dict, exit_code: int) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False))
    if not payload.get("ok", False):
        error = payload.get("error", {})
        print(
            f"{TOOL_NAME}: {error.get('code', 'E_UNKNOWN')}: {error.get('message', 'unknown error')}",
            file=sys.stderr,
        )
    return exit_code


def _result(payload: dict) -> int:
    payload.setdefault("ok", True)
    return _emit(payload, payload.get("exit_code", EXIT_OK))


def _failure(code: str, message: str, action: str, detail: dict | None = None, command: list[str] | None = None) -> int:
    payload = {
        "schema": SCHEMA,
        "tool": TOOL_NAME,
        "version": VERSION,
        "ok": False,
        "action": action,
        "exit_code": ERROR_EXIT_CODES.get(code, EXIT_UPSTREAM),
        "command": command or [],
        "error": {"code": code, "message": message, "detail": detail or {}},
    }
    return _emit(payload, payload["exit_code"])


# --------------------------------------------------------------------------- helpers


class _Parser(argparse.ArgumentParser):
    """Argument parser that reports usage errors as machine-readable JSON."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("formatter_class", argparse.RawDescriptionHelpFormatter)
        super().__init__(*args, **kwargs)

    def error(self, message: str):  # noqa: D102 - argparse API
        raise _UsageError(message)


class _UsageError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _decode(raw: bytes | str | None) -> str:
    """Decode captured child output as UTF-8 with replacement, never raising.

    ``text=True`` decoders use the locale codec on Windows (e.g. GBK) and can
    crash a reader thread on undecodable bytes; capturing bytes and decoding
    here keeps the one-JSON result promise intact.
    """

    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    return raw.decode("utf-8", errors="replace")


def _plan_complete(value: object) -> bool:
    return isinstance(value, dict) and all(field in value for field in PLAN_REQUIRED_FIELDS)


def _parse_plan(stdout: str) -> tuple[dict | None, str | None]:
    """Extract the upstream dry-run plan from complete stdout.

    The full (untruncated) stdout is parsed: first as one whole JSON document,
    then by scanning for JSON objects that contain every required plan field.
    Nested/truncated fragments never qualify. Returns ``(plan, error_reason)``.
    """

    text = (stdout or "").strip()
    if not text:
        return None, "upstream dry-run produced no stdout"
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        if _plan_complete(value):
            return value, None
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if _plan_complete(value):
            return value, None
    return None, "no JSON object with the expected top-level plan fields was found in upstream stdout"


def _parse_anchors(raw: list[str] | None) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    for item in raw or []:
        head, sep, tail = item.partition(",")
        if not sep or tail == "" or "," in tail:
            raise _AnchorFormatError(item)
        try:
            start = float(head.strip())
            end = float(tail.strip())
        except ValueError as error:
            raise _AnchorFormatError(item) from error
        if not (math.isfinite(start) and math.isfinite(end)):
            raise _AnchorValueError(item, "START and END must be finite numbers")
        if not (0 <= start < end):
            raise _AnchorValueError(item, "require 0 <= START < END (END must be > START)")
        spans.append((start, end))
    return spans


class _AnchorFormatError(Exception):
    def __init__(self, item: str):
        super().__init__(item)
        self.item = item


class _AnchorValueError(Exception):
    def __init__(self, item: str, reason: str):
        super().__init__(item)
        self.item = item
        self.reason = reason


def _sam_paths(args) -> tuple[Path, Path, Path, Path]:
    """Resolve (sam_root, entry script, model_dir, text_encoder_dir)."""
    root_raw = getattr(args, "sam_root", None) or os.environ.get("SAM_AUDIO_ROOT") or os.environ.get("AUDIO_TOOLBOX_SAM_ROOT")
    if not root_raw:
        raise _EnvironmentError("SAM Audio checkout not configured", {"hint": "pass --sam-root or set SAM_AUDIO_ROOT"})
    sam_root = Path(root_raw).expanduser().resolve()
    entry = sam_root / "scripts" / "run_inference.py"
    model_raw = getattr(args, "model_dir", None) or os.environ.get("SAM_AUDIO_MODEL_DIR")
    t5_raw = getattr(args, "text_encoder_dir", None) or os.environ.get("SAM_AUDIO_T5_DIR")
    model_dir = (Path(model_raw).expanduser().resolve() if model_raw else sam_root / "model-cache" / "sam-audio-small")
    t5_dir = (Path(t5_raw).expanduser().resolve() if t5_raw else sam_root / "model-cache" / "t5-base")
    return sam_root, entry, model_dir, t5_dir


class _EnvironmentError(Exception):
    def __init__(self, message: str, detail: dict | None = None):
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


def _upstream_env() -> dict:
    env = dict(os.environ)
    # Offline is a hard policy of this wrapper: never inherit a relaxed value
    # (setdefault would preserve an inherited HF_HUB_OFFLINE=0). The parent
    # process environment is left untouched.
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    # Explicit UTF-8 child stdio; output is then decoded UTF-8 with replacement.
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _upstream_public(upstream: dict) -> dict:
    """Diagnostics view of the upstream result: bounded tails, no raw full output."""

    return {key: value for key, value in upstream.items() if key != "stdout_full"}


def _run_upstream(argv: list[str], timeout_s: float) -> dict:
    """Run the SAM entry with an argv list. Never uses a shell."""
    started = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            timeout=timeout_s,
            shell=False,
            env=_upstream_env(),
        )
    except subprocess.TimeoutExpired as error:
        stdout_full = _decode(error.stdout)
        return {
            "timed_out": True,
            "exit_code": None,
            "duration_s": round(time.monotonic() - started, 3),
            "stdout_full": stdout_full,
            "stdout_tail": _tail(stdout_full),
            "stderr_tail": _tail(_decode(error.stderr)),
        }
    except OSError as error:
        return {
            "timed_out": False,
            "exit_code": None,
            "launch_error": f"{type(error).__name__}: {error}",
            "duration_s": round(time.monotonic() - started, 3),
            "stdout_full": "",
            "stdout_tail": "",
            "stderr_tail": "",
        }
    stdout_full = _decode(completed.stdout)
    return {
        "timed_out": False,
        "exit_code": completed.returncode,
        "duration_s": round(time.monotonic() - started, 3),
        "stdout_full": stdout_full,
        "stdout_tail": _tail(stdout_full),
        "stderr_tail": _tail(_decode(completed.stderr)),
    }


# --------------------------------------------------------------------------- commands


def cmd_separate(args, action: str) -> int:
    try:
        anchors = _parse_anchors(args.anchor)
    except _AnchorFormatError as error:
        return _failure(
            "E_ANCHOR_FORMAT",
            f"invalid --anchor {error.item!r}; expected START,END with a single comma",
            action,
        )
    except _AnchorValueError as error:
        return _failure(
            "E_ANCHOR_INVALID",
            f"invalid --anchor {error.item!r}; {error.reason}",
            action,
        )

    description = (args.description or "").strip()
    if not description:
        return _failure("E_DESCRIPTION_EMPTY", "--description must be a non-empty text prompt", action)

    audio = Path(args.audio).expanduser().resolve() if args.audio else None
    if audio is None:
        return _failure("E_USAGE", "--audio is required", action)
    if not audio.is_file():
        return _failure("E_AUDIO_NOT_FOUND", f"audio file not found: {audio}", action, {"audio": str(audio)})

    try:
        timeout_s = float(args.timeout)
    except (TypeError, ValueError):
        return _failure("E_TIMEOUT_INVALID", f"--timeout must be a number of seconds, got {args.timeout!r}", action)
    if not (math.isfinite(timeout_s) and timeout_s > 0):
        return _failure("E_TIMEOUT_INVALID", "--timeout must be a finite positive number of seconds", action)

    try:
        sam_root, entry, model_dir, t5_dir = _sam_paths(args)
    except _EnvironmentError as error:
        return _failure("E_ENVIRONMENT", error.message, action, error.detail)

    if not entry.is_file():
        return _failure(
            "E_ENVIRONMENT",
            f"SAM entry script not found: {entry}",
            action,
            {"sam_root": str(sam_root), "expected_entry": str(entry)},
        )

    python_exe = Path(args.python).expanduser() if args.python else Path(sys.executable)
    if not python_exe.exists():
        return _failure("E_ENVIRONMENT", f"interpreter not found: {python_exe}", action, {"python": str(python_exe)})

    argv = [
        str(python_exe),
        str(entry),
        "--audio",
        str(audio),
        "--description",
        description,
    ]
    for start, end in anchors:
        argv += ["--anchor", f"{start},{end}"]
    argv += ["--model-dir", str(model_dir), "--text-encoder-dir", str(t5_dir)]
    if args.device:
        argv += ["--device", args.device]
    if args.dtype:
        argv += ["--dtype", args.dtype]
    output_dir = getattr(args, "output_dir", None)
    if output_dir:
        argv += ["--output-dir", str(Path(output_dir).expanduser().resolve())]
    if action == "sam.dry-run":
        argv.append("--dry-run")

    upstream = _run_upstream(argv, timeout_s)
    base = {
        "schema": SCHEMA,
        "tool": TOOL_NAME,
        "version": VERSION,
        "action": action,
        "command": argv,
        "parameters": {
            "audio": str(audio),
            "description": description,
            "anchors": [{"start": s, "end": e} for s, e in anchors],
            "model_dir": str(model_dir),
            "text_encoder_dir": str(t5_dir),
            "device": args.device or None,
            "dtype": args.dtype or None,
            "output_dir": str(Path(output_dir).expanduser().resolve()) if output_dir else None,
            "timeout_s": timeout_s,
            "sam_root": str(sam_root),
        },
        "upstream": _upstream_public(upstream),
    }

    if upstream.get("timed_out"):
        return _emit(
            {**base, "ok": False, "exit_code": EXIT_TIMEOUT,
             "error": {"code": "E_UPSTREAM_TIMEOUT",
                       "message": f"SAM entry exceeded --timeout {timeout_s:g}s and was terminated",
                       "detail": {"timeout_s": timeout_s}}},
            EXIT_TIMEOUT,
        )
    if upstream.get("launch_error"):
        return _emit(
            {**base, "ok": False, "exit_code": EXIT_ENVIRONMENT,
             "error": {"code": "E_ENVIRONMENT", "message": upstream["launch_error"], "detail": {"python": str(python_exe)}}},
            EXIT_ENVIRONMENT,
        )
    if upstream.get("exit_code") != 0:
        return _emit(
            {**base, "ok": False, "exit_code": EXIT_UPSTREAM,
             "error": {"code": "E_UPSTREAM_FAILED",
                       "message": f"SAM entry exited with code {upstream.get('exit_code')}",
                       "detail": {"upstream_exit_code": upstream.get("exit_code")}}},
            EXIT_UPSTREAM,
        )

    if action == "sam.dry-run":
        plan, plan_error = _parse_plan(upstream.get("stdout_full", ""))
        if plan is None:
            return _emit(
                {**base, "ok": False, "exit_code": EXIT_OUTPUT, "plan": None, "outputs": {},
                 "error": {"code": "E_PLAN_INVALID",
                           "message": "SAM entry dry-run did not produce a valid plan: " + plan_error,
                           "detail": {"required_fields": list(PLAN_REQUIRED_FIELDS)}}},
                EXIT_OUTPUT,
            )
        return _result({**base, "exit_code": EXIT_OK, "plan": plan, "outputs": {}, "run_dir": plan.get("output_dir")})

    run_dir = Path(output_dir).expanduser().resolve() if output_dir else None
    if run_dir is None:
        match = re.search(r"^Wrote (.+)$", upstream.get("stdout_full", ""), flags=re.MULTILINE)
        if match:
            run_dir = Path(match.group(1).strip())
    if run_dir is None:
        return _emit(
            {**base, "ok": False, "exit_code": EXIT_OUTPUT,
             "error": {"code": "E_RUN_DIR_UNKNOWN",
                       "message": "SAM entry succeeded but its run directory could not be determined",
                       "detail": {"expected_marker": "Wrote <run_dir>"}}},
            EXIT_OUTPUT,
        )

    outputs = {}
    missing = []
    for name in EXPECTED_ARTIFACTS:
        path = run_dir / name
        if path.is_file():
            outputs[name] = str(path)
        else:
            missing.append(str(path))
    report = None
    report_error = None
    report_path = run_dir / "report.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            report_error = f"{type(error).__name__}: {error}"
        else:
            if not isinstance(report, dict):
                report = None
                report_error = "report.json does not contain a JSON object"
    if missing:
        return _emit(
            {**base, "ok": False, "exit_code": EXIT_OUTPUT, "run_dir": str(run_dir), "outputs": outputs,
             "error": {"code": "E_OUTPUT_MISSING",
                       "message": "expected artifacts were not written by the SAM entry",
                       "detail": {"missing": missing}}},
            EXIT_OUTPUT,
        )
    if report_error is not None:
        return _emit(
            {**base, "ok": False, "exit_code": EXIT_OUTPUT, "run_dir": str(run_dir), "outputs": outputs,
             "error": {"code": "E_REPORT_INVALID",
                       "message": "report.json could not be read or parsed; the run is not a valid machine-readable result",
                       "detail": {"report_path": str(report_path), "reason": report_error}}},
            EXIT_OUTPUT,
        )
    return _result({**base, "exit_code": EXIT_OK, "run_dir": str(run_dir), "outputs": outputs, "report": report})


def cmd_check_environment(args) -> int:
    action = "sam.check-environment"
    checks: list[dict] = []
    required_failed = []

    def add(name: str, ok: bool, detail, required: bool = True) -> None:
        checks.append({"name": name, "ok": ok, "required": required, "detail": detail})
        if required and not ok:
            required_failed.append(name)

    try:
        sam_root, entry, model_dir, t5_dir = _sam_paths(args)
    except _EnvironmentError as error:
        add("sam_root", False, {"message": error.message, **error.detail})
        return _emit(
            {"schema": SCHEMA, "tool": TOOL_NAME, "version": VERSION, "ok": False, "action": action,
             "exit_code": EXIT_ENVIRONMENT, "checks": checks,
             "error": {"code": "E_ENVIRONMENT", "message": error.message, "detail": error.detail}},
            EXIT_ENVIRONMENT,
        )

    add("sam_root", sam_root.is_dir(), {"path": str(sam_root)})
    add("sam_entry", entry.is_file(), {"path": str(entry)})
    add("model_dir", (model_dir / "checkpoint.pt").is_file() and (model_dir / "config.json").is_file(),
        {"path": str(model_dir), "expected": ["checkpoint.pt", "config.json"]})
    add("text_encoder_dir", (t5_dir / "config.json").is_file(),
        {"path": str(t5_dir), "expected": ["config.json", "model.safetensors (checked by scripts/verify_models.py)"]})
    add("model_manifest", (sam_root / "model-manifest.json").is_file(),
        {"path": str(sam_root / "model-manifest.json"),
         "note": "byte/SHA-256 integrity is checked by the SAM checkout's scripts/verify_models.py, never downloaded here"},
        required=False)

    python_exe = Path(args.python).expanduser() if args.python else Path(sys.executable)
    add("python", python_exe.exists(), {"path": str(python_exe)})

    upstream = {"skipped": True}
    if entry.is_file() and python_exe.exists():
        upstream = _run_upstream([str(python_exe), str(entry), "--check-environment"], CHECK_TIMEOUT_S)
        upstream_ok = (not upstream.get("timed_out")) and upstream.get("exit_code") == 0
        add("ffmpeg_preflight", upstream_ok, {
            "note": "runs the SAM entry's --check-environment (FFmpeg + shared DLL discovery, no torch import)",
            "upstream_exit_code": upstream.get("exit_code"),
            "stdout_tail": upstream.get("stdout_tail"),
            "stderr_tail": upstream.get("stderr_tail"),
        })
    else:
        add("ffmpeg_preflight", False, {"note": "skipped: SAM entry or interpreter missing"})

    add("offline_policy", True, {
        "network": "disabled by policy (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1)",
        "weights": "never downloaded or uploaded by audio-toolbox",
    }, required=False)

    ok = not required_failed
    payload = {
        "schema": SCHEMA,
        "tool": TOOL_NAME,
        "version": VERSION,
        "ok": ok,
        "action": action,
        "exit_code": EXIT_OK if ok else EXIT_ENVIRONMENT,
        "checks": checks,
        "upstream": _upstream_public(upstream),
        "failed_checks": required_failed,
    }
    if not ok:
        payload["error"] = {
            "code": "E_ENVIRONMENT",
            "message": "environment checks failed: " + ", ".join(required_failed),
            "detail": {"failed_checks": required_failed},
        }
    return _emit(payload, payload["exit_code"])


# --------------------------------------------------------------------------- CLI


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog=TOOL_NAME,
        description="Stable machine-readable wrapper for local SAM Audio separation.",
        epilog=(
            "stable commands:\n"
            "  audio-toolbox sam separate --audio <FILE> --description <TEXT> [--anchor START,END]\n"
            "      [--model-dir D] [--text-encoder-dir D] [--device D] [--dtype DTYPE]\n"
            "      [--output-dir D] [--sam-root D] [--python EXE] [--timeout SEC] [--dry-run]\n"
            "  audio-toolbox sam check-environment [--sam-root D] [--model-dir D]\n"
            "      [--text-encoder-dir D] [--python EXE]"
        ),
    )
    parser.add_argument("--version", action="store_true", help="print the tool version as JSON and exit")
    sub = parser.add_subparsers(dest="group", metavar="GROUP")

    sam = sub.add_parser("sam", help="SAM Audio commands")
    sam_sub = sam.add_subparsers(dest="command", metavar="COMMAND")

    separate = sam_sub.add_parser("separate", help="separate one described sound from a local audio file")
    separate.add_argument("--audio", required=True, help="input audio file (path may contain spaces)")
    separate.add_argument("--description", required=True, help="lowercase noun/verb phrase, e.g. 'bowed strings'")
    separate.add_argument("--anchor", action="append", metavar="START,END", help="positive time span in seconds; repeatable")
    separate.add_argument("--model-dir", help="SAM Audio checkpoint directory (default: <sam-root>/model-cache/sam-audio-small)")
    separate.add_argument("--text-encoder-dir", help="T5 directory (default: <sam-root>/model-cache/t5-base)")
    separate.add_argument("--device", help="torch device, e.g. cuda or cpu (default: SAM checkout config)")
    separate.add_argument("--dtype", choices=("auto", "bfloat16", "float32"), help="model dtype")
    separate.add_argument("--output-dir", "--run-dir", dest="output_dir", help="directory for target/residual/request/report")
    separate.add_argument("--sam-root", help="SAM Audio checkout root (or env SAM_AUDIO_ROOT)")
    separate.add_argument("--python", help="interpreter that has the SAM environment (default: current interpreter)")
    separate.add_argument("--timeout", default=str(DEFAULT_TIMEOUT_S), help=f"upstream timeout in seconds (default {DEFAULT_TIMEOUT_S:g})")
    separate.add_argument("--dry-run", action="store_true", help="validate inputs and print the plan; no model load, no GPU")

    check = sam_sub.add_parser("check-environment", help="offline preflight; no torch import, no GPU, no network")
    check.add_argument("--model-dir", help="SAM Audio checkpoint directory")
    check.add_argument("--text-encoder-dir", help="T5 directory")
    check.add_argument("--sam-root", help="SAM Audio checkout root (or env SAM_AUDIO_ROOT)")
    check.add_argument("--python", help="interpreter that has the SAM environment (default: current interpreter)")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except _UsageError as error:
        return _failure("E_USAGE", error.message, "usage")

    if getattr(args, "version", False):
        return _result({"schema": SCHEMA, "tool": TOOL_NAME, "version": VERSION, "ok": True, "action": "version",
                        "exit_code": EXIT_OK})
    if args.group == "sam" and args.command == "separate":
        action = "sam.dry-run" if args.dry_run else "sam.separate"
        return cmd_separate(args, action)
    if args.group == "sam" and args.command == "check-environment":
        return cmd_check_environment(args)
    return _failure("E_USAGE", "no command given; try: audio-toolbox sam separate --help", "usage")


if __name__ == "__main__":
    raise SystemExit(main())