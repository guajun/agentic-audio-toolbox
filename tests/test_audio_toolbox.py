"""CPU-only tests for the audio-toolbox SAM wrapper.

Standard library only (unittest). No torch, no SAM imports, no network, no GPU:
the SAM entry is replaced by small fake entry scripts placed in temporary
"sam root" directories (which intentionally contain spaces in their names).
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "skills" / "sam-audio" / "scripts" / "audio_toolbox.py"

spec = importlib.util.spec_from_file_location("audio_toolbox_under_test", MODULE_PATH)
at = importlib.util.module_from_spec(spec)
spec.loader.exec_module(at)


ECHO_BODY = """\
import json, os, sys
from pathlib import Path
out = Path(os.environ["FAKE_ARGV_OUT"])
out.write_text(json.dumps(sys.argv[1:]), encoding="utf-8")
args = sys.argv[1:]
out_dir = Path(args[args.index("--output-dir") + 1])
out_dir.mkdir(parents=True, exist_ok=True)
for name in ("target.wav", "residual.wav", "request.json"):
    (out_dir / name).write_text("x", encoding="utf-8")
(out_dir / "report.json").write_text(json.dumps({"elapsed_s": 0.1}), encoding="utf-8")
print("fake sam entry ok; Wrote", out_dir)
"""

PLAN_BODY = """\
import json, sys
plan = {"audio": sys.argv[sys.argv.index("--audio") + 1],
        "duration_s": 30.0,
        "description": sys.argv[sys.argv.index("--description") + 1],
        "anchors": [["+", 11.18, 11.5]], "model_dir": "m", "text_encoder_dir": "t",
        "device": "cuda", "dtype": "bfloat16", "output_dir": "out", "network": "disabled"}
print(json.dumps(plan))
"""

DRY_RUN_BODY = PLAN_BODY

LONG_PLAN_BODY = """\
import json, sys
plan = {"audio": "a.wav", "duration_s": 30.0, "description": "d" * 5000,
        "anchors": [], "model_dir": "m", "text_encoder_dir": "t",
        "device": "cuda", "dtype": "bfloat16", "output_dir": "out", "network": "disabled"}
print(json.dumps(plan))
"""

NO_JSON_BODY = """\
print("no json object here at all")
"""

INVALID_PLAN_BODY = """\
import json
print(json.dumps({"audio": "x", "duration_s": 1.0}))
"""

NESTED_ONLY_BODY = """\
print('noise {"inner": {"audio": "x", "duration_s": 1.0, "description": "d"}}')
"""

NESTED_WITH_PLAN_BODY = """\
import json, sys
print('noise {"inner": {"audio": "x"}}')
plan = {"audio": sys.argv[sys.argv.index("--audio") + 1],
        "duration_s": 30.0, "description": "melodic sound", "anchors": [],
        "model_dir": "m", "text_encoder_dir": "t", "device": "cuda",
        "dtype": "bfloat16", "output_dir": "out", "network": "disabled"}
print(json.dumps(plan))
"""

ENV_REPORT_BODY = """\
import json, os, sys
from pathlib import Path
Path(os.environ["FAKE_ENV_OUT"]).write_text(json.dumps({
    "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
    "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
    "PYTHONIOENCODING": os.environ.get("PYTHONIOENCODING"),
}), encoding="utf-8")
plan = {"audio": sys.argv[sys.argv.index("--audio") + 1],
        "duration_s": 30.0, "description": "melodic sound", "anchors": [],
        "model_dir": "m", "text_encoder_dir": "t", "device": "cuda",
        "dtype": "bfloat16", "output_dir": "out", "network": "disabled"}
print(json.dumps(plan))
"""

DECODE_BODY = """\
import json, sys
plan = {"audio": "x.wav", "duration_s": 1.0, "description": "d", "anchors": [],
        "model_dir": "m", "text_encoder_dir": "t", "device": "cpu",
        "dtype": "float32", "output_dir": "out", "network": "disabled"}
sys.stdout.buffer.write(json.dumps(plan).encode("utf-8"))
sys.stdout.buffer.write(b"\\n\\xff\\xfe\\x80 invalid tail\\n")
sys.stdout.flush()
sys.stderr.buffer.write(b"\\xff\\xfe bad stderr\\n")
sys.stderr.flush()
"""

RUN_BODY = """\
import json, sys
from pathlib import Path
args = sys.argv[1:]
out_dir = Path(args[args.index("--output-dir") + 1])
out_dir.mkdir(parents=True, exist_ok=True)
for name in ("target.wav", "residual.wav", "request.json"):
    (out_dir / name).write_text("x", encoding="utf-8")
(out_dir / "report.json").write_text(json.dumps({"elapsed_s": 1.0, "sample_rate": 48000}), encoding="utf-8")
print("Wrote", out_dir)
"""

RUN_DEFAULT_BODY = """\
import json
from pathlib import Path
out_dir = Path(__file__).resolve().parents[1] / "auto run dir"
out_dir.mkdir(parents=True, exist_ok=True)
for name in ("target.wav", "residual.wav", "request.json"):
    (out_dir / name).write_text("x", encoding="utf-8")
(out_dir / "report.json").write_text(json.dumps({"elapsed_s": 1.0}), encoding="utf-8")
print("Wrote", out_dir)
"""

PARTIAL_RUN_BODY = """\
import sys
from pathlib import Path
args = sys.argv[1:]
out_dir = Path(args[args.index("--output-dir") + 1])
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "request.json").write_text("x", encoding="utf-8")
print("Wrote", out_dir)
"""

SILENT_RUN_BODY = """\
print("done, no location given")
"""

SLEEP_BODY = """\
import time
time.sleep(60)
"""

FAIL_BODY = """\
import sys
print("cuda requested but torch.cuda.is_available() is false", file=sys.stderr)
raise SystemExit(3)
"""

CHECK_OK_BODY = """\
import sys
assert "--check-environment" in sys.argv, sys.argv
print("FFmpeg environment preflight passed (no model loaded).")
"""

CHECK_FAIL_BODY = """\
import sys
print("FFmpeg not found on PATH.", file=sys.stderr)
raise SystemExit(2)
"""


@contextlib.contextmanager
def env_with(*, remove=(), **values):
    """Temporarily set/remove environment variables without clearing the rest."""
    saved = {name: os.environ.get(name) for name in set(remove) | set(values)}
    for name in remove:
        os.environ.pop(name, None)
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class ToolboxTestCase(unittest.TestCase):
    def make_sam_root(self, tmp: Path, body: str) -> Path:
        root = Path(tmp) / "sam root with spaces"
        (root / "scripts").mkdir(parents=True, exist_ok=True)
        (root / "scripts" / "run_inference.py").write_text(body, encoding="utf-8")
        return root

    def make_models(self, root: Path) -> tuple[Path, Path]:
        model = root / "model-cache" / "sam-audio-small"
        t5 = root / "model-cache" / "t5-base"
        model.mkdir(parents=True)
        t5.mkdir(parents=True)
        (model / "checkpoint.pt").write_bytes(b"\0")
        (model / "config.json").write_text("{}", encoding="utf-8")
        (t5 / "config.json").write_text("{}", encoding="utf-8")
        return model, t5

    def make_audio(self, tmp: Path, name: str = "input clip & more.wav") -> Path:
        audio = Path(tmp) / name
        audio.write_bytes(b"RIFF0000WAVEfake")
        return audio

    def run_cli(self, argv, **env):
        remove = env.pop("remove", ())
        stdout, stderr = io.StringIO(), io.StringIO()
        with env_with(remove=remove, **env), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = at.main(argv)
        payload = json.loads(stdout.getvalue())
        return code, payload, stderr.getvalue()


class CliSurfaceTests(ToolboxTestCase):
    def test_help_exits_zero_without_heavy_imports(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as error:
                at.main(["--help"])
        self.assertEqual(error.exception.code, 0)
        text = stdout.getvalue()
        for fragment in ("sam separate", "sam check-environment", "--dry-run", "--anchor", "--timeout"):
            self.assertIn(fragment, text)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("sam_audio", sys.modules)

    def test_subcommand_help_lists_required_flags(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as error:
                at.main(["sam", "separate", "--help"])
        self.assertEqual(error.exception.code, 0)
        text = stdout.getvalue()
        for flag in ("--audio", "--description", "--anchor", "--model-dir", "--text-encoder-dir",
                     "--device", "--dtype", "--output-dir", "--sam-root", "--python", "--timeout", "--dry-run"):
            self.assertIn(flag, text)

    def test_version_payload(self):
        code, payload, _ = self.run_cli(["--version"])
        self.assertEqual(code, 0)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["action"], "version")
        self.assertEqual(payload["schema"], "audio-toolbox.sam/v1")

    def test_missing_command_is_structured_usage_error(self):
        code, payload, stderr = self.run_cli([])
        self.assertEqual(code, 2)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "E_USAGE")
        self.assertEqual(payload["exit_code"], code)
        self.assertIn("E_USAGE", stderr)

    def test_missing_required_flags_is_structured_usage_error(self):
        code, payload, _ = self.run_cli(["sam", "separate"])
        self.assertEqual(code, 2)
        self.assertEqual(payload["error"]["code"], "E_USAGE")
        self.assertEqual(payload["exit_code"], 2)

    def test_envelope_keys_are_stable(self):
        code, payload, _ = self.run_cli(["--version"])
        self.assertEqual(
            {"schema", "tool", "version", "ok", "action", "exit_code"},
            set(payload),
        )
        self.assertEqual(code, payload["exit_code"])


class ValidationTests(ToolboxTestCase):
    def test_audio_not_found(self):
        code, payload, _ = self.run_cli(
            ["sam", "separate", "--audio", "nope missing.wav", "--description", "melodic sound"])
        self.assertEqual(code, 3)
        self.assertEqual(payload["error"]["code"], "E_AUDIO_NOT_FOUND")
        self.assertIn("audio", payload["error"]["detail"])

    def test_empty_description(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = self.make_audio(tmp)
            code, payload, _ = self.run_cli(
                ["sam", "separate", "--audio", str(audio), "--description", "   "])
        self.assertEqual(code, 2)
        self.assertEqual(payload["error"]["code"], "E_DESCRIPTION_EMPTY")

    def test_anchor_format_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = self.make_audio(tmp)
            for bad in ("1.0", "1,2,3", "x,y", ",2", "1,"):
                with self.subTest(anchor=bad):
                    code, payload, _ = self.run_cli(
                        ["sam", "separate", "--audio", str(audio), "--description", "melodic sound",
                         "--anchor", bad])
                    self.assertEqual(code, 2)
                    self.assertEqual(payload["error"]["code"], "E_ANCHOR_FORMAT")
                    self.assertIn(bad, payload["error"]["message"])

    def test_anchor_value_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = self.make_audio(tmp)
            for bad in ("2,1", "1,1", "-1,2", "nan,1", "0,inf"):
                with self.subTest(anchor=bad):
                    code, payload, _ = self.run_cli(
                        ["sam", "separate", "--audio", str(audio), "--description", "melodic sound",
                         "--anchor", bad])
                    self.assertEqual(code, 3)
                    self.assertEqual(payload["error"]["code"], "E_ANCHOR_INVALID")

    def test_timeout_flag_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = self.make_audio(tmp)
            for bad in ("abc", "-5", "0", "inf"):
                with self.subTest(timeout=bad):
                    code, payload, _ = self.run_cli(
                        ["sam", "separate", "--audio", str(audio), "--description", "melodic sound",
                         "--timeout", bad])
                    self.assertEqual(code, 2)
                    self.assertEqual(payload["error"]["code"], "E_TIMEOUT_INVALID")


class EnvironmentTests(ToolboxTestCase):
    def test_missing_sam_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = self.make_audio(tmp)
            code, payload, _ = self.run_cli(
                ["sam", "separate", "--audio", str(audio), "--description", "melodic sound"],
                remove=("SAM_AUDIO_ROOT", "AUDIO_TOOLBOX_SAM_ROOT"))
        self.assertEqual(code, 4)
        self.assertEqual(payload["error"]["code"], "E_ENVIRONMENT")
        self.assertIn("hint", payload["error"]["detail"])

    def test_missing_entry_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = self.make_audio(tmp)
            root = Path(tmp) / "empty root"
            root.mkdir()
            code, payload, _ = self.run_cli(
                ["sam", "separate", "--audio", str(audio), "--description", "melodic sound",
                 "--sam-root", str(root)])
        self.assertEqual(code, 4)
        self.assertEqual(payload["error"]["code"], "E_ENVIRONMENT")
        self.assertIn("expected_entry", payload["error"]["detail"])

    def test_missing_python_interpreter(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = self.make_audio(tmp)
            root = self.make_sam_root(tmp, ECHO_BODY)
            code, payload, _ = self.run_cli(
                ["sam", "separate", "--audio", str(audio), "--description", "melodic sound",
                 "--sam-root", str(root), "--python", str(Path(tmp) / "no such python.exe")])
        self.assertEqual(code, 4)
        self.assertEqual(payload["error"]["code"], "E_ENVIRONMENT")
        self.assertIn("python", payload["error"]["detail"])

    def test_sam_root_from_environment_variable(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = self.make_audio(tmp)
            root = self.make_sam_root(tmp, DRY_RUN_BODY)
            code, payload, _ = self.run_cli(
                ["sam", "separate", "--audio", str(audio), "--description", "melodic sound", "--dry-run"],
                SAM_AUDIO_ROOT=str(root))
        self.assertEqual(code, 0)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["parameters"]["sam_root"], str(root))


class UpstreamInvocationTests(ToolboxTestCase):
    def test_argv_passthrough_without_shell_injection(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = self.make_audio(tmp, 'clip & del ; echo hi + plus.wav')
            self.assertTrue(audio.is_file())
            root = self.make_sam_root(tmp, ECHO_BODY)
            argv_out = Path(tmp) / "argv.json"
            description = 'bowed strings" & del /f /q C:\\ & echo "pwned'
            code, payload, stderr = self.run_cli(
                ["sam", "separate", "--audio", str(audio), "--description", description,
                 "--anchor", "0,1.5", "--anchor", "2,3",
                 "--model-dir", str(root / "model dir"), "--text-encoder-dir", str(root / "t5 dir"),
                 "--device", "cpu", "--dtype", "float32",
                 "--output-dir", str(root / "out dir"), "--sam-root", str(root)],
                FAKE_ARGV_OUT=str(argv_out))
            self.assertEqual(code, 0, stderr)
            forwarded = json.loads(argv_out.read_text(encoding="utf-8"))
            expected = [
                str(root.resolve() / "scripts" / "run_inference.py"),
                "--audio", str(audio.resolve()),
                "--description", description,
                "--anchor", "0.0,1.5", "--anchor", "2.0,3.0",
                "--model-dir", str((root / "model dir").resolve()),
                "--text-encoder-dir", str((root / "t5 dir").resolve()),
                "--device", "cpu", "--dtype", "float32",
                "--output-dir", str((root / "out dir").resolve()),
            ]
            self.assertEqual(forwarded, expected[1:])
            self.assertEqual(payload["command"], [sys.executable, *expected])

    def test_no_shell_apis_in_source(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        for forbidden in ("shell=True", "os.system", "os.popen", "eval(", "exec(", "commands."):
            self.assertNotIn(forbidden, source)
        self.assertIn("shell=False", source)

    def test_upstream_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = self.make_audio(tmp)
            root = self.make_sam_root(tmp, FAIL_BODY)
            code, payload, stderr = self.run_cli(
                ["sam", "separate", "--audio", str(audio), "--description", "melodic sound",
                 "--sam-root", str(root)])
        self.assertEqual(code, 5)
        self.assertEqual(payload["error"]["code"], "E_UPSTREAM_FAILED")
        self.assertEqual(payload["error"]["detail"]["upstream_exit_code"], 3)
        self.assertEqual(payload["upstream"]["exit_code"], 3)
        self.assertIn("cuda requested", payload["upstream"]["stderr_tail"])
        self.assertIn("E_UPSTREAM_FAILED", stderr)

    def test_upstream_timeout_is_terminated(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = self.make_audio(tmp)
            root = self.make_sam_root(tmp, SLEEP_BODY)
            started = time.monotonic()
            code, payload, _ = self.run_cli(
                ["sam", "separate", "--audio", str(audio), "--description", "melodic sound",
                 "--sam-root", str(root), "--timeout", "1"])
            elapsed = time.monotonic() - started
        self.assertEqual(code, 6)
        self.assertEqual(payload["error"]["code"], "E_UPSTREAM_TIMEOUT")
        self.assertTrue(payload["upstream"]["timed_out"])
        self.assertLess(elapsed, 30)

    def test_dry_run_returns_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = self.make_audio(tmp)
            root = self.make_sam_root(tmp, DRY_RUN_BODY)
            code, payload, _ = self.run_cli(
                ["sam", "separate", "--audio", str(audio), "--description", "melodic sound",
                 "--anchor", "11.18,11.5", "--dry-run", "--sam-root", str(root)])
        self.assertEqual(code, 0)
        self.assertEqual(payload["action"], "sam.dry-run")
        self.assertEqual(payload["plan"]["network"], "disabled")
        self.assertEqual(payload["parameters"]["anchors"], [{"start": 11.18, "end": 11.5}])
        self.assertEqual(payload["outputs"], {})
        self.assertNotIn("stdout_full", payload["upstream"])


class OutputTests(ToolboxTestCase):
    def test_real_run_collects_outputs_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = self.make_audio(tmp)
            root = self.make_sam_root(tmp, RUN_BODY)
            out_dir = Path(tmp) / "run dir with spaces"
            code, payload, _ = self.run_cli(
                ["sam", "separate", "--audio", str(audio), "--description", "melodic sound",
                 "--output-dir", str(out_dir), "--sam-root", str(root)])
            self.assertEqual(code, 0)
            self.assertEqual(payload["action"], "sam.separate")
            self.assertEqual(payload["run_dir"], str(out_dir.resolve()))
            self.assertEqual(
                set(payload["outputs"]),
                {"target.wav", "residual.wav", "request.json", "report.json"},
            )
            for path in payload["outputs"].values():
                self.assertTrue(Path(path).is_file())
            self.assertEqual(payload["report"]["sample_rate"], 48000)

    def test_run_dir_parsed_from_upstream_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = self.make_audio(tmp)
            root = self.make_sam_root(tmp, RUN_DEFAULT_BODY)
            code, payload, _ = self.run_cli(
                ["sam", "separate", "--audio", str(audio), "--description", "melodic sound",
                 "--sam-root", str(root)])
            self.assertEqual(code, 0)
            self.assertTrue(payload["run_dir"])
            self.assertTrue(Path(payload["run_dir"]).is_dir())

    def test_missing_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = self.make_audio(tmp)
            root = self.make_sam_root(tmp, PARTIAL_RUN_BODY)
            out_dir = Path(tmp) / "partial run"
            code, payload, _ = self.run_cli(
                ["sam", "separate", "--audio", str(audio), "--description", "melodic sound",
                 "--output-dir", str(out_dir), "--sam-root", str(root)])
        self.assertEqual(code, 7)
        self.assertEqual(payload["error"]["code"], "E_OUTPUT_MISSING")
        missing = payload["error"]["detail"]["missing"]
        self.assertEqual(len(missing), 3)
        self.assertTrue(all("request.json" not in item for item in missing))

    def test_unknown_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = self.make_audio(tmp)
            root = self.make_sam_root(tmp, SILENT_RUN_BODY)
            code, payload, _ = self.run_cli(
                ["sam", "separate", "--audio", str(audio), "--description", "melodic sound",
                 "--sam-root", str(root)])
        self.assertEqual(code, 7)
        self.assertEqual(payload["error"]["code"], "E_RUN_DIR_UNKNOWN")


class CheckEnvironmentTests(ToolboxTestCase):
    def test_check_environment_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_sam_root(tmp, CHECK_OK_BODY)
            self.make_models(root)
            (root / "model-manifest.json").write_text("[]", encoding="utf-8")
            code, payload, _ = self.run_cli(
                ["sam", "check-environment", "--sam-root", str(root)])
        self.assertEqual(code, 0)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["action"], "sam.check-environment")
        names = {check["name"] for check in payload["checks"]}
        self.assertTrue({"sam_root", "sam_entry", "model_dir", "text_encoder_dir",
                         "python", "ffmpeg_preflight", "model_manifest", "offline_policy"} <= names)
        self.assertEqual(payload["failed_checks"], [])
        self.assertIn("preflight passed", payload["upstream"]["stdout_tail"])

    def test_check_environment_missing_models_and_ffmpeg(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_sam_root(tmp, CHECK_FAIL_BODY)
            code, payload, _ = self.run_cli(
                ["sam", "check-environment", "--sam-root", str(root)])
        self.assertEqual(code, 4)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "E_ENVIRONMENT")
        self.assertIn("model_dir", payload["failed_checks"])
        self.assertIn("ffmpeg_preflight", payload["failed_checks"])
        by_name = {check["name"]: check for check in payload["checks"]}
        self.assertFalse(by_name["model_dir"]["ok"])
        self.assertFalse(by_name["ffmpeg_preflight"]["ok"])

    def test_check_environment_missing_sam_root(self):
        code, payload, _ = self.run_cli(
            ["sam", "check-environment"], remove=("SAM_AUDIO_ROOT", "AUDIO_TOOLBOX_SAM_ROOT"))
        self.assertEqual(code, 4)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "E_ENVIRONMENT")


class OfflinePolicyTests(ToolboxTestCase):
    def test_offline_env_forced_for_child_and_parent_unchanged(self):
        watched = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONIOENCODING")
        before = {name: os.environ.get(name) for name in watched}
        with tempfile.TemporaryDirectory() as tmp:
            audio = self.make_audio(tmp)
            root = self.make_sam_root(tmp, ENV_REPORT_BODY)
            env_out = Path(tmp) / "env.json"
            code, payload, _ = self.run_cli(
                ["sam", "separate", "--audio", str(audio), "--description", "melodic sound",
                 "--dry-run", "--sam-root", str(root)],
                FAKE_ENV_OUT=str(env_out),
                HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0")
            self.assertEqual(code, 0, payload.get("error"))
            seen = json.loads(env_out.read_text(encoding="utf-8"))
        self.assertEqual(seen["HF_HUB_OFFLINE"], "1")
        self.assertEqual(seen["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(seen["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(before, {name: os.environ.get(name) for name in watched})


class PlanTests(ToolboxTestCase):
    def test_long_plan_is_parsed_and_tails_stay_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = self.make_audio(tmp)
            root = self.make_sam_root(tmp, LONG_PLAN_BODY)
            code, payload, _ = self.run_cli(
                ["sam", "separate", "--audio", str(audio), "--description", "melodic sound",
                 "--dry-run", "--sam-root", str(root)])
        self.assertEqual(code, 0, payload.get("error"))
        self.assertIsNotNone(payload["plan"])
        self.assertEqual(len(payload["plan"]["description"]), 5000)
        tail = payload["upstream"]["stdout_tail"]
        self.assertTrue(tail.startswith("..."))
        self.assertLessEqual(len(tail), 2003)
        self.assertNotIn("stdout_full", payload["upstream"])
        self.assertNotIn("d" * 4000, json.dumps(payload["upstream"]))

    def test_missing_plan_is_structured_output_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = self.make_audio(tmp)
            root = self.make_sam_root(tmp, NO_JSON_BODY)
            code, payload, _ = self.run_cli(
                ["sam", "separate", "--audio", str(audio), "--description", "melodic sound",
                 "--dry-run", "--sam-root", str(root)])
        self.assertEqual(code, 7)
        self.assertEqual(payload["error"]["code"], "E_PLAN_INVALID")
        self.assertIsNone(payload["plan"])
        self.assertFalse(payload["ok"])

    def test_incomplete_plan_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = self.make_audio(tmp)
            root = self.make_sam_root(tmp, INVALID_PLAN_BODY)
            code, payload, _ = self.run_cli(
                ["sam", "separate", "--audio", str(audio), "--description", "melodic sound",
                 "--dry-run", "--sam-root", str(root)])
        self.assertEqual(code, 7)
        self.assertEqual(payload["error"]["code"], "E_PLAN_INVALID")
        self.assertIn("output_dir", payload["error"]["detail"]["required_fields"])

    def test_nested_fragment_is_not_misidentified_as_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = self.make_audio(tmp)
            root = self.make_sam_root(tmp, NESTED_ONLY_BODY)
            code, payload, _ = self.run_cli(
                ["sam", "separate", "--audio", str(audio), "--description", "melodic sound",
                 "--dry-run", "--sam-root", str(root)])
            self.assertEqual(code, 7)
            self.assertEqual(payload["error"]["code"], "E_PLAN_INVALID")

            root = self.make_sam_root(tmp, NESTED_WITH_PLAN_BODY)
            code, payload, _ = self.run_cli(
                ["sam", "separate", "--audio", str(audio), "--description", "melodic sound",
                 "--dry-run", "--sam-root", str(root)])
            self.assertEqual(code, 0, payload.get("error"))
            self.assertEqual(payload["plan"]["audio"], str(audio.resolve()))


class ReportTests(ToolboxTestCase):
    CORRUPT_REPORT_BODY = """\
import sys
from pathlib import Path
args = sys.argv[1:]
out_dir = Path(args[args.index("--output-dir") + 1])
out_dir.mkdir(parents=True, exist_ok=True)
for name in ("target.wav", "residual.wav", "request.json"):
    (out_dir / name).write_text("x", encoding="utf-8")
(out_dir / "report.json").write_text("{not valid json", encoding="utf-8")
print("Wrote", out_dir)
"""

    NONOBJECT_REPORT_BODY = """\
import sys
from pathlib import Path
args = sys.argv[1:]
out_dir = Path(args[args.index("--output-dir") + 1])
out_dir.mkdir(parents=True, exist_ok=True)
for name in ("target.wav", "residual.wav", "request.json"):
    (out_dir / name).write_text("x", encoding="utf-8")
(out_dir / "report.json").write_text("[1, 2, 3]", encoding="utf-8")
print("Wrote", out_dir)
"""

    def test_corrupt_report_is_structured_output_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = self.make_audio(tmp)
            root = self.make_sam_root(tmp, self.CORRUPT_REPORT_BODY)
            out_dir = Path(tmp) / "run dir"
            code, payload, _ = self.run_cli(
                ["sam", "separate", "--audio", str(audio), "--description", "melodic sound",
                 "--output-dir", str(out_dir), "--sam-root", str(root)])
        self.assertEqual(code, 7)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "E_REPORT_INVALID")
        self.assertIn("reason", payload["error"]["detail"])
        self.assertNotIn("report", payload)

    def test_non_object_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = self.make_audio(tmp)
            root = self.make_sam_root(tmp, self.NONOBJECT_REPORT_BODY)
            out_dir = Path(tmp) / "run dir"
            code, payload, _ = self.run_cli(
                ["sam", "separate", "--audio", str(audio), "--description", "melodic sound",
                 "--output-dir", str(out_dir), "--sam-root", str(root)])
        self.assertEqual(code, 7)
        self.assertEqual(payload["error"]["code"], "E_REPORT_INVALID")


class DecodingTests(ToolboxTestCase):
    def test_invalid_utf8_bytes_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = self.make_audio(tmp)
            root = self.make_sam_root(tmp, DECODE_BODY)
            code, payload, stderr = self.run_cli(
                ["sam", "separate", "--audio", str(audio), "--description", "melodic sound",
                 "--dry-run", "--sam-root", str(root)])
        self.assertEqual(code, 0, payload.get("error"))
        self.assertEqual(payload["plan"]["network"], "disabled")
        self.assertIn("\ufffd", payload["upstream"]["stderr_tail"])
        self.assertIn("\ufffd", payload["upstream"]["stdout_tail"])
        self.assertNotIn("Traceback", json.dumps(payload))
        self.assertNotIn("Traceback", stderr)

    def test_non_ascii_paths_pass_through_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = self.make_audio(tmp, "音频 clip 测试.wav")
            root = self.make_sam_root(tmp, ECHO_BODY)
            argv_out = Path(tmp) / "argv.json"
            out_dir = Path(tmp) / "输出 目录"
            code, payload, stderr = self.run_cli(
                ["sam", "separate", "--audio", str(audio), "--description", "旋律 声音",
                 "--output-dir", str(out_dir), "--sam-root", str(root)],
                FAKE_ARGV_OUT=str(argv_out))
            self.assertEqual(code, 0, stderr)
            forwarded = json.loads(argv_out.read_text(encoding="utf-8"))
            self.assertEqual(forwarded[1], str(audio.resolve()))
            self.assertEqual(forwarded[3], "旋律 声音")
            self.assertEqual(forwarded[-1], str(out_dir.resolve()))
            self.assertIn("旋律 声音", payload["parameters"]["description"])

    def test_subprocess_stdout_is_exactly_one_json_document(self):
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            audio = self.make_audio(tmp)
            root = self.make_sam_root(tmp, RUN_BODY)
            out_dir = Path(tmp) / "run dir"
            completed = subprocess.run(
                [sys.executable, str(MODULE_PATH), "sam", "separate",
                 "--audio", str(audio), "--description", "melodic sound",
                 "--output-dir", str(out_dir), "--sam-root", str(root)],
                capture_output=True, shell=False, timeout=120)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertNotIn("stdout_full", payload["upstream"])
        self.assertNotIn(b"Traceback", completed.stdout)
        self.assertNotIn(b"Traceback", completed.stderr)


if __name__ == "__main__":
    unittest.main()