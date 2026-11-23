"""Stdlib-only provenance and entrypoint tests; no Docker or torch required."""

import builtins
from datetime import datetime, timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docker/record_training_environment.py"
STAMP = datetime(2026, 8, 30, 1, 2, 3, tzinfo=timezone.utc)
IMAGE = "sha256:" + "a" * 64
COMMIT = "b" * 40
RUNTIME = {"python": "3.11.11", "torch": "2.7.1", "torch_geometric": "2.6.1",
           "cuda": "12.8", "gpu": None, "tf32": False, "cudnn_tf32": True}


def load_recorder():
    spec = importlib.util.spec_from_file_location("record_training_environment", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.is_file(), "provenance recorder is missing")
        self.recorder = load_recorder()

    def source(self, dirty=False):
        return {"git": {"commit": COMMIT, "dirty": dirty},
                "image_identifier": IMAGE, "metadata_unavailable": []}

    def record(self):
        return self.recorder.build_record(["python", "a;b", "$HOME", ""],
            timestamp=STAMP, source=self.source(), runtime=RUNTIME)

    def test_exact_schema_clean_dirty_cpu_and_literal_invocation(self):
        for dirty in (False, True):
            record = self.recorder.build_record(["python", "a;b", "$HOME", ""],
                timestamp=STAMP, source=self.source(dirty), runtime=RUNTIME)
            self.assertEqual(record, {"schema_version": 1,
                "timestamp_utc": "2026-08-30T01:02:03Z",
                "invocation": ["python", "a;b", "$HOME", ""],
                **self.source(dirty), "runtime": RUNTIME})

    def test_import_never_imports_torch_or_pyg(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            self.assertNotIn(name.split(".")[0], ("torch", "torch_geometric"))
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guarded):
            load_recorder()

    def test_runtime_probe_cpu_and_gpu(self):
        for available in (False, True):
            cuda = SimpleNamespace(is_available=lambda: available,
                get_device_name=lambda index: "RTX 5090")
            backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                                       cudnn=SimpleNamespace(allow_tf32=True))
            torch = SimpleNamespace(__version__="2.7.1", version=SimpleNamespace(cuda="12.8"),
                                    cuda=cuda, backends=backends)
            with patch.dict(sys.modules, {"torch": torch,
                 "torch_geometric": SimpleNamespace(__version__="2.6.1")}):
                runtime = self.recorder.probe_runtime()
            self.assertEqual(runtime, {**RUNTIME, "python": self.recorder.platform.python_version(),
                "gpu": "RTX 5090" if available else None})

    def test_tf32_probe_reads_the_newer_flag_when_the_legacy_read_raises(self):
        class Flags:
            def __init__(self, precision):
                self.fp32_precision = precision

            @property
            def allow_tf32(self):
                raise RuntimeError("mixed legacy and new TF32 APIs")

        self.assertTrue(self.recorder.tf32_enabled(Flags("tf32")))
        self.assertFalse(self.recorder.tf32_enabled(Flags("ieee")))

    def test_direct_dev_metadata_explicitly_unavailable_and_secrets_omitted(self):
        self.assertEqual(self.recorder.probe_source({"SECRET": "never-record-this"}), {
            "git": {"commit": None, "dirty": None}, "image_identifier": None,
            "metadata_unavailable": ["git", "image_identifier"]})

    def test_complete_gpu_metadata_and_no_git_subprocess(self):
        env = {"FLYARCH_REQUIRE_METADATA": "1", "FLYARCH_GIT_STATE": json.dumps(self.source(True)["git"]),
               "FLYARCH_IMAGE_IDENTIFIER": IMAGE, "SECRET": "never-record-this"}
        with patch("subprocess.run", side_effect=AssertionError("container must not invoke git")):
            self.assertEqual(self.recorder.probe_source(env), self.source(True))

    def test_gpu_metadata_rejects_missing_and_invalid_fields(self):
        good = {"FLYARCH_REQUIRE_METADATA": "1", "FLYARCH_GIT_STATE": json.dumps(self.source()["git"]),
                "FLYARCH_IMAGE_IDENTIFIER": IMAGE}
        for key, bad in [("FLYARCH_GIT_STATE", ""), ("FLYARCH_IMAGE_IDENTIFIER", ""),
                ("FLYARCH_GIT_STATE", "not-json"), ("FLYARCH_GIT_STATE", "[]"),
                ("FLYARCH_GIT_STATE", json.dumps({"commit": COMMIT, "dirty": "false"})),
                ("FLYARCH_GIT_STATE", json.dumps({"commit": "main", "dirty": False})),
                ("FLYARCH_GIT_STATE", json.dumps({"commit": COMMIT, "dirty": False, "secret": "x"})),
                ("FLYARCH_IMAGE_IDENTIFIER", "flyarch-training:dev")]:
            with self.subTest(key=key, bad=bad), self.assertRaisesRegex(ValueError, "metadata"):
                self.recorder.probe_source({**good, key: bad})

    def test_atomic_writes_unique_names_and_complete_json(self):
        with tempfile.TemporaryDirectory() as temp:
            runs = Path(temp) / "runs"
            runs.mkdir()
            original_replace = os.replace
            seen = []
            def replace(source, dest):
                self.assertEqual(json.loads(Path(source).read_text()), self.record())
                self.assertFalse(Path(dest).exists())
                seen.append(dest)
                original_replace(source, dest)
            with patch.object(self.recorder.os, "replace", side_effect=replace):
                paths = [self.recorder.write_record(self.record(), runs_dir=runs) for _ in range(2)]
            self.assertEqual(len(set(paths)), 2)
            self.assertEqual(len(seen), 2)
            self.assertEqual(set(runs.iterdir()), {runs / "environment"})
            self.assertEqual(set((runs / "environment").iterdir()), set(paths))
            for path in paths:
                self.assertTrue(path.name.startswith("20260830T010203Z-"))
                self.assertEqual(json.loads(path.read_text()), self.record())

    def test_missing_managed_runs_is_not_created(self):
        with tempfile.TemporaryDirectory() as temp:
            runs = Path(temp) / "missing/runs"
            with self.assertRaises(OSError):
                self.recorder.write_record(self.record(), runs_dir=runs)
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_atomic_replace_failure_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(self.recorder.os, "replace", side_effect=OSError("disk error")):
                with self.assertRaisesRegex(OSError, "disk error"):
                    self.recorder.write_record(self.record(), runs_dir=Path(temp))
            self.assertEqual(list((Path(temp) / "environment").iterdir()), [])

    def test_environment_symlink_cannot_redirect_records_outside_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            runs, outside = Path(temp) / "runs", Path(temp) / "outside"
            runs.mkdir()
            outside.mkdir()
            (runs / "environment").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(OSError, "symlink"):
                self.recorder.write_record(self.record(), runs_dir=runs)
            self.assertEqual(list(outside.iterdir()), [])

    def test_serialization_failure_leaves_no_partial_record(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(TypeError):
                self.recorder.write_record({"bad": object()}, runs_dir=Path(temp))
            self.assertFalse(list(Path(temp).rglob("*.json")))

    def test_injected_clock_runtime_and_source_probes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.recorder.record_environment(["python", "a;b", "$HOME", ""],
                runs_dir=Path(temp), clock=lambda: STAMP,
                runtime_probe=lambda: RUNTIME, source_probe=self.source)
            self.assertEqual(json.loads(path.read_text()), self.record())

    def test_cli_reports_record_failure_without_success_message(self):
        with patch.object(self.recorder, "record_environment", side_effect=OSError("read-only runs")), \
                patch("sys.stderr", new_callable=io.StringIO) as output:
            self.assertEqual(self.recorder.main(["python", "train.py"]), 1)
        self.assertIn("Cannot record training environment: read-only runs", output.getvalue())
        self.assertNotIn("Training environment recorded:", output.getvalue())

    def test_cli_rejects_empty_command_before_probes(self):
        with patch.object(self.recorder, "record_environment") as record, \
                patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(self.recorder.main([]), 1)
        record.assert_not_called()

    def test_metadata_failure_happens_before_runtime_or_filesystem_work(self):
        def failed_source():
            raise ValueError("missing metadata")
        with patch.object(self.recorder, "write_record") as write, \
                patch.object(self.recorder, "probe_runtime") as runtime:
            with self.assertRaisesRegex(ValueError, "missing metadata"):
                self.recorder.record_environment(["python"],
                    source_probe=failed_source, runtime_probe=runtime)
        write.assert_not_called()
        runtime.assert_not_called()


class EntrypointTests(unittest.TestCase):
    def run_entrypoint(self, status=0):
        path = ROOT / "docker/training-entrypoint.sh"
        self.assertTrue(path.is_file(), "training entrypoint is missing")
        # Feed production shell to bash, replacing only the container interpreter
        # and script locations. Real bash exec and argv semantics remain intact.
        script = path.read_text().replace("/opt/conda/bin/python", sys.executable)
        script = script.replace("/opt/flyarch/record_training_environment.py",
            "-c 'import json,sys; print(json.dumps(sys.argv[1:]), file=sys.stderr); sys.exit(" + str(status) + ")'")
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp) / "cache"
            cache.mkdir()
            script = script.replace("/cache", str(cache))
            home = cache / "home/1234"
            result = subprocess.run(["bash", "-c", script, "entrypoint", sys.executable, "-c",
                "import json,sys; print(json.dumps(sys.argv[1:])); sys.exit(17)", "a;b", "$HOME", "", "two words"],
                env={**os.environ, "HOME": str(home), "XDG_CACHE_HOME": str(home / ".cache")},
                capture_output=True, text=True)
            self.assertTrue((home / ".cache").is_dir())
            return result

    def test_exec_preserves_arguments_and_nonzero_exit(self):
        result = self.run_entrypoint()
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertEqual(json.loads(result.stdout), ["a;b", "$HOME", "", "two words"])
        recorded = json.loads(result.stderr)
        self.assertEqual(recorded[0], sys.executable)
        self.assertEqual(recorded[-4:], ["a;b", "$HOME", "", "two words"])

    def test_record_failure_prevents_command(self):
        result = self.run_entrypoint(status=23)
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
