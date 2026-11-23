"""Daemon-free semantic checks for the training Compose contract."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "compose.training.yml"
DOCKER = shutil.which("docker")
GIT = shutil.which("git")


@unittest.skipUnless(DOCKER, "Docker CLI is required for Compose config checks")
class TrainingComposeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = str(Path(self.temp.name) / "data")
        self.cache = str(Path(self.temp.name) / "cache")

    def compose_result(self, overrides=None):
        self.assertTrue(COMPOSE.is_file(), f"missing {COMPOSE}")
        env = os.environ.copy()
        for name in ("LOCAL_UID", "LOCAL_GID", "FLYARCH_DATA_ROOT", "FLYARCH_CACHE_ROOT",
                     "FLYARCH_TENSORBOARD_PORT", "COMPOSE_PROFILES", "COMPOSE_FILE",
                     "FLYARCH_IMAGE_IDENTIFIER", "FLYARCH_GIT_STATE", "FLYARCH_GIT_BRANCH",
                     "FLYARCH_GIT_SUBJECT", "FLYARCH_REQUIRE_METADATA",
                     "FLYARCH_SOURCE_STATE"):
            env.pop(name, None)
        env.update(LOCAL_UID="1234", LOCAL_GID="2345",
                   FLYARCH_DATA_ROOT=self.data, FLYARCH_CACHE_ROOT=self.cache)
        for name, value in (overrides or {}).items():
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value
        return subprocess.run(
            [DOCKER, "compose", "--env-file", os.devnull, "--profile", "*", "-f", str(COMPOSE),
             "config", "--format", "json"],
            cwd=ROOT, env=env, capture_output=True, text=True, check=False,
        )

    def config(self, overrides=None):
        result = self.compose_result(overrides)
        self.assertEqual(0, result.returncode, result.stderr)
        return json.loads(result.stdout)

    def assert_mounts(self, service, expected):
        mounts = service["volumes"]
        self.assertEqual(len(expected), len(mounts))
        self.assertEqual(set(expected), {mount["target"] for mount in mounts})
        for mount in mounts:
            with self.subTest(target=mount["target"]):
                source, readonly = expected[mount["target"]]
                self.assertEqual("bind", mount["type"])
                self.assertEqual(source, mount["source"])
                self.assertEqual(readonly, mount.get("read_only", False))
                # Compose omits false in normalized JSON, but short bind syntax
                # normalizes to create_host_path=true and must fail this check.
                self.assertIs(False, mount.get("bind", {}).get("create_host_path", False))

    def expected_mounts(self, data, cache):
        return {
            "/workspace": (str(ROOT), False),
            "/workspace/data/raw": (f"{data}/raw", True),
            "/workspace/data/interim": (f"{data}/interim", False),
            "/workspace/data/processed": (f"{data}/processed", False),
            "/workspace/data/runs": (f"{data}/runs", False),
            "/cache": (cache, False),
        }

    def test_three_services_share_fixed_image_build_and_nonroot_identity(self):
        config = self.config()
        self.assertIn("x-flyarch-common", config)
        self.assertNotIn("networks", config)
        for mount in config["x-flyarch-common"]["volumes"]:
            self.assertEqual("bind", mount["type"])
            self.assertIs(False, mount["bind"]["create_host_path"])
        self.assertEqual({"dev", "trainer", "tensorboard"}, set(config["services"]))
        for name, service in config["services"].items():
            with self.subTest(service=name):
                self.assertEqual("flyarch-training:dev", service["image"])
                self.assertEqual(str(ROOT), service["build"]["context"])
                self.assertEqual("docker/training.Dockerfile", service["build"]["dockerfile"])
                # One shared build: no service picks a target or a second image.
                self.assertNotIn("target", service["build"])
                self.assertEqual("1234:2345", service["user"])
                self.assertEqual("/workspace", service["working_dir"])
                self.assertEqual("/cache/home/1234", service["environment"]["HOME"])
                self.assertEqual("/cache/home/1234/.cache", service["environment"]["XDG_CACHE_HOME"])
                # The numeric uid is the container's user; no host name is forwarded.
                self.assertNotIn("USER", service["environment"])
                self.assertIn(service.get("restart", "no"), ("no", ""))
                self.assertFalse(service.get("privileged", False))
                self.assertEqual("bridge", service.get("network_mode"))
                self.assertNotIn("networks", service)
                self.assertFalse(service.get("devices"))
                self.assertFalse(service.get("deploy"))

    def test_only_trainer_requests_gpu_and_host_ipc(self):
        services = self.config()["services"]
        gpus = services["trainer"]["gpus"]
        self.assertEqual(1, len(gpus))
        self.assertEqual(-1, gpus[0]["count"])
        # The gpus field itself implies the GPU capability; Compose may omit it.
        self.assertIn("gpu", gpus[0].get("capabilities", ["gpu"]))
        self.assertEqual("host", services["trainer"]["ipc"])
        for name in ("dev", "tensorboard"):
            self.assertFalse(services[name].get("gpus"))
            self.assertNotEqual("host", services[name].get("ipc"))
        for name in ("dev", "trainer"):
            self.assertEqual(["bash"], services[name]["command"])
            self.assertFalse(services[name].get("profiles"))
        self.assertTrue(services["dev"]["stdin_open"])
        self.assertTrue(services["dev"]["tty"])

    def test_runtime_image_is_pinned_and_metadata_is_explicitly_forwarded(self):
        image = "sha256:" + "a" * 64
        git = json.dumps({"commit": "b" * 40, "dirty": True})
        source = json.dumps({"files": ["train/train.py"], "sha256": "c" * 64})
        services = self.config({"FLYARCH_IMAGE_IDENTIFIER": image, "FLYARCH_GIT_STATE": git,
                                "FLYARCH_SOURCE_STATE": source,
                                "FLYARCH_GIT_BRANCH": "feat/th-run-labels",
                                "FLYARCH_GIT_SUBJECT": "feat: label runs"})["services"]
        for name in ("dev", "trainer"):
            service = services[name]
            self.assertEqual(service["environment"]["FLYARCH_GIT_BRANCH"], "feat/th-run-labels")
            self.assertEqual(service["environment"]["FLYARCH_GIT_SUBJECT"], "feat: label runs")
            self.assertEqual(service["image"], image)
            self.assertEqual(service["entrypoint"], ["/opt/flyarch/training-entrypoint.sh"])
            self.assertEqual(service["environment"]["FLYARCH_IMAGE_IDENTIFIER"], image)
            self.assertEqual(service["environment"]["FLYARCH_GIT_STATE"], git)
            self.assertEqual(service["environment"]["FLYARCH_SOURCE_STATE"], source)
        for name in ("dev", "trainer"):
            self.assertEqual(services[name]["environment"]["FLYARCH_REQUIRE_METADATA"], "0")

    def test_metadata_is_optional_and_forwarded_empty_when_unset(self):
        services = self.config()["services"]
        for name in ("dev", "trainer"):
            env = services[name]["environment"]
            self.assertIn("FLYARCH_IMAGE_IDENTIFIER", env)
            self.assertIn("FLYARCH_GIT_STATE", env)
            self.assertEqual(env["FLYARCH_GIT_BRANCH"], "")
            self.assertEqual(env["FLYARCH_GIT_SUBJECT"], "")
            self.assertEqual(env["FLYARCH_IMAGE_IDENTIFIER"], "")
            self.assertEqual(env["FLYARCH_GIT_STATE"], "")
            self.assertEqual(env["FLYARCH_SOURCE_STATE"], "")
            self.assertEqual(env["FLYARCH_REQUIRE_METADATA"], "0")

    def test_training_mounts_use_overridden_roots_without_creating_host_paths(self):
        services = self.config()["services"]
        for name in ("dev", "trainer"):
            with self.subTest(service=name):
                self.assert_mounts(services[name], self.expected_mounts(self.data, self.cache))

    def test_default_storage_roots_are_relative_to_the_repository(self):
        services = self.config({"FLYARCH_DATA_ROOT": None, "FLYARCH_CACHE_ROOT": None,
                                "LOCAL_UID": "3456", "LOCAL_GID": "4567"})["services"]
        for name in ("dev", "trainer"):
            self.assert_mounts(services[name], self.expected_mounts(
                f"{ROOT}/data", f"{ROOT}/.cache/training"))
        for service in services.values():
            self.assertEqual("3456:4567", service["user"])
            self.assertEqual("/cache/home/3456", service["environment"]["HOME"])
            self.assertEqual("/cache/home/3456/.cache", service["environment"]["XDG_CACHE_HOME"])

    def test_tensorboard_has_only_readonly_runs_and_writable_cache(self):
        service = self.config()["services"]["tensorboard"]
        self.assert_mounts(service, {
            "/workspace/data/runs": (f"{self.data}/runs", True),
            "/cache": (self.cache, False),
        })
        self.assertEqual(["tools"], service["profiles"])
        self.assertEqual(["/opt/conda/bin/python", "-m", "tensorboard.main"], service["entrypoint"])
        self.assertEqual(["--logdir=/workspace/data/runs", "--host=0.0.0.0", "--port=6006"],
                         service["command"])

    def test_only_tensorboard_publishes_a_loopback_port(self):
        for override, expected_port in ((None, "6006"), ("16006", "16006")):
            services = self.config({"FLYARCH_TENSORBOARD_PORT": override})["services"]
            for name in ("dev", "trainer"):
                self.assertFalse(services[name].get("ports"))
            ports = services["tensorboard"]["ports"]
            self.assertEqual(1, len(ports))
            self.assertEqual("127.0.0.1", ports[0]["host_ip"])
            self.assertEqual(expected_port, ports[0]["published"])
            self.assertEqual(6006, ports[0]["target"])
            self.assertEqual("tcp", ports[0]["protocol"])

    def test_uid_and_gid_are_required_even_without_a_dotenv_file(self):
        for name in ("LOCAL_UID", "LOCAL_GID"):
            for missing in (None, ""):
                with self.subTest(variable=name, value=missing):
                    result = self.compose_result({name: missing})
                    self.assertNotEqual(0, result.returncode)
                    self.assertIn(f"set {name}", result.stderr)


class TrainingRunsIgnoreTests(unittest.TestCase):
    @unittest.skipUnless(GIT, "Git CLI is required for ignore checks")
    def test_generated_runs_are_ignored_but_suites_remain_trackable(self):
        result = subprocess.run(
            [GIT, "check-ignore", "--no-index", "--stdin"], cwd=ROOT,
            input="data/runs/example/events.out.tfevents\ndata/suites/example.json\n",
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(["data/runs/example/events.out.tfevents"], result.stdout.splitlines())


if __name__ == "__main__":
    unittest.main()
