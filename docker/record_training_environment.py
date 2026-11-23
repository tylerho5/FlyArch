#!/opt/conda/bin/python
"""Record environment provenance before commands, not scientific run manifests.

Git and immutable image metadata come only from the FLYARCH_GIT_STATE and
FLYARCH_IMAGE_IDENTIFIER environment variables: the image does not contain Git.
When they are unset the record marks them unavailable.
Configuration, seeds, dataset and negative-sampling digests belong to training.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
import uuid


RUNS_DIR = Path("/workspace/data/runs")
RUNTIME_FIELDS = ("python", "torch", "torch_geometric", "cuda", "gpu", "tf32", "cudnn_tf32")


def probe_source(environ=None):
    """Read only whitelisted environment metadata, never execute Git in the image."""
    env = os.environ if environ is None else environ
    git_text = env.get("FLYARCH_GIT_STATE", "")
    image = env.get("FLYARCH_IMAGE_IDENTIFIER") or None
    git = {"commit": None, "dirty": None}
    missing = []
    if git_text:
        try:
            git = json.loads(git_text)
            if (not isinstance(git, dict) or set(git) != {"commit", "dirty"}
                    or not isinstance(git["commit"], str)
                    or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", git["commit"]) is None
                    or type(git["dirty"]) is not bool):
                raise ValueError("invalid Git state")
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid Git metadata in FLYARCH_GIT_STATE.") from exc
    else:
        missing.append("git")
    if image is None:
        missing.append("image_identifier")
    elif re.fullmatch(r"sha256:[0-9a-f]{64}", image) is None:
        raise ValueError("Invalid image metadata in FLYARCH_IMAGE_IDENTIFIER: an exact sha256 image ID is required.")
    if env.get("FLYARCH_REQUIRE_METADATA") == "1" and missing:
        raise ValueError("Missing metadata: " + ", ".join(missing)
                         + "; set FLYARCH_GIT_STATE and FLYARCH_IMAGE_IDENTIFIER or unset FLYARCH_REQUIRE_METADATA.")
    return {"git": git, "image_identifier": image, "metadata_unavailable": missing}


def tf32_enabled(flags):
    """Newer PyTorch raises on the legacy ``allow_tf32`` read once ``fp32_precision`` is set."""
    try:
        return bool(flags.allow_tf32)
    except RuntimeError:
        return getattr(flags, "fp32_precision", "tf32") == "tf32"


def probe_runtime():
    """Import the ML stack only inside the container runtime probe."""
    import torch
    import torch_geometric

    gpu_available = torch.cuda.is_available()
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "torch_geometric": str(torch_geometric.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if gpu_available else None,
            "tf32": tf32_enabled(torch.backends.cuda.matmul),
            "cudnn_tf32": tf32_enabled(torch.backends.cudnn)}


def utc_now():
    return datetime.now(timezone.utc)


def build_record(invocation, *, timestamp, source, runtime):
    """Pure schema assembly from explicit observations; never copy environment."""
    return {"schema_version": 1,
            "timestamp_utc": timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "invocation": list(invocation), "image_identifier": source["image_identifier"],
            "git": {key: source["git"][key] for key in ("commit", "dirty")},
            "metadata_unavailable": list(source["metadata_unavailable"]),
            "runtime": {key: runtime[key] for key in RUNTIME_FIELDS}}


def write_record(record, *, runs_dir=RUNS_DIR):
    """Atomically publish under existing managed runs; never provision parents."""
    payload = json.dumps(record, indent=2, allow_nan=False) + "\n"
    directory = Path(runs_dir) / "environment"
    if directory.is_symlink():
        raise OSError("Refusing environment symlink: records must stay under managed runs.")
    directory.mkdir(exist_ok=True)  # Deliberately no parents=True.
    stamp = datetime.fromisoformat(record["timestamp_utc"].replace("Z", "+00:00"))
    destination = directory / (stamp.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex + ".json")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                         prefix=".environment-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def record_environment(invocation, *, runs_dir=RUNS_DIR, clock=utc_now,
                       runtime_probe=probe_runtime, source_probe=probe_source):
    source = source_probe()  # Reject incomplete metadata before runtime work.
    record = build_record(invocation, timestamp=clock(), source=source, runtime=runtime_probe())
    return write_record(record, runs_dir=runs_dir)


def main(argv=None):
    invocation = sys.argv[1:] if argv is None else argv
    try:
        if not invocation:
            raise ValueError("A command is required before recording the training environment.")
        path = record_environment(invocation)
    except Exception as exc:
        print(f"Cannot record training environment: {exc}", file=sys.stderr)
        return 1
    print(f"Training environment recorded: {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
