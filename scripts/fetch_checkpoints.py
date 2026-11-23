#!/usr/bin/env python3
"""Download the 36 gate-cohort expert checkpoints from the Zenodo deposit.

Fetches each ``best.pt`` listed in ``results/checkpoints.sha256`` into
``data/runs/<run_id>/best.pt`` and verifies its sha256 as it streams. A file
that is already in place with the right digest is skipped; a failed download
leaves no partial file behind. Stdlib-only, Python 3.9+.

The deposit holds the files flat, named ``<run_id>.best.pt``, under the record
set in ``sync_banc_dataset.ZENODO_RECORD``.

Usage:
    python scripts/fetch_checkpoints.py [--manifest FILE] [--out-dir DIR]
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sync_banc_dataset import ZENODO_RECORD, get_ssl_context, log, zenodo_record_is_set  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "results" / "checkpoints.sha256"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "runs"
CHUNK = 8 * 1024 * 1024


def read_manifest(path):
    """Parse ``sha256sum`` lines (``<hex>  checkpoints/<run_id>/best.pt``) into (run_id, digest)."""
    entries = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        digest, _, rel = raw.partition("  ")
        entries.append((Path(rel.strip()).parent.name, digest.strip()))
    return entries


def deposit_url(run_id):
    return f"https://zenodo.org/records/{ZENODO_RECORD}/files/{run_id}.best.pt?download=1"


def _open(url):
    return urllib.request.urlopen(urllib.request.Request(url), timeout=120, context=get_ssl_context())


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_one(run_id, digest, out_dir, opener=_open):
    """Fetch one checkpoint; True when ``<out_dir>/<run_id>/best.pt`` verifies."""
    dest = Path(out_dir) / run_id / "best.pt"
    if dest.exists() and _sha256(dest) == digest:
        log(f"checkpoint: {run_id} already present")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name("best.pt.part")
    h = hashlib.sha256()
    try:
        with opener(deposit_url(run_id)) as resp, open(part, "wb") as out:
            for chunk in iter(lambda: resp.read(CHUNK), b""):
                h.update(chunk)
                out.write(chunk)
    except OSError as exc:
        log(f"checkpoint: {run_id} download failed: {exc}")
        part.unlink(missing_ok=True)
        return False
    if h.hexdigest() != digest:
        log(f"checkpoint: {run_id} sha256 MISMATCH ({h.hexdigest()}), discarding")
        part.unlink(missing_ok=True)
        return False
    part.replace(dest)
    log(f"checkpoint: {run_id} OK")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download the gate-cohort checkpoints from the Zenodo deposit")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    if not zenodo_record_is_set():
        log(f"ZENODO_RECORD is {ZENODO_RECORD!r}: the Zenodo deposit id is not set in this checkout. "
            "Set ZENODO_RECORD in scripts/sync_banc_dataset.py to the deposit's record id.")
        return 1
    entries = read_manifest(args.manifest)
    ok = [fetch_one(run_id, digest, args.out_dir) for run_id, digest in entries]
    log(f"{sum(ok)}/{len(ok)} checkpoints verified")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
