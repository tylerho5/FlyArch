#!/usr/bin/env python3
"""Download and verify the BANC v888 female brain-and-nerve-cord connectome (F2).

Downloads the BANC connectivity + annotation files, CAVE materialization 888,
and verifies each against its pinned size and sha256. Resumable, stdlib-only
(no pip install), runs identically on Windows and macOS. Same shape as
``sync_flywire_dataset.py``.

The meta, edgelist and per-neuron files were pinned from BANC's public GCS
bucket (gs://lee-lab_brain-and-nerve-cord-fly-connectome), which overwrites
objects in place and keeps no old generations. BANC replaced the meta file on
2026-11-11, so the pinned bytes of these three files are not publicly hosted.
The download step stops for them with the current release's URLs; files placed
in the data directory by hand are verified against the pins. The enriched
synapse table downloads from Harvard Dataverse doi:10.7910/DVN/7WTH1N,
datafile 13992863, which carries it at the pinned size and sha256.

To add a file, download it by hand, run ``--pin DIR`` to print a ready-to-paste
MANIFEST entry (name, size, sha256), add its ``url``, and move it into MANIFEST.

Usage:
    python sync_banc_dataset.py [--step download|verify|all] [--data-dir DIR]
    python sync_banc_dataset.py --pin DIR

Steps:
    download   fetch missing files into data/raw/banc/ (resumable)
    verify     check size + checksum of every local file
    all        download -> verify (default)
"""

import argparse
import hashlib
import os
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw" / "banc"

# ---------------------------------------------------------------------------
# TLS: python.org / Homebrew builds often ship with no CA bundle configured.
# Find one portably (certifi -> known system paths -> OpenSSL default).
# ---------------------------------------------------------------------------
_SSL_CONTEXT = None


def _candidate_ca_files():
    cands = []
    try:
        import certifi
        cands.append(certifi.where())
    except ImportError:
        pass
    cands += [
        "/etc/ssl/cert.pem",                      # macOS, Debian/Ubuntu
        "/etc/pki/tls/certs/ca-bundle.crt",      # Fedora/RHEL
        "/etc/ssl/certs/ca-certificates.crt",     # Debian/Ubuntu (alt)
        "/opt/homebrew/etc/openssl@3/cert.pem",   # macOS ARM Homebrew
        "/usr/local/etc/openssl@3/cert.pem",      # macOS Intel Homebrew
    ]
    return cands


def get_ssl_context():
    global _SSL_CONTEXT
    if _SSL_CONTEXT is not None:
        return _SSL_CONTEXT
    for cafile in _candidate_ca_files():
        if os.path.exists(cafile):
            try:
                _SSL_CONTEXT = ssl.create_default_context(cafile=cafile)
                log(f"tls: using CA bundle {cafile}")
                return _SSL_CONTEXT
            except ssl.SSLError:
                continue
    # Fall back to the interpreter default; may fail at handshake on
    # systems with no CA store. Tell the user how to fix it.
    _SSL_CONTEXT = ssl.create_default_context()
    log("tls: no CA bundle found; trying interpreter default. "
        "If you see CERTIFICATE_VERIFY_FAILED, run: python -m pip install certifi")
    return _SSL_CONTEXT


# ---------------------------------------------------------------------------
# Manifest: name, source url, expected size (bytes), checksum (algo, hex).
# Pinned 2026-10-11 with `--pin DIR`. url None: the pinned bytes are not public;
# bucket_path names the current release's object.
# ---------------------------------------------------------------------------
BUCKET = "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome"
ZENODO_RECORD = "ZENODO_RECORD_TBD"
DATAVERSE_DATAFILE = "https://dataverse.harvard.edu/api/access/datafile"

MANIFEST = [
    dict(name="banc_888_meta.feather",
         url=None, bucket_path="compiled_data/banc_888/banc_888_meta.feather",
         size=57_503_026, algo="sha256",
         digest="86ccf5df0c67419f8c5f43e93a7ed38d23a080e9f7fde26737290252f3780098"),
    dict(name="banc_888_edgelist_neuropil_v2.feather",
         url=None, bucket_path="neuron_connectivity/v888/banc_888_edgelist_neuropil_v2.feather",
         size=4_501_005_898, algo="sha256",
         digest="c6c9b825be8ce8b542d97ea5fabba4e30726ae0d10ebebdcbaae9f2f713c2d1e"),
    dict(name="banc_888_synapses_per_neuron_neuropil_side_v2.parquet",
         url=None, bucket_path="synapses/v2.0/banc_888_synapses_per_neuron_neuropil_side_v2.parquet",
         size=920_069_257, algo="sha256",
         digest="f6f7558d63118af4e1549a54085dc0716d7659e2241c1d5a943e715d8e964aab"),
    dict(name="banc_888_synapses_v2_enriched.parquet",
         url=f"{DATAVERSE_DATAFILE}/13992863",
         size=17_146_564_854, algo="sha256",
         digest="e75d1430089a55ef0e2eb40ba88803d0ddc59175c98af03ba78ef58d26832092"),
]


def zenodo_record_is_set(record=None):
    """True once ZENODO_RECORD holds a numeric Zenodo record id."""
    return str(ZENODO_RECORD if record is None else record).isdigit()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def log(msg):
    print(f"[sync] {msg}", flush=True)


def _urlopen_retry(req, timeout):
    """urlopen that tolerates flaky DNS (common on VPN or captive-portal networks).

    Retries on transient name-resolution failures only; all other errors
    propagate immediately so the caller's own retry logic handles them.
    """
    last = None
    for attempt in range(1, 6):
        try:
            return urllib.request.urlopen(req, timeout=timeout, context=get_ssl_context())
        except urllib.error.URLError as exc:
            last = exc
            if isinstance(exc.reason, socket.gaierror):
                log(f"net: DNS hiccup (attempt {attempt}/5): {exc.reason}; retrying")
                time.sleep(2 * attempt)
                continue
            raise
    raise last


def digest_file(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Pin: hash a directory of manually-downloaded files into MANIFEST entries.
# ---------------------------------------------------------------------------
def step_pin(directory):
    """Print a ready-to-paste MANIFEST ``dict(...)`` line for every file in DIR.

    You cannot pin a checksum you have never seen -- this is the tool that
    pins it the moment you can, i.e. right after a manual download. The
    printed line omits ``url``: add that by hand when moving the entry into
    MANIFEST, since a local file does not carry its own download source.
    """
    directory = Path(directory)
    if not directory.is_dir():
        log(f"pin: {directory} is not a directory")
        return False
    files = sorted(p for p in directory.iterdir() if p.is_file())
    if not files:
        log(f"pin: no files found in {directory}")
        return False
    for path in files:
        size = path.stat().st_size
        digest = digest_file(path, "sha256")
        print(
            f'    dict(name={path.name!r}, size={size}, '
            f'algo="sha256", digest={digest!r}),  # url=...'
        )
    return True


# ---------------------------------------------------------------------------
# Download (resumable, retried)
# ---------------------------------------------------------------------------
def download_one(item, data_dir, ui):
    path = data_dir / item["name"]
    url = item["url"]

    if path.exists() and path.stat().st_size == item["size"]:
        return True

    for attempt in range(1, 4):
        # Re-derive the resume offset each attempt. A prior attempt may have
        # appended bytes before a drop; reusing the loop-entry offset would
        # duplicate those bytes, fail the size check, and force a restart.
        offset = path.stat().st_size if path.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            req = urllib.request.Request(url, headers=headers)
            total = offset
            with _urlopen_retry(req, 120) as resp, open(path, "ab" if offset else "wb") as out:
                while True:
                    chunk = resp.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    total += len(chunk)
                    ui.update(item["name"], total, len(chunk))
            if path.stat().st_size == item["size"]:
                ui.finish(item["name"])
                return True
            # Server may not honor Range; loop again from scratch once.
            if offset:
                log(f"download: {item['name']} size mismatch after resume, retrying from 0")
                path.unlink(missing_ok=True)
                continue
            log(f"download: {item['name']} size mismatch (got {path.stat().st_size}, want {item['size']})")
            return False
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                log(f"download: {item['name']} attempt {attempt} failed: {exc}")
                time.sleep(5 * attempt)
                continue
            log(
                f"download: {item['name']} 404 at {item['url']}: the URL is wrong or the "
                "record has moved. Check it against MANIFEST."
            )
            return False
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            log(f"download: {item['name']} attempt {attempt} failed: {exc}")
            time.sleep(5 * attempt)
    return False


class _NullUI:
    """Minimal progress sink used when the manifest is small or empty."""

    def update(self, name, done, chunk):
        pass

    def finish(self, name):
        log(f"download: {name} OK")


def step_download(manifest, data_dir):
    if not manifest:
        log("download: MANIFEST is empty -- fetch the files by hand first "
            "(see module docstring), then `--pin DIR` and populate MANIFEST")
        return True
    data_dir.mkdir(parents=True, exist_ok=True)
    unhosted = [i for i in manifest if i["url"] is None and not (data_dir / i["name"]).exists()]
    hosted = [i for i in manifest if i["url"] is not None]
    ok = True
    if hosted:
        get_ssl_context()
        ui = _NullUI()
        for item in hosted:
            ok = download_one(item, data_dir, ui) and ok
    if unhosted:
        log("download: the pinned bytes of these files are not publicly hosted (BANC replaced "
            "the meta file on 2026-11-11 and its bucket keeps no old versions):")
        for item in unhosted:
            log(f"  {item['name']}: current release at {BUCKET}/{item['bucket_path']}")
        log(f"To build F2 from the current release, download those into {data_dir}; "
            "the meta file will not match its pinned sha256, so skip `--step verify` for it.")
        ok = False
    return ok


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
def step_verify(manifest, data_dir):
    if not manifest:
        log("verify: MANIFEST is empty -- nothing to verify yet")
        return True
    ok = True
    for item in manifest:
        path = data_dir / item["name"]
        if not path.exists():
            log(f"verify: {item['name']} MISSING")
            ok = False
            continue
        size_ok = path.stat().st_size == item["size"]
        if not size_ok:
            log(f"verify: {item['name']} size mismatch ({path.stat().st_size} != {item['size']})")
            ok = False
            continue
        if item["digest"]:
            actual = digest_file(path, item["algo"])
            if actual != item["digest"]:
                log(f"verify: {item['name']} {item['algo']} MISMATCH ({actual})")
                ok = False
            else:
                log(f"verify: {item['name']} {item['algo']} OK")
        else:
            log(f"verify: {item['name']} size OK (no checksum pinned)")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Download and verify the BANC v888 dataset (F2)")
    parser.add_argument("--step", choices=["download", "verify", "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--pin", type=Path, default=None, metavar="DIR",
        help="Print MANIFEST entries (name, size, sha256) for every file in "
             "DIR and exit, instead of running --step",
    )
    args = parser.parse_args(argv)

    if args.pin is not None:
        return 0 if step_pin(args.pin) else 1

    steps = ["download", "verify"] if args.step == "all" else [args.step]
    ok = True
    for step in steps:
        fn = {"download": step_download, "verify": step_verify}[step]
        log(f"== step: {step} ==")
        ok = fn(MANIFEST, args.data_dir)
        if not ok:
            log(f"step '{step}' reported failures")
            break

    log("done")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
