#!/usr/bin/env python3
"""Download and verify the FlyWire v783 whole-brain connectome dataset (F1).

Downloads the authoritative FlyWire v783 connectivity + annotation files and
verifies each against its pinned size and sha256. Resumable, stdlib-only (no
pip install), runs identically on Windows and macOS.

Sources (all 783-aligned, immutable):
  * Connectivity: Zenodo record 10676866 (v783.0)
    https://zenodo.org/records/10676866
  * Annotations:  github.com/flyconnectome/flywire_annotations @ v3.1.0

Usage:
    python sync_flywire_dataset.py [--step download|verify|all] [--data-dir DIR]

Steps:
    download   fetch missing files into data/raw/ (resumable)
    verify     check size + checksum of every local file
    all        download -> verify (default)
"""

import argparse
import hashlib
import json
import os
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw"

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
# Zenodo md5s are published by Zenodo; GitHub sha256s are pinned after a
# verified local download of the v3.1.0 tag.
# ---------------------------------------------------------------------------
ZENODO = "https://zenodo.org/api/records/10676866/files"
GITHUB = "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/v3.1.0/supplemental_files"

MANIFEST = [
    # -- Connectivity (Zenodo 10676866, v783.0) -----------------------------
    dict(name="flywire_synapses_783.feather",
         url=f"{ZENODO}/flywire_synapses_783.feather/content",
         size=9_492_998_242, algo="md5", digest="f8f1b97c9d4b0ea9b4c8b287f6b99091"),
    dict(name="proofread_connections_783.feather",
         url=f"{ZENODO}/proofread_connections_783.feather/content",
         size=852_022_274, algo="md5", digest="f48f972d262323a102aed49af1396b8a"),
    dict(name="proofread_root_ids_783.npy",
         url=f"{ZENODO}/proofread_root_ids_783.npy/content",
         size=1_114_168, algo="md5", digest="e0e6c19732fd8c7a4e39a2d170105421"),
    dict(name="per_neuron_neuropil_count_post_783.feather",
         url=f"{ZENODO}/per_neuron_neuropil_count_post_783.feather/content",
         size=233_843_050, algo="md5", digest="bb5999f10920ade803d9f37097a43a56"),
    dict(name="per_neuron_neuropil_count_pre_783.feather",
         url=f"{ZENODO}/per_neuron_neuropil_count_pre_783.feather/content",
         size=16_853_770, algo="md5", digest="90fcdb42c1ba05ed92820840fa1e6ba0"),
    # -- Annotations (flyconnectome/flywire_annotations, tag v3.1.0) --------
    dict(name="Supplemental_file1_neuron_annotations.tsv",
         url=f"{GITHUB}/Supplemental_file1_neuron_annotations.tsv",
         size=31_718_505, algo="sha256", digest="9a4f8b2f843196074431ebd7cd883536afa1be86c8a4ce90970441e8be81d1be"),
    dict(name="Supplemental_file2_non_neuron_annotations.tsv",
         url=f"{GITHUB}/Supplemental_file2_non_neuron_annotations.tsv",
         size=127_761, algo="sha256", digest="2349fd5b4b9e53fd1b4a84aa73230bb7bebe66faa5d6a7490155aa2128861de1"),
    dict(name="Supplemental_file3_summary_with_ngl_links.csv",
         url=f"{GITHUB}/Supplemental_file3_summary_with_ngl_links.csv",
         size=758_670, algo="sha256", digest="8e9d92c0263c19863da3d537400bc2fe5ab6ed19e8327b0237e1e8a60ef9cfa5"),
    dict(name="Supplemental_file4_hemilineages_clustering.csv",
         url=f"{GITHUB}/Supplemental_file4_hemilineages_clustering.csv",
         size=2_315_552, algo="sha256", digest="daa1ebbede4ff3299d36d3224c3463f4b5a36116c057261d4f6e9a55ac4cbdc2"),
    dict(name="Supplemental_file5_hemibrain_meta.csv",
         url=f"{GITHUB}/Supplemental_file5_hemibrain_meta.csv",
         size=2_634_367, algo="sha256", digest="c9ca8421a5b6a5382348e06a7f62ef452bc78562983deff9ff479e30be2bd2a2"),
]


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


def _fmt_bytes(n):
    """Human-readable byte count, e.g. 1.23 GB."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0


def _fmt_eta(seconds):
    """Human-readable remaining time, e.g. 2m05s."""
    if seconds < 0 or seconds == float("inf"):
        return "--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class _DownloadUI:
    """Live download progress: an overall bar plus a row per manifest file.

    On a TTY, draws a block of lines and redraws it in place with ANSI escape
    codes. Every file is listed from the start: done files show 100% (marked
    "done"), the file actively downloading shows live speed/ETA, and the rest
    show 0% (marked "pending"), so the count of downloaded files is always
    visible. When stdout is not a TTY (piped, log capture), falls back to
    throttled one-line log messages.
    """
    def __init__(self, manifest, tty):
        self.order = [m["name"] for m in manifest]
        self.total = sum(m["size"] for m in manifest)
        self.name_w = max(len(m["name"]) for m in manifest)
        self.state = {m["name"]: {"total": m["size"], "done": 0,
                                  "start": None, "session": 0,
                                  "downloaded_this_run": False}
                      for m in manifest}
        self.start = time.monotonic()
        self.tty = tty
        self.rows = 0
        self.last_emit = self.start - 6

    def update(self, name, done, chunk):
        st = self.state[name]
        st["done"] = done
        st["session"] += chunk
        st["downloaded_this_run"] = True
        if st["start"] is None:
            st["start"] = time.monotonic()
        self._emit()

    def finish(self, name):
        st = self.state[name]
        st["done"] = st["total"]
        self._emit()
        if not self.tty:
            log(f"download: {name} OK ({_fmt_bytes(st['total'])})")

    def register(self, name, existing):
        """Record a file already present on disk (complete or a partial resume).

        Sets the done-amount so the file shows in the block immediately and
        counts toward the overall bar from the start, as though it had been
        downloaded in this session. `existing` is the on-disk size: a file at
        its target shows at 100%, a partial file shows its current amount until
        the download resumes and fills it in.
        """
        st = self.state[name]
        st["done"] = min(existing, st["total"])
        st["start"] = self.start
        st["session"] = 0
        st["downloaded_this_run"] = False

    def render(self):
        """Draw the current state without waiting for a download tick."""
        if self.tty:
            self._redraw_tty()

    # -- rendering -----------------------------------------------------
    def _emit(self):
        if self.tty:
            self._redraw_tty()
        else:
            self._emit_log()

    def _redraw_tty(self):
        rows = self._rows()
        # Each row: \r resets to column 0 (cursor-up alone keeps the column),
        # and \033[K erases any leftover text on that line from a longer frame.
        lines = ["\r" + r + "\033[K" for r in rows]
        # Move up rows-1 lines to reach the block's top row. After a draw the
        # cursor sits at the end of the LAST row, so rows-1 (not rows) lands on
        # the first row; overshooting would clobber text printed above the block
        # (e.g. the one-time TLS log line).
        up = f"\033[{self.rows - 1}A" if self.rows > 1 else ""
        buf = up + "\n".join(lines) + "\033[J"
        sys.stdout.write(buf)
        sys.stdout.flush()
        self.rows = len(rows)

    def _emit_log(self):
        now = time.monotonic()
        if now - self.last_emit < 5:
            return
        active = self._active()
        if active:
            name, st = active
            p = st["done"] / st["total"] * 100 if st["total"] else 0
            log(f"download: {name} {p:.1f}%  "
                f"{_fmt_bytes(st['done'])}/{_fmt_bytes(st['total'])}")
        else:
            done, _ = self.overall()
            log(f"download: overall {done / self.total * 100:.1f}%")
        self.last_emit = now

    def _active(self):
        for name in self.order:
            st = self.state[name]
            if st["start"] is not None and st["done"] < st["total"]:
                return name, st
        return None

    def overall(self):
        done = sum(self.state[n]["done"] for n in self.order)
        return done, self.total

    def _rows(self):
        rows = [self._overall_line()]
        # Done files first (already-downloaded files listed up top); within each
        # group keep manifest/download order via a stable sort.
        order = sorted(self.order,
                       key=lambda n: 0 if self.state[n]["done"] >= self.state[n]["total"] else 1)
        for name in order:
            rows.append(self._file_line(name))
        return rows

    def _overall_line(self):
        done, total = self.overall()
        pct = done / total * 100 if total else 0
        speed = self._overall_speed()
        eta = _fmt_eta((total - done) / speed) if speed > 0 else "--"
        bytes_col = f"{_fmt_bytes(done):>9} / {_fmt_bytes(total):>9}"
        return (f"  {'OVERALL':<{self.name_w}}  {pct:5.1f}% {self._bar(pct)} "
                f"{bytes_col}  {_fmt_bytes(speed)}/s  ETA {eta}")

    def _file_line(self, name):
        st = self.state[name]
        pct = st["done"] / st["total"] * 100 if st["total"] else 0
        bytes_col = f"{_fmt_bytes(st['done']):>9} / {_fmt_bytes(st['total']):>9}"
        line = (f"  {name:<{self.name_w}}  {pct:5.1f}% {self._bar(pct)} "
                f"{bytes_col}")
        if st["done"] >= st["total"]:
            line += "  (done)"
        elif st.get("downloaded_this_run"):
            # actively downloading this session -> live speed/ETA
            e = time.monotonic() - st["start"]
            speed = st["session"] / e if e > 0 else 0
            eta = _fmt_eta((st["total"] - st["done"]) / speed) if speed > 0 else "--"
            line += f"  {_fmt_bytes(speed)}/s  ETA {eta}"
        elif st["done"] > 0:
            line += "  (resume)"
        else:
            line += "  (pending)"
        return line

    def _overall_speed(self):
        sess = sum(self.state[n]["session"] for n in self.order)
        e = time.monotonic() - self.start
        return sess / e if e > 0 else 0

    @staticmethod
    def _bar(pct):
        filled = int(min(pct, 100.0) // 5)
        return "[" + "#" * filled + " " * (20 - filled) + "]"


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
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            log(f"download: {item['name']} attempt {attempt} failed: {exc}")
            time.sleep(5 * attempt)
    return False


def step_download(manifest, data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    # Warm the TLS context now so its one-time "using CA bundle" log lands
    # *before* the progress block is drawn, not inside it (which would corrupt
    # the in-place rendering).
    get_ssl_context()
    ui = _DownloadUI(manifest, sys.stdout.isatty())
    # Pre-register files already on disk so they show in the block (complete
    # files at 100%, partial files at their current amount) and count toward
    # the overall bar from the start.
    done = 0
    for item in manifest:
        p = data_dir / item["name"]
        if p.exists():
            ui.register(item["name"], p.stat().st_size)
            if p.stat().st_size == item["size"]:
                done += 1
    if done:
        log(f"download: {done}/{len(manifest)} files already complete")
    ui.render()
    ok = True
    for item in manifest:
        ok = download_one(item, data_dir, ui) and ok
    return ok


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
def step_verify(manifest, data_dir):
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
    parser = argparse.ArgumentParser(description="Download and verify the FlyWire v783 dataset (F1)")
    parser.add_argument("--step", choices=["download", "verify", "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)

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
