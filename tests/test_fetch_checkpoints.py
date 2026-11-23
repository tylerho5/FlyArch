"""fetch_checkpoints hands data/runs/<run_id>/best.pt to the scorers only when the bytes match checkpoints.sha256."""

import hashlib
import io

import pytest

from scripts import fetch_checkpoints as fc

GOOD = b"checkpoint bytes"


def _manifest(tmp_path, digest):
    path = tmp_path / "checkpoints.sha256"
    path.write_text(f"{digest}  checkpoints/f1-al-s000-607e8a0e/best.pt\n", encoding="utf-8")
    return path


def test_manifest_line_maps_to_run_dir_and_digest(tmp_path):
    digest = hashlib.sha256(GOOD).hexdigest()
    assert fc.read_manifest(_manifest(tmp_path, digest)) == [("f1-al-s000-607e8a0e", digest)]


@pytest.mark.parametrize("served,expect_ok", [(GOOD, True), (b"tampered", False)])
def test_fetch_writes_only_verified_bytes(tmp_path, served, expect_ok):
    digest = hashlib.sha256(GOOD).hexdigest()
    urls = []

    def opener(url):
        urls.append(url)
        return io.BytesIO(served)

    assert fc.fetch_one("f1-al-s000-607e8a0e", digest, tmp_path, opener=opener) is expect_ok
    dest = tmp_path / "f1-al-s000-607e8a0e" / "best.pt"
    assert dest.exists() is expect_ok
    assert not list(tmp_path.rglob("*.part"))
    assert urls[0].endswith("/files/f1-al-s000-607e8a0e.best.pt?download=1")


def test_fetch_skips_a_file_already_verified(tmp_path):
    digest = hashlib.sha256(GOOD).hexdigest()
    dest = tmp_path / "f1-al-s000-607e8a0e" / "best.pt"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(GOOD)

    def opener(url):
        raise AssertionError("no download expected")

    assert fc.fetch_one("f1-al-s000-607e8a0e", digest, tmp_path, opener=opener) is True


def test_unset_record_fails_before_any_download(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(fc, "zenodo_record_is_set", lambda: False)
    digest = hashlib.sha256(GOOD).hexdigest()
    assert fc.main(["--manifest", str(_manifest(tmp_path, digest)), "--out-dir", str(tmp_path / "runs")]) == 1
    assert "ZENODO_RECORD" in capsys.readouterr().out
    assert not (tmp_path / "runs").exists()
