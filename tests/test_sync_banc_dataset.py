"""BANC downloads: the synapse table from Dataverse; the unhosted pinned files stop with the current release's URLs."""

import io
import urllib.error

from scripts import sync_banc_dataset as sync


def test_download_stops_on_404_and_names_the_url(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(req.full_url)
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, io.BytesIO())

    monkeypatch.setattr(sync, "_urlopen_retry", fake_urlopen)
    monkeypatch.setattr(sync.time, "sleep", lambda s: None)
    item = dict(name="f.feather", url="https://example.test/f.feather", size=10)

    assert sync.download_one(item, tmp_path, sync._NullUI()) is False
    assert len(calls) == 1
    assert item["url"] in capsys.readouterr().out


def test_only_the_synapse_table_has_a_download_url():
    urls = {m["name"]: m["url"] for m in sync.MANIFEST}
    assert urls.pop("banc_888_synapses_v2_enriched.parquet") == "https://dataverse.harvard.edu/api/access/datafile/13992863"
    assert len(urls) == 3 and all(url is None for url in urls.values())


def test_unhosted_files_stop_with_current_release_urls(tmp_path, monkeypatch, capsys):
    fetched = []
    monkeypatch.setattr(sync, "get_ssl_context", lambda: None)
    monkeypatch.setattr(sync, "download_one", lambda item, d, ui: fetched.append(item["name"]) or True)
    (tmp_path / "banc_888_meta.feather").write_bytes(b"placed by hand")

    assert sync.step_download(sync.MANIFEST, tmp_path) is False
    out = capsys.readouterr().out
    assert fetched == ["banc_888_synapses_v2_enriched.parquet"]
    assert f"{sync.BUCKET}/neuron_connectivity/v888/banc_888_edgelist_neuropil_v2.feather" in out
    assert "banc_888_meta.feather:" not in out
