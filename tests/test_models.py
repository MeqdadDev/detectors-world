import hashlib
import http.client
import io
import shutil
import urllib.error

import pytest

from detectors_world import Hand, _models, cache_dir

NAME = "hand_landmarker.task"
GOOD = b"good model"


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


@pytest.fixture
def model_store(monkeypatch, tmp_path):
    """Use an empty cache and a fake server that serves GOOD, returning the requested URLs."""
    monkeypatch.setenv("DETECTORS_WORLD_CACHE", str(tmp_path))
    monkeypatch.setitem(_models.MODELS, NAME, ("some/path", hashlib.sha256(GOOD).hexdigest()))
    requests = []

    def urlopen(url, timeout):
        requests.append(url)
        return FakeResponse(GOOD)

    monkeypatch.setattr(_models.urllib.request, "urlopen", urlopen)
    return requests


def test_download_is_cached(model_store, tmp_path):
    assert _models.load_model(NAME) == GOOD
    assert _models.load_model(NAME) == GOOD
    assert len(model_store) == 1
    assert (tmp_path / NAME).read_bytes() == GOOD


def test_damaged_cached_file_is_replaced(model_store, tmp_path):
    (tmp_path / NAME).write_bytes(b"truncated")
    assert _models.load_model(NAME) == GOOD
    assert (tmp_path / NAME).read_bytes() == GOOD


def test_damaged_download_is_not_cached(model_store, monkeypatch, tmp_path):
    monkeypatch.setattr(_models.urllib.request, "urlopen", lambda url, timeout: FakeResponse(b"bad"))
    with pytest.raises(RuntimeError, match="damaged or incomplete"):
        _models.load_model(NAME)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "error",
    [urllib.error.URLError("offline"), http.client.IncompleteRead(b"partial")],
)
def test_failed_download_explains_the_manual_route(model_store, monkeypatch, tmp_path, error):
    def urlopen(url, timeout):
        raise error

    monkeypatch.setattr(_models.urllib.request, "urlopen", urlopen)
    with pytest.raises(RuntimeError, match="model_path="):
        _models.load_model(NAME)
    assert list(tmp_path.iterdir()) == []


def test_failed_cache_write_leaves_no_partial_file(model_store, monkeypatch, tmp_path):
    def replace(source, target):
        raise KeyboardInterrupt

    monkeypatch.setattr(_models.os, "replace", replace)
    with pytest.raises(KeyboardInterrupt):
        _models.load_model(NAME)
    assert list(tmp_path.iterdir()) == []


def test_model_urls_are_pinned():
    assert all("/latest/" not in path and len(digest) == 64 for path, digest in _models.MODELS.values())


def test_model_in_a_folder_with_non_ascii_name(image, tmp_path):
    Hand().close()
    folder = tmp_path / "مستخدم"
    folder.mkdir()
    shutil.copy(cache_dir() / NAME, folder / NAME)
    with Hand(static_image=True, model_path=folder / NAME) as hand:
        assert len(hand.detect(image("hands"), draw=False)) == 2


def test_cache_in_a_folder_with_non_ascii_name(image, monkeypatch, tmp_path):
    Hand().close()
    folder = tmp_path / "مستخدم"
    folder.mkdir()
    shutil.copy(cache_dir() / NAME, folder / NAME)
    monkeypatch.setenv("DETECTORS_WORLD_CACHE", str(folder))
    with Hand(static_image=True) as hand:
        assert len(hand.detect(image("hands"), draw=False)) == 2
