"""The vendored front-end libraries are served, and only them."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402

client = TestClient(main.app)


@pytest.mark.parametrize("path", [
    "/static/vendor/marked/marked.min.js",
    "/static/vendor/dompurify/purify.min.js",
    "/static/vendor/katex/katex.min.js",
    "/static/vendor/katex/katex.min.css",
    "/static/vendor/katex/fonts/KaTeX_Main-Regular.woff2",
])
def test_vendored_files_are_served(path):
    r = client.get(path)
    assert r.status_code == 200 and len(r.content) > 1000


@pytest.mark.parametrize("path", [
    "/static/index.html", "/static/vendor/../index.html", "/static/vendor/%2e%2e/index.html",
    "/static/vendor/../../main.py",
])
def test_nothing_outside_vendor_is_reachable(path):
    r = client.get(path)
    assert r.status_code == 404
    assert b"<html" not in r.content.lower() and b"import" not in r.content


def test_page_loads_the_libraries_it_uses():
    html = client.get("/").text
    assert '<script src="/static/vendor/marked/marked.min.js"></script>' in html
    assert '<script src="/static/vendor/dompurify/purify.min.js"></script>' in html
    assert "katex.min.js" in html and '<script src="/static/vendor/katex' not in html   # KaTeX is lazy
