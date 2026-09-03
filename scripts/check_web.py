from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from html.parser import HTMLParser


class AssetParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.paths: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "script" and attributes.get("src"):
            self.paths.append(attributes["src"] or "")
        if tag == "link" and attributes.get("href"):
            self.paths.append(attributes["href"] or "")


def check(base_url: str) -> None:
    base_url = base_url.rstrip("/") + "/"
    response = urllib.request.urlopen(base_url, timeout=5)
    html = response.read().decode()
    if response.status != 200 or "Sentinel Control Room" not in html:
        raise AssertionError("dashboard HTML did not load")

    parser = AssetParser()
    parser.feed(html)
    for path in parser.paths:
        asset_url = urllib.parse.urljoin(base_url, path)
        asset = urllib.request.urlopen(asset_url, timeout=5)
        if asset.status != 200 or not asset.read():
            raise AssertionError(f"asset did not load: {asset_url}")

    status = json.load(urllib.request.urlopen(urllib.parse.urljoin(base_url, "status"), timeout=5))
    if not status["bootstrapped"] or not status["aliases"].get("production"):
        raise AssertionError("pipeline status is not ready")
    print(f"web integration: pass ({len(parser.paths)} assets, production ready)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    arguments = parser.parse_args()
    check(arguments.base_url)
