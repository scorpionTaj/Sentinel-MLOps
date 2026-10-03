from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request


def request(base_url: str, path: str, method: str = "GET") -> dict[str, object]:
    endpoint = f"{base_url.rstrip('/')}{path}"
    response = urllib.request.urlopen(
        urllib.request.Request(endpoint, method=method), timeout=15
    )
    return json.loads(response.read().decode("utf-8"))


def check(base_url: str) -> None:
    health = request(base_url, "/health")
    if health != {"status": "ok"}:
        raise AssertionError(f"unexpected health response: {health}")

    catalog = request(base_url, "/datasets")
    datasets = catalog.get("datasets")
    if catalog.get("baseline") != "FD001" or not isinstance(datasets, list):
        raise AssertionError(f"invalid dataset catalog: {catalog}")
    ids = [item["id"] for item in datasets]
    if len(ids) != len(set(ids)):
        raise AssertionError("dataset catalog contains duplicate ids")

    request(base_url, "/demo/reset", "POST")
    tested: list[str] = []
    try:
        for dataset in datasets:
            if not dataset.get("injectable"):
                continue
            scenario_id = str(dataset["id"])
            encoded = urllib.parse.quote(scenario_id)
            # Each regime is checked against the FD001 baseline; after a promotion the drift
            # reference legitimately moves to the promoted model's data.
            request(base_url, "/demo/reset", "POST")
            result = request(base_url, f"/simulate-drift?domain={encoded}", "POST")
            drift = result.get("drift")
            quality = result.get("quality")
            if not isinstance(drift, dict) or drift.get("detected") is not True:
                raise AssertionError(f"{scenario_id}: drift was not detected")
            if not isinstance(quality, dict) or quality.get("accepted") is not True:
                raise AssertionError(f"{scenario_id}: quality gate did not accept the batch")
            tested.append(scenario_id)

        try:
            request(base_url, "/simulate-drift?domain=UNKNOWN", "POST")
        except urllib.error.HTTPError as error:
            if error.code != 422:
                raise AssertionError(f"invalid scenario returned HTTP {error.code}") from error
        else:
            raise AssertionError("invalid scenario was accepted")
    finally:
        request(base_url, "/demo/reset", "POST")

    print(f"api scenario integration: pass ({len(tested)} regimes: {', '.join(tested)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify every Sentinel API drift scenario")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    arguments = parser.parse_args()
    check(arguments.base_url)
