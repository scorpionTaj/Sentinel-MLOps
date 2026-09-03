from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient

    from sentinel.api import create_app
except ImportError:
    TestClient = None
    create_app = None


@unittest.skipIf(TestClient is None, "FastAPI test dependencies are not installed")
class ApiIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.client = TestClient(create_app(Path(self.temporary.name)))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_dashboard_and_every_declared_asset_load(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        paths = re.findall(r'(?:src|href)="([^"]+)"', response.text)
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path):
                asset = self.client.get(path)
                self.assertEqual(asset.status_code, 200)
                self.assertTrue(asset.content)

    def test_status_is_bootstrapped(self) -> None:
        status = self.client.get("/status")
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["aliases"], {"production": 1})

    def test_dataset_parameter_is_validated(self) -> None:
        self.assertEqual(self.client.post("/simulate-drift?domain=SENSOR_BIAS").status_code, 200)
        self.assertEqual(self.client.post("/simulate-drift?domain=UNKNOWN").status_code, 422)

    def test_dataset_catalog_is_discoverable(self) -> None:
        response = self.client.get("/datasets")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        ids = {item["id"] for item in payload["datasets"]}
        self.assertEqual(payload["baseline"], "FD001")
        self.assertTrue({"FD002", "FD003", "FD004", "SENSOR_BIAS", "SENSOR_DROPOUT"} <= ids)

    def test_every_injectable_dataset_can_run_through_the_api(self) -> None:
        datasets = self.client.get("/datasets").json()["datasets"]
        for dataset in datasets:
            if not dataset["injectable"]:
                continue
            with self.subTest(dataset=dataset["id"]):
                with tempfile.TemporaryDirectory() as temporary:
                    client = TestClient(create_app(Path(temporary)))
                    response = client.post(f"/simulate-drift?domain={dataset['id']}")
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.json()["drift"]["detected"])
