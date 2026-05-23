import base64
import importlib
import os
import sys
import types
from types import SimpleNamespace

from fastapi.testclient import TestClient


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def import_api_with_fake_analyzer(monkeypatch):
    fake_analyzer = types.ModuleType("src.core.analyzer")

    async def fake_run_analysis(image_path: str):
        return SimpleNamespace(image_path=image_path)

    fake_analyzer.run_analysis = fake_run_analysis
    monkeypatch.setitem(sys.modules, "src.core.analyzer", fake_analyzer)
    sys.modules.pop("src.api", None)

    return importlib.import_module("src.api")


def test_health_endpoint_returns_ok(monkeypatch):
    api = import_api_with_fake_analyzer(monkeypatch)
    client = TestClient(api.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_rejects_invalid_content_type(monkeypatch):
    api = import_api_with_fake_analyzer(monkeypatch)
    client = TestClient(api.app)

    response = client.post(
        "/analyze",
        files={"image": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 422
    assert "Invalid file type" in response.json()["detail"]


def test_analyze_rejects_empty_file(monkeypatch):
    api = import_api_with_fake_analyzer(monkeypatch)
    client = TestClient(api.app)

    response = client.post(
        "/analyze",
        files={"image": ("empty.png", b"", "image/png")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Uploaded file is empty."


def test_analyze_rejects_file_that_exceeds_size_limit(monkeypatch):
    api = import_api_with_fake_analyzer(monkeypatch)
    monkeypatch.setattr(api.settings, "max_image_bytes", 10)
    client = TestClient(api.app)

    response = client.post(
        "/analyze",
        files={"image": ("large.png", PNG_BYTES, "image/png")},
    )

    assert response.status_code == 413
    assert "File too large" in response.json()["detail"]


def test_analyze_rejects_invalid_image_bytes(monkeypatch):
    api = import_api_with_fake_analyzer(monkeypatch)
    client = TestClient(api.app)

    response = client.post(
        "/analyze",
        files={"image": ("fake.png", b"not really an image", "image/png")},
    )

    assert response.status_code == 422
    assert "File content is not a valid JPEG or PNG" in response.json()["detail"]


def test_analyze_returns_500_when_analysis_fails(monkeypatch):
    api = import_api_with_fake_analyzer(monkeypatch)

    async def fake_run_analysis(image_path: str):
        raise RuntimeError("simulated analysis crash")

    monkeypatch.setattr(api, "run_analysis", fake_run_analysis)

    client = TestClient(api.app)
    response = client.post(
        "/analyze",
        files={"image": ("food.png", PNG_BYTES, "image/png")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Analysis failed. Please try again."

def test_history_endpoint_returns_serialized_records(monkeypatch):
    api = import_api_with_fake_analyzer(monkeypatch)

    class SavedRecord:
        def model_dump(self, mode="json"):
            return {"id": 1, "image_path": "data/rice.png"}

    class FakeRepo:
        async def list_all(self, limit: int = 20):
            return [SavedRecord()]

    monkeypatch.setattr(api, "repo", FakeRepo())

    client = TestClient(api.app)
    response = client.get("/history?limit=1")

    assert response.status_code == 200
    assert response.json() == [{"id": 1, "image_path": "data/rice.png"}]
