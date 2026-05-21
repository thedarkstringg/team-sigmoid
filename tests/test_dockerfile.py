from pathlib import Path


def test_dockerfile_exists_and_uses_python_slim():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.12-slim" in dockerfile
    assert "WORKDIR /app" in dockerfile
    assert "PYTHONPATH=/app" in dockerfile


def test_dockerfile_installs_requirements_and_copies_app_code():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "COPY requirements.txt" in dockerfile
    assert "pip install --no-cache-dir -r requirements.txt" in dockerfile
    assert "COPY ai ./ai" in dockerfile
    assert "COPY src ./src" in dockerfile


def test_dockerfile_exposes_api_port_and_starts_uvicorn():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "EXPOSE 8000" in dockerfile
    assert "uvicorn" in dockerfile
    assert "src.api:app" in dockerfile
    assert "--host" in dockerfile
    assert "0.0.0.0" in dockerfile


def test_dockerignore_protects_local_and_secret_files():
    dockerignore = Path(".dockerignore").read_text(encoding="utf-8")

    assert ".env" in dockerignore
    assert ".git" in dockerignore
    assert "__pycache__/" in dockerignore
    assert ".pytest_cache/" in dockerignore
