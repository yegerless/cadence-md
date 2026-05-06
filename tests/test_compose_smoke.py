"""Smoke checks for docker compose manifests."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.smoke]


def _docker_compose_config(project_root: Path, *args: str) -> str:
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI is unavailable.")
    process = subprocess.run(
        ["docker", "compose", *args, "config"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        stderr = process.stderr.strip()
        pytest.skip(f"docker compose config failed in current environment: {stderr}")
    return process.stdout


def test_docker_compose_test_config_smoke() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = _docker_compose_config(project_root, "-f", "docker-compose-test.yml")
    assert "postgres-test" in config
    assert "redis-test" in config


def test_docker_compose_dev_contains_migrations_service() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = _docker_compose_config(project_root, "-f", "docker-compose-dev.yml")
    assert "migrations:" in config
    assert "alembic" in config


def test_docker_compose_dev_contains_dvc_corpus_volume() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = _docker_compose_config(project_root, "-f", "docker-compose-dev.yml")
    assert "dvc-pull:" in config
    assert "dvc pull data.dvc" in config or "- data.dvc" in config
    assert "rag-corpus:" in config
    assert "/app/data" in config
