"""Tests for ProjectSync: how the bot runs the sync script around a task."""

import os
import stat
from pathlib import Path

import pytest

from src.bot.features import project_sync as ps
from src.bot.features.project_sync import ProjectSync
from src.config import create_test_config


def _script(tmp_path: Path, body: str) -> str:
    path = tmp_path / "fake-sync.sh"
    path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    (ws / "proj").mkdir(parents=True)
    return ws


def _sync(workspace: Path, script: str | None) -> ProjectSync:
    return ProjectSync(
        create_test_config(
            approved_directory=str(workspace), project_sync_script=script
        )
    )


async def test_disabled_when_no_script(workspace):
    sync = _sync(workspace, None)
    assert not sync.enabled
    assert await sync.pull(workspace / "proj") == ""
    assert await sync.push(workspace / "proj") == ""


async def test_passes_command_dir_and_device(tmp_path, workspace):
    script = _script(tmp_path, 'echo "$1|$2|$SYNC_DEVICE|$PWD"\n')
    out = await _sync(workspace, script).pull(workspace / "proj")
    cmd, path, device, cwd = out.split("|")
    assert cmd == "pull"
    assert Path(path) == (workspace / "proj").resolve()
    assert device == "бот"
    assert Path(cwd) == (workspace / "proj").resolve()


async def test_push_returns_script_output(tmp_path, workspace):
    script = _script(tmp_path, 'echo "☁️ Отправлено на GitHub: 2 файла"\n')
    out = await _sync(workspace, script).push(workspace / "proj")
    assert out == "☁️ Отправлено на GitHub: 2 файла"


async def test_silent_output_stays_empty(tmp_path, workspace):
    script = _script(tmp_path, "exit 0\n")
    assert await _sync(workspace, script).push(workspace / "proj") == ""


async def test_skips_workspace_root(tmp_path, workspace):
    script = _script(tmp_path, 'echo "должно быть тихо"\n')
    assert await _sync(workspace, script).pull(workspace) == ""


async def test_skips_paths_outside_workspace(tmp_path, workspace):
    script = _script(tmp_path, 'echo "должно быть тихо"\n')
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    assert await _sync(workspace, script).push(outside) == ""


async def test_nonzero_exit_without_output_gives_warning(tmp_path, workspace):
    script = _script(tmp_path, "exit 3\n")
    out = await _sync(workspace, script).push(workspace / "proj")
    assert out.startswith("⚠️")


async def test_nonzero_exit_keeps_script_message(tmp_path, workspace):
    script = _script(tmp_path, 'echo "⚠️ GitHub недоступен"; exit 1\n')
    out = await _sync(workspace, script).push(workspace / "proj")
    assert out == "⚠️ GitHub недоступен"


async def test_missing_script_is_silent(workspace):
    sync = _sync(workspace, "/nonexistent/sync-project")
    assert await sync.pull(workspace / "proj") == ""


async def test_timeout_gives_warning(tmp_path, workspace, monkeypatch):
    monkeypatch.setattr(ps, "SYNC_TIMEOUT_SECONDS", 0.2)
    script = _script(tmp_path, "sleep 5\n")
    out = await _sync(workspace, script).push(workspace / "proj")
    assert "не ответила вовремя" in out
