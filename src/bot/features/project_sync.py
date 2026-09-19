"""Синхронизация проекта с GitHub вокруг задачи.

Бот работает в клонах репозиториев. Чтобы правки с телефона доезжали до
компьютера, а правки с компьютера — до бота, перед каждой задачей проект
подтягивается с GitHub, а после — сохраняется и отправляется. Саму работу
делает внешний скрипт (``tools/sync-project.sh`` в проекте бота): он один и
тот же на сервере и на Mac, здесь только его запуск.

Скрипт молчит, когда делать нечего, и говорит по-русски, когда что-то
сделал или не смог. Его вывод отправляется в чат как есть.
"""

import asyncio
import os
from pathlib import Path
from typing import Optional, Union

import structlog

from ...config.settings import Settings

logger = structlog.get_logger()

# Сеть может быть медленной, но бесконечно ждать нельзя: задача уже ждёт.
SYNC_TIMEOUT_SECONDS = 120


class ProjectSync:
    """Запуск скрипта синхронизации для папки проекта."""

    def __init__(self, settings: Settings):
        # getattr: в тестах настройки бывают заглушками без этого поля.
        script = getattr(settings, "project_sync_script", None)
        self.script: Optional[str] = script if isinstance(script, str) and script else None
        try:
            self.approved_dir = Path(str(settings.approved_directory)).resolve()
        except Exception:
            self.approved_dir = Path("/nonexistent")
            self.script = None

    @property
    def enabled(self) -> bool:
        return bool(self.script)

    async def pull(self, path: Union[str, Path]) -> str:
        """Забрать новое с GitHub. Возвращает текст для чата или пустую строку."""
        return await self._run("pull", path)

    async def push(self, path: Union[str, Path]) -> str:
        """Сохранить и отправить правки. Возвращает текст для чата или пустую строку."""
        return await self._run("push", path)

    async def _run(self, command: str, path: Union[str, Path]) -> str:
        if not self.script:
            return ""

        try:
            project = Path(path).resolve()
            if not project.is_relative_to(self.approved_dir):
                return ""
        except Exception:
            return ""
        # Корень рабочей зоны — это не проект, а папка с проектами.
        if project == self.approved_dir:
            return ""

        env = {**os.environ, "SYNC_DEVICE": "бот"}
        try:
            process = await asyncio.create_subprocess_exec(
                self.script,
                command,
                str(project),
                cwd=str(project),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError:
            logger.warning("Project sync script not found", script=self.script)
            return ""
        except Exception as e:
            logger.warning("Project sync failed to start", error=str(e))
            return "⚠️ Синхронизация с GitHub не запустилась"

        try:
            stdout, _ = await asyncio.wait_for(
                process.communicate(), timeout=SYNC_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            process.kill()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except Exception:
                pass
            logger.warning("Project sync timed out", command=command, path=str(project))
            return "⚠️ Синхронизация с GitHub не ответила вовремя — попробую после следующей задачи"

        text = stdout.decode("utf-8", errors="replace").strip()
        if process.returncode != 0:
            logger.warning(
                "Project sync exited with error",
                command=command,
                code=process.returncode,
                output=text[:500],
            )
            return text or "⚠️ Синхронизация с GitHub не удалась"
        return text
