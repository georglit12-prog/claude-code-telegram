"""Подтверждение системных действий кнопкой в Telegram.

Бот работает с правами root, поэтому перед системным действием он обязан
спросить владельца и дождаться ответа. Отказ и молчание означают «не делать».
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

from src.claude.sdk_integration import _make_can_use_tool_callback

APPROVED = Path("/srv/claude-bot/workspace")
CWD = APPROVED / "myproject"


def _callback(ask_channel=None, validator=None):
    return _make_can_use_tool_callback(
        security_validator=validator,
        working_directory=CWD,
        approved_directory=APPROVED,
        ask_channel=ask_channel,
    )


def _channel(answer: str) -> MagicMock:
    channel = MagicMock()
    channel.ask = AsyncMock(return_value=answer)
    return channel


class TestOrdinaryWorkIsNotInterrupted:
    """Обычная работа не должна дёргать владельца вопросами."""

    async def test_plain_command_does_not_ask(self) -> None:
        channel = _channel("Разрешить")
        result = await _callback(channel)("Bash", {"command": "pytest -q"}, None)
        assert isinstance(result, PermissionResultAllow)
        channel.ask.assert_not_awaited()

    async def test_write_inside_project_does_not_ask(self) -> None:
        channel = _channel("Разрешить")
        result = await _callback(channel)(
            "Write", {"file_path": str(CWD / "app.py")}, None
        )
        assert isinstance(result, PermissionResultAllow)
        channel.ask.assert_not_awaited()

    async def test_reading_outside_does_not_ask(self) -> None:
        """Чтение чужого конфига безопасно — вопроса быть не должно."""
        channel = _channel("Разрешить")
        result = await _callback(channel)(
            "Read", {"file_path": "/etc/nginx/nginx.conf"}, None
        )
        assert isinstance(result, PermissionResultAllow)
        channel.ask.assert_not_awaited()


class TestSystemActionsAsk:
    """Системные действия проходят только через вопрос владельцу."""

    async def test_systemctl_asks_and_allows_on_yes(self) -> None:
        channel = _channel("Разрешить")
        result = await _callback(channel)(
            "Bash", {"command": "systemctl restart nginx"}, None
        )
        assert isinstance(result, PermissionResultAllow)
        channel.ask.assert_awaited_once()
        question = channel.ask.call_args.args[0]
        assert "systemctl restart nginx" in question

    async def test_systemctl_blocked_on_no(self) -> None:
        channel = _channel("Отказать")
        result = await _callback(channel)(
            "Bash", {"command": "systemctl restart nginx"}, None
        )
        assert isinstance(result, PermissionResultDeny)

    async def test_write_outside_workspace_asks(self) -> None:
        channel = _channel("Разрешить")
        result = await _callback(channel)(
            "Write", {"file_path": "/etc/nginx/sites-available/new"}, None
        )
        assert isinstance(result, PermissionResultAllow)
        channel.ask.assert_awaited_once()

    async def test_write_outside_workspace_blocked_on_no(self) -> None:
        channel = _channel("Отказать")
        result = await _callback(channel)(
            "Write", {"file_path": "/srv/site/index.html"}, None
        )
        assert isinstance(result, PermissionResultDeny)

    async def test_denial_message_tells_claude_not_to_retry(self) -> None:
        """Отказ должен останавливать, а не запускать поиск обходного пути."""
        channel = _channel("Отказать")
        result = await _callback(channel)("Bash", {"command": "apt install nginx"}, None)
        assert isinstance(result, PermissionResultDeny)
        assert "не повторяй" in result.message.lower()
        assert "обходных путей" in result.message.lower()


class TestSilenceMeansNo:
    """Нет ответа — значит нет: молча менять систему нельзя."""

    async def test_timeout_answer_is_treated_as_refusal(self) -> None:
        channel = _channel(
            "Пользователь не ответил за отведённое время. Выбери сам "
            "разумный вариант, скажи в ответе, какой и почему."
        )
        result = await _callback(channel)(
            "Bash", {"command": "systemctl restart nginx"}, None
        )
        assert isinstance(result, PermissionResultDeny)

    async def test_broken_channel_is_treated_as_refusal(self) -> None:
        channel = MagicMock()
        channel.ask = AsyncMock(side_effect=RuntimeError("канал закрыт"))
        result = await _callback(channel)(
            "Bash", {"command": "systemctl restart nginx"}, None
        )
        assert isinstance(result, PermissionResultDeny)


class TestWithoutChannel:
    """Без канала вопросов подтверждения не спрашиваются (старое поведение)."""

    async def test_no_channel_allows_system_command(self) -> None:
        result = await _callback(None)(
            "Bash", {"command": "systemctl restart nginx"}, None
        )
        assert isinstance(result, PermissionResultAllow)
