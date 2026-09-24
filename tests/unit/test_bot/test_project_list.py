"""Список «📂 Проекты» следует за папками рабочей зоны.

Новый проект из /newproject сразу становится текущим, удалённый — перестаёт
быть текущим, а устаревшие кнопки на него не ведут в пустоту.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.bot.orchestrator import MessageOrchestrator
from src.config import create_test_config


@pytest.fixture
def workspace():
    # Настройки хранят путь разрешённым (на macOS /var — ссылка на /private/var).
    with tempfile.TemporaryDirectory() as d:
        yield Path(d).resolve()


@pytest.fixture
def orchestrator(workspace):
    settings = create_test_config(approved_directory=str(workspace), agentic_mode=True)
    deps = {
        "claude_integration": MagicMock(),
        "storage": MagicMock(),
        "security_validator": MagicMock(),
        "rate_limiter": MagicMock(),
        "audit_logger": MagicMock(),
    }
    return MessageOrchestrator(settings, deps)


def _button_names(markup):
    return [button.text for row in markup.inline_keyboard for button in row]


def _message_update(text):
    update = MagicMock()
    update.effective_user.id = 123
    update.message.text = text
    update.message.reply_text = AsyncMock()
    return update


def _callback_update(data):
    update = MagicMock()
    update.callback_query.data = data
    update.callback_query.from_user.id = 123
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    return update


def _context(current_directory=None, claude_integration=None):
    context = MagicMock()
    context.user_data = {}
    if current_directory is not None:
        context.user_data["current_directory"] = current_directory
        context.user_data["claude_session_id"] = "old-session"
    context.bot_data = {
        "claude_integration": claude_integration,
        "rate_limiter": None,
        "audit_logger": None,
    }
    return context


async def test_newproject_opens_created_project(orchestrator, workspace):
    """Проект создан — бот сразу в нём, и он есть в списке."""

    async def create_folder(update, context, prompt_override=None):
        (workspace / "my-app").mkdir()

    orchestrator.agentic_text = AsyncMock(side_effect=create_folder)
    update = _message_update("/newproject my-app")
    context = _context()

    await orchestrator.agentic_newproject(update, context)

    assert context.user_data["current_directory"] == workspace / "my-app"
    assert context.user_data["claude_session_id"] is None
    last_text = update.message.reply_text.call_args.args[0]
    assert "my-app" in last_text
    names = _button_names(orchestrator._repos_keyboard())
    assert any(name.endswith("my-app") for name in names)


async def test_newproject_stays_put_when_folder_not_created(orchestrator, workspace):
    """Создать не вышло — бот не делает вид, что проект открыт."""
    orchestrator.agentic_text = AsyncMock()
    update = _message_update("/newproject my-app")
    context = _context()

    await orchestrator.agentic_newproject(update, context)

    assert context.user_data["current_directory"] == workspace
    texts = [c.args[0] for c in update.message.reply_text.call_args_list]
    assert not any("Открыл" in t for t in texts)


async def test_task_in_deleted_project_offers_project_list(orchestrator, workspace):
    """Папку текущего проекта удалили — задача не уходит в пустоту, бот предлагает список."""
    (workspace / "alive").mkdir()
    claude_integration = AsyncMock()
    update = _message_update("Поправь README")
    context = _context(workspace / "gone", claude_integration)

    await orchestrator.agentic_text(update, context)

    claude_integration.run_command.assert_not_called()
    assert context.user_data["current_directory"] == workspace
    assert context.user_data["claude_session_id"] is None
    reply = update.message.reply_text.call_args
    assert "gone" in reply.args[0]
    names = _button_names(reply.kwargs["reply_markup"])
    assert any(name.endswith("alive") for name in names)
    assert not any(name.endswith("gone") for name in names)


async def test_home_screen_forgets_deleted_project(orchestrator, workspace):
    """Главный экран не показывает удалённый проект текущим."""
    update = _callback_update("ui:home")
    context = _context(workspace / "gone")

    await orchestrator._handle_ui_callback(update, context)

    text = update.callback_query.edit_message_text.call_args.args[0]
    assert "не выбран" in text
    assert "gone" not in text
    assert context.user_data["current_directory"] == workspace


async def test_old_button_of_deleted_project_shows_fresh_list(orchestrator, workspace):
    """Кнопка из старого списка ведёт на удалённый проект — бот показывает свежий список."""
    (workspace / "alive").mkdir()
    update = _callback_update("cd:gone")
    context = _context()

    await orchestrator._agentic_callback(update, context)

    call = update.callback_query.edit_message_text.call_args
    assert "gone" in call.args[0]
    assert "Directory not found" not in call.args[0]
    names = _button_names(call.kwargs["reply_markup"])
    assert any(name.endswith("alive") for name in names)
    assert "current_directory" not in context.user_data
