"""Сторож опроса Telegram.

22.09.2026 бот пережил ночной обрыв VPN, но опрос getUpdates после него так и
не ожил: процесс работал, служба была active, а сообщения двое суток никто не
забирал. Ошибок в журнале не было — повторы по тайм-ауту библиотека пишет
только на уровне debug. Сторож замечает, что Telegram давно не отвечал, и
завершает процесс, чтобы systemd поднял бота заново.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.error import NetworkError

import src.bot.core as core_module
from src.bot.core import ClaudeCodeBot, PollingRequest
from src.config import create_test_config
from src.exceptions import ClaudeCodeTelegramError


@pytest.fixture
def polling_bot(monkeypatch):
    """Бот в режиме опроса с подменённым Application."""
    settings = create_test_config()
    bot = ClaudeCodeBot(settings, {"storage": MagicMock(), "security": MagicMock()})

    builder = MagicMock()
    app = MagicMock()
    app.bot = MagicMock()
    app.initialize = AsyncMock()
    app.start = AsyncMock()
    app.updater.start_polling = AsyncMock()
    builder.build.return_value = app

    monkeypatch.setattr(
        core_module.Application, "builder", MagicMock(return_value=builder)
    )
    monkeypatch.setattr(
        core_module, "FeatureRegistry", MagicMock(return_value=MagicMock())
    )
    monkeypatch.setattr(bot, "_set_bot_commands", AsyncMock())
    monkeypatch.setattr(bot, "_register_handlers", MagicMock())
    monkeypatch.setattr(bot, "_add_middleware", MagicMock())

    hard_exit = MagicMock()
    monkeypatch.setattr(core_module, "_schedule_hard_exit", hard_exit)

    return bot, app, builder, hard_exit


async def test_polling_request_remembers_last_answer(monkeypatch):
    """Ответ Telegram на getUpdates отмечается временем."""
    monkeypatch.setattr(
        core_module.HTTPXRequest,
        "do_request",
        AsyncMock(return_value=(200, b"{}")),
    )
    request = PollingRequest()
    request.last_answer = 0.0
    before = time.monotonic()

    assert await request.do_request("https://example.org", "POST") == (200, b"{}")

    assert request.last_answer >= before
    await request.shutdown()


async def test_polling_request_failure_keeps_last_answer(monkeypatch):
    """Ошибка сети — не ответ: время последнего ответа не сдвигается."""
    monkeypatch.setattr(
        core_module.HTTPXRequest,
        "do_request",
        AsyncMock(side_effect=NetworkError("proxy down")),
    )
    request = PollingRequest()
    request.last_answer = 0.0

    with pytest.raises(NetworkError):
        await request.do_request("https://example.org", "POST")

    assert request.last_answer == 0.0
    await request.shutdown()


async def test_initialize_polls_through_watched_request(polling_bot):
    """getUpdates ходит через клиента, за которым следит сторож."""
    bot, _app, builder, _hard_exit = polling_bot

    await bot.initialize()

    builder.get_updates_request.assert_called_once()
    request = builder.get_updates_request.call_args.args[0]
    assert isinstance(request, PollingRequest)
    assert request is bot.polling_request


async def test_real_builder_uses_watched_request_for_get_updates(monkeypatch):
    """Настоящий ApplicationBuilder принимает клиента, и getUpdates идёт через него.

    Библиотека запрещает смешивать готовый клиент с настройками get_updates_*
    — подменённый builder этого бы не заметил.
    """
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.2:2080")
    monkeypatch.setattr(core_module.Application, "initialize", AsyncMock())
    monkeypatch.setattr(
        core_module, "FeatureRegistry", MagicMock(return_value=MagicMock())
    )
    bot = ClaudeCodeBot(
        create_test_config(), {"storage": MagicMock(), "security": MagicMock()}
    )
    monkeypatch.setattr(bot, "_set_bot_commands", AsyncMock())
    monkeypatch.setattr(bot, "_register_handlers", MagicMock())
    monkeypatch.setattr(bot, "_add_middleware", MagicMock())

    await bot.initialize()

    get_updates_request = bot.app.bot._request[0]
    assert get_updates_request is bot.polling_request


async def test_start_exits_when_telegram_silent_too_long(polling_bot):
    """Telegram молчит дольше срока — start() падает, чтобы процесс перезапустили."""
    bot, app, _builder, hard_exit = polling_bot
    await bot.initialize()
    bot.polling_request.last_answer = (
        time.monotonic() - core_module.POLLING_STALL_SECONDS - 1
    )

    with pytest.raises(ClaudeCodeTelegramError):
        await asyncio.wait_for(bot.start(), timeout=5)

    app.updater.start_polling.assert_awaited_once()
    hard_exit.assert_called_once()
    assert bot.is_running is False


async def test_start_keeps_running_while_telegram_answers(polling_bot):
    """Пока Telegram отвечает, сторож бота не трогает."""
    bot, _app, _builder, hard_exit = polling_bot
    await bot.initialize()

    async def stop_soon():
        await asyncio.sleep(0.05)
        bot.is_running = False

    asyncio.create_task(stop_soon())
    await asyncio.wait_for(bot.start(), timeout=5)

    hard_exit.assert_not_called()


async def test_polling_keeps_messages_sent_while_bot_was_down(polling_bot):
    """После перезапуска бот забирает сообщения, пришедшие, пока он лежал.

    Иначе сторож перезапустил бы зависшего бота, а задачу, которую владелец
    отправил в эти минуты, тот молча выбросил бы.
    """
    bot, app, _builder, _hard_exit = polling_bot
    await bot.initialize()

    async def stop_soon():
        await asyncio.sleep(0.05)
        bot.is_running = False

    asyncio.create_task(stop_soon())
    await asyncio.wait_for(bot.start(), timeout=5)

    kwargs = app.updater.start_polling.call_args.kwargs
    assert kwargs["drop_pending_updates"] is False
