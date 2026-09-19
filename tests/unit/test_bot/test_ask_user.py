"""Вопрос с вариантами: кнопки в Telegram вместо неработающего AskUserQuestion."""

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.bot.features.ask_user import AskUserChannel, build_ask_server


def make_channel(timeout: float = 5.0) -> AskUserChannel:
    bot = AsyncMock()
    bot.send_message = AsyncMock(return_value=AsyncMock())
    return AskUserChannel(bot=bot, chat_id=42, timeout=timeout)


async def test_button_answer_returns_choice():
    """Нажатая кнопка возвращается модели как выбор пользователя."""
    channel = make_channel()

    asking = asyncio.create_task(channel.ask("Какой смайлик?", ["✨", "🚀"]))
    await asyncio.sleep(0)  # дать вопросу отправиться

    sent = channel.bot.send_message.await_args
    assert "Какой смайлик?" in sent.kwargs["text"]
    rows = sent.kwargs["reply_markup"].inline_keyboard
    # Два варианта плюс «свой вариант»
    assert len(rows) == 3
    key = rows[0][0].callback_data.split(":")[1]

    assert await channel.answer_button(key, "1") == "🚀"
    assert await asking == "Пользователь выбрал: 🚀"
    assert channel.waiting is False


async def test_free_text_answer():
    """«Свой вариант» ждёт обычное сообщение и отдаёт его модели."""
    channel = make_channel()

    asking = asyncio.create_task(channel.ask("Как назвать?", ["Вариант А"]))
    await asyncio.sleep(0)

    rows = channel.bot.send_message.await_args.kwargs["reply_markup"].inline_keyboard
    key = rows[0][0].callback_data.split(":")[1]

    await channel.answer_button(key, "free")
    assert await channel.answer_text("Назови «Вайб»") is True
    assert await asking == "Пользователь выбрал: Назови «Вайб»"


async def test_text_without_question_is_not_an_answer():
    """Без заданного вопроса обычный текст остаётся задачей, а не ответом."""
    channel = make_channel()
    assert channel.waiting is False
    assert await channel.answer_text("сделай рефакторинг") is False


async def test_timeout_lets_claude_continue():
    """Если ответа нет, задача не висит вечно."""
    channel = make_channel(timeout=0.05)
    answer = await channel.ask("Ждём?", ["да"])
    assert "не ответил" in answer
    assert channel.waiting is False


async def test_stale_button_is_ignored():
    """Нажатие на старый вопрос не роняет бота."""
    channel = make_channel()
    assert await channel.answer_button("несуществующий", "0") is None


async def test_server_exposes_single_tool():
    """MCP-сервер задачи отдаёт ровно один инструмент — вопрос пользователю."""
    server = build_ask_server(make_channel())
    assert server["type"] == "sdk"
    assert server["name"] == "telegram"
