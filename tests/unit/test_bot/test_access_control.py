"""Доступ к боту: кнопки и чаты.

У бота права root, поэтому «кто может им пользоваться» — это «кто может
администрировать сервер». Проверка личности не должна держаться на побочных
эффектах: она проверяется явно и для сообщений, и для нажатий кнопок.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.bot.middleware.auth import callback_auth_middleware


OWNER = 1932944793
STRANGER = 55555555


def _callback_update(user_id: int, chat_id: int = OWNER, chat_type: str = "private"):
    update = MagicMock()
    update.callback_query = MagicMock()
    update.callback_query.from_user.id = user_id
    update.callback_query.data = "ask:abc:0"
    update.callback_query.answer = AsyncMock()
    update.effective_user.id = user_id
    update.effective_chat.id = chat_id
    update.effective_chat.type = chat_type
    update.effective_message = None
    return update


def _settings(allowed_users=None, allowed_chats=None):
    settings = MagicMock()
    settings.allowed_users = allowed_users if allowed_users is not None else [OWNER]
    settings.allowed_chat_ids = allowed_chats
    return settings


class TestCallbackButtonsCheckIdentity:
    """Нажатие кнопки — такое же действие, как сообщение, и проверяется так же."""

    async def test_owner_press_passes(self) -> None:
        handler = AsyncMock(return_value="выполнено")
        update = _callback_update(OWNER)
        data = {"settings": _settings()}

        result = await callback_auth_middleware(handler, update, data)

        handler.assert_awaited_once()
        assert result == "выполнено"

    async def test_stranger_press_is_blocked(self) -> None:
        handler = AsyncMock()
        update = _callback_update(STRANGER)
        data = {"settings": _settings()}

        await callback_auth_middleware(handler, update, data)

        handler.assert_not_awaited()
        update.callback_query.answer.assert_awaited_once()

    async def test_stranger_is_told_why(self) -> None:
        """Отказ должен быть понятным, а не молчаливым зависанием кнопки."""
        handler = AsyncMock()
        update = _callback_update(STRANGER)

        await callback_auth_middleware(handler, update, {"settings": _settings()})

        text = update.callback_query.answer.call_args.args[0]
        assert "нет доступа" in text.lower()

    async def test_no_settings_blocks_rather_than_opens(self) -> None:
        """Сломанная конфигурация должна закрывать доступ, а не открывать."""
        handler = AsyncMock()
        update = _callback_update(OWNER)

        await callback_auth_middleware(handler, update, {})

        handler.assert_not_awaited()

    async def test_empty_allowed_users_blocks_everyone(self) -> None:
        """Пустой список — это «никому», а не «всем»."""
        handler = AsyncMock()
        update = _callback_update(OWNER)

        await callback_auth_middleware(
            handler, update, {"settings": _settings(allowed_users=[])}
        )

        handler.assert_not_awaited()


class TestChatRestriction:
    """Второй уровень: бот отвечает только в разрешённых чатах."""

    async def test_owner_in_allowed_chat_passes(self) -> None:
        handler = AsyncMock()
        update = _callback_update(OWNER, chat_id=OWNER)
        data = {"settings": _settings(allowed_chats=[OWNER])}

        await callback_auth_middleware(handler, update, data)

        handler.assert_awaited_once()

    async def test_owner_in_foreign_group_is_blocked(self) -> None:
        """Токен украли и добавили бота в чужую группу — там он молчит.

        Даже если сам владелец что-то там нажмёт: чат не тот.
        """
        handler = AsyncMock()
        update = _callback_update(OWNER, chat_id=-100999, chat_type="supergroup")
        data = {"settings": _settings(allowed_chats=[OWNER])}

        await callback_auth_middleware(handler, update, data)

        handler.assert_not_awaited()

    async def test_no_chat_restriction_configured_allows_any_chat(self) -> None:
        """Ограничение по чатам необязательно: не задано — не проверяем."""
        handler = AsyncMock()
        update = _callback_update(OWNER, chat_id=-100999, chat_type="supergroup")
        data = {"settings": _settings(allowed_chats=None)}

        await callback_auth_middleware(handler, update, data)

        handler.assert_awaited_once()


class TestMessageMiddlewareAlsoChecksChat:
    """Та же проверка чата применяется и к обычным сообщениям."""

    @pytest.mark.parametrize(
        "chat_id,should_pass",
        [(OWNER, True), (-100999, False)],
    )
    async def test_chat_restriction_on_messages(
        self, chat_id: int, should_pass: bool
    ) -> None:
        from src.bot.middleware.auth import chat_is_allowed

        settings = _settings(allowed_chats=[OWNER])
        assert chat_is_allowed(settings, chat_id) is should_pass
