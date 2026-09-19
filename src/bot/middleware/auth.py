"""Telegram bot authentication middleware."""

from datetime import UTC, datetime
from typing import Any, Callable, Dict, Optional

import structlog

logger = structlog.get_logger()


async def auth_middleware(handler: Callable, event: Any, data: Dict[str, Any]) -> Any:
    """Check authentication before processing messages.

    This middleware:
    1. Checks if user is authenticated
    2. Attempts authentication if not authenticated
    3. Updates session activity
    4. Logs authentication events
    """
    # Extract user information
    user_id = event.effective_user.id if event.effective_user else None
    username = (
        getattr(event.effective_user, "username", None)
        if event.effective_user
        else None
    )

    if not user_id:
        logger.warning("No user information in update")
        return

    # Get dependencies from context
    auth_manager = data.get("auth_manager")
    audit_logger = data.get("audit_logger")

    # Чат тоже проверяется: с украденным токеном бота могут добавить в чужую
    # группу. Молчим там совсем — отвечать «нет доступа» значит подтвердить,
    # что бот жив и чего-то стоит.
    chat = getattr(event, "effective_chat", None)
    if not chat_is_allowed(data.get("settings"), chat.id if chat else None):
        logger.warning(
            "Message from a chat that is not allowed",
            user_id=user_id,
            chat_id=chat.id if chat else None,
        )
        return

    if not auth_manager:
        logger.error("Authentication manager not available in middleware context")
        if event.effective_message:
            await event.effective_message.reply_text(
                "🔒 Authentication system unavailable. Please try again later."
            )
        return

    # Check if user is already authenticated
    if auth_manager.is_authenticated(user_id):
        # Update session activity
        if auth_manager.refresh_session(user_id):
            session = auth_manager.get_session(user_id)
            logger.debug(
                "Session refreshed",
                user_id=user_id,
                username=username,
                auth_provider=session.auth_provider if session else None,
            )

        # Continue to handler
        return await handler(event, data)

    # User not authenticated - attempt authentication
    logger.info(
        "Attempting authentication for user", user_id=user_id, username=username
    )

    # Try to authenticate (providers will check whitelist and tokens)
    authentication_successful = await auth_manager.authenticate_user(user_id)

    # Log authentication attempt
    if audit_logger:
        await audit_logger.log_auth_attempt(
            user_id=user_id,
            success=authentication_successful,
            method="automatic",
            reason="message_received",
        )

    if authentication_successful:
        session = auth_manager.get_session(user_id)
        logger.info(
            "User authenticated successfully",
            user_id=user_id,
            username=username,
            auth_provider=session.auth_provider if session else None,
        )

        # Welcome message for new session
        if event.effective_message:
            await event.effective_message.reply_text(
                f"🔓 Welcome! You are now authenticated.\n"
                f"Session started at {datetime.now(UTC).strftime('%H:%M:%S UTC')}"
            )

        # Continue to handler
        return await handler(event, data)

    else:
        # Authentication failed
        logger.warning("Authentication failed", user_id=user_id, username=username)

        if event.effective_message:
            await event.effective_message.reply_text(
                "🔒 <b>Authentication Required</b>\n\n"
                "You are not authorized to use this bot.\n"
                "Please contact the administrator for access.\n\n"
                f"Your Telegram ID: <code>{user_id}</code>\n"
                "Share this ID with the administrator to request access.",
                parse_mode="HTML",
            )
        return  # Stop processing


async def require_auth(handler: Callable, event: Any, data: Dict[str, Any]) -> Any:
    """Decorator-style middleware that requires authentication.

    This is a stricter version that only allows authenticated users.
    """
    user_id = event.effective_user.id if event.effective_user else None
    auth_manager = data.get("auth_manager")

    if not auth_manager or not auth_manager.is_authenticated(user_id):
        if event.effective_message:
            await event.effective_message.reply_text(
                "🔒 Authentication required to use this command."
            )
        return

    return await handler(event, data)


async def admin_required(handler: Callable, event: Any, data: Dict[str, Any]) -> Any:
    """Middleware that requires admin privileges.

    Note: This is a placeholder - admin privileges would need to be
    implemented in the authentication system.
    """
    user_id = event.effective_user.id if event.effective_user else None
    auth_manager = data.get("auth_manager")

    if not auth_manager or not auth_manager.is_authenticated(user_id):
        if event.effective_message:
            await event.effective_message.reply_text("🔒 Authentication required.")
        return

    session = auth_manager.get_session(user_id)
    if not session or not session.user_info:
        if event.effective_message:
            await event.effective_message.reply_text(
                "🔒 Session information unavailable."
            )
        return

    # Check for admin permissions (placeholder logic)
    permissions = session.user_info.get("permissions", [])
    if "admin" not in permissions:
        if event.effective_message:
            await event.effective_message.reply_text(
                "🔒 <b>Admin Access Required</b>\n\n"
                "This command requires administrator privileges.",
                parse_mode="HTML",
            )
        return

    return await handler(event, data)


# --- Доступ к боту: кто и откуда ----------------------------------------
#
# У бота права root, поэтому «кто может им пользоваться» — это «кто может
# администрировать сервер». Две проверки ниже намеренно не полагаются на
# auth_manager: они читают настройки напрямую и по умолчанию ЗАКРЫВАЮТ
# доступ. Сломанная или недозагруженная конфигурация должна означать
# «никому», а не «всем».


def user_is_allowed(settings: Any, user_id: Optional[int]) -> bool:
    """Есть ли у пользователя доступ к боту."""
    if user_id is None or settings is None:
        return False
    allowed = getattr(settings, "allowed_users", None)
    if not allowed:
        # Пустой список — это «никому». Бот с правами root не должен
        # отвечать всем подряд из-за потерянной строки в настройках.
        return False
    return user_id in allowed


def chat_is_allowed(settings: Any, chat_id: Optional[int]) -> bool:
    """Разрешено ли боту работать в этом чате.

    Ограничение необязательное: если список чатов не задан, проверка не
    применяется. Когда задан — это защита от кражи токена: с украденным
    токеном бота добавят в чужую группу, но там он работать не станет.
    """
    allowed = getattr(settings, "allowed_chat_ids", None)
    if not allowed:
        return True
    return chat_id in allowed


async def callback_auth_middleware(
    handler: Callable, event: Any, data: Dict[str, Any]
) -> Any:
    """Проверка доступа для нажатий кнопок.

    Нажатие кнопки — такое же действие, как сообщение: им можно подтвердить
    системную операцию, сменить проект или остановить задачу. Обычный
    auth_middleware их не видит (он зарегистрирован на сообщения), поэтому
    кнопки проверяются здесь.
    """
    query = getattr(event, "callback_query", None)
    user_id = query.from_user.id if query and query.from_user else None
    chat = getattr(event, "effective_chat", None)
    chat_id = chat.id if chat else None
    settings = data.get("settings")

    if user_is_allowed(settings, user_id) and chat_is_allowed(settings, chat_id):
        return await handler(event, data)

    logger.warning(
        "Callback rejected: no access",
        user_id=user_id,
        chat_id=chat_id,
        callback_data=getattr(query, "data", None),
    )
    if query is not None:
        try:
            await query.answer("У вас нет доступа к этому боту.", show_alert=True)
        except Exception as e:
            logger.debug("Could not answer rejected callback", error=str(e))
    return None
