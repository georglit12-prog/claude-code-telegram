"""Вопрос владельцу с вариантами ответа — кнопками прямо в чате.

Штатный инструмент Claude Code `AskUserQuestion` рассчитан на терминал: в
Telegram он молча возвращает «пользователь не ответил», и разговор идёт
дальше без ответа. Здесь тот же смысл сделан по-телеграмному — вопрос
приходит сообщением с кнопками, нажатие возвращается модели как результат
инструмента, и задача продолжается с выбранным вариантом.

Инструмент живёт внутри процесса бота (SDK MCP server), поэтому у него есть
прямой доступ к чату и к ожидающей задаче — отдельный процесс и настройка
`ENABLE_MCP` для этого не нужны.
"""

import asyncio
import secrets
from typing import Any, Dict, List, Optional

import structlog
from claude_agent_sdk import create_sdk_mcp_server, tool
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from ..utils.html_format import escape_html

logger = structlog.get_logger()

# Больше шести кнопок в столбик — уже список, а не выбор.
MAX_OPTIONS = 6
# Сколько ждать ответа, прежде чем идти дальше самостоятельно.
ANSWER_TIMEOUT_SECONDS = 900

SERVER_NAME = "telegram"
TOOL_NAME = f"mcp__{SERVER_NAME}__ask_user"

# Подсказка модели: без неё Claude возьмёт привычный AskUserQuestion,
# который в Telegram не работает.
SYSTEM_HINT = (
    "Когда нужно, чтобы человек выбрал из вариантов, вызывай инструмент "
    f"{TOOL_NAME}: вопрос придёт ему кнопками в Telegram, а ответ вернётся "
    "тебе. Штатный AskUserQuestion здесь бесполезен — интерфейса для него "
    "нет, он всегда отвечает «не ответил». Спрашивай только тогда, когда "
    "ответ действительно меняет работу, а не ради подтверждения."
)

FREE_TEXT_LABEL = "✍️ Свой вариант"


class AskUserChannel:
    """Канал «спросить и дождаться» на время одной задачи."""

    def __init__(
        self,
        bot: Any,
        chat_id: int,
        message_thread_id: Optional[int] = None,
        timeout: float = ANSWER_TIMEOUT_SECONDS,
    ) -> None:
        self.bot = bot
        self.chat_id = chat_id
        self.message_thread_id = message_thread_id
        self.timeout = timeout
        # Заданные, но не отвеченные вопросы: ключ → состояние.
        self._pending: Dict[str, Dict[str, Any]] = {}

    @property
    def waiting(self) -> bool:
        """Есть ли вопрос, который ждёт ответа."""
        return bool(self._pending)

    async def ask(self, question: str, options: List[str]) -> str:
        """Задать вопрос и дождаться ответа. Возвращает текст ответа."""
        options = [str(o).strip() for o in options if str(o).strip()][:MAX_OPTIONS]
        if not options:
            return "Вопрос без вариантов ответа — спрашивать нечего."

        key = secrets.token_urlsafe(6)
        rows = [
            [InlineKeyboardButton(text[:64], callback_data=f"ask:{key}:{index}")]
            for index, text in enumerate(options)
        ]
        rows.append(
            [InlineKeyboardButton(FREE_TEXT_LABEL, callback_data=f"ask:{key}:free")]
        )

        body = "\n".join(
            [f"❓ <b>{escape_html(question)}</b>", "", "<i>Выберите вариант:</i>"]
        )
        message = await self.bot.send_message(
            chat_id=self.chat_id,
            text=body,
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(rows),
            message_thread_id=self.message_thread_id,
        )

        future: "asyncio.Future[str]" = asyncio.get_running_loop().create_future()
        self._pending[key] = {
            "future": future,
            "options": options,
            "message": message,
            "question": question,
        }

        try:
            answer = await asyncio.wait_for(future, timeout=self.timeout)
        except asyncio.TimeoutError:
            await self._close(key, "⏳ <i>ответа не было, продолжаю сам</i>")
            return (
                "Пользователь не ответил за отведённое время. Выбери сам "
                "разумный вариант, скажи в ответе, какой и почему."
            )
        finally:
            self._pending.pop(key, None)

        return f"Пользователь выбрал: {answer}"

    async def answer_button(self, key: str, choice: str) -> Optional[str]:
        """Нажата кнопка. Возвращает выбранный текст или None, если не наш вопрос."""
        state = self._pending.get(key)
        if not state or state["future"].done():
            return None

        if choice == "free":
            state["awaiting_text"] = True
            await self._close(
                key,
                "✍️ <i>напишите свой ответ сообщением</i>",
                keep_pending=True,
            )
            return FREE_TEXT_LABEL

        try:
            answer = state["options"][int(choice)]
        except (ValueError, IndexError):
            return None

        state["future"].set_result(answer)
        await self._close(key, f"✅ <b>{escape_html(answer)}</b>")
        return answer

    async def answer_text(self, text: str) -> bool:
        """Свой вариант текстом. True — если текст ушёл как ответ на вопрос."""
        for key, state in list(self._pending.items()):
            if state.get("awaiting_text") and not state["future"].done():
                state["future"].set_result(text)
                return True
        return False

    async def cancel(self) -> None:
        """Задача кончилась — снять кнопки у всех незакрытых вопросов."""
        for key in list(self._pending):
            state = self._pending.get(key)
            if state and not state["future"].done():
                state["future"].cancel()
            await self._close(key, "<i>вопрос снят</i>")
            self._pending.pop(key, None)

    async def _close(self, key: str, note: str, keep_pending: bool = False) -> None:
        """Заменить кнопки на итог выбора, чтобы нажать второй раз было нельзя."""
        state = self._pending.get(key)
        if not state:
            return
        try:
            await state["message"].edit_text(
                f"❓ <b>{escape_html(state['question'])}</b>\n\n{note}",
                parse_mode="HTML",
                reply_markup=None,
            )
        except Exception:
            logger.debug("Failed to close question message")
        if not keep_pending:
            state["closed"] = True


def build_ask_server(channel: AskUserChannel) -> Any:
    """Собрать MCP-сервер с одним инструментом для этой задачи."""

    @tool(
        "ask_user",
        "Задать владельцу вопрос и дать выбрать из вариантов кнопками в "
        "Telegram. Возвращает выбранный вариант. Используй, когда ответ "
        "меняет дальнейшую работу.",
        {"question": str, "options": list},
    )
    async def ask_user(args: Dict[str, Any]) -> Dict[str, Any]:
        question = str(args.get("question") or "").strip()
        raw_options = args.get("options") or []
        if isinstance(raw_options, str):
            raw_options = [raw_options]

        options: List[str] = []
        for item in raw_options:
            # Модель может прислать как строки, так и {label, description}.
            if isinstance(item, dict):
                label = item.get("label") or item.get("text") or item.get("title")
                if label:
                    options.append(str(label))
            else:
                options.append(str(item))

        logger.info("Asking user", question=question[:80], options=len(options))
        answer = await channel.ask(question, options)
        return {"content": [{"type": "text", "text": answer}]}

    return create_sdk_mcp_server(name=SERVER_NAME, version="1.0.0", tools=[ask_user])
