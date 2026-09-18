"""Message orchestrator — single entry point for all Telegram updates.

Routes messages based on agentic vs classic mode. In agentic mode, provides
a minimal conversational interface (3 commands, no inline keyboards). In
classic mode, delegates to existing full-featured handlers.
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import structlog
from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    Update,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ..claude.sdk_integration import StreamUpdate
from ..config.settings import Settings
from ..projects import PrivateTopicsUnavailableError
from .utils.draft_streamer import DraftStreamer, generate_draft_id
from .utils.html_format import escape_html
from .utils.image_extractor import (
    ImageAttachment,
    should_send_as_photo,
    validate_image_path,
)

logger = structlog.get_logger()

_MEDIA_TYPE_MAP = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
}

# Patterns that look like secrets/credentials in CLI arguments
_SECRET_PATTERNS: List[re.Pattern[str]] = [
    # API keys / tokens (sk-ant-..., sk-..., ghp_..., gho_..., github_pat_..., xoxb-...)
    re.compile(
        r"(sk-ant-api\d*-[A-Za-z0-9_-]{10})[A-Za-z0-9_-]*"
        r"|(sk-[A-Za-z0-9_-]{20})[A-Za-z0-9_-]*"
        r"|(ghp_[A-Za-z0-9]{5})[A-Za-z0-9]*"
        r"|(gho_[A-Za-z0-9]{5})[A-Za-z0-9]*"
        r"|(github_pat_[A-Za-z0-9_]{5})[A-Za-z0-9_]*"
        r"|(xoxb-[A-Za-z0-9]{5})[A-Za-z0-9-]*"
    ),
    # AWS access keys
    re.compile(r"(AKIA[0-9A-Z]{4})[0-9A-Z]{12}"),
    # Generic long hex/base64 tokens after common flags/env patterns
    re.compile(
        r"((?:--token|--secret|--password|--api-key|--apikey|--auth)"
        r"[= ]+)['\"]?[A-Za-z0-9+/_.:-]{8,}['\"]?"
    ),
    # Inline env assignments like KEY=value
    re.compile(
        r"((?:TOKEN|SECRET|PASSWORD|API_KEY|APIKEY|AUTH_TOKEN|PRIVATE_KEY"
        r"|ACCESS_KEY|CLIENT_SECRET|WEBHOOK_SECRET)"
        r"=)['\"]?[^\s'\"]{8,}['\"]?"
    ),
    # Bearer / Basic auth headers
    re.compile(r"(Bearer )[A-Za-z0-9+/_.:-]{8,}" r"|(Basic )[A-Za-z0-9+/=]{8,}"),
    # Connection strings with credentials  user:pass@host
    re.compile(r"://([^:]+:)[^@]{4,}(@)"),
]


def _redact_secrets(text: str) -> str:
    """Replace likely secrets/credentials with redacted placeholders."""
    result = text
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub(
            lambda m: next((g + "***" for g in m.groups() if g is not None), "***"),
            result,
        )
    return result


# Что показывать вместо технических названий инструментов.
# Человеку важно «читает файл», а не «Read».
_TOOL_LABELS: Dict[str, str] = {
    "Read": "читает",
    "Write": "пишет",
    "Edit": "правит",
    "MultiEdit": "правит",
    "NotebookRead": "читает блокнот",
    "NotebookEdit": "правит блокнот",
    "Bash": "выполняет",
    "Glob": "ищет файлы",
    "Grep": "ищет в коде",
    "LS": "смотрит папку",
    "Task": "думает",
    "TaskOutput": "думает",
    "WebFetch": "смотрит в интернете",
    "WebSearch": "ищет в интернете",
    "TodoRead": "смотрит план",
    "TodoWrite": "составляет план",
    "Skill": "применяет навык",
}


def _tool_label(name: str) -> str:
    """Понятное человеку действие вместо имени инструмента."""
    return _TOOL_LABELS.get(name, name)


# Кадры «живого» индикатора: меняются на каждом обновлении сообщения,
# поэтому сразу видно, что бот не завис.
_SPINNER = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


def _human_elapsed(seconds: float) -> str:
    """Время работы словами: 8 сек, 1 мин 20 сек, 3 мин."""
    total = int(seconds)
    if total < 60:
        return f"{total} сек"
    minutes, rest = divmod(total, 60)
    if rest == 0:
        return f"{minutes} мин"
    return f"{minutes} мин {rest} сек"


# Tool name -> friendly emoji mapping for verbose output
_TOOL_ICONS: Dict[str, str] = {
    "Read": "\U0001f4d6",
    "Write": "\u270f\ufe0f",
    "Edit": "\u270f\ufe0f",
    "MultiEdit": "\u270f\ufe0f",
    "Bash": "\U0001f4bb",
    "Glob": "\U0001f50d",
    "Grep": "\U0001f50d",
    "LS": "\U0001f4c2",
    "Task": "\U0001f9e0",
    "TaskOutput": "\U0001f9e0",
    "WebFetch": "\U0001f310",
    "WebSearch": "\U0001f310",
    "NotebookRead": "\U0001f4d3",
    "NotebookEdit": "\U0001f4d3",
    "TodoRead": "\u2611\ufe0f",
    "TodoWrite": "\u2611\ufe0f",
}


def _tool_icon(name: str) -> str:
    """Return emoji for a tool, with a default wrench."""
    return _TOOL_ICONS.get(name, "\U0001f527")


@dataclass
class ActiveRequest:
    """Tracks an in-flight Claude request so it can be interrupted."""

    user_id: int
    interrupt_event: asyncio.Event = field(default_factory=asyncio.Event)
    interrupted: bool = False
    progress_msg: Any = None  # telegram Message object


class MessageOrchestrator:
    """Routes messages based on mode. Single entry point for all Telegram updates."""

    def __init__(self, settings: Settings, deps: Dict[str, Any]):
        self.settings = settings
        self.deps = deps
        self._active_requests: Dict[int, ActiveRequest] = {}
        self._known_commands: frozenset[str] = frozenset()

    def _inject_deps(self, handler: Callable) -> Callable:  # type: ignore[type-arg]
        """Wrap handler to inject dependencies into context.bot_data."""

        async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            for key, value in self.deps.items():
                context.bot_data[key] = value
            context.bot_data["settings"] = self.settings
            context.user_data.pop("_thread_context", None)

            is_sync_bypass = handler.__name__ == "sync_threads"
            is_start_bypass = handler.__name__ in {"start_command", "agentic_start"}
            message_thread_id = self._extract_message_thread_id(update)
            should_enforce = self.settings.enable_project_threads

            if should_enforce:
                if self.settings.project_threads_mode == "private":
                    should_enforce = not is_sync_bypass and not (
                        is_start_bypass and message_thread_id is None
                    )
                else:
                    should_enforce = not is_sync_bypass

            if should_enforce:
                allowed = await self._apply_thread_routing_context(update, context)
                if not allowed:
                    return

            try:
                await handler(update, context)
            finally:
                if should_enforce:
                    self._persist_thread_state(context)

        return wrapped

    async def _apply_thread_routing_context(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """Enforce strict project-thread routing and load thread-local state."""
        manager = context.bot_data.get("project_threads_manager")
        if manager is None:
            await self._reject_for_thread_mode(
                update,
                "❌ <b>Project Thread Mode Misconfigured</b>\n\n"
                "Thread manager is not initialized.",
            )
            return False

        chat = update.effective_chat
        message = update.effective_message
        if not chat or not message:
            return False

        if self.settings.project_threads_mode == "group":
            if chat.id != self.settings.project_threads_chat_id:
                await self._reject_for_thread_mode(
                    update,
                    manager.guidance_message(mode=self.settings.project_threads_mode),
                )
                return False
        else:
            if getattr(chat, "type", "") != "private":
                await self._reject_for_thread_mode(
                    update,
                    manager.guidance_message(mode=self.settings.project_threads_mode),
                )
                return False

        message_thread_id = self._extract_message_thread_id(update)
        if not message_thread_id:
            await self._reject_for_thread_mode(
                update,
                manager.guidance_message(mode=self.settings.project_threads_mode),
            )
            return False

        project = await manager.resolve_project(chat.id, message_thread_id)
        if not project:
            await self._reject_for_thread_mode(
                update,
                manager.guidance_message(mode=self.settings.project_threads_mode),
            )
            return False

        state_key = f"{chat.id}:{message_thread_id}"
        thread_states = context.user_data.setdefault("thread_state", {})
        state = thread_states.get(state_key, {})

        project_root = project.absolute_path
        current_dir_raw = state.get("current_directory")
        current_dir = (
            Path(current_dir_raw).resolve() if current_dir_raw else project_root
        )
        if not self._is_within(current_dir, project_root) or not current_dir.is_dir():
            current_dir = project_root

        context.user_data["current_directory"] = current_dir
        context.user_data["claude_session_id"] = state.get("claude_session_id")
        context.user_data["_thread_context"] = {
            "chat_id": chat.id,
            "message_thread_id": message_thread_id,
            "state_key": state_key,
            "project_slug": project.slug,
            "project_root": str(project_root),
            "project_name": project.name,
        }
        return True

    def _persist_thread_state(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Persist compatibility keys back into per-thread state."""
        thread_context = context.user_data.get("_thread_context")
        if not thread_context:
            return

        project_root = Path(thread_context["project_root"])
        current_dir = context.user_data.get("current_directory", project_root)
        if not isinstance(current_dir, Path):
            current_dir = Path(str(current_dir))
        current_dir = current_dir.resolve()
        if not self._is_within(current_dir, project_root) or not current_dir.is_dir():
            current_dir = project_root

        thread_states = context.user_data.setdefault("thread_state", {})
        thread_states[thread_context["state_key"]] = {
            "current_directory": str(current_dir),
            "claude_session_id": context.user_data.get("claude_session_id"),
            "project_slug": thread_context["project_slug"],
        }

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        """Return True if path is within root."""
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _extract_message_thread_id(update: Update) -> Optional[int]:
        """Extract topic/thread id from update message for forum/direct topics."""
        message = update.effective_message
        if not message:
            return None
        message_thread_id = getattr(message, "message_thread_id", None)
        if isinstance(message_thread_id, int) and message_thread_id > 0:
            return message_thread_id
        dm_topic = getattr(message, "direct_messages_topic", None)
        topic_id = getattr(dm_topic, "topic_id", None) if dm_topic else None
        if isinstance(topic_id, int) and topic_id > 0:
            return topic_id
        # Telegram omits message_thread_id for the General topic in forum
        # supergroups; its canonical thread ID is 1.
        chat = update.effective_chat
        if chat and getattr(chat, "is_forum", False):
            return 1
        return None

    async def _reject_for_thread_mode(self, update: Update, message: str) -> None:
        """Send a guidance response when strict thread routing rejects an update."""
        query = update.callback_query
        if query:
            try:
                await query.answer()
            except Exception:
                pass
            if query.message:
                await query.message.reply_text(message, parse_mode="HTML")
            return

        if update.effective_message:
            await update.effective_message.reply_text(message, parse_mode="HTML")

    def register_handlers(self, app: Application) -> None:
        """Register handlers based on mode."""
        if self.settings.agentic_mode:
            self._register_agentic_handlers(app)
        else:
            self._register_classic_handlers(app)

    def _register_agentic_handlers(self, app: Application) -> None:
        """Register agentic handlers: commands + text/file/photo."""
        from .handlers import command

        # Commands
        handlers = [
            ("start", self.agentic_start),
            ("new", self.agentic_new),
            ("status", self.agentic_status),
            ("verbose", self.agentic_verbose),
            ("repo", self.agentic_repo),
            ("newproject", self.agentic_newproject),
            ("model", self.agentic_model),
            ("mode", self.agentic_mode),
            ("effort", self.agentic_effort),
            ("restart", command.restart_command),
        ]
        if self.settings.enable_project_threads:
            handlers.append(("sync_threads", command.sync_threads))

        # Derive known commands dynamically — avoids drift when new commands are added
        self._known_commands: frozenset[str] = frozenset(cmd for cmd, _ in handlers)

        for cmd, handler in handlers:
            app.add_handler(CommandHandler(cmd, self._inject_deps(handler)))

        # Text messages -> Claude
        app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._inject_deps(self.agentic_text),
            ),
            group=10,
        )

        # Unknown slash commands -> Claude (passthrough in agentic mode).
        # Registered commands are handled by CommandHandlers in group 0
        # (higher priority). This catches any /command not matched there
        # and forwards it to Claude, while skipping known commands to
        # avoid double-firing.
        app.add_handler(
            MessageHandler(
                filters.COMMAND,
                self._inject_deps(self._handle_unknown_command),
            ),
            group=10,
        )

        # File uploads -> Claude
        app.add_handler(
            MessageHandler(
                filters.Document.ALL, self._inject_deps(self.agentic_document)
            ),
            group=10,
        )

        # Photo uploads -> Claude
        app.add_handler(
            MessageHandler(filters.PHOTO, self._inject_deps(self.agentic_photo)),
            group=10,
        )

        # Voice messages -> transcribe -> Claude
        app.add_handler(
            MessageHandler(filters.VOICE, self._inject_deps(self.agentic_voice)),
            group=10,
        )

        # Stop button callback (must be before cd: handler)
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._handle_stop_callback),
                pattern=r"^stop:",
            )
        )

        # Кнопки интерфейса (проекты, модель, режим, глубина, статус)
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._handle_ui_callback),
                pattern=r"^ui:",
            )
        )

        # Only cd: callbacks (for project selection), scoped by pattern
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._agentic_callback),
                pattern=r"^cd:",
            )
        )

        logger.info("Agentic handlers registered")

    def _register_classic_handlers(self, app: Application) -> None:
        """Register full classic handler set (moved from core.py)."""
        from .handlers import callback, command, message

        handlers = [
            ("start", command.start_command),
            ("help", command.help_command),
            ("new", command.new_session),
            ("continue", command.continue_session),
            ("end", command.end_session),
            ("ls", command.list_files),
            ("cd", command.change_directory),
            ("pwd", command.print_working_directory),
            ("projects", command.show_projects),
            ("status", command.session_status),
            ("export", command.export_session),
            ("actions", command.quick_actions),
            ("git", command.git_command),
            ("restart", command.restart_command),
        ]
        if self.settings.enable_project_threads:
            handlers.append(("sync_threads", command.sync_threads))

        for cmd, handler in handlers:
            app.add_handler(CommandHandler(cmd, self._inject_deps(handler)))

        app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._inject_deps(message.handle_text_message),
            ),
            group=10,
        )
        app.add_handler(
            MessageHandler(
                filters.Document.ALL, self._inject_deps(message.handle_document)
            ),
            group=10,
        )
        app.add_handler(
            MessageHandler(filters.PHOTO, self._inject_deps(message.handle_photo)),
            group=10,
        )
        app.add_handler(
            MessageHandler(filters.VOICE, self._inject_deps(message.handle_voice)),
            group=10,
        )
        app.add_handler(
            CallbackQueryHandler(self._inject_deps(callback.handle_callback_query))
        )

        logger.info("Classic handlers registered (13 commands + full handler set)")

    async def get_bot_commands(self) -> list:  # type: ignore[type-arg]
        """Return bot commands appropriate for current mode."""
        if self.settings.agentic_mode:
            commands = [
                BotCommand("start", "Start the bot"),
                BotCommand("new", "Start a fresh session"),
                BotCommand("status", "Show session status"),
                BotCommand("verbose", "Set output verbosity (0/1/2)"),
                BotCommand("repo", "Проекты: список и переключение"),
                BotCommand("newproject", "Создать новый проект"),
                BotCommand("model", "Модель: opus | sonnet | haiku | fable"),
                BotCommand("mode", "Режим: план | правки | обычный | авто"),
                BotCommand("effort", "Глубина: low | medium | high | xhigh | max"),
                BotCommand("restart", "Restart the bot"),
            ]
            if self.settings.enable_project_threads:
                commands.append(BotCommand("sync_threads", "Sync project topics"))
            return commands
        else:
            commands = [
                BotCommand("start", "Start bot and show help"),
                BotCommand("help", "Show available commands"),
                BotCommand("new", "Clear context and start fresh session"),
                BotCommand("continue", "Explicitly continue last session"),
                BotCommand("end", "End current session and clear context"),
                BotCommand("ls", "List files in current directory"),
                BotCommand("cd", "Change directory (resumes project session)"),
                BotCommand("pwd", "Show current directory"),
                BotCommand("projects", "Show all projects"),
                BotCommand("status", "Show session status"),
                BotCommand("export", "Export current session"),
                BotCommand("actions", "Show quick actions"),
                BotCommand("git", "Git repository commands"),
                BotCommand("restart", "Restart the bot"),
            ]
            if self.settings.enable_project_threads:
                commands.append(BotCommand("sync_threads", "Sync project topics"))
            return commands

    # --- Agentic handlers ---

    # ------------------------------------------------------------------
    # Кнопки под сообщениями
    # ------------------------------------------------------------------

    def _main_keyboard(self) -> InlineKeyboardMarkup:
        """Главные кнопки: то, что нужно чаще всего."""
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("📁 Проекты", callback_data="ui:repos"),
                    InlineKeyboardButton("✨ Новый проект", callback_data="ui:newproject"),
                ],
                [
                    InlineKeyboardButton("🧠 Модель", callback_data="ui:model"),
                    InlineKeyboardButton("⚙️ Режим", callback_data="ui:mode"),
                    InlineKeyboardButton("🎯 Глубина", callback_data="ui:effort"),
                ],
                [
                    InlineKeyboardButton("📊 Статус", callback_data="ui:status"),
                    InlineKeyboardButton("🆕 Начать заново", callback_data="ui:new"),
                ],
            ]
        )

    def _back_row(self) -> List[InlineKeyboardButton]:
        return [InlineKeyboardButton("‹ Назад", callback_data="ui:home")]

    def _model_keyboard(self, current: str) -> InlineKeyboardMarkup:
        rows = []
        for key, (value, desc) in self.MODEL_CHOICES.items():
            mark = "✅ " if value == current else ""
            short = desc.split("—", 1)[-1].strip()
            rows.append(
                [InlineKeyboardButton(f"{mark}{key} · {short}", callback_data=f"ui:setmodel:{key}")]
            )
        rows.append(self._back_row())
        return InlineKeyboardMarkup(rows)

    def _mode_keyboard(self, current: str) -> InlineKeyboardMarkup:
        rows = []
        for key in ("план", "правки", "обычный", "авто"):
            value, desc = self.MODE_CHOICES[key]
            mark = "✅ " if value == current else ""
            rows.append(
                [InlineKeyboardButton(f"{mark}{key} · {desc}", callback_data=f"ui:setmode:{key}")]
            )
        rows.append(self._back_row())
        return InlineKeyboardMarkup(rows)

    def _effort_keyboard(self, current: str) -> InlineKeyboardMarkup:
        rows = []
        for key, desc in self.EFFORT_CHOICES.items():
            mark = "✅ " if key == current else ""
            rows.append(
                [InlineKeyboardButton(f"{mark}{key} · {desc}", callback_data=f"ui:seteffort:{key}")]
            )
        rows.append(self._back_row())
        return InlineKeyboardMarkup(rows)

    def _repos_keyboard(self) -> InlineKeyboardMarkup:
        """Список проектов кнопками, по два в ряд."""
        base = self.settings.approved_directory
        try:
            names = sorted(
                d.name for d in base.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            )
        except Exception:
            names = []

        rows, row = [], []
        for name in names[:20]:
            mark = "📦 " if (base / name / ".git").is_dir() else "📁 "
            row.append(InlineKeyboardButton(f"{mark}{name}", callback_data=f"cd:{name}"))
            if len(row) == 2:
                rows.append(row); row = []
        if row:
            rows.append(row)
        rows.append([InlineKeyboardButton("✨ Новый проект", callback_data="ui:newproject")])
        rows.append(self._back_row())
        return InlineKeyboardMarkup(rows)

    async def _handle_ui_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Нажатия на кнопки интерфейса."""
        query = update.callback_query
        parts = query.data.split(":")
        action = parts[1] if len(parts) > 1 else ""
        value = parts[2] if len(parts) > 2 else ""

        model = context.user_data.get("claude_model") or self.settings.claude_model or "opus"
        mode = context.user_data.get("permission_mode") or "bypassPermissions"
        effort = context.user_data.get("claude_effort") or "xhigh"
        mode_ru = {
            "plan": "план", "acceptEdits": "правки",
            "default": "обычный", "bypassPermissions": "авто",
        }

        async def show(text: str, markup: InlineKeyboardMarkup) -> None:
            try:
                await query.edit_message_text(text, reply_markup=markup, parse_mode="HTML")
            except Exception:
                pass

        if action == "home":
            await query.answer()
            current_dir = context.user_data.get("current_directory", self.settings.approved_directory)
            project = current_dir.name if current_dir != self.settings.approved_directory else "не выбран"
            await show(
                f"📁 Проект: <b>{escape_html(project)}</b>\n"
                f"🧠 Модель: <b>{escape_html(model)}</b> · "
                f"Режим: <b>{mode_ru.get(mode, mode)}</b> · "
                f"Глубина: <b>{escape_html(effort)}</b>\n\n"
                f"Напишите задачу словами или выберите действие.",
                self._main_keyboard(),
            )

        elif action == "repos":
            await query.answer()
            await show("📁 <b>Проекты</b>\n\nВыберите, с чем работаем:", self._repos_keyboard())

        elif action == "model":
            await query.answer()
            await show(
                "🧠 <b>Модель</b>\n\n"
                "Opus входит в подписку. Fable сильнее, но тратит деньги сверх неё.",
                self._model_keyboard(model),
            )

        elif action == "mode":
            await query.answer()
            await show(
                "⚙️ <b>Режим работы</b>\n\n"
                "«План» покажет замысел, ничего не меняя — удобно перед крупной задачей.",
                self._mode_keyboard(mode),
            )

        elif action == "effort":
            await query.answer()
            await show(
                "🎯 <b>Глубина проработки</b>\n\n"
                "Чем выше, тем дольше думает и тем лучше результат.",
                self._effort_keyboard(effort),
            )

        elif action == "setmodel" and value in self.MODEL_CHOICES:
            new_value, desc = self.MODEL_CHOICES[value]
            context.user_data["claude_model"] = new_value
            await query.answer(f"Модель: {value}")
            note = "\n\n⚠️ Fable тратит средства сверх подписки." if value == "fable" else ""
            await show(
                f"🧠 <b>Модель</b>\n\nВыбрано: <b>{value}</b> — {escape_html(desc)}{note}",
                self._model_keyboard(new_value),
            )

        elif action == "setmode" and value in self.MODE_CHOICES:
            new_value, desc = self.MODE_CHOICES[value]
            context.user_data["permission_mode"] = new_value
            await query.answer(f"Режим: {value}")
            await show(
                f"⚙️ <b>Режим работы</b>\n\nВыбрано: <b>{value}</b> — {escape_html(desc)}",
                self._mode_keyboard(new_value),
            )

        elif action == "seteffort" and value in self.EFFORT_CHOICES:
            context.user_data["claude_effort"] = value
            await query.answer(f"Глубина: {value}")
            await show(
                f"🎯 <b>Глубина проработки</b>\n\n"
                f"Выбрано: <b>{value}</b> — {escape_html(self.EFFORT_CHOICES[value])}",
                self._effort_keyboard(value),
            )

        elif action == "status":
            await query.answer()
            current_dir = context.user_data.get("current_directory", self.settings.approved_directory)
            project = current_dir.name if current_dir != self.settings.approved_directory else "не выбран"
            session = "продолжается" if context.user_data.get("claude_session_id") else "новая"
            await show(
                f"📊 <b>Сейчас</b>\n\n"
                f"📁 Проект: <b>{escape_html(project)}</b>\n"
                f"💬 Разговор: <b>{session}</b>\n"
                f"🧠 Модель: <b>{escape_html(model)}</b>\n"
                f"⚙️ Режим: <b>{mode_ru.get(mode, mode)}</b>\n"
                f"🎯 Глубина: <b>{escape_html(effort)}</b>",
                self._main_keyboard(),
            )

        elif action == "new":
            context.user_data["claude_session_id"] = None
            context.user_data["session_started"] = True
            context.user_data["force_new_session"] = True
            await query.answer("Начинаем заново")
            await show("🆕 <b>Начинаем заново</b>\n\nЧто делаем?", self._main_keyboard())

        elif action == "newproject":
            await query.answer()
            await show(
                "✨ <b>Новый проект</b>\n\n"
                "Отправьте команду с именем проекта:\n"
                "<code>/newproject имя-проекта</code>\n\n"
                "Имя латиницей без пробелов, например <code>my-landing</code>.\n"
                "Проект появится и на сервере, и на GitHub.",
                InlineKeyboardMarkup([self._back_row()]),
            )

        else:
            await query.answer()

    async def agentic_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Brief welcome, no buttons."""
        user = update.effective_user
        sync_line = ""
        if (
            self.settings.enable_project_threads
            and self.settings.project_threads_mode == "private"
        ):
            if (
                not update.effective_chat
                or getattr(update.effective_chat, "type", "") != "private"
            ):
                await update.message.reply_text(
                    "🚫 <b>Private Topics Mode</b>\n\n"
                    "Use this bot in a private chat and run <code>/start</code> there.",
                    parse_mode="HTML",
                )
                return
            manager = context.bot_data.get("project_threads_manager")
            if manager:
                try:
                    result = await manager.sync_topics(
                        context.bot,
                        chat_id=update.effective_chat.id,
                    )
                    sync_line = (
                        "\n\n🧵 Topics synced"
                        f" (created {result.created}, reused {result.reused})."
                    )
                except PrivateTopicsUnavailableError:
                    await update.message.reply_text(
                        manager.private_topics_unavailable_message(),
                        parse_mode="HTML",
                    )
                    return
                except Exception:
                    sync_line = "\n\n🧵 Topic sync failed. Run /sync_threads to retry."
        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        dir_display = f"<code>{current_dir}/</code>"

        safe_name = escape_html(user.first_name)
        model = context.user_data.get("claude_model") or self.settings.claude_model or "opus"
        mode = context.user_data.get("permission_mode") or "bypassPermissions"
        mode_ru = {
            "plan": "план",
            "acceptEdits": "правки",
            "default": "обычный",
            "bypassPermissions": "авто",
        }.get(mode, mode)
        project = current_dir.name if current_dir != self.settings.approved_directory else "не выбран"

        await update.message.reply_text(
            f"👋 Привет, {safe_name}!\n\n"
            f"Напишите задачу обычными словами — я прочитаю код, внесу правки, "
            f"выполню команды и сохраню изменения.\n\n"
            f"📁 Проект: <b>{escape_html(project)}</b>\n"
            f"🧠 Модель: <b>{escape_html(model)}</b> · Режим: <b>{mode_ru}</b>"
            f"{sync_line}",
            parse_mode="HTML",
            reply_markup=self._main_keyboard(),
        )

    async def agentic_new(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Reset session, one-line confirmation."""
        context.user_data["claude_session_id"] = None
        context.user_data["session_started"] = True
        context.user_data["force_new_session"] = True

        await update.message.reply_text(
            "🆕 Начинаем заново. Что делаем?",
            reply_markup=self._main_keyboard(),
        )

    async def agentic_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Compact one-line status, no buttons."""
        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        project = (
            current_dir.name
            if current_dir != self.settings.approved_directory
            else "не выбран"
        )
        session_id = context.user_data.get("claude_session_id")
        session_status = "продолжается" if session_id else "новая"

        model = context.user_data.get("claude_model") or self.settings.claude_model or "opus"
        mode = context.user_data.get("permission_mode") or "bypassPermissions"
        mode_ru = {
            "plan": "план", "acceptEdits": "правки",
            "default": "обычный", "bypassPermissions": "авто",
        }.get(mode, mode)
        effort = context.user_data.get("claude_effort") or "xhigh"

        await update.message.reply_text(
            f"📊 <b>Сейчас</b>\n\n"
            f"📁 Проект: <b>{escape_html(project)}</b>\n"
            f"💬 Разговор: <b>{session_status}</b>\n"
            f"🧠 Модель: <b>{escape_html(model)}</b>\n"
            f"⚙️ Режим: <b>{mode_ru}</b>\n"
            f"🎯 Глубина: <b>{escape_html(effort)}</b>",
            parse_mode="HTML",
            reply_markup=self._main_keyboard(),
        )

    def _user_overrides(self, context: ContextTypes.DEFAULT_TYPE) -> dict:
        """Выбор пользователя из команд /model, /mode, /effort.

        Пусто = работаем на значениях из настроек бота.
        """
        out: dict = {}
        if context.user_data.get("claude_model"):
            out["model"] = context.user_data["claude_model"]
        if context.user_data.get("permission_mode"):
            out["permission_mode"] = context.user_data["permission_mode"]
        if context.user_data.get("claude_effort"):
            out["effort"] = context.user_data["claude_effort"]
        return out

    def _get_verbose_level(self, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Return effective verbose level: per-user override or global default."""
        user_override = context.user_data.get("verbose_level")
        if user_override is not None:
            return int(user_override)
        return self.settings.verbose_level

    # --- Выбор модели, режима работы и глубины (команды /model /mode /effort) ---
    # Значения живут в user_data и переживают перезапуск (PicklePersistence).
    # Читает их sdk_integration при сборке вызова Claude.

    MODEL_CHOICES = {
        "opus": ("opus", "Opus 5 — входит в подписку Max"),
        "sonnet": ("sonnet", "Sonnet 5 — быстрее и дешевле"),
        "haiku": ("haiku", "Haiku 4.5 — самый быстрый, для мелочей"),
        "fable": ("claude-fable-5-1", "Fable 5.1 — сильнее всех, но тратит usage credits"),
    }

    MODE_CHOICES = {
        "план": ("plan", "сначала показывает план, ничего не меняет"),
        "plan": ("plan", "сначала показывает план, ничего не меняет"),
        "правки": ("acceptEdits", "правит файлы сразу, команды спрашивает"),
        "edits": ("acceptEdits", "правит файлы сразу, команды спрашивает"),
        "обычный": ("default", "обычный режим Claude Code"),
        "default": ("default", "обычный режим Claude Code"),
        "авто": ("bypassPermissions", "делает всё сам, без вопросов"),
        "auto": ("bypassPermissions", "делает всё сам, без вопросов"),
    }

    EFFORT_CHOICES = {
        "low": "низкая — быстрые простые задачи",
        "medium": "средняя",
        "high": "высокая",
        "xhigh": "очень высокая — сложный код",
        "max": "максимальная — когда важна точность, а не скорость",
    }

    async def agentic_model(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Выбор модели: /model [opus|sonnet|haiku|fable]."""
        args = update.message.text.split()[1:] if update.message.text else []
        current = context.user_data.get("claude_model") or self.settings.claude_model or "opus"

        if not args:
            await update.message.reply_text(
                "🧠 <b>Модель</b>\n\n"
                "Opus входит в подписку. Fable сильнее, но тратит деньги сверх неё.",
                parse_mode="HTML",
                reply_markup=self._model_keyboard(current),
            )
            return

        choice = args[0].lower()
        if choice not in self.MODEL_CHOICES:
            await update.message.reply_text(
                "Не знаю такую модель. Доступны: "
                + ", ".join(self.MODEL_CHOICES) + "."
            )
            return

        value, desc = self.MODEL_CHOICES[choice]
        context.user_data["claude_model"] = value
        note = ""
        if choice == "fable":
            note = "\n\n⚠️ Fable тратит usage credits сверх подписки."
        await update.message.reply_text(
            f"Модель: <b>{choice}</b> — {desc}{note}", parse_mode="HTML"
        )

    async def agentic_mode(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Режим работы: /mode [план|правки|обычный|авто]."""
        args = update.message.text.split()[1:] if update.message.text else []
        current = context.user_data.get("permission_mode") or "bypassPermissions"
        human = {
            "plan": "план",
            "acceptEdits": "правки",
            "default": "обычный",
            "bypassPermissions": "авто",
        }

        if not args:
            await update.message.reply_text(
                "⚙️ <b>Режим работы</b>\n\n"
                "«План» покажет замысел, ничего не меняя — удобно перед крупной задачей.",
                parse_mode="HTML",
                reply_markup=self._mode_keyboard(current),
            )
            return

        choice = args[0].lower()
        if choice not in self.MODE_CHOICES:
            await update.message.reply_text(
                "Не знаю такой режим. Доступны: план, правки, обычный, авто."
            )
            return

        value, desc = self.MODE_CHOICES[choice]
        context.user_data["permission_mode"] = value
        await update.message.reply_text(
            f"Режим: <b>{human.get(value, value)}</b> — {desc}", parse_mode="HTML"
        )

    async def agentic_effort(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Глубина проработки: /effort [low|medium|high|xhigh|max]."""
        args = update.message.text.split()[1:] if update.message.text else []
        current = context.user_data.get("claude_effort") or "xhigh"

        if not args:
            await update.message.reply_text(
                "🎯 <b>Глубина проработки</b>\n\n"
                "Чем выше, тем дольше думает и тем лучше результат.",
                parse_mode="HTML",
                reply_markup=self._effort_keyboard(current),
            )
            return

        choice = args[0].lower()
        if choice not in self.EFFORT_CHOICES:
            await update.message.reply_text(
                "Доступны: " + ", ".join(self.EFFORT_CHOICES) + "."
            )
            return

        context.user_data["claude_effort"] = choice
        await update.message.reply_text(
            f"Глубина: <b>{choice}</b> — {self.EFFORT_CHOICES[choice]}",
            parse_mode="HTML",
        )

    async def agentic_verbose(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Set output verbosity: /verbose [0|1|2]."""
        args = update.message.text.split()[1:] if update.message.text else []
        if not args:
            current = self._get_verbose_level(context)
            labels = {0: "quiet", 1: "normal", 2: "detailed"}
            await update.message.reply_text(
                f"Verbosity: <b>{current}</b> ({labels.get(current, '?')})\n\n"
                "Usage: <code>/verbose 0|1|2</code>\n"
                "  0 = quiet (final response only)\n"
                "  1 = normal (tools + reasoning)\n"
                "  2 = detailed (tools with inputs + reasoning)",
                parse_mode="HTML",
            )
            return

        try:
            level = int(args[0])
            if level not in (0, 1, 2):
                raise ValueError
        except ValueError:
            await update.message.reply_text(
                "Please use: /verbose 0, /verbose 1, or /verbose 2"
            )
            return

        context.user_data["verbose_level"] = level
        labels = {0: "quiet", 1: "normal", 2: "detailed"}
        await update.message.reply_text(
            f"Verbosity set to <b>{level}</b> ({labels[level]})",
            parse_mode="HTML",
        )

    def _format_verbose_progress(
        self,
        activity_log: List[Dict[str, Any]],
        verbose_level: int,
        start_time: float,
        tick: int = 0,
    ) -> str:
        """Сообщение о ходе работы.

        Задача бота здесь — показать человеку, что он не завис: кадр индикатора
        меняется на каждом обновлении, время идёт, а последние действия названы
        по-человечески («читает config.py», а не «Read»).
        """
        elapsed = time.time() - start_time
        spin = _SPINNER[tick % len(_SPINNER)]
        head = f"{spin} <b>Работаю</b> · {_human_elapsed(elapsed)}"

        if not activity_log:
            return head + "\n\n<i>обдумываю задачу…</i>"

        lines: List[str] = [head, ""]

        shown = activity_log[-12:]
        if len(activity_log) > len(shown):
            lines.append(f"<i>…ранее ещё {len(activity_log) - len(shown)} шагов</i>")

        for entry in shown:
            kind = entry.get("kind", "tool")
            if kind == "text":
                snippet = entry.get("detail", "") or ""
                limit = 300 if verbose_level >= 2 else 90
                if snippet:
                    lines.append(f"💭 <i>{escape_html(snippet[:limit])}</i>")
            else:
                name = entry.get("name", "")
                icon = _tool_icon(name)
                label = _tool_label(name)
                detail = entry.get("detail") or ""
                if detail and verbose_level >= 1:
                    lines.append(
                        f"{icon} {escape_html(label)} <code>{escape_html(detail[:70])}</code>"
                    )
                else:
                    lines.append(f"{icon} {escape_html(label)}")

        return "\n".join(lines)

    @staticmethod
    def _summarize_tool_input(tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Return a short summary of tool input for verbose level 2."""
        if not tool_input:
            return ""
        if tool_name in ("Read", "Write", "Edit", "MultiEdit"):
            path = tool_input.get("file_path") or tool_input.get("path", "")
            if path:
                # Show just the filename, not the full path
                return path.rsplit("/", 1)[-1]
        if tool_name in ("Glob", "Grep"):
            pattern = tool_input.get("pattern", "")
            if pattern:
                return pattern[:60]
        if tool_name == "Bash":
            cmd = tool_input.get("command", "")
            if cmd:
                return _redact_secrets(cmd[:100])[:80]
        if tool_name in ("WebFetch", "WebSearch"):
            return (tool_input.get("url", "") or tool_input.get("query", ""))[:60]
        if tool_name == "Task":
            desc = tool_input.get("description", "")
            if desc:
                return desc[:60]
        # Generic: show first key's value
        for v in tool_input.values():
            if isinstance(v, str) and v:
                return v[:60]
        return ""

    @staticmethod
    def _start_typing_heartbeat(
        chat: Any,
        interval: float = 2.0,
    ) -> "asyncio.Task[None]":
        """Start a background typing indicator task.

        Sends typing every *interval* seconds, independently of
        stream events. Cancel the returned task in a ``finally``
        block.
        """

        async def _heartbeat() -> None:
            try:
                while True:
                    await asyncio.sleep(interval)
                    try:
                        await chat.send_action("typing")
                    except Exception:
                        pass
            except asyncio.CancelledError:
                pass

        return asyncio.create_task(_heartbeat())

    def _start_progress_ticker(
        self,
        progress_msg: Any,
        tool_log: List[Dict[str, Any]],
        verbose_level: int,
        start_time: float,
        reply_markup: Optional[InlineKeyboardMarkup],
        interval: float = 4.0,
    ) -> "asyncio.Task[None]":
        """Двигать индикатор, даже когда от Claude ничего не приходит.

        Claude может несколько минут обдумывать задачу, не вызывая инструментов.
        Без этого сообщение замирало бы, и казалось бы, что бот завис.
        Здесь же меняется кадр и растёт время работы.
        """

        async def _ticker() -> None:
            frame = 0
            last_text = ""
            try:
                while True:
                    await asyncio.sleep(interval)
                    frame += 1
                    text = self._format_verbose_progress(
                        tool_log, verbose_level, start_time, frame
                    )
                    if text == last_text:
                        continue
                    last_text = text
                    try:
                        await progress_msg.edit_text(
                            text, reply_markup=reply_markup, parse_mode="HTML"
                        )
                    except Exception:
                        # «сообщение не изменилось» и лимиты Telegram — не беда
                        pass
            except asyncio.CancelledError:
                pass

        return asyncio.create_task(_ticker())

    def _make_stream_callback(
        self,
        verbose_level: int,
        progress_msg: Any,
        tool_log: List[Dict[str, Any]],
        start_time: float,
        reply_markup: Optional[InlineKeyboardMarkup] = None,
        mcp_images: Optional[List[ImageAttachment]] = None,
        approved_directory: Optional[Path] = None,
        draft_streamer: Optional[DraftStreamer] = None,
        interrupt_event: Optional[asyncio.Event] = None,
    ) -> Optional[Callable[[StreamUpdate], Any]]:
        """Create a stream callback for verbose progress updates.

        When *mcp_images* is provided, the callback also intercepts
        ``send_image_to_user`` tool calls and collects validated
        :class:`ImageAttachment` objects for later Telegram delivery.

        When *draft_streamer* is provided, tool activity and assistant
        text are streamed to the user in real time via
        ``sendMessageDraft``.

        Returns None when verbose_level is 0 **and** no MCP image
        collection or draft streaming is requested.
        Typing indicators are handled by a separate heartbeat task.
        """
        need_mcp_intercept = mcp_images is not None and approved_directory is not None

        if verbose_level == 0 and not need_mcp_intercept and draft_streamer is None:
            return None

        last_edit_time = [0.0]  # mutable container for closure
        tick = [0]  # кадр индикатора: меняется при каждом обновлении

        async def _on_stream(update_obj: StreamUpdate) -> None:
            # Stop all streaming activity after interrupt
            if interrupt_event is not None and interrupt_event.is_set():
                return

            # Intercept send_image_to_user MCP tool calls.
            # The SDK namespaces MCP tools as "mcp__<server>__<tool>",
            # so match both the bare name and the namespaced variant.
            if update_obj.tool_calls and need_mcp_intercept:
                for tc in update_obj.tool_calls:
                    tc_name = tc.get("name", "")
                    if tc_name == "send_image_to_user" or tc_name.endswith(
                        "__send_image_to_user"
                    ):
                        tc_input = tc.get("input", {})
                        file_path = tc_input.get("file_path", "")
                        caption = tc_input.get("caption", "")
                        img = validate_image_path(
                            file_path, approved_directory, caption
                        )
                        if img:
                            mcp_images.append(img)

            # Capture tool calls
            if update_obj.tool_calls:
                for tc in update_obj.tool_calls:
                    name = tc.get("name", "unknown")
                    detail = self._summarize_tool_input(name, tc.get("input", {}))
                    if verbose_level >= 1:
                        tool_log.append(
                            {"kind": "tool", "name": name, "detail": detail}
                        )
                    if draft_streamer:
                        icon = _tool_icon(name)
                        line = (
                            f"{icon} {name}: {detail}" if detail else f"{icon} {name}"
                        )
                        await draft_streamer.append_tool(line)

            # Capture assistant text (reasoning / commentary)
            if update_obj.type == "assistant" and update_obj.content:
                text = update_obj.content.strip()
                if text:
                    first_line = text.split("\n", 1)[0].strip()
                    if first_line:
                        if verbose_level >= 1:
                            tool_log.append(
                                {"kind": "text", "detail": first_line[:120]}
                            )
                        if draft_streamer:
                            await draft_streamer.append_tool(
                                f"\U0001f4ac {first_line[:120]}"
                            )

            # Stream text to user via draft (prefer token deltas;
            # skip full assistant messages to avoid double-appending)
            if draft_streamer and update_obj.content:
                if update_obj.type == "stream_delta":
                    await draft_streamer.append_text(update_obj.content)

            # Throttle progress message edits to avoid Telegram rate limits
            if not draft_streamer and verbose_level >= 1:
                now = time.time()
                if (now - last_edit_time[0]) >= 2.0:
                    last_edit_time[0] = now
                    tick[0] += 1
                    new_text = self._format_verbose_progress(
                        tool_log, verbose_level, start_time, tick[0]
                    )
                    try:
                        await progress_msg.edit_text(
                            new_text, reply_markup=reply_markup, parse_mode="HTML"
                        )
                    except Exception:
                        pass

        return _on_stream

    async def _send_images(
        self,
        update: Update,
        images: List[ImageAttachment],
        reply_to_message_id: Optional[int] = None,
        caption: Optional[str] = None,
        caption_parse_mode: Optional[str] = None,
    ) -> bool:
        """Send extracted images as a media group (album) or documents.

        If *caption* is provided and fits (≤1024 chars), it is attached to the
        photo / first album item so text + images appear as one message.

        Returns True if the caption was successfully embedded in the photo message.
        """
        photos: List[ImageAttachment] = []
        documents: List[ImageAttachment] = []
        for img in images:
            if should_send_as_photo(img.path):
                photos.append(img)
            else:
                documents.append(img)

        # Telegram caption limit
        use_caption = bool(
            caption and len(caption) <= 1024 and photos and not documents
        )
        caption_sent = False

        # Send raster photos as a single album (Telegram groups 2-10 items)
        if photos:
            try:
                if len(photos) == 1:
                    with open(photos[0].path, "rb") as f:
                        await update.message.reply_photo(
                            photo=f,
                            reply_to_message_id=reply_to_message_id,
                            caption=caption if use_caption else None,
                            parse_mode=caption_parse_mode if use_caption else None,
                        )
                    caption_sent = use_caption
                else:
                    media = []
                    file_handles = []
                    for idx, img in enumerate(photos[:10]):
                        fh = open(img.path, "rb")  # noqa: SIM115
                        file_handles.append(fh)
                        media.append(
                            InputMediaPhoto(
                                media=fh,
                                caption=caption if use_caption and idx == 0 else None,
                                parse_mode=(
                                    caption_parse_mode
                                    if use_caption and idx == 0
                                    else None
                                ),
                            )
                        )
                    try:
                        await update.message.chat.send_media_group(
                            media=media,
                            reply_to_message_id=reply_to_message_id,
                        )
                        caption_sent = use_caption
                    finally:
                        for fh in file_handles:
                            fh.close()
            except Exception as e:
                logger.warning("Failed to send photo album", error=str(e))

        # Send SVGs / large files as documents (one by one — can't mix in album)
        for img in documents:
            try:
                with open(img.path, "rb") as f:
                    await update.message.reply_document(
                        document=f,
                        filename=img.path.name,
                        reply_to_message_id=reply_to_message_id,
                    )
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning(
                    "Failed to send document image",
                    path=str(img.path),
                    error=str(e),
                )

        return caption_sent

    async def agentic_text(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        prompt_override: Optional[str] = None,
    ) -> None:
        """Direct Claude passthrough. Simple progress. No suggestions.

        prompt_override — текст задачи вместо текста сообщения. Нужен командам,
        которые сами формулируют задачу (например /newproject): объекты
        сообщений в python-telegram-bot неизменяемые, подменить текст нельзя.
        """
        user_id = update.effective_user.id
        message_text = prompt_override or update.message.text

        logger.info(
            "Agentic text message",
            user_id=user_id,
            message_length=len(message_text),
        )

        # Rate limit check
        rate_limiter = context.bot_data.get("rate_limiter")
        if rate_limiter:
            allowed, limit_message = await rate_limiter.check_rate_limit(user_id, 0.001)
            if not allowed:
                await update.message.reply_text(f"⏱️ {limit_message}")
                return

        chat = update.message.chat
        await chat.send_action("typing")

        verbose_level = self._get_verbose_level(context)

        # Create Stop button and interrupt event
        interrupt_event = asyncio.Event()
        stop_kb = InlineKeyboardMarkup(
            [[InlineKeyboardButton("⏹ Остановить", callback_data=f"stop:{user_id}")]]
        )
        progress_msg = await update.message.reply_text(
            f"{_SPINNER[0]} <b>Работаю</b> · 0 сек\n\n<i>обдумываю задачу…</i>",
            reply_markup=stop_kb,
            parse_mode="HTML",
        )

        # Register active request for stop callback
        active_request = ActiveRequest(
            user_id=user_id,
            interrupt_event=interrupt_event,
            progress_msg=progress_msg,
        )
        self._active_requests[user_id] = active_request

        claude_integration = context.bot_data.get("claude_integration")
        if not claude_integration:
            self._active_requests.pop(user_id, None)
            await progress_msg.edit_text(
                "⚠️ Claude недоступен — проверьте настройки бота.",
                reply_markup=None,
            )
            return

        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        session_id = context.user_data.get("claude_session_id")

        # Check if /new was used — skip auto-resume for this first message.
        # Flag is only cleared after a successful run so retries keep the intent.
        force_new = bool(context.user_data.get("force_new_session"))

        # --- Verbose progress tracking via stream callback ---
        tool_log: List[Dict[str, Any]] = []
        start_time = time.time()
        mcp_images: List[ImageAttachment] = []

        # Stream drafts (private chats only)
        draft_streamer: Optional[DraftStreamer] = None
        if self.settings.enable_stream_drafts and chat.type == "private":
            draft_streamer = DraftStreamer(
                bot=context.bot,
                chat_id=chat.id,
                draft_id=generate_draft_id(),
                message_thread_id=update.message.message_thread_id,
                throttle_interval=self.settings.stream_draft_interval,
            )

        on_stream = self._make_stream_callback(
            verbose_level,
            progress_msg,
            tool_log,
            start_time,
            reply_markup=stop_kb,
            mcp_images=mcp_images,
            approved_directory=self.settings.approved_directory,
            draft_streamer=draft_streamer,
            interrupt_event=interrupt_event,
        )

        # Independent typing heartbeat — stays alive even with no stream events
        heartbeat = self._start_typing_heartbeat(chat)
        # Живой индикатор: двигается даже когда Claude долго думает молча,
        # иначе сообщение замирает и кажется, что бот завис.
        ticker = (
            self._start_progress_ticker(
                progress_msg, tool_log, verbose_level, start_time, stop_kb
            )
            if verbose_level >= 1 and draft_streamer is None
            else None
        )

        success = True
        try:
            claude_response = await claude_integration.run_command(
                prompt=message_text,
                working_directory=current_dir,
                user_id=user_id,
                session_id=session_id,
                on_stream=on_stream,
                force_new=force_new,
                interrupt_event=interrupt_event,
                overrides=self._user_overrides(context),
            )

            # New session created successfully — clear the one-shot flag
            if force_new:
                context.user_data["force_new_session"] = False

            context.user_data["claude_session_id"] = claude_response.session_id

            # NOTE: upstream re-parsed Claude's reply for `cd ...` here and
            # switched the active project on any match — even a `cd` inside an
            # explanation or a code sample. The project is switched only by the
            # explicit /repo command now.

            # Store interaction
            storage = context.bot_data.get("storage")
            if storage:
                try:
                    await storage.save_claude_interaction(
                        user_id=user_id,
                        session_id=claude_response.session_id,
                        prompt=message_text,
                        response=claude_response,
                        ip_address=None,
                    )
                except Exception as e:
                    logger.warning("Failed to log interaction", error=str(e))

            # Format response (no reply_markup — strip keyboards)
            from .utils.formatting import ResponseFormatter

            formatter = ResponseFormatter(self.settings)

            response_content = claude_response.content
            if claude_response.interrupted:
                response_content = (
                    response_content or ""
                ) + "\n\n_(остановлено вами)_"

            formatted_messages = formatter.format_claude_response(response_content)

        except Exception as e:
            success = False
            logger.error("Claude integration failed", error=str(e), user_id=user_id)
            from .handlers.message import _format_error_message
            from .utils.formatting import FormattedMessage

            formatted_messages = [
                FormattedMessage(_format_error_message(e), parse_mode="HTML")
            ]
        finally:
            heartbeat.cancel()
            if ticker is not None:
                ticker.cancel()
            self._active_requests.pop(user_id, None)
            if draft_streamer:
                try:
                    await draft_streamer.flush()
                except Exception:
                    logger.debug("Draft flush failed in finally block", user_id=user_id)

        try:
            await progress_msg.delete()
        except Exception:
            logger.debug("Failed to delete progress message, ignoring")

        # Use MCP-collected images (from send_image_to_user tool calls)
        images: List[ImageAttachment] = mcp_images

        # Try to combine text + images in one message when possible
        caption_sent = False
        if images and len(formatted_messages) == 1:
            msg = formatted_messages[0]
            if msg.text and len(msg.text) <= 1024:
                try:
                    caption_sent = await self._send_images(
                        update,
                        images,
                        reply_to_message_id=update.message.message_id,
                        caption=msg.text,
                        caption_parse_mode=msg.parse_mode,
                    )
                except Exception as img_err:
                    logger.warning("Image+caption send failed", error=str(img_err))

        # Send text messages (skip if caption was already embedded in photos)
        if not caption_sent:
            for i, message in enumerate(formatted_messages):
                if not message.text or not message.text.strip():
                    continue
                try:
                    await update.message.reply_text(
                        message.text,
                        parse_mode=message.parse_mode,
                        reply_markup=None,  # No keyboards in agentic mode
                        reply_to_message_id=(
                            update.message.message_id if i == 0 else None
                        ),
                    )
                    if i < len(formatted_messages) - 1:
                        await asyncio.sleep(0.5)
                except Exception as send_err:
                    logger.warning(
                        "Failed to send HTML response, retrying as plain text",
                        error=str(send_err),
                        message_index=i,
                    )
                    try:
                        await update.message.reply_text(
                            message.text,
                            reply_markup=None,
                            reply_to_message_id=(
                                update.message.message_id if i == 0 else None
                            ),
                        )
                    except Exception as plain_err:
                        await update.message.reply_text(
                            f"Failed to deliver response "
                            f"(Telegram error: {str(plain_err)[:150]}). "
                            f"Please try again.",
                            reply_to_message_id=(
                                update.message.message_id if i == 0 else None
                            ),
                        )

            # Send images separately if caption wasn't used
            if images:
                try:
                    await self._send_images(
                        update,
                        images,
                        reply_to_message_id=update.message.message_id,
                    )
                except Exception as img_err:
                    logger.warning("Image send failed", error=str(img_err))

        # Audit log
        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=user_id,
                command="text_message",
                args=[message_text[:100]],
                success=success,
            )

    async def agentic_document(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Process file upload -> Claude, minimal chrome."""
        user_id = update.effective_user.id
        document = update.message.document

        logger.info(
            "Agentic document upload",
            user_id=user_id,
            filename=document.file_name,
        )

        # Security validation
        security_validator = context.bot_data.get("security_validator")
        if security_validator:
            valid, error = security_validator.validate_filename(document.file_name)
            if not valid:
                await update.message.reply_text(f"File rejected: {error}")
                return

        # Size check
        max_size = 10 * 1024 * 1024
        if document.file_size > max_size:
            await update.message.reply_text(
                f"File too large ({document.file_size / 1024 / 1024:.1f}MB). Max: 10MB."
            )
            return

        chat = update.message.chat
        await chat.send_action("typing")
        progress_msg = await update.message.reply_text(f"{_SPINNER[0]} Работаю…")

        # Try enhanced file handler, fall back to basic
        features = context.bot_data.get("features")
        file_handler = features.get_file_handler() if features else None
        prompt: Optional[str] = None

        if file_handler:
            try:
                processed_file = await file_handler.handle_document_upload(
                    document,
                    user_id,
                    update.message.caption or "Please review this file:",
                )
                prompt = processed_file.prompt
            except Exception:
                file_handler = None

        if not file_handler:
            file = await document.get_file()
            file_bytes = await file.download_as_bytearray()
            try:
                content = file_bytes.decode("utf-8")
                if len(content) > 50000:
                    content = content[:50000] + "\n... (truncated)"
                caption = update.message.caption or "Please review this file:"
                prompt = (
                    f"{caption}\n\n**File:** `{document.file_name}`\n\n"
                    f"```\n{content}\n```"
                )
            except UnicodeDecodeError:
                await progress_msg.edit_text(
                    "Unsupported file format. Must be text-based (UTF-8)."
                )
                return

        # Process with Claude
        claude_integration = context.bot_data.get("claude_integration")
        if not claude_integration:
            await progress_msg.edit_text(
                "Claude integration not available. Check configuration."
            )
            return

        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        session_id = context.user_data.get("claude_session_id")

        # Check if /new was used — skip auto-resume for this first message.
        # Flag is only cleared after a successful run so retries keep the intent.
        force_new = bool(context.user_data.get("force_new_session"))

        verbose_level = self._get_verbose_level(context)
        tool_log: List[Dict[str, Any]] = []
        mcp_images_doc: List[ImageAttachment] = []
        on_stream = self._make_stream_callback(
            verbose_level,
            progress_msg,
            tool_log,
            time.time(),
            mcp_images=mcp_images_doc,
            approved_directory=self.settings.approved_directory,
        )

        heartbeat = self._start_typing_heartbeat(chat)
        try:
            claude_response = await claude_integration.run_command(
                prompt=prompt,
                working_directory=current_dir,
                user_id=user_id,
                session_id=session_id,
                on_stream=on_stream,
                force_new=force_new,
                overrides=self._user_overrides(context),
            )

            if force_new:
                context.user_data["force_new_session"] = False

            context.user_data["claude_session_id"] = claude_response.session_id

            # NOTE: upstream re-parsed Claude's reply for `cd ...` here and
            # switched the active project on any match — even a `cd` inside an
            # explanation or a code sample. The project is switched only by the
            # explicit /repo command now.

            from .utils.formatting import ResponseFormatter

            formatter = ResponseFormatter(self.settings)
            formatted_messages = formatter.format_claude_response(
                claude_response.content
            )

            try:
                await progress_msg.delete()
            except Exception:
                logger.debug("Failed to delete progress message, ignoring")

            # Use MCP-collected images (from send_image_to_user tool calls)
            images: List[ImageAttachment] = mcp_images_doc

            caption_sent = False
            if images and len(formatted_messages) == 1:
                msg = formatted_messages[0]
                if msg.text and len(msg.text) <= 1024:
                    try:
                        caption_sent = await self._send_images(
                            update,
                            images,
                            reply_to_message_id=update.message.message_id,
                            caption=msg.text,
                            caption_parse_mode=msg.parse_mode,
                        )
                    except Exception as img_err:
                        logger.warning("Image+caption send failed", error=str(img_err))

            if not caption_sent:
                for i, message in enumerate(formatted_messages):
                    await update.message.reply_text(
                        message.text,
                        parse_mode=message.parse_mode,
                        reply_markup=None,
                        reply_to_message_id=(
                            update.message.message_id if i == 0 else None
                        ),
                    )
                    if i < len(formatted_messages) - 1:
                        await asyncio.sleep(0.5)

                if images:
                    try:
                        await self._send_images(
                            update,
                            images,
                            reply_to_message_id=update.message.message_id,
                        )
                    except Exception as img_err:
                        logger.warning("Image send failed", error=str(img_err))

        except Exception as e:
            from .handlers.message import _format_error_message

            await progress_msg.edit_text(_format_error_message(e), parse_mode="HTML")
            logger.error("Claude file processing failed", error=str(e), user_id=user_id)
        finally:
            heartbeat.cancel()

    async def agentic_photo(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Process photo -> Claude, minimal chrome."""
        user_id = update.effective_user.id

        features = context.bot_data.get("features")
        image_handler = features.get_image_handler() if features else None

        if not image_handler:
            await update.message.reply_text("⚠️ Обработка фото недоступна.")
            return

        chat = update.message.chat
        await chat.send_action("typing")
        progress_msg = await update.message.reply_text(f"{_SPINNER[0]} Работаю…")

        try:
            photo = update.message.photo[-1]
            processed_image = await image_handler.process_image(
                photo, update.message.caption
            )
            fmt = processed_image.metadata.get("format", "png")
            images = [
                {
                    "data": processed_image.base64_data,
                    "media_type": _MEDIA_TYPE_MAP.get(fmt, "image/png"),
                }
            ]

            await self._handle_agentic_media_message(
                update=update,
                context=context,
                prompt=processed_image.prompt,
                progress_msg=progress_msg,
                user_id=user_id,
                chat=chat,
                images=images,
            )

        except Exception as e:
            from .handlers.message import _format_error_message

            await progress_msg.edit_text(_format_error_message(e), parse_mode="HTML")
            logger.error(
                "Claude photo processing failed", error=str(e), user_id=user_id
            )

    async def agentic_voice(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Transcribe voice message -> Claude, minimal chrome."""
        user_id = update.effective_user.id

        features = context.bot_data.get("features")
        voice_handler = features.get_voice_handler() if features else None

        if not voice_handler:
            await update.message.reply_text(self._voice_unavailable_message())
            return

        chat = update.message.chat
        await chat.send_action("typing")
        progress_msg = await update.message.reply_text("🎧 Распознаю голосовое…")

        try:
            voice = update.message.voice
            processed_voice = await voice_handler.process_voice_message(
                voice, update.message.caption
            )

            await progress_msg.edit_text("Working...")
            await self._handle_agentic_media_message(
                update=update,
                context=context,
                prompt=processed_voice.prompt,
                progress_msg=progress_msg,
                user_id=user_id,
                chat=chat,
            )

        except Exception as e:
            from .handlers.message import _format_error_message

            await progress_msg.edit_text(_format_error_message(e), parse_mode="HTML")
            logger.error(
                "Claude voice processing failed", error=str(e), user_id=user_id
            )

    async def _handle_agentic_media_message(
        self,
        *,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        prompt: str,
        progress_msg: Any,
        user_id: int,
        chat: Any,
        images: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        """Run a media-derived prompt through Claude and send responses."""
        claude_integration = context.bot_data.get("claude_integration")
        if not claude_integration:
            await progress_msg.edit_text(
                "Claude integration not available. Check configuration."
            )
            return

        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        session_id = context.user_data.get("claude_session_id")
        force_new = bool(context.user_data.get("force_new_session"))

        verbose_level = self._get_verbose_level(context)
        tool_log: List[Dict[str, Any]] = []
        mcp_images_media: List[ImageAttachment] = []
        on_stream = self._make_stream_callback(
            verbose_level,
            progress_msg,
            tool_log,
            time.time(),
            mcp_images=mcp_images_media,
            approved_directory=self.settings.approved_directory,
        )

        heartbeat = self._start_typing_heartbeat(chat)
        try:
            claude_response = await claude_integration.run_command(
                prompt=prompt,
                working_directory=current_dir,
                user_id=user_id,
                session_id=session_id,
                on_stream=on_stream,
                force_new=force_new,
                images=images,
                overrides=self._user_overrides(context),
            )
        finally:
            heartbeat.cancel()

        if force_new:
            context.user_data["force_new_session"] = False

        context.user_data["claude_session_id"] = claude_response.session_id

        # NOTE: see the note above — no directory switching from reply text.

        from .utils.formatting import ResponseFormatter

        formatter = ResponseFormatter(self.settings)
        formatted_messages = formatter.format_claude_response(claude_response.content)

        try:
            await progress_msg.delete()
        except Exception:
            logger.debug("Failed to delete progress message, ignoring")

        # Use MCP-collected images (from send_image_to_user tool calls).
        images: List[ImageAttachment] = mcp_images_media

        caption_sent = False
        if images and len(formatted_messages) == 1:
            msg = formatted_messages[0]
            if msg.text and len(msg.text) <= 1024:
                try:
                    caption_sent = await self._send_images(
                        update,
                        images,
                        reply_to_message_id=update.message.message_id,
                        caption=msg.text,
                        caption_parse_mode=msg.parse_mode,
                    )
                except Exception as img_err:
                    logger.warning("Image+caption send failed", error=str(img_err))

        if not caption_sent:
            for i, message in enumerate(formatted_messages):
                if not message.text or not message.text.strip():
                    continue
                await update.message.reply_text(
                    message.text,
                    parse_mode=message.parse_mode,
                    reply_markup=None,
                    reply_to_message_id=(update.message.message_id if i == 0 else None),
                )
                if i < len(formatted_messages) - 1:
                    await asyncio.sleep(0.5)

            if images:
                try:
                    await self._send_images(
                        update,
                        images,
                        reply_to_message_id=update.message.message_id,
                    )
                except Exception as img_err:
                    logger.warning("Image send failed", error=str(img_err))

    async def _handle_unknown_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Forward unknown slash commands to Claude in agentic mode.

        Known commands are handled by their own CommandHandlers (group 0);
        this handler fires for *every* COMMAND message in group 10 but
        returns immediately when the command is registered, preventing
        double execution.
        """
        msg = update.effective_message
        if not msg or not msg.text:
            return
        cmd = msg.text.split()[0].lstrip("/").split("@")[0].lower()
        if cmd in self._known_commands:
            return  # let the registered CommandHandler take care of it
        # Forward unrecognised /commands to Claude as natural language
        await self.agentic_text(update, context)

    def _voice_unavailable_message(self) -> str:
        """Return provider-aware guidance when voice feature is unavailable."""
        if self.settings.voice_provider == "local":
            return (
                "Voice processing is not available. "
                "Ensure whisper.cpp is installed and the model file exists. "
                "Check WHISPER_CPP_BINARY_PATH and WHISPER_CPP_MODEL_PATH settings."
            )
        return (
            "Voice processing is not available. "
            f"Set {self.settings.voice_provider_api_key_env} "
            f"for {self.settings.voice_provider_display_name} and install "
            'voice extras with: pip install "claude-code-telegram[voice]"'
        )

    async def agentic_newproject(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Создать новый проект: /newproject имя-проекта [описание]."""
        text = update.message.text or ""
        args = text.split()[1:]
        base = self.settings.approved_directory

        if not args:
            await update.message.reply_text(
                "Создать новый проект:\n"
                "<code>/newproject имя-проекта</code>\n"
                "<code>/newproject имя-проекта Короткое описание</code>\n\n"
                "Имя — латиницей, без пробелов, например <code>my-landing</code>.\n"
                "Проект появится и на сервере, и на GitHub (приватным), "
                "и его можно будет открыть на компьютере.",
                parse_mode="HTML",
            )
            return

        name = args[0].lower()
        description = " ".join(args[1:]).strip()

        # Имя станет и папкой, и репозиторием — пускаем только безопасные символы.
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,60}", name):
            await update.message.reply_text(
                "Имя должно быть латиницей без пробелов: буквы, цифры, дефис.\n"
                "Например: <code>my-landing</code>",
                parse_mode="HTML",
            )
            return

        target = base / name
        if target.exists():
            await update.message.reply_text(
                f"Проект <code>{escape_html(name)}</code> уже есть. "
                f"Открыть: <code>/repo {escape_html(name)}</code>",
                parse_mode="HTML",
            )
            return

        await update.message.reply_text(
            f"Создаю проект <b>{escape_html(name)}</b>…", parse_mode="HTML"
        )

        # Дальше работает сам Claude: он умеет и git, и gh. Так создание проекта
        # проходит тем же путём, что и любая другая задача, — с отчётом в чат.
        desc_part = f' с описанием "{description}"' if description else ""
        context.user_data["current_directory"] = base
        prompt = (
            f"Создай новый проект «{name}»{desc_part}. По шагам:\n"
            f"1. Создай папку {target} и перейди в неё.\n"
            f"2. Положи README.md с названием проекта и парой строк о нём.\n"
            f"3. Положи .gitignore, закрывающий: .env, secrets/, *.key, "
            f"node_modules/, __pycache__/, .venv/, *.db, .DS_Store\n"
            f"4. Положи CLAUDE.md с краткой памяткой по проекту.\n"
            f"5. Выполни: git init, git branch -M main, git add -A, "
            f'git commit -m "Начало проекта"\n'
            f"6. Выполни: gh repo create {name} --private --source=. "
            f"--remote=origin --push\n"
            f"7. Коротко отчитайся, что получилось, и напомни команду "
            f"для открытия проекта на компьютере: "
            f'gh repo clone {name} "Имя папки"'
        )

        # Отдаём Claude как обычную задачу — со стримингом и отчётом в чат.
        await self.agentic_text(update, context, prompt_override=prompt)

    async def agentic_repo(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """List repos in workspace or switch to one.

        /repo          — list subdirectories with git indicators
        /repo <name>   — switch to that directory, resume session if available
        """
        args = update.message.text.split()[1:] if update.message.text else []
        base = self.settings.approved_directory
        current_dir = context.user_data.get("current_directory", base)

        if args:
            # Switch to named repo
            target_name = args[0]
            target_path = base / target_name
            if not target_path.is_dir():
                await update.message.reply_text(
                    f"Directory not found: <code>{escape_html(target_name)}</code>",
                    parse_mode="HTML",
                )
                return

            context.user_data["current_directory"] = target_path

            # Try to find a resumable session
            claude_integration = context.bot_data.get("claude_integration")
            session_id = None
            if claude_integration:
                existing = await claude_integration._find_resumable_session(
                    update.effective_user.id, target_path
                )
                if existing:
                    session_id = existing.session_id
            context.user_data["claude_session_id"] = session_id

            is_git = (target_path / ".git").is_dir()
            git_badge = " (git)" if is_git else ""
            session_badge = " · session resumed" if session_id else ""

            await update.message.reply_text(
                f"Switched to <code>{escape_html(target_name)}/</code>"
                f"{git_badge}{session_badge}",
                parse_mode="HTML",
            )
            return

        # No args — list repos
        try:
            entries = sorted(
                [
                    d
                    for d in base.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                ],
                key=lambda d: d.name,
            )
        except OSError as e:
            await update.message.reply_text(f"Error reading workspace: {e}")
            return

        if not entries:
            await update.message.reply_text(
                f"No repos in <code>{escape_html(str(base))}</code>.\n"
                'Clone one by telling me, e.g. <i>"clone org/repo"</i>.',
                parse_mode="HTML",
            )
            return

        lines: List[str] = []
        keyboard_rows: List[list] = []  # type: ignore[type-arg]
        current_name = current_dir.name if current_dir != base else None

        for d in entries:
            is_git = (d / ".git").is_dir()
            icon = "\U0001f4e6" if is_git else "\U0001f4c1"
            marker = " \u25c0" if d.name == current_name else ""
            lines.append(f"{icon} <code>{escape_html(d.name)}/</code>{marker}")

        # Build inline keyboard (2 per row)
        for i in range(0, len(entries), 2):
            row = []
            for j in range(2):
                if i + j < len(entries):
                    name = entries[i + j].name
                    row.append(InlineKeyboardButton(name, callback_data=f"cd:{name}"))
            keyboard_rows.append(row)

        reply_markup = InlineKeyboardMarkup(keyboard_rows)

        await update.message.reply_text(
            "<b>Repos</b>\n\n" + "\n".join(lines),
            parse_mode="HTML",
            reply_markup=reply_markup,
        )

    async def _handle_stop_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle stop: callbacks — interrupt a running Claude request."""
        query = update.callback_query
        target_user_id = int(query.data.split(":", 1)[1])

        # Only the requesting user can stop their own request
        if query.from_user.id != target_user_id:
            await query.answer(
                "Остановить может только тот, кто дал задачу.", show_alert=True
            )
            return

        active = self._active_requests.get(target_user_id)
        if not active:
            await query.answer("Уже готово.", show_alert=False)
            return
        if active.interrupted:
            await query.answer("Уже останавливаю…", show_alert=False)
            return

        active.interrupt_event.set()
        active.interrupted = True
        await query.answer("Останавливаю…", show_alert=False)

        try:
            await active.progress_msg.edit_text("⏹ Останавливаю…", reply_markup=None)
        except Exception:
            pass

    async def _agentic_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle cd: callbacks — switch directory and resume session if available."""
        query = update.callback_query
        await query.answer()

        data = query.data
        _, project_name = data.split(":", 1)

        base = self.settings.approved_directory
        new_path = base / project_name

        if not new_path.is_dir():
            await query.edit_message_text(
                f"Directory not found: <code>{escape_html(project_name)}</code>",
                parse_mode="HTML",
            )
            return

        context.user_data["current_directory"] = new_path

        # Look for a resumable session instead of always clearing
        claude_integration = context.bot_data.get("claude_integration")
        session_id = None
        if claude_integration:
            existing = await claude_integration._find_resumable_session(
                query.from_user.id, new_path
            )
            if existing:
                session_id = existing.session_id
        context.user_data["claude_session_id"] = session_id

        is_git = (new_path / ".git").is_dir()
        git_badge = " (git)" if is_git else ""
        session_badge = " · session resumed" if session_id else ""

        await query.edit_message_text(
            f"Switched to <code>{escape_html(project_name)}/</code>"
            f"{git_badge}{session_badge}",
            parse_mode="HTML",
        )

        # Audit log
        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=query.from_user.id,
                command="cd",
                args=[project_name],
                success=True,
            )
