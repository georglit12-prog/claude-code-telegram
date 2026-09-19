"""Что считать системным действием, требующим подтверждения владельца.

Бот работает с правами root: он может завести проект в ``/srv``, поднять
службу, поправить nginx. Всё это — настоящая работа, ради которой права и
выдавались. Но на том же сервере живут сайт и воронка с оплатой, поэтому
действия за пределами рабочей папки проектов не выполняются молча: бот
спрашивает владельца кнопкой в Telegram и ждёт ответа.

Правило простое: **разрушительное и системное — спрашиваем, чтение — нет.**
Посмотреть состояние службы, прочитать чужой конфиг, поискать файл можно
без вопросов: это ничего не ломает, а вопрос на каждый ``cat`` сделал бы
работу невыносимой и приучил бы нажимать «да» не глядя.

Модуль только классифицирует. Спрашивает и ждёт ответа ``can_use_tool``
в :mod:`src.claude.sdk_integration`.
"""

from pathlib import Path
from typing import Optional, Set

from .monitor import split_command_chains

# Папки, за которыми на этом сервере стоят деньги и лицо бизнеса. Причина
# подтверждения называет их прямо: владелец должен видеть не «путь вне
# рабочей папки», а «это воронка с оплатой».
_SENSITIVE_PATHS = (
    ("/srv/funnel", "воронка с оплатой — через неё принимаются деньги"),
    ("/srv/site", "сайт и приём заявок"),
    ("/srv/photo-bot", "бот обработки фото"),
    ("/srv/claude-bot/secrets", "секреты бота: токены и ключи"),
    ("/etc", "системные настройки"),
)

# Команды управления службами. Смотреть состояние безопасно, менять — нет.
_SERVICE_COMMANDS: Set[str] = {"systemctl", "service"}
_SERVICE_READ_ONLY: Set[str] = {
    "status",
    "is-active",
    "is-enabled",
    "is-failed",
    "list-units",
    "list-unit-files",
    "list-timers",
    "show",
    "cat",
    "get-default",
}

# Команды, которые меняют систему целиком. Подкоманда для чтения — ключ,
# значение — набор безопасных подкоманд (пусто = вся команда системная).
_SYSTEM_COMMANDS: dict[str, Set[str]] = {
    "apt": {"list", "search", "show", "policy"},
    "apt-get": set(),
    "dpkg": {"-l", "--list", "-s", "--status", "-L"},
    "snap": {"list", "info", "find"},
    "useradd": set(),
    "userdel": set(),
    "usermod": set(),
    "groupadd": set(),
    "groupdel": set(),
    "passwd": set(),
    "chown": set(),
    "chgrp": set(),
    "chmod": set(),
    "ufw": {"status", "show"},
    "iptables": {"-L", "--list", "-S"},
    "nft": {"list"},
    "nginx": {"-t", "-T", "-v", "-V"},
    "certbot": {"certificates"},
    "crontab": {"-l"},
    "mount": set(),
    "umount": set(),
    "mkfs": set(),
    "fdisk": {"-l"},
    "reboot": set(),
    "shutdown": set(),
    "halt": set(),
    "poweroff": set(),
    "init": set(),
    "kill": set(),
    "killall": set(),
    "pkill": set(),
}

# Команды, меняющие файлы. Для них проверяем, куда именно они пишут.
_WRITING_COMMANDS: Set[str] = {
    "mkdir",
    "rmdir",
    "touch",
    "cp",
    "mv",
    "rm",
    "ln",
    "install",
    "tee",
    "truncate",
    "dd",
    "rsync",
    "chattr",
    "sed",
}


def _describe_path(path: str) -> str:
    """Понятная причина для пути: чем именно он важен."""
    for prefix, what in _SENSITIVE_PATHS:
        if path == prefix or path.startswith(prefix + "/"):
            return f"{path} — {what}"
    return path


def _is_inside(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def classify_file_write(
    file_path: str,
    approved_directory: Path,
    working_directory: Optional[Path] = None,
) -> Optional[str]:
    """Нужно ли подтверждение на запись в файл.

    Возвращает причину для владельца или ``None``, если это обычная работа
    внутри рабочей папки проектов.
    """
    if not file_path:
        return None

    base = working_directory or approved_directory
    try:
        resolved = (
            Path(file_path).resolve()
            if file_path.startswith("/")
            else (base / file_path).resolve()
        )
    except (ValueError, OSError):
        return None

    if _is_inside(resolved, approved_directory.resolve()):
        return None

    return f"запись вне рабочей папки: {_describe_path(str(resolved))}"


def _classify_single(
    tokens: list[str],
    working_directory: Path,
    approved_directory: Path,
) -> Optional[str]:
    """Классифицировать одну команду из цепочки."""
    if not tokens:
        return None

    # sudo сам по себе — заявка на системные права.
    if Path(tokens[0]).name == "sudo":
        rest = " ".join(tokens[1:])
        return f"команда с правами root: sudo {rest}" if rest else "команда через sudo"

    name = Path(tokens[0]).name
    args = [t for t in tokens[1:] if not t.startswith("-")]
    flags = [t for t in tokens[1:] if t.startswith("-")]

    # Службы: смотреть можно, трогать — спрашиваем.
    if name in _SERVICE_COMMANDS:
        subcommand = args[0] if args else ""
        if subcommand in _SERVICE_READ_ONLY:
            return None
        return f"управление службами: {' '.join(tokens)}"

    # Системные команды со своим списком безопасных подкоманд.
    if name in _SYSTEM_COMMANDS:
        safe = _SYSTEM_COMMANDS[name]
        if safe:
            probe = set(args[:1]) | set(flags)
            if probe & safe:
                return None
        return f"системная команда: {' '.join(tokens)}"

    # Запись файлов — проверяем, куда.
    if name in _WRITING_COMMANDS:
        for token in tokens[1:]:
            if token.startswith("-"):
                continue
            reason = classify_file_write(token, approved_directory, working_directory)
            if reason:
                return f"{name}: {reason}"

    return None


def classify_bash_command(
    command: str,
    working_directory: Path,
    approved_directory: Path,
) -> Optional[str]:
    """Нужно ли подтверждение на bash-команду.

    Возвращает причину для владельца или ``None``, если это обычная работа.
    Проверяются все команды цепочки: опасное может стоять на любой строке
    и после любого ``&&``.
    """
    chains = split_command_chains(command)
    if not chains:
        # Не разобрали — не выдумываем причину; границы рабочей папки
        # проверяются отдельно, в check_bash_directory_boundary.
        return None

    for tokens in chains:
        reason = _classify_single(tokens, working_directory, approved_directory)
        if reason:
            return reason

    return None
