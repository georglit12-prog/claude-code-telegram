"""Bash directory boundary enforcement for Claude tool calls."""

import re
import shlex
from pathlib import Path
from typing import Optional, Set, Tuple

# Subdirectories under ~/.claude/ that Claude Code uses internally.
# "projects" holds Claude Code's own session transcripts and per-project memory.
# Without it every such write is denied and the reply fills up with refusals.
# "skills" holds the skill instructions themselves: the Skill tool reads
# SKILL.md and its helper files, and without this entry every skill is listed
# but fails to run ("Execute skill: <name>").
_CLAUDE_INTERNAL_SUBDIRS: Set[str] = {
    "plans",
    "todos",
    "projects",
    "skills",
    "settings.json",
}

# Commands that modify the filesystem or change context and should have paths checked
_FS_MODIFYING_COMMANDS: Set[str] = {
    "mkdir",
    "touch",
    "cp",
    "mv",
    "rm",
    "rmdir",
    "ln",
    "install",
    "tee",
    "cd",
}

# Commands that are read-only or don't take filesystem paths
_READ_ONLY_COMMANDS: Set[str] = {
    "cat",
    "ls",
    "head",
    "tail",
    "less",
    "more",
    "which",
    "whoami",
    "pwd",
    "echo",
    "printf",
    "env",
    "printenv",
    "date",
    "wc",
    "sort",
    "uniq",
    "diff",
    "file",
    "stat",
    "du",
    "df",
    "tree",
    "realpath",
    "dirname",
    "basename",
}

# Actions / expressions that make ``find`` a filesystem-modifying command
_FIND_MUTATING_ACTIONS: Set[str] = {"-delete", "-exec", "-execdir", "-ok", "-okdir"}

# Bash command separators. A newline separates commands exactly like ``;``
# does — Claude routinely sends multi-line scripts, and without this a
# perfectly normal second line is parsed as more arguments to the first.
_COMMAND_SEPARATORS: Set[str] = {"&&", "||", ";", "|", "&", "\n"}

# Redirection operators. The token *after* one of these is a filename for
# I/O redirection (``> out.log``, ``2>&1``, ``< /dev/null``), never a
# filesystem argument of the command itself — ``cd foo > /dev/null`` does
# not touch ``/dev/null`` as a directory. We drop the operator and skip the
# very next token when scanning for paths.
_REDIRECT_OPERATORS: Set[str] = {">", ">>", "<", "<>", "2>", "2>>", "&>", "&>>"}

# Matches a heredoc/herestring start such as ``<<'PY'``, ``<<EOF`` or
# ``<<- "EOF"``: the operator, optional ``-``, optional quotes, and the
# delimiter word. Everything up to the matching end-of-line delimiter is
# data piped to the command's stdin (Python source, JSON, …), not shell
# syntax — paths mentioned inside it are not filesystem arguments of any
# bash command and must not be checked or chained into the surrounding one.
_HEREDOC_START_RE = re.compile(r"<<-?\s*(['\"]?)(\w+)\1")


def _strip_heredocs(command: str) -> str:
    """Remove heredoc bodies from *command*, replacing each with a placeholder.

    Operates line by line so a delimiter word appearing later as plain text
    (e.g. inside a normal argument) can't be mistaken for the start of a new
    heredoc once we're already inside one.
    """
    out_lines: list[str] = []
    lines = command.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        match = _HEREDOC_START_RE.search(line)
        if match:
            delimiter = match.group(2)
            # Keep everything on the start line up to the heredoc operator —
            # it may itself contain a real command (``cmd <<'EOF'``).
            out_lines.append(line[: match.start()])
            i += 1
            while i < len(lines) and lines[i].strip() != delimiter:
                i += 1
            i += 1  # skip the delimiter line itself
        else:
            out_lines.append(line)
            i += 1
    return "\n".join(out_lines)


def check_bash_directory_boundary(
    command: str,
    working_directory: Path,
    approved_directory: Path,
) -> Tuple[bool, Optional[str]]:
    """Check if a bash command's paths stay within the approved directory."""
    command = _strip_heredocs(command)

    # shlex treats newlines as ordinary whitespace, same as spaces, so a
    # multi-line script would otherwise collapse into one giant argument
    # list — Line 2 of ``cd proj\nsleep 5`` would look like more arguments
    # to ``cd``. Splitting per line first keeps each line's own tokens
    # together; ``\n`` is then also a separator (see _COMMAND_SEPARATORS)
    # so command chains still can't cross line boundaries.
    tokens: list[str] = []
    for line in command.split("\n"):
        try:
            line_tokens = shlex.split(line, comments=False)
        except ValueError:
            # Malformed quoting on this line (e.g. an unbalanced quote that
            # continues on the next line). Let it through — the sandbox
            # catches it at the OS level — rather than mis-tokenize it.
            return True, None
        tokens.extend(line_tokens)
        tokens.append("\n")

    if not tokens or not any(t != "\n" for t in tokens):
        return True, None

    # Split tokens into individual commands based on separators
    command_chains: list[list[str]] = []
    current_chain: list[str] = []

    for token in tokens:
        if token in _COMMAND_SEPARATORS:
            if current_chain:
                command_chains.append(current_chain)
            current_chain = []
        else:
            current_chain.append(token)

    if current_chain:
        command_chains.append(current_chain)

    resolved_approved = approved_directory.resolve()

    # Check each command in the chain
    for cmd_tokens in command_chains:
        if not cmd_tokens:
            continue

        base_command = Path(cmd_tokens[0]).name

        # Read-only commands are always allowed
        if base_command in _READ_ONLY_COMMANDS:
            continue

        # Determine if this specific command in the chain needs path validation
        needs_check = False
        if base_command == "find":
            needs_check = any(t in _FIND_MUTATING_ACTIONS for t in cmd_tokens[1:])
        elif base_command in _FS_MODIFYING_COMMANDS:
            needs_check = True

        if not needs_check:
            continue

        # Check each argument for paths outside the boundary
        skip_next = False
        for token in cmd_tokens[1:]:
            if skip_next:
                # This token is a redirection target (``> file``, ``< file``,
                # ``2>&1``), not a filesystem argument of the command itself.
                skip_next = False
                continue
            if token in _REDIRECT_OPERATORS:
                skip_next = True
                continue
            # A combined form like ``2>file`` or ``>out.log`` with no space —
            # shlex keeps it as one token, so there is nothing further to
            # check on this token; it targets stderr/stdout redirection, not
            # a real path argument of the command.
            if re.fullmatch(r"\d*(>>?|<>?)\S*", token):
                continue

            # Skip flags
            if token.startswith("-"):
                continue

            # Resolve both absolute and relative paths against the working
            # directory so that traversal sequences like ``../../evil`` are
            # caught instead of being silently allowed.
            try:
                if token.startswith("/"):
                    resolved = Path(token).resolve()
                else:
                    resolved = (working_directory / token).resolve()

                if not _is_within_directory(resolved, resolved_approved):
                    return False, (
                        f"Directory boundary violation: '{base_command}' targets "
                        f"'{token}' which is outside approved directory "
                        f"'{resolved_approved}'"
                    )
            except (ValueError, OSError):
                # If path resolution fails, the command might be malformed or
                # using bash features we can't statically analyze.
                # We skip checking this token and rely on the OS-level sandbox.
                continue

    return True, None


def _is_claude_internal_path(file_path: str) -> bool:
    """Check whether *file_path* points inside ``~/.claude/`` (allowed subdirs only)."""
    try:
        resolved = Path(file_path).resolve()
        home = Path.home().resolve()
        claude_dir = home / ".claude"

        # Path must be inside ~/.claude/
        try:
            rel = resolved.relative_to(claude_dir)
        except ValueError:
            return False

        # Must be in one of the known subdirectories (or a known file)
        top_part = rel.parts[0] if rel.parts else ""
        return top_part in _CLAUDE_INTERNAL_SUBDIRS

    except Exception:
        return False


def _is_within_directory(path: Path, directory: Path) -> bool:
    """Check if path is within directory."""
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False
