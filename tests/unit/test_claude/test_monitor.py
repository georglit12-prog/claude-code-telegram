"""Test bash directory boundary checking."""

from pathlib import Path
from unittest.mock import patch

from src.claude.monitor import (
    _is_claude_internal_path,
    check_bash_directory_boundary,
)


class TestCheckBashDirectoryBoundary:
    """Test the check_bash_directory_boundary function."""

    def setup_method(self) -> None:
        self.approved = Path("/root/projects")
        self.cwd = Path("/root/projects/myapp")

    def test_mkdir_outside_approved_directory(self) -> None:
        valid, error = check_bash_directory_boundary(
            "mkdir -p /root/web1", self.cwd, self.approved
        )
        assert not valid
        assert "directory boundary violation" in error.lower()
        assert "/root/web1" in error

    def test_mkdir_inside_approved_directory(self) -> None:
        valid, error = check_bash_directory_boundary(
            "mkdir -p /root/projects/newdir", self.cwd, self.approved
        )
        assert valid
        assert error is None

    def test_touch_outside_approved_directory(self) -> None:
        valid, error = check_bash_directory_boundary(
            "touch /tmp/evil.txt", self.cwd, self.approved
        )
        assert not valid
        assert "/tmp/evil.txt" in error

    def test_cp_outside_approved_directory(self) -> None:
        valid, error = check_bash_directory_boundary(
            "cp file.txt /etc/passwd", self.cwd, self.approved
        )
        assert not valid
        assert "/etc/passwd" in error

    def test_mv_outside_approved_directory(self) -> None:
        valid, error = check_bash_directory_boundary(
            "mv /root/projects/file.txt /tmp/file.txt", self.cwd, self.approved
        )
        assert not valid
        assert "/tmp/file.txt" in error

    def test_relative_paths_inside_approved_pass(self) -> None:
        valid, error = check_bash_directory_boundary(
            "mkdir -p subdir/nested", self.cwd, self.approved
        )
        assert valid
        assert error is None

    def test_relative_path_traversal_escaping_approved_dir(self) -> None:
        """mkdir ../../evil from /root/projects/myapp resolves to /root/evil."""
        valid, error = check_bash_directory_boundary(
            "mkdir ../../evil", self.cwd, self.approved
        )
        assert not valid
        assert "directory boundary violation" in error.lower()
        assert "../../evil" in error

    def test_relative_path_traversal_staying_inside_approved_dir(self) -> None:
        """mkdir ../sibling from /root/projects/myapp -> /root/projects/sibling (ok)."""
        valid, error = check_bash_directory_boundary(
            "mkdir ../sibling", self.cwd, self.approved
        )
        assert valid
        assert error is None

    def test_relative_path_dot_dot_at_boundary_root(self) -> None:
        """mkdir .. from approved root itself should be blocked."""
        cwd_at_root = Path("/root/projects")
        valid, error = check_bash_directory_boundary(
            "touch ../outside.txt", cwd_at_root, self.approved
        )
        assert not valid
        assert "directory boundary violation" in error.lower()

    def test_read_only_commands_pass(self) -> None:
        for cmd in ["cat /etc/hosts", "ls /tmp", "head /var/log/syslog"]:
            valid, error = check_bash_directory_boundary(cmd, self.cwd, self.approved)
            assert valid, f"Expected read-only command to pass: {cmd}"
            assert error is None

    def test_non_fs_commands_pass(self) -> None:
        """Commands not in the filesystem-modifying set pass through."""
        for cmd in ["python script.py", "node app.js", "cargo build"]:
            valid, error = check_bash_directory_boundary(cmd, self.cwd, self.approved)
            assert valid, f"Expected non-fs command to pass: {cmd}"
            assert error is None

    def test_empty_command(self) -> None:
        valid, error = check_bash_directory_boundary("", self.cwd, self.approved)
        assert valid
        assert error is None

    def test_flags_are_skipped(self) -> None:
        valid, error = check_bash_directory_boundary(
            "mkdir -p -v /root/projects/dir", self.cwd, self.approved
        )
        assert valid
        assert error is None

    def test_unparseable_command_passes_through(self) -> None:
        """Malformed quoting should pass through (sandbox catches it at OS level)."""
        valid, error = check_bash_directory_boundary(
            "mkdir 'unclosed quote", self.cwd, self.approved
        )
        assert valid
        assert error is None

    def test_rm_outside_approved_directory(self) -> None:
        valid, error = check_bash_directory_boundary(
            "rm /var/tmp/somefile", self.cwd, self.approved
        )
        assert not valid
        assert "/var/tmp/somefile" in error

    def test_ln_outside_approved_directory(self) -> None:
        valid, error = check_bash_directory_boundary(
            "ln -s /root/projects/file /tmp/link", self.cwd, self.approved
        )
        assert not valid
        assert "/tmp/link" in error

    # --- find command handling ---

    def test_find_without_mutating_flags_passes(self) -> None:
        """Plain find (read-only) should pass regardless of search path."""
        valid, error = check_bash_directory_boundary(
            "find /tmp -name '*.log'", self.cwd, self.approved
        )
        assert valid
        assert error is None

    def test_find_delete_outside_approved_dir(self) -> None:
        """find /tmp -delete should be blocked because /tmp is outside."""
        valid, error = check_bash_directory_boundary(
            "find /tmp -name '*.log' -delete", self.cwd, self.approved
        )
        assert not valid
        assert "directory boundary violation" in error.lower()
        assert "/tmp" in error

    def test_find_exec_outside_approved_dir(self) -> None:
        """find /var -exec rm {} ; should be blocked."""
        valid, error = check_bash_directory_boundary(
            "find /var -exec rm {} ;", self.cwd, self.approved
        )
        assert not valid
        assert "/var" in error

    def test_find_delete_inside_approved_dir(self) -> None:
        """find inside approved dir with -delete should pass."""
        valid, error = check_bash_directory_boundary(
            "find /root/projects/myapp -name '*.pyc' -delete",
            self.cwd,
            self.approved,
        )
        assert valid
        assert error is None

    def test_find_delete_relative_path_inside(self) -> None:
        """find . -delete from inside approved dir should pass."""
        valid, error = check_bash_directory_boundary(
            "find . -name '*.pyc' -delete", self.cwd, self.approved
        )
        assert valid
        assert error is None

    def test_find_execdir_outside_approved_dir(self) -> None:
        """find with -execdir outside approved dir should be blocked."""
        valid, error = check_bash_directory_boundary(
            "find /etc -execdir cat {} ;", self.cwd, self.approved
        )
        assert not valid
        assert "/etc" in error

    # --- cd and command chaining handling ---

    def test_cd_outside_approved_directory(self) -> None:
        """cd to an outside directory should be blocked."""
        valid, error = check_bash_directory_boundary("cd /tmp", self.cwd, self.approved)
        assert not valid
        assert "directory boundary violation" in error.lower()
        assert "/tmp" in error

    def test_cd_inside_approved_directory(self) -> None:
        """cd to an inside directory should pass."""
        valid, error = check_bash_directory_boundary(
            "cd subdir", self.cwd, self.approved
        )
        assert valid
        assert error is None

    def test_chained_commands_outside_blocked(self) -> None:
        """Any command in a chain targeting outside should be blocked."""
        # Chained with &&
        valid, error = check_bash_directory_boundary(
            "ls && rm /etc/passwd", self.cwd, self.approved
        )
        assert not valid
        assert "/etc/passwd" in error

        # Chained with ;
        valid, error = check_bash_directory_boundary(
            "mkdir newdir; mv file.txt /tmp/", self.cwd, self.approved
        )
        assert not valid
        assert "/tmp/" in error

    def test_chained_commands_inside_pass(self) -> None:
        """Chain of valid commands should pass."""
        valid, error = check_bash_directory_boundary(
            "cd subdir && touch file.txt && ls -la", self.cwd, self.approved
        )
        assert valid
        assert error is None

    def test_chained_cd_outside_blocked(self) -> None:
        """cd /tmp && something should be blocked."""
        valid, error = check_bash_directory_boundary(
            "cd /tmp && ls", self.cwd, self.approved
        )
        assert not valid
        assert "/tmp" in error


class TestMultilineAndRedirection:
    """Regression tests: multi-line scripts, redirects and heredocs must not
    be mis-tokenized into bogus filesystem-boundary violations (#issue
    "cd targets '/dev/null'" and friends seen in production logs)."""

    def setup_method(self) -> None:
        self.approved = Path("/srv/claude-bot/workspace")
        self.cwd = self.approved / "claude-tg-bot"

    def test_newline_is_a_command_separator(self) -> None:
        """A second line is a separate command, not more args to the first."""
        cmd = "cd subdir\ntouch file.txt"
        valid, error = check_bash_directory_boundary(cmd, self.cwd, self.approved)
        assert valid
        assert error is None

    def test_multiline_with_background_and_redirects_inside_approved(self) -> None:
        """Real production command: cd, background job, stdout/stderr/stdin
        redirects, disown, sleep+cat — all inside the approved dir."""
        cmd = (
            "cd /srv/claude-bot/workspace/claude-tg-bot\n"
            "setsid nohup bash tools/update.sh > .engine/update.log 2>&1 < /dev/null &\n"
            "disown 2>/dev/null || true\n"
            "sleep 5; cat .engine/update.log"
        )
        valid, error = check_bash_directory_boundary(cmd, self.cwd, self.approved)
        assert valid
        assert error is None

    def test_stdin_redirect_target_outside_approved_is_not_flagged(self) -> None:
        """``< /dev/null`` must not be checked as a path argument of the
        preceding command — it targets stdin, not the filesystem."""
        cmd = "cd subdir < /dev/null"
        valid, error = check_bash_directory_boundary(cmd, self.cwd, self.approved)
        assert valid
        assert error is None

    def test_stdout_redirect_target_outside_approved_is_not_flagged(self) -> None:
        cmd = "touch file.txt > /dev/null 2>&1"
        valid, error = check_bash_directory_boundary(cmd, self.cwd, self.approved)
        assert valid
        assert error is None

    def test_real_cd_after_redirect_outside_is_still_blocked(self) -> None:
        """The redirect exemption must not swallow a genuine later violation."""
        cmd = "touch file.txt > /dev/null && cd /tmp"
        valid, error = check_bash_directory_boundary(cmd, self.cwd, self.approved)
        assert not valid
        assert "/tmp" in error

    def test_heredoc_body_is_not_parsed_as_shell_tokens(self) -> None:
        """Paths written inside a heredoc (Python source piped to stdin) are
        data for the child process, not filesystem arguments of any bash
        command in the surrounding script."""
        cmd = (
            "cd /srv/claude-bot/workspace/claude-tg-bot\n"
            "python3 - <<'PY'\n"
            "import pathlib\n"
            "p = pathlib.Path('/etc/passwd')\n"
            "print(p)\n"
            "PY\n"
            "echo done"
        )
        valid, error = check_bash_directory_boundary(cmd, self.cwd, self.approved)
        assert valid
        assert error is None

    def test_heredoc_does_not_hide_a_real_violation_after_it(self) -> None:
        cmd = (
            "python3 - <<'PY'\n"
            "print('hello')\n"
            "PY\n"
            "mkdir /etc/evil"
        )
        valid, error = check_bash_directory_boundary(cmd, self.cwd, self.approved)
        assert not valid
        assert "/etc/evil" in error

    def test_heredoc_does_not_hide_a_real_violation_before_it(self) -> None:
        cmd = (
            "mkdir /etc/evil\n"
            "python3 - <<'PY'\n"
            "print('hello')\n"
            "PY"
        )
        valid, error = check_bash_directory_boundary(cmd, self.cwd, self.approved)
        assert not valid
        assert "/etc/evil" in error

    def test_multiple_mkdir_new_project_folders_pass(self) -> None:
        """The original complaint: creating a brand-new project folder."""
        cmd = (
            "mkdir -p /srv/claude-bot/workspace/new-project && "
            "cd /srv/claude-bot/workspace/new-project && git init"
        )
        valid, error = check_bash_directory_boundary(cmd, self.cwd, self.approved)
        assert valid
        assert error is None


class TestIsClaudeInternalPath:
    """Test the _is_claude_internal_path helper function."""

    def test_plan_file_is_internal(self, tmp_path: Path) -> None:
        """~/.claude/plans/some-plan.md should be recognised as internal."""
        with patch("src.claude.monitor.Path.home", return_value=tmp_path):
            (tmp_path / ".claude" / "plans").mkdir(parents=True)
            plan_file = tmp_path / ".claude" / "plans" / "my-plan.md"
            plan_file.touch()
            assert _is_claude_internal_path(str(plan_file)) is True

    def test_todo_file_is_internal(self, tmp_path: Path) -> None:
        """~/.claude/todos/todo.md should be recognised as internal."""
        with patch("src.claude.monitor.Path.home", return_value=tmp_path):
            (tmp_path / ".claude" / "todos").mkdir(parents=True)
            todo_file = tmp_path / ".claude" / "todos" / "todo.md"
            todo_file.touch()
            assert _is_claude_internal_path(str(todo_file)) is True

    def test_settings_json_is_internal(self, tmp_path: Path) -> None:
        """~/.claude/settings.json should be recognised as internal."""
        with patch("src.claude.monitor.Path.home", return_value=tmp_path):
            (tmp_path / ".claude").mkdir(parents=True)
            settings_file = tmp_path / ".claude" / "settings.json"
            settings_file.touch()
            assert _is_claude_internal_path(str(settings_file)) is True

    def test_arbitrary_file_under_claude_dir_rejected(self, tmp_path: Path) -> None:
        """Files directly under ~/.claude/ (not in known subdirs) are rejected."""
        with patch("src.claude.monitor.Path.home", return_value=tmp_path):
            (tmp_path / ".claude").mkdir(parents=True)
            secret = tmp_path / ".claude" / "credentials.json"
            secret.touch()
            assert _is_claude_internal_path(str(secret)) is False

    def test_path_outside_claude_dir_rejected(self, tmp_path: Path) -> None:
        """Paths outside ~/.claude/ entirely are rejected."""
        with patch("src.claude.monitor.Path.home", return_value=tmp_path):
            assert _is_claude_internal_path("/etc/passwd") is False
            assert _is_claude_internal_path("/tmp/evil.txt") is False

    def test_empty_path_rejected(self, tmp_path: Path) -> None:
        """Empty paths are rejected."""
        assert _is_claude_internal_path("") is False

    def test_unknown_subdir_rejected(self, tmp_path: Path) -> None:
        """Unknown subdirectories under ~/.claude/ are rejected."""
        with patch("src.claude.monitor.Path.home", return_value=tmp_path):
            (tmp_path / ".claude" / "secrets").mkdir(parents=True)
            bad_file = tmp_path / ".claude" / "secrets" / "key.pem"
            bad_file.touch()
            assert _is_claude_internal_path(str(bad_file)) is False
