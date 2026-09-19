"""Что считается системным действием и требует подтверждения владельца.

Бот работает с правами root, но всё, что выходит за рабочую папку проектов
или трогает систему, должно спрашивать у владельца кнопкой в Telegram.
"""

from pathlib import Path

from src.claude.system_actions import (
    classify_bash_command,
    classify_file_write,
)

APPROVED = Path("/srv/claude-bot/workspace")
CWD = APPROVED / "myproject"


class TestOrdinaryWorkNeedsNoApproval:
    """Обычная работа в проекте идёт без вопросов — как и раньше."""

    def test_plain_commands_pass(self) -> None:
        for cmd in [
            "ls -la",
            "git status",
            "python3 -m pytest",
            "npm install",
            "mkdir new-folder",
            "rm old-file.txt",
            "cat README.md",
            "git commit -m 'fix' && git push",
        ]:
            assert classify_bash_command(cmd, CWD, APPROVED) is None, cmd

    def test_writing_inside_project_passes(self) -> None:
        assert classify_file_write(str(CWD / "src/app.py"), APPROVED) is None
        assert classify_file_write("relative/file.txt", APPROVED) is None


class TestSystemctlNeedsApproval:
    """Службы — это то, что держит сайт и оплату живыми."""

    def test_restart_needs_approval(self) -> None:
        reason = classify_bash_command("systemctl restart nginx", CWD, APPROVED)
        assert reason is not None
        assert "служб" in reason.lower()

    def test_enable_and_daemon_reload_need_approval(self) -> None:
        for cmd in [
            "systemctl enable myservice",
            "systemctl daemon-reload",
            "systemctl stop avtovydacha-bot",
            "service nginx reload",
        ]:
            assert classify_bash_command(cmd, CWD, APPROVED) is not None, cmd

    def test_status_is_read_only_and_passes(self) -> None:
        """Смотреть состояние служб безопасно — спрашивать незачем."""
        for cmd in [
            "systemctl status nginx",
            "systemctl is-active claude-bot",
            "systemctl list-units",
            "journalctl -u nginx -n 50",
        ]:
            assert classify_bash_command(cmd, CWD, APPROVED) is None, cmd


class TestWritesOutsideWorkspaceNeedApproval:
    """Запись вне рабочей папки — всегда вопрос."""

    def test_writing_to_srv_needs_approval(self) -> None:
        reason = classify_file_write("/srv/site/index.html", APPROVED)
        assert reason is not None
        assert "/srv/site/index.html" in reason

    def test_writing_to_etc_needs_approval(self) -> None:
        assert classify_file_write("/etc/nginx/nginx.conf", APPROVED) is not None

    def test_mkdir_outside_workspace_needs_approval(self) -> None:
        reason = classify_bash_command("mkdir -p /srv/newproject", CWD, APPROVED)
        assert reason is not None
        assert "/srv/newproject" in reason

    def test_cp_outside_workspace_needs_approval(self) -> None:
        assert (
            classify_bash_command(
                "cp config.json /etc/myapp/config.json", CWD, APPROVED
            )
            is not None
        )

    def test_reading_outside_workspace_passes(self) -> None:
        """Чтение чужих файлов не разрушительно — вопросами не мучаем."""
        for cmd in [
            "cat /etc/nginx/nginx.conf",
            "ls /srv",
            "tail /var/log/syslog",
        ]:
            assert classify_bash_command(cmd, CWD, APPROVED) is None, cmd


class TestDangerousSystemCommands:
    """Команды, которые меняют систему целиком."""

    def test_package_managers_need_approval(self) -> None:
        for cmd in [
            "apt install nginx",
            "apt-get remove python3",
            "apt update && apt upgrade -y",
        ]:
            assert classify_bash_command(cmd, CWD, APPROVED) is not None, cmd

    def test_user_and_permission_changes_need_approval(self) -> None:
        for cmd in [
            "useradd newuser",
            "usermod -aG sudo someone",
            "chown -R www-data /srv/site",
            "chmod 777 /etc/passwd",
            "passwd root",
        ]:
            assert classify_bash_command(cmd, CWD, APPROVED) is not None, cmd

    def test_firewall_and_network_need_approval(self) -> None:
        for cmd in ["ufw allow 8080", "iptables -F", "nginx -s reload"]:
            assert classify_bash_command(cmd, CWD, APPROVED) is not None, cmd

    def test_reboot_needs_approval(self) -> None:
        for cmd in ["reboot", "shutdown -h now", "init 6"]:
            assert classify_bash_command(cmd, CWD, APPROVED) is not None, cmd

    def test_apt_list_is_read_only(self) -> None:
        assert classify_bash_command("apt list --installed", CWD, APPROVED) is None


class TestPaymentAndSiteAreExtraSensitive:
    """Воронка с оплатой и сайт — особый случай, причина должна это называть."""

    def test_funnel_is_named_in_reason(self) -> None:
        reason = classify_bash_command("rm -rf /srv/funnel/app", CWD, APPROVED)
        assert reason is not None
        assert "оплат" in reason.lower()

    def test_site_is_named_in_reason(self) -> None:
        reason = classify_file_write("/srv/site/index.html", APPROVED)
        assert reason is not None
        assert "сайт" in reason.lower()

    def test_funnel_env_is_named(self) -> None:
        reason = classify_file_write("/srv/funnel/app/.env", APPROVED)
        assert reason is not None
        assert "оплат" in reason.lower()


class TestMultilineCommands:
    """Многострочные команды — опасное может быть на любой строке."""

    def test_danger_on_second_line_is_caught(self) -> None:
        cmd = "cd /srv/claude-bot/workspace/myproject\nsystemctl restart nginx"
        assert classify_bash_command(cmd, CWD, APPROVED) is not None

    def test_safe_multiline_passes(self) -> None:
        cmd = (
            "cd /srv/claude-bot/workspace/myproject\n"
            "python3 -m pytest -q 2>&1 | tail -5\n"
            "git add -A && git commit -m 'tests'"
        )
        assert classify_bash_command(cmd, CWD, APPROVED) is None

    def test_heredoc_body_is_not_scanned(self) -> None:
        """Слова внутри heredoc — это данные, а не команды."""
        cmd = (
            "cd /srv/claude-bot/workspace/myproject\n"
            "python3 - <<'PY'\n"
            "print('systemctl restart nginx')\n"
            "PY"
        )
        assert classify_bash_command(cmd, CWD, APPROVED) is None


class TestSudo:
    """sudo сам по себе — признак системного действия."""

    def test_sudo_needs_approval(self) -> None:
        reason = classify_bash_command("sudo mkdir /srv/newapp", CWD, APPROVED)
        assert reason is not None

    def test_sudo_with_safe_command_still_asks(self) -> None:
        assert classify_bash_command("sudo ls /srv", CWD, APPROVED) is not None
