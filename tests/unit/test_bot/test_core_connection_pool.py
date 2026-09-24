"""Tests for bot core HTTP connection pool wiring."""

from unittest.mock import AsyncMock, MagicMock

import pytest

import src.bot.core as core_module
from src.bot.core import ClaudeCodeBot
from src.config import create_test_config


@pytest.fixture
def bot_with_builder(monkeypatch):
    """Create a bot with mocked Application builder plumbing."""
    settings = create_test_config()
    deps = {
        "storage": MagicMock(),
        "security": MagicMock(),
    }
    bot = ClaudeCodeBot(settings, deps)

    builder = MagicMock()

    app = MagicMock()
    app.bot = MagicMock()
    app.bot.set_my_commands = AsyncMock()
    app.initialize = AsyncMock()
    builder.build.return_value = app

    monkeypatch.setattr(
        core_module.Application,
        "builder",
        MagicMock(return_value=builder),
    )
    monkeypatch.setattr(
        core_module,
        "FeatureRegistry",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(bot, "_set_bot_commands", AsyncMock())
    monkeypatch.setattr(bot, "_register_handlers", MagicMock())
    monkeypatch.setattr(bot, "_add_middleware", MagicMock())

    return bot, builder


@pytest.mark.asyncio
async def test_initialize_gives_get_updates_client_a_pool(bot_with_builder):
    """Long polling must not run on the default single-connection pool."""
    bot, builder = bot_with_builder

    await bot.initialize()

    builder.get_updates_request.assert_called_once()
    request = builder.get_updates_request.call_args.args[0]
    assert request._client_kwargs["limits"].max_connections > 1
