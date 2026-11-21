# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

import asyncio
import queue
import threading
import timeit
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import discord

from emet.llms.discord_bot import EmetDiscordBot
from emet.utils.discord_bot import ChannelList, DiscordBot, Task


def scoped_bot():
    channel = Mock(spec=discord.TextChannel)
    channel.id = 123
    channel.name = "simulator"
    other = Mock(spec=discord.TextChannel)
    other.id = 456
    other.name = channel.name
    other.send = AsyncMock()

    events = {}

    def event(callback):
        events[callback.__name__] = callback

    bot = EmetDiscordBot.__new__(EmetDiscordBot)
    bot.client = SimpleNamespace(
        user=SimpleNamespace(id=30, name="emet", global_name="Emet"),
        get_channel=Mock(return_value=channel),
        guilds=[SimpleNamespace(id=79, name="Other", text_channels=[other])],
        event=event,
        process_commands=AsyncMock(),
    )
    bot.home_channel_id = channel.id
    bot.restrict_to_home_channels = True
    bot.allowed_channels = ChannelList()
    bot.task_queue = queue.Queue()
    bot._ready_event = threading.Event()
    bot._plan_lock = threading.Lock()
    bot._plan_thread = None
    bot.start_plan_thread = Mock()
    bot.process_queue = Mock()
    bot._setup_hooks(bot.client)
    return bot, channel, other, events


def test_exact_discord_channel_has_no_fallback_after_reconnect():
    bot, channel, other, _ = scoped_bot()
    bot.on_ready()
    bot.on_ready()
    bot.client.get_channel.assert_called_with(channel.id)
    assert bot.is_ready
    assert list(bot.allowed_channels) == [channel]

    bot.client.get_channel.return_value = None
    bot.on_ready()
    assert not bot.is_ready
    assert not list(bot.allowed_channels)
    assert other not in bot.allowed_channels


def test_exact_discord_channel_blocks_outside_mentions_and_tasks():
    bot, channel, other, events = scoped_bot()
    bot.on_ready()
    bot.on_message = Mock(return_value=None)
    message = SimpleNamespace(channel=other, content="<@30> describe the scene")
    asyncio.run(events["on_message"](message))
    bot.client.process_commands.assert_not_awaited()
    bot.on_message.assert_not_called()
    assert other not in bot.allowed_channels
    asyncio.run(bot.handle_task(Task("status", other, explicit=True)))
    other.send.assert_not_awaited()

    message.channel = channel
    message.content = "Describe the scene"
    asyncio.run(events["on_message"](message))
    bot.client.process_commands.assert_awaited_once_with(message)
    bot.on_message.assert_called_once_with(message)


def test_exact_operator_stop_bypasses_the_conversation_queue():
    bot, channel, other, _ = scoped_bot()
    bot.on_ready()
    bot.navigation_stop_request = Mock()
    bot.push_task = Mock()
    bot._print_discord_inbound = Mock()
    author = SimpleNamespace(id=42, display_name="Operator", bot=False)
    bot.on_message(SimpleNamespace(channel=other, author=author, content="stop"))
    bot.navigation_stop_request.assert_not_called()

    bot.on_message(SimpleNamespace(channel=channel, author=author, content=" STOP "))
    bot.navigation_stop_request.assert_called_once_with()
    assert bot.push_task.call_args.kwargs["explicit"] is True

    bot.push_task.reset_mock()
    bot.on_message(SimpleNamespace(channel=channel, author=author, content="When did you stop?"))
    bot.navigation_stop_request.assert_called_once_with()
    assert bot.push_task.call_args.kwargs["message"] == "Operator: When did you stop?"


def test_outbound_stop_ack_does_not_merge_the_next_operator_turn():
    bot, channel, _, _ = scoped_bot()
    bot._started = True
    bot.timeout = 180
    bot.handle_task = AsyncMock()
    stamp = timeit.default_timer()
    ack = Task("Stop requested.", channel, explicit=True, t=stamp)
    followup = Task("Operator: Tell me what happened.", channel, t=stamp)
    bot.task_queue.put(ack)
    bot.task_queue.put(followup)

    asyncio.run(DiscordBot.process_queue.coro(bot))

    bot.handle_task.assert_awaited_once_with(ack)
    assert ack.message == "Stop requested."
    assert bot.task_queue.get_nowait() is followup
