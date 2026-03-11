# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import datetime
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import discord
from termcolor import colored

# import emet.utils.logger as logger
from emet.controller.controller_instance_memory import RobotAgent
from emet.controller.task.dynamem import DynamemTaskExecutor, EQAExecuter
from emet.controller.task.pickup import PickupExecutor
from emet.llms import PickupPromptBuilder, get_llm_client
from emet.utils.discord_bot import DiscordBot, Task
from emet.utils.logger import Logger

logger = Logger(__name__)


class EmetDiscordBot(DiscordBot):
    """Discord bot that connects to a robot agent (pickup, dynamem, eqa, graph_eqa)."""

    def __init__(
        self,
        agent: RobotAgent,
        token: Optional[str] = None,
        llm: str = "qwen25",
        task: str = "pickup",
        skip_confirmations: bool = False,
        output_path: str = ".",
        device_id: int = 0,
        visual_servo: bool = True,
        server_ip: str = "127.0.0.1",
        use_voice: bool = False,
        debug_llm: bool = False,
        manipulation_only: bool = False,
        kwargs: Dict[str, Any] = None,
        home_channel: str = "talk-to-stretch",
        executor: Any = None,
    ) -> None:
        """
        Create a new Discord bot that can interact with the robot.

        Args:
            agent: The robot agent that will be used to control the robot.
            executor: Optional existing executor (e.g. DynamemTaskExecutor). If provided for task dynamem, it is reused and its discord_bot is set to self.
            token: The token for the discord bot. Will be read from env if not available.
            llm: The language model to use.
            task: The task to perform. Currently only "pickup" is supported.
            skip_confirmations: Set to skip confirmations from the user.
            output_path: The path to save output files.
            device_id: The ID of the device to use for perception.
            visual_servo: Set to use visual servoing.
            kwargs: Additional parameters.

        Returns:
            None
        """
        super().__init__(token)
        robot = agent.robot

        # Create the prompt we will use to control the robot
        prompt = PickupPromptBuilder()

        # Save the parameters
        self.task = task
        self.agent = agent
        self.robot = self.agent.robot
        self.parameters = agent.parameters
        self.visual_servo = visual_servo
        self.device_id = device_id
        self.output_path = output_path
        self.server_ip = server_ip
        self.skip_confirmations = skip_confirmations
        self.kwargs = kwargs
        self.prompt = prompt

        self.home_channel = os.environ.get("EMET_DISCORD_CHANNEL", home_channel)
        self.sent_prompt = False

        if kwargs is None:
            # Default parameters
            kwargs = {
                "match_method": "feature",
                "mllm_for_visual_grounding": False,
            }

        # Executor handles outputs from the LLM client and converts them into executable actions
        # TODO: we should have an Executor abstract class here!
        if self.task == "pickup":
            self.executor = PickupExecutor(
                robot,
                agent,
                available_actions=prompt.get_available_actions(),
                dry_run=False,
                discord_bot=self,
            )  # type: ignore
        elif self.task == "dynamem":
            if executor is not None:
                self.executor = executor
                self.executor.discord_bot = self  # type: ignore
                self.executor.agent.discord_bot = self  # type: ignore
            else:
                self.executor = DynamemTaskExecutor(
                    robot,
                    agent.parameters,
                    visual_servo=visual_servo,
                    match_method=kwargs["match_method"],
                    device_id=device_id,
                    output_path=output_path,
                    server_ip=server_ip,
                    skip_confirmations=skip_confirmations,
                    mllm=kwargs["mllm_for_visual_grounding"],
                    manipulation_only=manipulation_only,
                    discord_bot=self,
                )  # type: ignore
                self.executor.agent.discord_bot = self  # type: ignore
        elif self.task == "eqa":
            self.executor = EQAExecuter(agent, discord_bot=self)  # type: ignore
        elif self.task == "graph_eqa":
            self.executor = EQAExecuter(agent, discord_bot=self)  # type: ignore
        else:
            raise NotImplementedError(f"Task {task} is not implemented.")

        # Get the LLM client
        # When llm is None (e.g. agent loop manages its own LLM), skip model loading.
        # When task is eqa/graph_eqa, all llms are created within self.agent.
        if llm is not None and self.task not in ("eqa", "graph_eqa"):
            self.llm_client = get_llm_client(llm, prompt=prompt)
        else:
            self.llm_client = None

        self._llm_lock = threading.Lock()
        self._ready_event = threading.Event()

    @property
    def is_ready(self) -> bool:
        return self._ready_event.is_set()

    def wait_until_ready(self, timeout: float = 30.0) -> bool:
        """Block until Discord connection is ready, or timeout. Returns True if ready."""
        return self._ready_event.wait(timeout=timeout)

    def on_ready(self):
        """Event listener called when the bot has switched from offline to online."""
        logger.debug(f"{self.client.user} has connected to Discord!")
        guild_count = 0

        logger.debug("Bot User name:", self.client.user.name)
        logger.debug("Bot Global name:", self.client.user.global_name)
        logger.debug("Bot User ID:", self.client.user.id)
        self._user_name = self.client.user.name
        self._user_id = self.client.user.id

        for guild in self.client.guilds:
            logger.debug(f"Joining Server {guild.id} (name: {guild.name})")
            guild_count = guild_count + 1

            found_home = False
            first_text_channel = None
            for channel in guild.text_channels:
                if first_text_channel is None:
                    first_text_channel = channel
                if channel.name == self.home_channel:
                    logger.info(f"Found home channel: #{channel.name}")
                    self.allowed_channels.add_home(channel)
                    found_home = True
                    break

            if not found_home and first_text_channel is not None:
                logger.warning(
                    f"Home channel '{self.home_channel}' not found in {guild.name}. "
                    f"Using #{first_text_channel.name} instead. "
                    f"Set EMET_DISCORD_CHANNEL or create a #{self.home_channel} channel."
                )
                self.allowed_channels.add_home(first_text_channel)

        self.next_plan = None
        self._plan_lock = threading.Lock()
        self._plan_thread = None

        if len(self.allowed_channels) == 0:
            logger.error("No Discord channels found! Messages will not be sent.")
            logger.error("Create a #talk-to-stretch channel or set EMET_DISCORD_CHANNEL=<channel-name>.")
        else:
            logger.info("Discord channels:", len(self.allowed_channels), "in", guild_count, "guild(s).")

        self.process_queue.start()

        # Start the plan thread
        self.start_plan_thread()

        # Signal that the bot is ready
        self._ready_event.set()

    def push_task_to_all_channels(
        self, message: Optional[str] = None, content: Optional[str] = None
    ):
        """Push a task to all channels. Message will be "as-is" with no processing.

        Args:
            message: The message to send to the channel.
            content: The content (image string) to send to the channel.
        """

        for channel in self.allowed_channels:
            self.push_task(channel, message=message, content=content, explicit=True)

    def on_message(self, message: discord.Message, verbose: bool = False):
        """Event listener for whenever a new message is sent to a channel that this bot is in."""
        if verbose:
            # Printing some information to learn about what this actually does
            print(message)
            print("Content =", message.content)
            print("Content type =", type(message.content))
            print("Author name:", message.author.name)
            print("Author global name:", message.author.global_name)

        # This is your actual username
        # sender_name = message.author.name
        sender_name = message.author.display_name
        # Only necessary once we want multi-server Friends
        # global_name = message.author.global_name

        # Skip anything that's from this bot
        if message.author.id == self._user_id:
            return None

        # TODO: make this a command line parameter for which channel(s) he should be in
        channel_name = message.channel.name
        print("Channel name:", channel_name)
        channel_id = message.channel.id
        print("Channel ID:", channel_id)
        # datetime = message.created_at

        timestamp = message.created_at.timestamp()
        print("Timestamp:", timestamp)

        print(self.allowed_channels)
        if not message.channel in self.allowed_channels:
            print(" -> Not in allowed channels. Skipping.")
            return None

        # Construct the text to prompt the AI
        # TODO: Do we ever want to add the channel name? If so we can revert this change
        # text = f"{sender_name} on #{channel_name}: " + message.content
        text = f"{sender_name}: " + message.content
        self.push_task(channel=message.channel, message=text)

        print("Current task queue: ", self.task_queue.qsize())
        # print(" -> Response:", response)
        return None

    async def handle_task(self, task: Task):
        """Handle a task by sending the message to the channel. This will make the necessary calls in its thread to the different child functions that send messages, for example."""
        print()
        print("-" * 40)
        print("Handling task from channel:", task.channel.name)
        print("Handling task: message =", task.message)

        text = task.message
        try:
            if task.explicit:
                print("This task was explicitly triggered.")
                if task.message:
                    await task.channel.send(task.message)
                if task.content is not None:
                    import io
                    import numpy as np
                    from PIL import Image as PILImage
                    buf = io.BytesIO()
                    img = task.content
                    if isinstance(img, np.ndarray):
                        img = PILImage.fromarray(img.astype(np.uint8))
                    img.save(buf, format="PNG")
                    buf.seek(0)
                    filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".png"
                    await task.channel.send(file=discord.File(buf, filename=filename))
                return
        except Exception as e:
            print(colored("Error in handling task: " + str(e), "red"))

        with self._llm_lock:
            if self.llm_client is None:
                # Agent mode: no local LLM; just log the incoming message
                print(colored(f"[Discord] {text}", "cyan"))
                return
            if self.task != "eqa":
                response = self.llm_client(text, verbose=True)
                print("Response:", response)
                parsed_response = self.prompt.parse_response(response)
                print("Parsed response:", parsed_response)
                self.add_robot_plan(parsed_response, channel=task.channel)
            else:
                self.add_robot_plan(text, channel=task.channel)

    def add_robot_plan(self, response: List[Tuple[str, str]], channel: discord.TextChannel):
        """Add a task to the task queue."""
        with self._plan_lock:
            self.next_plan = response, channel

    def plan_thread(self):
        """Loop. Check to see if next plan received. If so, execute it. Else, sleep. After execution, set next_plan to None."""

        while self.robot.is_running:
            if self.next_plan is not None:
                with self._plan_lock:
                    response, channel = self.next_plan
                    self.next_plan = None
                # For eqa/graph_eqa, response is the raw message string (optionally "User: question"); for pickup/dynamem it's List[Tuple[str, str]]
                if self.task in ("eqa", "graph_eqa") and isinstance(response, str):
                    question = response.split(":", 1)[-1].strip() if ":" in response else response
                    self.executor(question, channel=channel)
                else:
                    self.executor(response, channel=channel)
            else:
                time.sleep(0.01)

    def start_plan_thread(self):
        """Start the plan thread."""
        self._plan_thread = threading.Thread(target=self.plan_thread)
        self._plan_thread.start()

    def __del__(self):
        """Destructor. Stop the plan thread."""
        if self._plan_thread is not None:
            self._plan_thread.join()
