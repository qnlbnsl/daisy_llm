import datetime
import json
import logging
import re
import time

from enum import Enum
from ruamel.yaml import YAML
from typing import Any, List, Optional, Tuple, TypedDict
from typing_extensions import Self


from .chat import Chat
from .text import print_text
from .connection_pool import ConnectionPool


# Initialize YAML parser
yaml = YAML()


class Role(Enum):
    user = "user"
    system = "system"


class Message(TypedDict):
    timestamp: Optional[str]
    role: Role
    content: str


class StartPrompt(TypedDict):
    timestamp: Optional[str]
    role: Role
    content: str


class ContextHandlers:
    description = (
        "A class for handling and managing messages in the chatGPT context object"
    )

    def __init__(self: Self, db_path: str) -> None:
        self.chat: Chat = Chat()

        # Get and set conversation_id from configs.yaml
        self.conversation_id: str | None = None
        with open("configs.yaml", "r") as f:
            configs = yaml.load(f)
        if (
            "conversation_id" in configs
        ):  # TODO: Ask @myrakrusemark about this. What exactly is this?
            self.conversation_id = configs.get("conversation_id")
            logging.info(
                "Using conversation id from configs: " + str(self.conversation_id)
            )

        self.db_path = db_path
        self.messages: List[Message] = []
        self.start_prompts: List[StartPrompt] = []
        self.connection_pool = ConnectionPool(db_path)

    def load_context(self: Self) -> None:
        self.messages = []
        self.create_conversations_table_if_not_exists()
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            # If conversation_id is not set, create a new conversation ID
            if not self.conversation_id:
                self.conversation_id = str(int(time.time()))

                print_text("Creating new conversation: ", "yellow")
                print_text(str(self.conversation_id), None, "\n")

            logging.info("Conversation id: " + str(self.conversation_id))

            # Get the messages from the conversation ID
            cursor.execute(
                """
				SELECT * FROM messages WHERE conversation_id = ?
			""",
                (self.conversation_id,),
            )
            rows = cursor.fetchall()
            if rows:
                for row in rows:
                    message: Message = {
                        "timestamp": row[1],
                        "role": row[2],
                        "content": row[3],
                    }
                    self.messages.append(message)
                print_text(
                    "Loaded "
                    + str(len(rows))
                    + " messages from conversation id: "
                    + str(self.conversation_id),
                    "yellow",
                    "\n",
                )

    def create_conversations_table_if_not_exists(self: Self) -> None:
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
				CREATE TABLE IF NOT EXISTS messages (
					conversation_id TEXT NOT NULL,
					timestamp TEXT NOT NULL,
					role TEXT NOT NULL,
					message TEXT NOT NULL
				);
			"""
            )
            cursor.execute(
                """
				CREATE TABLE IF NOT EXISTS conversations (
					id TEXT PRIMARY KEY,
					name TEXT NOT NULL,
					summary TEXT NOT NULL
				);
			"""
            )

    def save_context(self: Self) -> None:
        logging.info("Saving context: " + str(self.conversation_id))
        with self.connection_pool.get_connection() as conn:
            conn.execute(
                """
				PRAGMA foreign_keys=OFF;
			"""
            )
            conn.execute(
                """
				BEGIN TRANSACTION;
			"""
            )

            # Insert conversation information if it doesn't already exist
            rows = conn.execute(
                """
				SELECT COUNT(*) FROM conversations WHERE id = ?;
			""",
                (self.conversation_id,),
            ).fetchone()[0]
            if rows == 0:
                conn.execute(
                    """
					INSERT INTO conversations (id, name, summary)
					VALUES (?, ?, ?);
				""",
                    (self.conversation_id, "No name", "No summary"),
                )

            # Save messages
            conn.execute(
                """
				DELETE FROM messages WHERE conversation_id = ?;
			""",
                (self.conversation_id,),
            )
            for message in self.messages:
                conn.execute(
                    """
					INSERT INTO messages (conversation_id, timestamp, role, message)
					VALUES (?, ?, ?, ?);
				""",
                    (
                        self.conversation_id,
                        message.get("timestamp"),
                        message.get("role"),
                        message.get("content"),
                    ),
                )
            conn.execute(
                """
				COMMIT;
			"""
            )
            row_count = conn.execute(
                """
				SELECT COUNT(*) FROM messages WHERE conversation_id = ?;
			""",
                (self.conversation_id,),
            ).fetchone()[0]
            logging.info(
                f"Inserted {row_count} rows for conversation {self.conversation_id}."
            )

    def get_context(
        self: Self, include_timestamp: bool = True, include_system: bool = True
    ) -> List[Message]:
        context: List[Message] = []
        # Append start prompts to messages
        for start_prompt in self.start_prompts:
            if include_system or start_prompt["role"] != Role.system:
                new_prompt: Message = {
                    "role": start_prompt["role"],
                    "content": start_prompt["content"],
                    "timestamp": None,
                }
                if include_timestamp:
                    new_prompt["timestamp"] = start_prompt["timestamp"]
                context.append(new_prompt)  # append new_message, not start_prompt

        # Do the same for the regular messages
        for message in self.messages:
            if include_system or message["role"] != Role.system:
                new_message: Message = {
                    "role": message["role"],
                    "content": message["content"],
                    "timestamp": None,
                }
                if include_timestamp:
                    new_message["timestamp"] = message["timestamp"]
                context.append(new_message)  # append new_message, not message

        return context

    def get_context_without_timestamp(self: Self) -> List[Message]:
        messages_without_timestamp: List[Message] = []
        for message in self.get_context():
            message_without_timestamp = message.copy()
            message_without_timestamp["timestamp"] = None
            messages_without_timestamp.append(message_without_timestamp)
        return messages_without_timestamp

    def get_conversation_name_summary(
        self: Self, limit: Optional[int] = None
    ) -> List[Tuple[Any, Any, Any]] | None:
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()
            query = """SELECT id, name, summary FROM conversations ORDER BY id DESC"""
            if limit:
                query += f" LIMIT {limit}"
            cursor.execute(query)
            rows = cursor.fetchall()
            if rows:
                return [(id, name, summary) for id, name, summary in rows]
            else:
                return None

    def single_message_context(
        self: Self, role: Role, user_message: str, incl_timestamp: bool = True
    ) -> Message:
        if incl_timestamp:
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            return Message(timestamp=timestamp, role=role, content=str(user_message))
        else:
            return Message(role=role, content=user_message, timestamp=None)

    def add_start_prompt(
        self: Self, role: Role = Role.system, user_message: str = ""
    ) -> None:
        start_prompt = self.single_message_context(role, user_message)
        self.start_prompts.append(start_prompt)

    def add_message_object(self: Self, role: Role, message: str) -> None:
        logging.debug("Adding " + role.value + " message to context")
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        new_message: Message = {
            "role": role,
            "timestamp": timestamp,
            "content": str(message),
        }
        self.messages.append(new_message)
        self.save_context()
        logging.debug(self.messages)

    def add_message_object_at_start(self: Self, role: Role, message: str) -> None:
        logging.debug("Appending " + role.value + " message at start of context")
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        new_message: Message = {
            "role": role,
            "timestamp": timestamp,
            "content": str(message),
        }
        self.messages.insert(0, new_message)
        self.save_context()
        logging.debug(self.messages)

    def remove_last_message_object(self: Self) -> None:
        if self.messages:
            self.messages.pop()
            self.save_context()

    def get_last_message_object(
        self: Self, user_type: Optional[Role] = None
    ) -> Message | bool:
        if user_type:
            for message in reversed(self.messages):
                if message["role"] == user_type:
                    return message
        else:
            if self.messages:
                return self.messages[-1]
        return False

    def replace_last_message_object(
        self: Self, message: str, user_type: Optional[Role] = None
    ) -> None:
        if user_type:
            for i in reversed(range(len(self.messages))):
                if self.messages[i]["role"] == user_type:
                    self.messages[i]["content"] = message
                    self.save_context()
                    return
        elif message and self.messages:
            self.messages[-1]["content"] = message
            self.save_context()

    def delete_message_at_index(self: Self, index: int) -> bool:
        try:
            if index < len(self.messages) and index >= 0:
                self.messages.pop(index)
                self.save_context()
                return True
        except ValueError:
            pass
        return False

    def update_message_at_index(self: Self, message: str, index: int) -> None:
        try:
            if index < len(self.messages) and index >= 0:
                self.messages[index]["content"] = message
                now = datetime.datetime.now()
                self.messages[index]["timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S")
                self.save_context()
            else:
                raise ValueError("Index out of range")
        except ValueError:
            raise ValueError("Index must be an integer")

    def update_conversation_name_summary(
        self: Self, conversation_id: Optional[str] = None, update_all: bool = False
    ) -> None:
        self.chat = Chat()

        conversation_ids: List[str] = []

        if conversation_id:
            conversation_ids.append(conversation_id)
        elif update_all:
            conversation_ids = self.get_conversation_ids()
        else:
            conversation_ids.append(
                self.conversation_id
            )  # TODO: Ask @myrakrusemark about this
            # Get conversations with missing name or summary
            conversations = self.get_conversation_name_summary(limit=None)
            if conversations is None:
                raise Exception("No conversations found")
            for conv_id, name, summary in conversations:
                if name == "No name" or summary == "No summary":
                    if conv_id not in conversation_ids:
                        conversation_ids.append(conv_id)

        for conv_id in conversation_ids:
            messages = self.get_conversation_context_by_id(
                conv_id, include_timestamp=False
            )
            # If there are no messages in the context, delete it

            if not messages:
                self.delete_conversation_by_id(
                    conv_id
                )  # TODO: Ask @myrakrusemark about this. Needs implementation.

            # Get the name of the current conversation from the LLM
            time.sleep(1)
            logging.info("Updating conversation name and summary for: " + conv_id)

            while True:
                prompt = """
				Please respond with a name, and summary for this conversation.
				1. The name should be a single word or short phrase, no more than 5 words."
				2. The summary should be a fairly verbose summary of the conversation, as short as possible while still containing all of the important topics, names, places, and sentiment of conversation.
				3. The output must follow the following JSON format: {"name": name, "summary": summary}
				4. If the conversation is empty, please respond with "Empty"
				"""
                if messages:
                    messages.append(
                        self.single_message_context(Role.system, prompt, False)
                    )

                    print_text("Conversation info (" + str(conv_id) + "): ", "yellow")
                    response = self.chat.request(
                        messages=messages, silent=False, response_label=False
                    )
                else:
                    response = '{"name": "Empty Conversation", "summary": "None"}'

                # Extract the JSON response from the string
                response_match = re.search(r"{.*}", response)
                if response_match:
                    response_json = response_match.group(0)
                    break
                elif response == "Empty":
                    response_json = '{"name": "Empty Conversation", "summary": "None"}'
                    break
                else:
                    logging.error(
                        "Invalid response format while setting conversation name and summary. Trying again..."
                    )

            # Convert the JSON response to an object
            try:
                response_obj = json.loads(response_json)
            except Exception as e:
                logging.error(
                    "Invalid JSON response while setting conversation name and summary: "
                    + str(e)
                )
                return

            # Update the name and summary of the current conversation in the database
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """UPDATE conversations SET name = ?, summary = ? WHERE id = ?""",
                    (response_obj["name"], response_obj["summary"], conv_id),
                )
                conn.commit()

            logging.info(
                "Name and summary updated for conversation "
                + conv_id
                + ": "
                + response_obj["name"]
            )

    def get_conversation_ids(self: Self) -> List[Any]:
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""SELECT id FROM conversations;""")
            rows = cursor.fetchall()
            return [row[0] for row in rows]

    def new_conversation(self: Self) -> None:
        # Generate a new conversation ID
        conversation_id = str(int(time.time()))
        logging.info("Creating a new conversation: " + conversation_id)

        # Set the new conversation ID in configs.yaml
        with open("configs.yaml", "r") as f:
            configs = yaml.load(f)
        configs["conversation_id"] = conversation_id
        with open("configs.yaml", "w") as f:
            yaml.dump(configs, f)

        # Update the conversation ID and load the context
        self.conversation_id = conversation_id
        self.load_context()

    def get_conversation_name_by_id(self: Self, conversation_id: str) -> Any | None:
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT name FROM conversations WHERE id = ?""", (conversation_id,)
            )
            row = cursor.fetchone()
            if row:
                return row[0]
            else:
                return None

    def get_conversation_context_by_id(
        self: Self,
        conversation_id: str,
        include_timestamp: bool = True,
        include_system: bool = False,
    ) -> List[Message] | None:
        # Check if the conversation ID exists in the database
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
				SELECT id FROM conversations WHERE id = ?;
			""",
                (conversation_id,),
            )
            row = cursor.fetchone()

        if row:
            # Get the messages from the specified conversation ID
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
					SELECT timestamp, role, message FROM messages WHERE conversation_id = ?;
				""",
                    (conversation_id,),
                )
                rows = cursor.fetchall()

            context: List[Message] = []
            if rows:
                for row in rows:
                    message: Message = {
                        "timestamp": row[0],
                        "role": row[1],
                        "content": row[2],
                    }
                    if not include_timestamp:
                        message["timestamp"] = None
                    if include_system or message["role"] != Role.system:
                        context.append(message)

            return context
        else:
            return None
