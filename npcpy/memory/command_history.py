import os
import sqlite3
import json
from datetime import datetime
import uuid
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np


def flush_messages(n: int, messages: list) -> dict:
    if n <= 0:
        return {
            "messages": messages,
            "output": "Error: 'n' must be a positive integer.",
        }

    removed_count = min(n, len(messages))  # Calculate how many to remove
    del messages[-removed_count:]  # Remove the last n messages

    return {
        "messages": messages,
        "output": f"Flushed {removed_count} message(s). Context count is now {len(messages)} messages.",
    }


def get_db_connection():
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_messages_for_conversation(conversation_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
        SELECT role, content, timestamp
        FROM conversation_history
        WHERE conversation_id = ?
        ORDER BY timestamp ASC
    """
    cursor.execute(query, (conversation_id,))
    messages = cursor.fetchall()
    conn.close()

    return [
        {
            "role": message["role"],
            "content": message["content"],
            "timestamp": message["timestamp"],
        }
        for message in messages
    ]



def deep_to_dict(obj):
    """
    Recursively convert objects that have a 'to_dict' method to dictionaries,
    otherwise drop them from the output.
    """
    if isinstance(obj, dict):
        return {key: deep_to_dict(val) for key, val in obj.items()}

    if isinstance(obj, list):
        return [deep_to_dict(item) for item in obj]

    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict", None)):
        return deep_to_dict(obj.to_dict())

    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj

    return None  # Drop objects that don't have a known conversion


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return deep_to_dict(obj)
        except TypeError:
            return super().default(obj)


def show_history(command_history, args):
    if args:
        search_results = command_history.search(args[0])
        if search_results:
            return "\n".join(
                [f"{item[0]}. [{item[1]}] {item[2]}" for item in search_results]
            )
        else:
            return f"No commands found matching '{args[0]}'"
    else:
        all_history = command_history.get_all()
        return "\n".join([f"{item[0]}. [{item[1]}] {item[2]}" for item in all_history])


def query_history_for_llm(command_history, query):
    results = command_history.search(query)
    formatted_results = [
        f"Command: {r[2]}\nOutput: {r[4]}\nLocation: {r[5]}" for r in results
    ]
    return "\n\n".join(formatted_results)


class CommandHistory:
    def __init__(self, path="~/npcsh_history.db"):
        self.db_path = os.path.expanduser(path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key support
        self.cursor = self.conn.cursor()
        self.create_command_table()
        self.create_conversation_table()
        self.create_attachment_table()

    def create_command_table(self):
        self.cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS command_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            command TEXT,
            subcommands TEXT,
            output TEXT,
            location TEXT
        )
        """
        )
        self.conn.commit()

    def create_conversation_table(self):
        self.cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id TEXT UNIQUE,
            timestamp TEXT,
            role TEXT,
            content TEXT,
            conversation_id TEXT,
            directory_path TEXT,
            model TEXT,
            provider TEXT,
            npc TEXT
        )
        """
        )
        self.conn.commit()

    def create_attachment_table(self):
        self.cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS message_attachments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id TEXT,
            attachment_name TEXT,
            attachment_type TEXT,
            attachment_data BLOB,
            attachment_size INTEGER,
            upload_timestamp TEXT,
            FOREIGN KEY (message_id) REFERENCES conversation_history(message_id)
        )
        """
        )
        self.conn.commit()

    def add_command(self, command, subcommands, output, location):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Convert everything to strings to ensure it can be stored
        safe_subcommands = str(subcommands)
        safe_output = str(output)

        self.cursor.execute(
            """
            INSERT INTO command_history (timestamp, command, subcommands, output, location)
            VALUES (?, ?, ?, ?, ?)
        """,
            (timestamp, command, safe_subcommands, safe_output, location),
        )
        self.conn.commit()

    def generate_message_id(self) -> str:
        """Generate a unique ID for a message."""
        return str(uuid.uuid4())

    def add_conversation(
        self,
        role,
        content,
        conversation_id,
        directory_path,
        model=None,
        provider=None,
        npc=None,
        attachments=None,
        message_id=None,
    ):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if message_id is None:
            message_id = self.generate_message_id()

        if isinstance(content, dict):
            content = json.dumps(content, cls=CustomJSONEncoder)
        # Check if message_id already exists
        self.cursor.execute(
            "SELECT content FROM conversation_history WHERE message_id = ?",
            (message_id,),
        )
        existing_row = self.cursor.fetchone()

        if existing_row:
            # Append to existing content
            new_content = existing_row[0] + content
            self.cursor.execute(
                "UPDATE conversation_history SET content = ?, timestamp = ? WHERE message_id = ?",
                (new_content, timestamp, message_id),
            )
        else:
            # Insert new message
            self.cursor.execute(
                """
                INSERT INTO conversation_history
                (message_id, timestamp, role, content, conversation_id, directory_path, model, provider, npc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    message_id,
                    timestamp,
                    role,
                    content,
                    conversation_id,
                    directory_path,
                    model,
                    provider,
                    npc,
                ),
            )

        self.conn.commit()
        if attachments:
            for attachment in attachments:
                self.add_attachment(
                    message_id,
                    attachment["name"],
                    attachment["type"],
                    attachment["data"],
                    attachment_size=attachment.get("size"),
                )

        return message_id

    def add_attachment(
        self,
        message_id,
        attachment_name,
        attachment_type,
        attachment_data,
        attachment_size=None,
    ):
        """
        Add an attachment to a message.

        Args:
            message_id: The ID of the message to attach to
            attachment_name: The name of the attachment file
            attachment_type: The MIME type or file extension
            attachment_data: The binary data of the attachment
            attachment_size: The size in bytes (calculated if None)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Calculate size if not provided
        if attachment_size is None and attachment_data is not None:
            attachment_size = len(attachment_data)

        self.cursor.execute(
            """
            INSERT INTO message_attachments
            (message_id, attachment_name, attachment_type, attachment_data, attachment_size, upload_timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                attachment_name,
                attachment_type,
                attachment_data,
                attachment_size,
                timestamp,
            ),
        )
        self.conn.commit()

    def get_full_message_content(self, message_id):
        self.cursor.execute(
            "SELECT content FROM conversation_history WHERE message_id = ? ORDER BY timestamp ASC",
            (message_id,),
        )
        return "".join(row[0] for row in self.cursor.fetchall())  # Merge chunks

    def update_message_content(self, message_id, full_content):
        self.cursor.execute(
            "UPDATE conversation_history SET content = ? WHERE message_id = ?",
            (full_content, message_id),
        )
        self.conn.commit()

    def get_message_attachments(self, message_id) -> List[Dict]:
        """
        Retrieve all attachments for a specific message.

        Args:
            message_id: The ID of the message

        Returns:
            List of dictionaries containing attachment metadata (without binary data)
        """
        self.cursor.execute(
            """
            SELECT id, message_id, attachment_name, attachment_type, attachment_size, upload_timestamp
            FROM message_attachments
            WHERE message_id = ?
            """,
            (message_id,),
        )

        attachments = []
        for row in self.cursor.fetchall():
            attachments.append(
                {
                    "id": row[0],
                    "message_id": row[1],
                    "name": row[2],
                    "type": row[3],
                    "size": row[4],
                    "timestamp": row[5],
                }
            )
        return attachments

    def get_attachment_data(self, attachment_id) -> Tuple[bytes, str, str]:
        """
        Retrieve the binary data of a specific attachment.

        Args:
            attachment_id: The ID of the attachment

        Returns:
            Tuple of (binary_data, attachment_name, attachment_type)
        """
        self.cursor.execute(
            """
            SELECT attachment_data, attachment_name, attachment_type
            FROM message_attachments
            WHERE id = ?
            """,
            (attachment_id,),
        )

        result = self.cursor.fetchone()
        if result:
            return result[0], result[1], result[2]
        return None, None, None

    def delete_attachment(self, attachment_id) -> bool:
        """
        Delete a specific attachment.

        Args:
            attachment_id: The ID of the attachment to delete

        Returns:
            Boolean indicating success
        """
        try:
            self.cursor.execute(
                """
                DELETE FROM message_attachments
                WHERE id = ?
                """,
                (attachment_id,),
            )
            self.conn.commit()
            return self.cursor.rowcount > 0
        except sqlite3.Error:
            return False

    def get_last_command(self):
        self.cursor.execute(
            """
        SELECT * FROM command_history ORDER BY id DESC LIMIT 1
        """
        )
        return self.cursor.fetchone()

    def close(self):
        self.conn.close()

    def get_most_recent_conversation_id(self):
        self.cursor.execute(
            """
        SELECT conversation_id FROM conversation_history
        ORDER BY id DESC LIMIT 1
        """
        )
        return self.cursor.fetchone()

    def get_last_conversation(self, conversation_id):
        self.cursor.execute(
            """
        SELECT * FROM conversation_history WHERE conversation_id = ? and role = 'user'
        ORDER BY id DESC LIMIT 1
        """,
            (conversation_id,),
        )
        return self.cursor.fetchone()

    def get_message_by_id(self, message_id):
        """
        Retrieve a message by its message_id.

        Args:
            message_id: The unique message ID

        Returns:
            The message row or None if not found
        """
        self.cursor.execute(
            """
            SELECT * FROM conversation_history
            WHERE message_id = ?
            """,
            (message_id,),
        )
        return self.cursor.fetchone()

    def get_last_conversation_by_path(self, directory_path):
        most_recent_conversation_id = self.get_most_recent_conversation_id_by_path(
            directory_path
        )
        if most_recent_conversation_id and most_recent_conversation_id[0]:
            convo = self.get_conversations_by_id(most_recent_conversation_id[0])
            return convo
        return None

    def get_most_recent_conversation_id_by_path(self, path) -> Optional[str]:
        """Retrieve the most recent conversation ID for the current path."""
        self.cursor.execute(
            """
            SELECT conversation_id FROM conversation_history
            WHERE directory_path = ?
            ORDER BY timestamp DESC LIMIT 1
            """,
            (path,),
        )
        result = self.cursor.fetchone()
        return result

    def get_conversations_by_id(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Retrieve all messages for a specific conversation ID."""
        self.cursor.execute(
            """
            SELECT * FROM conversation_history
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
            """,
            (conversation_id,),
        )

        results = []
        for row in self.cursor.fetchall():
            message_dict = {
                "id": row[0],
                "message_id": row[1],
                "timestamp": row[2],
                "role": row[3],
                "content": row[4],
                "conversation_id": row[5],
                "directory_path": row[6],
                "model": row[7],
                "provider": row[8],
                "npc": row[9],
            }

            # Get attachments for this message
            attachments = self.get_message_attachments(row[1])
            if attachments:
                message_dict["attachments"] = attachments

            results.append(message_dict)

        return results

    def get_npc_conversation_stats(self, start_date=None, end_date=None):
        """
        Get conversation statistics grouped by NPC.
        Returns metrics like:
        - Total messages per NPC
        - Average message length 
        - Most common conversation topics
        - Response times
        - Models/providers used
        """
        date_filter = ""
        params = []
        if start_date and end_date:
            date_filter = "WHERE timestamp BETWEEN ? AND ?"
            params = [start_date, end_date]

        query = f"""
        SELECT 
            npc,
            COUNT(*) as total_messages,
            AVG(LENGTH(content)) as avg_message_length,
            COUNT(DISTINCT conversation_id) as total_conversations,
            COUNT(DISTINCT model) as models_used,
            COUNT(DISTINCT provider) as providers_used,
            GROUP_CONCAT(DISTINCT model) as model_list,
            GROUP_CONCAT(DISTINCT provider) as provider_list,
            MIN(timestamp) as first_conversation,
            MAX(timestamp) as last_conversation
        FROM conversation_history
        {date_filter}
        GROUP BY npc
        ORDER BY total_messages DESC
        """
        
        self.cursor.execute(query, params)
        return pd.DataFrame(self.cursor.fetchall(), columns=[
            'npc', 'total_messages', 'avg_message_length', 'total_conversations',
            'models_used', 'providers_used', 'model_list', 'provider_list',
            'first_conversation', 'last_conversation'
        ])

    def get_conversation_topic_analysis(self, npc=None, limit=1000):
        """
        Analyze conversation content to extract common topics and patterns.
        Optionally filter by specific NPC.
        """
        query = """
        WITH message_pairs AS (
            SELECT 
                ch1.conversation_id,
                ch1.content as user_message,
                ch2.content as assistant_response,
                ch1.timestamp as msg_time,
                ch1.npc
            FROM conversation_history ch1
            JOIN conversation_history ch2 
                ON ch1.conversation_id = ch2.conversation_id
                AND ch1.id < ch2.id
            WHERE ch1.role = 'user' 
                AND ch2.role = 'assistant'
                AND ch2.id = (
                    SELECT MIN(id) 
                    FROM conversation_history ch3
                    WHERE ch3.conversation_id = ch1.conversation_id
                    AND ch3.id > ch1.id
                    AND ch3.role = 'assistant'
                )
        """
        
        if npc:
            query += f" AND ch1.npc = ?" 
            params = [npc, limit]
        else:
            params = [limit]

        query += """
            ORDER BY ch1.timestamp DESC
            LIMIT ?
        )
        SELECT 
            conversation_id,
            user_message,
            assistant_response,
            msg_time,
            npc
        FROM message_pairs
        """

        self.cursor.execute(query, params)
        return pd.DataFrame(self.cursor.fetchall(), columns=[
            'conversation_id', 'user_message', 'assistant_response', 
            'timestamp', 'npc'
        ])

    def get_command_patterns(self, timeframe='day'):
        """
        Analyze command usage patterns over time.
        Timeframe can be 'hour', 'day', 'week', or 'month'.
        """
        time_group = {
            'hour': "strftime('%Y-%m-%d %H', timestamp)",
            'day': "strftime('%Y-%m-%d', timestamp)",
            'week': "strftime('%Y-%W', timestamp)",
            'month': "strftime('%Y-%m', timestamp)"
        }[timeframe]

        query = f"""
        WITH parsed_commands AS (
            SELECT 
                {time_group} as time_bucket,
                CASE 
                    WHEN command LIKE '/%%' THEN substr(command, 2, instr(substr(command, 2), ' ') - 1)
                    WHEN command LIKE 'npc %%' THEN substr(command, 5, instr(substr(command, 5), ' ') - 1)
                    ELSE command
                END as base_command
            FROM command_history
        )
        SELECT 
            time_bucket,
            base_command,
            COUNT(*) as usage_count
        FROM parsed_commands
        GROUP BY time_bucket, base_command
        ORDER BY time_bucket DESC, usage_count DESC
        """

        self.cursor.execute(query)
        return pd.DataFrame(self.cursor.fetchall(), columns=[
            'time_period', 'command', 'count'
        ])

    def get_conversation_flows(self, conversation_id=None):
        """
        Analyze the flow and structure of conversations.
        Shows message chains, response patterns, and interaction flows.
        """
        where_clause = "WHERE conversation_id = ?" if conversation_id else ""
        params = [conversation_id] if conversation_id else []

        query = f"""
        WITH message_sequence AS (
            SELECT 
                conversation_id,
                role,
                content,
                timestamp,
                npc,
                LAG(role) OVER (PARTITION BY conversation_id ORDER BY timestamp) as prev_role,
                LAG(content) OVER (PARTITION BY conversation_id ORDER BY timestamp) as prev_content,
                LEAD(role) OVER (PARTITION BY conversation_id ORDER BY timestamp) as next_role,
                LEAD(content) OVER (PARTITION BY conversation_id ORDER BY timestamp) as next_content
            FROM conversation_history
            {where_clause}
        )
        SELECT 
            conversation_id,
            role,
            content,
            timestamp,
            npc,
            prev_role,
            prev_content,
            next_role,
            next_content
        FROM message_sequence
        ORDER BY conversation_id, timestamp
        """

        self.cursor.execute(query, params)
        return pd.DataFrame(self.cursor.fetchall(), columns=[
            'conversation_id', 'role', 'content', 'timestamp', 'npc',
            'prev_role', 'prev_content', 'next_role', 'next_content'
        ])

    def analyze_npc_performance(self, start_date=None, end_date=None):
        """
        Analyze NPC performance metrics including:
        - Response times
        - Message completion rates
        - Error rates
        - Model/provider effectiveness
        """
        date_filter = ""
        params = []
        if start_date and end_date:
            date_filter = "WHERE ch1.timestamp BETWEEN ? AND ?"
            params = [start_date, end_date]

        query = f"""
        WITH response_times AS (
            SELECT 
                ch1.conversation_id,
                ch1.npc,
                ch1.model,
                ch1.provider,
                ch1.timestamp as request_time,
                MIN(ch2.timestamp) as response_time,
                ch1.content as user_message,
                ch2.content as assistant_response
            FROM conversation_history ch1
            LEFT JOIN conversation_history ch2 
                ON ch1.conversation_id = ch2.conversation_id
                AND ch1.id < ch2.id
                AND ch2.role = 'assistant'
            {date_filter}
            WHERE ch1.role = 'user'
            GROUP BY ch1.id
        )
        SELECT 
            npc,
            model,
            provider,
            COUNT(*) as total_interactions,
            AVG(julianday(response_time) - julianday(request_time)) * 24 * 60 as avg_response_time_minutes,
            COUNT(CASE WHEN response_time IS NULL THEN 1 END) as incomplete_responses,
            COUNT(CASE WHEN lower(assistant_response) LIKE '%error%' 
                   OR lower(assistant_response) LIKE '%failed%' 
                   OR lower(assistant_response) LIKE '%invalid%' 
                   OR lower(assistant_response) LIKE '%exception%' THEN 1 END) as error_count
        FROM response_times
        GROUP BY npc, model, provider
        ORDER BY total_interactions DESC
        """

        self.cursor.execute(query, params)
        return pd.DataFrame(self.cursor.fetchall(), columns=[
            'npc', 'model', 'provider', 'total_interactions', 
            'avg_response_time_mins', 'incomplete_responses', 'error_count'
        ])

    def get_conversation_sentiment_trends(self, window='day'):
        """
        Track sentiment trends in conversations over time.
        This provides a base table for sentiment analysis.
        """
        time_group = {
            'hour': "strftime('%Y-%m-%d %H', timestamp)",
            'day': "strftime('%Y-%m-%d', timestamp)",
            'week': "strftime('%Y-%W', timestamp)",
            'month': "strftime('%Y-%m', timestamp)"
        }[window]

        query = f"""
        SELECT 
            {time_group} as time_period,
            npc,
            role,
            GROUP_CONCAT(content, ' ') as messages,
            COUNT(*) as message_count
        FROM conversation_history
        GROUP BY time_period, npc, role
        ORDER BY time_period DESC, npc, role
        """

        self.cursor.execute(query)
        return pd.DataFrame(self.cursor.fetchall(), columns=[
            'time_period', 'npc', 'role', 'messages', 'message_count'
        ])


def start_new_conversation() -> str:
    """
    Starts a new conversation and returns a unique conversation ID.
    """
    return f"conversation_{datetime.now().strftime('%Y%m%d%H%M%S')}"


def save_conversation_message(
    command_history: CommandHistory,
    conversation_id: str,
    role: str,
    content: str,
    wd: str = None,
    model: str = None,
    provider: str = None,
    npc: str = None,
    attachments: List[Dict] = None,
    message_id: str = None,
):
    """
    Saves a conversation message linked to a conversation ID with optional attachments.

    Args:
        command_history: The CommandHistory instance
        conversation_id: The conversation identifier
        role: The message sender role ('user', 'assistant', etc.)
        content: The message content
        wd: Working directory (defaults to current directory)
        model: The model identifier (optional)
        provider: The provider identifier (optional)
        npc: The NPC identifier (optional)
        attachments: List of attachment dictionaries (optional)
            Each attachment dict should have:
            - name: Filename/title
            - type: MIME type or extension
            - data: Binary blob data
            - size: Size in bytes (optional)

    Returns:
        The message ID
    """
    if wd is None:
        wd = os.getcwd()

    return command_history.add_conversation(
        role=role,
        content=content,
        conversation_id=conversation_id,
        directory_path=wd,
        model=model,
        provider=provider,
        npc=npc,
        attachments=attachments,
        message_id=message_id,
    )


def retrieve_last_conversation(
    command_history: CommandHistory, conversation_id: str
) -> str:
    """
    Retrieves and formats all messages from the last conversation.
    """
    last_message = command_history.get_last_conversation(conversation_id)
    if last_message:
        return last_message[3]  # content
    return "No previous conversation messages found."


def save_attachment_to_message(
    command_history: CommandHistory,
    message_id: str,
    file_path: str,
    attachment_name: str = None,
    attachment_type: str = None,
):
    """
    Helper function to save a file from disk as an attachment.

    Args:
        command_history: The CommandHistory instance
        message_id: The message ID to attach to
        file_path: Path to the file on disk
        attachment_name: Name to save (defaults to basename)
        attachment_type: MIME type (defaults to guessing from extension)

    Returns:
        Boolean indicating success
    """
    try:
        # Get file name if not specified
        if not attachment_name:
            attachment_name = os.path.basename(file_path)

        # Try to guess MIME type if not specified
        if not attachment_type:
            _, ext = os.path.splitext(file_path)
            if ext:
                attachment_type = ext.lower()[1:]  # Remove the dot

        # Read file data
        with open(file_path, "rb") as f:
            data = f.read()

        # Add attachment
        command_history.add_attachment(
            message_id=message_id,
            attachment_name=attachment_name,
            attachment_type=attachment_type,
            attachment_data=data,
            attachment_size=len(data),
        )
        return True
    except Exception as e:
        print(f"Error saving attachment: {str(e)}")
        return False

def get_available_tables(db_path: str) -> str:
    """
    Function Description:
        This function gets the available tables in the database.
    Args:
        db_path (str): The database path.
    Keyword Args:
        None
    Returns:
        str: The available tables in the database.
    """

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name != 'command_history'"
            )
            tables = cursor.fetchall()

            return tables
    except Exception as e:
        print(f"Error getting available tables: {e}")
        return ""