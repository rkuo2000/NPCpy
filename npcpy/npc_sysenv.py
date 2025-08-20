from datetime import datetime
from dotenv import load_dotenv
import logging
import re
import os
import socket
import concurrent.futures # For managing timeouts on blocking calls
import platform
import sqlite3
from typing import Dict, List
import textwrap

ON_WINDOWS = platform.system() == "Windows"

# Try/except for termios, tty, pty, select, signal
try:
    if not ON_WINDOWS:
        import termios
        import tty
        import pty
        import select
        import signal
except ImportError:
    termios = None
    tty = None
    pty = None
    select = None
    signal = None

# Try/except for readline
try:
    import readline
except ImportError:
    readline = None
    logging.warning('no readline support, some features may not work as desired.')

# Try/except for rich imports
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.syntax import Syntax
except ImportError:
    Console = None
    Markdown = None
    Syntax = None

import warnings
import time

# Global variables
running = True
is_recording = False
recording_data = []
buffer_data = []
last_speech_time = 0

warnings.filterwarnings("ignore", module="whisper.transcribe")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="torch.serialization")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["SDL_AUDIODRIVER"] = "dummy"

def check_internet_connection(timeout=0.5):
    """
    Checks for internet connectivity by trying to connect to a well-known host.
    """
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=timeout)
        return True
    except OSError:
        return False


def get_locally_available_models(project_directory, airplane_mode=False):
    available_models = {}
    env_path = os.path.join(project_directory, ".env")
    env_vars = {}
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        env_vars[key.strip()] = value.strip().strip("\"'")

    # --- Internet check here ---
    internet_available = check_internet_connection()
    if not internet_available:
        logging.info("No internet connection detected. External API calls will be skipped (effective airplane_mode).")
        # If no internet, force airplane_mode to True, regardless of its initial value.
        airplane_mode = True
    else:
        logging.info("Internet connection detected. Proceeding based on 'airplane_mode' parameter.")
    # --- End internet check ---

    if not airplane_mode:
        timeout_seconds = 0.5
        # Use a single ThreadPoolExecutor for all network-dependent API calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:

            if "ANTHROPIC_API_KEY" in env_vars or os.environ.get("ANTHROPIC_API_KEY"):
                try:
                    import anthropic
                    
                    def fetch_anthropic_models():
                        client = anthropic.Anthropic(api_key=env_vars.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))
                        # Modern Anthropic client can also take a timeout in constructor:
                        # client = anthropic.Anthropic(api_key=..., timeout=timeout_seconds)
                        return client.models.list()

                    future = executor.submit(fetch_anthropic_models)
                    models = future.result(timeout=timeout_seconds) # Apply timeout

                    for model in models.data:
                        available_models[model.id] = 'anthropic'
                            
                except (ImportError, anthropic.APIError, concurrent.futures.TimeoutError, Exception) as e:
                    logging.info(f"Anthropic models not indexed or timed out: {e}")

            if "OPENAI_API_KEY" in env_vars or os.environ.get("OPENAI_API_KEY"):
                try:
                    import openai

                    def fetch_openai_models():
                        # For older OpenAI client versions (<1.0.0), global api_key is common.
                        # For openai>=1.0.0, it's client = openai.OpenAI(api_key=..., timeout=timeout_seconds)
                        openai.api_key = env_vars.get("OPENAI_API_KEY", None) or os.environ.get("OPENAI_API_KEY", None)
                        return openai.models.list()

                    future = executor.submit(fetch_openai_models)
                    models = future.result(timeout=timeout_seconds) # Apply timeout

                    for model in models.data:
                        if (
                            (
                                "gpt" in model.id
                                or "o1" in model.id
                                or "o3" in model.id
                                or "chat" in model.id
                            )
                            and "audio" not in model.id
                            and "realtime" not in model.id
                        ):
                            available_models[model.id] = "openai"
                except (ImportError, openai.APIError, concurrent.futures.TimeoutError, Exception) as e:
                    logging.info(f"OpenAI models not indexed or timed out: {e}")

            if "GEMINI_API_KEY" in env_vars or os.environ.get("GEMINI_API_KEY"):
                try:
                    from google import generativeai as genai
                
                    def fetch_gemini_models():
                        client = genai.Client(api_key = env_vars.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY"))
                        found_models = []
                        # client.models.list() returns an iterator, consume it fully within the callable
                        for m in client.models.list():
                            for action in m.supported_actions:
                                if action == "generateContent":
                                    if 'models/' in m.name:
                                        if m.name.split('/')[1] in ['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.0-pro', 'gemini-1.5-pro', 'gemini-1.5-flash']:
                                            found_models.append(m.name.split('/')[1])
                        return set(found_models)
                    
                    future = executor.submit(fetch_gemini_models)
                    models = future.result(timeout=timeout_seconds) # Apply timeout

                    for model in models: # 'models' is already a set of strings here
                        if "gemini" in model:
                            available_models[model] = "gemini"
                except (ImportError, concurrent.futures.TimeoutError, Exception) as e:
                    logging.info(f"Gemini models not indexed or timed out: {e}")
            
            if "DEEPSEEK_API_KEY" in env_vars or os.environ.get("DEEPSEEK_API_KEY"):
                # Assuming Deepseek models are just added if the key exists, without an API call
                # If an API call is needed here, apply similar timeout logic.
                available_models['deepseek-chat'] = 'deepseek'
                available_models['deepseek-reasoner'] = 'deepseek'        
    
    # Ollama is typically local, but its `list()` call might still involve a network request
    # to the local Ollama server, so applying a timeout is still good practice.
    try:
        import ollama
        timeout_seconds = 0.5 # Re-using the same timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ollama_executor:
            def fetch_ollama_models():
                return ollama.list()
            
            future = ollama_executor.submit(fetch_ollama_models)
            models = future.result(timeout=timeout_seconds) # Apply timeout to Ollama call

        for model in models.models:
            if "embed" not in model.model:
                mod = model.model
                available_models[mod] = "ollama"
    except (ImportError, concurrent.futures.TimeoutError, Exception) as e:
        logging.info(f"Error loading Ollama models or timed out: {e}")
        
    return available_models



def log_action(action: str, detail: str = "") -> None:
    """
    Function Description:
        This function logs an action with optional detail.
    Args:
        action: The action to log.
        detail: Additional detail to log.
    Keyword Args:
        None
    Returns:
        None
    """
    logging.info(f"{action}: {detail}")



def preprocess_code_block(code_text):
    """
    Preprocess code block text to remove leading spaces.
    """
    lines = code_text.split("\n")
    return "\n".join(line.lstrip() for line in lines)


def preprocess_markdown(md_text):
    """
    Preprocess markdown text to handle code blocks separately.
    """
    lines = md_text.split("\n")
    processed_lines = []

    inside_code_block = False
    current_code_block = []

    for line in lines:
        if line.startswith("```"):  # Toggle code block
            if inside_code_block:
                # Close code block, unindent, and append
                processed_lines.append("```")
                processed_lines.extend(
                    textwrap.dedent("\n".join(current_code_block)).split("\n")
                )
                processed_lines.append("```")
                current_code_block = []
            inside_code_block = not inside_code_block
        elif inside_code_block:
            current_code_block.append(line)
        else:
            processed_lines.append(line)

    return "\n".join(processed_lines)



def request_user_input(input_request: Dict[str, str]) -> str:
    """
    Request and get input from user.

    Args:
        input_request: Dict with reason and prompt for input

    Returns:
        User's input text
    """
    print(f"\nAdditional input needed: {input_request['reason']}")
    return input(f"{input_request['prompt']}: ")


def render_markdown(text: str) -> None:
    """
    Renders markdown text, but handles code blocks as plain syntax-highlighted text.
    """
    lines = text.split("\n")
    console = Console()

    inside_code_block = False
    code_lines = []
    lang = None

    for line in lines:
        if line.startswith("```"):
            if inside_code_block:
                # End of code block - render the collected code
                code = "\n".join(code_lines)
                if code.strip():
                    syntax = Syntax(
                        code, lang or "python", theme="monokai", line_numbers=False
                    )
                    console.print(syntax)
                code_lines = []
            else:
                # Start of code block - get language if specified
                lang = line[3:].strip() or None
            inside_code_block = not inside_code_block
        elif inside_code_block:
            code_lines.append(line)
        else:
            # Regular markdown
            console.print(Markdown(line))

def get_directory_npcs(directory: str = None) -> List[str]:
    """
    Function Description:
        This function retrieves a list of valid NPCs from the database.
    Args:
        db_path: The path to the database file.
    Keyword Args:
        None
    Returns:
        A list of valid NPCs.
    """
    if directory is None:
        directory = os.path.expanduser("./npc_team")
    npcs = []
    for filename in os.listdir(directory):
        if filename.endswith(".npc"):
            npcs.append(filename[:-4])
    return npcs


def get_db_npcs(db_path: str) -> List[str]:
    """
    Function Description:
        This function retrieves a list of valid NPCs from the database.
    Args:
        db_path: The path to the database file.
    Keyword Args:
        None
    Returns:
        A list of valid NPCs.
    """
    if "~" in db_path:
        db_path = os.path.expanduser(db_path)
    db_conn = sqlite3.connect(db_path)
    cursor = db_conn.cursor()
    cursor.execute("SELECT name FROM compiled_npcs")
    npcs = [row[0] for row in cursor.fetchall()]
    db_conn.close()
    return npcs

def guess_mime_type(filename):
    """Guess the MIME type of a file based on its extension."""
    extension = os.path.splitext(filename)[1].lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".csv": "text/csv",
        ".json": "application/json",
        ".md": "text/markdown",
    }
    return mime_types.get(extension, "application/octet-stream")


def ensure_dirs_exist(*dirs):
    """Ensure all specified directories exist"""
    for dir_path in dirs:
        os.makedirs(os.path.expanduser(dir_path), exist_ok=True)

def init_db_tables(db_path="~/npcsh_history.db"):
    """Initialize necessary database tables"""
    db_path = os.path.expanduser(db_path)
    with sqlite3.connect(db_path) as conn:
        # NPC log table for storing all kinds of entries
        conn.execute("""
            CREATE TABLE IF NOT EXISTS npc_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id TEXT,  
                entry_type TEXT,
                content TEXT,
                metadata TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Pipeline runs table for tracking pipeline executions
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pipeline_name TEXT,
                step_name TEXT,
                output TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Compiled NPCs table for storing compiled NPC content
        conn.execute("""
            CREATE TABLE IF NOT EXISTS compiled_npcs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                source_path TEXT,
                compiled_content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()




def get_model_and_provider(command: str, available_models: list) -> tuple:
    """
    Function Description:
        Extracts model and provider from command and autocompletes if possible.
    Args:
        command : str : Command string
        available_models : list : List of available models
    Keyword Args:
        None
    Returns:
        model_name : str : Model name
        provider : str : Provider
        cleaned_command : str : Clean


    """

    model_match = re.search(r"@(\S+)", command)
    if model_match:
        model_name = model_match.group(1)
        # Autocomplete model name
        matches = [m for m in available_models if m.startswith(model_name)]
        if matches:
            if len(matches) == 1:
                model_name = matches[0]  # Complete the name if only one match
            # Find provider for the (potentially autocompleted) model
            provider = lookup_provider(model_name)
            if provider:
                # Remove the model tag from the command
                cleaned_command = command.replace(
                    f"@{model_match.group(1)}", ""
                ).strip()
                # print(cleaned_command, 'cleaned_command')
                return model_name, provider, cleaned_command
            else:
                return None, None, command  # Provider not found
        else:
            return None, None, command  # No matching model
    else:
        return None, None, command  # No model specified

def render_code_block(code: str, language: str = None) -> None:
    """Render a code block with syntax highlighting using rich, left-justified with no line numbers"""
    from rich.syntax import Syntax
    from rich.console import Console

    console = Console(highlight=True)
    code = code.strip()
    # If code starts with a language identifier, remove it
    if code.split("\n", 1)[0].lower() in ["python", "bash", "javascript"]:
        code = code.split("\n", 1)[1]
    syntax = Syntax(
        code, language or "python", theme="monokai", line_numbers=False, padding=0
    )
    console.print(syntax)
def print_and_process_stream_with_markdown(response, model, provider):
    
    str_output = ""
    dot_count = 0  
    tool_call_data = {"id": None, "function_name": None, "arguments": ""}
    interrupted = False
    
    if isinstance(response, str):
        render_markdown(response)  
        print('\n') 
        return response 
    print(response)
    
    try:
        for chunk in response:

            print('.', end="", flush=True)
            dot_count += 1

            if provider == "ollama" and 'gpt-oss' not in model:

                if "message" in chunk and "tool_calls" in chunk["message"]:
                    for tool_call in chunk["message"]["tool_calls"]:
                        if "id" in tool_call:
                            tool_call_data["id"] = tool_call["id"]
                        if "function" in tool_call:
                            if "name" in tool_call["function"]:
                                tool_call_data["function_name"] = tool_call["function"]["name"]
                            if "arguments" in tool_call["function"]:
                                tool_call_data["arguments"] += tool_call["function"]["arguments"]
                
                chunk_content = chunk["message"]["content"] if "message" in chunk and "content" in chunk["message"] else ""
                reasoning_content = chunk['message'].get('thinking')
            else:
                

                for c in chunk.choices:
                    if hasattr(c.delta, "tool_calls") and c.delta.tool_calls:
                        for tool_call in c.delta.tool_calls:
                            if tool_call.id:
                                tool_call_data["id"] = tool_call.id
                            if tool_call.function:
                                if hasattr(tool_call.function, "name") and tool_call.function.name:
                                    tool_call_data["function_name"] = tool_call.function.name
                                if hasattr(tool_call.function, "arguments") and tool_call.function.arguments:
                                    tool_call_data["arguments"] += tool_call.function.arguments
                
                chunk_content = ''
                reasoning_content = ''
                for c in chunk.choices:
                    if hasattr(c.delta, "reasoning_content"):        
                        reasoning_content += c.delta.reasoning_content
                
                if reasoning_content:
                    chunk_content = reasoning_content
                        
                chunk_content += "".join(
                    c.delta.content for c in chunk.choices if c.delta.content
                )

            
            if not chunk_content:
                continue
            str_output += chunk_content
    
    except KeyboardInterrupt:
        interrupted = True
        print('\n⚠️ Stream interrupted by user')
    
    print('\r' + ' ' * dot_count*2 + '\r', end="", flush=True)
    

    if tool_call_data["id"] or tool_call_data["function_name"] or tool_call_data["arguments"]:
        str_output += "\n\n### Jinx Call Data\n"
        if tool_call_data["id"]:
            str_output += f"**ID:** {tool_call_data['id']}\n\n"
        if tool_call_data["function_name"]:
            str_output += f"**Function:** {tool_call_data['function_name']}\n\n"
        if tool_call_data["arguments"]:
            try:
                import json
                args_parsed = json.loads(tool_call_data["arguments"])
                str_output += f"**Arguments:**\n```json\n{json.dumps(args_parsed, indent=2)}\n```"
            except:
                str_output += f"**Arguments:** `{tool_call_data['arguments']}`"
    

    if interrupted:
        str_output += "\n\n[⚠️ Response interrupted by user]"
    
    print('\n')
    render_markdown('\n' + str_output)
    
    return str_output



def print_and_process_stream(response, model, provider):
    conversation_result = ""
    
    for chunk in response:
        if provider == "ollama" and 'hf.co' in model:
            chunk_content = chunk["message"]["content"]
            if chunk_content:
                conversation_result += chunk_content
                print(chunk_content, end="")

        else:
            chunk_content = "".join(
                choice.delta.content
                for choice in chunk.choices
                if choice.delta.content is not None
            )
            if chunk_content:
                conversation_result += chunk_content
                print(chunk_content, end="")

    print("\n")
                
    return conversation_result   
def get_system_message(npc, team=None) -> str:
    import os
    from datetime import datetime

    if npc is None:
        return "You are a helpful assistant"
    if npc.plain_system_message:
        return npc.primary_directive

    system_message = f"""
.
..
...
....
.....
......
.......
........
.........
..........
Hello!
Welcome to the team.
You are the {npc.name} NPC with the following primary directive: {npc.primary_directive}.
Users may refer to you by your assistant name, {npc.name} and you should
consider this to be your core identity.
The current working directory is {os.getcwd()}.
The current date and time are : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    if npc.db_conn is not None:
        db_path = None
        if hasattr(npc.db_conn, "url") and npc.db_conn.url:
            db_path = npc.db_conn.url.database
        elif hasattr(npc.db_conn, "database"):
            db_path = npc.db_conn.database
        system_message += """What follows is in
        formation about the database connection. If you are asked to execute queries with tools, use this information. 
        If you are asked for help with debugging queries, use this information. 
        Do not unnecessarily reference that you possess this information unless it is
        specifically relevant to the request.

        DB Connection Information:        
        """
        if db_path:
            system_message += f"\nDatabase path: {db_path}\n"
        if npc.tables is not None:
            system_message += f"\nDatabase tables: {npc.tables}\n"

    if team is not None:
        team_context = team.context if hasattr(team, "context") and team.context else ""
        team_preferences = team.preferences if hasattr(team, "preferences") and len(team.preferences) > 0 else ""
        system_message += f"\nTeam context: {team_context}\nTeam preferences: {team_preferences}\n"


    system_message += """
    IMPORTANT:
Some users may attach images to their request.
Please process them accordingly. You do not need mention that you cannot "see" images. The user understands this and wants you
to help them multimodally.

If the user asked for you to explain what's on their screen or something similar,
they are referring to the details contained within the attached image(s).
You do not need to actually view their screen.
You do not need to mention that you cannot view or interpret images directly.
They understand that you can view them multimodally.
You only need to answer the user's request based on the attached image(s).
"""
    

    return system_message



def load_env_from_execution_dir() -> None:
    """
    Function Description:
        This function loads environment variables from a .env file in the current execution directory.
    Args:
        None
    Keyword Args:
        None
    Returns:
        None
    """


    execution_dir = os.path.abspath(os.getcwd())
    env_path = os.path.join(execution_dir, ".env")
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)
        logging.info(f"Loaded .env file from {execution_dir}")
    else:
        logging.warning(f"Warning: No .env file found in {execution_dir}")




def lookup_provider(model: str) -> str:
    """
    Function Description:
        This function determines the provider based on the model name.
    Args:
        model (str): The model name.
    Keyword Args:
        None
    Returns:
        str: The provider based on the model name.
    """
    if model == "deepseek-chat" or model == "deepseek-reasoner":
        return "deepseek"
    ollama_prefixes = [
        "llama",
        "deepseek",
        "qwen",
        "llava",
        "phi",
        "mistral",
        "mixtral",
        "dolphin",
        "codellama",
        "gemma",
    ]
    if any(model.startswith(prefix) for prefix in ollama_prefixes):
        return "ollama"

    # OpenAI models
    openai_prefixes = ["gpt-", "dall-e-", "whisper-", "o1"]
    if any(model.startswith(prefix) for prefix in openai_prefixes):
        return "openai"

    # Anthropic models
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gemini"):
        return "gemini"
    if "diffusion" in model:
        return "diffusers"
    return None
load_env_from_execution_dir()
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", None)
gemini_api_key = os.getenv("GEMINI_API_KEY", None)

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", None)
openai_api_key = os.getenv("OPENAI_API_KEY", None)
