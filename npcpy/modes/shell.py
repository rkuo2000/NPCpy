import os
import sys
import readline
import atexit
from inspect import isgenerator
from termcolor import colored
from sqlalchemy import create_engine
from npcpy.npc_sysenv import (
    print_and_process_stream_with_markdown,
    NPCSH_STREAM_OUTPUT,
    NPCSH_CHAT_MODEL,
    NPCSH_CHAT_PROVIDER,
    NPCSH_API_URL,
    setup_npcsh_config,
    is_npcsh_initialized,
    initialize_base_npcs_if_needed,
    
)

from npcpy.memory.command_history import (
    CommandHistory,
    start_new_conversation,
    save_conversation_message,
)
from npcpy.helpers import (
)
from npcpy.modes.shell_helpers import (
    complete,  # For command completion
    readline_safe_prompt,
    get_multiline_input,
    setup_readline,
    execute_command,

    orange,  # For colored prompt
)
from npcpy.npc_compiler import (
    NPC, Team
)

import argparse
import importlib.metadata  
try:
    VERSION = importlib.metadata.version(
        "npcpy"
    )  
except importlib.metadata.PackageNotFoundError:
    VERSION = "unknown"  


TERMINAL_EDITORS = ["vim", "emacs", "nano"]


EMBEDDINGS_DB_PATH = os.path.expanduser("~/npcsh_chroma.db")

try:
    import chromadb

    chroma_client = chromadb.PersistentClient(path=EMBEDDINGS_DB_PATH)
except:
    chroma_client = None


def readline_safe_prompt(prompt: str) -> str:
    """
    Function Description:
        Escapes ANSI escape sequences in the prompt.
    Args:
        prompt : str : Prompt
    Keyword Args:
        None
    Returns:
        prompt : str : Prompt

    """
    # This regex matches ANSI escape sequences
    ansi_escape = re.compile(r"(\033\[[0-9;]*[a-zA-Z])")

    # Wrap them with \001 and \002
    def escape_sequence(m):
        return "\001" + m.group(1) + "\002"

    return ansi_escape.sub(escape_sequence, prompt)


def parse_piped_command(current_command):
    """
    Parse a single command for additional arguments.
    """
    # Use shlex to handle complex argument parsing
    if "/" not in current_command:
        return current_command, []

    try:
        command_parts = shlex.split(current_command)
        # print(command_parts)
    except ValueError:
        # Fallback if quote parsing fails
        command_parts = current_command.split()
        # print(command_parts)
    # Base command is the first part
    base_command = command_parts[0]

    # Additional arguments are the rest
    additional_args = command_parts[1:] if len(command_parts) > 1 else []

    return base_command, additional_args

def print_tools(tools):
    output = "Available tools:"
    for tool in tools:
        output += f"  {tool.tool_name}"
        output += f"   Description: {tool.description}"
        output += f"   Inputs: {tool.inputs}"
    return output





def replace_pipe_outputs(command: str, piped_outputs: list, cmd_idx: int) -> str:
    """
    Replace {0}, {1}, etc. placeholders with actual piped outputs.

    Args:
        command (str): Command with potential {n} placeholders
        piped_outputs (list): List of outputs from previous commands

    Returns:
        str: Command with placeholders replaced
    """
    placeholders = [f"{{{cmd_idx-1}}}", f"'{{{cmd_idx-1}}}'", f'"{{{cmd_idx-1}}}"']
    if str(cmd_idx - 1) in command:
        for placeholder in placeholders:
            command = command.replace(placeholder, str(output))
    elif cmd_idx > 0 and len(piped_outputs) > 0:
        # assume to pipe the previous commands output to the next command
        command = command + " " + str(piped_outputs[-1])
    return command


def get_npc_from_command(command: str) -> Optional[str]:
    """
    Function Description:
        This function extracts the NPC name from a command string.
    Args:
        command: The command string.

    Keyword Args:
        None
    Returns:
        The NPC name if found, or None
    """

    parts = command.split()
    npc = None
    for part in parts:
        if part.startswith("npc="):
            npc = part.split("=")[1]
            break
    return npc

def open_terminal_editor(command: str) -> None:
    """
    Function Description:
        This function opens a terminal-based text editor.
    Args:
        command: The command to open the editor.
    Keyword Args:
        None
    Returns:
        None
    """

    try:
        os.system(command)
    except Exception as e:
        print(f"Error opening terminal editor: {e}")

def setup_readline() -> str:
    """
    Function Description:
        Sets up readline for the npcsh shell.
    Args:
        None
    Keyword Args:
        None
    Returns:
        history_file : str : History file
    """
    history_file = os.path.expanduser("~/.npcsh_history")
    try:
        readline.read_history_file(history_file)
    except FileNotFoundError:
        pass

    readline.set_history_length(1000)
    readline.parse_and_bind("set enable-bracketed-paste on")  # Enable paste mode
    readline.parse_and_bind(r'"\e[A": history-search-backward')
    readline.parse_and_bind(r'"\e[B": history-search-forward')
    readline.parse_and_bind(r'"\C-r": reverse-search-history')
    readline.parse_and_bind(r'\C-e: end-of-line')
    readline.parse_and_bind(r'\C-a: beginning-of-line')

    return history_file


def save_readline_history():
    readline.write_history_file(os.path.expanduser("~/.npcsh_readline_history"))



def get_multiline_input(prompt: str) -> str:
    """
    Function Description:
        Gets multiline input from the user.
    Args:
        prompt : str : Prompt
    Keyword Args:
        None
    Returns:
        lines : str : Lines

    """
    lines = []
    current_prompt = prompt
    while True:
        try:
            line = input(current_prompt)
        except EOFError:
            print("Goodbye!")
            break

        if line.endswith("\\"):
            lines.append(line[:-1])  # Remove the backslash
            # Use a continuation prompt for the next line
            current_prompt = readline_safe_prompt("> ")
        else:
            lines.append(line)
            break

    return "\n".join(lines)


def execute_slash_command(
    command: str,
    npc: NPC = None,
    team: Team = None,
    messages=None,
    model: str = None,
    provider: str = None,
    api_url: str = None,
    conversation_id: str = None,
    stream: bool = False,
):
    """
    Function Description:
        Executes a slash command.
    Args:
        command : str : Command

    Keyword Args:
        embedding_model : None : Embedding model
        current_npc : None : Current NPC
        text_data : None : Text data
        text_data_embedded : None : Embedded text data
        messages : None : Messages
    Returns:
        dict : dict : Dictionary
    """

    command = command[1:]

    log_action("Command Executed", command)

    command_parts = command.split()
    command_name = command_parts[0] if len(command_parts) >= 1 else None
    args = command_parts[1:] if len(command_parts) >= 1 else []

    current_npc = npc
    if team is not None:
        if command_name in team.npcs:
            current_npc = team.npcs.get(command_name)
            output = f"Switched to NPC: {current_npc.name}"
        return {"messages": messages, "output": output, "current_npc": current_npc}
    print(command)
    print(command_name == "ots")
       
    if command_name == "compile" or command_name == "com":
        try:
            """ 

            if len(args) > 0:  # Specific NPC file(s) provided
                for npc_file in args:
                    # differentiate between .npc and .pipe
                    if npc_file.endswith(".pipe"):
                        # Initialize the PipelineRunner with the appropriate parameters
                        pipeline_runner = PipelineRunner(
                            pipeline_file=npc_file,  # Uses the current NPC file
                            db_path="~/npcsh_history.db",  # Ensure this path is correctly set
                            npc_root_dir="./npc_team",  # Adjust this to your actual NPC directory
                        )

                        # Execute the pipeline and capture the output
                        output = pipeline_runner.execute_pipeline()

                        # Format the output if needed
                        output = f"Compiled Pipeline: {output}\n"
                    elif npc_file.endswith(".npc"):
                        compiled_script = npc_compiler.compile(npc_file)

                        output = f"Compiled NPC profile: {compiled_script}\n"
            elif current_npc:  # Compile current NPC
                compiled_script = npc_compiler.compile(current_npc)
                output = f"Compiled NPC profile: {compiled_script}"
            else:  # Compile all NPCs in the directory
                output = ""
                for filename in os.listdir(npc_compiler.npc_directory):
                    if filename.endswith(".npc"):
                        try:
                            compiled_script = npc_compiler.compile(
                                npc_compiler.npc_directory + "/" + filename
                            )
                            output += (
                                f"Compiled NPC profile: {compiled_script['name']}\n"
                            )
                        except Exception as e:
                            output += f"Error compiling {filename}: {str(e)}\n"
             """
        except Exception as e:
            import traceback

            output = f"Error compiling NPC profile: {str(e)}\n{traceback.format_exc()}"
            print(output)
    elif command_name == "tools":
        return {"messages": messages, "output": print_tools('Team tools: '+
                                                            team.tools_dict.values() if team else []
                                                            +
                                                            'NPC Tools: '+
                                                            npc.tools_dict.values() if npc else []
                                                            )}
    elif command_name == "plan":
        return execute_plan_command(
            command,
            npc=npc,
            model=model,
            provider=provider,
            api_url=api_url,
            messages=messages,
        )
    elif command_name == "trigger":
        return execute_trigger_command(
            command,
            npc=npc,
            model=model,
            provider=provider,
            api_url=api_url,
            messages=messages,
        )

    elif command_name == "plonk":
        request = " ".join(args)
        plonk_call = plonk(
            request, action_space, model=model, provider=provider, npc=npc
        )
        return {"messages": messages, "output": plonk_call, "current_npc": current_npc}
    elif command_name == "wander":
        return enter_wander_mode(args, messages, npc_compiler, npc, model, provider)


        
        
    elif command_name == "flush":
        n = float("inf")  # Default to infinite
        for arg in args:
            if arg.startswith("n="):
                try:
                    n = int(arg.split("=")[1])
                except ValueError:
                    return {
                        "messages": messages,
                        "output": "Error: 'n' must be an integer." + "\n",
                    }

        flush_result = flush_messages(n, messages)
        return flush_result  # Return the result of flushing messages

    # Handle /rehash command
    elif command_name == "rehash":
        rehash_result = rehash_last_message(
            conversation_id, model=model, provider=provider, npc=npc
        )
        return rehash_result  # Return the result of rehashing last message

    elif command_name == "pipe":
        # need to fix
        if len(args) > 0:  # Specific NPC file(s) provided
            for npc_file in args:
                # differentiate between .npc and .pipe
                pipeline_runner = PipelineRunner(
                    pipeline_file=npc_file,  # Uses the current NPC file
                    db_path="~/npcsh_history.db",  # Ensure this path is correctly set
                    npc_root_dir="./npc_team",  # Adjust this to your actual NPC directory
                )

                # run through the steps in the pipe
    elif command_name == "select":
        query = " ".join([command_name] + args)  # Reconstruct full query

        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchall()

                if not rows:
                    output = "No results found"
                else:
                    # Get column names
                    columns = [description[0] for description in cursor.description]

                    # Format output as table
                    table_lines = []
                    table_lines.append(" | ".join(columns))
                    table_lines.append("-" * len(table_lines[0]))

                    for row in rows:
                        table_lines.append(" | ".join(str(col) for col in row))

                    output = "\n".join(table_lines)

                return {"messages": messages, "output": output}

        except sqlite3.Error as e:
            output = f"Database error: {str(e)}"
            return {"messages": messages, "output": output}
    elif command_name == "init":
        output = initialize_npc_project()
        return {"messages": messages, "output": output}
    elif (
        command.startswith("vixynt")
        or command.startswith("vix")
        or (command.startswith("v") and command[1] == " ")
    ):
        # check if "filename=..." is in the command
        filename = None
        if "filename=" in command:
            filename = command.split("filename=")[1].split()[0]
            command = command.replace(f"filename={filename}", "").strip()
        # Get user prompt about the image BY joining the rest of the arguments
        user_prompt = " ".join(command.split()[1:])

        output = generate_image(
            user_prompt, npc=npc, filename=filename, model=model, provider=provider
        )

    elif command_name == "ots":
        print('fasdh')
        return {"messages":messages, "output":ots(
            command_parts, model=model, provider=provider, npc=npc, api_url=api_url, 
            stream=stream
        )}
    elif command_name == "help":  # New help command
        print(get_help())
        return {
            "messages": messages,
            "output": get_help(),
        }

    elif command_name == "whisper":
        # try:
        messages = enter_whisper_mode(npc=npc)
        output = "Exited whisper mode."
        # except Exception as e:
        #    print(f"Error entering whisper mode: {str(e)}")
        #    output = "Error entering whisper mode"

    elif command_name == "notes":
        output = enter_notes_mode(npc=npc)
    elif command_name == "data":
        # print("data")
        output = enter_data_mode(npc=npc)
        # output = enter_observation_mode(, npc=npc)
    elif command_name == "cmd" or command_name == "command":
        output = execute_llm_command(
            command,
            npc=npc,
            stream=stream,
            messages=messages,
        )

    elif command_name == "search":
        result = execute_search_command(
            command,
            messages=messages,
        )
        messages = result["messages"]
        output = result["output"]
        return {
            "messages": messages,
            "output": output,
            "current_npc": current_npc,
        }
    elif command_name == "rag":
        result = execute_rag_command(command, messages=messages)
        messages = result["messages"]
        output = result["output"]
        return {
            "messages": messages,
            "output": output,
            "current_npc": current_npc,
        }

    elif command_name == "roll":

        output = generate_video(
            command,
            model=NPCSH_VIDEO_GEN_MODEL,
            provider=NPCSH_VIDEO_GEN_PROVIDER,
            npc=npc,
            messages=messages,
        )
        messages = output["messages"]
        output = output["output"]

    elif command_name == "set":
        parts = command.split()
        if len(parts) == 3 and parts[1] in ["model", "provider", "db_path"]:
            output = execute_set_command(parts[1], parts[2])
        else:
            return {
                "messages": messages,
                "output": "Invalid set command. Usage: /set [model|provider|db_path] 'value_in_quotes' ",
            }
    elif command_name == "search":
        output = execute_search_command(
            command,
            messages=messages,
        )
        messages = output["messages"]
        # print(output, type(output))
        output = output["output"]
        # print(output, type(output))
    elif command_name == "sample":
        output = execute_llm_question(
            " ".join(command.split()[1:]),  # Skip the command name
            npc=npc,
            messages=[],
            model=model,
            provider=provider,
            stream=stream,
        )
    elif command_name == "spool" or command_name == "sp":
        inherit_last = 0
        device = "cpu"
        rag_similarity_threshold = 0.3
        for part in args:
            if part.startswith("inherit_last="):
                try:
                    inherit_last = int(part.split("=")[1])
                except ValueError:
                    return {
                        "messages": messages,
                        "output": "Error: inherit_last must be an integer",
                    }
            if part.startswith("device="):
                device = part.split("=")[1]
            if part.startswith("rag_similarity_threshold="):
                rag_similarity_threshold = float(part.split("=")[1])
            if part.startswith("model="):
                model = part.split("=")[1]

            if part.startswith("provider="):
                provider = part.split("=")[1]
            if part.startswith("api_url="):
                api_url = part.split("=")[1]
            if part.startswith("api_key="):
                api_key = part.split("=")[1]

                # load the npc properly

        match = re.search(r"files=\s*\[(.*?)\]", command)
        files = []
        if match:
            # Extract file list from the command
            files = [
                file.strip().strip("'").strip('"') for file in match.group(1).split(",")
            ]

            # Call the enter_spool_mode with the list of files
        else:
            files = None

        if len(command_parts) >= 2 and command_parts[1] == "reattach":
            command_history = CommandHistory()
            last_conversation = command_history.get_last_conversation_by_path(
                os.getcwd()
            )
            print(last_conversation)
            if last_conversation:
                spool_context = [
                    {"role": part["role"], "content": part["content"]}
                    for part in last_conversation
                ]

                print(f"Reattached to previous conversation:\n\n")
                output = enter_spool_mode(
                    inherit_last,
                    files=files,
                    npc=npc,
                    model=model,
                    provider=provider,
                    rag_similarity_threshold=rag_similarity_threshold,
                    device=device,
                    messages=spool_context,
                    conversation_id=conversation_id,
                    stream=stream,
                )
                return {"messages": output["messages"], "output": output}

            else:
                return {"messages": [], "output": "No previous conversation found."}

        output = enter_spool_mode(
            inherit_last,
            files=files,
            npc=npc,
            rag_similarity_threshold=rag_similarity_threshold,
            device=device,
            conversation_id=conversation_id,
            stream=stream,
        )
        return {"messages": output["messages"], "output": output}

    elif npc is not None:
        if command_name in npc.tools_dict:
            tool = npc.tools_dict.get(command_name) or team.tools_dict.get(command_name)
            return execute_tool_command(
                tool,
                args,
                messages,
                npc=npc,
            )
    elif team is not None:
        if command_name in team.tools_dict:
            tool = team.tools_dict.get(command_name)
            return execute_tool_command(
                tool,
                args,
                messages,
                npc=npc,
            )
    output = f"Unknown command: {command_name}"

    return {
        "messages": messages,
        "output": output,
        "current_npc": current_npc,
    }
def get_file_color(filepath: str) -> tuple:
    """
    Function Description:
        Returns color and attributes for a given file path.
    Args:
        filepath : str : File path
    Keyword Args:
        None
    Returns:
        color : str : Color
        attrs : list : List of attributes

    """

    if os.path.isdir(filepath):
        return "blue", ["bold"]
    elif os.access(filepath, os.X_OK):
        return "green", []
    elif filepath.endswith((".zip", ".tar", ".gz", ".bz2", ".xz", ".7z")):
        return "red", []
    elif filepath.endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff")):
        return "magenta", []
    elif filepath.endswith((".py", ".pyw")):
        return "yellow", []
    elif filepath.endswith((".sh", ".bash", ".zsh")):
        return "green", []
    elif filepath.endswith((".c", ".cpp", ".h", ".hpp")):
        return "cyan", []
    elif filepath.endswith((".js", ".ts", ".jsx", ".tsx")):
        return "yellow", []
    elif filepath.endswith((".html", ".css", ".scss", ".sass")):
        return "magenta", []
    elif filepath.endswith((".md", ".txt", ".log")):
        return "white", []
    elif filepath.startswith("."):
        return "cyan", []
    else:
        return "white", []


def wrap_text(text: str, width: int = 80) -> str:
    """
    Function Description:
        Wraps text to a specified width.
    Args:
        text : str : Text to wrap
        width : int : Width of text
    Keyword Args:
        None
    Returns:
        lines : str : Wrapped text
    """
    lines = []
    for paragraph in text.split("\n"):
        lines.extend(textwrap.wrap(paragraph, width=width))
    return "\n".join(lines)

def global_completions(text: str, command_parts: list) -> list:
    """
    Function Description:
        Handles global autocompletions for the npcsh shell.
    Args:
        text : str : Text to autocomplete
        command_parts : list : List of command parts
    Keyword Args:
        None
    Returns:
        completions : list : List of completions

    """
    if not command_parts:
        return [c + " " for c in valid_commands if c.startswith(text)]
    elif command_parts[0] in ["/compile", "/com"]:
        # Autocomplete NPC files
        return [f for f in os.listdir(".") if f.endswith(".npc") and f.startswith(text)]
    elif command_parts[0] == "/read":
        # Autocomplete filenames
        return [f for f in os.listdir(".") if f.startswith(text)]
    else:
        # Default filename completion
        return [f for f in os.listdir(".") if f.startswith(text)]

def complete(text: str, state: int) -> str:
    """
    Function Description:
        Handles autocompletion for the npcsh shell.
    Args:
        text : str : Text to autocomplete
        state : int : State
    Keyword Args:
        None
    Returns:
        None

    """
    buffer = readline.get_line_buffer()
    available_chat_models, available_reasoning_models = get_available_models()
    available_models = available_chat_models + available_reasoning_models

    # If completing a model name
    if "@" in buffer:
        at_index = buffer.rfind("@")
        model_text = buffer[at_index + 1 :]
        model_completions = [m for m in available_models if m.startswith(model_text)]

        try:
            # Return the full text including @ symbol
            return "@" + model_completions[state]
        except IndexError:
            return None

    # If completing a command
    elif text.startswith("/"):
        command_completions = [c for c in valid_commands if c.startswith(text)]
        try:
            return command_completions[state]
        except IndexError:
            return None

    return None
def main() -> None:
    """
    Main function for the npcsh shell and server.
    Starts either the Flask server or the interactive shell based on the argument provided.
    """
    # Set up argument parsing to handle 'serve' and regular commands

    check_old_par_name = os.environ.get("NPCSH_MODEL", None)
    if check_old_par_name is not None:
        # raise a deprecation warning
        print(
            """Deprecation Warning: NPCSH_MODEL and NPCSH_PROVIDER were deprecated in v0.3.5 in favor of NPCSH_CHAT_MODEL and NPCSH_CHAT_PROVIDER instead.\
                Please update your environment variables to use the new names.
                """
        )

    parser = argparse.ArgumentParser(description="npcsh CLI")
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"npcsh version {VERSION}",  # Use the dynamically fetched version
    )
    args = parser.parse_args()

    setup_npcsh_config()
    if "NPCSH_DB_PATH" in os.environ:
        db_path = os.path.expanduser(os.environ["NPCSH_DB_PATH"])
    else:
        db_path = os.path.expanduser("~/npcsh_history.db")

    command_history = CommandHistory(db_path)

    readline.set_completer_delims(" \t\n")
    readline.set_completer(complete)
    if sys.platform == "darwin":
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")

    # check if ./npc_team exists
    if os.path.exists("./npc_team"):

        npc_directory = os.path.abspath("./npc_team/")
    else:
        npc_directory = os.path.expanduser("~/.npcsh/npc_team/")


    os.makedirs(npc_directory, exist_ok=True)
    """ 
    # Compile all NPCs in the user's npc_team directory
    for filename in os.listdir(npc_directory):
        if filename.endswith(".npc"):
            npc_file_path = os.path.join(npc_directory, filename)
            npc_compiler.compile(npc_file_path)

    # Compile NPCs from project-specific npc_team directory
    if os.path.exists(npc_directory):
        for filename in os.listdir(npc_directory):
            if filename.endswith(".npc"):
                npc_file_path = os.path.join(npc_directory, filename)
                npc_compiler.compile(npc_file_path) """

    if not is_npcsh_initialized():
        print("Initializing NPCSH...")
        initialize_base_npcs_if_needed(db_path)
        print(
            "NPCSH initialization complete. Please restart your terminal or run 'source ~/.npcshrc' for the changes to take effect."
        )

    history_file = setup_readline()
    atexit.register(readline.write_history_file, history_file)
    atexit.register(command_history.close)
    # make npcsh into ascii art
    from colorama import init

    init()  # Initialize colorama for ANSI code support
    if sys.stdin.isatty():

        print(
            """
    Welcome to \033[1;94mnpc\033[0m\033[1;38;5;202msh\033[0m!
    \033[1;94m                    \033[0m\033[1;38;5;202m               \\\\
    \033[1;94m _ __   _ __    ___ \033[0m\033[1;38;5;202m ___  | |___    \\\\
    \033[1;94m| '_ \ | '_ \  / __|\033[0m\033[1;38;5;202m/ __/ | |_ _|    \\\\
    \033[1;94m| | | || |_) |( |__ \033[0m\033[1;38;5;202m\_  \ | | | |    //
    \033[1;94m|_| |_|| .__/  \___|\033[0m\033[1;38;5;202m|___/ |_| |_|   //
            \033[1;94m| |          \033[0m\033[1;38;5;202m              //
            \033[1;94m| |
            \033[1;94m|_|

    Begin by asking a question, issuing a bash command, or typing '/help' for more information.
            """
        )

    current_npc = None
    messages = None
    current_conversation_id = start_new_conversation()
    team = Team(team_path=npc_directory)
    sibiji = NPC(file=os.path.expanduser("~/.npcsh/npc_team/sibiji.npc")    )
    npc = sibiji
    if not sys.stdin.isatty():
        for line in sys.stdin:
            user_input = line.strip()
            if not user_input:
                continue  # Skip empty lines
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                sys.exit(0)
            result = execute_command(
                user_input,
                db_path,
                npc = npc,
                team=team,
                model=NPCSH_CHAT_MODEL,
                provider=NPCSH_CHAT_PROVIDER,
                messages=messages,
                conversation_id=current_conversation_id,
                stream=NPCSH_STREAM_OUTPUT,
                api_url=NPCSH_API_URL,
            )
            messages = result.get("messages", messages)
            if "current_npc" in result:
                current_npc = result["current_npc"]
            output = result.get("output")
            conversation_id = result.get("conversation_id")
            model = result.get("model")
            provider = result.get("provider")
            npc = result.get("npc")
            team = result.get("team")
            messages = result.get("messages")
            current_path = result.get("current_path")
            attachments = result.get("attachments")
            npc_name = (
                npc.name
                if isinstance(npc, NPC)
                else npc if isinstance(npc, str) else None
            )
            
            save_conversation_message(
                command_history,
                conversation_id,
                "user",
                user_input,
                wd=current_path,
                model=model,
                provider=provider,
                npc=npc_name,
                attachments=attachments,
            )
            if NPCSH_STREAM_OUTPUT and (
                isgenerator(output)
                or (hasattr(output, "__iter__") and hasattr(output, "__next__"))
            ):
                output = print_and_process_stream_with_markdown(    
                                                                output, model, provider)

                    
        save_conversation_message(
            command_history,
            conversation_id,
            "assistant",
            output,
            wd=current_path,
            model=model,
            provider=provider,
            npc=npc_name,
        )
        sys.exit(0)

    while True:
        try:
            if current_npc:
                prompt = f"{colored(os.getcwd(), 'blue')}:{orange(current_npc.name)}> "
            else:
                prompt = f"{colored(os.getcwd(), 'blue')}:\033[1;94mnpc\033[0m\033[1;38;5;202msh\033[0m!> "

            prompt = readline_safe_prompt(prompt)
            user_input = get_multiline_input(prompt).strip()
            if not user_input:
                continue
            
            if user_input.lower() in ["exit", "quit"]:
                if current_npc:
                    print(f"Exiting {current_npc.name} mode.")
                    current_npc = None
                    continue
                else:
                    print("Goodbye!")
                    break
            if npc is not None:
                print(f"{npc.name}>", end="")

            result = execute_command(
                user_input,
                npc= npc,
                team=team,
                model=NPCSH_CHAT_MODEL,
                provider=NPCSH_CHAT_PROVIDER,
                messages=messages,
                conversation_id=current_conversation_id,
                stream=NPCSH_STREAM_OUTPUT,
                api_url=NPCSH_API_URL,
            )

            messages = result.get("messages", messages)

            # need to adjust the output for the messages to all have
            # model, provider, npc, timestamp, role, content
            # also messages

            if "npc" in result:

                npc = result["npc"]
            output = result.get("output")

            conversation_id = result.get("conversation_id")
            model = result.get("model")
            provider = result.get("provider")

            messages = result.get("messages")
            current_path = result.get("current_path")
            attachments = result.get("attachments")

            if current_npc is not None:
                if isinstance(current_npc, NPC):
                    npc_name = current_npc.name
                elif isinstance(current_npc, str):
                    npc_name = current_npc
            else:
                npc_name = None
            message_id = save_conversation_message(
                command_history,
                conversation_id,
                "user",
                user_input,
                wd=current_path,
                model=model,
                provider=provider,
                npc=npc_name,
                attachments=attachments,
            )


            #import pdb 
            #pdb.set_trace()
            str_output = ""
            try:
                if NPCSH_STREAM_OUTPUT and hasattr(output, "__iter__"):

                    buffer = ""
                    in_code = False
                    code_buffer = ""

                    for chunk in output:

                        chunk_content = "".join(
                            c.delta.content for c in chunk.choices if c.delta.content
                        )
                        if not chunk_content:
                            continue

                        str_output += chunk_content
                        # print(str_output, "str_output")
                        # Process the content character by character
                        for char in chunk_content:
                            buffer += char

                            # Check for triple backticks
                            if buffer.endswith("```"):
                                if not in_code:
                                    # Start of code block
                                    in_code = True
                                    # Print everything before the backticks
                                    print(buffer[:-3], end="")
                                    buffer = ""
                                    code_buffer = ""
                                else:
                                    # End of code block
                                    in_code = False
                                    # Remove the backticks from the end of the buffer
                                    buffer = buffer[:-3]
                                    # Add buffer to code content and render
                                    code_buffer += buffer

                                    # Check for and strip language tag
                                    if (
                                        "\n" in code_buffer
                                        and code_buffer.index("\n") < 15
                                    ):
                                        first_line, rest = code_buffer.split("\n", 1)
                                        if (
                                            first_line.strip()
                                            and not "```" in first_line
                                        ):
                                            code_buffer = rest

                                    # Render the code block
                                    render_code_block(code_buffer)

                                    # Reset buffers
                                    buffer = ""
                                    code_buffer = ""
                            elif in_code:
                                # Just add to code buffer
                                code_buffer += char
                                if len(buffer) >= 3:  # Keep buffer small while in code
                                    buffer = buffer[-3:]
                            else:
                                # Regular text - print if buffer gets too large
                                if len(buffer) > 100:
                                    print(buffer[:-3], end="")
                                    buffer = buffer[
                                        -3:
                                    ]  # Keep last 3 chars to check for backticks

                    # Handle any remaining content
                    if in_code:
                        render_code_block(code_buffer)
                    else:
                        print(buffer, end="")

                    if str_output:
                        output = str_output
            except:
                output = None

            print("\n")

            if isinstance(output, str):
                save_conversation_message(
                    command_history,
                    conversation_id,
                    "assistant",
                    output,
                    wd=current_path,
                    model=model,
                    provider=provider,
                    npc=npc_name,
                )

            # if there are attachments in most recent user sent message, save them
            # save_attachment_to_message(command_history, message_id, # file_path, attachment_name, attachment_type)

            if (
                result["output"] is not None
                and not user_input.startswith("/")
                and not isinstance(result, dict)
            ):
                print("final", result)

        except (KeyboardInterrupt, EOFError):
            if current_npc:
                print(f"\nExiting {current_npc.name} mode.")
                current_npc = None
            else:
                print("\nGoodbye!")
                break


if __name__ == "__main__":
    main()
