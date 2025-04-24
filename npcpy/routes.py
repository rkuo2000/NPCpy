from typing import Callable, Dict, Any, List, Optional, Union
import functools

class CommandRouter:
    def __init__(self):
        self.routes = {}
        self.help_info = {}
        self.shell_only = { }
        
    def route(self, command: str, help_text: str = "", shell_only: bool = False) -> Callable:
        """Decorator to register a function as a command route.
        
        Args:
            command: The command that triggers this route
            help_text: Documentation for this command shown in help
        """
        def wrapper(func):
            self.routes[command] = func
            self.help_info[command] = help_text
            self.shell_only[command] = shell_only
            
            @functools.wraps(func)
            def wrapped_func(*args, **kwargs):
                return func(*args, **kwargs)
            
            return wrapped_func
        return wrapper
    
    def get_route(self, command: str) -> Optional[Callable]:
        """Get the command route for a given command."""
        return self.routes.get(command)
    
    def execute(self, command: str, *args, **kwargs) -> Any:
        """Execute a command route with the given arguments."""
        route_func = self.get_route(command)
        if route_func:
            return route_func(*args, **kwargs)
        return None
    
    def get_commands(self) -> List[str]:
        """Return a list of all registered commands."""
        return list(self.routes.keys())
    
    def get_help(self, command: str = None) -> Dict[str, str]:
        """Get help for a specific command or all commands."""
        if command:
            if command in self.help_info:
                return {command: self.help_info[command]}
            return {}
        return self.help_info

router = CommandRouter()


from npcpy.npc_sysenv import render_code_block, render_markdown
from npcpy.llm_funcs import get_llm_response, execute_llm_command, get_llm_response, enter_spool_mode
from npcpy.data.web import execute_rag_command, execute_search_command

from npcpy.work.plan import execute_plan_command
from npcpy.work.plonk import execute_plonk_command
from npcpy.work.trigger import execute_trigger_command
from npcpy.modes.spool import enter_spool_mode
from npcpy.modes import enter_spool_mode, enter_guac_mode, enter_notes_mode, enter_plonk_mode, enter_pti_mode, enter_serve_mode, enter_sleep_mode, enter_wander_mode, enter_whisper_mode, enter_deep_research_mode, enter_command_mode


from npcpy.memory.knowledge_graph import breathe

###
### breathe 
###

@router.route("breathe", "Condense context on a regular cadence", shell_only=True)
def breathe_handler(command: str, *args, **kwargs):
    """Route for the breathe command.
    # Implement breathe command logic
    ## breathe: a method for condensing context on a regular cadence (# messages, len(context), etc) (under construction)
    -every 10 messages/7500 characters, condense the conversation into lessons learned. write the lessons learned down by the np
    for the day, then the npc will see the lessons they have learned that day in that folder as part of the context.
    """

    # ...
    return {"output": "Breathe result", "messages": kwargs.get("messages", [])}

###
### chat 
###
@router.route("chat", "Chat with an NPC")
def chat_handler(command: str, *args, **kwargs):
    """Route for the chat command.
    # Implement chat command logic
    ## Chat with an NPC
    Use the `/chat` macro to have a conversation with an NPC. You can also use `/c` as an alias for `/chat`.
    ```npcsh
    npcsh> /chat <npc_name>
    ```
    """
    response = get_llm_response(command, 
                                **kwargs)
    enter_spool_mode...
    
    return response

###
### Compile 
###
@router.route("compile", "Compile NPC profiles")
def compile_handler(command: str, *args, **kwargs):
    """Route for the compile command.
    # Implement compile command logic
        
    ###
    ### Compile
    ###


    Compile a specified NPC profile. This will make it available for use in npcsh interactions.
    ```npcsh
    npcsh> /compile <npc_file>
    ```
    You can also use `/com` as an alias for `/compile`. If no NPC file is specified, all NPCs in the npc_team directory will be compiled.

    Begin a conversations with a specified NPC by referencing their name
    ```npcsh
    npcsh> /<npc_name>:
    ```
    """
    # ...
    return {"output": "Compile result", "messages": kwargs.get("messages", [])}
###
### Command 
###
@router.route("cmd", "Execute a command")
@router.route("command", "Execute a command")
def cmd_handler(command: str, **kwargs):
    """Route for the cmd command."""
    return execute_llm_command(command, **kwargs)


@router.route("conjure", "Conjure an NPC or tool")
def conjure_handler(command: str, *args, **kwargs):
    """Route for the conjure command. 
    ###
    ### Conjure
    ###

    Otherwise, here are some more detailed examples of macros that can be used in npcsh:
    ## Conjure (under construction)
    Use the `/conjure` macro to generate an NPC, a NPC tool, an assembly line, a job, or an SQL model

    ```bash
    npc conjure -n name -t 'templates'
    ```"""
    # Implement conjure command logic
    # ...
    return {"output": "Conjure result", "messages": kwargs.get("messages", [])}

###
### Data
###

@router.route("data", "Enter data analysis mode", shell_only=True)
def data_handler(command: str, *args, **kwargs):
    """Route for the data command."""
    return {"output": "Data result", "messages": kwargs.get("messages", [])}

@router.route('debate')
def debate_handler(command: str, *args, **kwargs):
    """Route for the debate command.
    
        
    ###
    ### Debate (under construction)
    ###
    Use the `/debate` macro to have two or more NPCs debate a topic, problem, or question.

    For example:
    ```npcsh
    npcsh> /debate Should humans colonize Mars? npcs = ['sibiji', 'mark', 'ted']
    ```

    """
    # Implement debate command logic
    # ...
    return {"output": "Debate result", "messages": kwargs.get("messages", [])}


@router.route("flush", "Flush messages" , shell_only=True)
def flush_handler( *args, **kwargs):
    """Route for the flush command."""
    # Implement flush command logic
    flush_result = flush_messages(args, **kwargs)
    return {"output": "Flush result", "messages": kwargs.get("messages", [])}

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

@router.route("help", "Show help information")
def help_handler(command: str, *args, **kwargs):
    """Route for the help command."""
    return get_help()


def init_handler(command: str, *args, **kwargs):
    """Route for the init command."""
    # Implement init command logic
    # ...
    return {"output": "Init result", "messages": kwargs.get("messages", [])}

    return {"output": "Search result", "messages": kwargs.get("messages", [])}


###
### Notes
###

@router.route("notes", "Enter notes mode", shell_only=True)
def notes_handler(command: str, *args, **kwargs):
    """## Notes
    Jot down notes and store them within the npcsh database and in the current directory as a text file.
    ```npcsh
    npcsh> /notes
    ```

    """
    # ...
    return {"output": "Notes result", "messages": kwargs.get("messages", [])}


@router.route("ots", "Execute OTS command")
def ots_handler(command: str, *args, **kwargs):
    """Route for the ots command.
## Over-the-shoulder: Screenshots and image analysis

Use the /ots macro to take a screenshot and write a prompt for an LLM to answer about the screenshot.
```npcsh
npcsh> /ots

Screenshot saved to: /home/caug/.npcsh/screenshots/screenshot_1735015011.png

Enter a prompt for the LLM about this image (or press Enter to skip): describe whats in this image

The image displays a source control graph, likely from a version control system like Git. It features a series of commits represented by colored dots connected by lines, illustrating the project's development history. Each commit message provides a brief description of the changes made, including tasks like fixing issues, merging pull requests, updating README files, and adjusting code or documentation. Notably, several commits mention specific users, particularly "Chris Agostino," indicating collaboration and contributions to the project. The graph visually represents the branching and merging of code changes.
```

In bash:
```bash
npc ots
```



Alternatively, pass an existing image in like :
```npcsh
npcsh> /ots test_data/catfight.PNG
Enter a prompt for the LLM about this image (or press Enter to skip): whats in this ?

The image features two cats, one calico and one orange tabby, playing with traditional Japanese-style toys. They are each holding sticks attached to colorful pom-pom balls, which resemble birds. The background includes stylized waves and a red sun, accentuating a vibrant, artistic style reminiscent of classic Japanese art. The playful interaction between the cats evokes a lively, whimsical scene.
```

```bash
npc ots -f test_data/catfight.PNG
```
"""
    # Implement ots command logic
    
    # ...
    

    def ots(
        command_parts,
        npc=None,
        model: str = NPCSH_VISION_MODEL,
        provider: str = NPCSH_VISION_PROVIDER,
        api_url: str = NPCSH_API_URL,
        api_key: str = None,
        stream: bool = False,
    ):
        # check if there is a filename
        if len(command_parts) > 1:
            filename = command_parts[1]
            file_path = os.path.join(os.getcwd(), filename)
            # Get user prompt about the image
            user_prompt = input(
                "Enter a prompt for the LLM about this image (or press Enter to skip): "
            )
            #get image ready here
            
            output = get_llm_response(
            )
        else:
            output = capture_screenshot(npc=npc)
            user_prompt = input(
                "Enter a prompt for the LLM about this image (or press Enter to skip): "
            )
            #get image ready here
            
            output = get_llm_response(
            
            )
        return {"messages": [], "output": output}  # Return the message



    return {"output": "OTS command result", "messages": kwargs.get("messages", [])}
    

@router.route("plan", "Execute a plan command")
def plan_handler(command: str, *args, **kwargs):
    """Route for the plan command 
    ###
    ### Plan
    ###


    ## Plan : Schedule tasks to be run at regular intervals (under construction)
    Use the /plan macro to schedule tasks to be run at regular intervals.
    ```npcsh
    npcsh> /plan run a rag search for 'moonbeam' on the files in the current directory every 5 minutes
    ```

    ```npcsh
    npcsh> /plan record the cpu usage every 5 minutes
    ```

    ```npcsh
    npcsh> /plan record the apps that are using the most ram every 5 minutes
    ```

    """
    # Implement plan command logic
    # ...
    return execute_plan_command(command, **kwargs)

@router.route("plonk", "Execute a plonk command")
def plonk_handler(command: str, *args, **kwargs):
    """Route for the plonk command."""
    # Implement plonk command logic
    # ...
    return execute_plonk_command(command, **kwargs)


@router.route("rag", "Execute a RAG command")
def rag_handler(command: str, *args, **kwargs):
    """Route for the rag command."""
    # Implement rag command logic
    output = execute_rag_command(command, **kwargs)
    return output
@router.route("rehash", "Rehash the last message",shell_only=True)
def rehash_handler(command: str, *args, **kwargs):
    """Route for the rehash command."""
    # Implement rehash command logic
    # ...
    return rehash_command(command, **kwargs)


@router.route("sample", "Sample a command")
def sample_handler(command: str, *args, **kwargs):
    """Route for the sample command."""
    # Implement sample command logic
    output = get_llm_response(
        " ".join(command.split()[1:]),  # Skip the command name
        npc=npc,
        messages=[],
        model=model,
        provider=provider,
        stream=stream,
    )
    return output
@router.route("search", "Execute a search command")
def search_handler(command: str, *args, **kwargs):
    """Route for the search command."""
    # Implement search command logic
    # ...


@router.route("set", "Set configuration values")
def set_handler(command: str, *args, **kwargs):
    """Route for the set command."""
    # Implement set command logic
    # ...
    return {"output": "Set result", "messages": kwargs.get("messages", [])}
@router.route("sleep", "sleep")
def set_handler(command: str, *args, **kwargs):
    """Route for the sleep command."""
    # Implement set command logic
    # ...
    return {"output": "sleep result", "messages": kwargs.get("messages", [])}

@router.route("spool", "Enter spool mode")
def spool_handler(command: str, *args, **kwargs):
    """Route for the spool command."""
    # Implement spool command logic
    return enter_spool_mode(command, **kwargs)



@router.route("tools", "Show available tools")
def tools_handler(command: str, *args, **kwargs):
    """Route for the tools command."""
    # Implement tools command logic
    # ...
    return {"output": "Available tools", "messages": kwargs.get("messages", [])}    
@router.route("trigger", "Execute a trigger command")
def trigger_handler(command: str, *args, **kwargs):
    """Route for the trigger command."""
    # Implement trigger command logic
    # ...
    return {"output": "Trigger command result", "messages": kwargs.get("messages", [])}
@router.route("vixynt", "Generate images from text descriptions")
def vixynt_handler(command: str, *args, **kwargs):
    """Route for the vixynt command."""
    # Implement vixynt command logic

    filename = None
    if "filename=" in command:
        filename = command.split("filename=")[1].split()[0]
        command = command.replace(f"filename={filename}", "").strip()
    # Get user prompt about the image BY joining the rest of the arguments
    user_prompt = " ".join(command.split()[1:])

    output = generate_image(
        user_prompt, npc=npc, filename=filename, model=model, provider=provider
    )
    return {"output": "Image generation result", "messages": kwargs.get("messages", [])}

@router.route("wander", "Enter wander mode")
def wander_handler(command: str, *args, **kwargs):
    """Route for the whisper command."""
    # Implement whisper command logic
    return enter_wander_mode(command, **kwargs)

@router.route("whisper", "Enter whisper mode")
def whisper_handler(command: str, *args, **kwargs):
    """Route for the whisper command."""
    # Implement whisper command logic
    return enter_whisper_mode(command, **kwargs)

def execute_command( 
    command: str,
    npc: NPC = None,
    team: Team = None,
    model: str = None,
    provider: str = None,
    api_key: str = None,
    api_url: str = None,
    messages: list = None,
    conversation_id: str = None,
    stream: bool = False,
    embedding_model=None,
):
    """
    Function Description:
        Executes a command, with support for piping outputs between commands.
    Args:
        command : str : Command

        db_path : str : Database path

    Keyword Args:
        embedding_model :  Embedding model
        current_npc : NPC : Current NPC
        messages : list : Messages
    Returns:
        dict : dict : Dictionary
    """
    output = ""
    if len(command.strip()) == 0:
        return {"messages": messages, "output": output, "current_npc": npc}

    if messages is None:
        messages = []

    # Split commands by pipe, preserving the original parsing logic
    commands = command.split("|")
    # print(commands)
    available_models = get_available_models()

    # Track piped output between commands
    piped_outputs = []

    for idx, single_command in enumerate(commands):
        # Modify command if there's piped output from previous command

        if idx > 0:
            single_command, additional_args = parse_piped_command(single_command)
            if len(piped_outputs) > 0:
                single_command = replace_pipe_outputs(
                    single_command, piped_outputs, idx
                )
            if len(additional_args) > 0:
                single_command = f"{single_command} {' '.join(additional_args)}"

        messages.append({"role": "user", "content": single_command})
        # print(messages)

        if model is None:
            # note the only situation where id expect this to take precedent is when a frontend is specifying the model
            # to pass through at each time
            model_override, provider_override, command = get_model_and_provider(
                single_command, available_models[0]
            )
            if model_override is None:
                model_override = os.getenv("NPCSH_CHAT_MODEL")
            if provider_override is None:
                provider_override = os.getenv("NPCSH_CHAT_PROVIDER")
        else:
            model_override = model
            provider_override = provider
        if single_command.startswith("/"):
            result = execute_slash_command(
                single_command,
                npc=npc,
                messages=messages,
                model=model_override,
                provider=provider_override,
                stream=stream,
            )
            ## deal with stream here

            output = result.get("output", "")
            new_messages = result.get("messages", None)
            npc = result.get("current_npc", npc)
        else:
            # print(single_command)
            try:
                command_parts = shlex.split(single_command)
                # print(command_parts)
            except ValueError as e:
                if "No closing quotation" in str(e):
                    # Attempt to close unclosed quotes
                    if single_command.count('"') % 2 == 1:
                        single_command += '"'
                    elif single_command.count("'") % 2 == 1:
                        single_command += "'"
                    try:
                        command_parts = shlex.split(single_command)
                    except ValueError:
                        # fall back to regular split
                        command_parts = single_command.split()
            if command_parts[0] in interactive_commands:
                print(f"Starting interactive {command_parts[0]} session...")
                return_code = start_interactive_session(
                    interactive_commands[command_parts[0]]
                )
                return {
                    "messages": messages,
                    "output": f"Interactive {command_parts[0]} session ended with return code {return_code}",
                    "npc": npc,
                }
            elif command_parts[0] == "cd":
                change_dir_result = change_directory(command_parts, messages)
                messages = change_dir_result["messages"]
                output = change_dir_result["output"]
            elif command_parts[0] in BASH_COMMANDS:
                if command_parts[0] in TERMINAL_EDITORS:
                    return {
                        "messages": messages,
                        "output": open_terminal_editor(command),
                        "npc": npc,
                    }
                elif command_parts[0] in ["cat", "find", "who", "open", "which"]:
                    if not validate_bash_command(command_parts):
                        output = "Error: Invalid command syntax or arguments"
                        output = check_llm_command(
                            command,
                            npc=npc,
                            team=team,
                            messages=messages,
                            model=model_override,
                            provider=provider_override,
                            stream=stream,
                        )

                    else:
                        try:
                            result = subprocess.run(
                                command_parts, capture_output=True, text=True
                            )
                            output = result.stdout + result.stderr
                        except Exception as e:
                            output = f"Error executing command: {e}"

                elif command.startswith("open "):
                    try:
                        path_to_open = os.path.expanduser(
                            single_command.split(" ", 1)[1]
                        )
                        absolute_path = os.path.abspath(path_to_open)
                        expanded_command = [
                            "open",
                            absolute_path,
                        ]
                        subprocess.run(expanded_command, check=True)
                        output = f"Launched: {command}"
                    except subprocess.CalledProcessError as e:
                        output = colored(f"Error opening: {e}", "red")
                    except Exception as e:
                        output = colored(f"Error executing command: {str(e)}", "red")

                # Rest of BASH_COMMANDS handling remains the same
                else:
                    try:
                        result = subprocess.run(
                            command_parts, capture_output=True, text=True
                        )
                        output = result.stdout
                        if result.stderr:
                            output += colored(f"\nError: {result.stderr}", "red")

                        colored_output = ""
                        for line in output.split("\n"):
                            parts = line.split()
                            if parts:
                                filepath = parts[-1]
                                color, attrs = get_file_color(filepath)
                                colored_filepath = colored(filepath, color, attrs=attrs)
                                colored_line = " ".join(parts[:-1] + [colored_filepath])
                                colored_output += colored_line + "\n"
                            else:
                                colored_output += line
                        output = colored_output.rstrip()

                        if not output and result.returncode == 0:
                            output = colored(
                                f"Command '{single_command}' executed successfully (no output).",
                                "green",
                            )
                        print(output)
                    except Exception as e:
                        output = colored(f"Error executing command: {e}", "red")

            else:
                output = check_llm_command(
                    single_command,
                    npc=npc,
                    team=team,
                    messages=messages,
                    model=model_override,
                    provider=provider_override,
                    stream=stream,
                    api_key=api_key,
                    api_url=api_url,
                )

        if isinstance(output, dict):
            response = output.get("output", "")
            new_messages = output.get("messages", None)
            if new_messages is not None:
                messages = new_messages
            output = response
        if output:
            if npc is not None:
                print(f"{npc.name}> ", end="")
            
            if not stream:
                try:
                    render_markdown(output)
                except AttributeError:
                    print(output)

                piped_outputs.append(f'"{output}"')

                try:
                    # Prepare text to embed (both command and response)
                    texts_to_embed = [command, str(output) if output else ""]

                    # Generate embeddings
                    embeddings = get_embeddings(
                        texts_to_embed,
                    )

                    # Prepare metadata
                    metadata = [
                        {
                            "type": "command",
                            "timestamp": datetime.datetime.now().isoformat(),
                            "path": os.getcwd(),
                            "npc": npc.name if npc else None,
                            "conversation_id": conversation_id,
                        },
                        {
                            "type": "response",
                            "timestamp": datetime.datetime.now().isoformat(),
                            "path": os.getcwd(),
                            "npc": npc.name if npc else None,
                            "conversation_id": conversation_id,
                        },
                    ]
                    embedding_model = os.environ.get("NPCSH_EMBEDDING_MODEL")
                    embedding_provider = os.environ.get("NPCSH_EMBEDDING_PROVIDER")
                    collection_name = (
                        f"{embedding_provider}_{embedding_model}_embeddings"
                    )

                    try:
                        collection = chroma_client.get_collection(collection_name)
                    except Exception as e:
                        print(f"Warning: Failed to get collection: {str(e)}")
                        print("Creating new collection...")
                        collection = chroma_client.create_collection(collection_name)
                    date_str = datetime.datetime.now().isoformat()
                    current_ids = [f"cmd_{date_str}", f"resp_{date_str}"]

                    collection.add(
                        embeddings=embeddings,
                        documents=texts_to_embed,  
                        metadatas=metadata, 
                        ids=current_ids,
                    )

                except Exception as e:
                    print(f"Warning: Failed to store embeddings: {str(e)}")

    return {
        "messages": messages,
        "output": output,
        "conversation_id": conversation_id,
        "model": model,
        "current_path": os.getcwd(),
        "provider": provider,
        "npc": npc ,
        "team": team,
    }