import re
import os
import sys
import code # Still used by is_python_code if that's kept, but not directly in execute_python_code
import yaml
from pathlib import Path
import atexit
import traceback
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import argparse
import io
import importlib.util 

from npcpy.memory.command_history import CommandHistory
from npcpy.npc_compiler import Team, NPC
from npcpy.llm_funcs import get_llm_response
from npcpy.modes._state import initial_state as npcsh_initial_state
from npcpy.npc_sysenv import render_markdown, print_and_process_stream_with_markdown

try:
    import readline
except ImportError:
    readline = None

GUAC_REFRESH_PERIOD = os.environ.get('GUAC_REFRESH_PERIOD', 100)
READLINE_HISTORY_FILE = os.path.expanduser("~/.guac_readline_history")
try:
    npcsh_initial_state.GUAC_REFRESH_PERIOD = int(GUAC_REFRESH_PERIOD)
except ValueError:
    npcsh_initial_state.GUAC_REFRESH_PERIOD = 100 

@dataclass
class GuacState:
    current_mode: str = "agent"
    current_path: str = field(default_factory=os.getcwd)
    npc: Optional[NPC] = None
    team: Optional[Team] = None
    messages: List[Dict[str, str]] = field(default_factory=list)
    locals: Dict[str, Any] = field(default_factory=dict)
    command_history: Optional[CommandHistory] = None
    chat_model: Optional[str] = npcsh_initial_state.chat_model
    chat_provider: Optional[str] = npcsh_initial_state.chat_provider
    stream_output: bool = True
    config_dir: Optional[Path] = None
    src_dir: Optional[Path] = None
    command_count: int = 0
    compile_buffer: List[str] = field(default_factory=list)

def get_multiline_input_guac(prompt_str: str, state: GuacState) -> str:
    lines = list(state.compile_buffer) 
    current_prompt = prompt_str if not lines else "... "

    while True:
        try:
            line = input(current_prompt)
            lines.append(line)
            current_prompt = "... "

            if not line and len(lines) > 1 and not lines[-2].strip(): # Double empty line (empty, then another empty)
                lines.pop() 
                lines.pop() 
                break
            
            if not line and len(lines) == 1: # Single empty line entered
                lines.pop() 
                break

            # Heuristic for single complete lines (not starting a block)
            if len(lines) == 1 and line.strip():
                temp_line = line.strip()
                is_block_starter = re.match(r"^\s*(def|class|for|while|if|try|with|@)", temp_line)
                ends_with_colon_for_block = temp_line.endswith(":") and is_block_starter
                
                if not is_block_starter and not ends_with_colon_for_block: # if not a block starter
                    # Check for balanced brackets if it's not a block starter
                    # This helps for single-line expressions that might have colons (e.g. dicts, slices)
                    open_brackets = temp_line.count('(') - temp_line.count(')') + \
                                    temp_line.count('[') - temp_line.count(']') + \
                                    temp_line.count('{') - temp_line.count('}')
                    if open_brackets <= 0: # If all brackets are closed or more closing than opening
                        break
            
        except EOFError:
            print("\nGoodbye!")
            sys.exit(0)
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt")
            state.compile_buffer.clear()
            return ""
    
    full_input = "\n".join(lines)
    state.compile_buffer.clear() 
    return full_input

def is_python_code(text: str) -> bool: # Used by AGENT mode to decide if input is code or NL
    text = text.strip()
    if not text:
        return False
    try:
        # code.compile_command is good for this check as it knows about interactive input
        code.compile_command(text, symbol="exec") # Try 'exec' first as it's more general
        return True
    except (SyntaxError, OverflowError, ValueError):
        try:
            code.compile_command(text, symbol="eval")
            return True
        except (SyntaxError, OverflowError, ValueError):
            return False


def setup_guac_readline(history_file: str):
    if not readline:
        return
    try:
        readline.read_history_file(history_file)
    except FileNotFoundError:
        pass
    except OSError:
        pass
    try:
        if sys.stdin.isatty():
            readline.set_history_length(1000)
            try:
                readline.parse_and_bind("set enable-bracketed-paste on")
            except Exception:
                pass
    except Exception:
        pass

def save_guac_readline_history(history_file: str):
    if not readline:
        return
    try:
        readline.write_history_file(history_file)
    except OSError:
        pass
    except Exception:
        pass

def _load_guac_helpers_into_state(state: GuacState):
    if state.src_dir:
        main_module_path = state.src_dir / "main.py"
        if main_module_path.exists():
            try:
                guac_config_parent_path = str(state.src_dir.parent)
                if guac_config_parent_path not in sys.path:
                    sys.path.insert(0, guac_config_parent_path)
                src_path_str = str(state.src_dir)
                if src_path_str not in sys.path:
                     sys.path.insert(0, src_path_str)

                spec = importlib.util.spec_from_file_location("guac_main_helpers", main_module_path)
                if spec and spec.loader:
                    guac_main = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(guac_main)
                    for name in dir(guac_main):
                        if not name.startswith('__'):
                            state.locals[name] = getattr(guac_main, name)
                    state.locals['pd'] = pd
                    state.locals['np'] = np
                    state.locals['plt'] = plt
                    state.locals['datetime'] = datetime
                    state.locals['Path'] = Path
                    state.locals['os'] = os
                    state.locals['sys'] = sys
                    state.locals['json'] = json
                    state.locals['yaml'] = yaml
                    state.locals['re'] = re
                    state.locals['traceback'] = traceback # Expose traceback module
            except Exception as e:
                print(f"Warning: Could not load helpers from {main_module_path}: {e}", file=sys.stderr)

def setup_guac_mode(config_dir=None, plots_dir=None, npc_team_dir=None):
    home_dir = Path.home()
    config_dir = Path(config_dir) if config_dir else home_dir / ".npcsh" / "guac"
    plots_dir = Path(plots_dir) if plots_dir else config_dir / "plots"
    npc_team_dir = Path(npc_team_dir) if npc_team_dir else config_dir / "npc_team"
    src_dir = config_dir / "src"

    for p in [src_dir, plots_dir, npc_team_dir]:
        p.mkdir(parents=True, exist_ok=True)

    if not (config_dir / "__init__.py").exists():
        (config_dir / "__init__.py").touch()

    config_file = config_dir / "config.json"
    default_mode = "agent"
    lang = "python"
    current_config = {}

    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                current_config = json.load(f)
            default_mode = current_config.get("default_mode", "agent")
        except json.JSONDecodeError:
            pass 

    if not current_config or current_config.get("preferred_language") != lang:
        current_config = {
            "preferred_language": lang,
            "plots_directory": str(plots_dir),
            "npc_team_directory": str(npc_team_dir),
            "default_mode": default_mode
        }
        with open(config_file, "w") as f:
            json.dump(current_config, f, indent=2)

    os.environ["NPCSH_GUAC_LANG"] = lang
    os.environ["NPCSH_GUAC_PLOTS"] = str(plots_dir)
    os.environ["NPCSH_GUAC_TEAM"] = str(npc_team_dir)
    npcsh_initial_state.GUAC_DEFAULT_MODE = default_mode

    if not (src_dir / "__init__.py").exists():
        with open(src_dir / "__init__.py", "w") as f:
            f.write("# Guac source directory\n")

    main_py_content = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from pathlib import Path

def save_plot(name=None, plots_dir=None):
    if plots_dir is None:
        plots_dir = os.environ.get("NPCSH_GUAC_PLOTS", Path.home() / ".npcsh" / "guac" / "plots")
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{name}.png" if name else f"{timestamp}_plot.png"
    filepath = plots_dir / filename
    try:
        if plt.get_fignums(): # Check if there are any active figures
            plt.savefig(filepath)
            print(f"Plot saved to {filepath}")
        else:
            print("No active matplotlib plot to save.")
            return None
    except Exception as e:
        print(f"Error saving plot: {e}")
        return None
    return filepath

def read_img(img_path):
    try:
        from PIL import Image
        img = Image.open(img_path)
        img.show()
    except ImportError:
        print("PIL (Pillow) not available. Please install it: pip install Pillow")
    except FileNotFoundError:
        print(f"Image file not found: {img_path}")
    except Exception as e:
        print(f"Error reading image {img_path}: {e}")
    return img_path
"""
    if not (src_dir / "main.py").exists():
        with open(src_dir / "main.py", "w") as f:
            f.write(main_py_content)
    
    if str(config_dir) not in sys.path:
        sys.path.insert(0, str(config_dir))
    if str(config_dir.parent) not in sys.path:
        sys.path.insert(0, str(config_dir.parent))

    setup_npc_team(npc_team_dir, lang)
    return {
        "language": lang, "src_dir": src_dir, "config_path": config_file,
        "plots_dir": plots_dir, "npc_team_dir": npc_team_dir,
        "config_dir": config_dir, "default_mode": default_mode
    }

def setup_npc_team(npc_team_dir, lang):
    npc_data_list = [
        {"name": "guac", "primary_directive": f"You are guac, a Python data analysis assistant. Generate and explain Python code. If given a natural language query for a task, provide the Python code. If asked for information, answer and provide relevant Python code examples."},
    ]
    for npc_data in npc_data_list:
        with open(npc_team_dir / f"{npc_data['name']}.npc", "w") as f:
            yaml.dump(npc_data, f, default_flow_style=False)

    team_ctx_model = os.environ.get("NPCSH_CHAT_MODEL", npcsh_initial_state.chat_model or "llama3")
    team_ctx_provider = os.environ.get("NPCSH_CHAT_PROVIDER", npcsh_initial_state.chat_provider or "ollama")

    team_ctx = {
        "team_name": "guac_team", "description": f"A team for {lang} analysis",
        "foreman": "guac",
        "model": team_ctx_model,
        "provider": team_ctx_provider
    }
    npcsh_initial_state.chat_model = team_ctx_model
    npcsh_initial_state.chat_provider = team_ctx_provider
    with open(npc_team_dir / "team.ctx", "w") as f:
        yaml.dump(team_ctx, f, default_flow_style=False)

def print_guac_bowl():
    print("""
  🟢🟢🟢🟢🟢 
🟢          🟢                 
🟢  
🟢      
🟢      
🟢      🟢🟢🟢   🟢    🟢   🟢🟢🟢    🟢🟢🟢
🟢           🟢  🟢    🟢    ⚫⚫🟢  🟢        
🟢           🟢  🟢    🟢  ⚫🥑🧅⚫  🟢     
🟢           🟢  🟢    🟢  ⚫🥑🍅⚫  🟢      
 🟢🟢🟢🟢🟢🟢    🟢🟢🟢🟢    ⚫⚫🟢   🟢🟢🟢 
""")

def get_guac_prompt_char(command_count: int) -> str:
    period = int(npcsh_initial_state.GUAC_REFRESH_PERIOD)
    period = max(1, period)
    stages = ["\U0001F951", "\U0001F951🔪", "\U0001F951🥣", "\U0001F951🥣🧂", "\U0001F958 REFRESH?"]
    divisor = max(1, period // (len(stages)-1) if len(stages) > 1 else period)
    stage_index = min(command_count // divisor, len(stages) - 1)
    return stages[stage_index]

def _handle_guac_refresh(state: GuacState):
    if not state.command_history or not state.npc:
        print("Cannot refresh: command history or NPC not available.")
        return

    history_entries = state.command_history.get_all()
    if not history_entries:
        print("No command history to analyze for refresh.")
        return
    
    py_commands = []
    for entry in history_entries:
        if len(entry) > 2 and isinstance(entry[2], str) and entry[2].strip():
            if not entry[2].startswith('/'): # Heuristic: not a slash command
                 py_commands.append(entry[2])
    
    if not py_commands:
        print("No relevant commands in history to analyze for refresh.")
        return

    prompt = f"Analyze the following Python commands or natural language queries that led to Python code execution by a user:\n\n"
    prompt += "\n".join(py_commands[-20:]) 
    prompt += f"\n\nBased on these, suggest 1-3 useful Python helper functions that the user might find valuable. Provide only the Python code for these functions, wrapped in ```python ... ``` blocks. Do not include any other text or explanation outside the code blocks."

    try:
        response = get_llm_response(
            prompt,
            model=state.chat_model,
            provider=state.chat_provider,
            npc=state.npc,
            stream=False 
        )
        suggested_code_raw = response.get("response", "").strip()
        
        code_blocks = re.findall(r'```python\s*(.*?)\s*```', suggested_code_raw, re.DOTALL)
        if not code_blocks:
            if "def " in suggested_code_raw:
                 code_blocks = [suggested_code_raw]
            else:
                print("\nNo functions suggested by LLM or format not recognized.")
                return

        suggested_functions_code = "\n\n".join(block.strip() for block in code_blocks)

        if not suggested_functions_code.strip():
            print("\nLLM did not suggest any functions.")
            return

        print("\n=== Suggested Helper Functions ===\n")
        render_markdown(f"```python\n{suggested_functions_code}\n```")
        print("\n===============================\n")

        user_choice = input("Add these functions to your main.py? (y/n): ").strip().lower()
        if user_choice == 'y':
            main_py_path = state.src_dir / "main.py"
            with open(main_py_path, "a") as f:
                f.write("\n\n# --- Functions suggested by /refresh ---\n")
                f.write(suggested_functions_code)
                f.write("\n# --- End of suggested functions ---\n")
            print(f"Functions appended to {main_py_path}.")
            print("To use them in the current session, you might need to: import importlib; importlib.reload(guac.src.main); from guac.src.main import *")
        else:
            print("Suggested functions not added.")

    except Exception as e:
        print(f"Error during /refresh: {e}")
        traceback.print_exc()

def execute_python_code(code_str: str, state: GuacState) -> Tuple[GuacState, Any]:
    output_capture = io.StringIO()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    final_output_str = None
    is_expression = False

    try:
        sys.stdout = output_capture
        sys.stderr = output_capture

        if '\n' not in code_str.strip() and not re.match(r"^\s*(def|class|for|while|if|try|with|import|from|@)", code_str.strip()):
            try:
                compiled_expr = compile(code_str, "<input>", "eval")
                exec_result = eval(compiled_expr, state.locals)
                if exec_result is not None:
                    if not output_capture.getvalue().strip():
                         print(repr(exec_result), file=sys.stdout)
                is_expression = True 
            except SyntaxError: 
                is_expression = False
            except Exception: 
                is_expression = False
                raise # Re-raise runtime error during eval to be caught by outer try-except
        
        if not is_expression: 
            compiled_code = compile(code_str, "<input>", "exec")
            exec(compiled_code, state.locals)

    except SyntaxError as e: 
        exc_type, exc_value, _ = sys.exc_info() # Use _ for tb if not used
        error_lines = traceback.format_exception_only(exc_type, exc_value)
        # Manually adjust filename in error if it's <input> to avoid confusion
        adjusted_error_lines = [line.replace('File "<input>"', 'Syntax error in input') for line in error_lines]
        print("".join(adjusted_error_lines), file=output_capture, end="")
    except Exception:
        exc_type, exc_value, exc_tb = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_tb, file=output_capture)
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        final_output_str = output_capture.getvalue().strip()
        output_capture.close()
    
    if state.command_history:
        state.command_history.add_command(code_str, [final_output_str if final_output_str else ""], "", state.current_path)

    return state, final_output_str


def execute_guac_command(command: str, state: GuacState) -> Tuple[GuacState, Any]:
    stripped_command = command.strip()
    output = None 
    history_output_list_for_this_command = []


    if not stripped_command:
        return state, None

    if stripped_command.lower() in ["exit", "quit", "exit()", "quit()"]:
        raise SystemExit("Exiting Guac Mode.")

    if stripped_command.startswith("/"):
        parts = stripped_command.split(maxsplit=1)
        cmd_name = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if cmd_name == "/agent":
            state.current_mode = "agent"
            output = "Switched to AGENT mode."
        elif cmd_name == "/chat":
            state.current_mode = "chat"
            output = "Switched to CHAT mode."
        elif cmd_name == "/cmd":
            state.current_mode = "cmd"
            output = "Switched to CMD mode."
        elif cmd_name == "/ride":
            state.current_mode = "ride"
            output = "Switched to RIDE mode (placeholder)."
        elif cmd_name == "/refresh":
            _handle_guac_refresh(state)
            # _handle_guac_refresh prints its own status, so 'output' can be minimal or None
            output = "Refresh process initiated." 
        elif cmd_name == "/mode":
            output = f"Current mode: {state.current_mode.upper()}"
        elif cmd_name == "/show_vars":
            temp_output_list = ["Current Python Environment Variables:"]
            if state.locals:
                for k, v_obj in state.locals.items():
                    if not k.startswith("__"):
                        try:
                            v_repr = repr(v_obj)
                            temp_output_list.append(f"  {k}: {v_repr[:100]}{'...' if len(v_repr) > 100 else ''}")
                        except Exception:
                             temp_output_list.append(f"  {k}: <Error representing value>")
            else:
                temp_output_list.append("  (empty)")
            output = "\n".join(temp_output_list)
        elif cmd_name == "cd": 
            target_dir = args.strip() if args.strip() else str(Path.home())
            try:
                os.chdir(target_dir)
                state.current_path = os.getcwd()
                output = f"Changed directory to {state.current_path}"
            except FileNotFoundError:
                output = f"Error: Directory not found: {target_dir}"
            except Exception as e:
                output = f"Error changing directory: {e}"
        elif cmd_name == "ls":
            try:
                ls_path = args.strip() if args.strip() else state.current_path
                output = "\n".join(os.listdir(ls_path))
            except Exception as e:
                output = f"Error listing directory: {e}"
        elif cmd_name == "pwd":
            output = state.current_path
        elif cmd_name == "run" and args.strip().endswith(".py"):
            script_path = Path(args.strip())
            if script_path.exists():
                try:
                    with open(script_path, "r") as f:
                        script_code = f.read()
                    # execute_python_code will handle its own history for the script's execution.
                    # The output here is just for the 'run' command itself.
                    _, script_exec_output = execute_python_code(script_code, state) 
                    output = f"Executed script '{script_path}'.\nOutput from script:\n{script_exec_output if script_exec_output else '(No direct output)'}"
                except Exception as e:
                    output = f"Error running script {script_path}: {e}"
            else:
                output = f"Error: Script not found: {script_path}"
        else:
            output = f"Unknown slash command: {cmd_name}"
        
        history_output_list_for_this_command = [str(output)] if output is not None else [""]
        if state.command_history: # Log the slash command itself
            state.command_history.add_command(command, history_output_list_for_this_command, "", state.current_path)
        return state, output


    if state.current_mode == "agent":
        # `is_python_code` is a heuristic. `execute_python_code` will definitively determine via compile.
        if is_python_code(stripped_command) or state.compile_buffer: # `compile_buffer` implies it's code
            # `execute_python_code` handles its own history logging for the executed code block
            state, output = execute_python_code(stripped_command, state) 
            # No separate history entry here for the *input* `stripped_command` if it was direct code,
            # as `execute_python_code` logs the actual `code_str` it ran.
        else: # Natural language in agent mode
            locals_repr = {k:type(v).__name__ for k,v in state.locals.items() if not k.startswith('__') and type(v).__module__ != 'builtins'}
            prompt = f"""The user is in a Python environment (guac AGENT mode) and typed:
"{stripped_command}"
Generate Python code to address this query.
Return ONLY the executable Python code, without any markdown or explanation.
If the query is a question that doesn't require code, answer it briefly.
If you cannot generate code or answer, state that clearly starting with '# Cannot'.
Current Python locals available (first level, non-builtin): {locals_repr}
"""
            llm_response = get_llm_response(prompt, model=state.chat_model, provider=state.chat_provider, npc=state.npc, stream=False)
            generated_text = llm_response.get("response", "").strip()
            
            generated_code = None
            code_match = re.search(r"```python\s*(.*?)\s*```", generated_text, re.DOTALL)
            if code_match:
                generated_code = code_match.group(1).strip()
            elif "def " in generated_text or "import " in generated_text or "=" in generated_text or "print(" in generated_text:
                generated_code = generated_text
            else:
                output = generated_text # Assume textual answer

            if generated_code:
                if generated_code.startswith("# Cannot"):
                    output = generated_code
                else:
                    print(f"\n# LLM Generated Code (Agent Mode):\n---\n{generated_code}\n---\n")
                    # `execute_python_code` logs history of `generated_code`
                    _, exec_output = execute_python_code(generated_code, state) 
                    output = f"# Generated code executed.\n# Output from generated code:\n{exec_output if exec_output else '(No direct output)'}"
            elif not output: 
                 output = "# LLM did not generate Python code or a direct answer for this query."
            
            # Log the original natural language command and the LLM's processed output/summary
            history_output_list_for_this_command = [str(output)] if output is not None else [""]
            if state.command_history:
                 state.command_history.add_command(stripped_command, history_output_list_for_this_command, "", state.current_path)


    elif state.current_mode == "chat":
        state.messages.append({"role": "user", "content": stripped_command})
        llm_response_dict = get_llm_response(
            stripped_command, 
            model=state.chat_model,
            provider=state.chat_provider,
            npc=state.npc,
            messages=state.messages,
            stream=state.stream_output 
        )
        output = llm_response_dict.get("response") 
        state.messages = llm_response_dict.get("messages", state.messages) 

        log_output_str = output if not (state.stream_output and hasattr(output, '__iter__') and not isinstance(output, (str, bytes))) else "[Streamed LLM Response]"
        history_output_list_for_this_command = [str(log_output_str)]
        if state.command_history:
            state.command_history.add_command(stripped_command, history_output_list_for_this_command, "", state.current_path)

    elif state.current_mode == "cmd":
        locals_repr = {k:type(v).__name__ for k,v in state.locals.items() if not k.startswith('__') and type(v).__module__ != 'builtins'}
        prompt_cmd = f"""User wants to execute the following in Python:
"{stripped_command}"
Generate ONLY the executable Python code to achieve this. No markdown, no explanation.
If not possible, start response with '# Error:'.
Current Python locals available (first level, non-builtin): {locals_repr}
"""
        llm_response = get_llm_response(prompt_cmd, model=state.chat_model, provider=state.chat_provider, npc=state.npc, stream=False)
        generated_text = llm_response.get("response", "").strip()
        
        generated_code = None
        code_match = re.search(r"```python\s*(.*?)\s*```", generated_text, re.DOTALL)
        if code_match:
            generated_code = code_match.group(1).strip()
        elif generated_text.strip() and not generated_text.startswith("# Error:"):
            generated_code = generated_text

        if generated_code and not generated_code.startswith("# Error:"):
            print(f"\n# LLM Generated Code (Cmd Mode):\n---\n{generated_code}\n---\n")
            # `execute_python_code` logs history of `generated_code`
            _, exec_output = execute_python_code(generated_code, state) 
            output = f"# Generated code executed.\n# Output from generated code:\n{exec_output if exec_output else '(No direct output)'}"
        else:
            output = f"# LLM indicated an error or did not generate valid code: {generated_text}"
        
        history_output_list_for_this_command = [str(output)] if output is not None else [""]
        if state.command_history: # Log the original natural language command
             state.command_history.add_command(stripped_command, history_output_list_for_this_command, "", state.current_path)


    elif state.current_mode == "ride":
        output = "RIDE mode is not yet implemented. Your input was: " + stripped_command
        history_output_list_for_this_command = [str(output)]
        if state.command_history:
            state.command_history.add_command(stripped_command, history_output_list_for_this_command, "", state.current_path)

    return state, output


def run_guac_repl(initial_guac_state: GuacState):
    state = initial_guac_state
    _load_guac_helpers_into_state(state) 
    print_guac_bowl()
    print(f"Welcome to Guac Mode! Current mode: {state.current_mode.upper()}. Type slash commands like /agent, /chat, /cmd.")

    while True:
        try:
            state.current_path = os.getcwd()
            path_display = Path(state.current_path).name
            prompt_char = get_guac_prompt_char(state.command_count)
            mode_display = state.current_mode.upper()
            npc_display = f":{state.npc.name}" if state.npc and state.npc.name else ""
            
            prompt_str = f"[{path_display}|{mode_display}{npc_display}] {prompt_char} > "
            
            user_input = get_multiline_input_guac(prompt_str, state)

            if not user_input.strip() and not state.compile_buffer:
                if state.compile_buffer: state.compile_buffer.clear()
                continue
            
            if not user_input.strip() and state.compile_buffer: # Should not happen if get_multiline clears buffer
                user_input = "\n".join(state.compile_buffer)
                state.compile_buffer.clear()
                if not user_input.strip(): continue

            state.command_count +=1
            new_state, result = execute_guac_command(user_input, state)
            state = new_state 

            if result is not None:
                # Handle streaming output for chat mode specifically
                if state.stream_output and state.current_mode == "chat" and hasattr(result, '__iter__') and not isinstance(result, (str, bytes, dict)):
                    full_streamed_output_for_history = print_and_process_stream_with_markdown(result, state.chat_model, state.chat_provider)
                    if state.messages and state.messages[-1].get("role") == "assistant": 
                         state.messages[-1]["content"] = full_streamed_output_for_history
                    
                    if state.command_history:
                        # Attempt to update the last history entry's output with the full streamed text
                        # This requires CommandHistory to have get_last_entry_id and update_command_output methods
                        try:
                            last_entry_id = state.command_history.get_last_entry_id() 
                            if last_entry_id:
                                state.command_history.update_command_output(last_entry_id, [full_streamed_output_for_history])
                        except AttributeError: 
                            pass # CommandHistory might not support this
                elif isinstance(result, str): # For non-streamed string results
                    if result.strip(): render_markdown(result) 
                # Non-string, non-stream results (e.g. direct output from Python code not captured as string)
                # This case should ideally be handled by execute_python_code returning a string.
                # If `result` is something else, it implies an unhandled output type.
                elif not (state.current_mode == "chat" and state.stream_output): 
                    if result: print(str(result)) 
            print() 

        except (KeyboardInterrupt, EOFError): 
            print("\nExiting Guac Mode...")
            break
        except SystemExit as e:
            print(f"\n{e}")
            break
        except Exception:
            print("An unexpected error occurred in the REPL:")
            traceback.print_exc()

def enter_guac_mode(npc_obj=None, team_obj=None, config_dir_str=None, plots_dir_str=None, npc_team_dir_str=None,
                    refresh_period_val=None, lang_choice=None, default_mode_choice=None): 
    
    if refresh_period_val is not None:
        try:
            npcsh_initial_state.GUAC_REFRESH_PERIOD = int(refresh_period_val)
        except ValueError:
            pass

    setup_result = setup_guac_mode(
        config_dir=config_dir_str,
        plots_dir=plots_dir_str,
        npc_team_dir=npc_team_dir_str
    )

    guac_config_dir = setup_result["config_dir"]
    guac_src_dir = setup_result["src_dir"]
    guac_npc_team_dir = setup_result["npc_team_dir"]
    guac_default_mode = default_mode_choice or setup_result.get("default_mode", "agent")

    cmd_history = CommandHistory() 
    
    current_npc = npc_obj
    current_team = team_obj

    if current_npc is None and current_team is None: 
        try:
            current_team = Team(team_path=str(guac_npc_team_dir), db_conn=None)
            if current_team and current_team.npcs:
                 current_npc = current_team.get_npc("guac") 
                 if not current_npc: 
                     current_npc = current_team.get_foreman() or next(iter(current_team.npcs.values()), None)
        except Exception as e:
            print(f"Warning: Could not load Guac NPC team from {guac_npc_team_dir}: {e}", file=sys.stderr)

    initial_guac_state = GuacState(
        current_mode=guac_default_mode,
        npc=current_npc,
        team=current_team,
        command_history=cmd_history,
        chat_model=npcsh_initial_state.chat_model,
        chat_provider=npcsh_initial_state.chat_provider,
        config_dir=guac_config_dir,
        src_dir=guac_src_dir,
        locals={} 
    )

    try:
        setup_guac_readline(READLINE_HISTORY_FILE)
        atexit.register(save_guac_readline_history, READLINE_HISTORY_FILE)
    except Exception as e:
        print(f'Could not set up readline: {e}', file=sys.stderr)
    
    atexit.register(cmd_history.close) 

    run_guac_repl(initial_guac_state)

def main():
    parser = argparse.ArgumentParser(description="Enter Guac Mode - Interactive Python with LLM assistance.")
    parser.add_argument("--config_dir", type=str, help="Guac configuration directory.")
    parser.add_argument("--plots_dir", type=str, help="Directory to save plots.")
    parser.add_argument("--npc_team_dir", type=str, default=os.path.expanduser('~/.npcsh/guac/npc_team/'), help="NPC team directory for Guac.")
    parser.add_argument("--refresh_period", type=int, help="Number of commands before suggesting /refresh.")
    parser.add_argument("--default_mode", type=str, choices=["agent", "chat", "cmd", "ride"], help="Default mode to start in.")

    args = parser.parse_args()

    enter_guac_mode(
        config_dir_str=args.config_dir,
        plots_dir_str=args.plots_dir,
        npc_team_dir_str=args.npc_team_dir,
        refresh_period_val=args.refresh_period,
        default_mode_choice=args.default_mode
    )

if __name__ == "__main__":
    main()