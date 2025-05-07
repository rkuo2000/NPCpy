import re
import os
import sys
import code
import yaml
from pathlib import Path
import atexit
import traceback
from npcpy.memory.command_history import CommandHistory, start_new_conversation
from npcpy.npc_compiler import Team, NPC
from npcpy.llm_funcs import get_llm_response
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import argparse
from npcpy.modes._state import initial_state
try:
    import readline
except:
    pass


GUAC_REFRESH_PERIOD  = os.environ.get('GUAC_REFRESH_PERIOD', 100)
READLINE_HISTORY_FILE = os.path.expanduser("~/.guac_readline_history")

initial_state.GUAC_REFRESH_PERIOD = GUAC_REFRESH_PERIOD


def setup_guac_mode(config_dir=None, plots_dir=None, npc_team_dir=None):
    home_dir = Path.home()
    if config_dir is None:
        config_dir = home_dir / ".npcsh" / "guac"
    else:
        config_dir = Path(config_dir)

    if plots_dir is None:
        plots_dir = config_dir / "plots"
    else:
        plots_dir = Path(plots_dir)

    if npc_team_dir is None:
        npc_team_dir = config_dir / "npc_team"
    else:
        npc_team_dir = Path(npc_team_dir)

    src_dir = config_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    npc_team_dir.mkdir(parents=True, exist_ok=True)

    init_path = config_dir / "__init__.py"
    if not init_path.exists():
        with open(init_path, "w") as f:
            pass

    config_file = config_dir / "config.json"
    if config_file.exists():
        with open(config_file, "r") as f:
            config = json.load(f)
            lang = config.get("preferred_language", "python")
    else:
        print("Welcome to guac mode!")
        print("Please select your preferred language:")
        print("1. Python")
        print("2. R")
        print("3. JavaScript")
        choice = input("Enter choice (1-3, default: 1): ").strip()

        lang_map = {"1": "python", "2": "r", "3": "javascript", "": "python"}
        lang = lang_map.get(choice, "python")

        config = {
            "preferred_language": lang,
            "plots_directory": str(plots_dir),
            "npc_team_directory": str(npc_team_dir)
        }
        with open(config_file, "w") as f:
            json.dump(config, f)

    os.environ["NPCSH_GUAC_LANG"] = lang
    os.environ["NPCSH_GUAC_PLOTS"] = str(plots_dir)
    os.environ["NPCSH_GUAC_TEAM"] = str(npc_team_dir)

    if lang == "python":
        src_init_path = src_dir / "__init__.py"
        if not src_init_path.exists():
            with open(src_init_path, "w") as f:
                f.write("from .main import *\n")

        main_path = src_dir / "main.py"
        if not main_path.exists():
            with open(main_path, "w") as f:
                f.write("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from pathlib import Path

def save_plot(name=None, plots_dir=None):
    if plots_dir is None:
        plots_dir = os.environ.get("NPCSH_GUAC_PLOTS")
        if plots_dir is None:
            plots_dir = Path.home() / ".npcsh" / "guac" / "plots"
        else:
            plots_dir = Path(plots_dir)

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if name:
        filename = f"{timestamp}_{name}.png"
    else:
        filename = f"{timestamp}_plot.png"

    filepath = plots_dir / filename
    plt.savefig(filepath)
    print(f"Plot saved to {filepath}")
    return filepath

def read_img(img_path):
    try:
        from PIL import Image
        img = Image.open(img_path)
        img.show()
    except ImportError:
        print("PIL not available, can't display image")

    return img_path
""")
    elif lang == "r":
        main_path = src_dir / "main.R"
        if not main_path.exists():
            with open(main_path, "w") as f:
                f.write("""library(dplyr)
library(ggplot2)

save_plot <- function(plot = last_plot(), name = NULL, plots_dir = NULL) {
  if (is.null(plots_dir)) {
    plots_dir <- Sys.getenv("NPCSH_GUAC_PLOTS", unset = NA)
    if (is.na(plots_dir)) {
      plots_dir <- file.path(Sys.getenv("HOME"), ".npcsh", "guac", "plots")
    }
  }

  dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)

  timestamp <- format(Sys.time(), "%Y%m%d%H%M%S")
  if (is.null(name)) {
    filename <- paste0(timestamp, "_plot.png")
  } else {
    filename <- paste0(timestamp, "_", name, ".png")
  }

  filepath <- file.path(plots_dir, filename)
  ggsave(filepath, plot)
  cat("Plot saved to", filepath, "\\n")
  return(filepath)
}

view_img <- function(img_path) {
  if (file.exists(img_path)) {
    os <- Sys.info()["sysname"]
    if (os == "Darwin") {
      system(paste("open", img_path))
    } else if (os == "Windows") {
      system(paste("start", img_path), wait = FALSE)
    } else {
      system(paste("xdg-open", img_path))
    }
    cat("Opened image:", img_path, "\\n")
  } else {
    cat("Image file not found:", img_path, "\\n")
  }
  return(img_path)
}
""")
    elif lang == "javascript":
        main_path = src_dir / "main.js"
        if not main_path.exists():
            with open(main_path, "w") as f:
                f.write("""const fs = require('fs');
const path = require('path');
const os = require('os');
const child_process = require('child_process');

function savePlot(svgElement, name = null, plotsDir = null) {
  if (!plotsDir) {
    plotsDir = process.env.NPCSH_GUAC_PLOTS;
    if (!plotsDir) {
      plotsDir = path.join(os.homedir(), '.npcsh', 'guac', 'plots');
    }
  }

  if (!fs.existsSync(plotsDir)) {
    fs.mkdirSync(plotsDir, { recursive: true });
  }

  const timestamp = new Date().toISOString().replace(/[:.]/g, '');
  const filename = name ? `${timestamp}_${name}.svg` : `${timestamp}_plot.svg`;
  const filepath = path.join(plotsDir, filename);

  fs.writeFileSync(filepath, svgElement.outerHTML || svgElement);
  console.log(`Plot saved to ${filepath}`);
  return filepath;
}

function viewImage(imgPath) {
  if (fs.existsSync(imgPath)) {
    let command;

    switch (process.platform) {
      case 'darwin':
        command = 'open';
        break;
      case 'win32':
        command = 'start';
        break;
      default:
        command = 'xdg-open';
        break;
    }

    try {
      child_process.execSync(`${command} "${imgPath}"`);
      console.log(`Opened image: ${imgPath}`);
    } catch (error) {
      console.error(`Error opening image: ${error.message}`);
    }
  } else {
    console.error(`Image file not found: ${imgPath}`);
  }

  return imgPath;
}

module.exports = {
  savePlot,
  viewImage
};
""")

    if str(config_dir) not in sys.path:
        sys.path.insert(0, str(config_dir))

    setup_npc_team(npc_team_dir, lang)

    return {
        "language": lang,
        "src_dir": src_dir,
        "config_path": config_file,
        "plots_dir": plots_dir,
        "npc_team_dir": npc_team_dir,
        "config_dir": config_dir
    }

def setup_npc_team(npc_team_dir, lang):
    guac_npc = {
        "name": "guac",
        "primary_directive": f"You are guac, the main coordinator for data analysis in {lang}."
    }

    caug_npc = {
        "name": "caug",
        "primary_directive": f"You are caug, a specialist in big data statistical methods in {lang}."
    }

    parsely_npc = {
        "name": "parsely",
        "primary_directive": f"You are parsely, a specialist in mathematical methods in {lang}."
    }

    toon_npc = {
        "name": "toon",
        "primary_directive": f"You are toon, a specialist in brute force methods in {lang}."
    }

    for npc_data in [guac_npc, caug_npc, parsely_npc, toon_npc]:
        npc_file = npc_team_dir / f"{npc_data['name']}.npc"
        with open(npc_file, "w") as f:
            yaml.dump(npc_data, f, default_flow_style=False)

    team_ctx = {
        "team_name": "guac_team",
        "description": f"A team of NPCs specialized in {lang} analysis",
        "foreman": "guac",
        "model": os.environ.get("NPCSH_CHAT_MODEL", "llama3.2"),
        "provider": os.environ.get("NPCSH_CHAT_PROVIDER", "ollama")
    }

    with open(npc_team_dir / "team.ctx", "w") as f:
        yaml.dump(team_ctx, f, default_flow_style=False)

def is_code(text):
    code_patterns = [
        r'=', r'def\s+\w+\s*\(', r'class\s+\w+', r'import\s+\w+', r'from\s+\w+\s+import',
        r'\w+\.\w+\s*\(', r'if\s+.*:', r'for\s+.*:', r'while\s+.*:', r'\[.*\]', r'\{.*\}',
        r'\d+\s*[\+\-\*\/]\s*\d+', r'return\s+', r'lambda\s+\w+\s*:',
        r'^\s*\w+\s*\([^)]*\)\s*$'
    ]

    for pattern in code_patterns:
        if re.search(pattern, text):
            return True

    return False

def setup_guac_readline():
    try:
        readline.read_history_file(READLINE_HISTORY_FILE)
    except FileNotFoundError:
        pass
    except OSError as e:
        print(f"Warning: Could not read readline history file {READLINE_HISTORY_FILE}: {e}", file=sys.stderr)

    try:
        if sys.stdin.isatty():
            readline.parse_and_bind("set enable-bracketed-paste on")
    except Exception as e:
        print(f"Warning: Could not enable bracketed paste mode: {e}", file=sys.stderr)
        print("Multi-line pastes might not work correctly if bracketed paste is unsupported.", file=sys.stderr)

    readline.set_history_length(1000)
    atexit.register(save_guac_readline_history)

def save_guac_readline_history():
    try:
        readline.write_history_file(READLINE_HISTORY_FILE)
    except OSError as e:
        print(f"Warning: Could not write readline history file {READLINE_HISTORY_FILE}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Error saving readline history: {e}", file=sys.stderr)
class GuacInteractiveConsole(code.InteractiveConsole):
    def __init__(self, locals=None, command_history=None, npc=None, config_dir=None, src_dir=None, lang="python"):
        super().__init__(locals=locals)
        self.locals = locals if locals is not None else {}
        self.command_history = command_history
        self.npc = npc
        self.command_count = 0
        self.config_dir = config_dir
        self.src_dir = src_dir
        self.lang = lang
        self.filename = "<console>"

    def raw_input(self, prompt=""):
        try:
            current_prompt = get_guac_prompt(self.command_count)
            line = input(current_prompt)
            if line.strip():
                # Log command BEFORE push, as push might exec directly
                self.command_history.add_command(line, [], "", os.getcwd())
                self.command_count += 1
            return line
        except EOFError:
            self.write("\n")
            raise SystemExit()
        except KeyboardInterrupt:
             self.write("\nKeyboardInterrupt\n")
             return "\n"
    def push(self, line):
        log_line = line.strip()
        if log_line:
            self.command_history.add_command(log_line, [], "", os.getcwd())

        more_input_needed = False
        stripped_line = line.strip()

        if stripped_line in ('exit()', 'quit()', '/exit'): raise SystemExit
        elif stripped_line.startswith('ls'): os.system(stripped_line); self.resetbuffer(); return False
        elif stripped_line.startswith('cd'):
            try:
                parts = stripped_line.split(' ', 1)
                target_dir = parts[1] if len(parts) > 1 else str(Path.home())
                if (target_dir.startswith('"') and target_dir.endswith('"')) or \
                   (target_dir.startswith("'") and target_dir.endswith("'")): target_dir = target_dir[1:-1]
                os.chdir(target_dir); print(f"Changed directory to {os.getcwd()}")
            except Exception as e: print(f"Error changing directory: {e}")
            self.resetbuffer(); return False
        elif stripped_line.startswith('pwd'): print(f"Current directory: {os.getcwd()}"); self.resetbuffer(); return False
        elif stripped_line.startswith('pip'): os.system(stripped_line); self.resetbuffer(); return False
        elif stripped_line.startswith('run'):
            parts = stripped_line.split(' ', 1)
            if len(parts) < 2:
                 print("Usage: run <script.py>")
                 self.resetbuffer(); return False
            script_name = parts[1]
            if not script_name.endswith('.py'):
                print("Error: Only Python scripts (.py) are supported by 'run'.")
                self.resetbuffer(); return False
            try:
                with open(script_name) as f:
                    script_code = f.read()
                old_keys = set(self.locals.keys())
                exec(script_code, self.locals)
                new_keys = set(self.locals.keys()) - old_keys
                if new_keys:
                    print(f"Items added/modified from {script_name}:")
                    for key in new_keys:
                        value = self.locals[key]
                        if callable(value): print(f"  {key}: <function or class>")
                        else: print(f"  {key}: {repr(value)[:100]}{'...' if len(repr(value)) > 100 else ''}")
                else:
                    print(f"Script {script_name} executed.")
                if stripped_line: readline.add_history(stripped_line)
                self.command_count += 1
                self.write("\n")
            except FileNotFoundError:
                print(f"Error: Script '{script_name}' not found.")
            except Exception as e:
                print(f"Error running script '{script_name}':")
                self.showtraceback()
                self.write("\n")
            self.resetbuffer(); return False

        elif stripped_line == '/refresh': self._handle_refresh(); self.resetbuffer(); return False
        elif stripped_line == 'show':
            print("Current environment:")
            for name, value in self.locals.items():
                 if not name.startswith('__'):
                    try: print(f"{name}: {repr(value)[:150]}{'...' if len(repr(value)) > 150 else ''}")
                    except Exception: pass
            self.resetbuffer(); return False

        if not self.buffer and not is_code(line) and line.strip():
             if self.npc:
                 try:
                     prompt = f"""
The user has entered the following in the guac {self.lang} shell:
"{line.strip()}"
Generate {self.lang} code that addresses their query.
Return ONLY executable {self.lang} code without any additional text or markdown.
"""
                     response = get_llm_response(prompt, model = initial_state.chat_model, provider = initial_state.chat_provider, npc=self.npc)
                     generated_code = response.get("response", "").strip()
                     generated_code = re.sub(r'^```(?:python|r|javascript)?\s*|```$', '', generated_code, flags=re.MULTILINE).strip()

                     print(f"\n# Generated {self.lang} code:\n{generated_code}\n")

                     if generated_code and not generated_code.startswith('# Cannot generate'):
                         try:
                             exec(generated_code, self.locals)
                             print("\n# Generated code executed successfully")
                             self.command_count += 1
                             self.write("\n")
                         except Exception:
                             print(f"\n# Error executing generated {self.lang} code:")
                             self.showtraceback()
                             self.write("\n")
                     elif generated_code.startswith('# Cannot generate'):
                         print(generated_code[2:]); self.write("\n")
                     else:
                         print("# No code generated."); self.write("\n")
                 except Exception as llm_err:
                     print(f"\n# Error during natural language processing: {llm_err}")
                     self.showtraceback(); self.write("\n")
             else:
                 print("Natural language query detected but no NPC available."); self.write("\n")

             self.resetbuffer(); return False

        try:
            more_input_needed = super().push(line)
        except KeyboardInterrupt:
            self.write("\nKeyboardInterrupt\n"); self.resetbuffer(); more_input_needed = False
        except Exception:
             self.showtraceback()
             self.resetbuffer()
             more_input_needed = False
             self.write("\n")

        return more_input_needed
    def _handle_refresh(self):
        history_entries = self.command_history.get_all()
        if not history_entries: return

        commands = []
        for entry in history_entries:
            try:
                 cmd_text = entry[2]
                 if isinstance(cmd_text, str) and cmd_text.strip() and \
                    not cmd_text.startswith('/') and \
                    cmd_text.lower() not in ['exit()', 'quit()']:
                     commands.append(cmd_text)
            except (IndexError, TypeError): continue
        if not commands: return

        prompt = f"Analyze these {self.lang} commands:\n\n```{self.lang}\n"
        prompt += "\n".join(commands)
        prompt += f"\n```\n\nSuggest 1-3 useful {self.lang} functions..."

        try:
            response = get_llm_response(prompt, npc=self.npc)
            suggested_functions = response.get("response", "")
            if not suggested_functions or len(suggested_functions) < 10: return

            print("\n=== Suggested Functions ===\n")
            print(suggested_functions)
            print("\n=========================\n")

            user_choice = input("Add to environment? (y/n): ").strip().lower()
            if user_choice != 'y': return

            if self.lang == "python": file_path = self.src_dir / "main.py"
            elif self.lang == "r": file_path = self.src_dir / "main.R"
            elif self.lang == "javascript": file_path = self.src_dir / "main.js"
            else: return

            code_sections = []
            if self.lang == "python":
                code_sections = re.findall(r'```python\s+(.*?)\s+```', suggested_functions, re.DOTALL)
                if not code_sections: code_sections = re.findall(r'(def\s+.*?(?=\n\s*def|\Z))', suggested_functions, re.DOTALL)
            elif self.lang == "r":
                code_sections = re.findall(r'```r\s+(.*?)\s+```', suggested_functions, re.DOTALL)
                if not code_sections: code_sections = re.findall(r'(\w+\s*<-\s*function.*?})', suggested_functions, re.DOTALL)
            elif self.lang == "javascript":
                code_sections = re.findall(r'```javascript\s+(.*?)\s+```', suggested_functions, re.DOTALL)
                if not code_sections: code_sections = re.findall(r'(function\s+\w+\s*\(.*?\)\s*{.*?}\s*;?)', suggested_functions, re.DOTALL)

            functions_to_append = "\n".join(code.strip() for code in code_sections) if code_sections else suggested_functions.strip()

            if functions_to_append:
                 with open(file_path, "a") as f:
                     f.write("\n\n")
                     f.write(functions_to_append)
                     f.write("\n")
            else: pass

        except Exception as e:
            print(f"Error in refresh: {e}")


    def write(self, data):
        sys.stderr.write(data)
        sys.stderr.flush()
def print_guac_bowl():
    guac = [
        "  🟢🟢🟢🟢🟢 ",
        "🟢          🟢                 ",
        "🟢  ",
        "🟢      ",                  
        "🟢      ",                          
        "🟢      🟢🟢🟢   🟢    🟢   🟢🟢🟢    🟢🟢🟢",
        "🟢           🟢  🟢    🟢    ⚫⚫🟢  🟢        ",
        "🟢           🟢  🟢    🟢  ⚫🥑🧅⚫  🟢     ",
        "🟢           🟢  🟢    🟢  ⚫🥑🍅⚫  🟢      ",
        " 🟢🟢🟢🟢🟢🟢    🟢🟢🟢🟢    ⚫⚫🟢   🟢🟢🟢 ",
        "                                            "
    ]
    
    for line in guac:
        print(line)




def get_guac_prompt(command_count):
    stages = [
        "\U0001F951",
        "\U0001F951 🔪",
        "\U0001F951 🥣",
        "\U0001F951 🥣🧂",
        "\U0001F958 TIME TO REFRESH"
    ]

    stage_index = min(command_count // int((GUAC_REFRESH_PERIOD/5)), len(stages) - 1)

    return stages[stage_index] + " "

def enter_guac_mode(npc=None, team=None, config_dir=None, plots_dir=None, npc_team_dir=None,
                    refresh_period=None, lang=None):
    if refresh_period is not None:
        try:
            GUAC_REFRESH_PERIOD = int(refresh_period)
        except ValueError:
            pass

    setup_result = setup_guac_mode(
        config_dir=config_dir,
        plots_dir=plots_dir,
        npc_team_dir=npc_team_dir
    )

    lang = lang or setup_result["language"]
    config_dir = setup_result["config_dir"]
    src_dir = setup_result["src_dir"]
    npc_team_dir = setup_result["npc_team_dir"]

    parent_dir = str(config_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    command_history = CommandHistory()

    if npc is None and team is None:
        try:
            team = Team(team_path=str(npc_team_dir), db_conn=None)
            npc = team.npcs.get("guac")
            if not npc and team.npcs:
                 npc = next(iter(team.npcs.values()))
        except Exception as e:
            pass
    initial_state.npc = npc 
    initial_state.team = team
    try:
        setup_guac_readline()
    except:
        print('couldnt set up readline.')
    print_guac_bowl()

    if lang == "python":
        namespace = {}
        try:
            guac_package_parent = config_dir.parent
            if str(guac_package_parent) not in sys.path:
                 sys.path.insert(0, str(guac_package_parent))

            import guac.src.main as guac_main
            user_modules = {name: getattr(guac_main, name) for name in dir(guac_main)
                             if not name.startswith('__')}
            namespace.update(user_modules)
        except Exception as e:
             pass

        if npc:
            namespace['npc'] = npc
        if team:
            namespace['team'] = team

        console = GuacInteractiveConsole(
            locals=namespace,
            command_history=command_history,
            npc=npc,
            config_dir=config_dir,
            src_dir=src_dir,
            lang=lang
        )

        console.interact(banner="")

    elif lang == "r":
        try:
            import rpy2.robjects as ro
            ro.r(f'source("{src_dir}/main.R")')

            command_count = 0
            while True:
                try:
                    command_count += 1
                    r_cmd = input(get_guac_prompt(command_count))

                    if r_cmd.strip().lower() in ('q()', 'quit()', 'exit()', '/exit'):
                        break

                    if r_cmd.strip() == '/refresh':
                        console = GuacInteractiveConsole(
                            command_history=command_history, npc=npc,
                            config_dir=config_dir, src_dir=src_dir, lang="r"
                        )
                        console._handle_refresh()
                        continue

                    command_history.add_command(r_cmd, [], "", os.getcwd())

                    is_r_code = any(x in r_cmd for x in ['<-', '(', ')', '{', '}', '$', 'function'])

                    if is_r_code:
                        result = ro.r(r_cmd)
                        print(result)
                    elif npc:
                        prompt = f"""Generate R code for: "{r_cmd}". Return ONLY executable R code."""
                        response = get_llm_response(prompt, npc=npc)
                        generated_code = response.get("response", "").strip()
                        generated_code = re.sub(r'^```r?\s*|```$', '', generated_code, flags=re.MULTILINE).strip()

                        if generated_code:
                             try:
                                 result = ro.r(generated_code)
                                 print(result)
                             except Exception as e_exec:
                                 print(f"Error executing R code: {e_exec}")
                        else:
                             pass
                    else:
                         pass

                except Exception as e_loop:
                    print(f"Error: {str(e_loop)}")

        except ImportError:
            import subprocess
            try:
                 subprocess.run(["R", "-q", "--vanilla"], check=True)
            except FileNotFoundError:
                 print("R executable not found.")
            except Exception as e_sub:
                 print(f"Error running R: {e_sub}")

    elif lang == "javascript":
        import subprocess
        try:
             subprocess.run(["node", "-i"], check=True)
        except FileNotFoundError:
             print("Node.js executable not found.")
        except Exception as e_sub:
             print(f"Error running Node.js: {e_sub}")


def main():
    parser = argparse.ArgumentParser(description="Enter guac mode")
    parser.add_argument("--config_dir", type=str, help="Configuration directory")
    parser.add_argument("--plots_dir", type=str, help="Plots directory")
    parser.add_argument("--npc_team_dir", type=str, default=os.path.expanduser('~/.npcsh/guac/npc_team/'), help="NPC team directory")
    parser.add_argument("--refresh_period", type=int, help="Number of commands before suggesting refresh")
    parser.add_argument("--lang", type=str, help="Preferred language (python, r, javascript)")

    args = parser.parse_args()

    enter_guac_mode(
        npc=None,
        team=None,
        config_dir=args.config_dir,
        plots_dir=args.plots_dir,
        npc_team_dir=args.npc_team_dir,
        refresh_period=args.refresh_period,
        lang=args.lang
    )

if __name__ == "__main__":
    main()