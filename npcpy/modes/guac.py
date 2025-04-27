# add the refresh period, default = 25 commands so each stage is ~ 5 commands
import re 
import os
import sys
import code
import re
import yaml
from pathlib import Path
from npcpy.memory.command_history import CommandHistory, start_new_conversation
from npcpy.npc_compiler import Team, NPC
from npcpy.llm_funcs import get_llm_response
import json

GUAC_REFRESH_PERIOD = 100  # Default refresh period - each stage is 5 commands


def setup_guac_mode(config_dir=None, plots_dir=None, npc_team_dir=None):
    """
    Set up guac mode environment with NPC team.
    
    Args:
        config_dir: Directory to store configuration (default: ~/.npcsh/guac)
        plots_dir: Directory to store plots (default: ~/.npcsh/guac/plots)
        npc_team_dir: Directory for NPC team (default: ~/.npcsh/guac/npc_team)
    
    Returns:
        Dict with setup information
    """
    
    # Set default directories if not provided
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
    
    # Create directories
    src_dir = config_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    npc_team_dir.mkdir(parents=True, exist_ok=True)
    
    # Create an empty __init__.py in config_dir to make it a package
    init_path = config_dir / "__init__.py"
    if not init_path.exists():
        with open(init_path, "w") as f:
            f.write("# guac package\n")
    
    # Get or prompt for preferred language
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
        
        # Save config
        config = {
            "preferred_language": lang,
            "plots_directory": str(plots_dir),
            "npc_team_directory": str(npc_team_dir)
        }
        with open(config_file, "w") as f:
            json.dump(config, f)
    
    # Set up environment variable
    os.environ["NPCSH_GUAC_LANG"] = lang
    os.environ["NPCSH_GUAC_PLOTS"] = str(plots_dir)
    os.environ["NPCSH_GUAC_TEAM"] = str(npc_team_dir)
    
    # Create source files structure
    if lang == "python":
        # Create __init__.py in src directory to make it a package
        src_init_path = src_dir / "__init__.py"
        if not src_init_path.exists():
            with open(src_init_path, "w") as f:
                f.write("# guac src package\n")
                f.write("from .main import *\n")
        
        # Create simple main.py with basic functionality
        main_path = src_dir / "main.py"
        if not main_path.exists():
            with open(main_path, "w") as f:
                f.write("""# guac main module
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from pathlib import Path

def save_plot(name=None, plots_dir=None):
    \"\"\"
    Save current plot to file with timestamp and return path
    \"\"\"
    # Get plots directory from arg, env, or default
    if plots_dir is None:
        plots_dir = os.environ.get("NPCSH_GUAC_PLOTS")
        if plots_dir is None:
            plots_dir = Path.home() / ".npcsh" / "guac" / "plots"
        else:
            plots_dir = Path(plots_dir)
    
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename with timestamp
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
    \"\"\"
    Display image and return path
    \"\"\"
    try:
        from PIL import Image
        img = Image.open(img_path)
        img.show()
    except ImportError:
        print("PIL not available, can't display image")
    
    return img_path
""")
    elif lang == "r":
        # Create main.R file
        main_path = src_dir / "main.R"
        if not main_path.exists():
            with open(main_path, "w") as f:
                f.write("""# guac main R functions
library(dplyr)
library(ggplot2)

# Simple function to save plots
save_plot <- function(plot = last_plot(), name = NULL, plots_dir = NULL) {
  # Determine plots directory
  if (is.null(plots_dir)) {
    plots_dir <- Sys.getenv("NPCSH_GUAC_PLOTS", unset = NA)
    if (is.na(plots_dir)) {
      plots_dir <- file.path(Sys.getenv("HOME"), ".npcsh", "guac", "plots")
    }
  }
  
  # Create directory if needed
  dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Create filename with timestamp
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

# Function to view image
view_img <- function(img_path) {
  if (file.exists(img_path)) {
    # Try different methods depending on platform
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
        # Create main.js file
        main_path = src_dir / "main.js"
        if not main_path.exists():
            with open(main_path, "w") as f:
                f.write("""// guac main JavaScript functions
const fs = require('fs');
const path = require('path');
const os = require('os');
const child_process = require('child_process');

// Function to save chart/plot
function savePlot(svgElement, name = null, plotsDir = null) {
  // Determine plots directory
  if (!plotsDir) {
    plotsDir = process.env.NPCSH_GUAC_PLOTS;
    if (!plotsDir) {
      plotsDir = path.join(os.homedir(), '.npcsh', 'guac', 'plots');
    }
  }
  
  // Create directory if it doesn't exist
  if (!fs.existsSync(plotsDir)) {
    fs.mkdirSync(plotsDir, { recursive: true });
  }
  
  // Create filename with timestamp
  const timestamp = new Date().toISOString().replace(/[:.]/g, '');
  const filename = name ? `${timestamp}_${name}.svg` : `${timestamp}_plot.svg`;
  const filepath = path.join(plotsDir, filename);
  
  // Write SVG to file
  fs.writeFileSync(filepath, svgElement.outerHTML || svgElement);
  console.log(`Plot saved to ${filepath}`);
  return filepath;
}

// Function to view an image
function viewImage(imgPath) {
  if (fs.existsSync(imgPath)) {
    let command;
    
    // Determine platform-specific open command
    switch (process.platform) {
      case 'darwin': // macOS
        command = 'open';
        break;
      case 'win32': // Windows
        command = 'start';
        break;
      default: // Linux and others
        command = 'xdg-open';
        break;
    }
    
    // Open the image
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
    
    # Add to Python path if it's not already there
    if str(config_dir) not in sys.path:
        sys.path.insert(0, str(config_dir))
    
    # Set up NPC team structure (without tools)
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
    """
    Set up the NPC team structure with specialized NPCs for data analysis (no tools)
    Using YAML format instead of JSON
    """
    
    # Create the guac NPC team with specialized roles
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
    
    # Save NPC files as YAML
    for npc_data in [guac_npc, caug_npc, parsely_npc, toon_npc]:
        npc_file = npc_team_dir / f"{npc_data['name']}.npc"
        with open(npc_file, "w") as f:
            yaml.dump(npc_data, f, default_flow_style=False)
    
    # Create team context file (as YAML)
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
    """
    Determines if input is code or natural language
    """
    # Python syntax patterns
    code_patterns = [
        r'=', r'def\s+\w+\s*\(', r'class\s+\w+', r'import\s+\w+', r'from\s+\w+\s+import',
        r'\w+\.\w+\s*\(', r'if\s+.*:', r'for\s+.*:', r'while\s+.*:', r'\[.*\]', r'\{.*\}',
        r'\d+\s*[\+\-\*\/]\s*\d+', r'return\s+', r'lambda\s+\w+\s*:',
        r'^\s*\w+\s*\([^)]*\)\s*$'  # Add this pattern to detect standalone function calls
    ]
    
    # Check if any code pattern matches
    for pattern in code_patterns:
        if re.search(pattern, text):
            return True
        
    return False
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
        self.messages = []
        self.memory = []
    def raw_input(self, prompt=""):            
        return input(get_guac_prompt(self.command_count))
            
    def push(self, line):
        self.command_count += 1
        
        # Handle special commands
        if line.strip() in ('exit()', 'quit()', '/exit'):
            return False
        elif line.strip().startswith('ls'):
            os.system('ls')
            return False
        elif line.strip().startswith('cd'):
            try:
                os.chdir(line.strip().split(' ')[1])
                print(f"Changed directory to {os.getcwd()}")
            except Exception as e:
                print(f"Error changing directory: {e}")
            return False
        elif line.strip().startswith('pwd'):
            print(f"Current directory: {os.getcwd()}")
            return False
        elif line.strip().startswith('pip'):
            os.system(line.strip())
            return False
        elif line.strip().startswith('run'):
            # Extract the script name from the command
            script_name = line.strip().split(' ')[1]
            if not script_name.endswith('.py'):
                print("Error: Only Python scripts are supported.")
                return False
            try:
                # Execute the script directly in the current namespace
                with open(script_name) as f:
                    code = f.read()
                    
                # Save current keys to identify what's new
                old_keys = set(self.locals.keys())
                
                # Execute in the current environment (self.locals)
                exec(code, self.locals)
                
                # Identify new items added to namespace
                new_keys = set(self.locals.keys()) - old_keys
                
                # Display what was added
                if new_keys:
                    print(f"Items added from {script_name}:")
                    for key in new_keys:
                        value = self.locals[key]
                        # For functions/classes, show their name rather than full representation
                        if callable(value):
                            print(f"{key}: <function or class>")
                        else:
                            print(f"{key}: {value}")
                else:
                    print(f"No new variables or functions were added from {script_name}")
                    
            except Exception as e:
                print(f"Error running script: {e}")
            return False


        elif line.strip() == '/refresh':
            self._handle_refresh()
            return False
        elif line.strip() == 'show':
            print("Current environment:")
            for name, value in self.locals.items():
                print(f"{name}: {value}")
            return False
        
        # Add to command history
        self.command_history.add_command(
            line, [], "", os.getcwd()
        )
        
        # Always try to execute as code first
        try:
            # For single variable names, handle specially to print them
            if re.match(r'^\s*\w+\s*$', line) and line.strip() in self.locals:
                print(f"{line.strip()} = {self.locals[line.strip()]}")
                return False
            
            if is_code(line):
                return super().push(line)
            else:

                # If execution fails and it doesn't look like code, treat as natural language
                if self.npc:
                    try:
                        # Process natural language using NPC
                        prompt = f"""
                        The user has entered the following in the guac shell:
                        "{line}"
                        
                        Generate Python code that addresses their query. 
                        Return ONLY executable Python code without any additional text or markdown.
                        """
                        
                        response = get_llm_response(prompt, npc=self.npc)
                        
                        self.messages = response.get("messages", [])
                        generated_code = response.get("response", "").strip()
                        
                        # Clean the code (remove markdown if present)
                        generated_code = re.sub(r'```python|```', '', generated_code).strip()
                                                
                        # Display the code and execute it
                        print(f"\n# Generated code:")
                        print(f"{generated_code}\n")

                        # Save existing variables before execution
                        old_keys = set(self.locals.keys())

                        # Execute the generated code
                        try:
                            exec(generated_code, self.locals)
                            
                            # Show which new variables were created, filtering out special variables
                            new_keys = set(k for k in self.locals.keys() if not k.startswith('__')) - \
                                    set(k for k in old_keys if not k.startswith('__'))
                            
                            if new_keys:
                                print("\n# New variables created:")
                                for key in new_keys:
                                    value = self.locals[key]
                                    if callable(value):
                                        print(f"{key}: <function or class>")
                                    else:
                                        print(f"{key}: {value}")
                                        
                            print("\n# Code executed successfully")
                        except Exception as e:
                            print(f"\n# Error executing code: {e}")                        

                    except Exception as e:
                        print(f"Error processing natural language: {str(e)}")
                        return False
                else:
                    print("Natural language query detected but no NPC available to process it.")
                    return False
        except Exception as e:
            # If execution fails, print the error
            print(f"Error executing code: {e}")
            return False
            
    def _handle_refresh(self):
        """
        Handles the /refresh command by analyzing command history and suggesting improvements
        """
        print("\nRefreshing guac environment and analyzing command history...\n")
        
        # Get command history
        history_entries = self.command_history.get_all()
        
        if not history_entries:
            print("No command history available to analyze.")
            return
        
        # Extract only commands (no output or system commands)
        commands = []
        for entry in history_entries:
            cmd = entry[2]  # Command text is at index 2
            # Skip system commands and empty commands
            if cmd and not cmd.startswith('/') and cmd not in ['exit()', 'quit()']:
                commands.append(cmd)
        
        if not commands:
            print("No substantial commands found in history to analyze.")
            return
        
        # Create prompt for the NPC to analyze and suggest improvements
        if self.lang == "python":
            prompt = f"""
            Analyze these Python commands from a recent data analysis session:
            
            ```python
            {chr(10).join(commands)}
            ```
            
            Based on the user's workflow patterns, suggest 1-3 functions, utilities, or automations 
            that would be useful to add to their guac environment. For each suggestion:
            
            1. Provide a clear name and purpose
            2. Include fully implemented Python code
            3. Explain how it builds on patterns observed in their workflow
            
            Format your response as complete, well-documented Python function(s) that could be 
            added to the guac.src.main module.
            """
        elif self.lang == "r":
            prompt = f"""
            Analyze these R commands from a recent data analysis session:
            
            ```r
            {chr(10).join(commands)}
            ```
            
            Based on the user's workflow patterns, suggest 1-3 functions, utilities, or automations 
            that would be useful to add to their guac environment. For each suggestion:
            
            1. Provide a clear name and purpose
            2. Include fully implemented R code
            3. Explain how it builds on patterns observed in their workflow
            
            Format your response as complete, well-documented R function(s) that could be 
            added to the guac src/main.R module.
            """
        elif self.lang == "javascript":
            prompt = f"""
            Analyze these JavaScript commands from a recent data analysis session:
            
            ```javascript
            {chr(10).join(commands)}
            ```
            
            Based on the user's workflow patterns, suggest 1-3 functions, utilities, or automations 
            that would be useful to add to their guac environment. For each suggestion:
            
            1. Provide a clear name and purpose
            2. Include fully implemented JavaScript code
            3. Explain how it builds on patterns observed in their workflow
            
            Format your response as complete, well-documented JavaScript function(s) that could be 
            added to the guac src/main.js module.
            """
        else:
            print(f"Language {self.lang} not supported for refresh analysis.")
            return
            
        # Get suggestions from NPC
        try:
            response = get_llm_response(prompt, npc=self.npc)
            suggested_functions = response.get("response", "")
            
            # Check if we got a reasonable response
            if not suggested_functions or len(suggested_functions) < 10:
                print("Could not generate meaningful suggestions based on command history.")
                return
                
            # Display suggestions to the user
            print("\n=== Suggested Functions Based on Your Workflow ===\n")
            print(suggested_functions)
            print("\n=== End of Suggestions ===\n")
            
            # Ask if user wants to save these functions
            user_choice = input("Would you like to add these functions to your guac environment? (y/n): ").strip().lower()
            
            if user_choice == 'y':
                # Append to the appropriate file
                if self.lang == "python":
                    file_path = self.src_dir / "main.py"
                elif self.lang == "r":
                    file_path = self.src_dir / "main.R"
                elif self.lang == "javascript":  
                    file_path = self.src_dir / "main.js"
                
                # Extract just the code from the suggestions
                if self.lang == "python":
                    # Simple python code extraction - this could be improved
                    code_sections = re.findall(r'```python\s+(.*?)\s+```', suggested_functions, re.DOTALL)
                    if not code_sections:
                        code_sections = re.findall(r'def\s+\w+\s*\([^)]*\)[^:]*:(.*?)(?=\n\s*def|\Z)', suggested_functions, re.DOTALL)
                elif self.lang == "r":
                    code_sections = re.findall(r'```r\s+(.*?)\s+```', suggested_functions, re.DOTALL)
                    if not code_sections:
                        code_sections = re.findall(r'(\w+\s*<-\s*function.*?})', suggested_functions, re.DOTALL)
                elif self.lang == "javascript":
                    code_sections = re.findall(r'```javascript\s+(.*?)\s+```', suggested_functions, re.DOTALL)
                    if not code_sections:
                        code_sections = re.findall(r'(function\s+\w+\s*\([^)]*\)\s*{.*?})', suggested_functions, re.DOTALL)
                
                # Format the code for appending
                if code_sections:
                    with open(file_path, "a") as f:
                        f.write("\n\n# Auto-generated functions from guac refresh\n")
                        for code in code_sections:
                            f.write(f"\n{code.strip()}\n")
                    
                    print(f"\nFunctions added to {file_path}")
                    print("Restart guac or import the module again to use the new functions.")
                else:
                    # If we couldn't extract code properly, just append the whole thing
                    with open(file_path, "a") as f:
                        f.write("\n\n# Auto-generated functions from guac refresh\n")
                        f.write(f"\n{suggested_functions.strip()}\n")
                    
                    print(f"\nSuggestions added to {file_path}")
                    print("You may need to edit the file to extract the proper code.")
            else:
                print("No functions were added to your environment.")
                
        except Exception as e:
            print(f"Error analyzing command history: {str(e)}")

def get_guac_prompt(command_count):
    """
    Returns an evolving avocado prompt that gradually turns into guacamole
    """
    # Define the avocado-to-guac transformation stages
    stages = [
        "\U0001F951",            # Fresh avocado
        "\U0001F951 🔪",         # Avocado being cut
        "\U0001F951 🥣",         # Avocado in bowl
        "\U0001F951 🥣 🧂",      # Avocado being seasoned
        "\U0001F958"             # Guacamole
    ]
    
    # Calculate which stage we're at
    stage_index = min(command_count // int((GUAC_REFRESH_PERIOD/5)), len(stages) - 1)
    
    # If we've reached the final stage, suggest a refresh
    if stage_index == len(stages) - 1 and command_count % GUAC_REFRESH_PERIOD == 0:
        return stages[stage_index] + " (Type '/refresh' to analyze your workflow) "
    
    return stages[stage_index] + " "





def enter_guac_mode(npc=None, team=None, config_dir=None, plots_dir=None, npc_team_dir=None, 
                    refresh_period=None, lang=None):
    """
    Enter guac mode - interactive shell with NPC team support that handles natural language
    
    Args:
        npc: Optional NPC object for LLM interactions
        team: Optional Team object containing NPCs
        config_dir: Directory for configuration
        plots_dir: Directory for plots
        npc_team_dir: Directory for NPC team
        refresh_period: Number of commands before suggesting refresh
        lang: Preferred language
    """
    # Update refresh period if provided
    global GUAC_REFRESH_PERIOD
    if refresh_period is not None:
        try:
            GUAC_REFRESH_PERIOD = int(refresh_period)
        except ValueError:
            print(f"Warning: Invalid refresh period '{refresh_period}', using default {GUAC_REFRESH_PERIOD}")

    # Set up guac environment
    setup_result = setup_guac_mode(
        config_dir=config_dir,
        plots_dir=plots_dir,
        npc_team_dir=npc_team_dir
    )
    
    lang = lang or setup_result["language"]
    config_dir = setup_result["config_dir"]
    src_dir = setup_result["src_dir"]
    npc_team_dir = setup_result["npc_team_dir"]
    
    # Make sure package directories are in path
    parent_dir = str(config_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Initialize command history
    command_history = CommandHistory()
    conversation_id = start_new_conversation()
    
    # Load NPC team if none provided
    if npc is None and team is None:
        try:
            team = Team(team_path=str(npc_team_dir), db_conn=None)
            npc = team.npcs.get("guac")  # Use guac as the main NPC
            print(f"Loaded NPC team from {npc_team_dir}")
        except Exception as e:
            print(f"Warning: Could not load NPC team: {e}")
    
    # Welcome message
    print(f"Entering guac mode with {lang}")
    print("Type code directly or natural language queries")
    print("Type 'exit()' to exit, '/refresh' to analyze your workflow and suggest improvements")
    
    # Start simple REPL based on language
    if lang == "python":
        # Try to import the guac package
        try:
            # First make sure we can import from guac.src
            sys.path.append(str(config_dir))
            
            # Try different import approaches
            try:
                import guac.src.main
                modules_from_guac = {name: getattr(guac.src.main, name) for name in dir(guac.src.main) 
                                    if not name.startswith('__')}
                print("Successfully imported from guac.src.main")
            except ImportError:
                try:
                    from guac.src import main
                    modules_from_guac = {name: getattr(main, name) for name in dir(main) 
                                        if not name.startswith('__')}
                    print("Successfully imported from guac.src")
                except ImportError:
                    try:
                        sys.path.append(str(src_dir))
                        import main
                        modules_from_guac = {name: getattr(main, name) for name in dir(main) 
                                            if not name.startswith('__')}
                        print("Successfully imported from main")
                    except ImportError:
                        print(f"Warning: Could not import guac package from any location")
                        print(f"sys.path: {sys.path}")
                        modules_from_guac = {}
        except Exception as e:
            print(f"Warning: Import error: {e}")
            modules_from_guac = {}
        
        # Create base namespace with modules from guac
        namespace = modules_from_guac        
        # Add NPC and team to the namespace
        namespace['npc'] = npc
        if team is not None:
            namespace['team'] = team
            
        # Add necessary imports for common data science tasks
        if 'pd' not in namespace:
            import pandas as pd
            namespace['pd'] = pd
        if 'np' not in namespace:
            import numpy as np
            namespace['np'] = np
        if 'plt' not in namespace:
            import matplotlib.pyplot as plt
            namespace['plt'] = plt
        
        # Start the Python REPL
        GuacInteractiveConsole(
            locals=namespace, 
            command_history=command_history, 
            npc=npc,
            config_dir=config_dir,
            src_dir=src_dir,
            lang=lang
        ).interact(
            banner=f"Python {sys.version} with guac mode\nType code or natural language queries directly.\nType 'exit()' to exit, '/refresh' to get suggestions."
        )
    
    elif lang == "r":
        # Simple R console with rpy2 if available
        try:
            import rpy2.robjects as ro
            
            # Source the main R file
            ro.r(f'source("{src_dir}/main.R")')
            
            # Start simple R REPL
            command_count = 0
            while True:
                try:
                    command_count += 1
                    r_cmd = input(get_guac_prompt(command_count))
                    
                    # Exit check
                    if r_cmd.strip() in ('q()', 'quit()', 'exit()', '/exit'):
                        break
                    
                    # Refresh command
                    if r_cmd.strip() == '/refresh':
                        # Create a temporary GuacInteractiveConsole to handle refresh
                        console = GuacInteractiveConsole(
                            command_history=command_history,
                            npc=npc,
                            config_dir=config_dir,
                            src_dir=src_dir,
                            lang="r"
                        )
                        console._handle_refresh()
                        continue
                    
                    # Add to command history
                    command_history.add_command(
                        r_cmd, [], "", os.getcwd()
                    )
                    
                    # Simple code detection for R
                    is_r_code = any(x in r_cmd for x in ['<-', '(', ')', '{', '}', '$', 'function'])
                    
                    if is_r_code:
                        # Execute as R code
                        result = ro.r(r_cmd)
                        print(result)
                    elif npc:
                        # Process natural language
                        prompt = f"""
                        The user has entered the following in the guac R shell:
                        "{r_cmd}"
                        
                        Generate R code that addresses their query.
                        Return ONLY executable R code without any additional text or markdown.
                        """
                        
                        response = get_llm_response(prompt, npc=npc)
                        generated_code = response.get("response", "").strip()
                        
                        # Clean code
                        generated_code = re.sub(r'```r|```', '', generated_code).strip()
                        
                        # Show and execute
                        print(f"\n# Generated R code:")
                              
                        print(f"{generated_code}\n")
                        
                        try:
                            result = ro.r(generated_code)
                            print(result)
                            print("\n# Code executed successfully")
                        except Exception as e:
                            print(f"\n# Error executing R code: {e}")
                    else:
                        print("Natural language query detected but no NPC available to process it.")
                    
                except Exception as e:
                    print(f"Error: {str(e)}")
                    
        except ImportError:
            print("rpy2 not available, using system R")
            import subprocess
            subprocess.run(["R", "-q", "--vanilla", "-f", f"{src_dir}/main.R"])
    
    elif lang == "javascript":
        # For JavaScript, we need a more custom approach since Node.js REPL is less customizable
        import subprocess
        
        # Create a custom script that handles both code and natural language
        temp_js_file = Path.home() / ".npcsh" / "guac" / "guac_repl.js"
        with open(temp_js_file, "w") as f:
            f.write(f"""
            const guac = require('{src_dir}/main.js');
            Object.assign(global, guac);
            
            // Setup to handle natural language if Node.js REPL allows
            const readline = require('readline');
            const fs = require('fs');
            
            // Create prompt stages for avocado-to-guac transformation
            const promptStages = [
                "🥑 ",
                "🥑 🔪 ",
                "🥑 🥣 ",
                "🥑 🥣 🧂 ",
                "🥬 "
            ];
            
            let commandCount = 0;
            
            // Store command history
            const commandHistory = [];
            
            // Custom prompt function 
            function getGuacPrompt() {{
                const refreshPeriod = {GUAC_REFRESH_PERIOD};
                const stageIndex = Math.min(Math.floor(commandCount / (refreshPeriod/5)), promptStages.length - 1);
                const stage = promptStages[stageIndex];
                
                // If we've reached the final stage, suggest a refresh
                if (stageIndex === promptStages.length - 1 && commandCount % refreshPeriod === 0) {{
                    return stage + " (Type '/refresh' to analyze your workflow) ";
                }}
                
                return stage;
            }}
            
            // Handle special commands
            const originalEval = global.eval;
            global.eval = function(code) {{
                // Track commands
                commandCount++;
                commandHistory.push(code);
                
                // Handle special commands
                if (code.trim() === '/refresh') {{
                    console.log('\\nAnalyzing your JavaScript workflow...\\n');
                    
                    // Save command history to a temp file
                    const historyFile = '{temp_js_file}.history.js';
                    fs.writeFileSync(historyFile, commandHistory.join('\\n'));
                    
                    // Execute a process to analyze and suggest improvements
                    // This is a placeholder - in a real implementation, you would call an API
                    console.log('Workflow analysis would happen here...');
                    console.log('For now, this feature is not fully implemented in JavaScript mode.');
                    return;
                }}
                
                // Execute normally
                return originalEval(code);
            }};
            
            // Set prompt when REPL is available
            if (global._replServer) {{
                const origPrompt = global._replServer.prompt;
                
                // Override prompt getter
                Object.defineProperty(global._replServer, 'prompt', {{
                    get: function() {{ 
                        return getGuacPrompt();
                    }},
                    set: function(val) {{ origPrompt = val; }}
                }});
            }}
            """)
        
        # Run Node with our custom script
        subprocess.run(["node", "-i", "-e", f"require('{temp_js_file}')"])

def main():
    """
    Main function to set up and enter guac mode.
    
    This function can be called directly or used as a script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Enter guac mode")
    parser.add_argument("--config_dir", type=str, help="Configuration directory")
    parser.add_argument("--plots_dir", type=str, help="Plots directory")
    parser.add_argument("--npc_team_dir", type=str, default=os.path.expanduser('~/.npcsh/guac/npc_team/'), help="NPC team directory")
    parser.add_argument("--refresh_period", type=int, help="Number of commands before suggesting refresh")
    parser.add_argument("--lang", type=str, help="Preferred language (python, r, javascript)")
    
    args = parser.parse_args()
    
    # Initialize guac environment
    setup_guac_mode(
        config_dir=args.config_dir, 
        plots_dir=args.plots_dir,
        npc_team_dir=args.npc_team_dir
    )
    
    # Create or load NPC and team
    try:
        npc = NPC(file=os.path.join(args.npc_team_dir, "guac.npc"))
        team = Team(team_path=args.npc_team_dir, db_conn=None)
    except Exception as e:
        print(f"Warning: Error loading NPC/team: {e}")
        npc = None
        team = None
    
    # Enter guac mode
    enter_guac_mode(
        npc=npc,
        team=team,
        config_dir=args.config_dir, 
        plots_dir=args.plots_dir,
        npc_team_dir=args.npc_team_dir, 
        refresh_period=args.refresh_period,
        lang=args.lang
    )
    
if __name__ == "__main__":
    main()