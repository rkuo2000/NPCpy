# add the refresh period, default = 25 commands so each stage is ~ 5 commands



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
    import os
    import json
    import sys
    from pathlib import Path
    
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
    """
    import os
    import json
    from pathlib import Path
    
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
    
    # Save NPC files
    for npc_data in [guac_npc, caug_npc, parsely_npc, toon_npc]:
        npc_file = npc_team_dir / f"{npc_data['name']}.npc"
        with open(npc_file, "w") as f:
            json.dump(npc_data, f, indent=2)
    
    # Create team context file
    team_ctx = {
        "team_name": "guac_team",
        "description": f"A team of NPCs specialized in {lang} analysis",
        "foreman": "guac",
        "model": os.environ.get("NPCSH_CHAT_MODEL", "llama3.2"),
        "provider": os.environ.get("NPCSH_CHAT_PROVIDER", "ollama")
    }
    
    with open(npc_team_dir / "team.ctx", "w") as f:
        json.dump(team_ctx, f, indent=2)

def enter_guac_mode(npc=None, team=None):
    """
    Enter guac mode - interactive shell with NPC team support that handles natural language
    
    Args:
        npc: Optional NPC object for LLM interactions
        team: Optional Team object containing NPCs
    """
    import os
    import sys
    import code
    import re
    from pathlib import Path
    from npcpy.memory.command_history import CommandHistory, start_new_conversation
    from npcpy.npc_compiler import Team
    from npcpy.llm_funcs import get_llm_response
    
    # Set up guac environment
    setup_result = setup_guac_mode()
    lang = setup_result["language"]
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
    print("Type 'exit()' to exit")
    
    # Track number of commands for avocado-to-guac transformation
    command_count = 0
    max_stages = 5  # Number of stages in the transformation
    
    # Function to get the current prompt based on command count
    def get_guac_prompt():
        nonlocal command_count
        # Define the avocado-to-guac transformation stages
        stages = [
            "\U0001F951",            # Fresh avocado
            "\U0001F951 🔪",         # Avocado being cut
            "\U0001F951 🥣",         # Avocado in bowl
            "\U0001F951 🥣 🧂",      # Avocado being seasoned
            "\U0001F958"             # Guacamole
        ]
        
        # Calculate which stage we're at
        stage_index = min(command_count // 3, len(stages) - 1)
        return stages[stage_index] + " "
    
    # Start simple REPL based on language
    if lang == "python":
        # Try to import the guac package
        try:
            # First make sure we can import from guac.src
            import sys
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
        
        # Add basic pandas/numpy/matplotlib if not already there
        if 'pd' not in namespace:
            import pandas as pd
            namespace['pd'] = pd
        if 'np' not in namespace:
            import numpy as np
            namespace['np'] = np
        if 'plt' not in namespace:
            import matplotlib.pyplot as plt
            namespace['plt'] = plt
        
        # Function to determine if input is code
        def is_code(text):
            # Python syntax patterns
            code_patterns = [
                r'=', r'def\s+\w+\s*\(', r'class\s+\w+', r'import\s+\w+', r'from\s+\w+\s+import',
                r'\w+\.\w+\s*\(', r'if\s+.*:', r'for\s+.*:', r'while\s+.*:', r'\[.*\]', r'\{.*\}',
                r'\d+\s*[\+\-\*\/]\s*\d+', r'return\s+', r'lambda\s+\w+\s*:'
            ]
            
            # Check if any code pattern matches
            for pattern in code_patterns:
                if re.search(pattern, text):
                    return True
                
            return False
        
        # Custom interactive console with natural language support
        class GuacInteractiveConsole(code.InteractiveConsole):
            def raw_input(self, prompt=""):
                # Use evolving avocado/guac prompt
                return input(get_guac_prompt())
                
            def push(self, line):
                nonlocal command_count
                command_count += 1
                
                # Exit check
                if line.strip() in ('exit()', 'quit()', '/exit'):
                    return False
                
                # Add to command history
                command_history.add_command(
                    line, [], "", os.getcwd()
                )
                
                # Check if this is code or natural language
                if is_code(line):
                    # Treat as code
                    return super().push(line)
                else:
                    # Process as natural language if we have an NPC
                    if npc:
                        try:
                            # Process natural language using NPC
                            prompt = f"""
                            The user has entered the following in the guac shell:
                            "{line}"
                            
                            Generate Python code that addresses their query. 
                            Return ONLY executable Python code without any additional text or markdown.
                            """
                            
                            response = get_llm_response(prompt, npc=npc)
                            generated_code = response.get("response", "").strip()
                            
                            # Clean the code (remove markdown if present)
                            generated_code = re.sub(r'```python|```', '', generated_code).strip()
                            
                            # Display the code and execute it
                            print(f"\n# Generated code:")
                            print(f"{generated_code}\n")
                            
                            # Execute the generated code
                            try:
                                exec(generated_code, self.locals)
                                print("\n# Code executed successfully")
                            except Exception as e:
                                print(f"\n# Error executing code: {e}")
                                
                            return False
                        except Exception as e:
                            print(f"Error processing natural language: {str(e)}")
                            return False
                    else:
                        print("Natural language query detected but no NPC available to process it.")
                        return False
        
        # Start the Python REPL
        GuacInteractiveConsole(locals=namespace).interact(
            banner=f"Python {sys.version} with guac mode\nType code or natural language queries directly.\nType 'exit()' to exit."
        )
    
    elif lang == "r":
        # Simple R console with rpy2 if available
        try:
            import rpy2.robjects as ro
            
            # Source the main R file
            ro.r(f'source("{src_dir}/main.R")')
            
            # Start simple R REPL
            while True:
                try:
                    command_count += 1
                    r_cmd = input(get_guac_prompt())
                    
                    # Exit check
                    if r_cmd.strip() in ('q()', 'quit()', 'exit()', '/exit'):
                        break
                    
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
            
            // Create prompt stages for avocado-to-guac transformation
            const promptStages = [
                "🥑 ",
                "🥑 🔪 ",
                "🥑 🥣 ",
                "🥑 🥣 🧂 ",
                "🥬 "
            ];
            
            let commandCount = 0;
            
            // Custom prompt function 
            function getGuacPrompt() {{
                const stageIndex = Math.min(Math.floor(commandCount / 3), promptStages.length - 1);
                return promptStages[stageIndex];
            }}
            
            // Set prompt when REPL is available
            if (global._replServer) {{
                const origPrompt = global._replServer.prompt;
                
                // Override prompt getter
                Object.defineProperty(global._replServer, 'prompt', {{
                    get: function() {{ 
                        commandCount++;
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
    parser.add_argument("--npc_team_dir", type=str, help="NPC team directory")
    
    args = parser.parse_args()
    
    setup_guac_mode(
        config_dir=args.config_dir, 
        plots_dir=args.plots_dir,
        npc_team_dir=args.npc_team_dir
    )
    enter_guac_mode()
if __name__ == "__main__":
    main()
    