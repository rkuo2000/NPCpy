
<p align="center">
  <img src="https://raw.githubusercontent.com/cagostino/npcsh/main/npcpy.png" alt="npcpy logo of a solarpunk sign">
</p>


# npcpy

Welcome to `npcpy`, the python library for the NPC Toolkit and the home of the core command-line programs that make up the NPC Shell (`npcsh`). `npcpy` is an agent-based framework designed to easily integrate AI models into one's daily workflow and it does this by providing users with a variety of interfaces through which they can use, test, and explore the capabilities of AI models, agents, and agent systems. These include the following:

- an extensible python library (`npcpy`) with convenient methods for getting LLM responses, loading data, creating agents, and implementing agentic capabilities in new custom systems.

- a bash-replacement shell (`npcsh`) that can process bash, natural language, or special macro calls for procedures like image generation (`/vixynt 'prompt'`), web searching (`/search -p perplexity 'cal bears football schedule'`), and one-off LLM response samples (`/sample 'prompt'`). Users can specify whether natural language is processed agentically (i.e. an LLM reviews and decides to pass to other agents or use tools) or directly through bash execution.

- a command line interface (`npc`) with the ability to process calls naturally (`npc 'prompt'`) or to call NPC macro commands like running a flask server to provide API access to an agent team (`npc serve`), web searching (`npc search -p perplexity 'cal bears football schedule'`), etc

- a replacement shell for interpreters like python/r/node/julia (`guac`) that brings a pomodoro-like approach to interactive coding.

- a reasoning REPL loop with explicit checks to request inputs from users following thinking traces  (`pti`) which can be accessed by running `pti` directly, or by running `npc pti` from the command line or `/pti` from `npcsh`.
- a simple agentic REPL loop (`spool`) which can be accessed by running `spool` directly, or by running `npc spool` from the command line or `/spool` from `npcsh`.

- a voice control REPL loop (`yap`) which can be accessed by running `yap` directly, or by running `npc yap` from the command line or `/yap` from `npcsh`.


`npcpy` works with local and enterprise LLM providers through its LiteLLM integration, allowing users to run inference from Ollama, LMStudio, OpenAI, Anthropic, Gemini, and Deepseek, making it a versatile tool for both simple commands and sophisticated AI-driven tasks. 

In `npcpy`, all agentic capabilities are built and tested using small local models (like `llama3.2`) to ensure it can function reliably even at the edge of computing.




Read the docs at [npcpy.readthedocs.io](https://npcpy.readthedocs.io/en/latest/)

## NPC Studio
There is a graphical user interface that makes use of the NPC Toolkit through the NPC Studio. See the open source code for NPC Studio [here](https://github.com/cagostino/npc-studio). Download the executables (soon) at [our website](https://www.npcworldwi.de/npc-studio).


## Mailing List
Interested to stay in the loop and to hear the latest and greatest about `npcpy`, `npcsh`, and NPC Studio? Be sure to sign up for the [newsletter](https://forms.gle/n1NzQmwjsV4xv1B2A)!





## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=cagostino/npcpy&type=Date)](https://star-history.com/#cagostino/npcpy&Date)

## TLDR Cheat Sheet for NPC shell and cli
The NPC shell and cli let users iterate and experiment with AI in a natural way. Below is a cheat sheet that shows how to use the NPC Toolkit's macro commands in both the shell and the CLI. For the `npcsh` commands to work, one must activate `npcsh` by typing it in a shell.

| Task | npc CLI | npcsh |
|----------|----------|----------|
| Ask a generic question | npc 'prompt' | 'prompt' |
| Compile an NPC | npc compile /path/to/npc.npc | /compile /path/to/npc.npc |
| Computer use | npc plonk -n 'npc_name' -sp 'task for plonk to carry out '| /plonk -n 'npc_name' -sp 'task for plonk to carry out ' |
| Enter a chat with an NPC (NPC needs to be compiled first) | npc spool -n npc_name | /spool -n <npc_name> |
| Generate image    | npc vixynt 'prompt'  | /vixynt prompt   |
| Get a sample LLM response  | npc sample 'prompt'   | /sample prompt for llm  |
| Search the web | npc search -q "cal golden bears football schedule" -sp perplexity | /search -p perplexity 'cal bears football schedule' |
| Serve an NPC team | npc serve --port 5337 --cors='http://localhost:5137/' | /serve --port 5337 --cors='http://localhost:5137/' |
| Screenshot analysis  | npc ots |  /ots  |
| Voice Chat    | npc yap   | /yap   |


When beginning, `npcsh` initializes a set of agents that you can use and tweak as you go. Our mascot agent is sibiji the spider and will help you weave your agent web! 

<p align="center">
  <img src="https://raw.githubusercontent.com/cagostino/npcsh/main/npcpy/npc_team/sibiji.png" alt="npcsh logo with sibiji the spider">
</p>


## Python Examples
Integrate `npcpy` into your Python projects for additional flexibility. Below are a few examples of how to use the library programmatically.

### Example 1: using npcpy's get_llm_response and get_stream

```python
from npcpy.llm_funcs import get_llm_response

# ollama's llama3.2
response = get_llm_response("What is the capital of France? Respond with a json object containing 'capital' as the key and the capital as the value.",
                            model='llama3.2',
                            provider='ollama',
                            format='json')
print(response)
# assistant's response is contained in the 'response' key for easier access
assistant_response = response['response']
print(assistant_response)
# access messages too
messages = response['messages']
print(messages)


#openai's gpt-4o-mini
from npcpy.llm_funcs import get_llm_response

response = get_llm_response("What is the capital of France? Respond with a json object containing 'capital' as the key and the capital as the value.",
                            model='gpt-4o-mini',
                            provider='openai',
                            format='json')
print(response)
# anthropic's claude haikue 3.5 latest
from npcpy.llm_funcs import get_llm_response

response = get_llm_response("What is the capital of France? Respond with a json object containing 'capital' as the key and the capital as the value.",
                            model='claude-3-5-haiku-latest',
                            provider='anthropic',
                            format='json')



# alternatively, if you have NPCSH_CHAT_MODEL / NPCSH_CHAT_PROVIDER set in your ~/.npcshrc, it will use those values
response = get_llm_response("What is the capital of France? Respond with a json object containing 'capital' as the key and the capital as the value.",
                            format='json')


# with stream
# alternatively, if you have NPCSH_CHAT_MODEL / NPCSH_CHAT_PROVIDER set in your ~/.npcshrc, it will use those values
response = get_llm_response("whats going on tonight?",
                            model='gpt-4o-mini',
                            provider='openai',
                            stream=True)

for chunk in response['response']:
    print(chunk)
```

### Example 2: Building a flow with check_llm_command

```python
#first let's demonstrate the capabilities of npcsh's check_llm_command
from npcpy.llm_funcs import check_llm_command

command = 'can you write a description of the idea of semantic degeneracy?'

response = check_llm_command(command,
                             model='gpt-4o-mini',
                             provider='openai')



# now to make the most of check_llm_command, let's add an NPC with a generic code execution tool


from npcpy.npc_compiler import NPC, Tool
from npcpy.llm_funcs import check_llm_command

code_execution_tool = Tool(
    {
        "tool_name": "execute_python",
        "description": """Executes a code block in python.
                Final output from script MUST be stored in a variable called `output`.
                          """,
        "inputs": ["script"],
        "steps": [
            {
                "engine": " python",
                "code": """{{ script }}""",
            }
        ],
    }
)


command = """can you write a description of the idea of semantic degeneracy and save it to a file?
             After, can you take that and make various versions of it from the points of
             views of different sub-disciplines of natural lanaguage processing?
             Finally produce a synthesis of the resultant various versions and save it."
            """
npc = NPC(
    name="NLP_Master",
    primary_directive="Provide astute anlayses on topics related to NLP. Carry out relevant tasks for users to aid them in their NLP-based analyses",
    model="gpt-4o-mini",
    provider="openai",
    tools=[code_execution_tool],
)
response = check_llm_command(
    command, model="gpt-4o-mini", provider="openai", npc=npc, stream=False
)


# or by attaching an NPC Team
from npcpy.npc_compiler import NPC

response = check_llm_command(command,
                             model='gpt-4o-mini',
                              provider='openai',)
```



### Example 3: Creating and Using an NPC
This example shows how to create and initialize an NPC and use it to answer a question.
```python
import sqlite3
from npcpy.npc_compiler import NPC

# Set up database connection
db_path = '~/npcsh_history.db'
conn = sqlite3.connect(db_path)

# Load NPC from a file
npc = NPC(
          name='Simon Bolivar',
          db_conn=conn,
          primary_directive='Liberate South America from the Spanish Royalists.',
          model='gpt-4o-mini',
          provider='openai',
          )

response = npc.get_llm_response("What is the most important territory to retain in the Andes mountains?")
print(response['response'])
```
```bash
'The most important territory to retain in the Andes mountains for the cause of liberation in South America would be the region of Quito in present-day Ecuador. This area is strategically significant due to its location and access to key trade routes. It also acts as a vital link between the northern and southern parts of the continent, influencing both military movements and the morale of the independence struggle. Retaining control over Quito would bolster efforts to unite various factions in the fight against Spanish colonial rule across the Andean states.'
```
### Example 4: Orchestrating a team



```python
import pandas as pd
import numpy as np
import os
from npcpy.npc_compiler import NPC, Team, Tool


# Create test data and save to CSV
def create_test_data(filepath="sales_data.csv"):
    sales_data = pd.DataFrame(
        {
            "date": pd.date_range(start="2024-01-01", periods=90),
            "revenue": np.random.normal(10000, 2000, 90),
            "customer_count": np.random.poisson(100, 90),
            "avg_ticket": np.random.normal(100, 20, 90),
            "region": np.random.choice(["North", "South", "East", "West"], 90),
            "channel": np.random.choice(["Online", "Store", "Mobile"], 90),
        }
    )

    # Add patterns to make data more realistic
    sales_data["revenue"] *= 1 + 0.3 * np.sin(
        np.pi * np.arange(90) / 30
    )  # Seasonal pattern
    sales_data.loc[sales_data["channel"] == "Mobile", "revenue"] *= 1.1  # Mobile growth
    sales_data.loc[
        sales_data["channel"] == "Online", "customer_count"
    ] *= 1.2  # Online customer growth

    sales_data.to_csv(filepath, index=False)
    return filepath, sales_data


code_execution_tool = Tool(
    {
        "tool_name": "execute_code",
        "description": """Executes a Python code block with access to pandas,
                          numpy, and matplotlib.
                          Results should be stored in the 'results' dict to be returned.
                          The only input should be a single code block with \n characters included.
                          The code block must use only the  libraries or methods contained withen the
                            pandas, numpy, and matplotlib libraries or using builtin methods.
                          do not include any json formatting or markdown formatting.

                          When generating your script, the final output must be encoded in a variable
                          named "output". e.g.

                          output  = some_analysis_function(inputs, derived_data_from_inputs)
                            Adapt accordingly based on the scope of the analysis

                          """,
        "inputs": ["script"],
        "steps": [
            {
                "engine": "python",
                "code": """{{script}}""",
            }
        ],
    }
)

# Analytics team definition
analytics_team = [
    {
        "name": "analyst",
        "primary_directive": "You analyze sales performance data, focusing on revenue trends, customer behavior metrics, and market indicators. Your expertise is in extracting actionable insights from complex datasets.",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "tools": [code_execution_tool],  # Only the code execution tool
    },
    {
        "name": "researcher",
        "primary_directive": "You specialize in causal analysis and experimental design. Given data insights, you determine what factors drive observed patterns and design tests to validate hypotheses.",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "tools": [code_execution_tool],  # Only the code execution tool
    },
    {
        "name": "engineer",
        "primary_directive": "You implement data pipelines and optimize data processing. When given analysis requirements, you create efficient workflows to automate insights generation.",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "tools": [code_execution_tool],  # Only the code execution tool
    },
]


def create_analytics_team():
    # Initialize NPCs with just the code execution tool
    npcs = []
    for npc_data in analytics_team:
        npc = NPC(
            name=npc_data["name"],
            primary_directive=npc_data["primary_directive"],
            model=npc_data["model"],
            provider=npc_data["provider"],
            tools=[code_execution_tool],  # Only code execution tool
        )
        npcs.append(npc)

    # Create coordinator with just code execution tool
    coordinator = NPC(
        name="coordinator",
        primary_directive="You coordinate the analytics team, ensuring each specialist contributes their expertise effectively. You synthesize insights and manage the workflow.",
        model="gpt-4o-mini",
        provider="openai",
        tools=[code_execution_tool],  # Only code execution tool
    )

    # Create team
    team = Team(npcs=npcs, foreman=coordinator)
    return team


def main():
    # Create and save test data
    data_path, sales_data = create_test_data()

    # Initialize team
    team = create_analytics_team()

    # Run analysis - updated prompt to reflect code execution approach
    results = team.orchestrate(
        f"""
    Analyze the sales data at {data_path} to:
    1. Identify key performance drivers
    2. Determine if mobile channel growth is significant
    3. Recommend tests to validate growth hypotheses

    Here is a header for the data file at {data_path}:
    {sales_data.head()}

    When working with dates, ensure that date columns are converted from raw strings. e.g. use the pd.to_datetime function.


    When working with potentially messy data, handle null values by using nan versions of numpy functions or
    by filtering them with a mask .

    Use Python code execution to perform the analysis - load the data and perform statistical analysis directly.
    """
    )

    print(results)

    # Cleanup
    os.remove(data_path)


if __name__ == "__main__":
    main()

```



## Installation
`npcpy` is available on PyPI and can be installed using pip. Before installing, make sure you have the necessary dependencies installed on your system. Below are the instructions for installing such dependencies on Linux, Mac, and Windows. If you find any other dependencies that are needed, please let us know so we can update the installation instructions to be more accommodating.

### Linux install
```bash

# for audio primarily
sudo apt-get install espeak
sudo apt-get install portaudio19-dev python3-pyaudio
sudo apt-get install alsa-base alsa-utils
sudo apt-get install libcairo2-dev
sudo apt-get install libgirepository1.0-dev
sudo apt-get install ffmpeg

# for triggers
sudo apt install inotify-tools


#And if you don't have ollama installed, use this:
curl -fsSL https://ollama.com/install.sh | sh

ollama pull llama3.2
ollama pull llava:7b
ollama pull nomic-embed-text
pip install npcpy
# if you want to install with the API libraries
pip install npcpy[lite]
# if you want the full local package set up (ollama, diffusers, transformers, cuda etc.)
pip install npcpy[local]
# if you want to use tts/stt
pip install npcpy[whisper]

# if you want everything:
pip install npcpy[all]




### Mac install
```bash
#mainly for audio
brew install portaudio
brew install ffmpeg
brew install pygobject3

# for triggers
brew install ...


brew install ollama
brew services start ollama
ollama pull llama3.2
ollama pull llava:7b
ollama pull nomic-embed-text
pip install npcsh
# if you want to install with the API libraries
pip install npcpy[lite]
# if you want the full local package set up (ollama, diffusers, transformers, cuda etc.)
pip install npcpy[local]
# if you want to use tts/stt
pip install npcpy[whisper]

# if you want everything:
pip install npcpy[all]

```
### Windows Install

Download and install ollama exe.

Then, in a powershell. Download and install ffmpeg.

```
ollama pull llama3.2
ollama pull llava:7b
ollama pull nomic-embed-text
pip install npcsh
# if you want to install with the API libraries
pip install npcsh[lite]
# if you want the full local package set up (ollama, diffusers, transformers, cuda etc.)
pip install npcpy[local]
# if you want to use tts/stt
pip install npcpy[yap]

# if you want everything:
pip install npcpy[all]

```
As of now, npcsh appears to work well with some of the core functionalities like /ots and /whisper.


### Fedora Install (under construction)

python3-dev (fixes hnswlib issues with chroma db)
xhost +  (pyautogui)
python-tkinter (pyautogui)

## Startup Configuration and Project Structure
After it has been pip installed, `npcsh` can be used as a command line tool. Start it by typing:
```bash
npcsh
```
When initialized, `npcsh` will generate a .npcshrc file in your home directory that stores your npcsh settings.
Here is an example of what the .npcshrc file might look like after this has been run.
```bash
# NPCSH Configuration File
export NPCSH_INITIALIZED=1
export NPCSH_CHAT_PROVIDER='ollama'
export NPCSH_CHAT_MODEL='llama3.2'
export NPCSH_DB_PATH='~/npcsh_history.db'
```
`npcsh` also comes with a set of tools and NPCs that are used in processing. It will generate a folder at ~/.npcsh/ that contains the tools and NPCs that are used in the shell and these will be used in the absence of other project-specific ones. Additionally, `npcsh` records interactions and compiled information about npcs within a local SQLite database at the path specified in the .npcshrc file. This will default to ~/npcsh_history.db if not specified. When the data mode is used to load or analyze data in CSVs or PDFs, these data will be stored in the same database for future reference.

The installer will automatically add this file to your shell config, but if it does not do so successfully for whatever reason you can add the following to your .bashrc or .zshrc:

```bash
# Source NPCSH configuration
if [ -f ~/.npcshrc ]; then
    . ~/.npcshrc
fi
```

We support inference via all providers supported by litellm. For openai-compatible providers that are not explicitly named in litellm, use simply `openai-like` as the provider. The default provider must be one of `['openai','anthropic','ollama', 'gemini', 'deepseek', 'openai-like']` and the model must be one available from those providers.

To use tools that require API keys, create an `.env` file in the folder where you are working or place relevant API keys as env variables in your ~/.npcshrc. If you already have these API keys set in a ~/.bashrc or a ~/.zshrc or similar files, you need not additionally add them to ~/.npcshrc or to an `.env` file. Here is an example of what an `.env` file might look like:

```bash
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export DEEPSEEK_API_KEY='your_deepseek_key'
export GEMINI_API_KEY='your_gemini_key'
export PERPLEXITY_API_KEY='your_perplexity_key'
```


 Individual npcs can also be set to use different models and providers by setting the `model` and `provider` keys in the npc files.
 Once initialized and set up, you will find the following in your ~/.npcsh directory:
```bash
~/.npcsh/
├── npc_team/           # Global NPCs
│   ├── tools/          # Global tools
│   └── assembly_lines/ # Workflow pipelines

```
For cases where you wish to set up a project specific set of NPCs, tools, and assembly lines, add a `npc_team` directory to your project and `npcsh` should be able to pick up on its presence, like so:
```bash
./npc_team/            # Project-specific NPCs
├── tools/             # Project tools #example tool next
│   └── example.tool
└── assembly_lines/    # Project workflows
    └── example.pipe
└── models/    # Project workflows
    └── example.model
└── example1.npc        # Example NPC
└── example2.npc        # Example NPC
└── team.ctx            # Example ctx


```

## IMPORTANT: migrations and deprecations and major changes

### v0.3.34
-In v0.3.34, there were many significant changes to the structure of npcpy, introducing various new submodules for data I/O (`data`), AI model generation and inference (`gen`), command history, knowledge graph, and search features (`memory`), mixture of agents methods and schemes (`mix`), modes for interaction like `spool`, `guac`, `wander`, `yap`, `pti`, and more (`modes`), SQL-focused tooling (`sql`) and computer automations like `cron`, `systemctl`, `pyautogui`, etc (`work`) .


 
### v0.3.33
-In v0.3.33, the NPCCompiler object was phased out and the global/project dichotomy was removed. 
-the primary python package entrypoint was renamed from npcsh to npcpy
-npcsh is still automatically installed and available, but we will have a better separation of responsibilities in the NPC framework when the shell handles these rather than integrating it across the library.
-context files are being introduced.


### v0.3.4
-In v0.3.4, the structure for tools was adjusted. If you have made custom tools please refer to the structure within npc_compiler to ensure that they are in the correct format. Otherwise, do the following
```bash
rm ~/.npcsh/npc_team/tools/*.tool
```
and then
```bash
npcsh
```
and the updated tools will be copied over into the correct location.

### v0.3.5
-Version 0.3.5 included a complete overhaul and refactoring of the llm_funcs module. This was done to make it not as horribly long and to make it easier to add new models and providers


-in version 0.3.5, a change was introduced to the database schema for messages to add npcs, models, providers, and associated attachments to data. If you have used `npcsh` before this version, you will need to run this migration script to update your database schema:   [migrate_conversation_history_v0.3.5.py](https://github.com/cagostino/npcsh/blob/cfb9dc226e227b3e888f3abab53585693e77f43d/npcsh/migrations/migrate_conversation_history_%3Cv0.3.4-%3Ev0.3.5.py)

-additionally, NPCSH_MODEL and NPCSH_PROVIDER have been renamed to NPCSH_CHAT_MODEL and NPCSH_CHAT_PROVIDER
to provide a more consistent naming scheme now that we have additionally introduced `NPCSH_VISION_MODEL` and `NPCSH_VISION_PROVIDER`, `NPCSH_EMBEDDING_MODEL`, `NPCSH_EMBEDDING_PROVIDER`, `NPCSH_REASONING_MODEL`, `NPCSH_REASONING_PROVIDER`, `NPCSH_IMAGE_GEN_MODEL`, and `NPCSH_IMAGE_GEN_PROVIDER`.
- In addition, we have added NPCSH_API_URL to better accommodate openai-like apis that require a specific url to be set as well as `NPCSH_STREAM_OUTPUT` to indicate whether or not to use streaming in one's responses. It will be set to 0 (false) by default as it has only been tested  and verified for a small subset of the models and providers we have available (openai, anthropic, and ollama). If you try it and run into issues, please post them here so we can correct them as soon as possible !



## Contributing
Contributions are welcome! Please submit issues and pull requests on the GitHub repository.

## Support
If you appreciate the work here, [consider supporting NPC Worldwide](https://buymeacoffee.com/npcworldwide). If you'd like to explore how to use `npcsh` to help your business, please reach out to info@npcworldwi.de .


## License
This project is licensed under the MIT License.
