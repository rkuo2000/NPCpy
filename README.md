
<p align="center">
  <img src="https://raw.githubusercontent.com/cagostino/npcpy/main/npcpy.png" alt="npcpy logo of a solarpunk sign">
</p>


# npcpy

Welcome to `npcpy`, the python library for the NPC Toolkit and the home of the core command-line programs that make up the NPC Shell (`npcsh`). 


`npcpy` is an agent-based framework designed to easily integrate AI models into one's daily workflow and it does this by providing users with a variety of interfaces through which they can use, test, and explore the capabilities of AI models, agents, and agent systems. 

<p align="center">
  <a href= "https://github.com/cagostino/npcpy/blob/main/docs/npcpy.md"> 
  <img src="https://raw.githubusercontent.com/cagostino/npcpy/main/npcpy/npc-python.png" alt="npc-python logo" width=250></a>
</p>


Here is an example for getting responses for a particular agent:

```
from npcpy.npc_compiler import NPC
simon = NPC(
          name='Simon Bolivar',
          primary_directive='Liberate South America from the Spanish Royalists.',
          model='gemma3',
          provider='ollama',
         
          )
response = simon.get_llm_response("What is the most important territory to retain in the Andes mountains?")
print(response['response'])
```
``` 
The most important territory to retain in the Andes mountains is **Cuzco**. 
It’s the heart of the Inca Empire, a crucial logistical hub, and holds immense symbolic value for our liberation efforts. Control of Cuzco is paramount.
```


Here is an example for setting up an agent team:

```
from npcpy.npc_compiler import NPC, Team
ggm = NPC(
          name='gabriel garcia marquez',
          primary_directive='You are the author gabriel garcia marquez. see the stars ',
          model='deepseek-chat',
          provider='deepseek', # anthropic, gemini, openai, any supported by litellm
         
          )

isabel = NPC(
          name='isabel allende',
          primary_directive='You are the author isabel allende. visit the moon',
          model='deepseek-chat',
          provider='deepseek', # anthropic, gemini, openai, any supported by litellm
          )
borges = NPC(
          name='jorge luis borges',
          primary_directive='You are the author jorge luis borges. listen to the earth and work with your team',
          model='gpt-4o-mini',
          provider='openai', # anthropic, gemini, openai, any supported by litellm
          )          

# set up an NPC team with a forenpc that orchestrates the other npcs
lit_team = Team(npcs = [ggm, isabel], forenpc=borges)

print(lit_team.orchestrate('whats isabel working on? '))

 • Action chosen: pass_to_npc                                                                                                                                          
handling agent pass

 • Action chosen: answer_question                                                                                                                                      
 
{'debrief': {'summary': 'Isabel is finalizing preparations for her lunar expedition, focusing on recalibrating navigation systems and verifying the integrity of life support modules.',
  'recommendations': 'Proceed with thorough system tests under various conditions, conduct simulation runs of key mission phases, and confirm backup systems are operational before launch.'},
 'execution_history': [{'messages': [],
   'output': 'I am currently finalizing preparations for my lunar expedition. It involves recalibrating my navigation systems and verifying the integrity of my life support modules. Details are quite...complex.'}]}





print(lit_team.orchestrate('which book are your team members most proud of? ask them please. '))


In [2]: print(lit_team.orchestrate('which book are your team members most proud of?'))

 • Action chosen: execute_sequence                                                                                                                 
handling agent pass

 • Action chosen: answer_question                                                                                                                                      
handling agent pass

 • Action chosen: answer_question                                                                                                                          
response was not complete.. The response included answers from both Gabriel Garcia Marquez and Isabel Allende, which satisfies the requirement to get input from each team member about the book they are most proud of. However, it does not include a response from Jorge Luis Borges, who was the initial NPC to receive the request. To fully address the user's request, Borges should have provided his own answer before passing the question to the others.

 • Action chosen: pass_to_npc                                                                                                                                          
response was not complete.. The result did not provide any specific information about the books that team members are proud of, which is the core of the user's request.

 • Action chosen: execute_sequence                                                                                                                                     
handling agent pass

 • Action chosen: answer_question                                                                                                                                      
handling agent pass

 • Action chosen: answer_question                                                                                                                                      
{'debrief': {'summary': "The responses provided detailed accounts of the books that the NPC team members, Gabriel Garcia Marquez and Isabel Allende, are most proud of. Gabriel highlighted 'Cien años de soledad,' while Isabel spoke of 'La Casa de los Espíritus.' Both authors expressed deep personal connections to their works, illustrating their significance in Latin American literature and their own identities.", 'recommendations': 'Encourage further engagement with each author to explore more about their literary contributions, or consider asking about themes in their works or their thoughts on current literary trends.'}, 'execution_history': [{'messages': ...}]}
```


See more examples of how to use `npcpy` to create agents and agentic systems [here](https://github.com/cagostino/npcpy/blob/main/docs/npcpy.md). `npcpy` can include images, pdfs, and csvs in its llm response generation. 


## Inference Capabilities
- `npcpy` works with local and enterprise LLM providers through its LiteLLM integration, allowing users to run inference from Ollama, LMStudio, OpenAI, Anthropic, Gemini, and Deepseek, making it a versatile tool for both simple commands and sophisticated AI-driven tasks. 

## Read the Docs

Read the docs at [npcpy.readthedocs.io](https://npcpy.readthedocs.io/en/latest/)


## NPC Studio
There is a graphical user interface that makes use of the NPC Toolkit through the NPC Studio. See the open source code for NPC Studio [here](https://github.com/cagostino/npc-studio). Download the executables (soon) at [our website](https://www.npcworldwi.de/npc-studio).


## Mailing List
Interested to stay in the loop and to hear the latest and greatest about `npcpy`, `npcsh`, and NPC Studio? Be sure to sign up for the [newsletter](https://forms.gle/n1NzQmwjsV4xv1B2A)!


## Support
If you appreciate the work here, [consider supporting NPC Worldwide](https://buymeacoffee.com/npcworldwide). If you'd like to explore how to use `npcpy` to help your business, please reach out to info@npcworldwi.de .



## NPC Shell

The NPC shell is a suite of executable command-line programs that allow users to easily interact with NPCs and LLMs through a command line shell. 
Programs within the NPC shell use the properties defined in `~/.npcshrc`, which is generated upon installation and running of `npcsh` for the first time.



The following are the current programs in the NPC shell:



## `npcsh`
<p align="center">
  <a href= "https://github.com/cagostino/npcpy/blob/main/docs/guide.md"> 
  <img src="https://raw.githubusercontent.com/cagostino/npcpy/main/npcpy/npcsh.png" alt="npcsh logo" width=250></a>
</p> 

- a bash-replacement shell (`npcsh`) that can process bash, natural language, or special macro calls. `npcsh` detects whether input is bash or natural language and processes it accordingly. 
    
    - Users can specify whether natural language commands are processed in one of three ways:
        - agentically (i.e. an NPC reviews and decides to pass to other NPCs or to use tools (called `jinxs`)) to carry out tasks.
        - conversationally (the NPC generates a response which the user can approve to run) 
        - directly through bash execution (the NPC responds by generating executable bash code which is then processed automatically in the shell.
    
        Switching between the modes within the session is straightforward and the user can specify the default mode in the `.npcshrc` file described in greater detail below. The default mode is agentic, but the user can switch by typing `/chat` to switch to conversational mode or `/cmd` to switch to bash execution mode.
    <details>  <summary>Click to see some examples</summary>
    
    - Web searching     
        ```
        /search -p perplexity 'cal bears football schedule'
        ```
    - One shot sampling 
        ```
        /sample 'prompt'
        ```

    - Image generation:      
        ```
        /vixynt 'an image of a dog eating a hat'
        ```
        
    - Process Identification:       
        ```    
        please identify the process consuming the most memory on my computer
        ```
        

    - Jinx Use:   
        ```
        identify what windows are open on my computer desktop at the moment
        ```
    
    - Screenshot analysis:     
        ```
        /ots
        ```
    - voice chat:     
        ```
        /yap
        ```
    - Computer use:     
        ```
        /plonk -n 'npc_name' -sp 'task for plonk to carry out '
        ```
    - Enter chat loop with an NPC:     
        ```
        /spool -n <npc_name>
        ```
    </details>





## `guac`

<p align="center"><a href ="https://github.com/cagostino/npcpy/blob/main/docs/guac.md"> 
  <img src="https://raw.githubusercontent.com/cagostino/npcpy/main/npcpy/npc_team/guac.png" alt="npcpy logo of a solarpunk sign", width=250></a>
</p> 

- a replacement shell for interpreters like python/r/node/julia with an avocado input marker 🥑 that brings a pomodoro-like approach to interactive coding. <details>  <summary>Click to see some examples</summary>

    - Simulation:      
        `🥑 Make a markov chain simulation of a random walk in 2D space with 1000 steps and visualize`
        ```
        # Generated python code:
        import numpy as np
        import matplotlib.pyplot as plt

        # Number of steps
        n_steps = 1000

        # Possible moves: up, down, left, right
        moves = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])

        # Initialize position array
        positions = np.zeros((n_steps+1, 2), dtype=int)

        # Generate random moves
        for i in range(1, n_steps+1):
            step = moves[np.random.choice(4)]
            positions[i] = positions[i-1] + step

        # Plot the random walk
        plt.figure(figsize=(8, 8))
        plt.plot(positions[:, 0], positions[:, 1], lw=1)
        plt.scatter([positions[0, 0]], [positions[0, 1]], color='green', label='Start')
        plt.scatter([positions[-1, 0]], [positions[-1, 1]], color='red', label='End')
        plt.title('2D Random Walk - 1000 Steps (Markov Chain)')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()
        # Generated code executed successfully

        ```
        Access the variables created in the code:    
        `🥑 print(positions)`
        ```
        [[  0   0]
        [  0  -1]
        [ -1  -1]
        ...
        [ 29 -23]
        [ 28 -23]
        [ 27 -23]]
        ```

    - Run a python script:   
        `🥑 run file.py`    
    - Refresh:    
        `🥑 /refresh`       
    - Show current variables:    
        `🥑 /show`    

    A guac session progresses through a series of stages, each of equal length. Each stage adjusts the emoji input prompt. Once the stages have passed, it is time to refresh. Stage 1: `🥑`, Stage 2: `🥑🔪` Stage 3: `🥑🥣` Stage:4 `🥑🥣🧂`, `Stage 5: 🥘 TIME TO REFRESH`. At stage 5, the user is reminded to refresh with the /refresh macro. This will evaluate the session so farand suggest and implement new functions or automations that will aid in future sessions, with the ultimate approval of the user.
 </details>

## `npc`
- a command line interface offering the capabilities of the npc shell from a regular bash shell. Our mascot agent sibiji the spider will help you weave your agent web with the `npc` cli. 

    <p align="center">
    <img src="https://raw.githubusercontent.com/cagostino/npcsh/main/npcpy/npc_team/sibiji.png" alt="npcsh logo with sibiji the spider">
    </p>
    
### Cheat Sheet for NPC shell and cli
 <details>  <summary> Toggle Cheat Sheet </summary>

- The NPC shell and CLI let users iterate and experiment with AI in a natural way. Below is a cheat sheet that shows how to use the NPC Toolkit's macro commands in both the shell and the CLI. For the `npcsh` commands to work, one must activate `npcsh` by typing it in a shell.

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
    



</details> 


## `alicanto` : a deep research focused agent flow. 

<p align="center"><a href ="https://github.com/cagostino/npcpy/blob/main/docs/deep.md"> 
  <img src="https://raw.githubusercontent.com/cagostino/npcpy/main/npcpy/npc_team/alicanto.png" alt="logo for deep research", width=250></a>
</p>

 
## `pti`
-  a reasoning REPL loop with explicit checks to request inputs from users following thinking traces.
 
<p align="center"><a href ="https://github.com/cagostino/npcpy/blob/main/docs/pti.md"> 
  <img src="https://raw.githubusercontent.com/cagostino/npcpy/main/npcpy/npc_team/frederic4.png" alt="npcpy logo of frederic the bear and the pti logo", width=250></a>
</p>


## `spool`
- a simple agentic REPL chat loop with a specified agent.

<p align="center"><a href ="https://github.com/cagostino/npcpy/blob/main/docs/spool.md"> 
  <img src="https://raw.githubusercontent.com/cagostino/npcpy/main/npcpy/npc_team/spool.png" alt="logo for spool", width=250></a>
</p>

## `yap`


<p align="center"><a href ="https://github.com/cagostino/npcpy/blob/main/docs/yap.md"> 
  <img src="https://raw.githubusercontent.com/cagostino/npcpy/main/npcpy/npc_team/yap.png" alt="logo for yap ", width=250></a>
</p>

- an agentic voice control loop with a specified agent. 


## `wander` 

<p align="center"><a href ="https://github.com/cagostino/npcpy/blob/main/docs/wander.md">
  <img src="https://raw.githubusercontent.com/cagostino/npcpy/main/npcpy/npc_team/kadiefa.png" alt="logo for wander", width=250></a>
</p>
- a system for thinking outside of the box

## Enabling Innovation
- `npcpy` is a framework that speeds up and simplifies the development of NLP-based or Agent-based applications and provides developers and researchers with methods to explore and test across dozens of models, providers, and personas as well as other model-level hyperparameters (e.g. `temperature`, `top_k`, etc.), incorporating an array of data sources and common tools.
- The `npcpy` agent data layer makes it easy to set up teams and serve them so you can focus more on the agent personas and less on the nitty gritty of inference.
- `npcpy` provides pioneering methods in the construction and updating of knowledge graphs as well as in the development and testing of novel mixture of agent scenarios.
- The agentic interfaces (`npcsh`, `guac`, etc.) provided as part of `npcpy` can serve as templates for developers to modify in order to create their own specialized loops that fit their own workflow best or to adapt even to their own full stack application. 

- In `npcpy`, all agentic capabilities are developed and tested using small local models (like `llama3.2`, `gemma3`) to ensure it can function reliably at the edge of computing.







## Installation
`npcpy` is available on PyPI and can be installed using pip. Before installing, make sure you have the necessary dependencies installed on your system. Below are the instructions for installing such dependencies on Linux, Mac, and Windows. If you find any other dependencies that are needed, please let us know so we can update the installation instructions to be more accommodating.

### Linux install
<details>  <summary> Toggle </summary>
  
```bash
# these are for audio primarily, skip if you dont need tts
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
pip install npcpy[yap]

# if you want everything:
pip install npcpy[all]
```
</details>


### Mac install
<details>  <summary> Toggle </summary>
```bash
#mainly for audio
brew install portaudio
brew install ffmpeg
brew install pygobject3

# for triggers
brew install inotify-tools


brew install ollama
brew services start ollama
ollama pull llama3.2
ollama pull llava:7b
ollama pull nomic-embed-text
pip install npcpy
# if you want to install with the API libraries
pip install npcpy[lite]
# if you want the full local package set up (ollama, diffusers, transformers, cuda etc.)
pip install npcpy[local]
# if you want to use tts/stt
pip install npcpy[yap]

# if you want everything:
pip install npcpy[all]
```
</details>

### Windows Install

<details>  <summary> Toggle </summary>
Download and install ollama exe.

Then, in a powershell. Download and install ffmpeg.

```powershell
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

</details>

### Fedora Install (under construction)

<details>  <summary> Toggle </summary>
```bash
python3-dev (fixes hnswlib issues with chroma db)
xhost +  (pyautogui)
python-tkinter (pyautogui)

```

</details>

## Startup Configuration and Project Structure
After `npcpy` has been pip installed, `npcsh`, `guac`, `pti`, `spool`, `yap` and the `npc` CLI can be used as command line tools. To initialize these correctly, first start by starting the NPC shell:
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

`npcsh` also comes with a set of jinxs and NPCs that are used in processing. It will generate a folder at ~/.npcsh/ that contains the tools and NPCs that are used in the shell and these will be used in the absence of other project-specific ones. Additionally, `npcsh` records interactions and compiled information about npcs within a local SQLite database at the path specified in the .npcshrc file. This will default to ~/npcsh_history.db if not specified. When the data mode is used to load or analyze data in CSVs or PDFs, these data will be stored in the same database for future reference.

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
For cases where you wish to set up a project specific set of NPCs, jinxs, and assembly lines, add a `npc_team` directory to your project and `npcsh` should be able to pick up on its presence, like so:
```bash
./npc_team/            # Project-specific NPCs
├── jinxs/             # Project jinxs #example jinx next
│   └── example.jinx
└── assembly_lines/    # Project workflows
    └── example.pipe
└── models/    # Project workflows
    └── example.model
└── example1.npc        # Example NPC
└── example2.npc        # Example NPC
└── team.ctx            # Example ctx


```

### Migrations and deprecations and major changes
### v0.3.37
<details>  <summary> Toggle </summary>
- added team to the conversation history table. 

 </details>

### v0.3.35
<details>  <summary> Toggle </summary>
-In v0.3.35, there were many significant changes to the structure of npcpy, introducing various new submodules for data I/O (`data`), AI model generation and inference (`gen`), command history, knowledge graph, and search features (`memory`), mixture of agents methods and schemes (`mix`), modes for interaction like `spool`, `guac`, `wander`, `yap`, `pti`, and more (`modes`), SQL-focused tooling (`sql`) and computer automations like `cron`, `systemctl`, `pyautogui`, etc (`work`) .

 </details>
 
### v0.3.33
<details>  <summary> Toggle </summary>
-In v0.3.33, the NPCCompiler object was phased out and the global/project dichotomy was removed. 
-the primary python package entrypoint was renamed from npcsh to npcpy
-npcsh is still automatically installed and available, but we will have a better separation of responsibilities in the NPC framework when the shell handles these rather than integrating it across the library.
-context files are being introduced.
 </details>

## Contributing
Contributions are welcome! Please submit issues and pull requests on the GitHub repository.


## License
This project is licensed under the MIT License.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=cagostino/npcpy&type=Date)](https://star-history.com/#cagostino/npcpy&Date)
