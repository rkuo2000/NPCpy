<p align="center">
  <img src="https://raw.githubusercontent.com/cagostino/npcpy/main/npcpy.png" alt="npcpy logo of a solarpunk sign">
</p>


# npcpy

Welcome to `npcpy`, the python library for the NPC Toolkit that supercharges natural language processing pipelines and agent tooling. `npcpy` is a python framework for building systems with LLMs that can generate text, images, and videos while allowing users to easily integrate data sources in their response handling.


<p align="center">
  <a href= "https://github.com/cagostino/npcpy/blob/main/docs/npcpy.md"> 
  <img src="https://raw.githubusercontent.com/cagostino/npcpy/main/npcpy/npc-python.png" alt="npc-python logo" width=250></a>
</p>


Here is an example for getting responses for a particular agent:

```python
from npcpy.npc_compiler import NPC
simon = NPC(
          name='Simon Bolivar',
          primary_directive='Liberate South America from the Spanish Royalists.',
          model='gemma3:4b',
          provider='ollama'
          )
response = simon.get_llm_response("What is the most important territory to retain in the Andes mountains?")
print(response['response'])
```
```python 
The most important territory to retain in the Andes mountains is **Cuzco**. 
It’s the heart of the Inca Empire, a crucial logistical hub, and holds immense symbolic value for our liberation efforts. Control of Cuzco is paramount.
```


Here is an example for setting up an agent team:

```python
from npcpy.npc_compiler import NPC, Team
ggm = NPC(
          name='gabriel garcia marquez',
          primary_directive='You are the author gabriel garcia marquez. see the stars ',
          model='gemma3:4b',
          provider='ollama', # anthropic, gemini, openai, any supported by litellm
          )

isabel = NPC(
          name='isabel allende',
          primary_directive='You are the author isabel allende. visit the moon',
          model='llama3.2:8b',
          provider='ollama', # anthropic, gemini, openai, any supported by litellm
          )
borges = NPC(
          name='jorge luis borges',
          primary_directive='You are the author jorge luis borges. listen to the earth and work with your team',
          model='qwen3:latest',
          provider='ollama', # anthropic, gemini, openai, any supported by litellm
          )          

# set up an NPC team with a forenpc that orchestrates the other npcs
lit_team = Team(npcs = [ggm, isabel], forenpc=borges)

print(lit_team.orchestrate('whats isabel working on? '))
```
```
 • Action chosen: pass_to_npc                                                                                                                                          
handling agent pass

 • Action chosen: answer_question                                                                                                                                      
 
{'debrief': {'summary': 'Isabel is finalizing preparations for her lunar expedition, focusing on recalibrating navigation systems and verifying the integrity of life support modules.',
  'recommendations': 'Proceed with thorough system tests under various conditions, conduct simulation runs of key mission phases, and confirm backup systems are operational before launch.'},
 'execution_history': [{'messages': [],
   'output': 'I am currently finalizing preparations for my lunar expedition. It involves recalibrating my navigation systems and verifying the integrity of my life support modules. Details are quite...complex.'}]}
```
```python
print(lit_team.orchestrate('which book are your team members most proud of? ask them please. '))
```  

```python
{'debrief': {'summary': "The responses provided detailed accounts of the books that the NPC team members, Gabriel Garcia Marquez and Isabel Allende, are most proud of. Gabriel highlighted 'Cien años de soledad,' while Isabel spoke of 'La Casa de los Espíritus.' Both authors expressed deep personal connections to their works, illustrating their significance in Latin American literature and their own identities.", 'recommendations': 'Encourage further engagement with each author to explore more about their literary contributions, or consider asking about themes in their works or their thoughts on current literary trends.'}, 'execution_history': [{'messages': ...}]}
```

LLM responses can be obtained without NPCs as well.

```python
from npcpy.llm_funcs import get_llm_response
response = get_llm_response("Who was the celtic Messenger god?", model='mistral:7b', provider='ollama')
print(response['response'])
```

```
The Celtic messenger god is often associated with the figure of Tylwyth Teg, also known as the Tuatha Dé Danann (meaning "the people of the goddess Danu"). However, among the various Celtic cultures, there are a few gods and goddesses that served similar roles.

One of the most well-known Celtic messengers is Brigid's servant, Líth (also spelled Lid or Lith), who was believed to be a spirit guide for messengers and travelers in Irish mythology.
```
The structure of npcpy also allows one to pass an npc
to `get_llm_response` in addition to using the NPC's wrapped method, 
allowing you to be flexible in your implementation and testing.
```python
from npcpy.npc_compiler import NPC
from npcpy.llm_funcs import get_llm_response
simon = NPC(
          name='Simon Bolivar',
          primary_directive='Liberate South America from the Spanish Royalists.',
          model='gemma3:4b',
          provider='ollama'
          )
response = get_llm_response("Who was the mythological chilean bird that guides lucky visitors to gold?", npc=simon)
print(response['response'])
```
Users are not required to pass agents to get_llm_response, so you can work with LLMs without requiring agents in each case.


`npcpy` also supports streaming responses, with the `response` key containing a generator in such cases which can be printed and processed through the print_and_process_stream method.


```python
from npcpy.npc_sysenv import print_and_process_stream
from npcpy.llm_funcs import get_llm_response
response = get_llm_response("When did the united states government begin sendinng advisors to vietnam?", model='qwen2.5:14b', provider='ollama', stream = True)

full_response = print_and_process_stream(response['response'], 'llama3.2', 'ollama')
```
Return structured outputs by specifying `format='json'` or passing a Pydantic schema. When specific formats are extracted, `npcpy`'s `get_llm_response` will convert the response from its string representation so you don't have to worry about that. 

```python
from npcpy.llm_funcs import get_llm_response
response = get_llm_response("What is the sentiment of the american people towards the repeal of Roe v Wade? Return a json object with `sentiment` as the key and a float value from -1 to 1 as the value", model='deepseek-coder', provider='deepseek', format='json')

print(response['response'])
```
```
{'sentiment': -0.7}
```

The `get_llm_response` function also can take a list of messages and will additionally return the messages with the user prompt and the assistant response appended if the response is not streamed. If it is streamed, the user must manually append the conversation result as part of their workflow if they want to then pass the messages back in.

Additionally, one can pass attachments. Here we demonstrate both
```python
from npcpy.llm_funcs import get_llm_response
messages = [{'role': 'system', 'content': 'You are an annoyed assistant.'}]

response = get_llm_response("What is the meaning of caesar salad", model='llama3.2', provider='ollama', images=['./Language_Evolution_and_Innovation_experiment.png'], messages=messages)



```
Easily create images with the generate_image function, using models available through Huggingface's diffusers library or from OpenAI or Gemini.
```python
from npcpy.llm_funcs import gen_image
image = gen_image("make a picture of the moon in the summer of marco polo", model='runwayml/stable-diffusion-v1-5', provider='diffusers')


image = gen_image("make a picture of the moon in the summer of marco polo", model='dall-e-2', provider='openai')


# edit images with 'gpt-image-1' or gemini's multimodal models, passing image paths, byte code images, or PIL instances.

image = gen_image("make a picture of the moon in the summer of marco polo", model='gpt-image-1', provider='openai', attachments=['/path/to/your/image.jpg', your_byte_code_image_here, your_PIL_image_here])


image = gen_image("edit this picture of the moon in the summer of marco polo so that it looks like it is in the winter of nishitani", model='gemini-2.0-flash', provider='gemini', attachments= [])

```

Likewise, generate videos :

```python
from npcpy.llm_funcs import gen_video
video = gen_video("make a video of the moon in the summer of marco polo", model='runwayml/stable-diffusion-v1-5', provider='diffusers')
```

## Tool Calling Examples

`npcpy` supports tool calling both with and without NPCs, allowing you to extend LLM capabilities with custom functions.

### Tool Calling without NPCs

```python
from npcpy.llm_funcs import get_llm_response
from npcpy.tools import auto_tools
import os
import subprocess

def read_file(filepath: str) -> str:
    """Read and return the contents of a file."""
    with open(filepath, 'r') as f:
        return f.read()

def write_file(filepath: str, content: str) -> str:
    """Write content to a file."""
    with open(filepath, 'w') as f:
        f.write(content)
    return f"Wrote {len(content)} characters to {filepath}"

def list_files(directory: str = ".") -> list:
    """List all files in a directory."""
    return os.listdir(directory)

def run_command(command: str) -> str:
    """Run a shell command and return the output."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout

# Auto-generate schemas from functions
tools_schema, tool_map = auto_tools([read_file, write_file, list_files, run_command])

# Use tools with natural language requests
response = get_llm_response(
    "List the files in the current directory, read the README.md file, and show me the git status",
    model='llama3.2',
    provider='ollama',
    tools=tools_schema,
    tool_map=tool_map
)

# Access raw tool data for manual processing
print("Tool calls made:")
for call in response.get('tool_calls', []):
    print(f"- Called {call['function']['name']} with {call['function']['arguments']}")

print("\nTool results:")
for result in response.get('tool_results', []):
    print(f"- {result}")

# The response object contains:
# response['tool_calls'] - List of tools the LLM decided to call
# response['tool_results'] - List of results from executing those tools
# response['messages'] - Full conversation history including tool interactions
# response['response'] - None when tools are used (no synthesized response)
```

#### Manual Tool Schema (Alternative)

If you prefer to create schemas manually:

```python
from npcpy.llm_funcs import get_llm_response
import json

def search_files(pattern: str, directory: str = ".") -> list:
    """Search for files matching a pattern."""
    import glob
    return glob.glob(f"{directory}/**/*{pattern}*", recursive=True)

def get_disk_usage(path: str = ".") -> dict:
    """Get disk usage information for a path."""
    import shutil
    total, used, free = shutil.disk_usage(path)
    return {"total_gb": total//1e9, "used_gb": used//1e9, "free_gb": free//1e9}

# Manual schema creation
tool_map = {"search_files": search_files, "get_disk_usage": get_disk_usage}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files matching a pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "File pattern to search for"},
                    "directory": {"type": "string", "description": "Directory to search in"}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_disk_usage",
            "description": "Get disk usage information",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path to check"}},
                "required": []
            }
        }
    }
]

response = get_llm_response(
    "Find all Python files in this project and check how much disk space we're using",
    model='llama3.2',
    provider='ollama',
    tools=tools_schema,
    tool_map=tool_map
)

# Process the tool results
if response.get('tool_calls'):
    print("Tools used:")
    for i, call in enumerate(response['tool_calls']):
        func_name = call['function']['name']
        args = call['function']['arguments']
        result = response['tool_results'][i]
        print(f"  {func_name}({args}) → {result}")
else:
    print("No tools were called")
    print(response.get('response', 'No response available'))
```

### Tool Calling with NPCs

```python
from npcpy.npc_compiler import NPC
from npcpy.tools import auto_tools
import requests
import os

def backup_file(filepath: str) -> str:
    """Create a backup copy of a file."""
    backup_path = f"{filepath}.backup"
    with open(filepath, 'r') as src, open(backup_path, 'w') as dst:
        dst.write(src.read())
    return f"Backed up {filepath} to {backup_path}"

def send_notification(message: str, channel: str = "general") -> str:
    """Send a notification message."""
    return f"Notification sent to #{channel}: {message}"

def check_server(url: str) -> dict:
    """Check if a server is responding."""
    try:
        response = requests.get(url, timeout=5)
        return {"url": url, "status": response.status_code, "online": True}
    except:
        return {"url": url, "status": "timeout", "online": False}

# Create assistant NPC
assistant = NPC(
    name='System Assistant',
    primary_directive='You are a helpful system assistant.', 
    model='qwen3:latest',
    provider='ollama'
)

# Auto-generate tools
tools_schema, tool_map = auto_tools([backup_file, send_notification, check_server])

# Use NPC with tools
response = assistant.get_llm_response(
    "Backup the README.md file, check if github.com is online, and send a notification about the status",
    tools=tools_schema,
    tool_map=tool_map
)

# Handle tool response data
if response.get('tool_calls'):
    print("Assistant performed these actions:")
    for i, call in enumerate(response['tool_calls']):
        func_name = call['function']['name']
        result = response['tool_results'][i]
        print(f"  ✓ {func_name}: {result}")
        
    # You can also access the full message history
    print(f"\nTotal messages in conversation: {len(response.get('messages', []))}")
else:
    print("No tools were used:", response.get('response', 'No response'))
```

### Simple Multi-Tool Example

Here's how tools work together naturally:

```python
from npcpy.llm_funcs import get_llm_response
from npcpy.tools import auto_tools
import os

def create_file(filename: str, content: str) -> str:
    """Create a new file with content."""
    with open(filename, 'w') as f:
        f.write(content)
    return f"Created {filename}"

def count_files(directory: str = ".") -> int:
    """Count files in a directory."""
    return len([f for f in os.listdir(directory) if os.path.isfile(f)])

def get_file_size(filename: str) -> str:
    """Get the size of a file."""
    size = os.path.getsize(filename)
    return f"{filename} is {size} bytes"

# Auto-generate tools
tools_schema, tool_map = auto_tools([create_file, count_files, get_file_size])

# Let the LLM use multiple tools naturally
response = get_llm_response(
    "Create a file called 'hello.txt' with the content 'Hello World!', then tell me how many files are in the current directory and what size the new file is",
    model='deepseek-reasoner',
    provider='deepseek',
    tools=tools_schema,
    tool_map=tool_map
)

# Process the multi-tool workflow results
print("Multi-tool workflow executed:")
for i, (call, result) in enumerate(zip(response.get('tool_calls', []), response.get('tool_results', []))):
    func_name = call['function']['name']
    args = call['function']['arguments']
    print(f"{i+1}. {func_name}({args}) → {result}")

# Example of building a summary from tool results
if response.get('tool_results'):
    file_created = any('Created' in str(result) for result in response['tool_results'])
    file_count = next((result for result in response['tool_results'] if isinstance(result, int)), None)
    file_size = next((result for result in response['tool_results'] if 'bytes' in str(result)), None)
    
    summary = f"Summary: File created: {file_created}, Directory has {file_count} files, {file_size}"
    print(f"\n{summary}")
```

## Understanding Tool Calling Response Structure

When using tools with `npcpy`, the response structure differs from regular LLM responses. Instead of a synthesized text response, you get structured data about the tools that were called and their results. This allows you to decide what actions to proceed with following the tool call.

### Response Object Structure

```python
response = {
    'response': None,                    # Always None when tools are used
    'tool_calls': [...],                 # List of tools the LLM decided to call
    'tool_results': [...],               # Results from executing those tools
    'messages': [...],                   # Full conversation history with tool interactions
    'usage': {...},                      # Token usage information
    'model': 'gpt-4o-mini',             # Model used
    'provider': 'openai'                 # Provider used
}
```



## Jinx Examples

Jinxs are powerful workflow templates that combine natural language processing with Python code execution. They're defined in YAML files and can be used by NPCs or called directly.

### Creating a Simple Jinx

Create a file called `data_analyzer.jinx`:

```yaml
jinx_name: "data_analyzer"
description: "Analyze CSV data and generate insights"
inputs:
  - "file_path"
  - "analysis_type"
steps:
  - name: "load_data"
    engine: "python"
    code: |
      import pandas as pd
      import numpy as np
      
      # Load the CSV file
      df = pd.read_csv('{{ file_path }}')
      print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
      
      # Store in context for next steps
      context['dataframe'] = df
      context['row_count'] = len(df)
      context['column_count'] = len(df.columns)
      
  - name: "analyze_data"
    engine: "python" 
    code: |
      df = context['dataframe']
      analysis_type = '{{ analysis_type }}'.lower()
      
      if analysis_type == 'basic':
          # Basic statistics
          stats = df.describe()
          context['statistics'] = stats.to_dict()
          output = f"Basic statistics computed for {len(df.columns)} columns"
      elif analysis_type == 'correlation':
          # Correlation analysis
          numeric_df = df.select_dtypes(include=[np.number])
          if len(numeric_df.columns) > 1:
              corr_matrix = numeric_df.corr()
              context['correlation_matrix'] = corr_matrix.to_dict()
              output = f"Correlation matrix computed for {len(numeric_df.columns)} numeric columns"
          else:
              output = "Not enough numeric columns for correlation analysis"
      else:
          output = f"Unknown analysis type: {analysis_type}"
          
  - name: "generate_report"
    engine: "natural"
    code: |
      Based on the data analysis results:
      - Dataset has {{ row_count }} rows and {{ column_count }} columns
      - Analysis type: {{ analysis_type }}
      
      {% if statistics %}
      Key statistics: {{ statistics }}
      {% endif %}
      
      {% if correlation_matrix %}
      Correlation insights: {{ correlation_matrix }}
      {% endif %}
      
      Please generate a comprehensive summary report of the key findings and insights.
```

### Using Jinx with NPCs

```python
from npcpy.npc_compiler import NPC, Jinx

# Create NPC with jinx
data_scientist = NPC(
    name='Data Scientist',
    primary_directive='You are an expert data scientist specializing in data analysis and insights.',
    jinxs=['data_analyzer'],  # Reference the jinx file
    model='llama3.2:13b',
    provider='ollama'
)

# Execute the jinx
result = data_scientist.execute_jinx(
    'data_analyzer',
    {
        'file_path': './sales_data.csv',
        'analysis_type': 'basic'
    }
)

print(result['output'])
```

### Complex Jinx with Multiple Steps

Create `research_pipeline.jinx`:

```yaml
jinx_name: "research_pipeline"
description: "Research a topic, analyze sources, and generate a report"
inputs:
  - "research_topic"
  - "output_format"
steps:
  - name: "gather_info" 
    engine: "natural"
    code: |
      Please research the topic: {{ research_topic }}
      
      Provide comprehensive information including:
      1. Key concepts and definitions
      2. Current trends and developments
      3. Major challenges or controversies
      4. Future outlook
      
      Focus on recent, credible sources and provide specific examples.
      
  - name: "analyze_findings"
    engine: "python"
    code: |
      # Extract key information from the research
      research_text = context.get('llm_response', '')
      
      # Simple analysis - count key terms
      import re
      from collections import Counter
      
      # Extract sentences and key phrases
      sentences = re.split(r'[.!?]', research_text)
      context['sentence_count'] = len([s for s in sentences if len(s.strip()) > 10])
      
      # Find common important terms (simple approach)
      words = re.findall(r'\b[A-Z][a-z]+\b', research_text)
      common_terms = Counter(words).most_common(10)
      context['key_terms'] = dict(common_terms)
      
      output = f"Analysis complete: {context['sentence_count']} sentences, top terms: {list(context['key_terms'].keys())[:5]}"
      
  - name: "format_report"
    engine: "natural" 
    code: |
      Based on the research findings about {{ research_topic }}, create a well-structured report in {{ output_format }} format.
      
      Research Summary:
      {{ llm_response }}
      
      Key Statistics:
      - Number of key points covered: {{ sentence_count }}
      - Most mentioned terms: {{ key_terms }}
      
      Please format this as a professional {{ output_format }} with:
      1. Executive Summary
      2. Main Findings  
      3. Analysis and Insights
      4. Recommendations
      5. Conclusion
      
      Ensure the content is well-organized and actionable.
```

### Using Jinxs Directly

```python
from npcpy.npc_compiler import Jinx
from npcpy.llm_funcs import get_llm_response

# Load and execute jinx directly
research_jinx = Jinx(jinx_path='./research_pipeline.jinx')

# Create a simple NPC context for execution
class SimpleNPC:
    def __init__(self):
        self.shared_context = {}
    
    def get_llm_response(self, prompt, **kwargs):
        return get_llm_response(prompt, model='deepseek-coder', provider='deepseek')

npc = SimpleNPC()

# Execute the jinx
result = research_jinx.execute(
    input_values={
        'research_topic': 'artificial intelligence in healthcare',
        'output_format': 'markdown'
    },
    jinxs_dict={'research_pipeline': research_jinx},
    npc=npc
)

print(result['output'])
```

### Team-based Jinx Usage

```python
from npcpy.npc_compiler import NPC, Team

# Create specialized NPCs for different tasks
researcher = NPC(
    name='Researcher',
    primary_directive='You are a thorough researcher who gathers comprehensive information.',
    jinxs=['research_pipeline'],
    model='gemini-2.0-flash',
    provider='gemini'
)

analyst = NPC(
    name='Data Analyst', 
    primary_directive='You are a data analyst who excels at finding patterns and insights.',
    jinxs=['data_analyzer'],
    model='claude-3-5-sonnet-latest',
    provider='anthropic'
)

writer = NPC(
    name='Technical Writer',
    primary_directive='You are a technical writer who creates clear, well-structured documents.',
    model='llama3.2',
    provider='ollama'
)

# Create team with forenpc (coordinator)
research_team = Team(
    npcs=[researcher, analyst],
    forenpc=writer
)

# Orchestrate complex workflow
result = research_team.orchestrate(
    "Research the impact of AI in education, analyze any available data, and create a comprehensive report"
)

print(result)
```

For more examples of how to use `npcpy` to simplify your LLM workflows  or to create agents or multi-agent systems, see [here](https://github.com/cagostino/npcpy/blob/main/docs/npcpy.md). `npcpy` can include images, pdfs, and csvs in its llm response generation. 


## Inference Capabilities
- `npcpy` works with local and enterprise LLM providers through its LiteLLM integration, allowing users to run inference from Ollama, LMStudio, OpenAI, Anthropic, Gemini, and Deepseek, making it a versatile tool for both simple commands and sophisticated AI-driven tasks. 

## Read the Docs

Read the docs at [npcpy.readthedocs.io](https://npcpy.readthedocs.io/en/latest/)


## NPC Studio
There is a graphical user interface that makes use of the NPC Toolkit through the NPC Studio. See the source code for NPC Studio [here](https://github.com/cagostino/npc-studio). Download the executables at [our website](https://enpisi.com/npc-studio).

## NPC Shell

The NPC shell is a suite of executable command-line programs that allow users to easily interact with NPCs and LLMs through a command line shell. 


[Try out the NPC Shell](https://github.com/npc-worldwide/npcsh)


## Mailing List
Interested to stay in the loop and to hear the latest and greatest about `npcpy`, `npcsh` and NPC Studio? Be sure to sign up for the [newsletter](https://forms.gle/n1NzQmwjsV4xv1B2A)!



## Support
If you appreciate the work here, [consider supporting NPC Worldwide with a monthly donation](https://buymeacoffee.com/npcworldwide), [buying NPC-WW themed merch](https://enpisi.com/shop), or hiring us to help you explore how to use `npcpy` and AI tools to help your business or research team, please reach out to info@npcworldwi.de .





## Enabling Innovation and Research
- `npcpy` is a framework that speeds up and simplifies the development of NLP-based or Agent-based applications and provides developers and researchers with methods to explore and test across dozens of models, providers, and personas as well as other model-level hyperparameters (e.g. `temperature`, `top_k`, etc.), incorporating an array of data sources and common tools.
- The `npcpy` agent data layer makes it easy to set up teams and serve them so you can focus more on the agent personas and less on the nitty gritty of inference.
- `npcpy` provides pioneering methods in the construction and updating of knowledge graphs as well as in the development and testing of novel mixture of agent scenarios.
- In `npcpy`, all agentic capabilities are developed and tested using small local models (like `llama3.2`, `gemma3`) to ensure it can function reliably at the edge of computing.

Check out our recent paper on the limitations of LLMs and on the quantum-like nature of natural language interpretation : [arxiv preprint](https://arxiv.org/abs/2506.10077), accepted for publication at [Quantum AI and NLP 2025](qnlp.ai)

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
pip install 'npcpy[lite]'
# if you want the full local package set up (ollama, diffusers, transformers, cuda etc.)
pip install 'npcpy[local]'
# if you want to use tts/stt
pip install 'npcpy[yap]'
# if you want everything:
pip install 'npcpy[all]'

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

### Fedora Install (under construction)

<details>  <summary> Toggle </summary>
  
```bash
python3-dev #(fixes hnswlib issues with chroma db)
xhost +  (pyautogui)
python-tkinter (pyautogui)
```

</details>


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


For cases where you wish to set up a team of NPCs, jinxs, and assembly lines, add a `npc_team` directory to your project and then initialize an NPC Team.
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


## Contributing
Contributions are welcome! Please submit issues and pull requests on the GitHub repository.


## License
This project is licensed under the MIT License.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=cagostino/npcpy&type=Date)](https://star-history.com/#cagostino/npcpy&Date)
