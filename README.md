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

```
from npcpy.npc_compiler import NPC
simon = NPC(
          name='Simon Bolivar',
          primary_directive='Liberate South America from the Spanish Royalists.',
          model='gemma3',
          provider='ollama'
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
```
print(lit_team.orchestrate('which book are your team members most proud of? ask them please. '))
```
```
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

LLM responses can be obtained without NPCs as well.

```
from npcpy.llm_funcs import get_llm_response
response = get_llm_response("Who was the celtic Messenger god?", model='llama3.2', provider='ollama')
print(response['response'])
```

```
The Celtic messenger god is often associated with the figure of Tylwyth Teg, also known as the Tuatha Dé Danann (meaning "the people of the goddess Danu"). However, among the various Celtic cultures, there are a few gods and goddesses that served similar roles.

One of the most well-known Celtic messengers is Brigid's servant, Líth (also spelled Lid or Lith), who was believed to be a spirit guide for messengers and travelers in Irish mythology.
```
The structure of npcpy also allows one to pass an npc
to `get_llm_response` in addition to using the NPC's wrapped method, 
allowing you to be flexible in your implementation and testing.
```
from npcpy.npc_compiler import NPC
from npcpy.llm_funcs import get_llm_response
simon = NPC(
          name='Simon Bolivar',
          primary_directive='Liberate South America from the Spanish Royalists.',
          model='qwen3',
          provider='ollama'
          )
response = get_llm_response("Who was the mythological chilean bird that guides lucky visitors to gold?", npc=simon)
print(response['response'])
```
Users are not required to pass agents to get_llm_response, so you can work with LLMs without requiring agents in each case.


`npcpy` also supports streaming responses, with the `response` key containing a generator in such cases which can be printed and processed through the print_and_process_stream method.


```
from npcpy.npc_sysenv import print_and_process_stream
from npcpy.llm_funcs import get_llm_response
response = get_llm_response("When did the united states government begin sendinng advisors to vietnam?", model='llama3.2', provider='ollama', stream = True)

full_response = print_and_process_stream(response['response'], 'llama3.2', 'ollama')
```
Return structured outputs by specifying `format='json'` or passing a Pydantic schema. When specific formats are extracted, `npcpy`'s `get_llm_response` will convert the response from its string representation so you don't have to worry about that. 

```
from npcpy.llm_funcs import get_llm_response
response = get_llm_response("What is the sentiment of the american people towards the repeal of Roe v Wade? Return a json object with `sentiment` as the key and a float value from -1 to 1 as the value", model='gemma3:1b', provider='ollama', format='json')

print(response['response'])
```
```
{'sentiment': -0.7}
```

The `get_llm_response` function also can take a list of messages and will additionally return the messages with the user prompt and the assistant response appended if the response is not streamed. If it is streamed, the user must manually append the conversation result as part of their workflow if they want to then pass the messages back in.

Additionally, one can pass attachments. Here we demonstrate both
```
from npcpy.llm_funcs import get_llm_response
messages = [{'role': 'system', 'content': 'You are an annoyed assistant.'}]

response = get_llm_response("What is the meaning of caesar salad", model='gpt-4o-mini', provider='openai', images=['./Language_Evolution_and_Innovation_experiment.png'], messages=messages)



```
Easily create images with the generate_image function, using models available through Huggingface's diffusers library or from OpenAI or Gemini.
```
from npcpy.llm_funcs import gen_image
image = gen_image("make a picture of the moon in the summer of marco polo", model='runwayml/stable-diffusion-v1-5', provider='diffusers')


image = gen_image("make a picture of the moon in the summer of marco polo", model='dall-e-2', provider='openai')


# edit images with 'gpt-image-1' or gemini's multimodal models, passing image paths, byte code images, or PIL instances.

image = gen_image("make a picture of the moon in the summer of marco polo", model='gpt-image-1', provider='openai', attachments=['/path/to/your/image.jpg', your_byte_code_image_here, your_PIL_image_here])


image = gen_image("edit this picture of the moon in the summer of marco polo so that it looks like it is in the winter of nishitani", model='gemini-2.0-flash', provider='gemini', attachments= [])

```

Likewise, generate videos :

```
from npcpy.llm_funcs import gen_video
video = gen_video("make a video of the moon in the summer of marco polo", model='runwayml/stable-diffusion-v1-5', provider='diffusers')
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
