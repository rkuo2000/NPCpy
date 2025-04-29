(npcsh) caug@pop-os:/media/caug/extradrive1/npcww/npcsh$ guac
gLoaded .env file from /media/caug/extradrive1/npcww/npcsh
hows iloading npc team from directory
Error loading team context: 'Team' object has no attribute 'jinja_env'
filename:  guac.npc
filename:  toon.npc
filename:  parsely.npc
filename:  caug.npc
filename:  team.ctx
🥑 hows it going

# Generated python code:
print("I'm doing well, thank you! How can I assist you with Python today?")

I'm doing well, thank you! How can I assist you with Python today?

# Generated code executed successfully

🥑 ls
composition.png  docs      image.png  Makefile     mkdocs.yml  npcpy.egg-info  npcsh.code-workspace  otter_.png        pirate.png  setup.py    template_tests  tests
data             examples  LICENSE    MANIFEST.in  npcpy       npcpy.png       otter.png             output_image.png  README.md   sprite.png  test_data
🥑 cd npcpy
Changed directory to /media/caug/extradrive1/npcww/npcsh/npcpy
🥑 ls
data  __init__.py   main.py  migrations  modes            npcsh.png  npc_sysenv.py  __pycache__  sql
gen   llm_funcs.py  memory   mix         npc_compiler.py  npcs.py    npc_team       routes.py    work
🥑 run llm_funcs.py
Items added/modified from llm_funcs.py:
  subprocess: <module 'subprocess' from '/home/caug/.pyenv/versions/3.11.0/lib/python3.11/subprocess.py'>
  Generator: <function or class>
  PIL: <module 'PIL' from '/home/caug/.pyenv/versions/3.11.0/envs/npcsh/lib/python3.11/site-packages/PIL/__...
  render_markdown: <function or class>
  NPCSH_VIDEO_GEN_PROVIDER: 'diffusers'
  available_chat_models: ['gemma3', 'llama3.3', 'llama3.2', 'llama3.1', 'phi4', 'phi3.5', 'mistral', 'llama3', 'gemma', 'qwen...
  generate_image: <function or class>
  NPCSH_IMAGE_GEN_PROVIDER: 'openai'
  Union: <function or class>
  requests: <module 'requests' from '/home/caug/.pyenv/versions/3.11.0/envs/npcsh/lib/python3.11/site-packages/r...
  NPCSH_CHAT_MODEL: 'gpt-4.1-mini'
  rehash_last_message: <function or class>
  get_model_and_provider: <function or class>
  generate_video: <function or class>
  available_reasoning_models: ['o1-mini', 'o1', 'o1-preview', 'o3-mini', 'o3-preview', 'deepseek-reasoner']
  get_system_message: <function or class>
  get_llm_response: <function or class>
  Optional: <function or class>
  FileSystemLoader: <function or class>
  NPCSH_EMBEDDING_MODEL: 'nomic-embed-text'
  NPCSH_EMBEDDING_PROVIDER: 'ollama'
  NPCSH_VIDEO_GEN_MODEL: 'runwayml/stable-diffusion-v1-5'
  NPCSH_VISION_MODEL: 'gpt-4o-mini'
  NPCSH_VISION_PROVIDER: 'openai'
  sqlite3: <module 'sqlite3' from '/home/caug/.pyenv/versions/3.11.0/lib/python3.11/sqlite3/__init__.py'>
  check_llm_command: <function or class>
  NPCSH_REASONING_MODEL: 'deepseek-reasoner'
  NPCSH_API_URL: ''
  NPCSH_REASONING_PROVIDER: 'deepseek'
  execute_llm_command: <function or class>
  get_available_models: <function or class>
  lookup_provider: <function or class>
  Dict: <function or class>
  create_engine: <function or class>
  decide_plan: <function or class>
  NPCSH_CHAT_PROVIDER: 'openai'
  get_litellm_response: <function or class>
  Any: <function or class>
  Undefined: <function or class>
  NPCSH_IMAGE_GEN_MODEL: 'dall-e-2'
  Template: <function or class>
  NPCSH_DEFAULT_MODE: 'chat'
  generate_image_litellm: <function or class>
  handle_tool_call: <function or class>
  List: <function or class>
  Environment: <function or class>

🥑 out = get_llm_response('hello', model='gpt-4.1-mini', provider='openai')
🥑 print(out)
{'response': 'Hello! How can I assist you today?', 'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': [{'type': 'text', 'text': 'hello'}]}, {'role': 'assistant', 'content': 'Hello! How can I assist you today?'}], 'raw_response': ModelResponse(id='chatcmpl-BR2eVV5PUAgvyFLSZNUkbSUUG3tdI', created=1745784751, model='gpt-4.1-mini-2025-04-14', object='chat.completion', system_fingerprint='fp_38647f5e19', choices=[Choices(finish_reason='stop', index=0, message=Message(content='Hello! How can I assist you today?', role='assistant', tool_calls=None, function_call=None, provider_specific_fields={'refusal': None}, annotations=[]))], usage=Usage(completion_tokens=10, prompt_tokens=18, total_tokens=28, completion_tokens_details=CompletionTokensDetailsWrapper(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0, text_tokens=None), prompt_tokens_details=PromptTokensDetailsWrapper(audio_tokens=0, cached_tokens=0, text_tokens=None, image_tokens=None)), service_tier='default'), 'tool_calls': []}


Guac lets users execute code snippets or to ask LLMs questions which respond by generating and executing code directly within the interpreter. The variables and functions generated during these executions are inspectable to the user. In addition, `guac` is set up to provide users with a sense of cyclicality by progressing from a raw avocado (🥑) through a series of intermediaite steps until it is a gross brown mush (🥘). At this point, the user is asked to refresh, which initiates an LLM review of the session's commands and results and then suggests automations and then after the user reviews them they will be added to the user's `guac` module that is installed locally within the `~/.npcsh/guac/` folder and which eveolves as the user uses it. This refresh period is meant to encourage frequent reviews for users to help them work more efficiently and cognizantly.  
