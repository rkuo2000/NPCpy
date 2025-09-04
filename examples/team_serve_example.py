from npcpy.serve import start_flask_server
from npcpy.npc_compiler import NPC, Team
from npcpy.tools import auto_tools
from npcpy.gen.image_gen import generate_image, edit_image
from npcpy.data.web import search_web
from npcpy.data.image import capture_screenshot
from npcpy.data.load import load_csv, load_pdf, load_json, load_excel
from npcpy.work.desktop import perform_action
from npcpy.work.plan import execute_plan_command
from npcpy.work.trigger import execute_trigger_command
import pandas as pd


image_generator = NPC(
    name='Image Creator',
    primary_directive='You are a creative image generator who creates stunning visuals based on descriptions and requirements.',
    model='gpt-4o',
    provider='openai',
    tools=[generate_image]
)

image_editor = NPC(
    name='Image Editor',
    primary_directive='You are a skilled image editor who enhances and modifies images to meet specific requirements.',
    model='claude-3-5-sonnet-latest', 
    provider='anthropic',
    tools=[edit_image]
)

web_researcher = NPC(
    name='Web Researcher',
    primary_directive='You are a thorough web researcher who finds relevant information and visual references online.',
    model='deepseek-chat',
    provider='deepseek',
    tools=[search_web]
)

screenshot_specialist = NPC(
    name='Screenshot Specialist',
    primary_directive='You are a screenshot specialist who captures screen content and visual elements as needed.',
    model='llama3.2',
    provider='ollama',
    tools=[capture_screenshot]
)

desktop_automator = NPC(
    name='Desktop Automation Specialist',
    primary_directive='You are a desktop automation specialist who can control the computer interface, click, type, and perform various desktop actions.',
    model='qwen3:latest',
    provider='ollama',
    tools=[perform_action]
)

data_analyst = NPC(
    name='Data Analyst',
    primary_directive='You are a data analyst who can load, process, and analyze various data formats including CSV, Excel, PDF, and JSON files.',
    model='gemini-1.5-pro',
    provider='gemini',
    tools=[load_csv, load_pdf, load_json, load_excel]
)

workflow_planner = NPC(
    name='Workflow Planner',
    primary_directive='You are a workflow automation specialist who can create scheduled tasks and automated triggers for various system operations.',
    model='claude-3-5-sonnet-latest',
    provider='anthropic',
    tools=[execute_plan_command, execute_trigger_command]
)

multimedia_coordinator = NPC(
    name='Multimedia Director',
    primary_directive='You are a multimedia director who coordinates image creation, editing, research, data analysis, desktop automation, and workflow planning.',
    model='gemini-2.0-flash',
    provider='gemini'
)


multimedia_team = Team(
    npcs=[
        image_generator,
        image_editor,
        web_researcher, 
        screenshot_specialist,
        desktop_automator,
        data_analyst,
        workflow_planner
    ],
    forenpc=multimedia_coordinator
)


if __name__ == "__main__":
    
    import npcpy.serve as serve_module
    
    
    if not hasattr(serve_module, 'registered_teams'):
        serve_module.registered_teams = {}
    serve_module.registered_teams['multimedia_team'] = multimedia_team
    
    
    if not hasattr(serve_module, 'registered_npcs'):
        serve_module.registered_npcs = {}
    
    for npc in list(multimedia_team.npcs.values()) + [multimedia_team.forenpc]:
        serve_module.registered_npcs[npc.name] = npc
    
    print(f"Registered team 'multimedia_team' with {len(multimedia_team.npcs)} NPCs")
    print(f"Available NPCs: {[npc.name for npc in list(multimedia_team.npcs.values()) + [multimedia_team.forenpc]]}")
    
    start_flask_server(
        port=5337,
        cors_origins=["*"],  
        debug=True,
        teams={'multimedia_team': multimedia_team},  
        npcs=serve_module.registered_npcs  
    )
