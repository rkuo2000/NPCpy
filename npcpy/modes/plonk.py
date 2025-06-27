from npcpy.data.image import capture_screenshot
from typing import Any, Dict
import json     
import time
import platform
from npcpy.llm_funcs import get_llm_response
from npcpy.work.desktop import perform_action, action_space
import pyautogui
import subprocess

def get_system_examples():
    system = platform.system()
    if system == "Windows":
        return "Examples: start firefox, notepad, calc, explorer"
    elif system == "Darwin":
        return "Examples: open -a Firefox, open -a TextEdit, open -a Calculator"
    else:
        return "Examples: firefox &, gedit &, gnome-calculator &"

def execute_plonk_command(request, action_space, model, provider, npc=None, max_iterations=10, debug=False):
    system = platform.system()
    system_examples = get_system_examples()
    
    messages = []
    last_action_feedback = "None" 

    iteration_count = 0
    while iteration_count < max_iterations:
        if debug:
            print(f"Iteration {iteration_count + 1}/{max_iterations}")

        prompt_template = f"""
        Goal: {request}
        Feedback from last action: {last_action_feedback}

        Your task is to control the computer to achieve the goal.
        
        THOUGHT PROCESS:
        1. Analyze the screen. Is the application I need (e.g., a web browser) already open?
        2. If YES, `click` it. If NO, use `bash` to launch it. Use the examples: {system_examples}.
        
        CRITICAL COMPLETION RULE:
        Once the goal is visually complete on the screen, your ONLY next action is to use the 'quit' action.

        Your response MUST be a JSON object with an "actions" key.
        
        ---
        EXAMPLE 1: Task "Create and save a file named 'memo.txt' with the text 'Meeting at 3pm'"
        {{
          "actions": [
            {{ "type": "bash", "command": "gedit &" }},
            {{ "type": "wait", "duration": 2 }},
            {{ "type": "type", "text": "Meeting at 3pm" }},
            {{ "type": "hotkey", "keys": ["ctrl", "s"] }},
            {{ "type": "wait", "duration": 1 }},
            {{ "type": "type", "text": "memo.txt" }},
            {{ "type": "key", "keys": ["enter"] }},
            {{ "type": "quit" }}
          ]
        }}
        ---
        EXAMPLE 2: Task "Search for news about space exploration"
        {{
          "actions": [
            {{ "type": "bash", "command": "firefox &" }},
            {{ "type": "wait", "duration": 3 }},
            {{ "type": "type", "text": "news about space exploration" }},
            {{ "type": "key", "keys": ["enter"] }},
            {{ "type": "quit" }}
          ]
        }}
        ---
        """


        screenshot = capture_screenshot(npc=npc, full=True)
        if not screenshot:
            time.sleep(2)
            continue
        
        response = get_llm_response(
            prompt=prompt_template,
            model=model,
            provider=provider,
            npc=npc,
            images=[screenshot.get('file_path')],
            messages=messages,
            format="json",
        )

        if "messages" in response:
            messages = response["messages"]
        
        response_data = response.get('response')

        if not isinstance(response_data, dict) or "actions" not in response_data:
            last_action_feedback = f"Invalid JSON response from model: {response_data}"
            continue

        actions_list = response_data.get("actions", [])
        
        if not isinstance(actions_list, list):
            last_action_feedback = "Model did not return a list in the 'actions' key."
            continue

        if not actions_list:
            last_action_feedback = "No actions were returned. The task is likely not complete. Re-evaluating."
            print(last_action_feedback)
            continue
        
        for action in actions_list:
            if debug:
                print(f"Executing action: {action}")
            if action.get("type") == "quit":
                print("Task complete: Model returned 'quit' action.")
                return "SUCCESS"
                
            result = perform_action(action)
            last_action_feedback = result.get("message") or result.get("output")
            if result.get("status") == "error":
                print(f"Action failed, providing feedback to model: {last_action_feedback}")
                break 
            time.sleep(1)
        
        iteration_count += 1
    
    return None





def main():
    
    tests = [
        "Open a web browser and find a stock price for apple inc ",
        "Open calculator and calculate 25 * 43", 
        "Open a text editor and write 'Hello World'",
    ]
    
    for i, test in enumerate(tests, 1):
        print(f"Test {i}: {test}")
        print("-" * 50)
        
        execute_plonk_command(
            request=test,
            action_space=action_space,
            model="gpt-4o-mini",
            provider="openai",
            max_iterations=8,
            debug=True
        )
        
        time.sleep(5)

if __name__ == "__main__":
    main()