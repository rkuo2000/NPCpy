
from npcpy.data.image import capture_screenshot
from typing import Any, Dict
import json     
import time
from npcpy.llm_funcs import get_llm_response
def execute_plonk_command(request, action_space, model=None, provider=None, npc=None):
    """
    Main interaction loop with LLM for action determination

    Args:
        request (str): The task to be performed
        action_space (dict): Available action types and the inputs they require
        npc (optional): NPC object for context and screenshot
        **kwargs: Additional arguments for LLM response
    """
    prompt = f"""
    Here is a request from a user:
    {request}

    Your job is to choose certain actions, take screenshots,
    and evaluate what the next step is to complete the task.

    You can choose from the following action types:
    {json.dumps(action_space)}



    Attached to the message is a screenshot of the current screen.

    Please use that information to determine the next steps.
    Your response must be a JSON with an 'actions' key containing a list of actions.
    Each action should have a 'type' and any necessary parameters.https://www.reddit.com


    For example:
        Your response should look like:

        {{
            "actions": [
                {{"type":"bash", "command":"firefox &"}},
                {{"type": "click", "x": 5, "y": 5}},
                {{'type': 'type', 'text': 'https://www.google.com'}}
                ]
                }}

    IF you have to type something, ensure that it iis first opened and selected. Do not
    begin with typing immediately.
    If you have to click, the numbers should range from 0 to 100 in x and y  with 0,0 being in the upper left.


    IF you have accomplished the task, return an empty list.
    Do not include any additional markdown formatting.
    """

    while True:
        # Capture screenshot using NPC-based method
        screenshot = capture_screenshot(npc=npc, full=True)

        # Ensure screenshot was captured successfully
        if not screenshot:
            print("Screenshot capture failed")
            return None

        # Get LLM response
        response = get_llm_response(
            prompt,
            images=[screenshot],
            model=model,
            provider=provider,
            npc=npc,
            format="json",
        )
        # print("LLM Response:", response, type(response))
        # Check if task is complete
        print(response["response"])
        if not response["response"].get("actions", []):
            return response

        # Execute actions
        for action in response["response"]["actions"]:
            print("Performing action:", action)
            action_result = perform_action(action)
            perform_action({"type": "wait", "duration": 5})

            # Optional: Add error handling or logging
            if action_result.get("status") == "error":
                print(f"Error performing action: {action_result.get('message')}")

        # Small delay between action batches
        time.sleep(1)


def test_open_reddit(npc: Any = None):
    """
    Test function to open a web browser and navigate to Reddit using plonk
    """
    # Define the action space for web navigation

    # Request to navigate to Reddit
    request = "Open a web browser and go to reddit.com"

    # Determine the browser launch hotkey based on the operating system
    import platform

    system = platform.system()

    if system == "Darwin":  # macOS
        browser_launch_keys = ["command", "space"]
        browser_search = "chrome"
    elif system == "Windows":
        browser_launch_keys = ["win", "r"]
        browser_search = "chrome"
    else:  # Linux or other
        browser_launch_keys = ["alt", "f2"]
        browser_search = "firefox"

    # Perform the task using plonk
    result = plonk(
        request,
        action_space,
        model="gpt-4o-mini",
        provider="openai",
    )

    # Optionally, you can add assertions or print results
    print("Reddit navigation test result:", result)

    return result
