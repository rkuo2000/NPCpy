

action_space = {
        "click": {"x": "int (0-100)", "y": "int (0-100)", "button": "left|right|middle (optional)"},
        "type": {"text": "string"},
        "key": {"keys": "list of key names or a single key string"},
        "scroll": {"direction": "up|down", "amount": "int (optional)"},
        "wait": {"duration": "float seconds"},
        "bash": {"command": "string"},
        "hotkey": {"keys": "list of keys to press simultaneously"},
        "drag": {"start_x": "int (0-100)", "start_y": "int (0-100)", "end_x": "int (0-100)", "end_y": "int (0-100)"},
        "quit": {"description": "Use this action when the main goal is 100% complete."},
    }
import pyautogui
import time
import subprocess
def perform_action(action):
    action_type = action.get("type")
    print(f"Action received: {action}")

    if action_type == "bash":
        command = action.get("command", "")
        print(f"Executing bash command: {command}")
        subprocess.Popen(command, shell=True)
        return {"status": "success", "output": f"Launched '{command}'."}
    
    try:
        if action_type == "click":
            width, height = pyautogui.size()
            x = int(action.get("x", 0) * width / 100)
            y = int(action.get("y", 0) * height / 100)
            print(f"Clicking at ({x}, {y})")
            pyautogui.click(x=x, y=y)
        
        elif action_type == "type":
            text_to_type = action.get("text", "")
            print(f"Typing text: {text_to_type}")
            pyautogui.typewrite(text_to_type)

        elif action_type == "key":
            keys_to_press = action.get("keys") or action.get("key", [])
            print(f"Pressing key(s): {keys_to_press}")
            if isinstance(keys_to_press, str):
                keys_to_press = [keys_to_press]
            for k in keys_to_press:
                pyautogui.press(k.lower())

        elif action_type == "wait":
            duration = action.get("duration", 1)
            print(f"Waiting for {duration} seconds")
            time.sleep(duration)
        
        else:
            return {"status": "error", "message": f"Unknown or malformed action: {action}"}

        return {"status": "success", "output": "Action performed."}
    except Exception as e:
        return {"status": "error", "message": str(e)}
