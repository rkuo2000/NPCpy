

action_space = {
    "hotkey": {"key": "string"},  # For pressing hotkeys
    "click": {
        "x": "int between 0 and 100",
        "y": "int between 0 and 100",
    },  # For clicking
    "drag": {
        "x": "int between 0 and 100",
        "y": "int between 0 and 100",
        "duration": "int",
    },  # For dragging
    "wait": {"duration": "int"},  # For waiting
    "type": {"text": "string"},
    "right_click": {"x": "int between 0 and 100", "y": "int between 0 and 100"},
    "double_click": {"x": "int between 0 and 100", "y": "int between 0 and 100"},
    "bash": {"command": "string"},
}
def perform_action(action):
    """
    Execute different types of actions using PyAutoGUI
    """
    try:
        pyautogui.PAUSE = 1  # Add a small pause between actions
        pyautogui.FAILSAFE = (
            True  # Enable fail-safe to stop script by moving mouse to corner
        )

        print(f"Action received: {action}")  # Debug print

        if action["type"] == "click":
            pyautogui.click(x=action.get("x"), y=action.get("y"))

        elif action["type"] == "double_click":
            pyautogui.doubleClick(x=action.get("x"), y=action.get("y"))

        elif action["type"] == "right_click":
            pyautogui.rightClick(x=action.get("x"), y=action.get("y"))

        elif action["type"] == "drag":
            pyautogui.dragTo(
                x=action.get("x"), y=action.get("y"), duration=action.get("duration", 1)
            )

        elif action["type"] == "type":
            text = action.get("text", "")
            if isinstance(text, dict):
                text = text.get("text", "")
            pyautogui.typewrite(text)

        elif action["type"] == "hotkey":
            keys = action.get("text", "")
            print(f"Hotkey action: {keys}")  # Debug print
            if isinstance(keys, str):
                keys = [keys]
            elif isinstance(keys, dict):
                keys = [keys.get("key", "")]
            pyautogui.hotkey(*keys)

        elif action["type"] == "wait":
            time.sleep(action.get("duration", 1))  # Wait for the given time in seconds

        elif action["type"] == "bash":
            command = action.get("command", "")
            print(f"Running bash command: {command}")  # Debug print
            subprocess.Popen(
                command, shell=True
            )  # Run the command without waiting for it to complete
            print(f"Bash Command Output: {result.stdout.decode()}")  # Debug output
            print(f"Bash Command Error: {result.stderr.decode()}")  # Debug error

        return {"status": "success"}

    except Exception as e:
        return {"status": "error", "message": str(e)}
    