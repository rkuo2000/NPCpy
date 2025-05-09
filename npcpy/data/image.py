
import time
import platform
import subprocess
from typing import Dict, Any
import os
import io
from PIL import Image


def _windows_snip_to_file(file_path: str) -> bool:
    """Helper function to trigger Windows snipping and save to file."""
    try:
        # Import Windows-specific modules only when needed
        import win32clipboard
        from PIL import ImageGrab
        from ctypes import windll

        # Simulate Windows + Shift + S
        windll.user32.keybd_event(0x5B, 0, 0, 0)  # WIN down
        windll.user32.keybd_event(0x10, 0, 0, 0)  # SHIFT down
        windll.user32.keybd_event(0x53, 0, 0, 0)  # S down
        windll.user32.keybd_event(0x53, 0, 0x0002, 0)  # S up
        windll.user32.keybd_event(0x10, 0, 0x0002, 0)  # SHIFT up
        windll.user32.keybd_event(0x5B, 0, 0x0002, 0)  # WIN up

        # Wait for user to complete the snip
        print("Please select an area to capture...")
        time.sleep(1)  # Give a moment for snipping jinx to start

        # Keep checking clipboard for new image
        max_wait = 30  # Maximum seconds to wait
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                image = ImageGrab.grabclipboard()
                if image:
                    image.save(file_path, "PNG")
                    return True
            except Exception:
                pass
            time.sleep(0.5)

        return False

    except ImportError:
        print("Required packages not found. Please install: pip install pywin32 Pillow")
        return False


def capture_screenshot(npc: Any = None, full=False) -> Dict[str, str]:
    """
    Function Description:
        This function captures a screenshot of the current screen and saves it to a file.
    Args:
        npc: The NPC object representing the current NPC.
        full: Boolean to determine if full screen capture is needed
    Returns:
        A dictionary containing the filename, file path, and model kwargs.
    """
    # Ensure the directory exists
    directory = os.path.expanduser("~/.npcsh/screenshots")
    os.makedirs(directory, exist_ok=True)

    # Generate a unique filename
    filename = f"screenshot_{int(time.time())}.png"
    file_path = os.path.join(directory, filename)

    system = platform.system()
    model_kwargs = {}

    if npc is not None:
        if npc.provider is not None:
            model_kwargs["provider"] = npc.provider
        if npc.model is not None:
            model_kwargs["model"] = npc.model

    if full:
        if system == "Darwin":
            subprocess.run(["screencapture", file_path])
        elif system == "Linux":
            if (
                subprocess.run(
                    ["which", "gnome-screenshot"], capture_output=True
                ).returncode
                == 0
            ):
                subprocess.Popen(["gnome-screenshot", "-f", file_path])
                while not os.path.exists(file_path):
                    time.sleep(0.5)
            elif (
                subprocess.run(["which", "scrot"], capture_output=True).returncode == 0
            ):
                subprocess.Popen(["scrot", file_path])
                while not os.path.exists(file_path):
                    time.sleep(0.5)
            else:
                print(
                    "No supported screenshot jinx found. Please install gnome-screenshot or scrot."
                )
                return None
        elif system == "Windows":
            # For full screen on Windows, we'll use a different approach
            try:
                import win32gui
                import win32ui
                import win32con
                from PIL import Image

                # Get screen dimensions
                width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
                height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)

                # Create device context
                hdesktop = win32gui.GetDesktopWindow()
                desktop_dc = win32gui.GetWindowDC(hdesktop)
                img_dc = win32ui.CreateDCFromHandle(desktop_dc)
                mem_dc = img_dc.CreateCompatibleDC()

                # Create bitmap
                screenshot = win32ui.CreateBitmap()
                screenshot.CreateCompatibleBitmap(img_dc, width, height)
                mem_dc.SelectObject(screenshot)
                mem_dc.BitBlt((0, 0), (width, height), img_dc, (0, 0), win32con.SRCCOPY)

                # Save
                screenshot.SaveBitmapFile(mem_dc, file_path)

                # Cleanup
                mem_dc.DeleteDC()
                win32gui.DeleteObject(screenshot.GetHandle())

            except ImportError:
                print(
                    "Required packages not found. Please install: pip install pywin32"
                )
                return None
        else:
            print(f"Unsupported operating system: {system}")
            return None
    else:
        if system == "Darwin":
            subprocess.run(["screencapture", "-i", file_path])
        elif system == "Linux":
            if (
                subprocess.run(
                    ["which", "gnome-screenshot"], capture_output=True
                ).returncode
                == 0
            ):
                subprocess.Popen(["gnome-screenshot", "-a", "-f", file_path])
                while not os.path.exists(file_path):
                    time.sleep(0.5)
            elif (
                subprocess.run(["which", "scrot"], capture_output=True).returncode == 0
            ):
                subprocess.Popen(["scrot", "-s", file_path])
                while not os.path.exists(file_path):
                    time.sleep(0.5)
            else:
                print(
                    "No supported screenshot jinx found. Please install gnome-screenshot or scrot."
                )
                return None
        elif system == "Windows":
            success = _windows_snip_to_file(file_path)
            if not success:
                print("Screenshot capture failed or timed out.")
                return None
        else:
            print(f"Unsupported operating system: {system}")
            return None

    # Check if screenshot was successfully saved
    if os.path.exists(file_path):
        print(f"Screenshot saved to: {file_path}")
        return {
            "filename": filename,
            "file_path": file_path,
            "model_kwargs": model_kwargs,
        }
    else:
        print("Screenshot capture failed or was cancelled.")
        return None

def compress_image(image_bytes, max_size=(800, 600)):
    # Create a copy of the bytes in memory
    buffer = io.BytesIO(image_bytes)
    img = Image.open(buffer)

    # Force loading of image data
    img.load()

    # Convert RGBA to RGB if necessary
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background

    # Resize if needed
    if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
        img.thumbnail(max_size)

    # Save with minimal compression
    out_buffer = io.BytesIO()
    img.save(out_buffer, format="JPEG", quality=95, optimize=False)
    return out_buffer.getvalue()

