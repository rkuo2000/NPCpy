#!/usr/bin/env python3
import os
import platform
import subprocess
import sys

def build_executable():
    # Path to your npcsh.py
    script_path = os.path.join('npcpy', 'modes', 'npcsh.py')
    
    if not os.path.exists(script_path):
        print(f"Error: Could not find {script_path}")
        sys.exit(1)

    # Determine platform-specific executable name
    system = platform.system().lower()
    if system == 'windows':
        exe_name = 'npcsh.exe'
    else:
        exe_name = 'npcsh'

    # PyInstaller command
    cmd = [
        'pyinstaller',
        '--onefile',
        '--name', exe_name,
        '--distpath', './dist',
        '--workpath', './build',
        '--specpath', './',
        '--clean',
        script_path
    ]

    print(f"Building {exe_name}...")
    print(" ".join(cmd))
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\nSuccess! Executable created in ./dist/{exe_name}")
    except subprocess.CalledProcessError as e:
        print(f"\nBuild failed with error: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("\nPyInstaller not found. Install it with: pip install pyinstaller")
        sys.exit(1)

if __name__ == '__main__':
    build_executable()