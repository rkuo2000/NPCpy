from setuptools import setup, find_packages

from pathlib import Path
import os


def package_files(directory):
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths



# Base requirements (no LLM packages)
base_requirements = [
    "jinja2",
    "litellm",    
    "scipy",
    "numpy",
    "requests",
    "matplotlib",
    "markdown",
    "networkx", 
    "PyYAML",
    "PyMuPDF",
    "pyautogui",
    "pydantic", 
    "pygments",
    "sqlalchemy",
    "termcolor",
    "rich",
    "colorama",
    "Pillow",
    "python-dotenv",
    "pandas",
    "beautifulsoup4",
    "duckduckgo-search",
    "flask",
    "flask_cors",
    "redis",
    "psycopg2-binary",
    "flask_sse",
]

# API integration requirements
api_requirements = [
    "anthropic",
    "openai",
    "google-generativeai",
    "google-genai",
]

# mcp integration requirements
mcp_requirements = [
    "mcp",
]
# Local ML/AI requirements
local_requirements = [
    "sentence_transformers",
    "opencv-python",
    "ollama",
    "kuzu",
    "chromadb",
    "diffusers",
    "nltk",
    "torch",
]

# Voice/Audio requirements
voice_requirements = [
    "pyaudio",
    "gtts",
    "playsound==1.2.2",
    "pygame", 
    "faster_whisper",
    "pyttsx3",
]

extra_files = package_files("npcpy/npc_team/")

setup(
    name="npcpy",
    version="1.0.31",
    packages=find_packages(exclude=["tests*"]),
    install_requires=base_requirements,  # Only install base requirements by default
    extras_require={
        "lite": api_requirements,  # Just API integrations
        "local": local_requirements,  # Local AI/ML features
        "yap": voice_requirements,  # Voice/Audio features
        "mcp": mcp_requirements,  # MCP integration
        "all": api_requirements + local_requirements + voice_requirements + mcp_requirements,  # Everything
    },
    author="Christopher Agostino",
    author_email="info@npcworldwi.de",
    description="npcpy is the premier open-source library for integrating LLMs and Agents into python systems.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/NPC-Worldwide/npcpy",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    include_package_data=True,
    python_requires=">=3.10",
)

