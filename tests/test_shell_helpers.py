import pytest
import os
import sqlite3
import tempfile
from pathlib import Path
from npcpy.modes.shell_helpers import execute_command
from npcpy.memory.command_history import CommandHistory
from npcpy.npc_sysenv import (
    get_system_message,
    lookup_provider,
    NPCSH_STREAM_OUTPUT,
    get_available_tables,
)

def test_execute_slash_commands():
    """Test various slash commands"""

    result = execute_command("/help")
    assert "Available Commands" in result["output"]


def test_execute_command_with_model_override():
    """Test command execution with model override"""
    result = execute_command(
        "@gpt-4o-mini What is 2+2?",
    )
    assert result["output"] is not None


def test_execute_command_who_was_simon_bolivar():
    """Test the command for querying information about Simón Bolívar."""
    result = execute_command(
        "What country was Simon Bolivar born in?",
    )
    assert "venezuela" in str(result["output"]).lower()


def test_execute_command_capital_of_france():
    """Test the command for querying the capital of France."""
    result = execute_command("What is the capital of France?")
    assert "paris" in str(result["output"]).lower()


def test_execute_command_weather_info( ):
    """Test the command for getting weather information."""
    result = execute_command(
        "search the web for the weather in Tokyo?" 
    )
    print(result)  # Add print for debugging
    assert "tokyo" in str(result["output"]).lower()


def test_execute_command_linked_list_implementation():
    """Test the command for querying linked list implementation in Python."""
    result = execute_command(
        " Tell me a way to implement a linked list in Python?",
    )
    assert "class Node:" in str(result["output"])
    assert "class LinkedList:" in str(result["output"])


def test_execute_command_inquiry_with_npcs( ):
    """Test inquiry using NPCs."""
    result = execute_command(
        "/search -p duckduckgo who is the current us president",
    )
    assert "President" in result["output"]  # Check for presence of expected output


def test_execute_command_rag_search( ):
    """Test the command for a RAG search."""
    result = execute_command(
        "/rag -f dummy_linked_list.py linked list",
    )

    print(result)  # Print the result for debugging visibility
    # Instead of specific class search, check if it includes any relevant text
    assert (
        "Found similar texts:" in result["output"]
    )  # Check for invocation acknowledgement
    assert "linked" in result["output"].lower()  # Check for mention of linked list
