import os
import asyncio
import json
from contextlib import AsyncExitStack
import sys

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    print("FATAL: 'mcp' library is required. Install it (`pip install mcp`).")
    exit(1)

async def _async_get_mcp_tools_definitions(server_path: str) -> list[dict]:
    formatted_tools = []
    abs_path = os.path.abspath(server_path)
    if not (os.path.exists(abs_path) and (abs_path.endswith('.py') or abs_path.endswith('.js'))):
        return []

    command = "python" if abs_path.endswith('.py') else "node"
    params = StdioServerParameters(command=command, args=[abs_path], env=os.environ.copy())

    try:
        async with AsyncExitStack() as stack:
            transport = await asyncio.wait_for(stack.enter_async_context(stdio_client(params)), 15.0)
            session = await stack.enter_async_context(ClientSession(*transport))
            await asyncio.wait_for(session.initialize(), 10.0)
            response = await asyncio.wait_for(session.list_tools(), 10.0)

            for tool in response.tools or []:
                schema = getattr(tool, "inputSchema", {}) or {}
                formatted_tools.append({
                    "type": "function",
                    "function": {
                        "name": getattr(tool, "name", "Unknown"),
                        "description": getattr(tool, "description", ""),
                        "parameters": schema
                    }
                })
    except Exception:import os
import asyncio
import json
import sys
from contextlib import AsyncExitStack
from typing import Any, Dict

# --- MCP Imports ---
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    print("FATAL: 'mcp' library is required. Install it (`pip install mcp`).")
    exit(1)

# --- Synchronous MCP Jinx Execution Helper (Uses asyncio.run) ---

async def _async_call_mcp_tool(
    abs_server_path: str,
    tool_name: str,
    tool_args: Dict
) -> Any:
    """Async helper to connect, call a specific tool, and disconnect."""
    if not MCP_AVAILABLE:
        raise ImportError("MCP library not installed.")

    result_content = {"error": "MCP call failed to complete"} # Default error
    command = "python" if abs_server_path.endswith('.py') else "node"
    server_params = StdioServerParameters(
        command=command,
        args=[abs_server_path],
        env=os.environ.copy()
    )
    timeout_seconds = 30.0 # Timeout for the entire operation

    try:
        async with AsyncExitStack() as stack:
            async def connect_and_call():
                nonlocal result_content
                stdio_transport = await stack.enter_async_context(stdio_client(server_params))
                session = await stack.enter_async_context(ClientSession(*stdio_transport))
                await session.initialize()
                call_result = await session.call_tool(tool_name, tool_args)

                content = call_result.content
                # Basic content extraction (may need adjustment based on server)
                if isinstance(content, list) and len(content) == 1 and hasattr(content[0], 'text'):
                    content = content[0].text
                elif hasattr(content, 'text'):
                    content = content.text
                result_content = content

            await asyncio.wait_for(connect_and_call(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        result_content = {"error": f"Timeout executing MCP tool '{tool_name}'"}
    except Exception as e:
        result_content = {"error": f"Error executing MCP tool '{tool_name}': {type(e).__name__} - {e}"}

    return result_content

def execute_mcp_tool_sync(
    server_path: str,
    tool_name: str,
    tool_args: Dict
) -> Any:
    """
    Synchronously executes a single MCP tool call using asyncio.run().
    This will BLOCK the calling thread.
    """
    if not MCP_AVAILABLE:
        return {"error": "MCP library not installed."}

    abs_server_path = os.path.abspath(server_path)
    if not os.path.exists(abs_server_path):
        return {"error": f"Server path not found: {abs_server_path}"}
    if not (abs_server_path.endswith('.py') or abs_server_path.endswith('.js')):
         return {"error": f"Server path must be .py or .js: {abs_server_path}"}

    try:
        # Check if a loop is already running
        asyncio.get_running_loop()
        return {"error": "Cannot run MCP tool sync within active async context."}
    except RuntimeError:
        # No loop running, safe to proceed
        pass

    try:
        # This is the blocking call
        result = asyncio.run(_async_call_mcp_tool(abs_server_path, tool_name, tool_args))
        return result
    except Exception as e:
        return {"error": f"Failed to run MCP tool '{tool_name}' synchronously: {e}"}

# --- Example Usage ---
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <mcp_server_script_path> <tool_name> '<json_arguments>'")
        print(f"Example: python {sys.argv[0]} ../server/weather.py get_weather '{{\"location\": \"London\"}}'")
        sys.exit(1)

    server_script_path = sys.argv[1]
    tool_to_call = sys.argv[2]
    args_json_string = sys.argv[3]

    try:
        tool_arguments = json.loads(args_json_string)
        if not isinstance(tool_arguments, dict):
            raise ValueError("Arguments must be a JSON object (dictionary).")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON provided for arguments: {args_json_string}")
        sys.exit(1)
    except ValueError as e:
         print(f"Error: {e}")
         sys.exit(1)

    print(f"--- Attempting Synchronous MCP Jinx Call ---")
    print(f"Server: {server_script_path}")
    print(f"Jinx:   {tool_to_call}")
    print(f"Args:   {tool_arguments}")
    print(f"---------------------------------------------")

    # This simulates what a modified NPC.execute_tool would do for an MCP tool
    result = execute_mcp_tool_sync(server_script_path, tool_to_call, tool_arguments)

    print("\n--- Result ---")
    # Try to pretty-print if it's likely JSON, otherwise print raw
    if isinstance(result, dict) or isinstance(result, list):
        print(json.dumps(result, indent=2))
    elif isinstance(result, str):
         try: # Maybe it's a JSON string?
              print(json.dumps(json.loads(result), indent=2))
         except json.JSONDecodeError:
              print(result) # Print as plain text
    else:
        print(result) # Print other types directly
        pass
    return formatted_tools

def load_mcp_tool_definitions_sync(server_path: str) -> list[dict]:
    definitions = []
    if not MCP_AVAILABLE: return definitions
    try:
        asyncio.get_running_loop()
        # Cannot run within an active loop
        return []
    except RuntimeError:
        try:
            definitions = asyncio.run(_async_get_mcp_tools_definitions(server_path))
        except Exception:
            pass
    return definitions

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path_to_mcp_server_script.py_or_js>")
        sys.exit(1)

    mcp_server_path_arg = sys.argv[1]
    loaded_mcp_definitions = load_mcp_tool_definitions_sync(mcp_server_path_arg)

    print(json.dumps(loaded_mcp_definitions, indent=2))