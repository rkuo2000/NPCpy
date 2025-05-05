import os
import asyncio
import json
import sys
from contextlib import AsyncExitStack
from typing import Any, Dict, List

# --- MCP Imports ---
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    print("FATAL: 'mcp' library is required. Install it (`pip install mcp`).")
    exit(1)

# --- Synchronous MCP Tool Execution Helper (Uses asyncio.run) ---

async def _async_call_mcp_tool(
    abs_server_path: str,
    tool_name: str,
    tool_args: Dict,
    debug: bool = True # Add debug flag back
) -> Any:
    if not MCP_AVAILABLE:
        raise ImportError("MCP library not installed.")

    result_content: Any = {"error": "MCP call failed to complete"} # Default error
    server_name = os.path.basename(abs_server_path) # For logging

    # Define log helper inside
    def _log(msg):
        if debug: print(f"[_async_call_mcp: {server_name}/{tool_name}] {msg}")

    _log(f"Attempting to connect to {abs_server_path}...")
    command = "python" if abs_server_path.endswith('.py') else "node"
    server_params = StdioServerParameters(
        command=command,
        args=[abs_server_path],
        env=os.environ.copy()
    )
    timeout_seconds = 30.0

    try:
        async with AsyncExitStack() as stack:
            _log(f"Awaiting connect_and_call() with timeout {timeout_seconds}s...")
            async def connect_and_call():
                nonlocal result_content
                _log(f"Entering stdio_client context...")
                stdio_transport = await stack.enter_async_context(stdio_client(server_params))
                _log(f"Entering ClientSession context...")
                session = await stack.enter_async_context(ClientSession(*stdio_transport))
                _log(f"Awaiting session.initialize()...")
                await session.initialize()
                _log(f"Session initialized. Awaiting session.call_tool({tool_name}, {tool_args})...")
                call_result = await session.call_tool(tool_name, tool_args)
                _log(f"session.call_tool completed. Raw result: {call_result}") # Log raw result

                content = call_result.content
                _log(f"Extracted content type: {type(content)}")

                # --- Corrected Content Handling ---
                if isinstance(content, list) and content and all(hasattr(item, 'text') for item in content):
                    result_content = [item.text for item in content]
                    _log(f"Processed list of TextContent: {result_content}")
                elif isinstance(content, list) and len(content) == 1 and hasattr(content[0], 'text'):
                     result_content = content[0].text
                     _log(f"Processed single TextContent in list: {result_content!r}")
                elif hasattr(content, 'text'):
                    result_content = content.text
                    _log(f"Processed direct TextContent: {result_content!r}")
                else:
                    result_content = content
                    _log(f"Using content directly (not TextContent): {str(result_content)[:200]}...")
                # --- End Corrected Content Handling ---

            await asyncio.wait_for(connect_and_call(), timeout=timeout_seconds)
            _log(f"connect_and_call() finished successfully.")

    except asyncio.TimeoutError:
        _log(f"Timeout Error!")
        result_content = {"error": f"Timeout executing MCP tool '{tool_name}'"}
    except Exception as e:
        _log(f"Exception: {type(e).__name__} - {e}")
        # import traceback # Optional for more detail
        # traceback.print_exc() # Optional
        result_content = {"error": f"Error executing MCP tool '{tool_name}': {type(e).__name__} - {e}"}

    return result_content

# --- execute_mcp_tool_sync (Passes debug flag) ---
def execute_mcp_tool_sync(
    server_path: str,
    tool_name: str,
    tool_args: Dict,
    debug: bool = True # Add debug flag
) -> Any:
    if not MCP_AVAILABLE:
        return {"error": "MCP library not installed."}

    abs_server_path = os.path.abspath(server_path)
    if not os.path.exists(abs_server_path):
        return {"error": f"Server path not found: {abs_server_path}"}
    if not (abs_server_path.endswith('.py') or abs_server_path.endswith('.js')):
         return {"error": f"Server path must be .py or .js: {abs_server_path}"}

    try:
        asyncio.get_running_loop()
        if debug: print("[execute_mcp_tool_sync] Error: Cannot run sync within active async loop.")
        return {"error": "Cannot run MCP tool sync within active async context."}
    except RuntimeError:
        pass # No loop running

    if debug: print(f"[execute_mcp_tool_sync] Calling asyncio.run for {tool_name} on {os.path.basename(abs_server_path)}...")
    try:
        # Pass debug flag down
        result = asyncio.run(_async_call_mcp_tool(abs_server_path, tool_name, tool_args, debug))
        if debug: print(f"[execute_mcp_tool_sync] asyncio.run completed.")
        return result
    except Exception as e:
        if debug: print(f"[execute_mcp_tool_sync] Error during asyncio.run: {e}")
        return {"error": f"Failed to run MCP tool '{tool_name}' synchronously: {e}"}


# --- Example Usage (Passes debug=True) ---
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <mcp_server_script_path> <tool_name> '<json_arguments>'")
        print(f"Example using your server: python {sys.argv[0]} ../npcpy/work/mcp_server.py get_available_tables '{{\"db_path\": \"~/npcsh_history.db\"}}'")
        sys.exit(1)

    server_script_path = sys.argv[1]
    tool_to_call = sys.argv[2]
    args_json_string = sys.argv[3]

    try:
        tool_arguments = json.loads(args_json_string)
        if not isinstance(tool_arguments, dict):
            raise ValueError("Arguments must be a JSON object.")
    except Exception as e:
        print(f"Error parsing arguments JSON: {e}")
        sys.exit(1)

    print(f"--- Attempting Synchronous MCP Tool Call ---")
    print(f"Server: {server_script_path}")
    print(f"Tool:   {tool_to_call}")
    print(f"Args:   {tool_arguments}")
    print(f"---------------------------------------------")

    # Execute with debug=True to see internal logs
    result = execute_mcp_tool_sync(server_script_path, tool_to_call, tool_arguments, debug=True)

    # --- Result Printing (Remains the same) ---
    print("\n--- Result ---")
    print(f"Type: {type(result)}")
    print("Content:")
    if isinstance(result, (dict, list)):
        print(json.dumps(result, indent=2))
    elif isinstance(result, str):
        print(repr(result))
    else:
        print(result)