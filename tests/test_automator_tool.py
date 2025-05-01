from npcpy.npc_compiler import NPC, Tool

# Load the tool
automator = Tool(tool_path="~/npcww/npcsh/npcpy/npc_team/tools/automator.tool")

# Create an NPC instance
npc = NPC(name="sibiji", 
          primary_directive="You're an assistant focused on helping users understand their documents.", 
          tools=[automator])

result = npc.execute_tool(
    "automator", 
    {
        "request": "any time a new download appears, open the downloads folder",
        "type": "trigger"
    }
)

print(result)

