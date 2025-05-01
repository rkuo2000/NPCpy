from npcpy.npc_compiler import NPC, Tool

# Load the tool
file_chat = Tool(tool_path="~/npcww/npcsh/npcpy/npc_team/tools/file_chat.tool")

# Create an NPC instance
npc = NPC(name="sibiji", primary_directive="You're an assistant focused on helping users understand their documents.", tools=[file_chat])

result = npc.execute_tool(
    "file_chat", 
    {
        "files_list": ["/home/caug/npcww/npcsh/test_data/yuan2004.pdf"]
    }
)

print(result)

