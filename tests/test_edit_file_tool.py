from npcpy.npc_compiler import NPC, Tool

# Load the tool
file_editor = Tool(tool_path="~/npcww/npcsh/npcpy/npc_team/tools/edit_file.tool")

# Create an NPC instance
npc = NPC(name="editor", primary_directive="You're a code editor assistant", tools=[file_editor])

# Execute the tool
result = npc.execute_tool(
    "file_editor", 
    {
        "file_path": "~/test_file.py",
        "edit_instructions": "Add a new function called multiply_numbers that takes two arguments and returns their product. Also modify the main section to call this new function with arguments 3 and 4, and print the result."
    }
)

print(result)


# Execute the tool
result = npc.execute_tool(
    "file_editor", 
    {
        "file_path": "~/test_file.py",
        "edit_instructions": "add a markov chain monte carlo sampler and come up with a simulation and add it to main."
    }
)

print(result)