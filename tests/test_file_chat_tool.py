from npcpy.npc_compiler import NPC, Jinx

# Load the jinx
file_chat = Jinx(jinx_path="~/npcww/npcsh/npcpy/npc_team/jinxs/file_chat.jinx")

# Create an NPC instance
npc = NPC(name="sibiji", primary_directive="You're an assistant focused on helping users understand their documents.", jinxs=[file_chat])

result = npc.execute_jinx(
    "file_chat", 
    {
        "files_list": ["/home/caug/npcww/npcsh/test_data/yuan2004.pdf"]
    }
)

print(result)

