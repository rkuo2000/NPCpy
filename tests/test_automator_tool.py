from npcpy.npc_compiler import NPC, Jinx

# Load the jinx
automator = Jinx(jinx_path="~/npcww/npcsh/npcpy/npc_team/jinxs/automator.jinx")

# Create an NPC instance
npc = NPC(name="sibiji", 
          primary_directive="You're an assistant focused on helping users understand their documents.", 
          jinxs=[automator])

result = npc.execute_jinx(
    "automator", 
    {
        "request": "any time a new download appears, open the downloads folder",
        "type": "trigger"
    }
)

print(result)

