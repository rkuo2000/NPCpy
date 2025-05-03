tool_name: npcsh_executor
description: Issue npc shell requests. Uses one of the NPC macros.
inputs:
  - request to make. 
steps:
  - engine: "python"
    code: |
      {{code}}
