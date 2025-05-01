tool_name: npcsh_executor
description: Issue npc shell requests. Uses one of the NPC macros.
inputs:
  - request 
  - type # one of plan or trigger
steps:
  - engine: "python"
    code: |
        type = '{{type}}'
        request = '{{request}}'
        if type == 'plan':
            from npcpy.work.plan import execute_plan_command 
            output = execute_plan_command(request)
        elif type == 'trigger':
            from npcpy.work.trigger import execute_trigger_command
            output = execute_trigger_command(request)
        else:
            raise ValueError("Invalid type. Must be 'plan' or 'trigger'.")
