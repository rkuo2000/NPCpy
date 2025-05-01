tool_name: pdf_chat
description: Execute queries on the ~/npcsh_history.db to pull data. The database contains only information about conversations and other user-provided data. It does not store any information about individual files.
inputs:
  - files_list:  # list of files 
steps:
  - engine: python
    code: |
      from npcpy.modes.spool import enter_spool_mode
      files_list = {{files_list}}
      output = enter_spool_mode(
        files = files_list
      )



