from npcpy.npc_compiler import NPC
simon = NPC(
          name='Simon Bolivar',
          primary_directive='Liberate South America from the Spanish Royalists.',
          model='gemini-2.5-flash',
          provider='gemini'
          )
response = simon.get_llm_response("What is the most important territory to retain in the Andes mountains?")
print(response['response'])

