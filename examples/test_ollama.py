from npcpy.npc_compiler import NPC
simon = NPC(
          name='Simon Bolivar',
          primary_directive='Liberate South America from the Spanish Royalists.',
          model='gpt-oss',
          provider='ollama'
          )
#response = simon.get_llm_response("What is the most important territory to retain in the Andes mountains?")
response = simon.get_llm_response("What is your name? and Where do you live?")
print(response['response'])

