from npcpy.llm_funcs import get_llm_response 

gemini_test = get_llm_response('who is the moon', 
                               model='gemini-2.0-flash', 
                               provider='gemini')
