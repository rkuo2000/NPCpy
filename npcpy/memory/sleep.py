#from sympy import Q
from npcpy.llm_funcs import get_llm_response

    
def sleep():
    """
    Sleep for a short duration to allow the system to process other tasks.
    """
    # load in the existing knowledge graph
    # try to update it using the more recent information.
    # knowledge table will will just append memories 
    # each memory will be either a fact, a mistake, or a lesson learned, essentially representing neutral, negative, and positive memories.
    # the knowledge graph gets adjusted by looking at the groupings and deciding whether or not they might be
    # better off combined or separated or refactored. 
    pass
def forget():
    """
    Forget the current task and reset the state.
    """
    # once the knowledge graph gets sufficiently big, we will want to have a way to 
    # prune it without being overly prescriptive.
    pass
