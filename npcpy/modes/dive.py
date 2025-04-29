
def dive(request, 
               npcs : list = None, 
               num_npcs: int =10, 
               depth: int =5, 
               memory: int =3, 
               context: str = None,
               ):
    """
    Function Description:
        This function generates a list of NPCs based on the provided request.
    Args:
        request (str): The request for generating the NPCs.
        n_agents (int): The number of NPCs to generate.
        depth (int): The depth of the NPCs.
    Returns:
        List[NPC]: A list of generated NPCs.
    """
    # Initialize the list of NPCs
    
    if npcs is None:
        npcs = generate_random_npcs(num_npcs, model, provider)
    chains = {}
    chain_facts={}
    
    for npc in npcs: 
        chains[npc.name] = generate_research_chain(request, npc, depth, memory=memory, context=context)
        chain_facts[npc.name] =  extract_facts("\n".join(chains[npc.name]))

    chain_fact_groups = identify_groups([chain_fact for chain_fact in chain_facts.values()], model=model, provider=provider)
    consolidated_research = consolidate_research(chains, chain_facts, chain_fact_groups)
    return consolidated_research