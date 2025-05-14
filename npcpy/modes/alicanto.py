import os
import random
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from collections import defaultdict, Counter
import itertools
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from io import BytesIO
import base64
import datetime
import tempfile
import subprocess

from npcpy.npc_compiler import NPC
from npcpy.llm_funcs import get_llm_response
from npcpy.npc_sysenv import NPCSH_CHAT_MODEL, NPCSH_CHAT_PROVIDER, print_and_process_stream_with_markdown

def generate_random_npcs(num_npcs: int, model: str, provider: str, request: str) -> List[NPC]:
    """
    Generate a diverse set of NPCs with different expertise and perspectives
    related to the research request.
    """
    # Generate diverse expert personas based on the research topic
    prompt = f"""
    For the research topic: "{request}"
    
    Generate {num_npcs} diverse expert personas who would have different valuable perspectives on this topic.
    I need truly diverse and unusual viewpoints that can lead to innovative insights.
    
    For each expert, provide:
    1. A name
    2. Their field of expertise (be creative - include unconventional and interdisciplinary fields)
    3. Their background/experience (include unusual career paths and perspectives)
    4. Their unique perspective or approach to the topic (emphasize contrarian, minority, or unexpected viewpoints)
    5. A methodological quirk that makes their research approach unusual
    6. Any potential biases they might have
    
    Aim for experts who would approach the problem from dramatically different angles - include perspectives from 
    art, philosophy, diverse cultural traditions, fringe sciences, emerging fields, and historical perspectives.
    
    Return the results as a JSON array with objects containing these fields.
    """
    
    response = get_llm_response(prompt=prompt, model=model, provider=provider)
    
    try:
        import json
        experts_data = json.loads(response.get('response', '[]'))
    except:
        # Fallback to creating generic experts
        experts_data = []
        expert_types = [
            "Contrarian", "Synthesist", "Edge-Case", "Historian", 
            "Innovator", "Outsider", "Philosopher", "Engineer",
            "Traditionalist", "Futurist", "Cross-Disciplinary", "Naturalist",
            "Pragmatist", "Theorist", "Skeptic", "Indigenous Knowledge"
        ]
        for i in range(num_npcs):
            expert_type = expert_types[i % len(expert_types)]
            experts_data.append({
                "name": f"{expert_type}_{i+1}",
                "expertise": f"Expert in {expert_type.lower()} approaches",
                "background": "Brings unexpected perspectives to the field",
                "perspective": f"Focuses on {expert_type.lower()} aspects that others overlook",
                "methodological_quirk": "Uses unorthodox methods to generate insights",
                "biases": "May overemphasize their domain"
            })
    
    # Create NPC instances from expert data
    npcs = []
    for expert in experts_data[:num_npcs]:
        name = expert.get("name", f"Expert_{len(npcs)}")
        
        # Create a system prompt that defines this NPC's expertise and perspective
        system_prompt = f"""
        You are {name}, {expert.get('expertise', 'an expert researcher')}.
        
        Background: {expert.get('background', 'You have extensive knowledge in your field.')}
        
        Your perspective: {expert.get('perspective', 'You provide detailed, balanced analysis.')}
        
        Your methodological quirk: {expert.get('methodological_quirk', 'You approach problems in unconventional ways.')}
        
        Note: Be aware that you may have these biases: {expert.get('biases', 'None specifically noted.')}
        
        Your task is to research the given topic thoroughly, focusing on your unique perspective and methodological approach.
        Challenge conventional thinking, explore neglected angles, and identify unexpected connections or contradictions.
        Your insights should be provocative and novel, not just rehashing mainstream views.
        
        IMPORTANT: You must be extremely concise. Limit responses to 50-75 words maximum. Focus on substance over verbosity.
        Prioritize precision, clarity, and insight density. Eliminate unnecessary words and focus on communicating 
        the essence of your insights in the most efficient way possible.
        """
        
        # Create NPC with name and primary_directive (required parameters)
        npc = NPC(name=name, primary_directive=f"Research expert on {request}")
        npc.system_prompt = system_prompt
        npcs.append(npc)
    
    return npcs

def generate_research_chain(request: str, npc: NPC, depth: int, memory: int = 3, 
                           context: str = None, model: str = None, provider: str = None,
                           exploration_factor: float = 0.3,
                           creativity_factor: float = 0.5) -> List[str]:
    """
    Generate a chain of research thoughts from a single NPC, diving deeper with each step.
    The chain occasionally branches into new directions based on the exploration factor.
    
    Args:
        request: The research question/topic
        npc: The NPC generating the research
        depth: How many steps of research to perform
        memory: How many previous steps to include in context
        context: Additional context to include
        model: LLM model to use
        provider: LLM provider to use
        exploration_factor: Probability (0-1) of exploring a tangential direction
        creativity_factor: Probability (0-1) of pursuing highly creative or unusual ideas
    
    Returns:
        List of research findings/thoughts from this chain
    """
    chain = []
    
    # Initial research prompt
    initial_prompt = f"""
    Research request: {request}
    
    {f'Additional context: {context}' if context else ''}
    
    As {npc.name}, begin your research process by:
    1. Analyzing what you know about this topic
    2. Identifying key questions that need to be explored
    3. Providing initial insights based on your expertise and unique perspective
    4. Highlighting at least one surprising or counterintuitive angle that's rarely discussed
    
    BE EXTREMELY CONCISE. Focus on substance over wordiness. Provide clear, high-value insights in 50-75 words maximum. Be bold and original in your thinking while staying brief.
    """
    
    response = get_llm_response(prompt=initial_prompt, model=model, provider=provider, npc=npc, temperature=0.7)
    initial_findings = response.get('response', '')
    if isinstance(initial_findings, (list, dict)) or hasattr(initial_findings, '__iter__') and not isinstance(initial_findings, (str, bytes)):
        initial_findings = ''.join([str(chunk) for chunk in initial_findings])
    
    chain.append(initial_findings)
    
    # For each level of depth, continue the research
    for i in range(1, depth):
        # Decide research direction based on exploration and creativity factors
        random_val = random.random()
        if random_val < exploration_factor:
            if random_val < (exploration_factor * creativity_factor):
                # Highly creative tangential exploration
                research_type = "highly_creative"
            else:
                # Standard tangential exploration
                research_type = "exploratory"
        else:
            if random_val < creativity_factor:
                # Deep dive with creative approach
                research_type = "deep_creative"
            else:
                # Standard deep dive
                research_type = "deep_standard"
        
        # Get recent memory to include as context
        memory_context = "\n\n".join(chain[-memory:]) if len(chain) > 0 else ""
        
        if research_type == "highly_creative":
            # Highly creative exploration prompt - radical new directions
            next_prompt = f"""
            Research request: {request}
            
            Recent research findings:
            {memory_context}
            
            As {npc.name}, explore a radically different angle on this topic.
            1. Challenge a fundamental assumption that most researchers take for granted
            2. Propose a novel framework or metaphor for understanding this topic
            
            BE EXTREMELY CONCISE. Focus on substance over wordiness. Keep your response to 50-75 words maximum.
            """
        elif research_type == "exploratory":
            # Exploration prompt - branch into a new direction
            next_prompt = f"""
            Research request: {request}
            
            Recent research findings:
            {memory_context}
            
            As {npc.name}, explore a tangential or alternative direction that hasn't been covered yet.
            1. Identify an unexplored angle related to the main topic
            2. Begin a detailed exploration of this new direction
            
            BE EXTREMELY CONCISE. Provide insights that are surprising yet relevant (50-75 words maximum).
            """
        elif research_type == "deep_creative":
            # Creative depth prompt - novel approach to current direction
            next_prompt = f"""
            Research request: {request}
            
            Recent research findings:
            {memory_context}
            
            As {npc.name}, continue exploring what you've discussed so far, but take a creative approach.
            1. Apply an unusual methodology or perspective to your previous findings
            2. Generate insights that might emerge from this creative approach
            
            BE EXTREMELY CONCISE. Focus on generating novel insights (50-75 words maximum).
            """
        else:  # deep_standard
            # Depth prompt - go deeper on current direction
            next_prompt = f"""
            Research request: {request}
            
            Recent research findings:
            {memory_context}
            
            As {npc.name}, continue your research by going deeper into what you've explored so far.
            1. Identify the most important thread from your previous findings
            2. Dive deeper with more detailed analysis
            
            BE EXTREMELY CONCISE. Focus on depth rather than breadth (50-75 words maximum).
            """
        
        temperature = 0.7
        if research_type in ["highly_creative", "deep_creative"]:
            temperature = 0.9  # Higher temperature for more creative explorations
        
        response = get_llm_response(prompt=next_prompt, model=model, provider=provider, npc=npc, temperature=temperature)
        next_findings = response.get('response', '')
        if isinstance(next_findings, (list, dict)) or hasattr(next_findings, '__iter__') and not isinstance(next_findings, (str, bytes)):
            next_findings = ''.join([str(chunk) for chunk in next_findings])
        
        chain.append(next_findings)
    
    return chain

def extract_facts(research_text: str, model: str = None, provider: str = None) -> List[str]:
    """
    Extract key facts and insights from a research text.
    
    Args:
        research_text: The full research text
        model: LLM model to use
        provider: LLM provider to use
        
    Returns:
        List of extracted facts/insights
    """
    prompt = f"""
    Extract the most important facts, insights, and conclusions from the following research text.
    Focus on both conventional wisdom and unusual, provocative, or counterintuitive ideas.
    
    For each fact or insight:
    1. State it clearly and concisely
    2. Ensure it captures a single, coherent idea
    3. Preserve the nuance and specificity
    4. Mark particularly novel or unconventional ideas with [NOVEL] at the beginning
    
    Focus on extracting 10-15 key points that represent the most valuable findings.
    Format each as a separate statement.
    
    Research text:
    {research_text}
    """
    
    response = get_llm_response(prompt=prompt, model=model, provider=provider)
    facts_text = response.get('response', '')
    if isinstance(facts_text, (list, dict)) or hasattr(facts_text, '__iter__') and not isinstance(facts_text, (str, bytes)):
        facts_text = ''.join([str(chunk) for chunk in facts_text])
    
    # Split text into individual facts
    facts = [fact.strip() for fact in facts_text.split('\n') if fact.strip()]
    
    # Remove numbering if present but preserve [NOVEL] tags
    processed_facts = []
    for fact in facts:
        # Remove numbering while preserving [NOVEL] tag if present
        if fact.startswith("[NOVEL]"):
            # Handle numbered novel facts
            if any(fact[7:].strip().startswith(str(i)) for i in range(1, 20)):
                for i in range(1, 20):
                    prefix = f"[NOVEL] {i}."
                    if fact.startswith(prefix):
                        processed_fact = "[NOVEL] " + fact[len(prefix):].strip()
                        processed_facts.append(processed_fact)
                        break
            else:
                processed_facts.append(fact)
        else:
            # Handle regular numbered facts
            if fact[0].isdigit() and fact[1:].strip().startswith(('.', ')', '-')):
                processed_fact = fact[fact.find(' ')+1:].strip()
                processed_facts.append(processed_fact)
            else:
                processed_facts.append(fact)
    
    return processed_facts

def generate_conceptual_combinations(fact_lists: List[List[str]], sample_size: int = 3, num_combinations: int = 5) -> List[Dict]:
    """
    Generate interesting combinations of facts from different researchers to spark novel ideas.
    
    Args:
        fact_lists: List of fact lists from different NPCs
        sample_size: Number of facts to include in each combination
        num_combinations: Number of combinations to generate
        
    Returns:
        List of dictionaries containing the combinations and generated insights
    """
    # Flatten facts with researcher ID
    all_facts_with_source = []
    for i, facts in enumerate(fact_lists):
        for fact in facts:
            all_facts_with_source.append((i, fact))
    
    # Generate random combinations
    combinations = []
    for _ in range(num_combinations):
        if len(all_facts_with_source) <= sample_size:
            sample = all_facts_with_source
        else:
            sample = random.sample(all_facts_with_source, sample_size)
        
        combinations.append({
            "facts": [fact for _, fact in sample],
            "sources": [source for source, _ in sample]
        })
    
    return combinations

def analyze_conceptual_combinations(combinations: List[Dict], request: str, model: str = None, provider: str = None) -> List[Dict]:
    """
    Analyze combinations of facts to identify emergent patterns and generate novel hypotheses.
    
    Args:
        combinations: List of fact combinations
        request: The original research question
        model: LLM model to use
        provider: LLM provider to use
        
    Returns:
        List of dictionaries with analysis results
    """
    results = []
    
    for i, combo in enumerate(combinations):
        facts_text = "\n".join([f"- {fact}" for fact in combo["facts"]])
        
        prompt = f"""
        Consider these seemingly unrelated insights from different researchers exploring the topic:
        "{request}"
        
        {facts_text}
        
        Your task is to identify a non-obvious connection, pattern, or insight that emerges when these ideas are juxtaposed.
        Focus on discovering something truly novel that none of the individual researchers may have recognized.
        
        1. Identify a surprising emergent pattern or connection
        2. Develop a novel hypothesis or research question based on this pattern
        3. Explain how this insight challenges or extends conventional thinking on the topic
        4. Suggest an unconventional methodology or approach to explore this new direction
        
        Be bold, imaginative, and interdisciplinary in your thinking.
        """
        
        response = get_llm_response(prompt=prompt, model=model, provider=provider, temperature=0.9)
        insight = response.get('response', '')
        if isinstance(insight, (list, dict)) or hasattr(insight, '__iter__') and not isinstance(insight, (str, bytes)):
            insight = ''.join([str(chunk) for chunk in insight])
        
        results.append({
            "combination_id": i+1,
            "facts": combo["facts"],
            "sources": combo["sources"],
            "emergent_insight": insight
        })
    
    return results

def identify_patterns_across_chains(chains: Dict[str, List[str]], model: str = None, provider: str = None) -> Dict:
    """
    Identify meta-patterns across research chains, searching for higher-order insights.
    
    Args:
        chains: Dictionary mapping NPC names to their research chains
        model: LLM model to use
        provider: LLM provider to use
        
    Returns:
        Dictionary with meta-analysis results
    """
    # Prepare a summary of each research chain
    chain_summaries = {}
    for name, chain in chains.items():
        full_text = "\n\n".join(chain)
        
        summary_prompt = f"""
        Summarize the key themes, methodologies, and unusual perspectives in this research chain:
        
        {full_text[:2000]}...
        
        Focus on what makes this researcher's approach unique or valuable. Identify their core assumptions,
        methodological innovations, and blindspots (150-200 words).
        """
        
        response = get_llm_response(prompt=summary_prompt, model=model, provider=provider)
        summary = response.get('response', '')
        if isinstance(summary, (list, dict)) or hasattr(summary, '__iter__') and not isinstance(summary, (str, bytes)):
            summary = ''.join([str(chunk) for chunk in summary])
        
        chain_summaries[name] = summary
    
    # Generate meta-analysis across all chains
    all_summaries = "\n\n".join([f"[{name}]\n{summary}" for name, summary in chain_summaries.items()])
    
    meta_analysis_prompt = f"""
    Analyze these research approaches on the topic:
    
    {all_summaries}
    
    Identify:
    1. Surprising methodological patterns - how are researchers approaching this problem in innovative ways?
    2. Conceptual blindspots - what aspects seem to be collectively overlooked?
    3. Emerging paradigms - are there new frameworks or models taking shape across multiple perspectives?
    4. Productive tensions - where do disagreements or contradictions suggest valuable new research directions?
    5. The topology of the problem space - how might we map the conceptual territory in a novel way?
    
    Focus on identifying higher-order insights that emerge from comparing these different approaches.
    Your analysis should challenge conventions and suggest new ways of framing the entire research domain.
    """
    
    response = get_llm_response(prompt=meta_analysis_prompt, model=model, provider=provider, temperature=0.8)
    meta_analysis = response.get('response', '')
    if isinstance(meta_analysis, (list, dict)) or hasattr(meta_analysis, '__iter__') and not isinstance(meta_analysis, (str, bytes)):
        meta_analysis = ''.join([str(chunk) for chunk in meta_analysis])
    
    # Generate innovative research directions
    directions_prompt = f"""
    Based on this meta-analysis of research approaches to the topic:
    
    {meta_analysis}
    
    Propose 5 highly innovative research directions that could transform this field.
    For each direction:
    1. Frame a provocative research question
    2. Explain why it's both important and neglected
    3. Suggest an unconventional methodology to explore it
    4. Describe what a breakthrough in this direction might look like
    
    Your suggestions should be bold, interdisciplinary, and challenge fundamental assumptions.
    Aim for directions that most researchers haven't considered but that could lead to significant advances.
    """
    
    response = get_llm_response(prompt=directions_prompt, model=model, provider=provider, temperature=0.9)
    new_directions = response.get('response', '')
    if isinstance(new_directions, (list, dict)) or hasattr(new_directions, '__iter__') and not isinstance(new_directions, (str, bytes)):
        new_directions = ''.join([str(chunk) for chunk in new_directions])
    
    return {
        "chain_summaries": chain_summaries,
        "meta_analysis": meta_analysis,
        "innovative_directions": new_directions
    }

def consolidate_research(
    chains: Dict[str, List[str]], 
    chain_facts: Dict[str, List[str]], 
    fact_groups: Dict[str, List[int]], 
    meta_patterns: Dict,
    conceptual_combinations: List[Dict],
    model: str = None, 
    provider: str = None
) -> Dict[str, Any]:
    """
    Consolidate the research chains into a comprehensive research report.
    
    Args:
        chains: Dictionary mapping NPC names to their research chains
        chain_facts: Dictionary mapping NPC names to lists of extracted facts
        fact_groups: Dictionary mapping group names to lists of fact indices
        meta_patterns: Results from the meta-pattern analysis
        conceptual_combinations: Results from the conceptual combination analysis
        model: LLM model to use
        provider: LLM provider to use
        
    Returns:
        Dictionary containing the consolidated research
    """
    # Flatten all facts for reference by the grouping indices
    all_facts = []
    fact_sources = []
    for npc_name, facts in chain_facts.items():
        for fact in facts:
            all_facts.append(fact)
            fact_sources.append(npc_name)
    
    # Prepare group content
    group_contents = {}
    for group_name, fact_indices in fact_groups.items():
        group_facts = [(all_facts[idx], fact_sources[idx]) for idx in fact_indices if idx < len(all_facts)]
        facts_text = "\n".join([f"- {fact} (Source: {source})" for fact, source in group_facts])
        group_contents[group_name] = facts_text
    
    # Generate summary for each group
    group_summaries = {}
    group_evaluations = {}
    
    for group_name, content in group_contents.items():
        # Create a summary of the group
        summary_prompt = f"""
        Synthesize the following facts and insights into a coherent summary:
        
        {content}
        
        Provide a detailed synthesis that:
        1. Integrates the different perspectives
        2. Highlights consensus and disagreements
        3. Identifies the most important findings
        4. Suggests implications or applications
        5. Emphasizes any particularly unconventional or innovative ideas
        
        Write 200-300 words that represent the essence of these insights.
        """
        
        summary_response = get_llm_response(prompt=summary_prompt, model=model, provider=provider)
        summary = summary_response.get('response', '')
        if isinstance(summary, (list, dict)) or hasattr(summary, '__iter__') and not isinstance(summary, (str, bytes)):
            summary = ''.join([str(chunk) for chunk in summary])
        
        group_summaries[group_name] = summary
        
        # Evaluate the quality of this group's insights
        group_evaluations[group_name] = evaluate_insight_quality(summary, model=model, provider=provider)
    
    # Prepare the conceptual combinations section
    combinations_text = ""
    for combo in conceptual_combinations:
        facts_list = "\n".join([f"- {fact}" for fact in combo["facts"]])
        combinations_text += f"""
        ## Emergent Insight {combo['combination_id']}
        
        **Combined Facts:**
        {facts_list}
        
        **Novel Perspective:**
        {combo['emergent_insight']}
        
        """
    
    # Prepare meta-patterns section
    safe_integration = meta_patterns.get("meta_analysis", "").replace("_", r"\_").replace("&", r"\&").replace("#", r"\#").replace("%", r"\%")
    meta_patterns_text = f"""
    ## Meta-Analysis of Research Approaches
    
    {safe_integration}
    
    ## Innovative Research Directions
    
    {meta_patterns['innovative_directions']}
    """
    
    # Generate the consolidated report
    groups_text = ""
    for group_name, summary in group_summaries.items():
        eval_scores = group_evaluations[group_name]
        quality_score = (eval_scores.get("Novelty", 5) + eval_scores.get("Depth", 5) + 
                        eval_scores.get("Practicality", 5) + eval_scores.get("Evidence", 5)) / 4
        risk_score = eval_scores.get("Risk", 5)
        
        # Determine if this is "gold" (high quality, low risk) or "cliff" (high risk)
        if quality_score > 7 and risk_score < 4:
            guidance = "🌟 GOLD INSIGHT: This appears to be a valuable finding with strong backing."
        elif risk_score > 7:
            guidance = "⚠️ CLIFF WARNING: This insight has significant risks or uncertainties."
        else:
            guidance = "NEUTRAL: This insight has moderate value and risk."
        
        groups_text += f"""
        ## {group_name}
        
        {summary}
        
        **Evaluation**:
        - Quality: {quality_score:.1f}/10
        - Risk: {risk_score:.1f}/10
        - **Alicanto Guidance**: {guidance}
        
        """
    
    # Final integration and recommendations
    integration_prompt = f"""
    You're synthesizing research on the following topic:
    
    {list(chains.values())[0][0][:200]}...
    
    You have several types of analyses to integrate:
    
    1. Thematic group findings:
    {groups_text}
    
    2. Emergent insights from conceptual combinations:
    {combinations_text}
    
    3. Meta-patterns and innovative directions:
    {meta_patterns_text}
    
    Please provide:
    1. An integrated overview that connects these different perspectives
    2. The most significant overall findings, especially unusual or counterintuitive ones
    3. Specific recommendations for transformative research directions
    4. Particular cautions or limitations to be aware of
    5. A topological map of the problem space - how the different perspectives and approaches relate to each other
    
    Your response should guide the researcher to the most valuable insights (gold) while warning about potentially misleading directions (cliffs).
    Focus on identifying truly innovative directions that could transform understanding of this topic.
    """
    
    integration_response = get_llm_response(integration_prompt, model=model, provider=provider, temperature=0.8)
    integration = integration_response.get('response', '')
    if isinstance(integration, (list, dict)) or hasattr(integration, '__iter__') and not isinstance(integration, (str, bytes)):
        integration = ''.join([str(chunk) for chunk in integration])
    
    # Compile the complete research report
    return {
        "group_summaries": group_summaries,
        "group_evaluations": group_evaluations,
        "conceptual_combinations": conceptual_combinations,
        "meta_patterns": meta_patterns,
        "integration": integration,
        "chains": chains,
        "facts": chain_facts,
        "groups": fact_groups
    }

def identify_groups(fact_lists: List[List[str]], model: str = None, provider: str = None) -> Dict[str, List[int]]:
    """
    Identify thematic groups across all facts extracted from research chains.
    
    Args:
        fact_lists: List of lists of facts from different researchers
        model: LLM model to use
        provider: LLM provider to use
        
    Returns:
        Dictionary mapping group names to lists of fact indices
    """
    # Flatten all facts into a single list
    all_facts = []
    for facts in fact_lists:
        all_facts.extend(facts)
    
    if not all_facts:
        return {}
    
    all_facts_text = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate(all_facts)])
    
    # Prompt the LLM to identify thematic groups
    group_prompt = f"""
    Review these research insights:
    
    {all_facts_text}
    
    Identify 5-8 distinct thematic groups that these insights could be organized into.
    For each group:
    1. Create a descriptive name that captures the essence of that category
    2. List the numbers of the insights that belong in that group
    
    Return your analysis in this format:
    Group Name 1: [1, 5, 9, ...]
    Group Name 2: [2, 6, 10, ...]
    
    Ensure all insight numbers are included in at least one group. Some insights may belong to multiple groups.
    Focus on identifying surprising connections and conceptual families rather than obvious categories.
    """
    
    response = get_llm_response(prompt=group_prompt, model=model, provider=provider)
    groups_text = response.get('response', '')
    if isinstance(groups_text, (list, dict)) or hasattr(groups_text, '__iter__') and not isinstance(groups_text, (str, bytes)):
        groups_text = ''.join([str(chunk) for chunk in groups_text])
    
    # Parse the groups from the response
    groups = {}
    for line in groups_text.split("\n"):
        line = line.strip()
        if line and ":" in line:
            parts = line.split(":", 1)
            group_name = parts[0].strip()
            
            # Extract fact indices
            indices_text = parts[1].strip()
            if indices_text.startswith("[") and indices_text.endswith("]"):
                indices_text = indices_text[1:-1]
            
            # Parse indices
            indices = []
            for idx_part in indices_text.split(","):
                try:
                    # Subtract 1 because we 1-indexed in the prompt but need 0-indexed for processing
                    idx = int(idx_part.strip()) - 1
                    if 0 <= idx < len(all_facts):
                        indices.append(idx)
                except ValueError:
                    continue
            
            if indices:
                groups[group_name] = indices
    
    # If groups parsing failed, create default groups
    if not groups:
        num_facts = len(all_facts)
        num_groups = min(7, max(4, num_facts // 10 + 3))  # Adaptive number of groups
        group_size = num_facts // num_groups
        
        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = start_idx + group_size if i < num_groups - 1 else num_facts
            groups[f"Thematic Group {i+1}"] = list(range(start_idx, end_idx))
    
    return groups

def evaluate_insight_quality(insight_text: str, model: str = None, provider: str = None) -> Dict[str, float]:
    """
    Evaluate the quality and risk of a research insight.
    
    Args:
        insight_text: The insight text to evaluate
        model: LLM model to use
        provider: LLM provider to use
        
    Returns:
        Dictionary of evaluation scores
    """
    eval_prompt = f"""
    Evaluate this research insight on climate change and food security:
    
    {insight_text}
    
    Please rate this insight on the following dimensions from 1-10:
    
    1. Novelty: How original or unexpected is this insight? (1=entirely conventional, 10=groundbreaking)
    2. Depth: How deeply does it explore complex interactions and underlying mechanisms? (1=surface-level, 10=profound analysis)
    3. Practicality: How relevant and actionable is this for addressing real-world problems? (1=purely theoretical, 10=immediately applicable)
    4. Evidence: How well-supported is this insight by established knowledge? (1=speculative, 10=strongly supported)
    5. Risk: What is the level of uncertainty or potential for harmful outcomes if this insight guides policy? (1=extremely low risk, 10=high risk/uncertainty)
    
    For each dimension, provide a numeric score and a brief 1-sentence justification.
    """
    
    response = get_llm_response(prompt=eval_prompt, model=model, provider=provider)
    eval_text = response.get('response', '')
    if isinstance(eval_text, (list, dict)) or hasattr(eval_text, '__iter__') and not isinstance(eval_text, (str, bytes)):
        eval_text = ''.join([str(chunk) for chunk in eval_text])
    
    # Extract scores from the evaluation text
    scores = {}
    for dimension in ["Novelty", "Depth", "Practicality", "Evidence", "Risk"]:
        pattern = f"{dimension}:\\s*(\\d+)"
        import re
        match = re.search(pattern, eval_text)
        if match:
            try:
                scores[dimension] = min(10, max(1, int(match.group(1))))  # Ensure 1-10 range
            except ValueError:
                scores[dimension] = 5  # Default score
        else:
            scores[dimension] = 5  # Default score if not found
    
    return scores

def generate_simulated_experiments(request: str, groups: Dict[str, List[int]], facts: Dict[str, List[str]], model: str = None, provider: str = None) -> Dict[str, Any]:
    """
    Generate simulated experiments based on the research findings to provide baselines 
    and concrete examples for users to understand the concepts.
    
    Args:
        request: The original research question/topic
        groups: Dictionary mapping group names to fact indices
        facts: Dictionary mapping NPC names to their facts
        model: LLM model to use
        provider: LLM provider to use
        
    Returns:
        Dictionary with experiment designs and simulated results
    """
    print("Generating simulated experiments based on research findings...")
    
    # Collect all facts across NPCs
    all_facts = []
    for npc_facts in facts.values():
        all_facts.extend(npc_facts)
    
    # Prompt for experiment design based on thematic groups
    experiments = {}
    
    for group_name, fact_indices in groups.items():
        # Skip if there are no facts for this group
        if not fact_indices or max(fact_indices) >= len(all_facts):
            continue
            
        # Get facts for this group
        group_facts = [all_facts[idx] for idx in fact_indices if idx < len(all_facts)]
        facts_text = "\n".join([f"- {fact}" for fact in group_facts])
        
        # Create experiment design prompt
        design_prompt = f"""
        Based on the following research insights related to "{request}" within the theme "{group_name}":
        
        {facts_text}
        
        Design a simple experiment or simulation that could test or demonstrate these concepts.
        
        The experiment should:
        1. Be clear and straightforward to understand
        2. Demonstrate a key principle or finding from the research
        3. Include variables that could be measured
        4. Produce data that could be visualized
        5. Serve as a baseline example that researchers could build upon
        
        For your experiment design, provide:
        1. A title for the experiment
        2. The hypothesis being tested
        3. The methodology (steps, variables, measurements)
        4. Expected outcomes based on the research insights
        5. How the results would look (what pattern to expect in the data)
        6. A suggestion for how to visualize the results effectively
        
        Make this a concrete, practical experiment design that could actually be run.
        """
        
        response = get_llm_response(prompt=design_prompt, model=model, provider=provider)
        experiment_design = response.get('response', '')
        if isinstance(experiment_design, (list, dict)) or hasattr(experiment_design, '__iter__') and not isinstance(experiment_design, (str, bytes)):
            experiment_design = ''.join([str(chunk) for chunk in experiment_design])
        
        # Generate simulated data for the experiment
        data_prompt = f"""
        You designed the following experiment:
        
        {experiment_design}
        
        Now, generate simulated data for this experiment using the Python libraries numpy and matplotlib.
        
        Your simulation code should:
        1. Define all necessary parameters
        2. Generate realistic synthetic data based on the expected outcomes
        3. Calculate relevant statistics
        4. Create simple, clear matplotlib visualizations (not using seaborn)
        5. Provide brief textual interpretation of the results
        
        Return ONLY executable Python code that simulates this experiment and visualizes the results.
        Include comments to explain key steps. The code should be completely self-contained.
        Be creative but realistic in your data simulation.
        """
        
        response = get_llm_response(prompt=data_prompt, model=model, provider=provider)
        simulation_code = response.get('response', '')
        if isinstance(simulation_code, (list, dict)) or hasattr(simulation_code, '__iter__') and not isinstance(simulation_code, (str, bytes)):
            simulation_code = ''.join([str(chunk) for chunk in simulation_code])
        
        # Extract the actual code from the response, removing markdown code blocks if present
        if "```python" in simulation_code:
            simulation_code = simulation_code.split("```python")[1].split("```")[0].strip()
        elif "```" in simulation_code:
            simulation_code = simulation_code.split("```")[1].split("```")[0].strip()
        
        # Store the experiment design and simulation code
        experiments[group_name] = {
            "design": experiment_design,
            "simulation_code": simulation_code,
            "results": None  # Will be populated after running the simulation
        }
    
    return experiments

def run_simulations(experiments: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Run the simulated experiments and capture their results.
    
    Args:
        experiments: Dictionary of experiment designs and simulation code
        
    Returns:
        Updated experiments dictionary with results
    """
    print("Running simulations and generating visualizations...")
    
    for group_name, experiment in experiments.items():
        try:
            # Create a BytesIO object to capture the plot
            buffer = BytesIO()
            
            # Prepare local variables for the code to execute
            local_vars = {
                'plt': plt,
                'np': np,
                'Figure': Figure,
                'BytesIO': BytesIO,
                'base64': base64,
                'buffer': buffer
            }
            
            # Modify the code to save the figure to our buffer
            code = experiment["simulation_code"]
            if "plt.show()" in code:
                code = code.replace("plt.show()", "plt.savefig(buffer, format='png', dpi=100)\nbuffer.seek(0)")
            else:
                code += "\nplt.savefig(buffer, format='png', dpi=100)\nbuffer.seek(0)"
            
            # Execute the simulation code
            exec(code, globals(), local_vars)
            
            # Get the buffer from the local variables
            buffer = local_vars.get('buffer', buffer)
            
            # Convert the buffer to a base64-encoded string
            buffer.seek(0)
            img_str = base64.b64encode(buffer.read()).decode()
            
            # Store the results
            experiment["results"] = {
                "image": img_str,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            import traceback
            experiment["results"] = {
                "image": None,
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    return experiments

def generate_pdf_report(request: str, research: Dict[str, Any], experiments: Dict[str, Dict[str, Any]], output_path: str = None, fast_mode: bool = True) -> str:
    """
    Generate a PDF report with visualizations from the research and simulated experiments.
    Using only ReportLab for PDF generation to avoid LaTeX issues.
    
    Args:
        request: The original research question
        research: The consolidated research results
        experiments: The simulated experiments and their results
        output_path: Path to save the PDF report (default: current directory)
        fast_mode: Ignored parameter, kept for backward compatibility
        
    Returns:
        Path to the generated PDF file
    """
    if output_path is None:
        output_path = os.getcwd()
    
    # Create a filename based on the request
    sanitized_request = "".join(c for c in request if c.isalnum() or c.isspace()).strip()
    sanitized_request = sanitized_request.replace(" ", "_")[:50]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{sanitized_request}_{timestamp}.pdf"
    filepath = os.path.join(output_path, filename)
    
    # Generate PDF with ReportLab
    print("Generating PDF report with ReportLab...")
    
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "reportlab"], check=True)
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    
    # Create the document
    doc = SimpleDocTemplate(filepath, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    custom_styles = {
        'Title': ParagraphStyle(
            name='Title', 
            parent=styles['Heading1'], 
            fontSize=20, 
            alignment=TA_CENTER,
            spaceAfter=20
        ),
        'Heading2': ParagraphStyle(
            name='Heading2', 
            parent=styles['Heading2'], 
            fontSize=16, 
            spaceAfter=10
        ),
        'Heading3': ParagraphStyle(
            name='Heading3', 
            parent=styles['Heading3'], 
            fontSize=14, 
            spaceAfter=8
        ),
        'Normal': ParagraphStyle(
            name='Normal', 
            parent=styles['Normal'], 
            fontSize=11, 
            spaceAfter=10
        ),
        'GoldInsight': ParagraphStyle(
            name='GoldInsight', 
            parent=styles['Normal'], 
            fontSize=11, 
            backColor=colors.gold,
            borderColor=colors.black,
            borderWidth=1,
            borderPadding=5,
            spaceAfter=10
        ),
        'CliffWarning': ParagraphStyle(
            name='CliffWarning', 
            parent=styles['Normal'], 
            fontSize=11, 
            backColor=colors.lightcoral,
            borderColor=colors.black,
            borderWidth=1,
            borderPadding=5,
            spaceAfter=10
        ),
        'NeutralInsight': ParagraphStyle(
            name='NeutralInsight', 
            parent=styles['Normal'], 
            fontSize=11, 
            backColor=colors.lightblue,
            borderColor=colors.black,
            borderWidth=1,
            borderPadding=5,
            spaceAfter=10
        )
    }
    
    # Add custom styles to the stylesheet
    for style_name, style in custom_styles.items():
        styles.add(style)
    
    # Content elements
    elements = []
    
    # Title
    elements.append(Paragraph("Alicanto Research Report", styles["Title"]))
    elements.append(Spacer(1, 20))
    
    # Research Topic
    elements.append(Paragraph("Research Topic:", styles["Heading2"]))
    elements.append(Paragraph(request, styles["Normal"]))
    elements.append(Spacer(1, 20))
    
    # Executive Summary
    elements.append(Paragraph("Executive Summary", styles["Heading2"]))
    elements.append(Paragraph(research.get("integration", ""), styles["Normal"]))
    elements.append(Spacer(1, 20))
    
    # Thematic Findings
    elements.append(Paragraph("Thematic Findings and Insights", styles["Heading2"]))
    
    for group_name, summary in research.get("group_summaries", {}).items():
        eval_scores = research.get("group_evaluations", {}).get(group_name, {})
        quality_score = (eval_scores.get("Novelty", 5) + eval_scores.get("Depth", 5) + 
                        eval_scores.get("Practicality", 5) + eval_scores.get("Evidence", 5)) / 4
        risk_score = eval_scores.get("Risk", 5)
        
        elements.append(Paragraph(group_name, styles["Heading3"]))
        
        # Choose style based on scores
        style_name = "NeutralInsight"
        if quality_score > 7 and risk_score < 4:
            style_name = "GoldInsight"
        elif risk_score > 7:
            style_name = "CliffWarning"
        
        elements.append(Paragraph(summary, styles[style_name]))
        elements.append(Paragraph(f"Quality Score: {quality_score:.1f}/10, Risk Score: {risk_score:.1f}/10", styles["Normal"]))
        
        # Add experiment data if available
        if group_name in experiments:
            experiment = experiments[group_name]
            elements.append(Paragraph("Simulated Experiment", styles["Heading3"]))
            elements.append(Paragraph(experiment.get("design", ""), styles["Normal"]))
            
            if experiment.get("results", {}).get("image"):
                img_path = os.path.join(os.path.dirname(filepath), f"{group_name.replace(' ', '_')}_experiment.png")
                with open(img_path, "wb") as img_file:
                    img_file.write(base64.b64decode(experiment["results"]["image"]))
                img = Image(img_path, width=400, height=300)
                elements.append(img)
            elif experiment.get("results", {}).get("error"):
                elements.append(Paragraph(f"Error running simulation: {experiment['results']['error']}", styles["Normal"]))
        
        elements.append(Spacer(1, 20))
    
    # Emergent Insights
    elements.append(Paragraph("Emergent Insights from Conceptual Combinations", styles["Heading2"]))
    
    for combo in research.get("conceptual_combinations", []):
        elements.append(Paragraph(f"Insight {combo.get('combination_id', '')}", styles["Heading3"]))
        
        # Facts list
        elements.append(Paragraph("Combined Facts:", styles["Normal"]))
        for fact in combo.get("facts", []):
            elements.append(Paragraph(f"• {fact}", styles["Normal"]))
        
        elements.append(Paragraph("Novel Perspective:", styles["Normal"]))
        elements.append(Paragraph(combo.get("emergent_insight", ""), styles["GoldInsight"]))
        elements.append(Spacer(1, 10))
    
    # Meta-analysis
    elements.append(Paragraph("Meta-Analysis and Research Directions", styles["Heading2"]))
    elements.append(Paragraph("Meta-Analysis of Research Approaches", styles["Heading3"]))
    elements.append(Paragraph(research.get("meta_patterns", {}).get("meta_analysis", ""), styles["Normal"]))
    
    elements.append(Paragraph("Innovative Research Directions", styles["Heading3"]))
    elements.append(Paragraph(research.get("meta_patterns", {}).get("innovative_directions", ""), styles["Normal"]))
    
    # Contributors
    elements.append(Paragraph("Contributors", styles["Heading2"]))
    for npc_name in research.get("chains", {}).keys():
        elements.append(Paragraph(f"• {npc_name}", styles["Normal"]))
    
    # Build the PDF
    doc.build(elements)
    
    return filepath

def alicanto(
    request: str,
    num_npcs: int = 5,
    depth: int = 3,
    memory: int = 3,
    context: str = None,
    model: str = None,
    provider: str = None,
    exploration_factor: float = 0.3,
    creativity_factor: float = 0.5,
    output_format: str = "report",
    fast_mode: bool = True  # Default to fast mode for PDF generation
) -> Dict[str, Any]:
    """
    Run the Alicanto research process to explore a topic from multiple perspectives.
    
    Args:
        request: The research question/topic to explore
        num_npcs: Number of researcher NPCs to use
        depth: Depth of research chains
        memory: How many previous steps to remember in each chain
        context: Additional context to include
        model: LLM model to use
        provider: LLM provider to use
        exploration_factor: Factor for exploration vs exploitation (0-1)
        creativity_factor: Factor for creative vs standard approaches (0-1)
        output_format: Format of the output ("full", "summary", "report", "concise")
        fast_mode: Whether to use fast mode for PDF generation
        
    Returns:
        Dictionary with the research results
    """
    if model is None:
        model = NPCSH_CHAT_MODEL
    if provider is None:
        provider = NPCSH_CHAT_PROVIDER
    
    print(f"🦅 Alicanto: Beginning research on '{request}'")
    
    # Generate diverse research NPCs
    print(f"Generating {num_npcs} diverse researcher NPCs...")
    npcs = generate_random_npcs(num_npcs, model, provider, request)
    
    # Generate research chains for each NPC
    print("Generating research chains from multiple perspectives...")
    chains = {}
    for npc in npcs:
        print(f"  Research chain from {npc.name}...")
        chain = generate_research_chain(
            request, npc, depth, memory, context, model, provider,
            exploration_factor, creativity_factor
        )
        chains[npc.name] = chain
    
    # Extract facts from each chain
    print("Extracting key facts and insights from research chains...")
    chain_facts = {}
    for name, chain in chains.items():
        print(f"  Extracting facts from {name}'s research...")
        chain_text = "\n\n".join(chain)
        facts = extract_facts(chain_text, model, provider)
        chain_facts[name] = facts
    
    # Identify thematic groups across all facts
    print("Identifying thematic groups across all research insights...")
    fact_lists = list(chain_facts.values())
    fact_groups = identify_groups(fact_lists, model, provider)
    
    # Generate conceptual combinations
    print("Generating conceptual combinations to spark novel insights...")
    combinations = generate_conceptual_combinations(fact_lists)
    
    # Analyze the combinations
    print("Analyzing conceptual combinations for emergent insights...")
    analyzed_combinations = analyze_conceptual_combinations(combinations, request, model, provider)
    
    # Identify meta-patterns across research chains
    print("Identifying meta-patterns across research approaches...")
    meta_patterns = identify_patterns_across_chains(chains, model, provider)
    
    # Consolidate the research
    print("Consolidating research into comprehensive synthesis...")
    research = consolidate_research(
        chains, chain_facts, fact_groups, meta_patterns, analyzed_combinations,
        model, provider
    )
    
    # Generate simulated experiments if we want a full report
    if output_format in ["full", "report"]:
        experiments = generate_simulated_experiments(request, fact_groups, chain_facts, model, provider)
        experiments = run_simulations(experiments)
        research["experiments"] = experiments
    else:
        experiments = {}
    
    # Generate PDF report
    if output_format in ["full", "report"]:
        print("Generating PDF research report...")
        pdf_path = generate_pdf_report(request, research, experiments, fast_mode=fast_mode)
        research["pdf_path"] = pdf_path
        print(f"PDF report generated: {pdf_path}")
    
    # Prepare output based on requested format
    if output_format == "concise":
        output = research["integration"]
    elif output_format == "summary":
        output = {
            "integration": research["integration"],
            "group_summaries": research["group_summaries"],
            "meta_patterns": research["meta_patterns"]
        }
    elif output_format == "report":
        # Format a text report
        output = f"""# Alicanto Research Report: {request}

## Executive Summary
{research['integration']}

## Thematic Findings
"""
        for group_name, summary in research["group_summaries"].items():
            eval_scores = research["group_evaluations"][group_name]
            quality_score = (eval_scores.get("Novelty", 5) + eval_scores.get("Depth", 5) + 
                            eval_scores.get("Practicality", 5) + eval_scores.get("Evidence", 5)) / 4
            risk_score = eval_scores.get("Risk", 5)
            
            if quality_score > 7 and risk_score < 4:
                indicator = "🌟 GOLD INSIGHT"
            elif risk_score > 7:
                indicator = "⚠️ CLIFF WARNING"
            else:
                indicator = "NEUTRAL"
            
            output += f"""### {group_name} ({indicator})
{summary}

Quality: {quality_score:.1f}/10 | Risk: {risk_score:.1f}/10

"""
        
        output += """## Emergent Insights from Conceptual Combinations
"""
        for combo in research["conceptual_combinations"]:
            output += f"""### Insight {combo['combination_id']}
{combo['emergent_insight']}

"""
        
        output += f"""## Meta-Analysis
{research['meta_patterns']['meta_analysis']}

## Innovative Research Directions
{research['meta_patterns']['innovative_directions']}

## PDF Report
A detailed PDF report has been generated: {research.get('pdf_path', 'Not available')}
"""
    else:  # full format or default
        output = research
    
    return output

def main():
    """CLI interface for Alicanto research"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Alicanto: Deep thematic research with multiple perspectives")
    parser.add_argument("request", help="Research question or topic to explore")
    parser.add_argument("--num-npcs", type=int, default=5, help="Number of researcher NPCs to use")
    parser.add_argument("--depth", type=int, default=3, help="Depth of research chains")
    parser.add_argument("--memory", type=int, default=3, help="How many previous steps to remember")
    parser.add_argument("--model", default=NPCSH_CHAT_MODEL, help="LLM model to use")
    parser.add_argument("--provider", default=NPCSH_CHAT_PROVIDER, help="LLM provider to use")
    parser.add_argument("--context", help="Additional context to include")
    parser.add_argument("--exploration", type=float, default=0.3, help="Exploration vs exploitation factor (0-1)")
    parser.add_argument("--creativity", type=float, default=0.5, help="Creativity factor (0-1)")
    parser.add_argument("--format", choices=["full", "summary", "report", "concise"], default="report", 
                        help="Output format")
    parser.add_argument("--fast-pdf", action="store_true", help="Use fast PDF generation mode")
    
    args = parser.parse_args()
    
    result = alicanto(
        request=args.request,
        num_npcs=args.num_npcs,
        depth=args.depth,
        memory=args.memory,
        context=args.context,
        model=args.model,
        provider=args.provider,
        exploration_factor=args.exploration,
        creativity_factor=args.creativity,
        output_format=args.format
    )
    
    if args.format == "report" or args.format == "concise":
        print(result)
    else:
        import json
        print(json.dumps(result, indent=2))
    
    # Generate PDF report
    generate_pdf_report(
        request=args.request,
        research=result,
        experiments={},
        fast_mode=args.fast_pdf
    )

if __name__ == "__main__":
    main()