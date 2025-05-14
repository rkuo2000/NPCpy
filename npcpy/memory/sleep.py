import os
import sys
import datetime
from typing import List, Dict, Any, Optional, Union

import npcpy.memory.knowledge_graph as kg
from npcpy.llm_funcs import get_llm_response
from npcpy.npc_compiler import NPC

def sleep(
    messages: Optional[List[Dict[str, str]]],
    model: str,
    provider: str,
    db_path: str = os.path.expanduser("~/npcsh_graph.db"),
    chroma_path: str = os.path.expanduser("~/npcsh_chroma.db"),
    npc: Any = None,
    context: str = None,
) -> Union[str, Dict[str, Any]]:
    """
    Process conversation history to update the knowledge graph.
    This function extracts facts, mistakes, and lessons learned from the conversation
    and stores them in the knowledge graph for long-term memory.
    
    Args:
        messages: The conversation history
        model: The model to use for the LLM
        provider: The provider for the LLM
        db_path: Path to the knowledge graph database
        chroma_path: Path to the vector database for embeddings
        npc: The NPC object (optional)
        context: Additional context (optional)
        
    Returns:
        Summary string or dict with extraction results
    """
    if not messages or not isinstance(messages, list):
        return "No valid messages provided to process"
    
    # Convert messages to text
    conversation_text = "\n".join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in messages])
    
    try:
        # Initialize database connection
        conn = kg.init_db(db_path, drop=False)
        if not conn:
            return "Failed to connect to the knowledge graph database"
        
        # Extract insights from conversation
        extraction_prompt = f"""
        You are analyzing a conversation to extract valuable information for long-term memory.
        Identify the following from the conversation:

        1. Facts: Important factual information worth remembering
        2. Mistakes: Errors made or misconceptions corrected
        3. Lessons: Insights, learning points, or principles discovered
        4. Actions: Tasks completed or important actions taken
        5. Decisions: Key decisions or choices made

        Format your response as a JSON object with these categories as keys,
        and lists of extracted items as values. Be concise and specific.

        Conversation:
        {conversation_text}
        """

        if npc and hasattr(npc, 'name'):
            extraction_prompt += f"\n\nThis is for an AI assistant named {npc.name}."

        response = get_llm_response(
            prompt=extraction_prompt,
            model=model,
            provider=provider,
            messages=[{"role": "system", "content": "You are a helpful assistant for memory extraction"}],
            stream=False,
            format="json"
        )

        extraction_data = response.get("response", {})
        if isinstance(extraction_data, str):
            try:
                import json
                extraction_data = json.loads(extraction_data)
            except:
                extraction_data = {"facts": [], "mistakes": [], "lessons": [], "actions": [], "decisions": []}
        
        # Store extractions in knowledge graph
        timestamp = datetime.datetime.now().isoformat()
        source = f"conversation_{timestamp}"
        
        facts = extraction_data.get("facts", [])
        mistakes = extraction_data.get("mistakes", [])
        lessons = extraction_data.get("lessons", [])
        actions = extraction_data.get("actions", [])
        decisions = extraction_data.get("decisions", [])
        
        # Store facts
        for fact in facts:
            kg.add_fact(conn, fact, source)
        
        # Store mistakes
        for mistake in mistakes:
            kg.add_mistake(conn, mistake, source)
        
        # Store lessons
        for lesson in lessons:
            kg.add_lesson(conn, lesson, source)
            
        # Store actions (if the function exists)
        if hasattr(kg, 'add_action'):
            for action in actions:
                kg.add_action(conn, action, source)
                
        # Store decisions (if the function exists)
        if hasattr(kg, 'add_decision'):
            for decision in decisions:
                kg.add_decision(conn, decision, source)
        
        # Generate summary
        summary = []
        if facts:
            summary.append(f"📚 **Facts Learned**: {len(facts)}")
            for fact in facts[:3]:  # Show only first 3 for brevity
                summary.append(f"- {fact}")
            if len(facts) > 3:
                summary.append(f"- _(and {len(facts) - 3} more facts)_")
                
        if mistakes:
            summary.append(f"⚠️ **Mistakes Noted**: {len(mistakes)}")
            for mistake in mistakes[:3]:
                summary.append(f"- {mistake}")
            if len(mistakes) > 3:
                summary.append(f"- _(and {len(mistakes) - 3} more mistakes)_")
                
        if lessons:
            summary.append(f"💡 **Lessons Learned**: {len(lessons)}")
            for lesson in lessons[:3]:
                summary.append(f"- {lesson}")
            if len(lessons) > 3:
                summary.append(f"- _(and {len(lessons) - 3} more lessons)_")
                
        if actions:
            summary.append(f"🏃 **Actions Taken**: {len(actions)}")
            for action in actions[:3]:
                summary.append(f"- {action}")
            if len(actions) > 3:
                summary.append(f"- _(and {len(actions) - 3} more actions)_")
                
        if decisions:
            summary.append(f"🔀 **Decisions Made**: {len(decisions)}")
            for decision in decisions[:3]:
                summary.append(f"- {decision}")
            if len(decisions) > 3:
                summary.append(f"- _(and {len(decisions) - 3} more decisions)_")
        
        summary_text = "\n\n".join(summary)
        if not summary_text:
            summary_text = "No significant information extracted from this conversation."
            
        # Also store in vector DB if possible
        try:
            kg.store_in_vector_db(
                chroma_path, 
                conversation_text, 
                extraction_data, 
                source
            )
        except Exception as e:
            summary_text += f"\n\n⚠️ Note: Failed to store in vector database: {str(e)}"
            
        return summary_text
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error processing memory: {str(e)}"

def forget(
    entity: str,
    model: str,
    provider: str, 
    db_path: str = os.path.expanduser("~/npcsh_graph.db"),
    chroma_path: str = os.path.expanduser("~/npcsh_chroma.db"),
    npc: Any = None,
) -> Union[str, Dict[str, Any]]:
    """
    Remove specific information from the knowledge graph.
    
    Args:
        entity: The entity or concept to forget
        model: The model to use
        provider: The provider to use
        db_path: Path to the knowledge graph database
        chroma_path: Path to the vector database
        npc: The NPC object (optional)
        
    Returns:
        Summary of what was forgotten
    """
    if not entity or not entity.strip():
        return "Please specify what to forget"
    
    try:
        # Initialize database connection
        conn = kg.init_db(db_path, drop=False)
        if not conn:
            return "Failed to connect to the knowledge graph database"
        
        # Search for facts, mistakes, and lessons related to the entity
        facts = kg.search_facts(conn, entity)
        mistakes = kg.search_mistakes(conn, entity)
        lessons = kg.search_lessons(conn, entity)
        
        # If the specific functions exist, also search for actions and decisions
        actions = []
        decisions = []
        if hasattr(kg, 'search_actions'):
            actions = kg.search_actions(conn, entity)
        if hasattr(kg, 'search_decisions'):
            decisions = kg.search_decisions(conn, entity)
        
        # Generate a verification prompt to confirm what to forget
        items_to_forget = []
        items_to_forget.extend([("fact", fact_id, fact_text) for fact_id, fact_text, _ in facts])
        items_to_forget.extend([("mistake", mistake_id, mistake_text) for mistake_id, mistake_text, _ in mistakes])
        items_to_forget.extend([("lesson", lesson_id, lesson_text) for lesson_id, lesson_text, _ in lessons])
        items_to_forget.extend([("action", action_id, action_text) for action_id, action_text, _ in actions])
        items_to_forget.extend([("decision", decision_id, decision_text) for decision_id, decision_text, _ in decisions])
        
        if not items_to_forget:
            return f"No memories found related to '{entity}'"
        
        # Generate a confirmation message
        summary = []
        summary.append(f"🗑️ **Forgetting information about '{entity}'**\n")
        
        # Delete the items
        forgotten_count = 0
        for item_type, item_id, item_text in items_to_forget:
            if item_type == "fact":
                kg.delete_fact(conn, item_id)
            elif item_type == "mistake":
                kg.delete_mistake(conn, item_id)
            elif item_type == "lesson":
                kg.delete_lesson(conn, item_id)
            elif item_type == "action" and hasattr(kg, 'delete_action'):
                kg.delete_action(conn, item_id)
            elif item_type == "decision" and hasattr(kg, 'delete_decision'):
                kg.delete_decision(conn, item_id)
            
            summary.append(f"- Forgot {item_type}: {item_text}")
            forgotten_count += 1
        
        # Also remove from vector DB if possible
        try:
            kg.remove_from_vector_db(chroma_path, entity)
            summary.append(f"\nAlso removed related embeddings from vector database.")
        except Exception as e:
            summary.append(f"\n⚠️ Note: Failed to remove from vector database: {str(e)}")
        
        summary_text = "\n".join(summary)
        if forgotten_count == 0:
            summary_text = f"No memories were forgotten related to '{entity}'."
            
        return summary_text
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error forgetting memory: {str(e)}"
