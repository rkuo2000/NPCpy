from npcpy.memory.knowledge_graph import *
import os
import time
from datetime import datetime


def print_section(title):
    """Print a formatted section title"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


def print_facts_in_groups(conn):
    """Print all facts organized by groups"""
    print_section("CURRENT KNOWLEDGE GRAPH STATE")
    
    # Get all groups
    result = conn.execute("MATCH (g:Groups) RETURN g.name").get_as_df()
    groups = [row["g.name"] for _, row in result.iterrows()]
    
    # For each group, get its facts
    for group in groups:
        print(f"\nGROUP: {group}")
        print("-" * 50)
        
        escaped_group = group.replace('"', '\\"')
        result = conn.execute(f"""
            MATCH (g:Groups)-[:Contains]->(f:Fact)
            WHERE g.name = "{escaped_group}"
            RETURN f.content
        """).get_as_df()
        
        if result.empty:
            print("  No facts in this group yet")
        else:
            for i, row in enumerate(result.iterrows(), 1):
                print(f"  {i}. {row[1]['f.content']}")
    
    # Count facts not assigned to any group
    result = conn.execute("""
        MATCH (f:Fact) 
        WHERE NOT EXISTS { MATCH (g:Groups)-[:Contains]->(f) }
        RETURN count(f) as orphan_count
    """).get_as_df()
    
    orphan_count = result.iloc[0]["orphan_count"]
    print(f"\nFacts not assigned to any group: {orphan_count}")


def get_analytics(conn):
    """Print analytics about the knowledge graph"""
    print_section("KNOWLEDGE GRAPH ANALYTICS")
    
    # Count all facts
    result = conn.execute("MATCH (f:Fact) RETURN count(f) as fact_count").get_as_df()
    fact_count = result.iloc[0]["fact_count"]
    
    # Count all groups
    result = conn.execute("MATCH (g:Groups) RETURN count(g) as group_count").get_as_df()
    group_count = result.iloc[0]["group_count"]
    
    # Average facts per group
    result = conn.execute("""
        MATCH (g:Groups)-[:Contains]->(f:Fact)
        RETURN g.name, count(f) as fact_count
    """).get_as_df()
    
    if not result.empty:
        avg_facts_per_group = result["fact_count"].mean()
    else:
        avg_facts_per_group = 0
    
    print(f"Total Facts: {fact_count}")
    print(f"Total Groups: {group_count}")
    print(f"Average Facts per Group: {avg_facts_per_group:.2f}")


# Monkey patch the datetime.now function in the knowledge_graph module
# This is needed because it's using datetime.now() directly instead of datetime.datetime.now()
import datetime as dt
dt.now = dt.datetime.now


if __name__ == "__main__":
    # Define the database path - use current time to avoid conflicts with previous tests
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_path = os.path.expanduser(f"~/knowledge_graph_test_{timestamp}.db")
    path = os.path.expanduser("~/npcww/npcsh/tests/")
    
    print(f"Creating test knowledge graph database at: {db_path}")
    
    # Initialize database with a fresh start
    conn = init_db(db_path, drop=True)
    
    # PHASE 1: Initial Knowledge Graph Population
    print_section("PHASE 1: CREATING INITIAL KNOWLEDGE GRAPH")
    
    # Initial text about npcsh
    initial_text = """
    npcsh is a python based command line tool designed to integrate Large Language Models (LLMs) into one's daily workflow by making them available through the command line shell.
    
    Smart Interpreter: npcsh leverages the power of LLMs to understand your natural language commands and questions, executing tasks, answering queries, and providing relevant information from local files and the web.
    
    Command History: npcsh remembers your command history and can reference previous commands and their outputs.
    """
    
    print("Extracting facts from initial text...")
    facts = extract_facts(initial_text, model="gpt-4o-mini", provider="openai")
    
    print("\nIdentifying initial groups...")
    initial_groups = identify_groups(facts, model="gpt-4o-mini", provider="openai")
    print("Initial groups identified:")
    for group in initial_groups:
        print(f"- {group}")
        create_group(conn, group)
    
    print("\nAssigning facts to appropriate groups...")
    for fact in facts:
        print(f"Processing fact: {fact}")
        # First insert the fact into the database
        insert_fact(conn, fact, path)
        # Then assign it to groups
        group_assignments = assign_groups_to_fact(fact, initial_groups, model="gpt-4o-mini", provider="openai")
        print(f"Assigned to groups: {group_assignments.get('groups', [])}")
        for group in group_assignments.get("groups", []):
            assign_fact_to_group_graph(conn, fact, group)
    
    print_facts_in_groups(conn)
    get_analytics(conn)
    
    # PHASE 2: Adding More Knowledge
    print_section("PHASE 2: EVOLVING THE KNOWLEDGE GRAPH WITH NEW INFORMATION")
    
    # New information about npcsh
    new_text = """
    Macros: npcsh provides macros to accomplish common tasks with LLMs like voice control (/yap), image generation (/vixynt), screenshot capture and analysis (/ots), one-shot questions (/sample), and more.

    NPC-Driven Interactions: npcsh allows users to coordinate agents (i.e. NPCs) to form assembly lines that can reliably accomplish complicated multi-step procedures. Define custom "NPCs" (Non-Player Characters) with specific personalities, directives, and tools.
    
    Advanced Customization: npcsh supports user-defined configuration files that allow for extensive customization of the tool's behavior, appearance, and functionality.
    """
    
    print("Extracting facts from new text...")
    new_facts = extract_facts(new_text, model="gpt-4o-mini", provider="openai")
    
    print("\nRe-evaluating groups with new knowledge...")
    all_facts = facts + new_facts
    updated_groups = identify_groups(all_facts, model="gpt-4o-mini", provider="openai")
    
    # Find new groups
    new_groups = [g for g in updated_groups if g not in initial_groups]
    print(f"New groups identified: {new_groups}")
    
    # Create new groups
    for group in new_groups:
        print(f"Creating new group: {group}")
        create_group(conn, group)
    
    # Assign new facts to groups
    print("\nAssigning new facts to groups...")
    for fact in new_facts:
        print(f"Processing fact: {fact}")
        # First insert the fact into the database
        insert_fact(conn, fact, path)
        # Then assign it to groups
        group_assignments = assign_groups_to_fact(fact, updated_groups, model="gpt-4o-mini", provider="openai")
        print(f"Assigned to groups: {group_assignments.get('groups', [])}")
        for group in group_assignments.get("groups", []):
            assign_fact_to_group_graph(conn, fact, group)
    
    print_facts_in_groups(conn)
    get_analytics(conn)
    
    # PHASE 3: Knowledge Graph Reorganization
    print_section("PHASE 3: KNOWLEDGE GRAPH REORGANIZATION AND EVOLUTION")
    
    print("Analyzing groups for potential subgrouping...")
    for group in updated_groups:
        analysis = analyze_group_for_subgrouping(conn, group, model="gpt-4o-mini", provider="openai")
        print(f"Group: {group}")
        print(f"Should split: {analysis.get('should_split', False)}")
        print(f"Reason: {analysis.get('reason', 'N/A')}")
        
        if analysis.get('should_split', False) and 'suggested_subgroups' in analysis:
            print("Implementing suggested subgroups:")
            for subgroup in analysis['suggested_subgroups']:
                subgroup_name = subgroup.get('name', '')
                if subgroup_name:
                    print(f"Creating subgroup: {subgroup_name}")
                    create_group(conn, subgroup_name)
                    
                    # Assign facts to this subgroup
                    for fact_content in subgroup.get('facts', []):
                        print(f"Assigning fact to subgroup: {fact_content[:50]}...")
                        assign_fact_to_group_graph(conn, fact_content, subgroup_name)
    
    print("\nSuggesting fact reassignments...")
    reassignments = suggest_fact_reassignments(conn, model="gpt-4o-mini", provider="openai")
    
    for suggestion in reassignments:
        if suggestion.get('needs_reassignment', False):
            fact = suggestion.get('fact', '')
            current_groups = suggestion.get('current_groups', [])
            suggested_groups = suggestion.get('suggested_groups', [])
            
            print(f"\nFact: {fact[:50]}...")
            print(f"Current groups: {current_groups}")
            print(f"Suggested groups: {suggested_groups}")
            print(f"Reason: {suggestion.get('reason', 'N/A')}")
            print(f"Confidence: {suggestion.get('confidence', 0)}")
            
            # Implement high-confidence reassignments
            if suggestion.get('confidence', 0) > 0.7:
                print("Implementing this high-confidence reassignment...")
                
                # Remove from groups not in the suggested list
                for group in current_groups:
                    if group not in suggested_groups:
                        # Note: We would need to implement a remove_fact_from_group function
                        # This is a placeholder for that functionality
                        print(f"  Would remove from: {group}")
                        
                # Add to new suggested groups
                for group in suggested_groups:
                    if group not in current_groups:
                        print(f"  Adding to: {group}")
                        assign_fact_to_group_graph(conn, fact, group)
    
    # PHASE 4: Final Analysis
    print_section("FINAL KNOWLEDGE GRAPH STATE")
    print_facts_in_groups(conn)
    get_analytics(conn)
    
    # Visualize the graph if available
    try:
        print("\nVisualizing the knowledge graph...")
        visualize_graph(conn)
        print("Graph visualization completed. Check for the generated image.")
    except Exception as e:
        print(f"Could not visualize graph: {str(e)}")
    
    # Close database connection
    conn.close()
    print(f"\nKnowledge graph test completed. Database saved at: {db_path}")
