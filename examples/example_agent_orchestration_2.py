import os
import sys

from npcpy.npc_compiler import NPC, Team

def test_multi_expertise_task():
    """Test a task that genuinely requires multiple types of expertise"""
    print("=== Multi-Expertise Task Test ===")
    
    # Create specialized NPCs
    coordinator = NPC(
        name="coordinator",
        primary_directive="""You coordinate projects but are not a technical expert. 
        Pass technical analysis tasks to the analyst and content creation to the writer.""",
        model="llama3.2",
        provider="ollama"
    )
    
    writer = NPC(
        name="writer", 
        primary_directive="""You are a technical writer who creates documentation. 
        Handle all writing and documentation tasks directly.""",
        model="llama3.2",
        provider="ollama"
    )
    
    analyst = NPC(
        name="analyst",
        primary_directive="""You are a data analyst. Analyze data, create insights, 
        and provide technical analysis. Handle all analytical tasks directly.""",
        model="llama3.2", 
        provider="ollama"
    )
    
    team = Team(npcs=[coordinator, writer, analyst], forenpc=coordinator)
    
    # Task that requires technical analysis (should pass to analyst)
    technical_request = """
    Analyze this sales data and provide insights:
    - Q1: $150K revenue, 1200 customers
    - Q2: $180K revenue, 1350 customers  
    - Q3: $165K revenue, 1280 customers
    - Q4: $220K revenue, 1500 customers
    
    Calculate growth rates, customer acquisition costs, and identify trends.
    """
    
    print(f"Technical Request: {technical_request}")
    print("\n" + "="*50)
    
    result = team.orchestrate(technical_request)
    
    print("Technical Analysis Result:")
    print(f"Output: {result.get('output', 'No output')[:200]}...")
    
    return result

if __name__ == "__main__":
    test_multi_expertise_task()