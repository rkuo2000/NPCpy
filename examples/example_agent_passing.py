import os
import sys
sys.path.append('/media/caug/extradrive1/npcww/npcpy')

from npcpy.npc_compiler import NPC, Team
import tempfile
import yaml

def test_simple_agent_passing():
    """Simple test to debug agent passing"""
    print("=== Debug Agent Passing ===")
    
    # Create NPCs directly without files
    coordinator = NPC(
        name="coordinator",
        primary_directive="You coordinate tasks between team members",
        model="llama3.2",
        provider="ollama"
    )
    
    writer = NPC(
        name="writer", 
        primary_directive="You write content and documentation",
        model="llama3.2",
        provider="ollama"
    )
    
    # Create team with NPCs
    team = Team(npcs=[coordinator, writer], forenpc=coordinator)
    
    print(f"Team NPCs: {list(team.npcs.keys())}")
    print(f"Forenpc: {team.get_forenpc().name if team.get_forenpc() else 'None'}")
    
    # Test direct agent passing
    request = "Please pass this task to the writer to create a simple document"
    
    result = coordinator.check_llm_command(request, team=team)
    print("Result:", result)

if __name__ == "__main__":
    test_simple_agent_passing()