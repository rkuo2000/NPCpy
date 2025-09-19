from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import threading
import queue
import time

@dataclass
class MemoryItem:
    message_id: str
    conversation_id: str
    npc: str
    team: str
    directory_path: str
    content: str
    context: str
    model: str
    provider: str


def memory_approval_ui(memories: List[Dict]) -> List[Dict]:
    """Simple CLI interface for memory approval"""
    if not memories:
        return []
    
    print(f"\n📝 {len(memories)} memories ready for approval:")
    
    approvals = []
    for i, memory in enumerate(memories, 1):
        print(f"\n--- Memory {i}/{len(memories)} ---")
        print(f"NPC: {memory['npc']}")
        print(f"Content: {memory['content'][:200]}{'...' if len(memory['content']) > 200 else ''}")
        
        while True:
            choice = input("(a)pprove, (r)eject, (e)dit, (s)kip, (q)uit, (A)pprove all: ").strip().lower()
            
            if choice == 'a':
                approvals.append({"memory_id": memory['memory_id'], "decision": "human-approved"})
                break
            elif choice == 'r':
                approvals.append({"memory_id": memory['memory_id'], "decision": "human-rejected"})
                break
            elif choice == 'e':
                edited = input("Edit memory: ").strip()
                if edited:
                    approvals.append({
                        "memory_id": memory['memory_id'], 
                        "decision": "human-edited",
                        "final_memory": edited
                    })
                break
            elif choice == 's':
                break
            elif choice == 'q':
                return approvals
            elif choice == 'A':
                
                for remaining_memory in memories[i-1:]:
                    approvals.append({
                        "memory_id": remaining_memory['memory_id'], 
                        "decision": "human-approved"
                    })
                return approvals
    
    return approvals