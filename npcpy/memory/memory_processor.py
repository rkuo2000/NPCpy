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

class MemoryApprovalQueue:
    def __init__(self, command_history):
        self.command_history = command_history
        self.pending_queue = queue.Queue()
        self.approval_results = queue.Queue()
        self.processing_thread = None
        self.running = False

    def add_memory(self, memory_item: MemoryItem):
        """Add memory to processing queue (non-blocking)"""
        self.pending_queue.put(memory_item)
        
    def start_background_processing(self):
        """Start background thread for memory processing"""
        if self.processing_thread and self.processing_thread.is_alive():
            return
            
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def _process_queue(self):
        """Background processing of memory queue"""
        while self.running:
            try:
                
                batch = []
                try:
                    
                    memory = self.pending_queue.get(timeout=1.0)
                    batch.append(memory)
                    
                    
                    while len(batch) < 10:
                        try:
                            memory = self.pending_queue.get_nowait()
                            batch.append(memory)
                        except queue.Empty:
                            break
                    
                    self._process_memory_batch(batch)
                    
                except queue.Empty:
                    continue
                    
            except Exception as e:
                print(f"Error in memory processing: {e}")
                time.sleep(1)

    def _process_memory_batch(self, memories: List[MemoryItem]):
        """Process a batch of memories"""
        for memory in memories:
            
            memory_id = self.command_history.add_memory_to_database(
                message_id=memory.message_id,
                conversation_id=memory.conversation_id,
                npc=memory.npc,
                team=memory.team,
                directory_path=memory.directory_path,
                initial_memory=memory.content,
                status="pending_approval",
                model=memory.model,
                provider=memory.provider
            )
            
            
            self.approval_results.put({
                "memory_id": memory_id,
                "content": memory.content,
                "context": memory.context,
                "npc": memory.npc
            })

    def get_approval_batch(self, max_items: int = 5) -> List[Dict]:
        """Get batch of memories ready for approval"""
        batch = []
        try:
            while len(batch) < max_items:
                item = self.approval_results.get_nowait()
                batch.append(item)
        except queue.Empty:
            pass
        return batch

    def stop_processing(self):
        """Stop background processing"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)

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