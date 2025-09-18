from npcpy.npc_compiler import NPC
from sqlalchemy import create_engine
import tempfile

def test_write_code():
    npc = NPC(name="test", primary_directive="Execute code", model="llama3.2", provider="ollama")
    result = npc.write_code("x = 5 + 3; output = f'Result: {x}'")


def test_memory_search():
    db_path = tempfile.mktemp(suffix='.db')
    engine = create_engine(f'sqlite:///{db_path}')
    npc = NPC(name="test", primary_directive="Test memory", model="llama3.2", provider="ollama", db_conn=engine, memory=True)
    
    result = npc.search_my_conversations("test query")
    print(result)
    
    result = npc.search_my_memories("test memory")
    print(result)
test_write_code()
test_memory_search()
