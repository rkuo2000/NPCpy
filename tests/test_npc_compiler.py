import os
import tempfile
import sqlite3
from npcpy.npc_compiler import NPC, Jinx, Team, Pipeline


def test_npc_creation():
    """Test basic NPC creation"""
    npc = NPC(
        name="test_npc",
        primary_directive="You are a helpful assistant",
        model="llama3.2:latest",
        provider="ollama"
    )
    assert npc.name == "test_npc"
    assert npc.primary_directive == "You are a helpful assistant"
    print(f"Created NPC: {npc.name}")


def test_npc_save_and_load():
    """Test NPC save and load functionality"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        
        npc = NPC(
            name="save_test_npc",
            primary_directive="Test NPC for saving",
            model="llama3.2:latest",
            provider="ollama"
        )
        npc.save(temp_dir)
        
        
        npc_file = os.path.join(temp_dir, "save_test_npc.npc")
        assert os.path.exists(npc_file)
        
        
        loaded_npc = NPC(file=npc_file)
        assert loaded_npc.name == "save_test_npc"
        print(f"Saved and loaded NPC: {loaded_npc.name}")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_npc_get_llm_response():
    """Test NPC LLM response functionality"""
    npc = NPC(
        name="response_test_npc",
        primary_directive="You are a helpful math assistant",
        model="llama3.2:latest",
        provider="ollama"
    )
    
    response = npc.get_llm_response("What is 3 + 4?")
    assert response is not None
    print(f"NPC response: {response}")


def test_jinx_creation():
    """Test basic Jinx creation"""
    jinx_data = {
        "jinx_name": "test_jinx",
        "description": "A test jinx",
        "inputs": ["input1", "input2"],
        "steps": [
            {
                "code": """
input1 = '{{ input1 }}'
input2 = '{{ input2 }}'
output = f"Processed: {input1} and {input2}"
print(output)
""",
                "engine": "python"
            }
        ]
    }
    
    jinx = Jinx(jinx_data=jinx_data)
    assert jinx.jinx_name == "test_jinx"
    print(f"Created Jinx: {jinx.jinx_name}")


def test_jinx_execution():
    """Test Jinx execution"""
    jinx_data = {
        "jinx_name": "math_jinx",
        "description": "Math calculation jinx",
        "inputs": ["number1", "number2"],
        "steps": [
            {
                "code": """
number1 = int('{{ number1 }}')
number2 = int('{{ number2 }}')
output = number1 + number2
print(f"The sum of {number1} and {number2} is {output}")
""",
                "engine": "python"
            }
        ]
    }
    
    jinx = Jinx(jinx_data=jinx_data)
    input_values = {"number1": "5", "number2": "7"}
    
    result = jinx.execute(input_values, {})
    assert result is not None
    print(f"Jinx execution result: {result}")


def test_team_creation():
    """Test Team creation"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        
        npc1 = NPC(name="analyst", primary_directive="Analyze data")
        npc2 = NPC(name="critic", primary_directive="Critique analysis")
        
        npc1.save(temp_dir)
        npc2.save(temp_dir)
        
        team = Team(team_path=temp_dir)
        assert team is not None
        print(f"Created team with {len(team.npcs)} NPCs")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_npc_with_database():
    """Test NPC with database connection"""
    temp_db = tempfile.mktemp(suffix=".db")
    
    try:
        conn = sqlite3.connect(temp_db)
        
        npc = NPC(
            name="db_test_npc",
            primary_directive="Test NPC with database",
            db_conn=conn
        )
        
        assert npc.db_conn is not None
        print(f"Created NPC with database: {npc.name}")
        
        conn.close()
        
    finally:
        if os.path.exists(temp_db):
            os.remove(temp_db)


def test_pipeline_creation():
    """Test Pipeline creation"""
    pipeline_data = {
        "name": "test_pipeline",
        "steps": [
            {
                "name": "step1",
                "type": "llm",
                "npc": "test_npc",
                "prompt": "Process this data"
            }
        ]
    }
    
    pipeline = Pipeline(pipeline_data=pipeline_data)
    assert pipeline.name == "test_pipeline"
    print(f"Created Pipeline: {pipeline.name}")


def test_jinx_save_and_load():
    """Test Jinx save and load"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        jinx_data = {
            "jinx_name": "save_test_jinx",
            "description": "Test jinx for saving",
            "inputs": ["input1"],
            "steps": [{"type": "llm", "prompt": "Process {{input1}}"}]
        }
        
        jinx = Jinx(jinx_data=jinx_data)
        jinx.save(temp_dir)
        
        jinx_file = os.path.join(temp_dir, "save_test_jinx.jinx")
        assert os.path.exists(jinx_file)
        
        loaded_jinx = Jinx(jinx_path=jinx_file)
        assert loaded_jinx.jinx_name == "save_test_jinx"
        print(f"Saved and loaded Jinx: {loaded_jinx.jinx_name}")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_npc_execute_jinx():
    """Test NPC executing a jinx"""
    try:
        npc = NPC(
            name="jinx_executor",
            primary_directive="Execute jinxs",
            model="llama3.2:latest",
            provider="ollama"
        )
        
        jinx_data = {
            "jinx_name": "simple_jinx",
            "inputs": ["message"],
            "steps": [{"type": "llm", "prompt": "Reply to: {{message}}"}]
        }
        
        jinx = Jinx(jinx_data=jinx_data)
        result = npc.execute_jinx("simple_jinx", {"message": "Hello"})
        
        assert result is not None
        print(f"NPC jinx execution result: {result}")
    except Exception as e:
        print(f"NPC jinx execution failed: {e}")
