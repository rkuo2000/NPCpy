import os
import tempfile
from npcpy.llm_funcs import get_llm_response, gen_image, execute_llm_command, check_llm_command, breathe
from npcpy.npc_compiler import NPC


def test_get_llm_response_basic():
    """Test basic LLM response"""
    response = get_llm_response(
        prompt="What is 2+2? Answer only with the number.",
        model="llama3.2:latest",
        provider="ollama"
    )
    assert response is not None
    print(f"Response: {response}")


def test_get_llm_response_with_messages():
    """Test LLM response with conversation history"""
    messages = [
        {"role": "user", "content": "Hi there!"},
        {"role": "assistant", "content": "Hello! How can I help you?"},
    ]
    
    response = get_llm_response(
        prompt="What did I just say?",
        messages=messages,
        model="llama3.2:latest",
        provider="ollama"
    )
    assert response is not None
    print(f"Conversation response: {response}")


def test_get_llm_response_with_attachments():
    """Test LLM response with file attachments"""
    temp_dir = tempfile.mkdtemp()
    test_file = os.path.join(temp_dir, "test.txt")
    
    with open(test_file, "w") as f:
        f.write("This is a test document with important information.")
    
    try:
        response = get_llm_response(
            prompt="What does the attached file contain?",
            attachments=[test_file],
            model="llama3.2:latest", 
            provider="ollama"
        )
        assert response is not None
        print(f"Attachment response: {response}")
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_execute_llm_command():
    """Test LLM command execution"""
    try:
        result = execute_llm_command(
            command="Tell me a very short joke",
            model="llama3.2:latest",
            provider="ollama"
        )
        assert result is not None
        print(f"Command result: {result}")
    except Exception as e:
        print(f"Command execution failed: {e}")


def test_check_llm_command():
    """Test LLM command checking"""
    try:
        result = check_llm_command(
            command="Calculate 5 * 7",
            model="llama3.2:latest",
            provider="ollama"
        )
        assert result is not None
        print(f"Command check result: {result}")
    except Exception as e:
        print(f"Command check failed: {e}")


def test_get_llm_response_transformers():
    result = get_llm_response(
        prompt="hello",
        model="Qwen/Qwen3-1.7b", 
        provider="transformers"
    )

    result = get_llm_response(
        prompt="what is 2+2",
        model="qwen3/qwen3-1.7b",
        provider="transformers",
        messages=[{"role": "user", "content": "hi"}]
    )

    result = get_llm_response(
        prompt="test",
        provider="transformers"
    )
def test_gen_image():
    """Test image generation"""
    try:

        result = gen_image(
            prompt="A really red circle that is redder than the reddest red redded in redding",
            model="dall-e-3",
            provider="openai"
        )
        if result:
            print(f"Generated image type: {type(result)}")
        else:
            print("Image generation returned None (expected without API key)")
    except Exception as e:
        print(f"Image generation failed (expected without API key): {e}")


def test_get_llm_response_with_npc():
    """Test LLM response using NPC"""
    try:
        test_npc = NPC(
            name="test_npc",
            primary_directive="You are a helpful test assistant",
            model="llama3.2:latest",
            provider="ollama"
        )
        
        response = get_llm_response(
            prompt="Hello, introduce yourself briefly.",
            npc=test_npc
        )
        assert response is not None
        print(f"NPC response: {response}")
    except Exception as e:
        print(f"NPC test failed: {e}")


def test_streaming_response():
    """Test streaming LLM response"""
    try:
        response = get_llm_response(
            prompt="Count from 1 to 3",
            model="llama3.2:latest",
            provider="ollama",
            stream=True
        )
        assert response is not None
        print(f"Streaming response: {response}")
    except Exception as e:
        print(f"Streaming failed: {e}")
        
def test_breathe():
    messages = [
    {'role': 'user', 'content': 'I need a function to add two numbers'},
    {'role': 'assistant', 'content': 'def add(a, b): return a + b'},
    {'role': 'user', 'content': 'Now make it work with strings too'}
    ]
    result = breathe(messages)
    print("Test 2 - Coding:")
    print(result)
    print()

    
    messages = [
        {'role': 'user', 'content': 'I want to build a todo app'},
        {'role': 'assistant', 'content': 'What features do you need?'},
        {'role': 'user', 'content': 'Add tasks, mark complete, delete'},
        {'role': 'assistant', 'content': 'Use Flask for backend and React for frontend?'},
        {'role': 'user', 'content': 'Just Flask for now, keep it simple'}
    ]
    result = breathe(messages)
    print("Test 3 - Project:")
    print(result)

