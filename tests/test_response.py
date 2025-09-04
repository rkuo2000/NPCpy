import os
import tempfile
import json
from npcpy.gen.response import get_litellm_response, get_ollama_response, process_tool_calls


def test_get_litellm_response_basic():
    """Test basic litellm response"""
    response = get_litellm_response(
        prompt="What is 1+1?",
        model="llama3.2:latest",
        provider="ollama"
    )
    assert response is not None
    assert "response" in response or "messages" in response
    print(f"LiteLLM response: {response}")


def test_get_ollama_response_basic():
    """Test basic ollama response"""
    messages = [{"role": "user", "content": "Say hello"}]
    response = get_ollama_response(
        prompt="Hello",
        model="llama3.2:latest",
        messages=messages
    )
    assert response is not None
    print(f"Ollama response: {response}")


def test_get_litellm_response_with_images():
    """Test litellm response with images"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='red')
        img_path = os.path.join(temp_dir, "test_image.png")
        img.save(img_path)
        
        response = get_litellm_response(
            prompt="What color is this image?",
            model="gemma3:4b",
            provider="ollama",
            images=[img_path]
        )
        assert response is not None
        print(f"Image response: {response}")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_get_litellm_response_with_tools():
    """Test litellm response with tools"""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Calculate math expressions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"}
                    }
                }
            }
        }
    ]
    
    response = get_litellm_response(
        prompt="Calculate 15 * 8",
        model="llama3.2:latest",
        provider="ollama",
        tools=tools
    )
    assert response is not None
    print(f"Tools response: {response}")


def test_get_litellm_response_json_format():
    """Test litellm response with JSON format"""
    response = get_litellm_response(
        prompt="Return a JSON object with name 'John' and age 30",
        model="llama3.2:latest",
        provider="ollama",
        format="json"
    )
    assert response is not None
    
    if "response" in response and response["response"]:
        
        assert isinstance(response["response"], (dict, list)), f"Expected dict/list, got {type(response['response'])}"
        print("JSON format test passed - response is already parsed")
    
    print(f"JSON response: {response}")


def test_get_litellm_response_streaming():
    """Test litellm streaming response"""
    response = get_litellm_response(
        prompt="Count from 1 to 5",
        model="llama3.2:latest",
        provider="ollama",
        stream=True
    )
    assert response is not None
    print(f"Streaming response: {type(response['response'])}")


def test_get_ollama_response_with_format():
    """Test ollama response with format specification"""
    messages = [{"role": "user", "content": "Return JSON with 'status': 'ok'"}]
    response = get_ollama_response(
        prompt=None,
        model="llama3.2:latest",
        messages=messages,
        format="json"
    )
    assert response is not None
    print(f"Ollama JSON response: {response}")


def test_process_tool_calls():
    """Test tool call processing"""
    
    response_dict = {
        "response": "I'll calculate that for you",
        "messages": [{"role": "assistant", "content": "Let me calculate"}],
        "tool_calls": [
            {
                "function": {
                    "name": "calculate",
                    "arguments": '{"expression": "2+2"}'
                }
            }
        ]
    }
    
    def mock_calculate(expression):
        return {"result": eval(expression)}
    
    tool_map = {"calculate": mock_calculate}
    
    result = process_tool_calls(
        response_dict,
        tool_map,
        "llama3.2:latest",
        "ollama",
        []
    )
    
    assert result is not None
    print(f"Tool call processing result: {result}")


def test_get_litellm_response_with_api_key():
    """Test litellm with API key"""
    response = get_litellm_response(
        prompt="What is AI?",
        model="llama3.2:latest",
        provider="ollama",
        api_key=None
    )
    
    assert response is not None
    print(f"API key test response: {response}")




def test_get_litellm_response_with_attachments():
    """Test litellm with file attachments"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("This is test content for attachment processing.")
        
        response = get_litellm_response(
            prompt="Summarize the attached file",
            model="llama3.2:latest",
            provider="ollama",
            attachments=[test_file]
        )
        assert response is not None
        print(f"Attachment response: {response}")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)
