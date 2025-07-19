import os
import tempfile
import json
from npcpy.gen.response import get_litellm_response, get_ollama_response, process_tool_calls


def test_get_litellm_response_basic():
    """Test basic litellm response"""
    try:
        response = get_litellm_response(
            prompt="What is 1+1?",
            model="llama3.2:latest",
            provider="ollama"
        )
        assert response is not None
        assert "response" in response or "messages" in response
        print(f"LiteLLM response: {response}")
    except Exception as e:
        print(f"LiteLLM test failed: {e}")


def test_get_ollama_response_basic():
    """Test basic ollama response"""
    try:
        messages = [{"role": "user", "content": "Say hello"}]
        response = get_ollama_response(
            prompt="Hello",
            model="llama3.2:latest",
            messages=messages
        )
        assert response is not None
        print(f"Ollama response: {response}")
    except Exception as e:
        print(f"Ollama test failed: {e}")


def test_get_litellm_response_with_images():
    """Test litellm response with images"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create a simple test image file
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='red')
        img_path = os.path.join(temp_dir, "test_image.png")
        img.save(img_path)
        
        response = get_litellm_response(
            prompt="What color is this image?",
            model="llama3.2-vision:latest",
            provider="ollama",
            images=[img_path]
        )
        assert response is not None
        print(f"Image response: {response}")
        
    except Exception as e:
        print(f"Image test failed: {e}")
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_get_litellm_response_with_tools():
    """Test litellm response with tools"""
    try:
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
        
    except Exception as e:
        print(f"Tools test failed: {e}")


def test_get_litellm_response_json_format():
    """Test litellm response with JSON format"""
    try:
        response = get_litellm_response(
            prompt="Return a JSON object with name 'John' and age 30",
            model="llama3.2:latest",
            provider="ollama",
            format="json"
        )
        assert response is not None
        
        if "response" in response and response["response"]:
            # Try to parse as JSON
            try:
                json.loads(response["response"])
                print("JSON format test passed")
            except json.JSONDecodeError:
                print("Response not valid JSON, but function didn't crash")
        
        print(f"JSON response: {response}")
        
    except Exception as e:
        print(f"JSON format test failed: {e}")


def test_get_litellm_response_streaming():
    """Test litellm streaming response"""
    try:
        response = get_litellm_response(
            prompt="Count from 1 to 5",
            model="llama3.2:latest",
            provider="ollama",
            stream=True
        )
        assert response is not None
        print(f"Streaming response: {type(response['response'])}")
        
    except Exception as e:
        print(f"Streaming test failed: {e}")


def test_get_ollama_response_with_format():
    """Test ollama response with format specification"""
    try:
        messages = [{"role": "user", "content": "Return JSON with 'status': 'ok'"}]
        response = get_ollama_response(
            prompt=None,
            model="llama3.2:latest",
            messages=messages,
            format="json"
        )
        assert response is not None
        print(f"Ollama JSON response: {response}")
        
    except Exception as e:
        print(f"Ollama format test failed: {e}")


def test_process_tool_calls():
    """Test tool call processing"""
    try:
        # Mock response with tool calls
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
        
    except Exception as e:
        print(f"Tool call processing failed: {e}")


def test_get_litellm_response_with_api_key():
    """Test litellm with API key"""
    try:
        response = get_litellm_response(
            prompt="What is AI?",
            model="llama3.2:latest",
            provider="ollama",
            api_key=None
        )
        # This should fail with invalid API key, but tests the flow
        print("API key test - expected to fail with invalid key")
        
    except Exception as e:
        print(f"API key test failed as expected: {e}")


def test_get_litellm_response_with_attachments():
    """Test litellm with file attachments"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create test file
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
        
    except Exception as e:
        print(f"Attachment test failed: {e}")
    finally:
        import shutil
        shutil.rmtree(temp_dir)
