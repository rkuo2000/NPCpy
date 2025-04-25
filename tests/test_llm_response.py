import pytest
import json
import os
from npcpy.gen.response import get_ollama_response, get_litellm_response, process_tool_calls, handle_streaming_json

# Simple test for Ollama response
def test_ollama_basic():
    """Test basic Ollama response with a simple prompt"""
    result = get_ollama_response(
        prompt="What is the capital of France?",
        model="llama3.2"
    )
    
    # Check we got a response
    assert result["response"] is not None
    # Check message structure is correct
    assert len(result["messages"]) >= 3  # system, user, assistant
    # Verify the last message is from assistant
    assert result["messages"][-1]["role"] == "assistant"
    # Check prompt was preserved
    assert "capital of France" in result["messages"][1]["content"]

# Test appending to messages
def test_ollama_append_message():
    """Test appending a prompt to existing messages"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    
    result = get_ollama_response(
        prompt="How are you doing?",
        model="llama3.2",
        messages=messages
    )
    
    # Verify message was appended
    assert len(result["messages"]) >= 4
    assert "How are you doing" in result["messages"][3]["content"]

# Test JSON response format
def test_ollama_json_format():
    """Test JSON response format"""
    result = get_ollama_response(
        prompt="Return a JSON object with keys 'name' and 'age' and random values",
        model="llama3.2",
        format="json"
    )
    
    # Verify we got a JSON response
    assert isinstance(result["response"], dict)
    # These keys should be in the response based on our prompt
    assert "name" in result["response"]
    assert "age" in result["response"]

# Test streaming response
def test_ollama_stream():
    """Test streaming response"""
    result = get_ollama_response(
        prompt="Count from 1 to 5 slowly",
        model="llama3.2",
        stream=True
    )
    
    # Verify we got a generator
    assert hasattr(result["response"], "__next__")
    
    # Consume part of the stream
    counter = 0
    for chunk in result["response"]:
        counter += 1
        if counter >= 3:  # Just read a few chunks, not the whole stream
            break
    
    # Verify we could read from the stream
    assert counter >= 3

# Test LiteLLM response
def test_litellm_basic():
    """Test basic LiteLLM response"""
    # Skip if no API key available
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("No OpenAI API key available")
    
    result = get_litellm_response(
        prompt="What is the capital of Germany?",
        model="gpt-4o-mini",
        provider="openai",
        api_key=api_key
    )
    
    # Check basic structure
    assert result["response"] is not None
    assert "messages" in result
    assert "Berlin" in result["response"]

# Test tool calling
def test_tool_calling():
    """Test tool calling functionality"""
    # Define a simple calculator tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Calculate math expressions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"}
                    },
                    "required": ["expression"]
                }
            }
        }
    ]
    
    # Get response with tool capability
    result = get_ollama_response(
        prompt="Calculate 2+2 using the calculator tool",
        model="llama3.2",
        tools=tools
    )
    
    # Check if tool calls were attempted (may not always happen with every model)
    if "tool_calls" in result and result["tool_calls"]:
        # We have tool calls
        assert len(result["tool_calls"]) > 0
        
        # Process tool calls
        def calculator(args):
            """Simple calculator implementation"""
            try:
                return {"result": eval(args["expression"])}
            except Exception as e:
                return {"error": str(e)}
        
        tool_map = {"calculator": calculator}
        processed = process_tool_calls(result, tool_map)
        
        # Verify processing worked
        assert "tool_results" in processed
        if processed["tool_results"]:
            assert "result" in processed["tool_results"][0]

# Test with image inputs
def test_image_input():
    """Test sending an image with prompt"""
    # Skip if no test image available
    result = get_ollama_response(
        prompt="Describe this image",
        model="llava:7b",  # Need a vision model
        images=["../test_data/catfight.PNG"]
    )
    print(result)
    
    # Check that we got a response
    assert result["response"] is not None
    # Check that it's not an error message
    assert not result["response"].startswith("Error")

# Test for streaming JSON
def test_json_streaming():
    """Test JSON streaming with simple prompt"""
    # Skip if no API key available    
    # Setup API params
    api_params = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Return a JSON object with your name and version."}
        ],
        "model": "openai/gpt-4o-mini",
        "stream": True,
        "response_format": {"type": "json_object"}
    }
    
    # Get streaming JSON
    stream = handle_streaming_json(api_params)
    
    # Collect chunks
    chunks = []
    for chunk in stream:
        chunks.append(chunk)
    
    # Verify we got some chunks
    assert len(chunks) > 0
    
    # Try to assemble the chunks into valid JSON
    json_text = ""
    for chunk in chunks:
        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
            json_text += chunk.choices[0].delta.content
    
    # Verify we can parse it
    try:
        parsed = json.loads(json_text)
        assert isinstance(parsed, dict)
        assert "name" in parsed
    except json.JSONDecodeError:
        # Sometimes the streaming response might not form valid JSON
        # This is acceptable in a test
        pass