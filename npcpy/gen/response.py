from typing import Any, Dict, List, Union
from pydantic import BaseModel
from npcpy.data.image import compress_image
from npcpy.npc_sysenv import get_system_message, lookup_provider, render_markdown
import base64
import json
import uuid
import os 
try: 
    import ollama
except ImportError:
    pass
except OSError:
    # Handle case where ollama is not installed or not available
    print("Ollama is not installed or not available. Please install it to use this feature.")
try:
    from litellm import completion
except ImportError:
    pass
except OSError:
    # Handle case where litellm is not installed or not available
    pass

def handle_streaming_json(api_params):
    """
    Handles streaming responses when JSON format is requested from LiteLLM.
    """
    json_buffer = ""
    stream = completion(**api_params)
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            json_buffer += content
            try:
                json.loads(json_buffer)
                yield chunk
            except json.JSONDecodeError:
                pass


        
def get_ollama_response(
    prompt: str,
    model: str,
    images: List[str] = None,
    tools: list = None,
    tool_choice: Dict = None,
    tool_map: Dict = None,
    format: Union[str, BaseModel] = None,
    messages: List[Dict[str, str]] = None,
    stream: bool = False,
    attachments: List[str] = None,
    follow_up_with_llm: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generates a response using the Ollama API, supporting both streaming and non-streaming.
    """


    image_paths = []
    if images:
        image_paths.extend(images)
    
    if attachments:
        for attachment in attachments:
            if os.path.exists(attachment):
                _, ext = os.path.splitext(attachment)
                ext = ext.lower()
                
                if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                    image_paths.append(attachment)
                elif ext == '.pdf':
                    try:
                        from npcpy.data.load import load_pdf
                        pdf_data = load_pdf(attachment)
                        if pdf_data is not None:
                            texts = json.loads(pdf_data['texts'].iloc[0])
                            pdf_text = "\n\n".join([item.get('content', '') for item in texts])
                            if prompt:
                                prompt += f"\n\nContent from PDF: {os.path.basename(attachment)}\n{pdf_text[:2000]}..."
                            else:
                                prompt = f"Content from PDF: {os.path.basename(attachment)}\n{pdf_text[:2000]}..."
                    except Exception:
                        pass
                elif ext == '.csv':
                    try:
                        from npcpy.data.load import load_csv
                        csv_data = load_csv(attachment)
                        if csv_data is not None:
                            csv_sample = csv_data.head(10).to_string()
                            if prompt:
                                prompt += f"\n\nContent from CSV: {os.path.basename(attachment)} (first 10 rows):\n{csv_sample}"
                            else:
                                prompt = f"Content from CSV: {os.path.basename(attachment)} (first 10 rows):\n{csv_sample}"
                    except Exception:
                        pass

    if prompt:
        if messages and messages[-1]["role"] == "user":
            if isinstance(messages[-1]["content"], str):
                messages[-1]["content"] = prompt
            elif isinstance(messages[-1]["content"], list):
                for i, item in enumerate(messages[-1]["content"]):
                    if item.get("type") == "text":
                        messages[-1]["content"][i]["text"] = prompt
                        break
                else:
                    messages[-1]["content"].append({"type": "text", "text": prompt})
        else:
            messages.append({"role": "user", "content": prompt})
    if format == "json" and not stream:
        json_instruction = """If you are a returning a json object, begin directly with the opening {.
            If you are returning a json array, begin directly with the opening [.
            Do not include any additional markdown formatting or leading
            ```json tags in your response. The item keys should be based on the ones provided
            by the user. Do not invent new ones."""
            
        if messages and messages[-1]["role"] == "user":
            if isinstance(messages[-1]["content"], list):
                messages[-1]["content"].append({
                    "type": "text", 
                    "text": json_instruction
                })
            elif isinstance(messages[-1]["content"], str):
                messages[-1]["content"] += "\n" + json_instruction
                        
    if image_paths:
        last_user_idx = -1
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                last_user_idx = i
        if last_user_idx == -1:
            messages.append({"role": "user", "content": ""})
            last_user_idx = len(messages) - 1
        messages[last_user_idx]["images"] = image_paths
    
    api_params = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }

    
    if tools:
        api_params["tools"] = tools
    if tool_choice:
        api_params["tool_choice"] = tool_choice
    options = {}
    for key, value in kwargs.items():
        if key in [
            "stop", "temperature", "top_p", "max_tokens", "max_completion_tokens",
            "tools", "tool_choice", "extra_headers", "parallel_tool_calls",
            "response_format", "user",
        ]:
            options[key] = value

    if isinstance(format, type) and not stream:
        api_params["format"] = format.model_json_schema()
    elif isinstance(format, str) and format == "json" and not stream:
        api_params["format"] = "json"
    


    options = {}
    for key, value in kwargs.items():
        if key in [
            "stop", "temperature", "top_p", "max_tokens", "max_completion_tokens",
            "tools", "tool_choice", "extra_headers", "parallel_tool_calls",
            "response_format", "user",
        ]:
            options[key] = value

    if isinstance(format, type) and not stream:
        api_params["format"] = format.model_json_schema()
    elif isinstance(format, str) and format == "json" and not stream:
        api_params["format"] = "json"

    # Create standardized response structure
    result = {
        "response": None,
        "messages": messages.copy(),
        "raw_response": None,
        "tool_calls": []
    }
    
    # Handle streaming
    if stream:
        result["response"] = ollama.chat(**api_params, options=options)
        return result
    
    # Non-streaming case
    res = ollama.chat(**api_params, options = options)
    result["raw_response"] = res

    if tools and hasattr(res.get('message', {}), 'tool_calls') and res['message']['tool_calls']:
        if tool_map:
            response_for_processing = {
                "response": res['message'].get('content'),
                "raw_response": res,
                "messages": messages,
                "tool_calls": res['message']['tool_calls']
            }
            return process_tool_calls(response_for_processing, tool_map, model, 'ollama', messages, stream)
    
    final_messages = messages + [{"role": "assistant", "content": res.get('message', {}).get('content')}]
    return {
        "response": res.get('message', {}).get('content'),
        "raw_response": res,
        "messages": final_messages,
        "tool_calls": [],
        "tool_results": []
    }


def get_litellm_response(
    prompt: str = None,
    model: str = None,
    provider: str = None,
    images: List[str] = None,
    tools: list = None,
    tool_choice: Dict = None,
    tool_map: Dict = None,
    format: Union[str, BaseModel] = None,
    messages: List[Dict[str, str]] = None,
    api_key: str = None,
    api_url: str = None,
    stream: bool = False,
    attachments: List[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    result = {
        "response": None,
        "messages": messages.copy() if messages else [],
        "raw_response": None,
        "tool_calls": []
    }
    
    if provider == "ollama":
        kwargs["tool_map"] = tool_map
        return get_ollama_response(
            prompt, model, images=images, tools=tools, tool_choice=tool_choice,
            format=format, messages=messages, stream=stream, attachments=attachments, **kwargs
        )
    
    if format == "json" and not stream:
        json_instruction = """If you are a returning a json object, begin directly with the opening {.
            If you are returning a json array, begin directly with the opening [.
            Do not include any additional markdown formatting or leading
            ```json tags in your response. The item keys should be based on the ones provided
            by the user. Do not invent new ones."""
            
        if result["messages"] and result["messages"][-1]["role"] == "user":
            if isinstance(result["messages"][-1]["content"], list):
                result["messages"][-1]["content"].append({"type": "text", "text": json_instruction})
            elif isinstance(result["messages"][-1]["content"], str):
                result["messages"][-1]["content"] += "\n" + json_instruction
    
    if images:
        last_user_idx = -1
        for i, msg in enumerate(result["messages"]):
            if msg["role"] == "user":
                last_user_idx = i
        if last_user_idx == -1:
            result["messages"].append({"role": "user", "content": []})
            last_user_idx = len(result["messages"]) - 1
        if isinstance(result["messages"][last_user_idx]["content"], str):
            result["messages"][last_user_idx]["content"] = [{"type": "text", "text": result["messages"][last_user_idx]["content"]}]
        elif not isinstance(result["messages"][last_user_idx]["content"], list):
            result["messages"][last_user_idx]["content"] = []
        for image_path in images:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(compress_image(image_file.read())).decode("utf-8")
                result["messages"][last_user_idx]["content"].append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                )
    
    api_params = {"messages": result["messages"]}
    
    if api_url is not None and provider == "openai-like":
        api_params["api_base"] = api_url
        provider = "openai"
    
    if format == "json" and not stream:
        api_params["response_format"] = {"type": "json_object"}
    elif isinstance(format, BaseModel):
        api_params["response_format"] = format
    if model is None:
        model = os.environ.get("NPCSH_CHAT_MODEL", "llama3.2")
    if provider is None:
        provider = os.environ.get("NPCSH_CHAT_PROVIDER", "openai")

    api_params["model"] = f"{provider}/{model}" if "/" not in model else model
    if api_key is not None: api_params["api_key"] = api_key
    if tools: api_params["tools"] = tools
    if tool_choice: api_params["tool_choice"] = tool_choice
    
    if kwargs:
        for key, value in kwargs.items():
            if key in [
                "stop", "temperature", "top_p", "max_tokens", "max_completion_tokens",
                "tools", "tool_choice", "extra_headers", "parallel_tool_calls",
                "response_format", "user",
            ]:
                api_params[key] = value
    
    if tools and tool_map:
        resp = completion(**{**api_params, "stream": False})
        result["raw_response"] = resp
        if hasattr(resp.choices[0].message, 'tool_calls') and resp.choices[0].message.tool_calls:
            result["tool_calls"] = resp.choices[0].message.tool_calls
            return process_tool_calls(result, tool_map, model, provider, messages, stream)
    
    api_params["stream"] = stream
    resp = completion(**api_params)

    if stream:
        result["response"] = resp
    else:
        result["raw_response"] = resp
        llm_response = resp.choices[0].message.content
        result["response"] = llm_response
        result["messages"].append({"role": "assistant", "content": llm_response})

    return result


def process_tool_calls(response_dict, tool_map, model, provider, messages, stream=False):
    result = response_dict.copy()
    result["tool_results"] = []
    
    if "messages" not in result:
        result["messages"] = messages if messages else []
    
    tool_calls = result.get("tool_calls", [])
    
    if not tool_calls:
        return result

    for tool_call in tool_calls:
        tool_id = str(uuid.uuid4())
        tool_name = None
        arguments = {}

        if isinstance(tool_call, dict):
            tool_id = tool_call.get("id", str(uuid.uuid4()))
            tool_name = tool_call.get("function", {}).get("name")
            arguments_str = tool_call.get("function", {}).get("arguments", "{}")
        else:
            tool_id = getattr(tool_call, "id", str(uuid.uuid4()))
            if hasattr(tool_call, "function"):
                func_obj = tool_call.function
                tool_name = getattr(func_obj, "name", None)
                arguments_str = getattr(func_obj, "arguments", "{}")
            else:
                continue

        try:
            arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
        except json.JSONDecodeError:
            arguments = {"raw_arguments": arguments_str}

        if tool_name in tool_map:
            tool_result = None
            tool_result_str = ""
            serializable_result = None

            try:
                tool_result = tool_map[tool_name](**arguments)
            except Exception as e:
                tool_result = f"Error executing tool '{tool_name}': {str(e)}"

            try:
                tool_result_str = json.dumps(tool_result, default=str)
                try:
                    serializable_result = json.loads(tool_result_str)
                except json.JSONDecodeError:
                    serializable_result = {"result": tool_result_str}
            except Exception as e_serialize:
                tool_result_str = f"Error serializing result for {tool_name}: {str(e_serialize)}"
                serializable_result = {"error": tool_result_str}
            
            result["tool_results"].append({
                "tool_call_id": tool_id,
                "tool_name": tool_name,
                "arguments": arguments,
                "result": serializable_result
            })
            
            result["messages"].append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(arguments)
                        }
                    }
                ]
            })
            
            result["messages"].append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": tool_result_str
            })
    
    return result