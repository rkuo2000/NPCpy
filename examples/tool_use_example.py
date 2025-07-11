from npcpy.llm_funcs import get_llm_response
import json
import os
from npcpy.npc_sysenv import print_and_process_stream_with_markdown

# Create dummy files for simulation if they don't exist
if not os.path.exists("reports/chart_image.png"):
    os.makedirs("reports", exist_ok=True)
    with open("reports/chart_image.png", "w") as f:
        f.write("dummy image content")

if not os.path.exists("documents/config.json"):
    os.makedirs("documents", exist_ok=True)
    with open("documents/config.json", "w") as f:
        f.write('{"sample": "document content"}')

def translate_text(text: str, target_language: str) -> str:
    if target_language.lower() == "spanish":
        return f"Simulated Spanish translation of '{text}': Hola!"
    elif target_language.lower() == "french":
        return f"Simulated French translation of '{text}': Bonjour!"
    return f"Translation for '{text}' to {target_language} not available."

def analyze_sentiment(text: str) -> str:
    text_lower = text.lower()
    if "happy" in text_lower or "great" in text_lower or "clear" in text_lower:
        return "Positive"
    elif "sad" in text_lower or "unhappy" in text_lower:
        return "Negative"
    return "Neutral"

def summarize_data(data_content: str, max_words: int = 50) -> str:
    words = data_content.split()
    if len(words) > max_words:
        return f"Simulated Summary ({max_words} words): {' '.join(words[:max_words])}..."
    return f"Simulated Summary: {data_content}"

def image_recognition(image_path: str) -> dict:
    if os.path.exists(image_path):
        return {"identified_objects": ["chart", "data points"], "scenes": ["report"], "image_path": image_path}
    return {"error": f"Image file not found at {image_path}"}

def text_to_speech(text_to_convert: str) -> str:
    return f"Simulated audio file generated for: '{text_to_convert}' (e.g., /tmp/audio.mp3)"

def parse_documents(document_content: str, format_type: str = "text") -> dict:
    if format_type == "json":
        try:
            return json.loads(document_content)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON content provided for parsing."}
    return {"extracted_text_start": document_content[:100], "total_length": len(document_content)}

def review_code(code_snippet: str) -> dict:
    suggestions = []
    if "print(" in code_snippet:
        suggestions.append("Consider using logging instead of print for production.")
    if "  " in code_snippet and "    " not in code_snippet:
        suggestions.append("Ensure consistent indentation (e.g., 4 spaces).")
    if "total / len(numbers)" in code_snippet and "len(numbers) == 0" not in code_snippet:
        suggestions.append("Add a check to prevent division by zero if 'numbers' can be empty.")
    return {"review_status": "Completed", "suggestions": suggestions if suggestions else ["No major issues found."]}

def organize_notes(notes_raw_text: str, category: str = "general") -> dict:
    action_items = [line.strip() for line in notes_raw_text.split('\n') if "TODO:" in line or "ACTION:" in line]
    return {"category": category, "action_items": action_items, "summary_length": len(notes_raw_text)}

tool_map = {
    "translate_text": translate_text,
    "analyze_sentiment": analyze_sentiment,
    "summarize_data": summarize_data,
    "image_recognition": image_recognition,
    "text_to_speech": text_to_speech,
    "parse_documents": parse_documents,
    "review_code": review_code,
    "organize_notes": organize_notes
}

tools_schema_for_llm = [
    {"type": "function", "function": {"name": "translate_text", "description": "Translates text.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}}, "required": ["text", "target_language"]}}},
    {"type": "function", "function": {"name": "analyze_sentiment", "description": "Analyzes sentiment.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "summarize_data", "description": "Summarizes data.", "parameters": {"type": "object", "properties": {"data_content": {"type": "string"}, "max_words": {"type": "integer"}}, "required": ["data_content"]}}},
    {"type": "function", "function": {"name": "image_recognition", "description": "Recognizes images.", "parameters": {"type": "object", "properties": {"image_path": {"type": "string"}}, "required": ["image_path"]}}},
    {"type": "function", "function": {"name": "text_to_speech", "description": "Converts text to speech.", "parameters": {"type": "object", "properties": {"text_to_convert": {"type": "string"}}, "required": ["text_to_convert"]}}},
    {"type": "function", "function": {"name": "parse_documents", "description": "Parses documents.", "parameters": {"type": "object", "properties": {"document_content": {"type": "string"}, "format_type": {"type": "string", "enum": ["text", "json"]}}, "required": ["document_content"]}}},
    {"type": "function", "function": {"name": "review_code", "description": "Reviews code.", "parameters": {"type": "object", "properties": {"code_snippet": {"type": "string"}}, "required": ["code_snippet"]}}},
    {"type": "function", "function": {"name": "organize_notes", "description": "Organizes notes.", "parameters": {"type": "object", "properties": {"notes_raw_text": {"type": "string"}, "category": {"type": "string"}}, "required": ["notes_raw_text"]}}}
]

query = """
I need assistance with several tasks for my data analysis project:
1. Translate the phrase 'How is the data looking?' to Spanish.
2. Analyze the sentiment of the user feedback: 'This report is absolutely great and very clear!'.
3. Summarize a long analysis document that starts with 'This document details...'.
4. Perform object recognition on an image located at 'reports/chart_image.png'.
5. Convert the concluding remark 'The project is on track for completion.' to an audio file.
6. Parse the following JSON configuration string: '{"server": "prod", "port": 8080}'.
7. Review this Python function: 'def calculate_average(numbers): total=sum(numbers); return total / len(numbers)'.
8. Organize my raw notes from the last team sync-up: 'Attendees: John, Jane. ACTION: Jane to update dashboard.'.
"""

# --- Non-Streaming Examples ---
print("--- STARTING NON-STREAMING EXAMPLES ---")

# Ollama
results_ollama = get_llm_response(prompt=query, model="qwen3:latest", provider="ollama", messages=[{"role": "user", "content": query}], tools=tools_schema_for_llm, tool_map=tool_map, stream=False)
print("\n--- Ollama Results (Non-Streaming) ---")
print("Final LLM Response:", results_ollama.get("response"))
print("Tool Results:", json.dumps(results_ollama.get("tool_results", []), indent=2))

# OpenAI
results_openai = get_llm_response(prompt=query, model="gpt-4o-mini", provider="openai", messages=[{"role": "user", "content": query}], tools=tools_schema_for_llm, tool_map=tool_map, stream=False)
print("\n--- OpenAI Results (Non-Streaming) ---")
print("Final LLM Response:", results_openai.get("response"))
print("Tool Results:", json.dumps(results_openai.get("tool_results", []), indent=2))

# Google Gemini
results_gemini = get_llm_response(prompt=query, model="gemini-2.0-flash", provider="gemini", messages=[{"role": "user", "content": query}], tools=tools_schema_for_llm, tool_map=tool_map, stream=False)
print("\n--- Gemini Results (Non-Streaming) ---")
print("Final LLM Response:", results_gemini.get("response"))
print("Tool Results:", json.dumps(results_gemini.get("tool_results", []), indent=2))

# Anthropic Claude
results_claude = get_llm_response(prompt=query, model="claude-3-5-haiku-latest", provider="anthropic", messages=[{"role": "user", "content": query}], tools=tools_schema_for_llm, tool_map=tool_map, stream=False)
print("\n--- Claude Results (Non-Streaming) ---")
print("Final LLM Response:", results_claude.get("response"))
print("Tool Results:", json.dumps(results_claude.get("tool_results", []), indent=2))

# DeepSeek
results_deepseek = get_llm_response(prompt=query, model="deepseek-chat", provider="deepseek", messages=[{"role": "user", "content": query}], tools=tools_schema_for_llm, tool_map=tool_map, stream=False)
print("\n--- DeepSeek Results (Non-Streaming) ---")
print("Final LLM Response:", results_deepseek.get("response"))
print("Tool Results:", json.dumps(results_deepseek.get("tool_results", []), indent=2))

