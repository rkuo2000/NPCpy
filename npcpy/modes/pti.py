
# pti
import json 
from typing import Dict, List, Optional, Any, Generator 
from npcpy.npc_sysenv import (NPCSH_REASONING_MODEL, NPCSH_REASONING_PROVIDER, NPCSH_CHAT_MODEL, NPCSH_CHAT_PROVIDER, NPCSH_API_URL)
def request_user_input(input_request: Dict[str, str]) -> str:
    """
    Request and get input from user.

    Args:
        input_request: Dict with reason and prompt for input

    Returns:
        User's input text
    """
    print(f"\nAdditional input needed: {input_request['reason']}")
    return input(f"{input_request['prompt']}: ")





def handle_request_input(
    context: str,
    model: str = NPCSH_CHAT_MODEL,
    provider: str = NPCSH_CHAT_PROVIDER,
    whisper: bool = False,
):
    """
    Analyze text and decide what to request from the user
    """
    prompt = f"""
    Analyze the text:
    {context}
    and determine what additional input is needed.
    Return a JSON object with:
    {{
        "input_needed": boolean,
        "request_reason": string explaining why input is needed,
        "request_prompt": string to show user if input needed
    }}

    Do not include any additional markdown formatting or leading ```json tags. Your response
    must be a valid JSON object.
    """

    response = get_llm_response(
        prompt,
        model=model,
        provider=provider,
        messages=[],
        format="json",
    )

    result = response.get("response", {})
    if isinstance(result, str):
        result = json.loads(result)

    user_input = request_user_input(
        {"reason": result["request_reason"], "prompt": result["request_prompt"]},
    )
    return user_input


def analyze_thoughts_for_input(
    thought_text: str,
    model: str = NPCSH_CHAT_MODEL,
    provider: str = NPCSH_CHAT_PROVIDER,
    api_url: str = NPCSH_API_URL,
    api_key: str = None,
) -> Optional[Dict[str, str]]:
    """
    Analyze accumulated thoughts to determine if user input is needed.

    Args:
        thought_text: Accumulated text from think block
        messages: Conversation history

    Returns:
        Dict with input request details if needed, None otherwise
    """

    prompt = (
        f"""
         Analyze these thoughts:
         {thought_text}
         and determine if additional user input would be helpful.
        Return a JSON object with:"""
        + """
        {
            "input_needed": boolean,
            "request_reason": string explaining why input is needed,
            "request_prompt": string to show user if input needed
        }
        Consider things like:
        - Ambiguity in the user's request
        - Missing context that would help provide a better response
        - Clarification needed about user preferences/requirements
        Only request input if it would meaningfully improve the response.
        Do not include any additional markdown formatting or leading ```json tags. Your response
        must be a valid JSON object.
        """
    )

    response = get_llm_response(
        prompt,
        model=model,
        provider=provider,
        api_url=api_url,
        api_key=api_key,
        messages=[],
        format="json",
    )

    result = response.get("response", {})
    if isinstance(result, str):
        result = json.loads(result)

    if result.get("input_needed"):
        return {
            "reason": result["request_reason"],
            "prompt": result["request_prompt"],
        }
def enter_reasoning_human_in_the_loop(
    messages: List[Dict[str, str]],
    reasoning_model: str = NPCSH_REASONING_MODEL,
    reasoning_provider: str = NPCSH_REASONING_PROVIDER,
    chat_model: str = NPCSH_CHAT_MODEL,
    chat_provider: str = NPCSH_CHAT_PROVIDER,
    npc: Any = None,
    answer_only: bool = False,
    context=None,
) -> Generator[str, None, None]:
    """
    Stream responses while checking for think tokens and handling human input when needed.

    Args:
        messages: List of conversation messages
        model: LLM model to use
        provider: Model provider
        npc: NPC instance if applicable

    Yields:
        Streamed response chunks
    """
    # Get the initial stream
    if answer_only:
        messages[-1]["content"] = (
            messages[-1]["content"].replace(
                "Think first though and use <think> tags", ""
            )
            + " Do not think just answer. "
        )
    else:
        messages[-1]["content"] = (
            messages[-1]["content"]
            + "         Think first though and use <think> tags.  "
        )

    response_stream = get_stream(
        messages,
        model=reasoning_model,
        provider=reasoning_provider,
        npc=npc,
        context=context,
    )

    thoughts = []
    response_chunks = []
    in_think_block = False
    for chunk in response_stream:
        # Check for user interrupt
        
        try:
            # Extract content based on provider
            if reasoning_provider == "ollama":
                chunk_content = chunk.get("message", {}).get("content", "")
            elif reasoning_provider == "openai" or reasoning_provider == "deepseek":
                chunk_content = "".join(
                    choice.delta.content
                    for choice in chunk.choices
                    if choice.delta.content is not None
                )
            elif reasoning_provider == "anthropic":
                if chunk.type == "content_block_delta":
                    chunk_content = chunk.delta.text
                else:
                    chunk_content = ""
            else:
                chunk_content = str(chunk)

            response_chunks.append(chunk_content)
            combined_text = "".join(response_chunks)

            # Check for LLM request block
            if (
                "<request_for_input>" in combined_text
                and "</request_for_input>" not in combined_text
            ):
                in_think_block = True

            if in_think_block:
                thoughts.append(chunk_content)
                yield chunk

            if "</request_for_input>" in combined_text:
                # Process the LLM's input request
                request_text = "".join(thoughts)
                yield "\nPlease provide the requested information: "

                # Wait for user input (blocking here is OK since we explicitly asked)
                user_input = input()

                # Add the interaction to messages and restart stream
                messages.append({"role": "assistant", "content": request_text})
                messages.append({"role": "user", "content": user_input})

                yield "\n[Continuing with provided information...]\n"
                yield from enter_reasoning_human_in_the_loop(
                    messages,
                    reasoning_model=reasoning_model,
                    reasoning_provider=reasoning_provider,
                    chat_model=chat_model,
                    chat_provider=chat_provider,
                    npc=npc,
                    answer_only=True,
                )
                return

            if not in_think_block:
                yield chunk

            
        except KeyboardInterrupt:        
            user_interrupt = input("\n[Stream interrupted by user]\n Enter your additional input: ")
            

            # Add the interruption to messages and restart stream
            messages.append(
                {"role": "user", "content": f"[INTERRUPT] {user_interrupt}"}
            )

            print(f"\n[Continuing with added context...]\n")
            yield from enter_reasoning_human_in_the_loop(
                messages,
                reasoning_model=reasoning_model,
                reasoning_provider=reasoning_provider,
                chat_model=chat_model,
                chat_provider=chat_provider,
                npc=npc,
                answer_only=True,
            )
            return