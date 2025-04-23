
# pti
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