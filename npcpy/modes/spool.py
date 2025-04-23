def enter_spool_mode(
    inherit_last: int = 0,
    model: str = None,
    provider: str = None,
    npc: Any = None,
    files: List[str] = None,  # New files parameter
    rag_similarity_threshold: float = 0.3,
    device: str = "cpu",
    messages: List[Dict] = None,
    conversation_id: str = None,
    stream: bool = False,
) -> Dict:
    """
    Function Description:
        This function is used to enter the spool mode where files can be loaded into memory.
    Args:

        inherit_last : int : The number of last commands to inherit.
        npc : Any : The NPC object.
        files : List[str] : List of file paths to load into the context.
    Returns:
        Dict : The messages and output.

    """

    command_history = CommandHistory()
    npc_info = f" (NPC: {npc.name})" if npc else ""
    print(f"Entering spool mode{npc_info}. Type '/sq' to exit spool mode.")

    spool_context = (
        messages.copy() if messages else []
    )  # Initialize context with messages

    loaded_content = {}  # New dictionary to hold loaded content

    # Create conversation ID if not provided
    if not conversation_id:
        conversation_id = start_new_conversation()

    command_history = CommandHistory()
    # Load specified files if any
    if files:
        for file in files:
            extension = os.path.splitext(file)[1].lower()
            try:
                if extension == ".pdf":
                    content = load_pdf(file)["texts"].iloc[0]
                elif extension == ".csv":
                    content = load_csv(file)
                else:
                    print(f"Unsupported file type: {file}")
                    continue
                loaded_content[file] = content
                print(f"Loaded content from: {file}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")

    # Add system message to context
    system_message = get_system_message(npc) if npc else "You are a helpful assistant."
    if len(spool_context) > 0:
        if spool_context[0]["role"] != "system":
            spool_context.insert(0, {"role": "system", "content": system_message})
    else:
        spool_context.append({"role": "system", "content": system_message})
    # Inherit last n messages if specified
    if inherit_last > 0:
        last_commands = command_history.get_all(limit=inherit_last)
        for cmd in reversed(last_commands):
            spool_context.append({"role": "user", "content": cmd[2]})
            spool_context.append({"role": "assistant", "content": cmd[4]})

    if npc is not None:
        if model is None:
            model = npc.model
        if provider is None:
            provider = npc.provider

    while True:
        try:
            user_input = input("spool> ").strip()
            if len(user_input) == 0:
                continue
            if user_input.lower() == "/sq":
                print("Exiting spool mode.")
                break
            if user_input.lower() == "/rehash":  # Check for whisper command
                # send the most recent message
                print("Rehashing last message...")
                output = rehash_last_message(
                    conversation_id,
                    model=model,
                    provider=provider,
                    npc=npc,
                    stream=stream,
                )
                print(output["output"])
                messages = output.get("messages", [])
                output = output.get("output", "")

            if user_input.lower() == "/whisper":  # Check for whisper command
                messages = enter_whisper_mode(spool_context, npc)
                # print(messages)  # Optionally print output from whisper mode
                continue  # Continue with spool mode after exiting whisper mode

            if user_input.startswith("/ots"):
                command_parts = user_input.split()
                file_path = None
                filename = None

                # Handle image loading/capturing
                if len(command_parts) > 1:
                    filename = command_parts[1]
                    file_path = os.path.join(os.getcwd(), filename)
                else:
                    output = capture_screenshot(npc=npc)
                    if output and "file_path" in output:
                        file_path = output["file_path"]
                        filename = output["filename"]

                if not file_path or not os.path.exists(file_path):
                    print(f"Error: Image file not found at {file_path}")
                    continue

                # Get user prompt about the image
                user_prompt = input(
                    "Enter a prompt for the LLM about this image (or press Enter to skip): "
                )

                # Read image file as binary data
                try:
                    with open(file_path, "rb") as img_file:
                        img_data = img_file.read()

                    # Create an attachment for the image
                    image_attachment = {
                        "name": filename,
                        "type": guess_mime_type(filename),
                        "data": img_data,
                        "size": len(img_data),
                    }

                    # Save user message with image attachment
                    message_id = save_conversation_message(
                        command_history,
                        conversation_id,
                        "user",
                        (
                            user_prompt
                            if user_prompt
                            else f"Please analyze this image: {filename}"
                        ),
                        wd=os.getcwd(),
                        model=model,
                        provider=provider,
                        npc=npc.name if npc else None,
                        attachments=[image_attachment],
                    )

                    # Now use analyze_image which will process the image
                    output = analyze_image(
                        command_history,
                        user_prompt,
                        file_path,
                        filename,
                        npc=npc,
                        stream=stream,
                        message_id=message_id,  # Pass the message ID for reference
                    )

                    # Save assistant's response
                    if output and isinstance(output, str):
                        save_conversation_message(
                            command_history,
                            conversation_id,
                            "assistant",
                            output,
                            wd=os.getcwd(),
                            model=model,
                            provider=provider,
                            npc=npc.name if npc else None,
                        )

                    # Update spool context with this exchange
                    spool_context.append(
                        {"role": "user", "content": user_prompt, "image": file_path}
                    )
                    spool_context.append({"role": "assistant", "content": output})

                    if isinstance(output, dict) and "filename" in output:
                        message = f"Screenshot captured: {output['filename']}\nFull path: {output['file_path']}\nLLM-ready data available."
                    else:
                        message = output

                    render_markdown(
                        output["response"]
                        if isinstance(output["response"], str)
                        else str(output["response"])
                    )
                    continue

                except Exception as e:
                    print(f"Error processing image: {str(e)}")
                    continue

            # Prepare kwargs for get_conversation
            kwargs_to_pass = {}
            if npc:
                kwargs_to_pass["npc"] = npc
                if npc.model:
                    kwargs_to_pass["model"] = npc.model

                if npc.provider:
                    kwargs_to_pass["provider"] = npc.provider

            # Incorporate the loaded content into the prompt for conversation
            if loaded_content:
                context_content = ""
                for filename, content in loaded_content.items():
                    # now do a rag search with the loaded_content
                    retrieved_docs = rag_search(
                        user_input,
                        content,
                        similarity_threshold=rag_similarity_threshold,
                        device=device,
                    )
                    if retrieved_docs:
                        context_content += (
                            f"\n\nLoaded content from: {filename}\n{content}\n\n"
                        )
                if len(context_content) > 0:
                    user_input += f"""
                    Here is the loaded content that may be relevant to your query:
                        {context_content}
                    Please reference it explicitly in your response and use it for answering.
                    """

            # Add user input to spool context
            spool_context.append({"role": "user", "content": user_input})

            # Save user message to conversation history
            message_id = save_conversation_message(
                command_history,
                conversation_id,
                "user",
                user_input,
                wd=os.getcwd(),
                model=model,
                provider=provider,
                npc=npc.name if npc else None,
            )

            if stream:
                conversation_result = ""
                output = get_stream(spool_context, **kwargs_to_pass)
                conversation_result = print_and_process_stream(output, model, provider)
                conversation_result = spool_context + [
                    {"role": "assistant", "content": conversation_result}
                ]
            else:
                conversation_result = get_conversation(spool_context, **kwargs_to_pass)

            # Handle potential errors in conversation_result
            if isinstance(conversation_result, str) and "Error" in conversation_result:
                print(conversation_result)  # Print the error message
                continue  # Skip to the next loop iteration
            elif (
                not isinstance(conversation_result, list)
                or len(conversation_result) == 0
            ):
                print("Error: Invalid response from get_conversation")
                continue

            spool_context = conversation_result  # update spool_context

            # Extract assistant's reply, handling potential KeyError
            try:
                # print(spool_context[-1])
                # print(provider)
                if provider == "gemini":
                    assistant_reply = spool_context[-1]["parts"][0]
                else:
                    assistant_reply = spool_context[-1]["content"]

            except (KeyError, IndexError) as e:
                print(f"Error extracting assistant's reply: {e}")
                print(spool_context[-1])
                print(
                    f"Conversation result: {conversation_result}"
                )  # Print for debugging
                continue

            # Save assistant's response to conversation history
            save_conversation_message(
                command_history,
                conversation_id,
                "assistant",
                assistant_reply,
                wd=os.getcwd(),
                model=model,
                provider=provider,
                npc=npc.name if npc else None,
            )

            # sometimes claude responds with unfinished markdown notation. so we need to check if there are two sets
            # of markdown notation and if not, we add it. so if # markdown notations is odd we add one more
            if assistant_reply.count("```") % 2 != 0:
                assistant_reply = assistant_reply + "```"

            if not stream:
                render_markdown(assistant_reply)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting spool mode.")
            break

    return {
        "messages": spool_context,
        "output": "\n".join(
            [msg["content"] for msg in spool_context if msg["role"] == "assistant"]
        ),
    }

