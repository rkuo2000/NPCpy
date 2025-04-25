from npcpy.data.web import search_web
from npcpy.data.text import rag_search
def execute_search_command(
    command: str,
    messages=None,
    provider: str = None,
):
    """
    Function Description:
        Executes a search command.
    Args:
        command : str : Command
        db_path : str : Database path

    Keyword Args:
        embedding_model : None : Embedding model
        current_npc : None : Current NPC
        text_data : None : Text data
        text_data_embedded : None : Embedded text data
        messages : None : Messages
    Returns:
        dict : dict : Dictionary

    """
    # search commands will bel ike :
    # '/search -p default = google "search term" '
    # '/search -p perplexity ..
    # '/search -p google ..
    # extract provider if its there
    # check for either -p or --p

    search_command = command.split()
    if any("-p" in s for s in search_command) or any(
        "--provider" in s for s in search_command
    ):
        provider = (
            search_command[search_command.index("-p") + 1]
            if "-p" in search_command
            else search_command[search_command.index("--provider") + 1]
        )
    else:
        provider = None
    if any("-n" in s for s in search_command) or any(
        "--num_results" in s for s in search_command
    ):
        num_results = (
            search_command[search_command.index("-n") + 1]
            if "-n" in search_command
            else search_command[search_command.index("--num_results") + 1]
        )
    else:
        num_results = 5

    # remove the -p and provider from the command string
    command = command.replace(f"-p {provider}", "").replace(
        f"--provider {provider}", ""
    )
    result = search_web(command, num_results=num_results, provider=provider)
    if messages is None:
        messages = []
        messages.append({"role": "user", "content": command})

    messages.append(
        {"role": "assistant", "content": result[0] + f" \n Citation Links: {result[1]}"}
    )

    return {
        "messages": messages,
        "output": result[0] + f"\n\n\n Citation Links: {result[1]}",
    }

def execute_rag_command(
    command: str,
    messages=None,
) -> dict:
    """
    Execute the RAG command with support for embedding generation using
    nomic-embed-text.
    """

    if messages is None:
        messages = []

    parts = command.split()
    search_terms = []
    params = {}
    file_list = []

    # Parse command parts
    for i, part in enumerate(parts):
        if "=" in part:  # This is a parameter
            key, value = part.split("=", 1)
            params[key.strip()] = value.strip()
        elif part.startswith("-f"):  # Handle the file list
            if i + 1 < len(parts):
                wildcard_pattern = parts[i + 1]
                file_list.extend(glob.glob(wildcard_pattern))
        else:  # This is part of the search term
            search_terms.append(part)

    # print(params)
    # -top_k  will also be a flaggable param
    if "-top_k" in params:
        top_k = int(params["-top_k"])
    else:
        top_k = 5

    # If no files found, inform the user
    if not file_list:
        return {
            "messages": messages,
            "output": "No files found matching the specified pattern.",
        }

    search_term = " ".join(search_terms)
    docs_to_embed = []

    # try:
    # Load each file and generate embeddings
    for filename in file_list:
        extension = os.path.splitext(filename)[1].lower()
        if os.path.exists(filename):
            if extension in [
                ".txt",
                ".csv",
                ".yaml",
                ".json",
                ".md",
                ".r",
                ".c",
                ".java",
                ".cpp",
                ".h",
                ".hpp",
                ".xlsx",
                ".py",
                ".js",
                ".ts",
                ".html",
                ".css",
                ".ipynb",
                ".pdf",
                ".docx",
                ".pptx",
                ".ppt",
                ".npc",
                ".tool",
                ".doc",
                ".xls",
            ]:
                if extension == ".csv":
                    df = pd.read_csv(filename)
                    file_texts = df.apply(
                        lambda row: " ".join(row.values.astype(str)), axis=1
                    ).tolist()
                else:
                    with open(filename, "r", encoding="utf-8") as file:
                        file_texts = file.readlines()
                    file_texts = [
                        line.strip() for line in file_texts if line.strip() != ""
                    ]
                    docs_to_embed.extend(file_texts)
            else:
                return {
                    "messages": messages,
                    "output": f"Unsupported file type: {extension} for file {filename}",
                }

    similar_texts = search_similar_texts(
        search_term,
        docs_to_embed=docs_to_embed,
        top_k=top_k,  # Adjust as necessary
    )

    # Format results
    output = "Found similar texts:\n\n"
    if similar_texts:
        for result in similar_texts:
            output += f"Score: {result['score']:.3f}\n"
            output += f"Text: {result['text']}\n"
            if "id" in result:
                output += f"ID: {result['id']}\n"
            output += "\n"
    else:
        output = "No similar texts found in the database."

    # Additional information about processed files
    output += "\nProcessed Files:\n"
    output += "\n".join(file_list)

    return {"messages": messages, "output": output}

    # except Exception as e:
    #    return {
    #        "messages": messages,
    #        "output": f"Error during RAG search: {str(e)}",
    #    }