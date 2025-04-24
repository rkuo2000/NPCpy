
def enter_data_mode(npc: Any = None) -> None:
    """
    Function Description:
        This function is used to enter the data mode.
    Args:

    Keyword Args:
        npc : Any : The NPC object.
    Returns:
        None
    """
    npc_name = npc.name if npc else "data_analyst"
    print(f"Entering data mode (NPC: {npc_name}). Type '/dq' to exit.")

    exec_env = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "os": os,
        "npc": npc,
    }

    while True:
        try:
            user_input = input(f"{npc_name}> ").strip()
            if user_input.lower() == "/dq":
                break
            elif user_input == "":
                continue

            # First check if input exists in exec_env
            if user_input in exec_env:
                result = exec_env[user_input]
                if result is not None:
                    if isinstance(result, pd.DataFrame):
                        print(result.to_string())
                    else:
                        print(result)
                continue

            # Then check if it's a natural language query
            if not any(
                keyword in user_input
                for keyword in [
                    "=",
                    "+",
                    "-",
                    "*",
                    "/",
                    "(",
                    ")",
                    "[",
                    "]",
                    "{",
                    "}",
                    "import",
                ]
            ):
                if "df" in exec_env and isinstance(exec_env["df"], pd.DataFrame):
                    df_info = {
                        "shape": exec_env["df"].shape,
                        "columns": list(exec_env["df"].columns),
                        "dtypes": exec_env["df"].dtypes.to_dict(),
                        "head": exec_env["df"].head().to_dict(),
                        "summary": exec_env["df"].describe().to_dict(),
                    }

                    analysis_prompt = f"""Based on this DataFrame info: {df_info}
                    Generate Python analysis commands to answer: {user_input}
                    Return each command on a new line. Do not use markdown formatting or code blocks."""

                    analysis_response = npc.get_llm_response(analysis_prompt).get(
                        "response", ""
                    )
                    analysis_commands = [
                        cmd.strip()
                        for cmd in analysis_response.replace("```python", "")
                        .replace("```", "")
                        .split("\n")
                        if cmd.strip()
                    ]
                    results = []

                    print("\nAnalyzing data...")
                    for cmd in analysis_commands:
                        if cmd.strip():
                            try:
                                result = eval(cmd, exec_env)
                                if result is not None:
                                    render_markdown(f"\n{cmd} ")
                                    if isinstance(result, pd.DataFrame):
                                        render_markdown(result.to_string())
                                    else:
                                        render_markdown(result)
                                    results.append((cmd, result))
                            except SyntaxError:
                                try:
                                    exec(cmd, exec_env)
                                except Exception as e:
                                    print(f"Error in {cmd}: {str(e)}")
                            except Exception as e:
                                print(f"Error in {cmd}: {str(e)}")

                    if results:
                        interpretation_prompt = f"""Based on these analysis results:
                        {[(cmd, str(result)) for cmd, result in results]}

                        Provide a clear, concise interpretation of what we found in the data.
                        Focus on key insights and patterns. Do not use markdown formatting."""

                        print("\nInterpretation:")
                        interpretation = npc.get_llm_response(
                            interpretation_prompt
                        ).get("response", "")
                        interpretation = interpretation.replace("```", "").strip()
                        render_markdown(interpretation)
                    continue

            # If not in exec_env and not natural language, try as Python code
            try:
                result = eval(user_input, exec_env)
                if result is not None:
                    if isinstance(result, pd.DataFrame):
                        print(result.to_string())
                    else:
                        print(result)
            except SyntaxError:
                exec(user_input, exec_env)
            except Exception as e:
                print(f"Error: {str(e)}")

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected. Exiting data mode.")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

    return

def enter_data_analysis_mode(npc: Any = None) -> None:
    """
    Function Description:
        This function is used to enter the data analysis mode.
    Args:

    Keyword Args:
        npc : Any : The NPC object.
    Returns:
        None
    """

    npc_name = npc.name if npc else "data_analyst"
    print(f"Entering data analysis mode (NPC: {npc_name}). Type '/daq' to exit.")

    dataframes = {}  # Dict to store dataframes by name
    context = {"dataframes": dataframes}  # Context to store variables
    messages = []  # For conversation history if needed

    while True:
        user_input = input(f"{npc_name}> ").strip()

        if user_input.lower() == "/daq":
            break

        # Add user input to messages for context if interacting with LLM
        messages.append({"role": "user", "content": user_input})

        # Process commands
        if user_input.lower().startswith("load "):
            # Command format: load <file_path> as <df_name>
            try:
                parts = user_input.split()
                file_path = parts[1]
                if "as" in parts:
                    as_index = parts.index("as")
                    df_name = parts[as_index + 1]
                else:
                    df_name = "df"  # Default dataframe name
                # Load data into dataframe
                df = pd.read_csv(file_path)
                dataframes[df_name] = df
                print(f"Data loaded into dataframe '{df_name}'")
            except Exception as e:
                print(f"Error loading data: {e}")

        elif user_input.lower().startswith("sql "):
            # Command format: sql <SQL query>
            try:
                query = user_input[4:]  # Remove 'sql ' prefix
                df = pd.read_sql_query(query, npc.db_conn)
                print(df)
                # Optionally store result in a dataframe
                dataframes["sql_result"] = df
                print("Result stored in dataframe 'sql_result'")

            except Exception as e:
                print(f"Error executing SQL query: {e}")

        elif user_input.lower().startswith("plot "):
            # Command format: plot <pandas plotting code>
            try:
                code = user_input[5:]  # Remove 'plot ' prefix
                # Prepare execution environment
                exec_globals = {"pd": pd, "plt": plt, **dataframes}
                exec(code, exec_globals)
                plt.show()
            except Exception as e:
                print(f"Error generating plot: {e}")

        elif user_input.lower().startswith("exec "):
            # Command format: exec <Python code>
            try:
                code = user_input[5:]  # Remove 'exec ' prefix
                # Prepare execution environment
                exec_globals = {"pd": pd, "plt": plt, **dataframes}
                exec(code, exec_globals)
                # Update dataframes with any new or modified dataframes
                dataframes.update(
                    {
                        k: v
                        for k, v in exec_globals.items()
                        if isinstance(v, pd.DataFrame)
                    }
                )
            except Exception as e:
                print(f"Error executing code: {e}")

        elif user_input.lower().startswith("help"):
            # Provide help information
            print(
                """
Available commands:
- load <file_path> as <df_name>: Load CSV data into a dataframe.
- sql <SQL query>: Execute SQL query.
- plot <pandas plotting code>: Generate plots using matplotlib.
- exec <Python code>: Execute arbitrary Python code.
- help: Show this help message.
- /daq: Exit data analysis mode.
"""
            )

        else:
            # Unrecognized command
            print("Unrecognized command. Type 'help' for a list of available commands.")

    print("Exiting data analysis mode.")
def execute_data_operations(
    query: str,
    dataframes: Dict[str, pd.DataFrame],
    npc: Any = None,
    db_path: str = "~/npcsh_history.db",
):
    """This function executes data operations.
    Args:
        query (str): The query to execute.

        dataframes (Dict[str, pd.DataFrame]): The dictionary of dataframes.
    Keyword Args:
        npc (Any): The NPC object.
        db_path (str): The database path.
    Returns:
        Any: The result of the data operations.
    """

    location = os.getcwd()
    db_path = os.path.expanduser(db_path)

    try:
        try:
            # Create a safe namespace for pandas execution
            namespace = {
                "pd": pd,
                "np": np,
                "plt": plt,
                **dataframes,  # This includes all our loaded dataframes
            }
            # Execute the query
            result = eval(query, namespace)

            # Handle the result
            if isinstance(result, (pd.DataFrame, pd.Series)):
                # render_markdown(result)
                return result, "pd"
            elif isinstance(result, plt.Figure):
                plt.show()
                return result, "pd"
            elif result is not None:
                # render_markdown(result)

                return result, "pd"

        except Exception as exec_error:
            print(f"Pandas Error: {exec_error}")

        # 2. Try SQL
        # print(db_path)
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                print(query)
                print(get_available_tables(db_path))

                cursor.execute(query)
                # get available tables

                result = cursor.fetchall()
                if result:
                    for row in result:
                        print(row)
                    return result, "sql"
        except Exception as e:
            print(f"SQL Error: {e}")

        # 3. Try R
        try:
            result = subprocess.run(
                ["Rscript", "-e", query], capture_output=True, text=True
            )
            if result.returncode == 0:
                print(result.stdout)
                return result.stdout, "r"
            else:
                print(f"R Error: {result.stderr}")
        except Exception as e:
            pass

        # If all engines fail, ask the LLM
        print("Direct execution failed. Asking LLM for SQL query...")
        llm_prompt = f"""
        The user entered the following query which could not be executed directly using pandas, SQL, R, Scala, or PySpark:
        ```
        {query}
        ```

        The available tables in the SQLite database at {db_path} are:
        ```sql
        {get_available_tables(db_path)}
        ```

        Please provide a valid SQL query that accomplishes the user's intent.  If the query requires data from a file, provide instructions on how to load the data into a table first.
        Return only the SQL query, or instructions for loading data followed by the SQL query.
        """

        llm_response = get_llm_response(llm_prompt, npc=npc)

        print(f"LLM suggested SQL: {llm_response}")
        command = llm_response.get("response", "")
        if command == "":
            return "LLM did not provide a valid SQL query.", None
        # Execute the LLM-generated SQL
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(command)
                result = cursor.fetchall()
                if result:
                    for row in result:
                        print(row)
                    return result, "llm"
        except Exception as e:
            print(f"Error executing LLM-generated SQL: {e}")
            return f"Error executing LLM-generated SQL: {e}", None

    except Exception as e:
        print(f"Error executing query: {e}")
        return f"Error executing query: {e}", None


def check_output_sufficient(
    request: str,
    data: pd.DataFrame,
    query: str,
    model: str = None,
    provider: str = None,
    npc: Any = None,
) -> Dict[str, Any]:
    """
    Check if the query results are sufficient to answer the user's request.
    """
    prompt = f"""
    Given:
    - User request: {request}
    - Query executed: {query}
    - Results:
      Summary: {data.describe()}
      data schema: {data.dtypes}
      Sample: {data.head()}

    Is this result sufficient to answer the user's request?
    Return JSON with:
    {{
        "IS_SUFFICIENT": <boolean>,
        "EXPLANATION": <string : If the answer is not sufficient specify what else is necessary.
                                IFF the answer is sufficient, provide a response that can be returned to the user as an explanation that answers their question.
                                The explanation should use the results to answer their question as long as they wouold be useful to the user.
                                    For example, it is not useful to report on the "average/min/max/std ID" or the "min/max/std/average of a string column".

                                Be smart about what you report.
                                It should not be a conceptual or abstract summary of the data.
                                It should not unnecessarily bring up a need for more data.
                                You should write it in a tone that answers the user request. Do not spout unnecessary self-referential fluff like "This information gives a clear overview of the x landscape".
                                >
    }}
    DO NOT include markdown formatting or ```json tags.

    """

    response = get_llm_response(
        prompt, format="json", model=model, provider=provider, npc=npc
    )

    # Clean response if it's a string
    result = response.get("response", {})
    if isinstance(result, str):
        result = result.replace("```json", "").replace("```", "").strip()
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            return {"IS_SUFFICIENT": False, "EXPLANATION": "Failed to parse response"}

    return result


def process_data_output(
    llm_response: Dict[str, Any],
    db_conn,
    request: str,
    tables: str = None,
    history: str = None,
    npc: Any = None,
    model: str = None,
    provider: str = None,
) -> Dict[str, Any]:
    """
    Process the LLM's response to a data request and execute the appropriate query.
    """
    try:
        choice = llm_response.get("choice")
        query = llm_response.get("query")

        if not query:
            return {"response": "No query provided", "code": 400}

        # Create SQLAlchemy engine based on connection type
        if "psycopg2" in db_conn.__class__.__module__:
            engine = create_engine("postgresql://caug:gobears@localhost/npc_test")
        else:
            engine = create_engine("sqlite:///test_sqlite.db")

        if choice == 1:  # Direct answer query
            try:
                df = pd.read_sql_query(query, engine)
                result = check_output_sufficient(
                    request, df, query, model=model, provider=provider, npc=npc
                )

                if result.get("IS_SUFFICIENT"):
                    return {"response": result["EXPLANATION"], "data": df, "code": 200}
                return {
                    "response": f"Results insufficient: {result.get('EXPLANATION')}",
                    "code": 400,
                }

            except Exception as e:
                return {"response": f"Query execution failed: {str(e)}", "code": 400}

        elif choice == 2:  # Exploratory query
            try:
                df = pd.read_sql_query(query, engine)
                extra_context = f"""
                Exploratory query results:
                Query: {query}
                Results summary: {df.describe()}
                Sample data: {df.head()}
                """

                return get_data_response(
                    request,
                    db_conn,
                    tables=tables,
                    extra_context=extra_context,
                    history=history,
                    model=model,
                    provider=provider,
                    npc=npc,
                )

            except Exception as e:
                return {"response": f"Exploratory query failed: {str(e)}", "code": 400}

        return {"response": "Invalid choice specified", "code": 400}

    except Exception as e:
        return {"response": f"Processing error: {str(e)}", "code": 400}


def get_data_response(
    request: str,
    db_conn,
    tables: str = None,
    n_try_freq: int = 5,
    extra_context: str = None,
    history: str = None,
    model: str = None,
    provider: str = None,
    npc: Any = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Generate a response to a data request, with retries for failed attempts.
    """

    # Extract schema information based on connection type
    schema_info = ""
    if "psycopg2" in db_conn.__class__.__module__:
        cursor = db_conn.cursor()
        # Get all tables and their columns
        cursor.execute(
            """
            SELECT
                t.table_name,
                array_agg(c.column_name || ' ' || c.data_type) as columns,
                array_agg(
                    CASE
                        WHEN tc.constraint_type = 'FOREIGN KEY'
                        THEN kcu.column_name || ' REFERENCES ' || ccu.table_name || '.' || ccu.column_name
                        ELSE NULL
                    END
                ) as foreign_keys
            FROM information_schema.tables t
            JOIN information_schema.columns c ON t.table_name = c.table_name
            LEFT JOIN information_schema.table_constraints tc
                ON t.table_name = tc.table_name
                AND tc.constraint_type = 'FOREIGN KEY'
            LEFT JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
            LEFT JOIN information_schema.constraint_column_usage ccu
                ON tc.constraint_name = ccu.constraint_name
            WHERE t.table_schema = 'public'
            GROUP BY t.table_name;
        """
        )
        for table, columns, fks in cursor.fetchall():
            schema_info += f"\nTable {table}:\n"
            schema_info += "Columns:\n"
            for col in columns:
                schema_info += f"  - {col}\n"
            if any(fk for fk in fks if fk is not None):
                schema_info += "Foreign Keys:\n"
                for fk in fks:
                    if fk:
                        schema_info += f"  - {fk}\n"

    elif "sqlite3" in db_conn.__class__.__module__:
        cursor = db_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        for (table_name,) in tables:
            schema_info += f"\nTable {table_name}:\n"
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            schema_info += "Columns:\n"
            for col in columns:
                schema_info += f"  - {col[1]} {col[2]}\n"

            cursor.execute(f"PRAGMA foreign_key_list({table_name});")
            foreign_keys = cursor.fetchall()
            if foreign_keys:
                schema_info += "Foreign Keys:\n"
                for fk in foreign_keys:
                    schema_info += f"  - {fk[3]} REFERENCES {fk[2]}({fk[4]})\n"

    prompt = f"""
    User request: {request}

    Database Schema:
    {schema_info}

    {extra_context or ''}
    {f'Query history: {history}' if history else ''}

    Provide either:
    1) An SQL query to directly answer the request
    2) An exploratory query to gather more information

    Return JSON with:
    {{
        "query": <sql query string>,
        "choice": <1 or 2>,
        "explanation": <reason for choice>
    }}
    DO NOT include markdown formatting or ```json tags.
    """

    failures = []
    for attempt in range(max_retries):
        # try:
        llm_response = get_llm_response(
            prompt, npc=npc, format="json", model=model, provider=provider
        )

        # Clean response if it's a string
        response_data = llm_response.get("response", {})
        if isinstance(response_data, str):
            response_data = (
                response_data.replace("```json", "").replace("```", "").strip()
            )
            try:
                response_data = json.loads(response_data)
            except json.JSONDecodeError:
                failures.append("Invalid JSON response")
                continue

        result = process_data_output(
            response_data,
            db_conn,
            request,
            tables=tables,
            history=failures,
            npc=npc,
            model=model,
            provider=provider,
        )

        if result["code"] == 200:
            return result

        failures.append(result["response"])

        if attempt == max_retries - 1:
            return {
                "response": f"Failed after {max_retries} attempts. Errors: {'; '.join(failures)}",
                "code": 400,
            }

    # except Exception as e:
    #    failures.append(str(e))



def enter_data_analysis_mode(npc: Any = None) -> None:
    """
    Function Description:
        This function is used to enter the data analysis mode.
    Args:

    Keyword Args:
        npc : Any : The NPC object.
    Returns:
        None
    """

    npc_name = npc.name if npc else "data_analyst"
    print(f"Entering data analysis mode (NPC: {npc_name}). Type '/daq' to exit.")

    dataframes = {}  # Dict to store dataframes by name
    context = {"dataframes": dataframes}  # Context to store variables
    messages = []  # For conversation history if needed

    while True:
        user_input = input(f"{npc_name}> ").strip()

        if user_input.lower() == "/daq":
            break

        # Add user input to messages for context if interacting with LLM
        messages.append({"role": "user", "content": user_input})

        # Process commands
        if user_input.lower().startswith("load "):
            # Command format: load <file_path> as <df_name>
            try:
                parts = user_input.split()
                file_path = parts[1]
                if "as" in parts:
                    as_index = parts.index("as")
                    df_name = parts[as_index + 1]
                else:
                    df_name = "df"  # Default dataframe name
                # Load data into dataframe
                df = pd.read_csv(file_path)
                dataframes[df_name] = df
                print(f"Data loaded into dataframe '{df_name}'")
            except Exception as e:
                print(f"Error loading data: {e}")

        elif user_input.lower().startswith("sql "):
            # Command format: sql <SQL query>
            try:
                query = user_input[4:]  # Remove 'sql ' prefix
                df = pd.read_sql_query(query, npc.db_conn)
                print(df)
                # Optionally store result in a dataframe
                dataframes["sql_result"] = df
                print("Result stored in dataframe 'sql_result'")

            except Exception as e:
                print(f"Error executing SQL query: {e}")

        elif user_input.lower().startswith("plot "):
            # Command format: plot <pandas plotting code>
            try:
                code = user_input[5:]  # Remove 'plot ' prefix
                # Prepare execution environment
                exec_globals = {"pd": pd, "plt": plt, **dataframes}
                exec(code, exec_globals)
                plt.show()
            except Exception as e:
                print(f"Error generating plot: {e}")

        elif user_input.lower().startswith("exec "):
            # Command format: exec <Python code>
            try:
                code = user_input[5:]  # Remove 'exec ' prefix
                # Prepare execution environment
                exec_globals = {"pd": pd, "plt": plt, **dataframes}
                exec(code, exec_globals)
                # Update dataframes with any new or modified dataframes
                dataframes.update(
                    {
                        k: v
                        for k, v in exec_globals.items()
                        if isinstance(v, pd.DataFrame)
                    }
                )
            except Exception as e:
                print(f"Error executing code: {e}")

        elif user_input.lower().startswith("help"):
            # Provide help information
            print(
                """
Available commands:
- load <file_path> as <df_name>: Load CSV data into a dataframe.
- sql <SQL query>: Execute SQL query.
- plot <pandas plotting code>: Generate plots using matplotlib.
- exec <Python code>: Execute arbitrary Python code.
- help: Show this help message.
- /daq: Exit data analysis mode.
"""
            )

        else:
            # Unrecognized command
            print("Unrecognized command. Type 'help' for a list of available commands.")

    print("Exiting data analysis mode.")


def enter_data_mode(npc: Any = None) -> None:
    """
    Function Description:
        This function is used to enter the data mode.
    Args:

    Keyword Args:
        npc : Any : The NPC object.
    Returns:
        None
    """
    npc_name = npc.name if npc else "data_analyst"
    print(f"Entering data mode (NPC: {npc_name}). Type '/dq' to exit.")

    exec_env = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "os": os,
        "npc": npc,
    }

    while True:
        try:
            user_input = input(f"{npc_name}> ").strip()
            if user_input.lower() == "/dq":
                break
            elif user_input == "":
                continue

            # First check if input exists in exec_env
            if user_input in exec_env:
                result = exec_env[user_input]
                if result is not None:
                    if isinstance(result, pd.DataFrame):
                        print(result.to_string())
                    else:
                        print(result)
                continue

            # Then check if it's a natural language query
            if not any(
                keyword in user_input
                for keyword in [
                    "=",
                    "+",
                    "-",
                    "*",
                    "/",
                    "(",
                    ")",
                    "[",
                    "]",
                    "{",
                    "}",
                    "import",
                ]
            ):
                if "df" in exec_env and isinstance(exec_env["df"], pd.DataFrame):
                    df_info = {
                        "shape": exec_env["df"].shape,
                        "columns": list(exec_env["df"].columns),
                        "dtypes": exec_env["df"].dtypes.to_dict(),
                        "head": exec_env["df"].head().to_dict(),
                        "summary": exec_env["df"].describe().to_dict(),
                    }

                    analysis_prompt = f"""Based on this DataFrame info: {df_info}
                    Generate Python analysis commands to answer: {user_input}
                    Return each command on a new line. Do not use markdown formatting or code blocks."""

                    analysis_response = npc.get_llm_response(analysis_prompt).get(
                        "response", ""
                    )
                    analysis_commands = [
                        cmd.strip()
                        for cmd in analysis_response.replace("```python", "")
                        .replace("```", "")
                        .split("\n")
                        if cmd.strip()
                    ]
                    results = []

                    print("\nAnalyzing data...")
                    for cmd in analysis_commands:
                        if cmd.strip():
                            try:
                                result = eval(cmd, exec_env)
                                if result is not None:
                                    render_markdown(f"\n{cmd} ")
                                    if isinstance(result, pd.DataFrame):
                                        render_markdown(result.to_string())
                                    else:
                                        render_markdown(result)
                                    results.append((cmd, result))
                            except SyntaxError:
                                try:
                                    exec(cmd, exec_env)
                                except Exception as e:
                                    print(f"Error in {cmd}: {str(e)}")
                            except Exception as e:
                                print(f"Error in {cmd}: {str(e)}")

                    if results:
                        interpretation_prompt = f"""Based on these analysis results:
                        {[(cmd, str(result)) for cmd, result in results]}

                        Provide a clear, concise interpretation of what we found in the data.
                        Focus on key insights and patterns. Do not use markdown formatting."""

                        print("\nInterpretation:")
                        interpretation = npc.get_llm_response(
                            interpretation_prompt
                        ).get("response", "")
                        interpretation = interpretation.replace("```", "").strip()
                        render_markdown(interpretation)
                    continue

            # If not in exec_env and not natural language, try as Python code
            try:
                result = eval(user_input, exec_env)
                if result is not None:
                    if isinstance(result, pd.DataFrame):
                        print(result.to_string())
                    else:
                        print(result)
            except SyntaxError:
                exec(user_input, exec_env)
            except Exception as e:
                print(f"Error: {str(e)}")

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected. Exiting data mode.")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

    return
