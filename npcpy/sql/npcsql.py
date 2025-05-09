import pandas as pd
import yaml
from typing import List, Dict, Any, Union
from npcpy.npc_compiler import 



def execute_squish_command():
    return


def execute_splat_command():
    return




class NPCSQLOperations:
    def __init__(self, npc_directory, db_path):
        super().__init__(npc_directory, db_path)

    def _get_context(
        self, df: pd.DataFrame, context: Union[str, Dict, List[str]]
    ) -> str:
        """Resolve context from different sources"""
        if isinstance(context, str):
            # Check if it's a column reference
            if context in df.columns:
                return df[context].to_string()
            # Assume it's static text
            return context
        elif isinstance(context, list):
            # List of column names to include
            return " ".join(df[col].to_string() for col in context if col in df.columns)
        elif isinstance(context, dict):
            # YAML-style context
            return yaml.dump(context)
        return ""

    # SINGLE PROMPT OPERATIONS
    def synthesize(
        self,
        query,
        df: pd.DataFrame,
        columns: List[str],
        npc: str,
        context: Union[str, Dict, List[str]],
        framework: str,
    ) -> pd.Series:
        context_text = self._get_context(df, context)

        def apply_synthesis(row):
            # we have f strings from the query, we want to fill those back in in the request
            request = query.format(**row[columns])
            prompt = f"""Framework: {framework}
                        Context: {context_text}
                        Text to synthesize: {request}
                        Synthesize the above text."""

            result = self.execute_stage(
                {"step_name": "synthesize", "npc": npc, "task": prompt},
                {},
                self.jinja_env,
            )

            return result[0]["response"]

        # columns a list
        columns_str = "_".join(columns)
        df_out = df[columns].apply(apply_synthesis, axis=1)
        return df_out

    # MULTI-PROMPT/PARALLEL OPERATIONS
    def spread_and_sync(
        self,
        df: pd.DataFrame,
        column: str,
        npc: str,
        variations: List[str],
        sync_strategy: str,
        context: Union[str, Dict, List[str]],
    ) -> pd.Series:
        context_text = self._get_context(df, context)

        def apply_spread_sync(text):
            results = []
            for variation in variations:
                prompt = f"""Variation: {variation}
                            Context: {context_text}
                            Text to analyze: {text}
                            Analyze the above text with {variation} perspective."""

                result = self.execute_stage(
                    {"step_name": f"spread_{variation}", "npc": npc, "task": prompt},
                    {},
                    self.jinja_env,
                )

                results.append(result[0]["response"])

            # Sync results
            sync_result = self.aggregate_step_results(
                [{"response": r} for r in results], sync_strategy
            )

            return sync_result

        return df[column].apply(apply_spread_sync)
        # COMPARISON OPERATIONS

    def contrast(
        self,
        df: pd.DataFrame,
        col1: str,
        col2: str,
        npc: str,
        context: Union[str, Dict, List[str]],
        comparison_framework: str,
    ) -> pd.Series:
        context_text = self._get_context(df, context)

        def apply_contrast(row):
            prompt = f"""Framework: {comparison_framework}
                        Context: {context_text}
                        Text 1: {row[col1]}
                        Text 2: {row[col2]}
                        Compare and contrast the above texts."""

            result = self.execute_stage(
                {"step_name": "contrast", "npc": npc, "task": prompt},
                {},
                self.jinja_env,
            )

            return result[0]["response"]

        return df.apply(apply_contrast, axis=1)

    def sql_operations(self, sql: str) -> pd.DataFrame:
        # Execute the SQL query

        """
        1. delegate(COLUMN, npc, query, context, jinxs, reviewers)
        2. dilate(COLUMN, npc, query, context, scope, reviewers)
        3. erode(COLUMN, npc, query, context, scope, reviewers)
        4. strategize(COLUMN, npc, query, context, timeline, constraints)
        5. validate(COLUMN, npc, query, context, criteria)
        6. synthesize(COLUMN, npc, query, context, framework)
        7. decompose(COLUMN, npc, query, context, granularity)
        8. criticize(COLUMN, npc, query, context, framework)
        9. summarize(COLUMN, npc, query, context, style)
        10. advocate(COLUMN, npc, query, context, perspective)

        MULTI-PROMPT/PARALLEL OPERATIONS
        11. spread_and_sync(COLUMN, npc, query, variations, sync_strategy, context)
        12. bootstrap(COLUMN, npc, query, sample_params, sync_strategy, context)
        13. resample(COLUMN, npc, query, variation_strategy, sync_strategy, context)

        COMPARISON OPERATIONS
        14. mediate(COL1, COL2, npc, query, context, resolution_strategy)
        15. contrast(COL1, COL2, npc, query, context, comparison_framework)
        16. reconcile(COL1, COL2, npc, query, context, alignment_strategy)

        MULTI-COLUMN INTEGRATION
        17. integrate(COLS[], npc, query, context, integration_method)
        18. harmonize(COLS[], npc, query, context, harmony_rules)
        19. orchestrate(COLS[], npc, query, context, workflow)
        """

    # Example usage in SQL-like syntax:
    """
    def execute_sql(self, sql: str) -> pd.DataFrame:
        # This would be implemented to parse and execute SQL with our custom functions
        # Example SQL:
        '''
        SELECT
            customer_id,
            synthesize(feedback_text,
                      npc='analyst',
                      context=customer_segment,
                      framework='satisfaction') as analysis,
            spread_and_sync(price_sensitivity,
                          npc='pricing_agent',
                          variations=['conservative', 'aggressive'],
                          sync_strategy='balanced_analysis',
                          context=market_context) as price_strategy
        FROM customer_data
        '''
        pass
    """


class NPCDBTAdapter:
    def __init__(self, npc_sql: NPCSQLOperations):
        self.npc_sql = npc_sql
        self.models = {}

    def ref(self, model_name: str) -> pd.DataFrame:
        # Implementation for model referencing
        return self.models.get(model_name)

    def parse_model(self, model_sql: str) -> pd.DataFrame:
        # Parse the SQL model and execute with our custom functions
        pass


class AIFunctionParser:
    """Handles parsing and extraction of AI function calls from SQL"""

    @staticmethod
    def extract_function_params(sql: str) -> Dict[str, Dict]:
        """Extract AI function parameters from SQL"""
        ai_functions = {}

        pattern = r"(\w+)\s*\(((?:[^()]*|\([^()]*\))*)\)"
        matches = re.finditer(pattern, sql)

        for match in matches:
            func_name = match.group(1)
            if func_name in ["synthesize", "spread_and_sync"]:
                params = match.group(2).split(",")
                ai_functions[func_name] = {
                    "query": params[0].strip().strip("\"'"),
                    "npc": params[1].strip().strip("\"'"),
                    "context": params[2].strip().strip("\"'"),
                }

        return ai_functions


class SQLModel:
    def __init__(self, name: str, content: str, path: str, npc_directory: str):
        self.name = name
        self.content = content
        self.path = path
        self.npc_directory = npc_directory  # This sets the npc_directory attribute

        self.dependencies = self._extract_dependencies()
        self.has_ai_function = self._check_ai_functions()
        self.ai_functions = self._extract_ai_functions()
        print(f"Initializing SQLModel with NPC directory: {npc_directory}")

    def _extract_dependencies(self) -> Set[str]:
        """Extract model dependencies using ref() calls"""
        pattern = r"\{\{\s*ref\(['\"]([^'\"]+)['\"]\)\s*\}\}"
        return set(re.findall(pattern, self.content))

    def _check_ai_functions(self) -> bool:
        """Check if the model contains AI function calls"""
        ai_functions = [
            "synthesize",
            "spread_and_sync",
            "delegate",
            "dilate",
            "erode",
            "strategize",
            "validate",
            "decompose",
            "criticize",
            "summarize",
            "advocate",
            "bootstrap",
            "resample",
            "mediate",
            "contrast",
            "reconcile",
            "integrate",
            "harmonize",
            "orchestrate",
        ]
        return any(func in self.content for func in ai_functions)

    def _extract_ai_functions(self) -> Dict[str, Dict]:
        """Extract all AI functions and their parameters from the SQL content."""
        ai_functions = {}
        pattern = r"(\w+)\s*\(((?:[^()]*|\([^()]*\))*)\)"
        matches = re.finditer(pattern, self.content)

        for match in matches:
            func_name = match.group(1)
            if func_name in [
                "synthesize",
                "spread_and_sync",
                "delegate",
                "dilate",
                "erode",
                "strategize",
                "validate",
                "decompose",
                "criticize",
                "summarize",
                "advocate",
                "bootstrap",
                "resample",
                "mediate",
                "contrast",
                "reconcile",
                "integrate",
                "harmonize",
                "orchestrate",
            ]:
                params = [
                    param.strip().strip("\"'") for param in match.group(2).split(",")
                ]
                npc = params[1]
                if not npc.endswith(".npc"):
                    npc = npc.replace(".npc", "")
                if self.npc_directory in npc:
                    npc = npc.replace(self.npc_directory, "")

                # print(npc)
                ai_functions[func_name] = {
                    "column": params[0],
                    "npc": npc,
                    "query": params[2],
                    "context": params[3] if len(params) > 3 else None,
                }
        return ai_functions


class ModelCompiler:
    def __init__(self, models_dir: str, db_path: str, npc_directory: str):
        self.models_dir = Path(models_dir)
        self.db_path = db_path
        self.models: Dict[str, SQLModel] = {}
        self.npc_operations = NPCSQLOperations(npc_directory, db_path)
        self.npc_directory = npc_directory

    def discover_models(self):
        """Discover all SQL models in the models directory"""
        self.models = {}
        for sql_file in self.models_dir.glob("**/*.sql"):
            model_name = sql_file.stem
            with open(sql_file, "r") as f:
                content = f.read()
            self.models[model_name] = SQLModel(
                model_name, content, str(sql_file), self.npc_directory
            )
            print(f"Discovered model: {model_name}")
        return self.models

    def build_dag(self) -> Dict[str, Set[str]]:
        """Build dependency graph"""
        dag = {}
        for model_name, model in self.models.items():
            dag[model_name] = model.dependencies
        print(f"Built DAG: {dag}")
        return dag

    def topological_sort(self) -> List[str]:
        """Generate execution order using topological sort"""
        dag = self.build_dag()
        in_degree = defaultdict(int)

        for node, deps in dag.items():
            for dep in deps:
                in_degree[dep] += 1
                if dep not in dag:
                    dag[dep] = set()

        queue = deque([node for node in dag.keys() if len(dag[node]) == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            for dependent, deps in dag.items():
                if node in deps:
                    deps.remove(node)
                    if len(deps) == 0:
                        queue.append(dependent)

        if len(result) != len(dag):
            raise ValueError("Circular dependency detected")

        print(f"Execution order: {result}")
        return result

    def _replace_model_references(self, sql: str) -> str:
        ref_pattern = r"\{\{\s*ref\s*\(\s*['\"]([^'\"]+)['\"]\s*\)\s*\}\}"

        def replace_ref(match):
            model_name = match.group(1)
            if model_name not in self.models:
                raise ValueError(
                    f"Model '{model_name}' not found during ref replacement."
                )
            return model_name

        replaced_sql = re.sub(ref_pattern, replace_ref, sql)
        return replaced_sql

    def compile_model(self, model_name: str) -> str:
        """Compile a single model, resolving refs."""
        model = self.models[model_name]
        compiled_sql = model.content
        compiled_sql = self._replace_model_references(compiled_sql)
        print(f"Compiled SQL for {model_name}:\n{compiled_sql}")
        return compiled_sql

    def _extract_base_query(self, sql: str) -> str:
        for dep in self.models[self.current_model].dependencies:
            sql = sql.replace(f"{{{{ ref('{dep}') }}}}", dep)

        parts = sql.split("FROM", 1)
        if len(parts) != 2:
            raise ValueError("Invalid SQL syntax")

        select_part = parts[0].replace("SELECT", "").strip()
        from_part = "FROM" + parts[1]

        columns = re.split(r",\s*(?![^()]*\))", select_part.strip())

        final_columns = []
        for col in columns:
            if "synthesize(" not in col:
                final_columns.append(col)
            else:
                alias_match = re.search(r"as\s+(\w+)\s*$", col, re.IGNORECASE)
                if alias_match:
                    final_columns.append(f"NULL as {alias_match.group(1)}")

        final_sql = f"SELECT {', '.join(final_columns)} {from_part}"
        print(f"Extracted base query:\n{final_sql}")

        return final_sql

    def execute_model(self, model_name: str) -> pd.DataFrame:
        """Execute a model and materialize it to the database"""
        self.current_model = model_name
        model = self.models[model_name]
        compiled_sql = self.compile_model(model_name)

        try:
            if model.has_ai_function:
                df = self._execute_ai_model(compiled_sql, model)
            else:
                df = self._execute_standard_sql(compiled_sql)

            self._materialize_to_db(model_name, df)
            return df

        except Exception as e:
            print(f"Error executing model {model_name}: {str(e)}")
            raise

    def _execute_standard_sql(self, sql: str) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            try:
                sql = re.sub(r"--.*?\n", "\n", sql)
                sql = re.sub(r"\s+", " ", sql).strip()
                return pd.read_sql(sql, conn)
            except Exception as e:
                print(f"Failed to execute SQL: {sql}")
                print(f"Error: {str(e)}")
                raise

    def execute_ai_function(self, query, npc, column_value, context):
        """Execute a specific AI function logic - placeholder"""
        print(f"Executing AI function on value: {column_value}")
        synthesized_value = (
            f"Processed({query}): {column_value} in context {context} with npc {npc}"
        )
        return synthesized_value

    def _execute_ai_model(self, sql: str, model: SQLModel) -> pd.DataFrame:
        try:
            base_sql = self._extract_base_query(sql)
            print(f"Executing base SQL:\n{base_sql}")
            df = self._execute_standard_sql(base_sql)

            # extract the columns they are between {} pairs
            columns = re.findall(r"\{([^}]+)\}", sql)

            # Handle AI function a
            for func_name, params in model.ai_functions.items():
                if func_name == "synthesize":
                    query_template = params["query"]

                    npc = params["npc"]
                    # only take the after the split "/"
                    npc = npc.split("/")[-1]
                    context = params["context"]
                    # Call the synthesize method using DataFrame directly
                    synthesized_df = self.npc_operations.synthesize(
                        query=query_template,  # The raw query to format
                        df=df,  # The DataFrame containing the data
                        columns=columns,  # The column(s) used to format the query
                        npc=npc,  # NPC parameter
                        context=context,  # Context parameter
                        framework="default_framework",  # Adjust this as per your needs
                    )

                    # Optionally pull the synthesized data into a new column
                    df[
                        "ai_analysis"
                    ] = synthesized_df  # Adjust as per what synthesize returns

            return df

        except Exception as e:
            print(f"Error in AI model execution: {str(e)}")
            raise

    def _materialize_to_db(self, model_name: str, df: pd.DataFrame):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {model_name}")
            df.to_sql(model_name, conn, index=False)
            print(f"Materialized model {model_name} to database")

    def _table_exists(self, table_name: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name=?;
            """,
                (table_name,),
            )
            return cursor.fetchone() is not None

    def run_all_models(self):
        """Execute all models in dependency order"""
        self.discover_models()
        execution_order = self.topological_sort()
        print(f"Running models in order: {execution_order}")

        results = {}
        for model_name in execution_order:
            print(f"\nExecuting model: {model_name}")

            model = self.models[model_name]
            for dep in model.dependencies:
                if not self._table_exists(dep):
                    raise ValueError(
                        f"Dependency {dep} not found in database for model {model_name}"
                    )

            results[model_name] = self.execute_model(model_name)

        return results


def create_example_models(
    models_dir: str = os.path.abspath("./npc_team/factory/models/"),
    db_path: str = "~/npcsh_history.db",
    npc_directory: str = "./npc_team/",
):
    """Create example SQL model files"""
    os.makedirs(os.path.abspath("./npc_team/factory/"), exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    db_path = os.path.expanduser(db_path)
    conn = sqlite3.connect(db_path)
    df = pd.DataFrame(
        {
            "feedback": ["Great product!", "Could be better", "Amazing service"],
            "customer_id": [1, 2, 3],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        }
    )

    df.to_sql("raw_customer_feedback", conn, index=False, if_exists="replace")
    print("Created raw_customer_feedback table")

    compiler = ModelCompiler(models_dir, db_path, npc_directory)
    results = compiler.run_all_models()

    for model_name, df in results.items():
        print(f"\nResults for {model_name}:")
        print(df.head())

    customer_feedback = """
    SELECT
        feedback,
        customer_id,
        timestamp
    FROM raw_customer_feedback
    WHERE LENGTH(feedback) > 10;
    """

    customer_insights = """
    SELECT
        customer_id,
        feedback,
        timestamp,
        synthesize(
            "feedback text: {feedback}",
            "analyst",
            "feedback_analysis"
        ) as ai_analysis
    FROM {{ ref('customer_feedback') }};
    """

    models = {
        "customer_feedback.sql": customer_feedback,
        "customer_insights.sql": customer_insights,
    }

    for name, content in models.items():
        path = os.path.join(models_dir, name)
        with open(path, "w") as f:
            f.write(content)
        print(f"Created model: {name}")
