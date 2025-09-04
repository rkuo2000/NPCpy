import pandas as pd
import re
import os
from pathlib import Path
from typing import Dict, List, Set, Union, Any
from collections import defaultdict, deque
from sqlalchemy import create_engine, text, Engine
import inspect
from npcpy.llm_funcs import *
from npcpy.memory.command_history import create_engine_from_path

try:
    import modin.pandas as pd
    import snowflake.snowpark.modin.plugin
    PANDAS_BACKEND = 'snowflake'
except ImportError:
    try:
        import modin.pandas as pd
        PANDAS_BACKEND = 'modin'
    except ImportError:
        import pandas as pd
        PANDAS_BACKEND = 'pandas'

class NPCSQLOperations:
    def __init__(self, npc_directory: str, db_engine: Union[str, Engine] = "~/npcsh_history.db"):
        self.npc_directory = npc_directory
        
        if isinstance(db_engine, str):
            self.engine = create_engine_from_path(db_engine)
        else:
            self.engine = db_engine
            
        self.npc_loader = None
        self.function_map = self._build_function_map()
        
    def _get_team(self):
        return self.npc_loader if hasattr(self.npc_loader, 'npcs') else None

    def _build_function_map(self):
        import npcpy.llm_funcs as llm_funcs
        import types
        
        function_map = {}
        for name in dir(llm_funcs):
            if name.startswith('_'):
                continue
            obj = getattr(llm_funcs, name)
            if isinstance(obj, types.FunctionType) and obj.__module__ == 'npcpy.llm_funcs':
                function_map[name] = obj
        
        return function_map

    def _resolve_npc_reference(self, npc_ref: str):
        if not npc_ref or not self.npc_loader:
            return None
            
        if npc_ref.endswith('.npc'):
            npc_ref = npc_ref[:-4]
        
        npc = self.npc_loader.get_npc(npc_ref)
        if npc:
            return npc
            
        if ',' in npc_ref:
            npc_names = [name.strip() for name in npc_ref.split(',')]
            npcs = [self.npc_loader.get_npc(name) for name in npc_names]
            npcs = [npc for npc in npcs if npc is not None]
            
            if npcs:
                from npcpy.npc_compiler import Team
                temp_team = Team(npcs=npcs)
                return temp_team
                
        return None
        
    def execute_ai_function(self, func_name: str, df: pd.DataFrame, **params):
        if func_name not in self.function_map:
            raise ValueError(f"Unknown AI function: {func_name}")
            
        func = self.function_map[func_name]
        
        npc_ref = params.get('npc', '')
        resolved_npc = self._resolve_npc_reference(npc_ref)
        
        resolved_team = self._get_team()
        if not resolved_team and hasattr(resolved_npc, 'team'):
            resolved_team = resolved_npc.team
        
        def apply_function(row):
            try:
                query_template = params.get('query', '')
                if query_template:
                    row_data = {col: str(row[col]) for col in df.columns}
                    query = query_template.format(**row_data)
                else:
                    query = ''
                
                sig = inspect.signature(func)
                func_params = {k: v for k, v in {
                    'prompt': query,
                    'npc': resolved_npc,
                    'team': resolved_team,
                    'context': params.get('context', '')
                }.items() if k in sig.parameters}
                
                result = func(**func_params)
                return result.get("response", "") if isinstance(result, dict) else str(result)
                
            except Exception as e:
                print(f"Error applying function {func_name}: {e}")
                return f"Error: {str(e)}"
        
        return df.apply(apply_function, axis=1)
    

class SQLModel:
    def __init__(self, name: str, content: str, path: str, npc_directory: str):
        self.name = name
        self.content = content
        self.path = path
        self.npc_directory = npc_directory

        self.dependencies = self._extract_dependencies()
        self.has_ai_function = self._check_ai_functions()
        self.ai_functions = self._extract_ai_functions()

    def _extract_dependencies(self) -> Set[str]:
        pattern = r"\{\{\s*ref\(['\"]([^'\"]+)['\"]\)\s*\}\}"
        return set(re.findall(pattern, self.content))
    
    def _check_ai_functions(self) -> bool:
        return "nql." in self.content

    def _extract_ai_functions(self) -> Dict[str, Dict]:
        import npcpy.llm_funcs as llm_funcs
        import types
        
        ai_functions = {}
        pattern = r"nql\.(\w+)\s*\(((?:[^()]*|\([^()]*\))*)\)"
        matches = re.finditer(pattern, self.content)

        # Get available function names dynamically
        available_functions = []
        for name in dir(llm_funcs):
            if name.startswith('_'):
                continue
            obj = getattr(llm_funcs, name)
            if isinstance(obj, types.FunctionType) and obj.__module__ == 'npcpy.llm_funcs':
                available_functions.append(name)

        for match in matches:
            func_name = match.group(1)
            if func_name in available_functions:
                params = [
                    param.strip().strip("\"'") for param in match.group(2).split(",")
                ]
                npc = params[1] if len(params) > 1 else ""
                if not npc.endswith(".npc"):
                    npc = npc.replace(".npc", "")
                if self.npc_directory in npc:
                    npc = npc.replace(self.npc_directory, "")

                ai_functions[func_name] = {
                    "column": params[0] if params else "",
                    "npc": npc,
                    "query": params[2] if len(params) > 2 else "",
                    "context": params[3] if len(params) > 3 else None,
                }
        return ai_functions
    


class ModelCompiler:
    def __init__(self, models_dir: str, db_engine: Union[str, Engine] = "~/npcsh_history.db", 
                 npc_directory: str = "./npc_team/", external_engines: Dict[str, Engine] = None):
        self.models_dir = Path(os.path.expanduser(models_dir))
        
        if isinstance(db_engine, str):
            self.engine = create_engine_from_path(db_engine)
        else:
            self.engine = db_engine
            
        self.external_engines = external_engines or {}
        self.models: Dict[str, SQLModel] = {}
        self.npc_operations = NPCSQLOperations(npc_directory, self.engine)
        self.npc_directory = npc_directory
        
        from npcpy.npc_compiler import Team
        try:
            self.npc_team = Team(team_path=npc_directory)
            self.npc_operations.npc_loader = self.npc_team
        except:
            self.npc_team = None
            
    def _get_engine(self, source_name: str) -> Engine:
        """Get database engine by source name"""
        if source_name == 'local' or source_name not in self.external_engines:
            return self.engine
        return self.external_engines[source_name]
        
    def _has_native_ai_functions(self, source_name: str) -> bool:
        """Check if database has native AI function support"""
        ai_enabled = {'snowflake', 'databricks', 'bigquery'}
        return source_name in ai_enabled

    def discover_models(self):
        """Discover SQL models in directory structure"""
        self.models = {}
        print(self.models_dir)
        print(list(self.models_dir.glob("**/*.sql")))
        
        for sql_file in self.models_dir.glob("**/*.sql"):
            model_name = sql_file.stem
            with open(sql_file, "r") as f:
                content = f.read()
                
            model_npc_dir = sql_file.parent
            
            self.models[model_name] = SQLModel(
                model_name, content, str(sql_file), str(model_npc_dir)
            )
            print(f"Discovered model: {model_name}")
            print(sql_file, )
            
        return self.models

    def build_dag(self) -> Dict[str, Set[str]]:
        """Build dependency graph"""
        dag = {}
        for model_name, model in self.models.items():
            dag[model_name] = model.dependencies
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

        return result

    def _replace_model_references(self, sql: str) -> str:
        ref_pattern = r"\{\{\s*ref\s*\(\s*['\"]([^'\"]+)['\"]\s*\)\s*\}\}"

        def replace_ref(match):
            model_name = match.group(1)
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found during ref replacement.")
            return model_name

        replaced_sql = re.sub(ref_pattern, replace_ref, sql)
        return replaced_sql



    def _extract_base_query(self, sql: str) -> str:
        for dep in self.models[self.current_model].dependencies:
            sql = sql.replace(f"{{{{ ref('{dep}') }}}}", dep)

        nql_pattern = r"nql\.\w+\s*\([^)]*(?:\([^)]*\)[^)]*)*\)\s+as\s+(\w+)"
        
        def replace_nql_func(match):
            alias_name = match.group(1)
            return f"NULL as {alias_name}"
        
        cleaned_sql = re.sub(nql_pattern, replace_nql_func, sql, flags=re.DOTALL)
        return cleaned_sql

    def _execute_standard_sql(self, sql: str) -> pd.DataFrame:
        try:
            sql = re.sub(r"--.*?\n", "\n", sql)
            sql = re.sub(r"\s+", " ", sql).strip()
            return pd.read_sql(sql, self.engine)
        except Exception as e:
            print(f"Failed to execute SQL: {sql}")
            print(f"Error: {str(e)}")
            raise

    def _execute_ai_model(self, sql: str, model: SQLModel) -> pd.DataFrame:
        """Execute SQL with AI functions"""
        source_pattern = r'FROM\s+(\w+)\.(\w+)'
        matches = re.findall(source_pattern, sql)
        
        if matches:
            source_name, table_name = matches[0]
            engine = self._get_engine(source_name)
            
            if self._has_native_ai_functions(source_name):
                return pd.read_sql(sql.replace(f"{source_name}.", ""), engine)
            else:
                base_sql = self._extract_base_query(sql)
                df = pd.read_sql(base_sql.replace(f"{source_name}.", ""), engine)
        else:
            base_sql = self._extract_base_query(sql)
            df = pd.read_sql(base_sql, self.engine)
        
        for func_name, params in model.ai_functions.items():
            result_series = self.npc_operations.execute_ai_function(
                func_name, df, **params
            )
            df[f"{func_name}_result"] = result_series
            
        return df

    def execute_model(self, model_name: str) -> pd.DataFrame:
        """Execute a model and materialize it to the database"""
        self.current_model = model_name
        model = self.models[model_name]

        try:
            if model.has_ai_function:
                df = self._execute_ai_model(model.content, model)
            else:
                compiled_sql = self._replace_model_references(model.content)
                df = self._execute_standard_sql(compiled_sql)

            self._materialize_to_db(model_name, df)
            return df

        except Exception as e:
            print(f"Error executing model {model_name}: {str(e)}")
            raise

    def _materialize_to_db(self, model_name: str, df: pd.DataFrame):
        with self.engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {model_name}"))
        df.to_sql(model_name, self.engine, index=False)
        print(f"Materialized model {model_name} to database")

    def _table_exists(self, table_name: str) -> bool:
        with self.engine.connect() as conn:
            try:
                result = conn.execute(text(f"SELECT 1 FROM {table_name} LIMIT 1"))
                return True
            except:
                return False

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
                    raise ValueError(f"Dependency {dep} not found in database for model {model_name}")

            results[model_name] = self.execute_model(model_name)

        return results