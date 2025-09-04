

"""Test the auto_tools functionality with some real examples."""

from npcpy.tools import auto_tools

def get_weather(location: str) -> str:
    """Get weather information for a location"""
    return f"The weather in {location} is sunny and 75°F"

def calculate_math(expression: str) -> str:
    """Calculate a mathematical expression"""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except:
        return "Invalid mathematical expression"

def process_data(data: list, operation: str = "sum") -> float:
    """
    Process a list of numbers with various operations.
    
    Args:
        data: List of numbers to process
        operation: Operation to perform ('sum', 'avg', 'max', 'min')
    """
    if operation == "sum":
        return sum(data)
    elif operation == "avg":
        return sum(data) / len(data) if data else 0
    elif operation == "max":
        return max(data) if data else 0
    elif operation == "min":
        return min(data) if data else 0
    else:
        return 0


print("Testing auto_tools with docstring_parser...")
tools_schema, tool_map = auto_tools([get_weather, calculate_math, process_data])

print("Generated schema:")
import json
print(json.dumps(tools_schema, indent=2))

print("\nTool map keys:", list(tool_map.keys()))


print("\nTesting actual function calls:")
print("Weather:", get_weather("Tokyo"))
print("Math:", calculate_math("5 * 7"))
print("Data processing:", process_data([1, 2, 3, 4, 5], "avg"))


"""
Test the auto_tools functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from npcpy.tools import auto_tools
import json

def get_weather(location: str) -> str:
    """Get weather information for a location"""
    return f"The weather in {location} is sunny and 75°F"

def calculate_math(expression: str) -> str:
    """Calculate a mathematical expression"""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except:
        return "Invalid mathematical expression"

def process_data(data: list, operation: str = "sum") -> float:
    """
    Process a list of numbers with various operations.
    
    Args:
        data: List of numbers to process
        operation: Operation to perform ('sum', 'avg', 'max', 'min')
        
    Returns:
        Result of the operation
    """
    if operation == "sum":
        return sum(data)
    elif operation == "avg":
        return sum(data) / len(data)
    elif operation == "max":
        return max(data)
    elif operation == "min":
        return min(data)
    else:
        return 0.0


"""
Practical tool calling examples with npcpy - real-world use cases
"""

import os
import json
import requests
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from npcpy.llm_funcs import get_llm_response
from npcpy.tools import auto_tools


def create_project_structure(project_name: str, project_type: str = "python") -> str:
    """
    Create a complete project directory structure with common files.
    
    Args:
        project_name: Name of the project to create
        project_type: Type of project ('python', 'web', 'data', 'ml')
    """
    base_path = f"./{project_name}"
    
    
    dirs = ["src", "tests", "docs", "data", "scripts"]
    
    
    if project_type == "python":
        dirs.extend(["requirements", "config"])
    elif project_type == "web":
        dirs.extend(["static", "templates", "api"])
    elif project_type == "data":
        dirs.extend(["notebooks", "models", "reports"])
    elif project_type == "ml":
        dirs.extend(["notebooks", "models", "experiments", "datasets"])
    
    
    for dir_name in dirs:
        os.makedirs(f"{base_path}/{dir_name}", exist_ok=True)
    
    
    files = {
        "README.md": f"
        ".gitignore": "*.pyc\n__pycache__/\n.env\n.DS_Store\n",
        "requirements.txt": "
    }
    
    if project_type == "python":
        files["setup.py"] = f'from setuptools import setup, find_packages\n\nsetup(\n    name="{project_name}",\n    version="0.1.0",\n    packages=find_packages(),\n)\n'
        files["src/__init__.py"] = ""
    
    for filename, content in files.items():
        filepath = f"{base_path}/{filename}"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(content)
    
    return f"Created {project_type} project '{project_name}' with {len(dirs)} directories and {len(files)} files"


def analyze_csv_file(filepath: str, analysis_type: str = "summary") -> Dict:
    """
    Analyze a CSV file and return insights.
    
    Args:
        filepath: Path to the CSV file
        analysis_type: Type of analysis ('summary', 'correlation', 'missing', 'outliers')
    """
    if not os.path.exists(filepath):
        return {"error": f"File {filepath} not found"}
    
    try:
        df = pd.read_csv(filepath)
        result = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict()
        }
        
        if analysis_type == "summary":
            result["summary"] = df.describe().to_dict()
            result["info"] = {
                "memory_usage": df.memory_usage(deep=True).sum(),
                "null_counts": df.isnull().sum().to_dict()
            }
        
        elif analysis_type == "correlation":
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 1:
                result["correlation"] = df[numeric_cols].corr().to_dict()
            else:
                result["warning"] = "Not enough numeric columns for correlation"
        
        elif analysis_type == "missing":
            result["missing_data"] = {
                "total_missing": df.isnull().sum().sum(),
                "missing_by_column": df.isnull().sum().to_dict(),
                "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict()
            }
        
        elif analysis_type == "outliers":
            numeric_cols = df.select_dtypes(include=['number']).columns
            outliers = {}
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers[col] = len(df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)])
            result["outliers"] = outliers
        
        return result
    
    except Exception as e:
        return {"error": str(e)}


def query_database(db_path: str, query: str, params: Optional[List] = None) -> Dict:
    """
    Execute a SQL query on SQLite database and return results.
    
    Args:
        db_path: Path to SQLite database file
        query: SQL query to execute
        params: Optional parameters for the query
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if query.strip().lower().startswith('select'):
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                return {
                    "columns": columns,
                    "rows": rows,
                    "count": len(rows)
                }
            else:
                conn.commit()
                return {"message": f"Query executed successfully, {cursor.rowcount} rows affected"}
    
    except Exception as e:
        return {"error": str(e)}


def fetch_web_content(url: str, content_type: str = "text") -> Dict:
    """
    Fetch content from a web URL.
    
    Args:
        url: URL to fetch content from
        content_type: Type of content to extract ('text', 'json', 'html')
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        result = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "url": response.url
        }
        
        if content_type == "json":
            result["content"] = response.json()
        elif content_type == "html":
            result["content"] = response.text
            result["length"] = len(response.text)
        else:  
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            result["content"] = text[:5000] + "..." if len(text) > 5000 else text
            result["full_length"] = len(text)
        
        return result
    
    except Exception as e:
        return {"error": str(e)}


def get_system_info() -> Dict:
    """Get system information including disk usage, memory, and processes."""
    import psutil
    
    try:
        
        cpu_info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
        }
        
        
        memory = psutil.virtual_memory()
        memory_info = {
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent,
            "used": memory.used
        }
        
        
        disk = psutil.disk_usage('/')
        disk_info = {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": (disk.used / disk.total) * 100
        }
        
        
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'cpu_percent']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        top_processes = sorted(processes, key=lambda x: x['memory_percent'] or 0, reverse=True)[:10]
        
        return {
            "cpu": cpu_info,
            "memory": memory_info,
            "disk": disk_info,
            "top_processes": top_processes,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        return {"error": str(e)}


def git_status_summary(repo_path: str = ".") -> Dict:
    """
    Get a summary of git repository status.
    
    Args:
        repo_path: Path to git repository (default: current directory)
    """
    import subprocess
    
    try:
        
        original_cwd = os.getcwd()
        os.chdir(repo_path)
        
        result = {}
        
        
        branch_result = subprocess.run(['git', 'branch', '--show-current'], 
                                     capture_output=True, text=True)
        result['current_branch'] = branch_result.stdout.strip()
        
        
        status_result = subprocess.run(['git', 'status', '--porcelain'], 
                                     capture_output=True, text=True)
        status_lines = status_result.stdout.strip().split('\n') if status_result.stdout.strip() else []
        
        result['changes'] = {
            'modified': [line[3:] for line in status_lines if line.startswith(' M')],
            'added': [line[3:] for line in status_lines if line.startswith('A ')],
            'deleted': [line[3:] for line in status_lines if line.startswith(' D')],
            'untracked': [line[3:] for line in status_lines if line.startswith('??')],
            'total_changes': len(status_lines)
        }
        
        
        log_result = subprocess.run(['git', 'log', '--oneline', '-5'], 
                                  capture_output=True, text=True)
        result['recent_commits'] = log_result.stdout.strip().split('\n') if log_result.stdout.strip() else []
        
        
        remote_result = subprocess.run(['git', 'remote', '-v'], 
                                     capture_output=True, text=True)
        result['remotes'] = remote_result.stdout.strip().split('\n') if remote_result.stdout.strip() else []
        
        return result
    
    except Exception as e:
        return {"error": str(e)}
    finally:
        os.chdir(original_cwd)


def analyze_text_document(filepath: str, analysis_type: str = "summary") -> Dict:
    """
    Analyze a text document for various metrics.
    
    Args:
        filepath: Path to text file
        analysis_type: Type of analysis ('summary', 'readability', 'keywords', 'sentiment')
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        import re
        from collections import Counter
        
        
        words = re.findall(r'\b\w+\b', text.lower())
        sentences = re.split(r'[.!?]+', text)
        paragraphs = text.split('\n\n')
        
        result = {
            "file_size": len(text),
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "paragraph_count": len([p for p in paragraphs if p.strip()]),
            "character_count": len(text),
            "avg_words_per_sentence": len(words) / max(len(sentences), 1)
        }
        
        if analysis_type == "summary":
            word_freq = Counter(words)
            result["most_common_words"] = word_freq.most_common(10)
            result["unique_words"] = len(set(words))
            result["lexical_diversity"] = len(set(words)) / len(words) if words else 0
        
        elif analysis_type == "readability":
            
            avg_sentence_length = len(words) / max(len(sentences), 1)
            long_words = [w for w in words if len(w) > 6]
            
            result["avg_sentence_length"] = avg_sentence_length
            result["long_words_count"] = len(long_words)
            result["long_words_percentage"] = len(long_words) / len(words) * 100 if words else 0
            
            
            result["estimated_reading_level"] = "easy" if avg_sentence_length < 15 else "moderate" if avg_sentence_length < 25 else "difficult"
        
        elif analysis_type == "keywords":
            
            stop_words = set(['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
            keywords = [word for word in words if len(word) > 4 and word not in stop_words]
            keyword_freq = Counter(keywords)
            result["keywords"] = keyword_freq.most_common(20)
            result["keyword_density"] = len(keywords) / len(words) * 100 if words else 0
        
        return result
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    
    print("Testing practical tools with auto_tools...")
    
    
    practical_tools = [
        create_project_structure,
        analyze_csv_file,
        query_database,
        fetch_web_content,
        get_system_info,
        git_status_summary,
        analyze_text_document
    ]
    
    tools_schema, tool_map = auto_tools(practical_tools)
    
    print(f"Generated {len(tools_schema)} practical tools:")
    for i, tool in enumerate(tools_schema):
        print(f"{i+1}. {tool['function']['name']}: {tool['function']['description']}")
    
    
    print("\n" + "="*50)
    print("Example: Using tools with LLM")
    
    
    test_csv_path = "./test_data/books.csv"
    
    response = get_llm_response(
        f"""I need to analyze the CSV file at {test_csv_path}. Can you:
        1. Give me a summary analysis of the data
        2. Tell me what the current git status is for this project
        3. Check the system information
        
        Please use the appropriate tools to gather this information.""",
        model='gpt-4o-mini',
        provider='openai',
        tools=tools_schema,
        tool_map=tool_map
    )
    
    print("\nLLM Response:")
    print(response.get('response', 'No response'))
    
    print("\nTool Results:")
    for result in response.get('tool_results', []):
        print(f"- {result['tool_name']}: {str(result['result'])[:200]}...")


"""
Advanced practical tool examples - specialized workflows
"""

import os
import json
import base64
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
import subprocess
from npcpy.llm_funcs import get_llm_response
from npcpy.tools import auto_tools


def run_code_analysis(directory: str, language: str = "python") -> Dict:
    """
    Run static code analysis on a directory of code files.
    
    Args:
        directory: Directory containing code files
        language: Programming language ('python', 'javascript', 'typescript')
    """
    try:
        result = {"directory": directory, "language": language, "files_analyzed": []}
        
        
        extensions = {
            "python": [".py"],
            "javascript": [".js"],
            "typescript": [".ts", ".tsx"]
        }
        
        target_extensions = extensions.get(language, [".py"])
        code_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in target_extensions):
                    code_files.append(os.path.join(root, file))
        
        result["total_files"] = len(code_files)
        
        
        total_lines = 0
        total_functions = 0
        total_classes = 0
        issues = []
        
        for filepath in code_files[:10]:  
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = len(content.splitlines())
                total_lines += lines
                
                file_info = {
                    "file": os.path.relpath(filepath, directory),
                    "lines": lines,
                    "size": len(content)
                }
                
                if language == "python":
                    import ast
                    try:
                        tree = ast.parse(content)
                        functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
                        classes = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
                        file_info.update({"functions": functions, "classes": classes})
                        total_functions += functions
                        total_classes += classes
                        
                        
                        if lines > 500:
                            issues.append(f"{file_info['file']}: Very long file ({lines} lines)")
                        if functions > 20:
                            issues.append(f"{file_info['file']}: Too many functions ({functions})")
                            
                    except SyntaxError as e:
                        issues.append(f"{file_info['file']}: Syntax error - {str(e)}")
                
                result["files_analyzed"].append(file_info)
                
            except Exception as e:
                issues.append(f"{filepath}: Error reading file - {str(e)}")
        
        result.update({
            "total_lines": total_lines,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "issues": issues,
            "avg_lines_per_file": total_lines / max(len(code_files), 1)
        })
        
        return result
    
    except Exception as e:
        return {"error": str(e)}


def manage_docker_containers(action: str, container_name: Optional[str] = None) -> Dict:
    """
    Manage Docker containers - list, start, stop, or get logs.
    
    Args:
        action: Action to perform ('list', 'start', 'stop', 'logs', 'stats')
        container_name: Name of container (required for start/stop/logs)
    """
    try:
        if action == "list":
            result = subprocess.run(['docker', 'ps', '-a', '--format', 'json'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                containers = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        containers.append(json.loads(line))
                return {"containers": containers, "count": len(containers)}
            else:
                return {"error": result.stderr}
        
        elif action == "stats":
            result = subprocess.run(['docker', 'stats', '--no-stream', '--format', 'json'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                stats = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        stats.append(json.loads(line))
                return {"stats": stats}
            else:
                return {"error": result.stderr}
        
        elif container_name:
            if action == "start":
                result = subprocess.run(['docker', 'start', container_name], 
                                      capture_output=True, text=True)
            elif action == "stop":
                result = subprocess.run(['docker', 'stop', container_name], 
                                      capture_output=True, text=True)
            elif action == "logs":
                result = subprocess.run(['docker', 'logs', '--tail', '50', container_name], 
                                      capture_output=True, text=True)
            else:
                return {"error": f"Unknown action: {action}"}
            
            if result.returncode == 0:
                return {"action": action, "container": container_name, "output": result.stdout}
            else:
                return {"error": result.stderr}
        
        else:
            return {"error": f"Container name required for action: {action}"}
    
    except Exception as e:
        return {"error": str(e)}


def analyze_log_file(log_path: str, log_type: str = "generic", lines: int = 1000) -> Dict:
    """
    Analyze log files for errors, patterns, and statistics.
    
    Args:
        log_path: Path to log file
        log_type: Type of log ('apache', 'nginx', 'syslog', 'application', 'generic')
        lines: Number of recent lines to analyze
    """
    try:
        if not os.path.exists(log_path):
            return {"error": f"Log file {log_path} not found"}
        
        
        with open(log_path, 'rb') as f:
            f.seek(0, 2)  
            file_size = f.tell()
            
            
            lines_found = []
            chunk_size = 8192
            position = file_size
            
            while len(lines_found) < lines and position > 0:
                chunk_size = min(chunk_size, position)
                position -= chunk_size
                f.seek(position)
                chunk = f.read(chunk_size).decode('utf-8', errors='ignore')
                lines_found = chunk.split('\n') + lines_found
            
            log_lines = lines_found[-lines:] if len(lines_found) > lines else lines_found
        
        
        result = {
            "file_path": log_path,
            "total_lines_analyzed": len(log_lines),
            "file_size": file_size
        }
        
        
        error_patterns = []
        warning_patterns = []
        
        if log_type in ["apache", "nginx"]:
            error_patterns = [r"\s[45]\d\d\s", r"error", r"fail"]  
            warning_patterns = [r"warn", r"timeout"]
        elif log_type == "syslog":
            error_patterns = [r"error", r"fail", r"critical", r"alert", r"emerg"]
            warning_patterns = [r"warn", r"notice"]
        else:  
            error_patterns = [r"error", r"exception", r"fail", r"critical", r"fatal"]
            warning_patterns = [r"warn", r"warning", r"deprecated"]
        
        
        import re
        errors = []
        warnings = []
        
        for line in log_lines:
            if line.strip():
                for pattern in error_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        errors.append(line.strip())
                        break
                else:
                    for pattern in warning_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            warnings.append(line.strip())
                            break
        
        
        timestamps = []
        timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}',  
            r'\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}',   
            r'\w{3}\s+\d{1,2}\s\d{2}:\d{2}:\d{2}'     
        ]
        
        for line in log_lines[-100:]:  
            for pattern in timestamp_patterns:
                match = re.search(pattern, line)
                if match:
                    timestamps.append(match.group())
                    break
        
        result.update({
            "errors": {
                "count": len(errors),
                "recent_errors": errors[-10:] if errors else []
            },
            "warnings": {
                "count": len(warnings),
                "recent_warnings": warnings[-10:] if warnings else []
            },
            "timestamps_found": len(timestamps),
            "analysis_summary": f"Found {len(errors)} errors and {len(warnings)} warnings in {len(log_lines)} lines"
        })
        
        return result
    
    except Exception as e:
        return {"error": str(e)}


def network_diagnostics(target: str, test_type: str = "ping") -> Dict:
    """
    Run network diagnostic tests.
    
    Args:
        target: Target hostname or IP address
        test_type: Type of test ('ping', 'traceroute', 'nslookup', 'port_scan')
    """
    try:
        if test_type == "ping":
            result = subprocess.run(['ping', '-c', '4', target], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                
                lines = result.stdout.split('\n')
                stats_line = [line for line in lines if 'packets transmitted' in line]
                rtt_line = [line for line in lines if 'min/avg/max' in line]
                
                return {
                    "target": target,
                    "test_type": test_type,
                    "success": True,
                    "output": result.stdout,
                    "statistics": stats_line[0] if stats_line else None,
                    "rtt_stats": rtt_line[0] if rtt_line else None
                }
            else:
                return {"target": target, "success": False, "error": result.stderr}
        
        elif test_type == "nslookup":
            result = subprocess.run(['nslookup', target], 
                                  capture_output=True, text=True)
            return {
                "target": target,
                "test_type": test_type,
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None
            }
        
        elif test_type == "traceroute":
            
            cmd = ['traceroute'] if os.name != 'nt' else ['tracert']
            result = subprocess.run(cmd + [target], 
                                  capture_output=True, text=True, timeout=30)
            return {
                "target": target, 
                "test_type": test_type,
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None
            }
        
        else:
            return {"error": f"Unsupported test type: {test_type}"}
    
    except subprocess.TimeoutExpired:
        return {"error": f"Timeout running {test_type} for {target}"}
    except Exception as e:
        return {"error": str(e)}


def manage_processes(action: str, process_name: Optional[str] = None, pid: Optional[int] = None) -> Dict:
    """
    Manage system processes - list, kill, or get detailed info.
    
    Args:
        action: Action to perform ('list', 'kill', 'info', 'top_cpu', 'top_memory')
        process_name: Name of process (for kill/info actions)
        pid: Process ID (alternative to process_name)
    """
    try:
        import psutil
        
        if action == "list":
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'status', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return {"processes": processes[:50], "total_count": len(processes)}
        
        elif action == "top_cpu":
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                try:
                    proc.cpu_percent()  
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            
            top_processes = sorted(processes, key=lambda x: x['cpu_percent'] or 0, reverse=True)[:10]
            return {"top_cpu_processes": top_processes}
        
        elif action == "top_memory":
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'memory_info']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            top_processes = sorted(processes, key=lambda x: x['memory_percent'] or 0, reverse=True)[:10]
            return {"top_memory_processes": top_processes}
        
        elif action in ["kill", "info"]:
            target_procs = []
            
            if pid:
                try:
                    target_procs = [psutil.Process(pid)]
                except psutil.NoSuchProcess:
                    return {"error": f"No process found with PID {pid}"}
            elif process_name:
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        if process_name.lower() in proc.info['name'].lower():
                            target_procs.append(proc)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            else:
                return {"error": "Either process_name or pid must be provided"}
            
            if not target_procs:
                return {"error": f"No processes found matching '{process_name}'"}
            
            results = []
            for proc in target_procs[:5]:  
                try:
                    proc_info = proc.as_dict(['pid', 'name', 'status', 'cpu_percent', 'memory_percent', 'create_time'])
                    
                    if action == "kill":
                        proc.terminate()
                        proc_info['action'] = 'terminated'
                    
                    results.append(proc_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    results.append({"error": str(e), "pid": proc.pid})
            
            return {"action": action, "results": results}
        
        else:
            return {"error": f"Unknown action: {action}"}
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    
    print("Testing advanced practical tools...")
    
    advanced_tools = [
        run_code_analysis,
        manage_docker_containers,
        analyze_log_file,
        network_diagnostics,
        manage_processes
    ]
    
    tools_schema, tool_map = auto_tools(advanced_tools)
    
    print(f"\nGenerated {len(tools_schema)} advanced tools:")
    for i, tool in enumerate(tools_schema):
        print(f"{i+1}. {tool['function']['name']}: {tool['function']['description'][:60]}...")
    
    
    print("\n" + "="*60)
    print("Example: DevOps assistant with advanced tools")
    
    response = get_llm_response(
        """I'm debugging a performance issue on my system. Can you help me:
        
        1. Check what processes are using the most CPU and memory
        2. Analyze any Python code in the current directory for potential issues
        3. Check if there are any Docker containers running
        4. Test network connectivity to google.com
        
        Use the appropriate tools to gather this information and provide insights.""",
        model='gpt-4o-mini',
        provider='openai',
        tools=tools_schema,
        tool_map=tool_map
    )
    
    print("\nDevOps Assistant Response:")
    print(response.get('response', 'No response'))
    
    if response.get('tool_results'):
        print(f"\nExecuted {len(response['tool_results'])} tools:")
        for result in response['tool_results']:
            print(f"- {result['tool_name']}: {str(result['result'])[:100]}...")


if __name__ == "__main__":
    functions = [get_weather, calculate_math, process_data]
    
    tools_schema, tool_map = auto_tools(functions)
    
    print("Generated Tools Schema:")
    print(json.dumps(tools_schema, indent=2))
    
    print("\nTool Map:")
    print(tool_map)
    
    
    print("\nTesting tool calls:")
    print("Weather:", tool_map["get_weather"]("Paris"))
    print("Math:", tool_map["calculate_math"]("15 * 23"))
    print("Data processing:", tool_map["process_data"]([1, 2, 3, 4, 5], "avg"))

