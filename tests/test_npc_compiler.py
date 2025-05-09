import pytest
import os
import tempfile
import sqlite3
from unittest.mock import patch, MagicMock
from npcpy.npc_compiler import  NPC, Jinx



# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------

def initialize_project(directory=None, context=None):
    """Initialize a new NPC project"""
    if directory is None:
        directory = os.getcwd()
        
    # Create directory structure
    npc_team_dir = os.path.join(directory, "npc_team")
    tools_dir = os.path.join(npc_team_dir, "tools")
    pipes_dir = os.path.join(npc_team_dir, "pipes")
    
    ensure_dirs_exist(npc_team_dir, tools_dir, pipes_dir)
    
    # Create default context
    default_ctx = {
        "project_name": os.path.basename(os.path.abspath(directory)),
        "created_at": datetime.now().isoformat()
    }
    
    if context:
        default_ctx.update(context)
        
    # Save context file
    ctx_path = os.path.join(npc_team_dir, "team.ctx")
    write_yaml_file(ctx_path, default_ctx)
    
    # Create default NPC (sibiji)
    sibiji_path = os.path.join(npc_team_dir, "sibiji.npc")
    sibiji_data = {
        "name": "sibiji",
        "primary_directive": "You are sibiji, a foundational AI assistant. Your role is to provide basic support and information. Respond to queries concisely and accurately.",
        "model": os.environ.get("NPCSH_CHAT_MODEL", "llama3.2"),
        "provider": os.environ.get("NPCSH_CHAT_PROVIDER", "ollama"),
        "use_global_tools": True
    }
    
    write_yaml_file(sibiji_path, sibiji_data)
    
    # Create basic README
    readme_path = os.path.join(npc_team_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"""# {default_ctx["project_name"]} NPC Team
        
Created: {default_ctx["created_at"]}

This directory contains NPCs, tools, and pipelines for the {default_ctx["project_name"]} project.

## Structure
- `.npc` files define AI agents with specific capabilities
- `tools/` directory contains tools that NPCs can use
- `pipes/` directory contains multi-step pipelines
- `team.ctx` contains team-wide context variables
        """)
        
    return f"NPC project initialized in {npc_team_dir}"

# ---------------------------------------------------------------------------
# Example 2: Creating an NPC Team
# ---------------------------------------------------------------------------

def example_2_npc_team():
    """Example demonstrating creation and use of an NPC team"""
    print("\nExample 2: Creating and using an NPC team")
    
    # Create team directory
    team_dir = "./example_team"
    ensure_dirs_exist(team_dir)
    
    # Create team context
    ctx = {
        "project_name": "Data Analysis Project",
        "foreman": "{{ref('coordinator')}}"
    }
    ctx_path = os.path.join(team_dir, "team.ctx")
    write_yaml_file(ctx_path, ctx)
    
    # Create coordinator NPC
    coordinator = NPC(
        "coordinator",
        primary_directive="You are the coordinator for the Data Analysis team. Your role is to understand requests and direct them to the appropriate specialist on your team.",
        model="llama3.2",
        provider="ollama"
    )
    coordinator.save(team_dir)
    
    # Create specialist NPCs
    data_analyst = NPC(
        "analyst",
        primary_directive="You are a data analyst specializing in statistical analysis and data interpretation. You excel at finding patterns and insights in structured data.",
        model="llama3.2",
        provider="ollama"
    )
    data_analyst.save(team_dir)
    
    visualizer = NPC(
        "visualizer",
        primary_directive="You are a data visualization specialist. Your expertise is in creating effective visual representations of data and explaining visual insights.",
        model="llama3.2",
        provider="ollama"
    )
    visualizer.save(team_dir)
    
    # Create tools directory
    tools_dir = os.path.join(team_dir, "tools")
    ensure_dirs_exist(tools_dir)
    
    # Create analysis tool
    analysis_tool = Jinx(tool_data={
        "tool_name": "analyze_statistics",
        "description": "Perform statistical analysis on data",
        "inputs": [
            {"name": "data", "description": "Data to analyze"}
        ],
        "steps": [
            {
                "name": "calculate",
                "engine": "python",
                "code": """
import numpy as np
import json

# Parse data if it's a string
data = context.get('data')
if isinstance(data, str):
    try:
        data = json.loads(data)
    except:
        pass

# Extract numeric values
numeric_values = []
if isinstance(data, list):
    for item in data:
        if isinstance(item, (int, float)):
            numeric_values.append(item)
        elif isinstance(item, dict):
            for k, v in item.items():
                if isinstance(v, (int, float)):
                    numeric_values.append(v)

if not numeric_values:
    output = {"error": "No numeric values found in data"}
else:
    output = {
        "count": len(numeric_values),
        "min": min(numeric_values),
        "max": max(numeric_values),
        "mean": np.mean(numeric_values),
        "median": np.median(numeric_values),
        "std_dev": np.std(numeric_values)
    }
"""
            }
        ]
    })
    analysis_tool.save(tools_dir)
    
    # Create visualization tool
    viz_tool = Jinx(tool_data={
        "tool_name": "describe_visualization",
        "description": "Describe how to visualize data",
        "inputs": [
            {"name": "data", "description": "Data to visualize"},
            {"name": "chart_type", "description": "Type of chart to create"}
        ],
        "steps": [
            {
                "name": "suggest",
                "engine": "natural",
                "code": """
Based on the following data:
{{data}}

Create a detailed description of how to visualize this data using a {{chart_type}} chart.
Include:
1. What should be on the x and y axes
2. What colors would be most effective
3. What insights the visualization would highlight
4. Any potential issues with using this chart type for this data
"""
            }
        ]
    })
    viz_tool.save(tools_dir)
    
    # Load the team
    team = Team(team_path=team_dir)
    
    # Test the team with a request
    request = "I have sales data for the last quarter and want to understand the trends and patterns. The data shows increasing sales in the first month, a plateau in the second month, and a decline in the third month."
    
    print("\nSending request to team...")
    result = team.orchestrate(request)
    print("\nTeam response:")
    if 'debrief' in result and isinstance(result['debrief'], dict):
        print(f"Summary: {result['debrief'].get('summary')}")
        print(f"Recommendations: {result['debrief'].get('recommendations')}")
    else:
        print(result)
    
    return team

# ---------------------------------------------------------------------------
# Example 3: Creating and Running a Pipeline
# ---------------------------------------------------------------------------

def example_3_pipeline():
    """Example demonstrating creation and use of a pipeline"""
    print("\nExample 3: Creating and running a pipeline")
    
    # Create a team for the pipeline
    team_dir = "./example_pipeline_team"
    ensure_dirs_exist(team_dir)
    
    # Create team context
    ctx = {
        "project_name": "Data Processing Pipeline",
    }
    ctx_path = os.path.join(team_dir, "team.ctx")
    write_yaml_file(ctx_path, ctx)
    
    # Create NPCs
    data_processor = NPC(
        "processor",
        primary_directive="You are a data processor. Your job is to clean and prepare data for analysis.",
        model="llama3.2",
        provider="ollama"
    )
    data_processor.save(team_dir)
    
    analyst = NPC(
        "analyst",
        primary_directive="You are a data analyst. Your job is to analyze prepared data and extract insights.",
        model="llama3.2",
        provider="ollama"
    )
    analyst.save(team_dir)
    
    reporter = NPC(
        "reporter",
        primary_directive="You are a report writer. Your job is to take analysis results and create clear, concise reports.",
        model="llama3.2",
        provider="ollama"
    )
    reporter.save(team_dir)
    
    # Create pipeline directory
    pipes_dir = os.path.join(team_dir, "pipes")
    ensure_dirs_exist(pipes_dir)
    
    # Create a pipeline definition
    pipeline_def = {
        "name": "data_analysis_pipeline",
        "steps": [
            {
                "step_name": "data_cleaning",
                "npc": "processor",
                "task": """
You are given the following raw data:
{{input}}

Clean this data by:
1. Removing any missing or malformed entries
2. Standardizing formats
3. Flagging any anomalies

Provide the cleaned data in JSON format and explain what changes you made.
"""
            },
            {
                "step_name": "data_analysis",
                "npc": "analyst",
                "task": """
Analyze the following cleaned data:
{{data_cleaning}}

Identify key trends, patterns, and insights in this data.
Focus on the most important findings that would be valuable for business decisions.
"""
            },
            {
                "step_name": "report_generation",
                "npc": "reporter",
                "task": """
Create a concise executive summary based on the following analysis:
{{data_analysis}}

The summary should be in a professional tone, highlight the most important findings,
and include clear recommendations for action.
"""
            }
        ]
    }
    
    # Save the pipeline
    pipeline_path = os.path.join(pipes_dir, "data_analysis.pipe")
    write_yaml_file(pipeline_path, pipeline_def)
    
    # Load the team
    team = Team(team_path=team_dir)
    
    # Create and run the pipeline
    pipeline = Pipeline(pipeline_path=pipeline_path, npc_team=team)
    
    # Sample data for pipeline
    sample_data = """
Date, Product, Revenue, Units Sold
2023-01-01, Product A, 12500, 250
2023-01-02, Product B, 8700, 145
2023-01-03, Product A, 13200, 264
2023-01-03, Product C, 5300, 106
2023-01-04, Product B, 9100, 152
2023-01-04, Product C, 5100, 102
2023-01-05, Product A, 14000, 280
2023-01-05, missing data, , 
2023-01-06, Product B, ERROR, 158
2023-01-06, Product C, 5500, 110
"""
    
    print("\nRunning pipeline with sample data...")
    result = pipeline.execute(initial_context={"input": sample_data})
    
    print("\nPipeline results:")
    for step_name, output in result["results"].items():
        print(f"\n--- {step_name} ---")
        print(output[:200] + "..." if len(str(output)) > 200 else output)
    
    return pipeline

# ---------------------------------------------------------------------------
# Example 4: Sub-teams and Hierarchical Structures
# ---------------------------------------------------------------------------

def example_4_hierarchical_teams():
    """Example demonstrating hierarchical team structures"""
    print("\nExample 4: Creating hierarchical team structures")
    
    # Create main team directory
    main_team_dir = "./example_hierarchical_team"
    ensure_dirs_exist(main_team_dir)
    
    # Create main team context
    main_ctx = {
        "project_name": "Multi-Department Analytics",
        "foreman": "{{ref('ceo')}}"
    }
    main_ctx_path = os.path.join(main_team_dir, "team.ctx")
    write_yaml_file(main_ctx_path, main_ctx)
    
    # Create CEO NPC
    ceo = NPC(
        "ceo",
        primary_directive="You are the CEO of the analytics organization. You delegate tasks to the appropriate department and synthesize final results.",
        model="llama3.2",
        provider="ollama"
    )
    ceo.save(main_team_dir)
    
    # Create departments (sub-teams)
    # 1. Sales Department
    sales_dept_dir = os.path.join(main_team_dir, "sales_dept")
    ensure_dirs_exist(sales_dept_dir)
    
    sales_ctx = {
        "department": "Sales Analytics",
        "foreman": "{{ref('sales_manager')}}"
    }
    sales_ctx_path = os.path.join(sales_dept_dir, "team.ctx")
    write_yaml_file(sales_ctx_path, sales_ctx)
    
    # Sales department NPCs
    sales_manager = NPC(
        "sales_manager",
        primary_directive="You are the sales department manager. You coordinate sales analytics and reporting.",
        model="llama3.2",
        provider="ollama"
    )
    sales_manager.save(sales_dept_dir)
    
    sales_analyst = NPC(
        "sales_analyst",
        primary_directive="You analyze sales data to identify trends, forecast future sales, and provide insights.",
        model="llama3.2",
        provider="ollama"
    )
    sales_analyst.save(sales_dept_dir)
    
    # 2. Marketing Department
    marketing_dept_dir = os.path.join(main_team_dir, "marketing_dept")
    ensure_dirs_exist(marketing_dept_dir)
    
    marketing_ctx = {
        "department": "Marketing Analytics",
        "foreman": "{{ref('marketing_manager')}}"
    }
    marketing_ctx_path = os.path.join(marketing_dept_dir, "team.ctx")
    write_yaml_file(marketing_ctx_path, marketing_ctx)
    
    # Marketing department NPCs
    marketing_manager = NPC(
        "marketing_manager",
        primary_directive="You are the marketing department manager. You coordinate marketing analytics and campaign optimization.",
        model="llama3.2",
        provider="ollama"
    )
    marketing_manager.save(marketing_dept_dir)
    
    marketing_analyst = NPC(
        "marketing_analyst",
        primary_directive="You analyze marketing campaign data to measure effectiveness and recommend improvements.",
        model="llama3.2",
        provider="ollama"
    )
    marketing_analyst.save(marketing_dept_dir)
    
    # Load the hierarchical team
    main_team = Team(team_path=main_team_dir)
    
    # Test the hierarchical team
    print("\nTeam structure:")
    print(f"Main team NPCs: {list(main_team.npcs.keys())}")
    for sub_team_name, sub_team in main_team.sub_teams.items():
        print(f"{sub_team_name} NPCs: {list(sub_team.npcs.keys())}")
    
    # Test request that should be routed to sales department
    request = "Analyze our recent sales performance for Q1 and identify which products are underperforming."
    
    print("\nSending sales-related request...")
    result = main_team.orchestrate(request)
    print("\nTeam response to sales request:")
    if 'debrief' in result and isinstance(result['debrief'], dict):
        print(f"Summary: {result['debrief'].get('summary')}")
    else:
        print(result)
    
    return main_team

# ---------------------------------------------------------------------------
# Example 5: Jinx Creation and Conversion from MCP
# ---------------------------------------------------------------------------

def example_5_tool_conversion():
    """Example demonstrating tool creation and conversion from MCP"""
    print("\nExample 5: Jinx creation and conversion from MCP")
    
    # Define an MCP-style function
    def analyze_sentiment(text: str, include_details: bool = False) -> dict:
        """
        Analyze the sentiment of a text.
        
        Args:
            text: The text to analyze
            include_details: Whether to include detailed analysis
            
        Returns:
            Dictionary with sentiment scores
        """
        # Mock implementation
        import random
        
        sentiment = random.choice(["positive", "neutral", "negative"])
        score = random.random()
        
        if sentiment == "positive":
            score = 0.5 + (score / 2)
        elif sentiment == "negative":
            score = score / 2
        else:
            score = 0.4 + (score / 5)
            
        result = {
            "sentiment": sentiment,
            "score": score
        }
        
        if include_details:
            result["details"] = {
                "confidence": random.random(),
                "keywords": ["example", "sentiment", "analysis"]
            }
            
        return result
    
    # Convert MCP function to NPCSH tool
    sentiment_tool = Jinx.from_mcp(analyze_sentiment)
    
    print("Generated Jinx:")
    print(f"Name: {sentiment_tool.tool_name}")
    print(f"Description: {sentiment_tool.description}")
    print(f"Inputs: {sentiment_tool.inputs}")
    print(f"Step count: {len(sentiment_tool.steps)}")
    
    # Create a test directory and save the tool
    tools_dir = "./example_tools"
    ensure_dirs_exist(tools_dir)
    sentiment_tool.save(tools_dir)
    
    # Create a manual tool
    translation_tool = Jinx(tool_data={
        "tool_name": "translate_text",
        "description": "Translate text from one language to another",
        "inputs": [
            {"name": "text", "description": "Text to translate"},
            {"name": "source_lang", "description": "Source language"},
            {"name": "target_lang", "description": "Target language"}
        ],
        "steps": [
            {
                "name": "translate",
                "engine": "natural",
                "code": """
Please translate the following text from {{source_lang}} to {{target_lang}}:

{{text}}

Provide only the translated text without explanations.
"""
            }
        ]
    })
    
    # Save the manual tool
    translation_tool.save(tools_dir)
    
    # Create an NPC with these tools
    language_npc = NPC(
        "linguist",
        primary_directive="You are a language specialist who can analyze and translate text.",
        model="llama3.2",
        provider="ollama",
        tools=[sentiment_tool, translation_tool]
    )
    
    # Test the sentiment tool
    print("\nTesting sentiment tool...")
    sentiment_result = language_npc.execute_tool("analyze_sentiment", {
        "text": "I absolutely love this product! It's amazing and works perfectly.",
        "include_details": True
    })
    print("Sentiment analysis result:")
    print(sentiment_result)
    
    # Test the translation tool
    print("\nTesting translation tool...")
    translation_result = language_npc.execute_tool("translate_text", {
        "text": "Hello world, this is a test.",
        "source_lang": "English",
        "target_lang": "Spanish"
    })
    print("Translation result:")
    print(translation_result)
    
    return language_npc

# ---------------------------------------------------------------------------
# Complete Example - Running All Examples
# ---------------------------------------------------------------------------

def run_all_examples():
    """Run all examples to demonstrate framework functionality"""
    # Initialize database tables
    init_db_tables()
    
    # Run examples
    example_1_simple_npc()
    example_2_npc_team()
    example_3_pipeline()
    example_4_hierarchical_teams()
    example_5_tool_conversion()
    
    print("\nAll examples completed successfully!")

if __name__ == "__main__":
    # Run all examples if executed directly
    run_all_examples()