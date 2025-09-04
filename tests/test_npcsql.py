# Updated test with alphabetized functions

import npcpy.sql.npcsql as nql
import pandas as pd
import os

def create_real_intelligence_models():
    models_dir = os.path.expanduser("~/.npcsh/npc_team/models")
    os.makedirs(models_dir, exist_ok=True)
    
    productivity_intelligence = """
SELECT 
    DATE(timestamp) as date,
    npc,
    COUNT(*) as daily_interactions,
    AVG(LENGTH(content)) as avg_message_depth,
    
    nql.synthesize(
        "Daily activity {date}: {daily_interactions} interactions with {npc}, avg depth {avg_message_depth}. What productivity patterns emerge?",
        "sibiji",
        "productivity_pattern_analysis"
    ) as productivity_insights

FROM conversation_history 
WHERE timestamp >= date('now', '-14 days')
  AND npc IS NOT NULL
GROUP BY DATE(timestamp), npc
ORDER BY date DESC
"""

    command_evolution = """
SELECT 
    command,
    COUNT(*) as usage_count,
    AVG(LENGTH(output)) as avg_complexity,
    
    nql.criticize(
        "Command: {command} used {usage_count} times with {avg_complexity} avg output complexity",
        "frederic",
        "constructive_analysis"  
    ) as command_critique

FROM command_history
WHERE timestamp >= date('now', '-30 days')
GROUP BY command
HAVING usage_count > 3
ORDER BY usage_count DESC
"""

    models = {
        "command_evolution.sql": command_evolution,
        "productivity_intelligence.sql": productivity_intelligence
    }
    
    for filename, content in models.items():
        filepath = os.path.join(models_dir, filename)
        with open(filepath, "w") as f:
            f.write(content)
        print(f"Created {filepath}")

def run_intelligence_pipeline():
    compiler = nql.ModelCompiler(
        models_dir="~/.npcsh/npc_team/models",
        npc_directory="~/.npcsh/npc_team"
    )
    
    results = compiler.run_all_models()
    
    for model_name, df in results.items():
        print(f"\n=== {model_name.upper()} ===")
        print(df.to_string())

if __name__ == "__main__":
    create_real_intelligence_models()
    run_intelligence_pipeline()