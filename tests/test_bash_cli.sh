# Basic tests and help commands
npc help
npc --help

# Test serve command with different port specification formats
npc serve --port=5340
npc serve --port 5341
npc serve -port=5342
npc serve -port 5343
npc serve --cors="http://localhost:3000,http://localhost:8080"

# Test search commands with different providers
npc search "python asyncio tutorial"
npc search -p google "python asyncio tutorial"
npc search --provider perplexity "machine learning basics"

# Test sample/LLM commands
npc sample "Write a hello world program in Python"
npc sample "Compare Python and JavaScript" --model gpt-4 --provider openai

# Test file and image processing
npc rag "litellm" -f ../setup.py
npc rag --file /path/to/document.txt "What is this document about?"
npc ots /path/to/screenshot.png "What's happening in this image?"
npc vixynt "A beautiful sunset over mountains" filename=sunset.png height=512 width=768

# Test with different NPC selection
npc -n custom_npc "Tell me about yourself"
npc --npc alternative_assistant "How can you help me?"

# Test other route commands
npc sleep 3
npc jinxs
npc init /tmp/new_npc_project
npc wander "How to implement a cache system"
npc plan "Create a new Python project with virtual environment"
npc trigger "Update all npm packages in the current directory"

# Testing command with multiple arguments and options
npc serve --port 5338 --cors "http://localhost:3000"
npc vixynt "A city made by pepsi" height=1024 width=1024 filename=pepsi_city.png
npc rag -f /path/to/file1.txt -f /path/to/file2.txt "Compare these two documents"



# Pipe file content to NPC sample command
cat data.json | npc sample "Summarize this JSON data"

# Pipe command output to NPC
ls -la | npc sample "Explain what these files are"

# Use grep to filter logs and have NPC analyze them
grep ERROR /var/log/application.log | npc sample "What are the common error patterns?"

# Use curl to get API response and analyze with NPC
curl -s https://api.example.com/data | npc sample "Analyze this API response"

# Create a multi-line prompt using heredoc
cat << EOF | npc sample
I need to understand how to structure a React application.
What are the best practices for component organization?
Should I use Redux or Context API for state management?
EOF




# Chain NPC commands using xargs
npc search "machine learning algorithms" | xargs -I {} npc sample "Explain {} in detail"

# Use output from one NPC command as input to another
npc sample "Generate 5 test cases" | npc sample "Convert these test cases to JavaScript code"

# Use NPC to generate code and then analyze it
npc sample "Write a Python sorting algorithm" | npc sample "Review this code for efficiency"

# Generate image description and then create image
npc sample "Describe a futuristic cityscape" | xargs npc vixynt