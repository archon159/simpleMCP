# simpleMCP
Simple MCP client and server implementation.

### Usage
1. Run the custom MCP server.
```
python3 custom_server.py
```

2. Run the brave-search MCP server.
```
npx -y @brave/brave-search-mcp-server --transport http --brave-api-key BRAVE_API_KEY
```

3. Run the MCP client.
```
python3 mcp_client.py -m "Where does the computer scientist Seungeon Lee work?"
```


### Requirements

python == 3.10.19

mcp == 1.26.0

numpy == 1.26.4

torch == 2.2.2

transformers == 4.57.1
