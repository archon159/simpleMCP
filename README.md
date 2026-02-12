# simpleMCP
Simple MCP client and server implementation.

### Usage

1. Create "secrets.env". You can insert API keys for running servers here.
```
touch ./secrets.env
```


2. Run MCP servers. This will run all MCP servers defined in "mcp_servers" directory.
```
python3 run_mcp_servers.py
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
