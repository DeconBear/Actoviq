import asyncio
import json
import os
import shlex
import threading
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, create_model

# LangChain tools
from langchain_core.tools import StructuredTool

# MCP / FastMCP clients
try:
    from fastmcp.client import FastMCPClient
    from mcp.client.stdio import stdio_client, StdioServerParameters
    from mcp import ClientSession, Tool
except ImportError:
    FastMCPClient = None
    stdio_client = None
    ClientSession = None
    Tool = Any


class MCPManager:
    """
    Manages connections to multiple MCP Servers and exposes their tools as LangChain tools.
    Since KlynxAgent is synchronous, this manager runs an internal asyncio event loop
    in a background thread to maintain the long-lived MCP stdio connections.
    """

    def __init__(self):
        self._servers: Dict[str, Any] = {}
        self._langchain_tools: List[StructuredTool] = []
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        """Run the asyncio event loop in a background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def load_from_json(self, json_path: str) -> List[StructuredTool]:
        """
        Loads MCP servers from a Claude Desktop style config file.
        Format expected:
        {
          "mcpServers": {
            "name": {
              "command": "node",
              "args": ["..."],
              "env": {"KEY": "VAL"}
            }
          }
        }
        """
        if not stdio_client:
            raise ImportError("mcp or fastmcp package not found. Please install them.")

        if not os.path.exists(json_path):
            print(f"[MCP] Config file not found: {json_path}")
            return []

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
            print(f"[MCP] Error reading config: {e}")
            return []

        servers = config.get("mcpServers", {})
        if not servers:
            return []

        new_tools = []
        
        # Schedule the async connection tasks safely on the loop
        future = asyncio.run_coroutine_threadsafe(
            self._connect_all_servers(servers), self._loop
        )
        
        # Wait synchronously for all servers to connect and tools to be fetched
        try:
            tools_by_server = future.result(timeout=30.0)
            
            for server_name, mcp_tools in tools_by_server.items():
                for tool in mcp_tools:
                    lc_tool = self._wrap_mcp_tool_as_langchain(server_name, tool)
                    new_tools.append(lc_tool)
                    self._langchain_tools.append(lc_tool)
                    
        except Exception as e:
            print(f"[MCP] Failed to connect and fetch tools: {e}")

        return new_tools

    async def _connect_all_servers(self, servers_dict: Dict[str, dict]) -> Dict[str, List[Tool]]:
        """Async method to connect to all configured servers in parallel."""
        tasks = []
        server_names = []
        
        for name, config in servers_dict.items():
            command = config.get("command")
            args = config.get("args", [])
            env = config.get("env", None)
            
            if not command:
                continue
                
            server_names.append(name)
            tasks.append(self._connect_single_server(name, command, args, env))
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        tools_dict = {}
        for name, result in zip(server_names, results):
            if isinstance(result, Exception):
                print(f"[MCP] Error connecting to server '{name}': {result}")
            else:
                tools_dict[name] = result
                print(f"[MCP] Successfully connected to server '{name}' and loaded {len(result)} tools.")
                
        return tools_dict

    async def _connect_single_server(self, name: str, command: str, args: List[str], env: Optional[Dict[str, str]]) -> List[Tool]:
        """Connect to a single MCP server via stdio and fetch tools."""
        
        # Merge environment variables if provided
        server_env = None
        if env:
            server_env = os.environ.copy()
            server_env.update(env)
            
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=server_env
        )
        
        # We need to keep the context managers alive for the lifetime of the application.
        # So we manage them manually using AsyncEnv
        from contextlib import AsyncExitStack
        stack = AsyncExitStack()
        
        try:
            stdio_transport = await stack.enter_async_context(stdio_client(server_params))
            read, write = stdio_transport
            session = await stack.enter_async_context(ClientSession(read, write))
            
            await session.initialize()
            
            # Store the live session and stack so it doesn't close
            self._servers[name] = {
                "session": session,
                "stack": stack
            }
            
            # Fetch tools
            response = await session.list_tools()
            return response.tools
            
        except Exception as e:
            await stack.aclose()
            raise e

    def _wrap_mcp_tool_as_langchain(self, server_name: str, tool: Tool) -> StructuredTool:
        """Dynamically create a LangChain StructuredTool from an MCP Tool definition."""
        name = f"mcp_{server_name}.{tool.name}"
        description = tool.description or f"MCP tool {tool.name} from server {server_name}"
        
        # Convert JSON schema to Pydantic Model dynamically
        schema = tool.inputSchema
        fields = {}
        
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        for prop_name, prop_data in properties.items():
            prop_type = Any
            # Basic basic type mapping
            if prop_data.get("type") == "string":
                prop_type = str
            elif prop_data.get("type") == "integer":
                prop_type = int
            elif prop_data.get("type") == "number":
                prop_type = float
            elif prop_data.get("type") == "boolean":
                prop_type = bool
            elif prop_data.get("type") == "array":
                prop_type = list
            elif prop_data.get("type") == "object":
                prop_type = dict
                
            default_val = ... if prop_name in required else None
            fields[prop_name] = (prop_type, default_val)
            
        args_schema = create_model(f"{tool.name}Schema", **fields)
        
        # Define the sync function that will execute the tool
        # By submitting a coroutine back to our background event loop
        def sync_caller(**kwargs) -> Any:
            try:
                print(f"[MCP] Calling '{name}' with args: {kwargs}")
                session = self._servers[server_name]["session"]
                
                # Execute asynchronously on the background loop
                coro = session.call_tool(tool.name, kwargs)
                future = asyncio.run_coroutine_threadsafe(coro, self._loop)
                
                # Block and wait for result
                result = future.result(timeout=60.0)
                
                # Format the MCP CallToolResult into a string for the LLM
                output = []
                for content in result.content:
                    if content.type == "text":
                        output.append(content.text)
                    elif content.type == "resource":
                        output.append(f"Resource [{content.resource.uri}]:\n{content.resource.text}")
                
                if result.isError:
                    return "Error: " + "\n".join(output)
                return "\n".join(output)
                
            except Exception as e:
                return f"MCP Tool Execution Error: {e}"
        
        return StructuredTool(
            name=name,
            description=description,
            args_schema=args_schema,
            func=sync_caller
        )

    def get_all_tools(self) -> List[StructuredTool]:
        """Returns all currently registered LangChain tools from MCP."""
        return self._langchain_tools

    def close(self):
        """Cleanup all connections and stop the event loop."""
        async def _close_all():
            for name, server_dict in self._servers.items():
                try:
                    await server_dict["stack"].aclose()
                    print(f"[MCP] Closed connection to {name}")
                except Exception:
                    pass
        
        if self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(_close_all(), self._loop)
            try:
                future.result(timeout=5.0)
            except Exception:
                pass
            
            self._loop.call_soon_threadsafe(self._loop.stop)
        
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
