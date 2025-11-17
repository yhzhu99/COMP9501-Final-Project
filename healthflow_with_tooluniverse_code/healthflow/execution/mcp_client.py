import asyncio
import logging
import os
from typing import List, Dict, Any

from dotenv import load_dotenv

load_dotenv()

from langchain_mcp_adapters.client import MultiServerMCPClient

async def fetch_mcp_tools():
    """Fetch MCP tools with error handling and fallback options."""

    # client = MultiServerMCPClient(
    #     {
    #         "web-search": {
    #             "args": [
    #                 "open-websearch@latest"
    #             ],
    #             "transport": "stdio",
    #             "command": "npx",
    #             "env": {
    #                 "MODE": "stdio",
    #                 "DEFAULT_SEARCH_ENGINE": "bing",
    #                 "ALLOWED_SEARCH_ENGINES": "bing"
    #             }
    #         }
    #     }
    # )

    # Dynamically assemble servers to avoid passing None values
    servers: Dict[str, Dict[str, Any]] = {}

    serper_api_key = os.getenv("SERPER_API_KEY")
    if serper_api_key:
        servers["serper"] = {
            "command": "uvx",
            "args": ["serper-mcp-server"],
            "env": {"SERPER_API_KEY": serper_api_key},
            "transport": "stdio",
        }

    tool_universe_path = os.getenv("TOOL_UNIVERSE_PATH")
    
    if tool_universe_path:
        servers["tooluniverse"] = {
            "command": "uv",
            "args": ["--directory", tool_universe_path, "run", "tooluniverse-mcp-claude"],
            "transport": "stdio",
        }

    # Optional: log what will be registered (mask secrets)
    try:
        masked = {
            name: {
                **cfg,
                "env": {k: ("***" if k.lower().endswith("key") else v) for k, v in (cfg.get("env") or {}).items()}
            }
            for name, cfg in servers.items()
        }
        logging.debug(f"MCP servers configured: {list(masked.keys())}")
    except Exception:
        pass

    if not servers:
        return []

    client = MultiServerMCPClient(servers)

    tools = await client.get_tools()
    return tools

def format_mcp_tools_for_planning(tools: List[Any]) -> str:
    """
    Format MCP tools information for the planning agent.

    Args:
        tools: List of MCP tools

    Returns:
        Formatted string describing available MCP tools for planning
    """
    if not tools:
        return "No MCP tools are currently available."

    tool_descriptions = []
    for tool in tools:
        # Get tool name and description
        tool_name = getattr(tool, 'name', 'Unknown Tool')
        tool_description = getattr(tool, 'description', 'No description available')

        # Get tool schema information
        if hasattr(tool, 'args_schema'):
            if isinstance(tool.args_schema, dict):
                schema = tool.args_schema
            elif hasattr(tool.args_schema, 'model_json_schema'):
                schema = tool.args_schema.model_json_schema()
            else:
                schema = getattr(tool.args_schema, 'schema', lambda: {})()
        else:
            schema = {"properties": {}, "required": []}

        # Format parameters
        params = []
        for prop_name, prop_info in schema.get("properties", {}).items():
            param_type = prop_info.get('type', 'Any')
            if prop_name in schema.get("required", []):
                params.append(f"{prop_name}: {param_type}")
            else:
                default = prop_info.get('default', 'None')
                params.append(f"{prop_name}: {param_type} = {default}")

        param_str = "(" + ", ".join(params) + ")" if params else "()"

        tool_descriptions.append(f"- **{tool_name}**{param_str}: {tool_description}")

    return "Available MCP Tools:\n" + "\n".join(tool_descriptions)

async def get_mcp_tools_info() -> Dict[str, Any]:
    """
    Get comprehensive information about MCP tools for planning purposes.

    Returns:
        Dictionary containing tools count and formatted descriptions
    """
    try:
        tools = await fetch_mcp_tools()
        tools_description = format_mcp_tools_for_planning(tools)

        return {
            "count": len(tools),
            "tools": tools,
            "description": tools_description
        }
    except Exception as e:
        logging.error(f"Failed to fetch MCP tools information: {e}")
        return {
            "count": 0,
            "tools": [],
            "description": f"Error fetching MCP tools: {str(e)}"
        }
