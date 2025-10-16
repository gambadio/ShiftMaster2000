"""
MCP (Model Context Protocol) Configuration and Integration

Handles MCP server connections and tool management
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from models import MCPServerConfig


def validate_mcp_config(server: MCPServerConfig) -> tuple[bool, Optional[str]]:
    """
    Validate MCP server configuration

    Returns:
        (is_valid, error_message)
    """
    if not server.name or not server.name.strip():
        return False, "Server name is required"

    if not server.command or not server.command.strip():
        return False, "Command is required"

    return True, None


def format_mcp_tools_for_prompt(mcp_servers: List[MCPServerConfig]) -> str:
    """
    Format MCP server configurations for inclusion in system prompt

    Args:
        mcp_servers: List of MCP server configurations

    Returns:
        Formatted string describing available tools
    """
    if not mcp_servers:
        return ""

    lines = ["### Available MCP Tools", ""]
    lines.append("You have access to the following external tools via MCP servers:")
    lines.append("")

    for i, server in enumerate(mcp_servers, 1):
        lines.append(f"{i}. **{server.name}**")
        lines.append(f"   - Command: `{server.command}`")
        if server.args:
            lines.append(f"   - Args: {' '.join(server.args)}")
        lines.append("")

    lines.append("Use these tools when you need additional information or capabilities beyond the provided data.")
    lines.append("")

    return "\n".join(lines)


async def connect_to_mcp_servers(servers: List[MCPServerConfig]) -> List[Any]:
    """
    Connect to MCP servers (placeholder for actual MCP integration)

    Args:
        servers: List of MCP server configurations

    Returns:
        List of connected MCP clients

    Note:
        This is a placeholder. Actual implementation requires:
        - Installing mcp package: pip install mcp
        - Implementing connection logic
        - Error handling for failed connections
    """
    # Placeholder implementation
    # In production, this would use the MCP SDK to connect to servers

    connected_clients = []

    for server in servers:
        # Example connection code (commented out - requires mcp package):
        # try:
        #     from mcp import Client, StdioServerParameters
        #     client = await Client.connect(
        #         StdioServerParameters(
        #             command=server.command,
        #             args=server.args,
        #             env=server.env
        #         )
        #     )
        #     connected_clients.append(client)
        # except Exception as e:
        #     print(f"Failed to connect to MCP server {server.name}: {e}")

        # For now, just log the attempt
        print(f"MCP server configured: {server.name} (command: {server.command})")

    return connected_clients


def get_mcp_server_examples() -> List[Dict[str, Any]]:
    """
    Get example MCP server configurations

    Returns:
        List of example server configs as dictionaries
    """
    return [
        {
            "name": "Filesystem",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/files"],
            "description": "Access to local filesystem for reading/writing files"
        },
        {
            "name": "Web Search",
            "command": "python",
            "args": ["-m", "mcp_server_brave_search"],
            "description": "Web search capabilities via Brave Search API"
        },
        {
            "name": "Database",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-postgres"],
            "description": "PostgreSQL database access"
        },
        {
            "name": "Git",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-git"],
            "description": "Git repository operations"
        }
    ]
