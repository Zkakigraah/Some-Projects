from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class ToolResult:
    """Encapsulates the outcome of an invoked agent tool."""
    success: bool
    output: str
    metadata: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILURE"
        return f"[{status}] {self.output}"


class BaseTool(ABC):
    """
    Abstract interface for all tools accessible by TinyAgent.
    Every tool must define its name, description, expected parameters, and execution logic.
    """

    def __init__(self, name: str, description: str, usage_doc: str):
        self.name = name
        self.description = description
        self.usage_doc = usage_doc

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Executes the tool given specific argument parameters."""
        pass

    def to_schema(self) -> Dict[str, Any]:
        """Provides schema specification ready for future LLM tool-calling format."""
        return {
            "name": self.name,
            "description": self.description,
            "usage": self.usage_doc
        }


class ToolRegistry:
    """Manages discovery and execution of available tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                output=f"Tool '{tool_name}' is not registered."
            )
        try:
            return tool.execute(**kwargs)
        except Exception as e:
            return ToolResult(
                success=False,
                output=f"Error during execution of '{tool_name}': {str(e)}"
            )

    def list_tools(self) -> Dict[str, str]:
        return {name: tool.description for name, tool in self._tools.items()}