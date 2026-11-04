"""Explicit, lazy agent registration."""
from __future__ import annotations
import importlib
from typing import Any
from .contracts import Agent

_AGENT_REGISTRY: dict[str, Any] = {}


def register_agent(name: str):
    """Register an agent class.

    Usage:
        @register_agent("codex")
        class CodexAgent:
            def run(self, ...): ...
    """

    def wrapper(agent_cls):
        _AGENT_REGISTRY[name] = agent_cls
        return agent_cls

    return wrapper


def get_agent_backend(name: str) -> Any:
    """Resolve a registered agent or lazily load a built-in SDK adapter."""
    if name not in _AGENT_REGISTRY:
        path = _BUILTIN_AGENTS.get(name)
        if not path:
            available = ", ".join(sorted(_BUILTIN_AGENTS))
            raise ValueError(f"Unknown agent backend: {name!r}. Available: {available}")
        module_path, cls_name = path.rsplit(":", 1)
        _AGENT_REGISTRY[name] = getattr(importlib.import_module(module_path), cls_name)
    return _AGENT_REGISTRY[name]


# Agent registry

_BUILTIN_AGENTS: dict[str, str] = {
    "claude_code": "uni_robot_agent.agents.claude_code.agent:ClaudeCodeAgent",
    "codex": "uni_robot_agent.agents.codex.agent:CodexAgent",
    "steward": "uni_robot_agent.agents.steward.agent:StewardAgent",
}


def create_agent(backend: str, **kwargs) -> Agent:
    """Create an agent by backend name with lazy loading."""
    return get_agent_backend(backend)(**kwargs)


def list_agent_backends() -> tuple[str, ...]:
    """Return built-in and explicitly registered backends without SDK imports."""
    return tuple(sorted(set(_BUILTIN_AGENTS) | set(_AGENT_REGISTRY)))
