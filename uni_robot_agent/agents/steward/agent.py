"""DeepSeek Harness (steward) adapter; the agent loop runs in a dsh child process."""
from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from uni_robot_agent.agents.contracts import AgentCapabilities, AgentResult
from uni_robot_agent.events import AgentRunStarted
from uni_robot_agent.services.mcp.http import HttpMcpServer

from .config import StewardConfig

MCP_SERVER_NAME = "uni_robot_agent"
MCP_TOOL_TIMEOUT_MS = 600_000
_INITIALIZE_TIMEOUT_S = 120.0
# dsh plugins that would expose host shell, files or ambient skills to a robot episode.
_NATIVE_TOOL_PLUGINS = (
    "tool-bash", "tool-pwsh", "tool-fs", "tool-fs-search",
    "skill", "skill-filesystem", "tool-skill", "tool-web",
)


class _RecordingExecutor:
    """Count dispatched calls and retain the finish outcome for the agent result."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self.tool_calls = 0
        self.finish: Any = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        result = self._inner.call(name, arguments)
        with self._lock:
            self.tool_calls += 1
            if name == "finish":
                self.finish = result
        return result


class StewardAgent(StewardConfig):
    capabilities = AgentCapabilities()

    def __init__(self, *, model: str, dsh_home: str | None = None, profile: str = "sdk",
                 patches: tuple[str, ...] = (), provider: str | None = None,
                 base_url: str | None = None, api_key: str | None = None,
                 context_window: int = 262144, max_tokens: int = 32768,
                 native_tools: bool = True, cwd: str | None = None) -> None:
        self._model = model
        self._dsh_home = dsh_home
        self._profile = profile
        self._patches = tuple(patches)
        self._provider = provider
        self._base_url = base_url.rstrip("/") if base_url else None
        self._api_key = api_key
        self._context_window = context_window
        self._max_tokens = max_tokens
        self._native_tools = native_tools
        self._cwd = cwd
        self._harness = None
        self._cancel_reason: str | None = None
        self._cleanup_error: str | None = None

    def model_runtime(self, profile, event_bus=None):
        raise RuntimeError(
            "steward routes model traffic through dsh; use --steward-base-url "
            "instead of --provider-url"
        )

    def finalize_without_provider(self, result, session_dir):
        if session_dir is not None:
            result.train_error = (
                "steward model records live in the dsh home, not in uni-robot-agent"
            )

    def cancel(self, reason):
        self._cancel_reason = reason
        harness = self._harness
        if harness is not None:
            try:
                harness.close()
            except Exception:  # noqa: BLE001 - cancellation is best effort
                pass

    def close(self):
        if self._cleanup_error:
            raise RuntimeError(self._cleanup_error)
        if self._harness is not None:
            raise RuntimeError("steward run has not finished cleanup")

    def run(self, *, prompt, task, toolkit, executor, max_turns=0,
            event_bus=None, session_dir=None, model_endpoint=None) -> AgentResult:
        started = time.monotonic()
        proxy = _RecordingExecutor(executor)
        server = HttpMcpServer(proxy, event_bus=event_bus)
        mcp_url = server.start()
        error = None
        stop_reason = None
        cleanup_error = None
        response = ""
        run = None
        try:
            patch = self._write_patch(session_dir, prompt.system, mcp_url)
            if event_bus:
                event_bus.emit(AgentRunStarted(
                    instruction=task, model=self._model,
                    agent_backend="steward", tool_event_source="execution",
                ))
            from deepseek_harness import DeepSeekHarness

            env = {"STEWARD_API_KEY": self._api_key or "local"}
            harness = DeepSeekHarness(
                dsh_home=self._dsh_home,
                cwd=self._workspace(session_dir),
                profile=self._profile,
                patches=(str(patch), *self._patches),
                provider=self._provider,
                model=self._model,
                env=env,
                initialize_timeout_seconds=_INITIALIZE_TIMEOUT_S,
            )
            try:
                self._harness = harness
                harness.start()
                run = harness.run(prompt.user)
            finally:
                try:
                    harness.close()
                except Exception as exc:  # noqa: BLE001 - report cleanup without masking
                    cleanup_error = f"{type(exc).__name__}: {exc}"
                    error = error or cleanup_error
                self._harness = None
            response = run.final_response
            if self._cancel_reason:
                stop_reason = self._cancel_reason
            elif run.finish_reason not in (None, "completed"):
                stop_reason = str(run.finish_reason)
        except Exception as exc:  # noqa: BLE001 - surface the failure in AgentResult
            error = f"{type(exc).__name__}: {exc}"
        finally:
            self._harness = None
            try:
                server.stop()
            except Exception as exc:  # noqa: BLE001 - server shutdown is best effort
                cleanup_error = cleanup_error or f"{type(exc).__name__}: {exc}"
                error = error or cleanup_error

        finish = proxy.finish
        finish_ok = finish is not None and (
            getattr(finish, "status", "") == "accepted"
            or (getattr(finish, "output", None) is not None
                and finish.output.success and not finish.output.error)
        )
        message = self._message(finish, response)
        turns = sum(1 for event in (run.events if run is not None else [])
                    if isinstance(event, dict) and event.get("type") == "turn/end")
        return AgentResult(
            success=finish_ok and not error and not stop_reason,
            message=message, error=error, cleanup_error=cleanup_error,
            stop_reason=stop_reason, turns_used=turns, tool_calls=proxy.tool_calls,
            steps_used=turns + proxy.tool_calls,
            elapsed_s=time.monotonic() - started,
            trace_format="steward_events",
        )

    def _workspace(self, session_dir) -> str:
        if self._cwd:
            return str(Path(self._cwd).resolve())
        if session_dir is not None:
            return str(Path(session_dir).resolve())
        return str(Path.cwd())

    def _write_patch(self, session_dir, system_prompt: str, mcp_url: str) -> Path:
        root = Path(session_dir) if session_dir is not None else Path(
            tempfile.mkdtemp(prefix="steward-"))
        path = root / "steward.patch.yml"
        document: list[dict[str, Any]] = []
        providers: dict[str, Any] = {}
        if self._base_url:
            profile: dict[str, Any] = {
                "api": "openai-completions",
                "baseURL": self._base_url,
                "models": [{
                    "id": self._model,
                    "name": self._model,
                    "contextWindow": self._context_window,
                    "maxTokens": self._max_tokens,
                    # Declaring images is what admits observation tool results.
                    "input": ["text", "image"],
                    "reasoningEfforts": False,
                }],
            }
            # pi-ai refuses a route with no credential source; local servers
            # accept any placeholder, so the env reference is unconditional.
            profile["apiKeyEnv"] = "STEWARD_API_KEY"
            providers[str(self._provider)] = profile
            document.append({"id": "llm-pi-ai", "config": {"providers": providers}})
        document.append({
            "id": "system-prompt",
            "config": {"personaPrefix": system_prompt, "includeHarnessIdentity": False},
        })
        document.append({"insert": [{
            "id": "mcp-uni-robot-agent",
            "name": "@deepseek-ai/dsh-mcp-client",
            "config": {
                "serverName": MCP_SERVER_NAME,
                "transport": "streamable-http",
                "url": mcp_url,
                "toolCallTimeoutMs": MCP_TOOL_TIMEOUT_MS,
                "failOnStartupError": True,
            },
        }]})
        if not self._native_tools:
            # A robot episode exposes only the harness tools; ambient shell and
            # filesystem access would reach simulator files and host state.
            document.extend({"id": name, "disabled": True} for name in _NATIVE_TOOL_PLUGINS)
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def _message(finish, response: str) -> str:
        extra = getattr(getattr(finish, "output", None), "extra", None)
        if isinstance(extra, dict):
            summary = extra.get("summary") or extra.get("message")
            if isinstance(summary, str) and summary.strip():
                return summary
        return response or "steward session ended without output"
