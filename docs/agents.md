# Agent integration

`uni_robot_agent/agents/` owns agent-specific configuration, SDK lifecycle,
tool dispatch, event interpretation and native artifact export. `session.py`
invokes a registered agent without switching on its name. Robot tools and
resource scheduling remain in `tools/` and `execution/`.

## Codex calculation interpreter

`--agent-python /path/to/compute-venv/bin/python` (Python API:
`AgentConfig(agent_python=...)`) prepends that interpreter's directory to the
Codex built-in shell PATH. Use an environment containing both `python` and
`python3` and the calculation libraries the agent needs. The executable is
validated before startup and its absolute path is recorded in resolved config;
venv symlinks are preserved. This option does not select the uni-robot-agent, simulator,
or perception-service interpreter and does not install packages.

When set, Codex login shells, shell profiles and environment snapshots are
disabled so they cannot restore another Python PATH. An explicit `login=true`
tool call is rejected; omitted or false login uses a non-login shell. Explicit
absolute interpreter paths or commands that change PATH can still bypass this
selection. Without the option, existing SDK environment behavior is unchanged.
This setting requires the Codex backend; Claude Code rejects it.

## Adding an agent

1. Add a package under `agents/<name>/` with `__init__.py`.
2. Declare `model_protocol` (`responses` or `messages`) on the agent configuration
   and implement the synchronous, single-task contract in `agents/contracts.py`. An SDK may
   use an event loop internally; it must finish cleanup before `run` returns.
3. Add its lazy import path to `agents/registry.py`. SDK imports belong in
   the run path, so configuration validation remains offline.
4. Add lifecycle and protocol tests. New SDK dependencies require a dependency
   declaration; new upstream vendor conversions are separate provider work.

Configuration methods are `add_cli_arguments`, `resolve_defaults`,
`validate_config`, and `config_options`. Shared model transport flags are
registered by `agents/runtime.py`. Use distinct names for SDK-only options;
reject unsupported settings before connecting to robot or model services.

`model_runtime(profile, event_bus)` returns a `ModelRuntime` containing:

- `trace_directory`: the agent's native model record directory;
- `adapter`: the upstream URL and request/response codec;
- `protocol`: request matching, generation controls, context preparation,
  complete-response error detection, usage extraction and error encoding;
- `binding`: request identity and response prepare/delivery/revocation hooks;
- `finalize_records`: diagnosis/export of immutable model records.

Standard protocols live in `provider/protocols/`. An agent can provide a
private protocol implementation from its own package using the same methods.
The gateway does not need another agent-name branch. A protocol conversion
must explicitly support both its input and upstream wire formats.

`run` receives the assembled `Prompt`, executor, event bus, budgets, output
directory and model endpoint. Tool execution must go through that executor.
The provider may register proposed calls, but must never execute them in
addition to SDK tool callbacks. `cancel` requests termination; `close` must
report unfinished cleanup. `finalize_without_provider` handles runs without
model recording. The environment remains authoritative for task success.

## Codex compatibility

The Codex plugin configures SDK options and prompt modes, runs its synchronous
worker, and translates SDK events and results. Native provider records use
`codex_responses/`; validated SFT exports use `train.jsonl`.

Provider names and registration are documented in [Model providers](providers.md).
Agent configuration declares `model_protocol`; profile selection and validation
use the common registry rather than vendor branches in the planner.
Context selection precedes adapter
encoding. DeepSeek Files preparation remains scoped to the model client.
Complete responses are buffered before retry/delivery; tool batches commit
only after the final response send, and failed delivery revokes them.

## Claude Code

Select `--agent claude_code --model <provider-model>` and supply
`--provider-url <native-messages-api-root>`. `ANTHROPIC_BASE_URL` is a fallback
when no uni-robot-agent provider URL is configured. The URL may end in `/v1`.
Credentials come from the configured uni-robot-agent API key, `ANTHROPIC_AUTH_TOKEN`,
or the SDK's `ANTHROPIC_API_KEY`; keep them outside command arguments and Git.

Claude runs through the uni-robot-agent gateway using native Anthropic Messages.
Use `--provider-profile passthrough.messages` or `aliyun.messages`; Responses-to-Chat adapters are not Messages
adapters. Shared provider timeout/retry flags apply. The SDK retry count is
set to zero so uni-robot-agent owns full-exchange retries.

The SDK receives the robot system prompt and user input separately. Native file,
shell and Agent tools are enabled alongside the robot MCP tools. Hooks approve
native tools automatically, including access outside the working directory, and
the SDK sandbox is disabled. The MCP allowed-tools list is not a filesystem
boundary. Ambient MCP configuration, settings and SDK skill discovery are disabled;
robot skills remain available through the harness tools. Claude state is isolated
under the run's `.claude/` directory.
The SDK JSON buffer remains 64 MiB for large image tool results. Claude Code
may itself transform images before its subsequent model request; the gateway
records those actual requests.

`--claude-effort` configures the SDK effort. `--claude-max-budget-usd` optionally
sets a cost ceiling. Codex-specific
MCP controls and non-`none` `--reasoning-effort` are rejected for Claude.
`--no-model-parallel-tool-calls` maps to native `disable_parallel_tool_use`.

The model binding records tool-use IDs and batch order. A pre-tool hook carries
that exact ID to MCP outside the model-visible schema. Calls wait for response
delivery and the previous call's executor acknowledgement. Background jobs
can overlap under the common resource scheduler; immediate calls retain their
waiting semantics. Duplicate, revoked or unidentified callbacks cannot execute.
Cancelling the SDK waiter does not claim that a running robot action stopped.

The bridge isolates one private dependency in `sdk_compat.py`: the server
builder from the pinned `claude-agent-sdk==0.2.165`. It supports the SDK's MCP
1.x/2.x transport boundary while allowing removal of hook metadata before
schema validation. Revalidate this boundary on SDK upgrades.

Claude success requires an actual successful `finish` result and normal SDK
termination. Plain text, a proposed finish, or an SDK error is insufficient.
Records use `claude_messages/`. Native requests/responses and usage are retained;
provider-faithful SFT export is explicitly unsupported. The Messages
implementation supports only `identity` context policy and single-run SDK
sessions. Each TUI accepts one task; start another TUI for a new task.

Validate SDK changes against the pinned Claude CLI and a local scripted Messages
provider, including large image round trips, multiple tool calls, finish, retries
and cancellation. These checks can run without a commercial model or robot.

## steward (DeepSeek Harness)

`--agent steward` runs the [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)
loop as a child process through `deepseek-harness-sdk` (the `steward` extra).
The adapter starts the same `HttpMcpServer` the Codex adapter uses, writes a
generated Cordis patch, and lets dsh connect to it, so the shared tools, resource
scheduling and episode state are unchanged.

Configuration: `--steward-dsh-home` (or `DSH_HOME`) is the harness home the SDK
requires. `--steward-provider`/`--steward-base-url` (or `--provider-url`) declare
the model route the generated patch registers through `llm-pi-ai`; the route
declares `[text, image]` so observation images survive the MCP image gate.
uni-robot-agent's own provider gateway stays closed: dsh owns model transport,
retries and records. `--no-steward-native-tools` disables dsh's bash, filesystem,
web and skill plugins when a real robot must not reach host state. Only the
`identity` context policy is supported. Agent success requires a `finish` call
that was accepted or completed, plus a normal SDK termination.

Native model records live in the dsh home, not under the session's trace
directory; provider-faithful SFT export is unsupported. Validate SDK changes
against the pinned `deepseek-harness-sdk` and a local OpenAI-compatible endpoint,
including an image-bearing tool result, a background `finish` acknowledgement and
cancellation.
