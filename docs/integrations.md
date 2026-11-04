# Environment integrations

uni-robot-agent uses shared robot tools across simulator and physical environments.
Both `run` and `tui` require an explicit `--integration`; there is no default.
Choose an integration, then prepare its external services. Python extras install
harness-side dependencies; simulators, hardware runtimes, assets and model
checkpoints remain separate.

## Included integrations

| Integration | Use | External services |
|---|---|---|
| `robodojo` | Official RoboDojo simulation, task requirements and bundled gripper skills | RoboDojo environment; Hybrid adds an identified Pi0.5 model and a reviewer by default |
| `robouse` | Direct simulation tasks, including LIBERO and RoboCasa | Prepared profile-specific RoboUse environment |
| `robolab` | RoboLab DROID, Direct scalar-gripper control | Prepared RoboLab/Isaac environment; RGB cameras |
| `cobot_magic` | Steward WSim twin of the dual-Piper Cobot Magic, Direct control | Twin episode service; the `steward` Python package in the service interpreter |

These simulation integrations share scene, gripper, skill and episode tools.
RoboDojo supports Direct and Hybrid; RoboUse and RoboLab support Direct only.
RoboLab has no automatic setup/deploy recipe. Use
SAM3 only where the integration supports it; otherwise pass `--no-sam3`. Depth-based pixel
localization requires paired depth and calibration; RGB triangulation requires
at least two calibrated views.

## Real-robot release status

Public support for **ARX One** and **Franka** is **Coming soon**.
Real-robot integrations and deployment guides will be released separately.

## Inspect integration options

```bash
uni-robot-agent run --integration robouse --help
uni-robot-agent run --integration robodojo --help
```

See [getting started](getting-started.md) for installation and a first offline
configuration check. The CLI is `uni-robot-agent`; the Python package is `uni_robot_agent`.

## Prepared episodes and shared tools

`--env-url` points to an already prepared episode, including its full allocated
path when using a pool. The harness checks optional expected identity fields;
it does not use them to create or reset a simulator. The episode's metadata
provides its task instruction, arms, cameras, calibration and action binding.
`--instruction` overrides the instruction for the agent and actor.

| Tool family | Interface |
|---|---|
| Scene | `view_env_state` reads recorded RGB/state; `capture_scene` freezes RGB/state and available calibration and paired depth; `locate` queries the immutable snapshot using the advertised capabilities |
| Direct motion | `move(targets, profile="standard")`, `grasp`, `release`, `hold` |
| Actor | `actor_act` runs bounded chunks with fresh observations and optional reviewer decisions |
| Guidance | `list_skills()` lists shared general/ and current-environment Markdown paths recursively; `read_skill(path=...)` reads a file's content and SHA256; `--skill-writing` adds `write_skill` within the episode's namespace and requires an explicit external skills root |
| Completion | `finish` submits termination; the official task outcome stays outside model-visible feedback |

Tool definitions live in [`tools/gripper/`](../uni_robot_agent/tools/gripper/),
[`tools/scene/`](../uni_robot_agent/tools/scene/), [`tools/skills.py`](../uni_robot_agent/tools/skills.py)
and [`tools/episode.py`](../uni_robot_agent/tools/episode.py).

## RoboDojo

`--integration robodojo` adds the RoboDojo prompt and optional task-requirements
loader to the shared prepared-episode lifecycle. `--task` and `--run-id` identify
the expected episode. `--include-extra-task-requirements` loads the repository's
task requirements; it is enabled by default for RoboDojo and loads the simplified
`resources/prompts/robodojo/instruction_supplements.json` catalog. These supplements
clarify the task instruction, including the required sequence.
The full `task_requirements.json` remains a separate reference resource.

### Realistic observation settings

The paper's **Realistic** principle excludes simulator-provided depth and
task-verifier feedback from agent/tool interaction. RGB images, proprioception,
camera calibration, robot descriptions and task instructions remain available.
Static task requirements describe the task; they are separate from a verifier's
live assessment of progress or success. `finish` acknowledges submission without
returning the official outcome to the agent. Outcomes are saved for evaluation.

RoboDojo captures depth by default. For the paper's RGB-only input setting,
pass `--no-depth` when starting the environment through `run --deploy local`,
`deploy env` or the native environment server. This retains RGB and camera
calibration. With at least two calibrated RGB-only cameras, `locate` supports
multi-view triangulation. When connecting with `--env-url`, configure these
capabilities on the existing environment service before starting the agent;
the agent connection does not disable service-side depth acquisition.

The harness also supports optional depth localization and physical RGB-D sensors.
These capabilities do not establish that a run uses the paper's RGB-only setting.
See [runtime defaults](running.md#single-episode-cli) for capability selection.

### Camera-calibration ablation

For RoboDojo, add `--no-camera-calibration` at environment startup to disable
camera intrinsics and extrinsics as well; this option requires `--no-depth`.
Use `run --deploy local`, `deploy env`, or the native RoboDojo server. The environment
declares `has_calibration=false`. RoboUse and RoboLab do not support this switch;
explicit calibration overrides are rejected for those integrations and for actor
or SAM3 deployment. In `run --env-url` mode, configure the external service at
startup instead; the agent connection cannot change its camera capabilities.
Captures and action records contain no camera matrices or depth data; image paths,
sizes, robot proprioception and robot geometry remain available. `locate` exposes
only 2D SAM segmentation, or is absent when SAM is unavailable. Triangulation,
pixel-to-world localization and plane fitting are not exposed.
Use matching agent and service versions; older clients may reject the added
camera capability field.

### Hybrid control

Hybrid requires `--action-model-url` and an expected
`--action-model-checkpoint-sha256`. Its reviewer is enabled by default and uses
`--vla-review-base-url` or the inherited `--provider-url`. Reviewer model, adapter
and credentials are independently configurable; see
[reviewer configuration](running.md#vla-reviewer). `--no-vla-review` leaves the
[actor execution budgets](running.md#actor-execution-budgets) unchanged.

### Coordinated arm targets

`move`, `grasp` and `release` accept one or two distinct arm targets.
RoboDojo applies them together in each native control step, including in serial
execution mode. Unlisted arms hold; an arm reaching tolerance holds its achieved
pose while the other continues. A fault on either arm stops the call.

List order does not set execution order. Separate tool calls remain separate
actions even when emitted together. This coordinates target tracking, without
planning collision-free paths or maintaining a rigid relative pose between hands.

`move` requires absolute world [x,y,z,roll,pitch,yaw] in metres and fixed-axis
XYZ radians (R=Rz(yaw)@Ry(pitch)@Rx(roll)). State, primitive poses and Hybrid
previews keep wxyz quaternions at the same site. Feedback adds six-value
`target_pose`/`actual_pose` alongside existing position/quaternion fields.
Retain measured XYZ for an in-place turn. Translation and rotation interpolate
together; dependent stages need separate calls and completed feedback checks. Each integration
registers `control_frame`, `grasp_site_description` and `rotation_radius_m` on its
arm descriptors; shared tools interpolate poses without loading a robot model.
RoboDojo merges a fixed-site URDF extension into the native X5 chain for FK/DLS,
and converts site targets once for native link6 IK. RoboUse supplies backend-specific
grasp-site and controller conversions.
The site is a geometric convention, not object attachment evidence.

Environment protocol 3 and tool schema 7 define this interface. Upgrade services
and clients together. When replaying saved commands, verify that their recorded
coordinate conventions and schema match the target environment.

### Active gripper guidance

The [RoboDojo index](../uni_robot_agent/resources/skills/robodojo/index.md)
links to the [shared skill index](../uni_robot_agent/resources/skills/general/index.md)
for gripper control, handover, pose targeting, contact, placement, verification
and recovery. It adds X5 and environment-specific supplements. The agent selects
references from the observed situation and available capabilities.

Start with [the Direct example](../examples/README.md#robodojo). For actor-backed
control, use [the Hybrid example](../examples/robodojo_hybrid.sh) and
[environment service scripts](../scripts/deploy/services/robodojo/).
[RoboDojo SOURCE.md](../uni_robot_agent/integrations/robodojo/SOURCE.md) records pinned
upstream revisions and retained licenses. The official environment server requires
`--native-root` pointing to a clean pinned official checkout, including its submodules.
External runtime trees are not packaged.

## RoboUse

`--integration robouse` composes the shared scene, gripper, skill and episode tools
for RoboUse tasks, including LIBERO and RoboCasa. It supports Direct control;
there is no registered Hybrid actor binding. Each native backend keeps its own
dependencies, task admission, step budget, scoring and camera configuration.

Use `setup --integration robouse --runtime-profile PROFILE` to prepare a native
runtime, then `run --integration robouse --deploy local --task TASK` with the
model and provider settings. For an unreleased build, pass your matching runtime wheel to
`setup --wheel`; a published installation can fetch its matching version from the
package index. See [deployment](deployment.md).
See [deployment profiles](deployment.md#environment-profiles-and-limits),
[RoboUse scripts](../scripts/robouse/README.md) and the
[implementation](../uni_robot_agent/integrations/robouse/).

## RoboLab

`--integration robolab` composes the shared scene, gripper, skill and finish providers
for a prepared single-DROID episode. See [RoboLab setup](robolab.md) for runtime
requirements, coordinate frames and supported features. The integration uses RGB
and calibrated multi-view triangulation, with no actor or SAM3 binding.

## Cobot Magic twin

`--integration cobot_magic` drives the steward project's MuJoCo twin of a dual
Piper Cobot Magic over the shared gripper, scene, skill and finish providers.
The twin runs as a standalone episode service; the agent connects with
`--env-url`. Start it from a checkout with the steward package installed in the
same interpreter:

```bash
python -m uni_robot_agent.integrations.cobot_magic.environment.server \
  --root /path/to/steward/var/steward --scenario cobot-table \
  --task bowl-on-plate --instruction "Put the bowl on the plate." \
  --output-dir ./runs/twin --run-id twin-1 --port 18770
```

One service owns one episode: restart it for the next run. `--task` names the
final check as `<upper>-on-<lower>` object categories, resolved from the twin's
world relations, and is never shown to the agent. `--no-depth` keeps calibrated
RGB; `--no-camera-calibration` requires `--no-depth`.

Arms expose `left` and `right` with world-metre wxyz grasp-site targets, six
joints and a normalized gripper opening. Each 25 Hz step solves IK against the
twin's own model, commands joint targets and advances one control period.
Measured pose error is folded back into the target so a loaded arm settles on
the requested site, matching the twin's own integral compensation.

Cameras are `head`, `left_wrist` and `right_wrist`, each returning RGB, metric
depth and same-frame `intrinsic_K`/`T_world_from_optical`. Optical axes follow
this harness's convention (+x right, +y down, +z forward); the twin's mount
frame (+x forward, +y left, +z up) is converted at capture. In the bundled
`cobot-table` scene the wrist cameras sit inside the gripper and return a
near-black image; the head camera is the useful view.

## Model bindings and backends

RoboDojo uses `ActorClient`, `ActorService` and `Pi05Backend`.
Its `action_binding.py` translates snapshots and returned action chunks,
preserving camera orientation, state layout and gripper conventions.

| Environment | Binding | Actor profile | Native runtime |
|---|---|---|---|
| RoboDojo | `robodojo_pi05` | `robodojo_pi05` | `openpi_jax` |

A compatible checkpoint change is deployment configuration. A new observation
or action convention requires an explicit binding/profile.
See [service profiles](services.md#pi05-deployment-profiles).

## Integration contract

Expose `get_integration_spec()` from an integration module. The
[`IntegrationSpec` and `IntegrationSession` contracts](../uni_robot_agent/integrations/base.py)
keep the runtime independent of a simulator or transport:

1. **Configuration:** declare CLI arguments, validation, typed `config_model`
   options and tools available in task runs. Declare `deployment_module` only
   when automatic preparation/deployment is supported.
2. **Lifecycle:** return a session from `create_session(IntegrationContext)`
   implementing `start`, `finalize` and `close`.
3. **Tools:** provide `tool_providers`, `episode`, `task_succeeded()` and
   `action_model_name`. Return structured `ToolOutput` values.
4. **Scheduling:** declare resource policies for physical operations and
   lifecycle barriers; transport order does not establish exclusion.
5. **Outcomes:** return `EpisodeOutcome` with acceptance and failure
   classification. Retryable infrastructure failures are not accepted evaluation
   results. Set `allow_automatic_retry=False` when replay is unsuitable.

Register environment prompt builders explicitly; otherwise the generic builder
is used. Keep heavy imports in service processes or behind startup so CLI help
and offline validation work without native runtimes.

## Integration layout

```text
integrations/<name>/
├── __init__.py             # integration registration
├── prompt.py               # environment-specific context
├── action_binding.py       # model observation/action conversion, when needed
└── environment/
    ├── adapter.py          # EnvironmentBackend implementation
    ├── native_compat.py    # pinned runtime compatibility checks
    ├── server.py           # prepare episode and mount shared routes
    └── ...                 # native geometry, controllers and kinematics
```

RoboDojo and RoboUse keep native adapters in `environment/adapter.py`.
Shared tools and HTTP protocols stay in `tools/` and
`services/`.

### Prepared episode composition

Reuse `PreparedEpisodeSession(context, configure_tools=...)` for URL-backed
episodes. Its callback creates providers against `session.episode` and registers
auxiliary clients with `session.own()`. The session closes them even if later
setup fails, then releases its environment client. The outer session owns
`IntegrationContext.artifacts`; integrations use that store without closing it
or creating their own image copies.

Observation providers return `ObservationData.image_paths` pointing to ready
files in the shared store. Publish image bytes through the store at the provider
boundary, then pass `local_path(artifact_id)` to `ObservationHub`. Runtime paths are absolute;
persisted observation references are relative to the session directory.

RoboUse and RoboDojo reuse `integrations.gripper.create_session`; RoboDojo also
supplies a task-requirements loader. Another tool family can use a different callback
without adding action-name dispatch to the runtime or duplicating episode state.

See [architecture](architecture.md) for module ownership and
[services](services.md#add-an-actor-backend) for backend extensions.
