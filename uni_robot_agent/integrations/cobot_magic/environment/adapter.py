"""WSim twin backend: world poses in, primitive steps and calibrated frames out."""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from uni_robot_agent.services.environment.contracts import (
    ActionCommand,
    ArmCommand,
    ArmDescription,
    ArmState,
    CameraDescription,
    EnvironmentDescription,
    EnvironmentState,
    JointCommand,
    JointTarget,
    NativeCommand,
    Pose,
    PoseCommand,
    RequestRejected,
    StepReceipt,
)
from uni_robot_agent.services.environment.snapshots import CameraFrame, CapturedObservation

ARMS = ("left", "right")
PREFIX = {"left": "L", "right": "R"}
# Mounted sensor ids in the twin; the first is the scene camera.
CAMERAS = {
    "head": "top_camera",
    "left_wrist": "L_wrist_camera",
    "right_wrist": "R_wrist_camera",
}
CONTROL_HZ = 25.0
JOINT_TOL = 0.03
HOLD_TOL = 0.005
MIN_TICKS = 8
SETTLE_TICKS = 400
_APERTURE_M = 0.10
# Optical axes in the mount frame the twin publishes (+x forward, +y left, +z up).
_R_MOUNT_FROM_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def _require_steward():
    try:
        import steward  # noqa: F401
    except ImportError as error:  # pragma: no cover - depends on the service interpreter
        raise RuntimeError(
            "The cobot_magic twin backend needs the steward package in this interpreter; "
            "install it with `pip install -e /path/to/steward/steward`."
        ) from error


class Adapter:
    """Drive one WSim Cobot Magic twin through uni-robot-agent's primitive protocol."""

    def __init__(self, sim, robot: str, *, episode_id: str, task: str, instruction: str,
                 max_steps: int = 2500, depth_enabled: bool = True,
                 calibration_enabled: bool = True, max_tilt: float = 1.0) -> None:
        _require_steward()
        from steward.assets.cobot import TCP_OFFSET
        from steward.ego.ik import CobotIK
        from steward.world.sim import WorldSim  # noqa: F401 - validates the service interpreter

        if depth_enabled and not calibration_enabled:
            raise ValueError("Disabling camera calibration requires depth_enabled=False")
        self.sim, self.robot = sim, robot
        self.episode_id, self.task, self.instruction = episode_id, task, instruction
        self.max_steps = max_steps
        self.depth_enabled, self.calibration_enabled = depth_enabled, calibration_enabled
        self.max_tilt = float(max_tilt)
        self._tcp_offset = np.asarray(TCP_OFFSET, dtype=float)
        self._ik = CobotIK()
        self._steps = 0
        self._ended = self._sealed = self._closed = False
        self.stop_reason: str | None = None
        self.finish_request: dict[str, Any] | None = None
        self._gripper_commands = {"left": 1.0, "right": 1.0}
        self._substeps = max(1, round(1.0 / (CONTROL_HZ * float(sim.model.opt.timestep))))
        self.description = EnvironmentDescription(
            arms=[ArmDescription(
                name=arm, control_frame="gripper_grasp_site",
                grasp_site_description=(
                    "Pinch point between the jaws, world metres and wxyz; "
                    "z follows the approach axis"
                ),
                rotation_radius_m=float(np.linalg.norm(self._tcp_offset)),
                joint_names=[f"{PREFIX[arm]}_joint{i}" for i in range(1, 7)],
                joint_tracking_tolerance_rad=0.05,
            ) for arm in ARMS],
            cameras=[CameraDescription(
                name=name, role="scene" if name == "head" else "wrist",
                has_depth=depth_enabled, has_calibration=calibration_enabled,
            ) for name in CAMERAS],
            control_frequency_hz=CONTROL_HZ, max_episode_steps=max_steps,
            metadata={
                "workspace_bounds": [[-1.5, -1.5, 0.0], [1.5, 1.5, 1.5]],
                "skill_namespace": "cobot_magic",
                "robot_manual": "cobot_magic/robot-reference.md",
            },
        )

    # ---------------------------------------------------------------- state
    @property
    def steps(self) -> int:
        return self._steps

    @property
    def ended(self) -> bool:
        return self._ended or self._steps >= self.max_steps

    def read_state(self) -> EnvironmentState:
        sensors = self.sim.sensors(self.robot)
        joints = sensors.get("joints") or {}
        odometry = sensors.get("odometry") or {}
        grippers = sensors.get("grippers") or {}
        lift = joints.get("lift")
        arms: dict[str, ArmState] = {}
        for arm in ARMS:
            prefix = PREFIX[arm]
            values = [float(joints.get(f"{prefix}_joint{i}", 0.0)) for i in range(1, 7)]
            self._ik.set_base(float(odometry.get("x", 0.0)), float(odometry.get("y", 0.0)),
                              float(odometry.get("yaw", 0.0)), lift)
            # Servos holding against a stop report a fraction of a radian past the limit, and the
            # solver's limit constraint reads the whole configuration: one such value in the other
            # arm makes every later solve infeasible. Seed through the clamping setter.
            self._ik.set_joints(prefix, values)
            link6 = self._ik.fk(prefix)
            position, quaternion = self._tcp_pose(link6)
            opening = self._opening(joints, prefix)
            held = bool((grippers.get(f"{prefix}_link6") or {}).get("holding"))
            arms[arm] = ArmState(
                control_pose=Pose(position=position, quaternion=quaternion),
                joints=values,
                gripper=max(0.0, min(1.0, self._gripper_commands[arm] if held else opening)),
                gripper_command=self._gripper_commands[arm],
                metadata={
                    "gripper_opening_m": opening * _APERTURE_M,
                    "gripper_holding": held,
                    "gripper_state": (grippers.get(f"{prefix}_link6") or {}).get("state", "open"),
                },
            )
        return EnvironmentState(
            arms=arms, state_version=self._steps, native_steps=self._steps,
            ended=self.ended,
            metadata={
                "instruction": self.instruction,
                "remaining_steps": max(0, self.max_steps - self._steps),
            },
        )

    @staticmethod
    def _opening(joints: dict[str, float], prefix: str) -> float:
        left = float(joints.get(f"{prefix}_joint7", 0.0))
        right = float(joints.get(f"{prefix}_joint8", 0.0))
        return max(0.0, min(1.0, (left - right) / _APERTURE_M))

    def _tcp_pose(self, link6: dict[str, Any]) -> tuple[list[float], list[float]]:
        from steward.math3d import matrix_from_quat

        quaternion = [float(v) for v in link6["q"]]
        position = np.asarray(link6["p"], dtype=float) + matrix_from_quat(quaternion) @ self._tcp_offset
        return [float(v) for v in position], quaternion

    def _wrist_target(self, pinch_position, tool_quaternion):
        """Wrist pose whose grasp site, ``TCP_OFFSET`` along the tool axis, is the requested site."""
        from steward.math3d import matrix_from_quat

        quaternion = [float(v) for v in tool_quaternion]
        position = np.asarray(pinch_position, dtype=float) - matrix_from_quat(quaternion) @ self._tcp_offset
        return [float(v) for v in position], quaternion

    # -------------------------------------------------------------- commands
    def _active(self) -> None:
        if self._sealed or self.ended:
            raise RequestRejected("Native episode is not active")

    @staticmethod
    def _complete(command) -> None:
        if set(command.arms) != set(ARMS):
            raise RequestRejected("A control step must contain both complete arms")

    def _joint_limits(self, arm: str) -> list[tuple[float, float]]:
        import mujoco

        model = self.sim.model
        build = getattr(self.sim, "build", None)
        entity = (getattr(build, "names", None) or {}).get(self.robot)
        # MuJoCo names every joint of an entity `w:<robot>::<joint>`; looking up the bare
        # joint id finds nothing (mj_name2id returns -1) and silently reads the last
        # joint's range, which rejects every target. Resolve through the build's name map.
        known = entity.joints if entity is not None else {}
        limits = []
        for index in range(1, 7):
            suffix = f"{PREFIX[arm]}_joint{index}"
            name = known.get(suffix) or suffix
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint < 0:
                for candidate in range(model.njnt):
                    full = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, candidate) or ""
                    if full == suffix or full.endswith(f"::{suffix}"):
                        joint = candidate
                        break
            if joint < 0:
                raise RequestRejected(f"{arm} joint {index} is not in the twin model")
            low, high = model.jnt_range[joint]
            limits.append((float(low), float(high)))
        return limits

    def _check_joints(self, arm: str, values) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.shape != (6,) or not np.isfinite(array).all():
            raise RequestRejected(f"Invalid {arm} joint target")
        limits = np.asarray(self._joint_limits(arm), dtype=float)
        if np.any(array < limits[:, 0] - 1e-5) or np.any(array > limits[:, 1] + 1e-5):
            raise RequestRejected(f"{arm} joint target exceeds limits")
        return array

    def validate_commands(self, commands: list[ActionCommand], mode: str) -> None:
        self._active()
        state = self.read_state()
        previous = {arm: np.asarray(state.arms[arm].joints) for arm in ARMS}
        for command in commands:
            if isinstance(command, NativeCommand):
                raise RequestRejected("cobot_magic does not accept opaque native actions")
            self._complete(command)
            for arm, target in command.arms.items():
                if isinstance(target, JointTarget):
                    values = self._check_joints(arm, target.joints)
                    if np.max(np.abs(values - previous[arm])) > 0.4:
                        raise RequestRejected(f"{arm} action prefix has a joint jump over 0.4 rad")
                    previous[arm] = values
                elif mode == "eef":
                    pose = state.arms[arm].control_pose
                    displacement = np.linalg.norm(
                        np.asarray(target.pose.position) - np.asarray(pose.position))
                    rotation = _quaternion_distance(target.pose.quaternion, pose.quaternion)
                    if displacement > 0.06 or rotation > 0.4:
                        raise RequestRejected(
                            "EEF target exceeds 6 cm or 0.4 rad from the measured grasp site")

    def step_pose(self, command: PoseCommand) -> StepReceipt:
        """Drive both grasp sites to the requested poses and hold them until the arms settle.

        One step is one increment: the motion executor walks a longer move in ``max_step_m``
        increments and rejects an active arm that travels more than 0.2 m inside one step, so
        covering the whole path here would trip its guard. What this step adds is the settle —
        the twin's servos need tens of ticks to close on a new joint target, and a single-tick
        step would report a pose the arm never reached.
        """
        self._active()
        self._complete(command)
        state = self.read_state()
        action: dict[str, dict[str, Any]] = {}
        solved: dict[str, Any] = {}
        tilts: dict[str, float] = {}
        for arm, target in command.arms.items():
            if isinstance(target, JointTarget):
                action[arm] = {"joints": self._check_joints(arm, target.joints).tolist(),
                               "gripper": target.gripper}
                continue
            assert isinstance(target, ArmCommand)
            pinch = np.asarray(target.pose.position, dtype=float)
            quaternion = [float(v) for v in target.pose.quaternion]
            pose = state.arms[arm].control_pose
            if (np.linalg.norm(pinch - np.asarray(pose.position, dtype=float)) <= HOLD_TOL
                    and _quaternion_distance(quaternion, pose.quaternion) <= HOLD_TOL):
                # Holding an arm where it is must not depend on the solver: the agent holds one
                # arm still through every step of the other, and a re-solve of an unchanged pose is
                # where "no solution toward a pose the arm is already in" comes from.
                action[arm] = {"joints": list(state.arms[arm].joints), "gripper": target.gripper}
                continue
            # Position servos settle with a steady-state error under load. Command the
            # residual the arm is holding, as an arm's integral term would.
            measured = np.asarray(pose.position, dtype=float)
            residual = np.clip(pinch - measured, -0.02, 0.02)
            position, quaternion = self._wrist_target(pinch + residual, quaternion)
            try:
                sensors = self.sim.sensors(self.robot)
                odometry = sensors.get("odometry") or {}
                joints = sensors.get("joints") or {}
                current = [float(joints[f"{PREFIX[arm]}_joint{i}"]) for i in range(1, 7)]
                self._ik.set_base(float(odometry.get("x", 0.0)), float(odometry.get("y", 0.0)),
                                  float(odometry.get("yaw", 0.0)), joints.get("lift"))
                self._ik.set_joints(PREFIX[arm], current)
                current = self._ik.current(PREFIX[arm])
                # The Piper wrist cannot hold a strictly vertical tool over most of the table:
                # joint5 stops at +-70 deg, so a straight-down command misses by centimetres.
                # solve_tool keeps the grasp site fixed while the tool tilts, which is how the
                # twin's own skills reach the table.
                values, tilt = self._ik.solve_tool(
                    PREFIX[arm], position, quaternion, max_tilt=self.max_tilt, q_init=current)
            except Exception as error:  # noqa: BLE001 - IK failures are recoverable feedback
                return StepReceipt(accepted=False, state=self.read_state(), diagnostics={
                    "reason": "ik_failed", "arm": arm, "message": str(error),
                })
            action[arm] = {"joints": [float(v) for v in values], "gripper": target.gripper}
            solved[arm] = [float(v) for v in values]
            tilts[arm] = float(tilt)
        self._apply(action)
        self._settle(action)
        return StepReceipt(accepted=True, state=self.read_state(), diagnostics={
            "commanded_joints": solved,
            "tool_tilt_rad": tilts,
        })

    def _settle(self, action: dict[str, dict[str, Any]]) -> None:
        """Tick until every commanded joint is within ``JOINT_TOL``, or the budget runs out."""
        targets = {arm: np.asarray(action[arm]["joints"], dtype=float) for arm in action}
        compensated: set[str] = set()
        for tick in range(SETTLE_TICKS):
            if self.ended:
                return
            joints = (self.sim.sensors(self.robot).get("joints") or {})
            error = 0.0
            for arm, target in targets.items():
                now = np.asarray([float(joints.get(f"{PREFIX[arm]}_joint{i}", 0.0))
                                  for i in range(1, 7)], dtype=float)
                error = max(error, float(np.max(np.abs(now - target))))
            # Contacts and the gripper need sim time even when the joints already match: a step
            # that only commands a closure must still let the fingers travel.
            if error <= JOINT_TOL and tick >= MIN_TICKS:
                return
            if 0 < tick and error < 0.1 and any(arm not in compensated for arm in targets):
                # Integral action once per settle, the way a real servo removes a load offset.
                for arm, target in targets.items():
                    if arm in compensated:
                        continue
                    now = np.asarray([float(joints.get(f"{PREFIX[arm]}_joint{i}", 0.0))
                                      for i in range(1, 7)], dtype=float)
                    targets[arm] = target + (target - now)
                    compensated.add(arm)
                self.sim.command(self.robot, {"joints": {
                    f"{PREFIX[arm]}_joint{i}": float(value)
                    for arm, target in targets.items()
                    for i, value in enumerate(target, 1)}})
            self.sim.step(self._substeps)
            self._steps += 1

    def step_joints(self, command: JointCommand) -> StepReceipt:
        self._active()
        self._complete(command)
        action = {}
        for arm, target in command.arms.items():
            self._check_joints(arm, target.joints)
            action[arm] = {"joints": [float(v) for v in target.joints], "gripper": target.gripper}
        self._apply(action)
        return StepReceipt(accepted=True, state=self.read_state(), diagnostics={
            "commanded_joints": {arm: action[arm]["joints"] for arm in action},
        })

    def step_native(self, command: NativeCommand) -> StepReceipt:
        raise RequestRejected("cobot_magic does not accept opaque native actions")

    def preview_joints(self, commands: list[JointCommand]) -> dict[str, Any]:
        self._active()
        previews = []
        for command in commands:
            self._complete(command)
            row = {}
            for arm in ARMS:
                values = self._check_joints(arm, command.arms[arm].joints)
                row[arm] = self._ik.fk(PREFIX[arm], [float(v) for v in values])["p"]
            previews.append(row)
        return {"positions": previews}

    def _apply(self, action: dict[str, dict[str, Any]]) -> None:
        joints = {}
        for arm, command in action.items():
            for index, value in enumerate(command["joints"], 1):
                joints[f"{PREFIX[arm]}_joint{index}"] = float(value)
        self.sim.command(self.robot, {"joints": joints})
        for arm, command in action.items():
            state = "close" if command["gripper"] < 0.5 else "open"
            self.sim.command(self.robot, {"gripper": {f"{PREFIX[arm]}_link6": state}})
            self._gripper_commands[arm] = float(command["gripper"])
        self.sim.step(self._substeps)
        self._steps += 1
        if self._steps >= self.max_steps:
            self._ended = True

    # ------------------------------------------------------------ observation
    def capture(self, cameras: list[str] | None = None, *,
                high_resolution: bool = False) -> CapturedObservation:
        if self._closed:
            raise RequestRejected("Native episode is not active")
        selected = list(CAMERAS) if cameras is None else cameras
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= CAMERAS.keys():
            raise RequestRejected("cameras must name distinct supported views")
        from steward.ego.perception import decode_rgbd
        from steward.math3d import matrix_from_quat

        frames: dict[str, CameraFrame] = {}
        for name in selected:
            sensor = CAMERAS[name]
            decoded = decode_rgbd(self.sim.rgbd(self.robot, sensor))
            rgb = np.asarray(decoded["rgb"], dtype=np.uint8)
            depth = np.asarray(decoded["depth"], dtype=np.float32) if self.depth_enabled else None
            mount, definition = self.sim.camera_pose(self.robot, sensor)
            intrinsic = extrinsic = None
            calibration: dict[str, Any] = {
                "status": "disabled",
                "reason": "Camera calibration is disabled for this episode",
            }
            if self.calibration_enabled:
                intrinsic = _intrinsics(rgb.shape[:2], definition)
                rotation = matrix_from_quat([float(v) for v in mount["q"]])
                extrinsic = np.eye(4)
                extrinsic[:3, :3] = rotation @ _R_MOUNT_FROM_OPTICAL
                extrinsic[:3, 3] = np.asarray(mount["p"], dtype=float)
                calibration = {"status": "valid", "source": "mujoco_render",
                               "pixel_coordinates": "integer_pixel_centers"}
            frames[name] = CameraFrame(rgb, depth, intrinsic, extrinsic, calibration)
        state = self.read_state()
        truncated = self.ended and self._steps >= self.max_steps and not self._success()
        return CapturedObservation(
            self.episode_id, state.state_version, time.time(), state.model_dump(mode="json"),
            frames, terminated=bool(self._ended and not truncated), truncated=bool(truncated),
        ).freeze()

    # --------------------------------------------------------------- outcome
    def _success(self) -> bool:
        """Task verdict from exact world relations; never exposed to the agent."""
        if self.task == "drawer-open":
            return self._drawer_open()
        relations = (self.sim.world_state() or {}).get("relations") or []
        upper, _, lower = self.task.partition("-on-")
        upper_id, lower_id = self._entity(upper), self._entity(lower or "")
        if upper_id is None or lower_id is None:
            return False
        return any(
            relation.get("type") == "supported-by"
            and (relation.get("a") or {}).get("entity") == upper_id
            and (relation.get("b") or {}).get("entity") == lower_id
            for relation in relations
        )

    def _drawer_open(self) -> bool:
        """The drawer's own slide joint has travelled at least 10 cm (the twin's test bar)."""
        for entity in (self.sim.world_state() or {}).get("entities") or []:
            for joint in (entity.get("body") or {}).get("joints") or []:
                if joint.get("id") != "drawer_slide":
                    continue
                value = (joint.get("q") or {}).get("value") or []
                if value and float(value[0]) >= 0.10:
                    return True
        return False

    def _entity(self, category: str) -> str | None:
        if not category:
            return None
        for entity in (self.sim.world_state() or {}).get("entities") or []:
            value = (entity.get("category") or {}).get("value")
            if value == category and entity.get("role") == "object":
                return entity.get("id")
        return None

    def outcome(self) -> dict[str, Any]:
        complete = self._ended
        truncated = complete and self._steps >= self.max_steps and not self._success()
        return {
            "task_success": bool(self._success()) if complete else None,
            "complete": complete,
            "terminated": bool(complete and not truncated),
            "truncated": bool(truncated),
            "steps": self._steps,
            "step_limit": self.max_steps,
            "finish_request": self.finish_request,
            "stop_reason": self.stop_reason or ("native_terminal" if complete else None),
        }

    def finish(self, reason: str, status: str | None = None) -> dict[str, Any]:
        if not self._sealed:
            self._sealed = True
            self.stop_reason = reason
            self.finish_request = {"reason": reason, "status": status}
            if status is not None or reason in {"give_up", "max_turns", "wall_time_limit"}:
                self._ended = True
        return self.outcome()

    def close(self) -> None:
        if not self._closed:
            self._closed = True


def _intrinsics(shape: tuple[int, int], definition: dict[str, Any]) -> np.ndarray:
    height, width = shape
    fovy = float((definition.get("intrinsics") or {}).get("fovy", 60.0))
    focal = (height / 2.0) / math.tan(math.radians(fovy) / 2.0)
    return np.array([
        [focal, 0.0, (width - 1) / 2.0],
        [0.0, focal, (height - 1) / 2.0],
        [0.0, 0.0, 1.0],
    ])


def _quaternion_distance(left, right) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    return 2.0 * math.acos(float(np.clip(abs(np.dot(left, right)), 0.0, 1.0)))
