"""Single-episode twin service. Every MuJoCo call runs on this process's main thread."""

from __future__ import annotations

import argparse
import os
import threading
import traceback
from concurrent.futures import Future
from pathlib import Path
from queue import Empty, Queue

INTEGRATION = "cobot_magic"


class _OwnerThreadBackend:
    """Marshal twin operations onto the owner thread while request workers wait."""

    def __init__(self, backend):
        self.backend, self.description = backend, backend.description
        self._owner = threading.get_ident()
        self._queue = Queue()

    def _call(self, operation, *args, **kwargs):
        if threading.get_ident() == self._owner:
            return operation(*args, **kwargs)
        result = Future()
        self._queue.put((operation, args, kwargs, result))
        return result.result()

    def pump(self, timeout=.1):
        if threading.get_ident() != self._owner:
            raise RuntimeError("Only the owner thread can drain its queue")
        try:
            operation, args, kwargs, result = self._queue.get(timeout=timeout)
        except Empty:
            return
        try:
            result.set_result(operation(*args, **kwargs))
        except BaseException as error:  # noqa: BLE001 - preserve native failures for the caller
            result.set_exception(error)

    def read_state(self):
        return self._call(self.backend.read_state)

    def step_pose(self, command):
        return self._call(self.backend.step_pose, command)

    def step_joints(self, command):
        return self._call(self.backend.step_joints, command)

    def step_native(self, command):
        return self._call(self.backend.step_native, command)

    def validate_commands(self, commands, mode):
        return self._call(self.backend.validate_commands, commands, mode)

    def preview_joints(self, commands):
        return self._call(self.backend.preview_joints, commands)

    def capture(self, cameras=None, *, high_resolution=False):
        return self._call(self.backend.capture, cameras, high_resolution=high_resolution)

    def finish(self, reason, status=None):
        return self._call(self.backend.finish, reason, status)

    def close(self):
        return self._call(self.backend.close)


def _serve(adapter, args):
    from uni_robot_agent.artifacts.store import ArtifactStore
    from uni_robot_agent.services.environment.contracts import EpisodeInfo
    from uni_robot_agent.services.environment.routes import build_router
    from uni_robot_agent.services.environment.service import EnvironmentService
    from uni_robot_agent.services.hosting import HttpServer, create_app

    owner = _OwnerThreadBackend(adapter)
    info = EpisodeInfo(
        episode_id=args.run_id, task=args.task, environment=owner.description,
        integration=INTEGRATION, capabilities=["motion", "capture"],
        metadata={"task_id": args.task, "seed": 0, "scenario": args.scenario,
                  "instruction": args.instruction, "skill_namespace": INTEGRATION},
    )
    service = EnvironmentService(owner, info, ArtifactStore(args.output_dir))
    try:
        service.initialize()
        server = HttpServer(create_app(build_router(service)), host=args.host, port=args.port)
    except BaseException:
        service.close(reason="initialization_failed")
        raise
    host_errors: list[BaseException] = []

    def run_host():
        try:
            server.start()
        except BaseException as error:  # noqa: BLE001 - keep the thread error for the main loop
            host_errors.append(error)

    host = threading.Thread(target=run_host, name="episode-http", daemon=True)
    host.start()
    try:
        while host.is_alive():
            owner.pump()
    finally:
        server.request_stop()
        while host.is_alive():
            owner.pump()
        host.join()
        closed = Future()

        def close_service():
            try:
                service.close()
                closed.set_result(None)
            except BaseException as error:  # noqa: BLE001 - shutdown must surface native failures
                closed.set_exception(error)

        closer = threading.Thread(target=close_service, name="episode-shutdown")
        closer.start()
        while closer.is_alive():
            owner.pump()
        closer.join()
        closed.result()
    if host_errors:
        raise RuntimeError("Episode HTTP host failed") from host_errors[0]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="uni-robot-agent Cobot Magic twin episode")
    parser.add_argument("--root", type=Path, default=None,
                        help="Steward data root holding assets/, scenes/ and scenarios/ "
                             "(default: STEWARD_TWIN_ROOT)")
    parser.add_argument("--scenario", default="cobot-table")
    parser.add_argument("--task", default="bowl-on-plate",
                        help="'<upper>-on-<lower>' category pair used by the final check")
    parser.add_argument("--instruction", default="Put the bowl on the plate.")
    parser.add_argument("--max-steps", type=int, default=2500, help="25 Hz control steps")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--no-depth", action="store_true",
                        help="Capture calibrated RGB only; never return depth")
    parser.add_argument("--no-camera-calibration", action="store_true",
                        help="With --no-depth, capture RGB without intrinsics or extrinsics")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18770)
    args = parser.parse_args(argv)
    if args.no_camera_calibration and not args.no_depth:
        parser.error("--no-camera-calibration requires --no-depth")
    root = args.root or (Path(os.environ["STEWARD_TWIN_ROOT"])
                         if os.environ.get("STEWARD_TWIN_ROOT") else None)
    if root is None:
        parser.error("--root or STEWARD_TWIN_ROOT is required")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    args.root = root.expanduser().resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from uni_robot_agent.integrations.cobot_magic.environment.adapter import Adapter

    sim = adapter = None
    try:
        from steward.authoring.service import AuthoringService
        from steward.world.sim import WorldSim

        service = AuthoringService(args.root)
        service.ensure_scenario(args.scenario)
        scenario = service.load_scenario(args.scenario)
        sim = WorldSim(scenario["sceneState"], speed=args.speed)
        adapter = Adapter(
            sim, scenario["robots"][0], episode_id=args.run_id, task=args.task,
            instruction=args.instruction, max_steps=args.max_steps,
            depth_enabled=not args.no_depth,
            calibration_enabled=not args.no_camera_calibration,
        )
        _serve(adapter, args)
    except BaseException:
        (args.output_dir / "server_error.txt").write_text(traceback.format_exc())
        raise
    finally:
        if adapter is not None:
            adapter.close()
        if sim is not None:
            sim.stop()


if __name__ == "__main__":
    main()
