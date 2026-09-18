#!/usr/bin/env python3
"""Replay an F1 dual-arm trajectory from a LeRobot v2.1 episode.

The supplied F1 URDF/USD has no gripper joints, so this script deliberately
maps only the seven left-arm and seven right-arm joints.  Gripper columns in
the dataset are reported and ignored.

Isaac Sim's bundled Python does not need PyArrow installed.  Parquet reading
is delegated to a separate Python interpreter (normally the ``.venv`` next
to the converted dataset), then the small numerical trajectory is passed to
Isaac Python through a temporary NumPy archive.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROBOT_PRIM_PATH = "/F1"
LEFT_ARM_JOINTS = tuple(f"arm_L_joint{i}" for i in range(1, 8))
RIGHT_ARM_JOINTS = tuple(f"arm_R_joint{i}" for i in range(1, 8))
ARM_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
BODY_JOINTS = tuple(f"body_joint{i}" for i in range(1, 6))
HEAD_JOINTS = tuple(f"head_joint{i}" for i in range(1, 4))
WHEEL_JOINTS = tuple(
    f"chassis_{side}_wheel_{position}_joint{joint}"
    for side in ("L", "R")
    for position in ("front", "rear")
    for joint in (1, 2)
)

_PARQUET_READER = r"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

parquet_path = Path(sys.argv[1])
field = sys.argv[2]
output_path = Path(sys.argv[3])
table = pq.read_table(parquet_path, columns=[field, "timestamp", "frame_index"])
values = np.asarray(table[field].to_pylist(), dtype=np.float32)
timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
np.savez(output_path, values=values, timestamps=timestamps, frame_indices=frame_indices)
"""


def _repo_root() -> Path:
    override = os.environ.get("GENIESIM_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def _default_robot_usd() -> Path:
    return (
        _repo_root()
        / "output"
        / "f1_wheel"
        / "f1_wheel_0602_new"
        / "f1_wheel_0602_new.usda"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the 14 F1 arm joints from one LeRobot episode; "
            "gripper data is intentionally ignored."
        ),
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Converted LeRobot dataset root containing meta/info.json and data/.",
    )
    parser.add_argument("--episode", type=int, default=0, help="Converted episode index (default: 0).")
    parser.add_argument(
        "--field",
        choices=("action", "observation.state"),
        default="action",
        help="Replay recorded commands (action) or measured arm state (default: action).",
    )
    parser.add_argument(
        "--reader-python",
        type=Path,
        default=None,
        help="Python interpreter containing PyArrow; auto-detected from a nearby .venv by default.",
    )
    parser.add_argument("--robot-usd", type=Path, default=_default_robot_usd())
    parser.add_argument("--start-frame", type=int, default=0, help="First episode row to replay.")
    parser.add_argument("--end-frame", type=int, default=None, help="Exclusive final episode row.")
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth data frame (default: 1).")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit selected data frames; useful for a short validation run.",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (default: 1).")
    parser.add_argument(
        "--interpolation",
        choices=("linear", "hold"),
        default="linear",
        help="Interpolate at physics rate, or hold each recorded command (default: linear).",
    )
    parser.add_argument("--physics-hz", type=float, default=60.0, help="Physics frequency (default: 60).")
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.5,
        help="Settle at the first recorded pose before replay (default: 0.5).",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=1.0,
        help="Hold the final target before exiting (default: 1).",
    )
    parser.add_argument("--headless", action="store_true", help="Run without the Isaac Sim GUI.")
    parser.add_argument("--no-render", action="store_true", help="Skip explicit rendering.")
    parser.add_argument(
        "--with-table",
        action="store_true",
        help="Add the primitive table and cube; default is robot plus ground only.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Read and validate data without starting Isaac Sim.")
    parser.add_argument("--save-stage", type=Path, default=None, help="Optionally save the composed stage.")
    args = parser.parse_args()

    if args.episode < 0:
        parser.error("--episode must be non-negative")
    if args.start_frame < 0:
        parser.error("--start-frame must be non-negative")
    if args.end_frame is not None and args.end_frame <= args.start_frame:
        parser.error("--end-frame must be greater than --start-frame")
    if args.stride < 1:
        parser.error("--stride must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be at least 1")
    if args.speed <= 0:
        parser.error("--speed must be positive")
    if args.physics_hz <= 0:
        parser.error("--physics-hz must be positive")
    if args.settle_seconds < 0 or args.hold_seconds < 0:
        parser.error("--settle-seconds and --hold-seconds must be non-negative")
    return args


def _is_lerobot_dataset(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file()


def _validate_dataset_root(requested_path: Path) -> Path:
    """Validate an explicit converted LeRobot dataset root."""
    requested_path = requested_path.expanduser().resolve()
    if not requested_path.is_dir():
        raise FileNotFoundError(f"Dataset path is not a directory: {requested_path}")
    if not _is_lerobot_dataset(requested_path):
        raise FileNotFoundError(
            f"Not a converted LeRobot dataset root (missing meta/info.json): {requested_path}"
        )
    return requested_path


def _python_has_pyarrow(python: Path) -> bool:
    try:
        result = subprocess.run(
            [str(python), "-c", "import numpy, pyarrow.parquet"],
            env=_reader_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _reader_environment() -> dict[str, str]:
    """Remove Isaac's CPython 3.12 module path before launching another Python."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    return environment


def _absolute_without_resolving_symlinks(path: Path) -> Path:
    # Resolving a venv's bin/python symlink loses the virtual-environment
    # prefix and therefore its site-packages.
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _resolve_reader_python(explicit: Path | None, dataset_root: Path) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(_absolute_without_resolving_symlinks(explicit))
    env_value = os.environ.get("LEROBOT_READER_PYTHON")
    if env_value:
        candidates.append(_absolute_without_resolving_symlinks(Path(env_value)))
    candidates.append(_absolute_without_resolving_symlinks(Path(sys.executable)))
    for base in (dataset_root, *dataset_root.parents):
        candidates.extend((base / ".venv" / "bin" / "python", base / "venv" / "bin" / "python"))

    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    for candidate in unique_candidates:
        if candidate.is_file() and _python_has_pyarrow(candidate):
            return candidate

    checked = "\n".join(f"  - {path}" for path in unique_candidates)
    raise RuntimeError(
        "Could not find a Python interpreter containing NumPy and PyArrow. "
        "Pass --reader-python or set LEROBOT_READER_PYTHON.\nChecked:\n" + checked
    )


def _feature_names(info: dict, field: str) -> list[str]:
    try:
        names = info["features"][field]["names"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Dataset metadata does not describe field {field!r}") from exc
    while isinstance(names, list) and len(names) == 1 and isinstance(names[0], list):
        names = names[0]
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise ValueError(f"Dataset field {field!r} has unusable feature names: {names!r}")
    return names


def _canonical_arm_name(name: str) -> str | None:
    if name in ARM_JOINTS:
        return name
    match = re.fullmatch(r"arm_([lLrR])_(?:j|joint)([1-7])", name)
    if match:
        side = match.group(1).upper()
        return f"arm_{side}_joint{match.group(2)}"
    return None


def _episode_parquet(dataset_root: Path, info: dict, episode: int) -> Path:
    total_episodes = int(info.get("total_episodes", 0))
    if episode >= total_episodes:
        raise IndexError(
            f"Episode {episode} does not exist; this dataset has converted indices "
            f"0..{max(total_episodes - 1, 0)}"
        )
    chunks_size = int(info.get("chunks_size", 1000))
    pattern = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    relative_path = pattern.format(
        episode_chunk=episode // chunks_size,
        episode_index=episode,
    )
    parquet_path = (dataset_root / relative_path).resolve()
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Episode parquet does not exist: {parquet_path}")
    return parquet_path


def _read_parquet(
    reader_python: Path,
    parquet_path: Path,
    field: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with tempfile.TemporaryDirectory(prefix="f1_lerobot_replay_") as temporary_dir:
        archive_path = Path(temporary_dir) / "trajectory.npz"
        result = subprocess.run(
            [str(reader_python), "-c", _PARQUET_READER, str(parquet_path), field, str(archive_path)],
            env=_reader_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown reader error"
            raise RuntimeError(f"Parquet reader failed with exit code {result.returncode}:\n{detail}")
        with np.load(archive_path, allow_pickle=False) as archive:
            return (
                np.asarray(archive["values"], dtype=np.float32).copy(),
                np.asarray(archive["timestamps"], dtype=np.float64).copy(),
                np.asarray(archive["frame_indices"], dtype=np.int64).copy(),
            )


def _select_arm_trajectory(
    values: np.ndarray,
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    names: list[str],
    start_frame: int,
    end_frame: int | None,
    stride: int,
    max_frames: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError(
            f"Trajectory shape {values.shape} does not match {len(names)} metadata names"
        )
    if timestamps.shape != (values.shape[0],) or frame_indices.shape != (values.shape[0],):
        raise ValueError("timestamp/frame_index columns do not match the trajectory length")

    canonical_to_column: dict[str, int] = {}
    ignored: list[str] = []
    for column, source_name in enumerate(names):
        canonical = _canonical_arm_name(source_name)
        if canonical is None:
            ignored.append(source_name)
            continue
        if canonical in canonical_to_column:
            raise ValueError(f"Multiple dataset columns map to F1 joint {canonical}")
        canonical_to_column[canonical] = column
    missing = [name for name in ARM_JOINTS if name not in canonical_to_column]
    if missing:
        raise ValueError("Dataset is missing required F1 arm fields: " + ", ".join(missing))

    stop = min(end_frame if end_frame is not None else values.shape[0], values.shape[0])
    row_indices = np.arange(start_frame, stop, stride, dtype=np.int64)
    if max_frames is not None:
        row_indices = row_indices[:max_frames]
    if row_indices.size == 0:
        raise ValueError(
            f"Frame selection is empty: dataset has {values.shape[0]} rows, "
            f"requested start={start_frame}, end={end_frame}, stride={stride}"
        )

    arm_columns = [canonical_to_column[name] for name in ARM_JOINTS]
    arm_values = values[np.ix_(row_indices, arm_columns)]
    selected_timestamps = timestamps[row_indices]
    selected_frame_indices = frame_indices[row_indices]
    if not np.all(np.isfinite(arm_values)) or not np.all(np.isfinite(selected_timestamps)):
        raise ValueError("Selected trajectory contains NaN or infinite values")
    return arm_values, selected_timestamps, selected_frame_indices, ignored


def _validate_asset_closure(robot_usd: Path) -> Path:
    robot_usd = robot_usd.expanduser().resolve()
    if not robot_usd.is_file():
        raise FileNotFoundError(f"F1 root USD does not exist: {robot_usd}")
    payload_dir = robot_usd.parent / "payloads"
    required = (
        payload_dir / "base.usda",
        payload_dir / "geometries.usd",
        payload_dir / "robot.usda",
        payload_dir / "Physics" / "physics.usda",
        payload_dir / "Physics" / "physx.usda",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "F1 USD package is incomplete:\n  - " + "\n  - ".join(str(path) for path in missing)
        )
    return robot_usd


def _array1d(value: object) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _print_trajectory_summary(
    dataset_root: Path,
    parquet_path: Path,
    reader_python: Path,
    args: argparse.Namespace,
    trajectory: np.ndarray,
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    ignored: list[str],
) -> None:
    if len(timestamps) > 1:
        duration = max(0.0, float(timestamps[-1] - timestamps[0]))
        max_step_deg = float(np.rad2deg(np.max(np.abs(np.diff(trajectory, axis=0)))))
    else:
        duration = 0.0
        max_step_deg = 0.0
    print(f"[f1-replay] Dataset: {dataset_root}")
    print(f"[f1-replay] Episode parquet: {parquet_path}")
    print(f"[f1-replay] Parquet reader: {reader_python}")
    print(f"[f1-replay] Source field: {args.field}")
    print("[f1-replay] Joint mapping: " + ", ".join(ARM_JOINTS))
    if ignored:
        print("[f1-replay] Ignored non-arm fields: " + ", ".join(ignored))
    print(
        f"[f1-replay] Selected {len(trajectory)} frames "
        f"(episode frame_index {int(frame_indices[0])}..{int(frame_indices[-1])}, "
        f"recorded duration {duration:.3f}s, playback {duration / args.speed:.3f}s)"
    )
    print(f"[f1-replay] Maximum selected sample-to-sample arm step: {max_step_deg:.3f} deg")
    if max_step_deg > 20.0:
        print(
            "[f1-replay] WARNING: the selected data contains a joint step above 20 deg; "
            "linear interpolation and drive velocity limits will soften it."
        )


def main() -> int:
    args = _parse_args()
    try:
        dataset_root = _validate_dataset_root(args.dataset)
        info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
        parquet_path = _episode_parquet(dataset_root, info, args.episode)
        names = _feature_names(info, args.field)
        reader_python = _resolve_reader_python(args.reader_python, dataset_root)
        values, timestamps, frame_indices = _read_parquet(reader_python, parquet_path, args.field)
        trajectory, timestamps, frame_indices, ignored = _select_arm_trajectory(
            values,
            timestamps,
            frame_indices,
            names,
            args.start_frame,
            args.end_frame,
            args.stride,
            args.max_frames,
        )
        robot_usd = _validate_asset_closure(args.robot_usd)
    except (FileNotFoundError, IndexError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[f1-replay] ERROR: {exc}", file=sys.stderr)
        return 2

    _print_trajectory_summary(
        dataset_root,
        parquet_path,
        reader_python,
        args,
        trajectory,
        timestamps,
        frame_indices,
        ignored,
    )
    if args.dry_run:
        print("[f1-replay] Dry run complete; Isaac Sim was not started")
        return 0

    # Omniverse/Isaac modules must only be imported after SimulationApp.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless, "hide_ui": args.headless})

    try:
        import isaacsim.core.experimental.utils.app as app_utils
        import isaacsim.core.experimental.utils.stage as stage_utils
        from isaacsim.core.experimental.objects import Cube, DistantLight, GroundPlane
        from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim
        from isaacsim.core.rendering_manager import RenderingManager
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.utils.viewports import set_camera_view

        def add_static_cube(
            path: str,
            position: list[float],
            scale: list[float],
            color: list[float],
        ) -> None:
            shape = Cube(paths=path, positions=position, sizes=1.0, scales=scale, colors=color)
            GeomPrim(paths=shape.paths, apply_collision_apis=True)

        def step_once() -> bool:
            if not simulation_app.is_running():
                return False
            SimulationManager.step()
            if not args.no_render:
                RenderingManager.render()
            simulation_app.update()
            return simulation_app.is_running()

        print(f"[f1-replay] Isaac Python: {sys.executable}")
        print(f"[f1-replay] Robot USD: {robot_usd}")

        stage_utils.create_new_stage()
        stage_utils.set_stage_units(meters_per_unit=1.0)
        GroundPlane("/World/GroundPlane", positions=[0.0, 0.0, 0.0])
        light = DistantLight("/World/DistantLight")
        light.set_intensities(500)

        if args.with_table:
            add_static_cube(
                "/World/Table/Top",
                [0.65, 0.0, 0.72],
                [0.70, 1.00, 0.08],
                [0.36, 0.20, 0.10],
            )
            for name, x, y in (
                ("LegFrontLeft", 0.92, 0.42),
                ("LegFrontRight", 0.92, -0.42),
                ("LegBackLeft", 0.38, 0.42),
                ("LegBackRight", 0.38, -0.42),
            ):
                add_static_cube(
                    f"/World/Table/{name}",
                    [x, y, 0.34],
                    [0.07, 0.07, 0.68],
                    [0.28, 0.14, 0.07],
                )
            cube_shape = Cube(
                paths="/World/GraspCube",
                positions=[0.55, -0.18, 0.805],
                sizes=1.0,
                scales=[0.07, 0.07, 0.07],
                colors=[0.10, 0.65, 0.95],
            )
            RigidPrim(paths=cube_shape.paths)
            GeomPrim(paths=cube_shape.paths, apply_collision_apis=True)

        stage_utils.add_reference_to_stage(usd_path=str(robot_usd), path=ROBOT_PRIM_PATH)
        for _ in range(600):
            if not stage_utils.is_stage_loading():
                break
            simulation_app.update()
        if stage_utils.is_stage_loading():
            raise RuntimeError("Timed out while loading the F1 USD package")

        stage = stage_utils.get_current_stage(backend="usd")
        if not stage.GetPrimAtPath(ROBOT_PRIM_PATH).IsValid():
            raise RuntimeError(f"Robot prim was not composed at {ROBOT_PRIM_PATH}")

        robot = Articulation(ROBOT_PRIM_PATH)
        dof_names = list(robot.dof_names)
        dof_index = {name: index for index, name in enumerate(dof_names)}
        missing_arm_dofs = [name for name in ARM_JOINTS if name not in dof_index]
        if missing_arm_dofs:
            raise RuntimeError("F1 USD is missing arm DOFs: " + ", ".join(missing_arm_dofs))
        arm_indices = [dof_index[name] for name in ARM_JOINTS]
        missing_wheel_dofs = [name for name in WHEEL_JOINTS if name not in dof_index]
        if missing_wheel_dofs:
            raise RuntimeError("F1 USD is missing wheel DOFs: " + ", ".join(missing_wheel_dofs))
        wheel_indices = [dof_index[name] for name in WHEEL_JOINTS]

        # Query the imported USD limits before constructing the initial pose.
        lower_raw, upper_raw = robot.get_dof_limits(dof_indices=arm_indices)
        lower_limits = _array1d(lower_raw)
        upper_limits = _array1d(upper_raw)
        if lower_limits.shape != (len(ARM_JOINTS),) or upper_limits.shape != (len(ARM_JOINTS),):
            raise RuntimeError("Isaac returned unexpected F1 arm-limit shapes")

        violations = np.maximum(
            np.maximum(lower_limits[None, :] - trajectory, trajectory - upper_limits[None, :]),
            0.0,
        )
        violation_mask = violations > 1.0e-6
        if np.any(violation_mask):
            affected = [
                f"{name} ({int(np.count_nonzero(violation_mask[:, column]))} samples)"
                for column, name in enumerate(ARM_JOINTS)
                if np.any(violation_mask[:, column])
            ]
            print(
                f"[f1-replay] LIMIT CLAMP: {int(np.count_nonzero(violation_mask))} values, "
                f"maximum violation {float(np.rad2deg(np.max(violations))):.3f} deg"
            )
            print("[f1-replay] Clamped joints: " + ", ".join(affected))
            trajectory = np.clip(trajectory, lower_limits, upper_limits)
        else:
            print("[f1-replay] All selected arm targets are inside the imported F1 limits")

        initial_positions = np.zeros(len(dof_names), dtype=np.float32)
        initial_positions[arm_indices] = trajectory[0]
        robot.set_default_state(
            positions=[0.0, 0.0, 0.22],
            orientations=[1.0, 0.0, 0.0, 0.0],
            dof_positions=initial_positions,
        )

        moving_joint_names = (*BODY_JOINTS, *HEAD_JOINTS, *ARM_JOINTS)
        moving_indices = [dof_index[name] for name in moving_joint_names if name in dof_index]
        physics_dt = 1.0 / args.physics_hz
        SimulationManager.set_physics_dt(physics_dt)
        set_camera_view(
            eye=[2.2, 2.2, 1.8],
            target=[0.15, 0.0, 0.75],
            camera_prim_path="/OmniverseKit_Persp",
        )

        app_utils.play()
        simulation_app.update()
        robot.reset_to_default_state()

        # Several source-URDF joints declare zero effort/velocity.  Runtime
        # overrides make the imported articulation controllable without
        # modifying the user's source asset.
        robot.set_dof_max_efforts(100.0, dof_indices=moving_indices)
        robot.set_dof_max_velocities(3.14, dof_indices=moving_indices)
        robot.set_dof_gains(stiffnesses=1000.0, dampings=100.0, dof_indices=moving_indices)

        # The URDF declares the eight wheel-group joints as revolute while
        # also assigning lower=upper=effort=velocity=0.  Isaac's importer
        # creates weak angular drives for that contradictory description,
        # and contact can make the unconstrained numerical response explode.
        # This arm-only replay explicitly parks every steering/rolling DOF.
        wheel_zeros = np.zeros(len(wheel_indices), dtype=np.float32)
        robot.set_dof_max_efforts(100.0, dof_indices=wheel_indices)
        robot.set_dof_max_velocities(0.1, dof_indices=wheel_indices)
        robot.set_dof_gains(stiffnesses=1000.0, dampings=100.0, dof_indices=wheel_indices)
        robot.set_dof_positions(wheel_zeros, dof_indices=wheel_indices)
        robot.set_dof_velocities(wheel_zeros, dof_indices=wheel_indices)
        controlled_indices = moving_indices + wheel_indices
        robot.set_dof_position_targets(
            initial_positions[controlled_indices],
            dof_indices=controlled_indices,
        )
        robot.set_dof_velocity_targets(
            np.zeros(len(controlled_indices), dtype=np.float32),
            dof_indices=controlled_indices,
        )

        settle_steps = int(round(args.settle_seconds / physics_dt))
        for _ in range(settle_steps):
            if not step_once():
                break

        print(
            f"[f1-replay] Replaying episode {args.episode} at {args.speed:g}x, "
            f"{args.interpolation} interpolation, {args.physics_hz:g} Hz physics"
        )
        simulated_steps = 0
        interrupted = not simulation_app.is_running()
        fallback_sample_dt = float(info.get("fps", 20))
        fallback_sample_dt = args.stride / fallback_sample_dt if fallback_sample_dt > 0 else 0.05

        for sample in range(max(0, len(trajectory) - 1)):
            segment_dt = float(timestamps[sample + 1] - timestamps[sample])
            if segment_dt <= 0:
                segment_dt = fallback_sample_dt
            segment_steps = max(1, int(round(segment_dt / args.speed / physics_dt)))
            start_target = trajectory[sample]
            end_target = trajectory[sample + 1]
            for substep in range(segment_steps):
                if args.interpolation == "linear":
                    alpha = float(substep + 1) / float(segment_steps)
                    target = start_target + alpha * (end_target - start_target)
                else:
                    target = start_target
                robot.set_dof_position_targets(target, dof_indices=arm_indices)
                if not step_once():
                    interrupted = True
                    break
                simulated_steps += 1
            if interrupted:
                break
            if (sample + 1) % 100 == 0 or sample + 2 == len(trajectory):
                print(f"[f1-replay] Progress: {sample + 2}/{len(trajectory)} data frames")

        if not interrupted:
            robot.set_dof_position_targets(trajectory[-1], dof_indices=arm_indices)
            hold_steps = int(round(args.hold_seconds / physics_dt))
            for _ in range(hold_steps):
                if not step_once():
                    interrupted = True
                    break
                simulated_steps += 1

        final_positions = _array1d(robot.get_dof_positions())
        final_wheel_velocities = _array1d(robot.get_dof_velocities(dof_indices=wheel_indices))
        final_error = final_positions[arm_indices] - trajectory[-1]
        print(
            "[f1-replay] Final arm target error: "
            f"RMSE={float(np.rad2deg(np.sqrt(np.mean(final_error**2)))):.3f} deg, "
            f"max={float(np.rad2deg(np.max(np.abs(final_error)))):.3f} deg"
        )
        print(
            f"[f1-replay] {'Interrupted' if interrupted else 'Completed'} "
            f"{len(trajectory)} selected frames in {simulated_steps} replay physics steps"
        )
        print(
            "[f1-replay] Maximum final parked-wheel speed: "
            f"{float(np.max(np.abs(final_wheel_velocities))):.6f} rad/s"
        )

        if args.save_stage is not None:
            save_path = args.save_stage.expanduser().resolve()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if not stage_utils.save_stage(str(save_path)):
                raise RuntimeError(f"Failed to save stage to {save_path}")
            print(f"[f1-replay] Saved stage: {save_path}")

        app_utils.stop()
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
