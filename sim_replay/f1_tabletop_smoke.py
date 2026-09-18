#!/usr/bin/env python3
"""Load the locally converted F1 robot for a basic arm-control smoke test.

This is the first integration step for a new robot asset:

* load the Isaac Sim 6.0 Asset Structure 3.0 USD generated from the F1 URDF;
* optionally create a table and a dynamic cube from primitives;
* verify the F1 articulation and its expected joint names;
* repair zero runtime effort/velocity limits on movable joints; and
* cycle a small, joint-space right-arm motion.

The supplied ``f1_wheel_0602_new.urdf`` has no gripper/finger links or
joints.  Consequently this script intentionally does not claim to pick up
the cube.  It establishes the loading, physics, and low-level control layer
that a real or temporary gripper can be added to next.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROBOT_PRIM_PATH = "/F1"
RIGHT_ARM_JOINTS = tuple(f"arm_R_joint{i}" for i in range(1, 8))
LEFT_ARM_JOINTS = tuple(f"arm_L_joint{i}" for i in range(1, 8))
BODY_JOINTS = tuple(f"body_joint{i}" for i in range(1, 6))
HEAD_JOINTS = tuple(f"head_joint{i}" for i in range(1, 4))
WHEEL_JOINTS = tuple(
    f"chassis_{side}_wheel_{position}_joint{joint}"
    for side in ("L", "R")
    for position in ("front", "rear")
    for joint in (1, 2)
)

# Zero is the CAD/URDF reference pose.  The inspect pose stays comfortably
# inside every published F1 limit and is deliberately small: this is a
# joint-control validation, not a calibrated grasp pose.
RIGHT_ARM_HOME = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
RIGHT_ARM_INSPECT = (0.15, -0.25, 0.0, -0.45, 0.0, 0.20, 0.0)


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
        description="Load F1 and validate named joint control; optionally add a primitive table/cube.",
    )
    parser.add_argument(
        "--robot-usd",
        type=Path,
        default=_default_robot_usd(),
        help="Converted F1 root USD (default: <repo>/output/f1_wheel/...).",
    )
    parser.add_argument("--headless", action="store_true", help="Run without the Isaac Sim GUI.")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Stop after N physics frames; default is 240 headless or run-until-close with GUI.",
    )
    parser.add_argument(
        "--no-motion",
        action="store_true",
        help="Load and simulate without sending the right-arm test trajectory.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip explicit rendering (useful for fast headless validation).",
    )
    parser.add_argument(
        "--with-table",
        action="store_true",
        help="Add the primitive table and cube; default is robot plus ground only.",
    )
    parser.add_argument(
        "--save-stage",
        type=Path,
        default=None,
        help="Optionally save the composed F1/table scene as USD/USDA.",
    )
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.headless and args.steps is None:
        args.steps = 240
    return args


def _validate_asset_closure(robot_usd: Path) -> Path:
    robot_usd = robot_usd.expanduser().resolve()
    if not robot_usd.is_file():
        raise FileNotFoundError(
            f"F1 USD does not exist: {robot_usd}\n"
            "Import the URDF first with the command documented in this script's handoff."
        )

    # Isaac Sim 6's importer emits an Asset Structure 3.0 package.  Checking
    # the payload closure up front turns otherwise cryptic composition
    # warnings into an actionable error.
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
        joined = "\n  - ".join(str(path) for path in missing)
        raise FileNotFoundError(f"F1 USD package is incomplete:\n  - {joined}")
    return robot_usd


def main() -> int:
    args = _parse_args()
    try:
        robot_usd = _validate_asset_closure(args.robot_usd)
    except FileNotFoundError as exc:
        print(f"[f1-smoke] ERROR: {exc}", file=sys.stderr)
        return 2

    # Omniverse/Isaac modules must only be imported after SimulationApp.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": args.headless,
            "hide_ui": args.headless,
        }
    )

    try:
        import numpy as np
        import isaacsim.core.experimental.utils.app as app_utils
        import isaacsim.core.experimental.utils.stage as stage_utils
        from isaacsim.core.experimental.objects import Cube, DistantLight, GroundPlane
        from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim
        from isaacsim.core.rendering_manager import RenderingManager
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.utils.viewports import set_camera_view

        def add_static_cube(path: str, position: list[float], scale: list[float], color: list[float]) -> None:
            shape = Cube(paths=path, positions=position, sizes=1.0, scales=scale, colors=color)
            GeomPrim(paths=shape.paths, apply_collision_apis=True)

        print(f"[f1-smoke] Isaac Python: {sys.executable}")
        print(f"[f1-smoke] Robot USD: {robot_usd}")

        stage_utils.create_new_stage()
        stage_utils.set_stage_units(meters_per_unit=1.0)
        GroundPlane("/World/GroundPlane", positions=[0.0, 0.0, 0.0])
        light = DistantLight("/World/DistantLight")
        light.set_intensities(500)

        cube = None
        if args.with_table:
            # The optional tabletop uses primitives only.  Its top surface is
            # z=0.76 m and the cube begins 5 mm above it.
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
            cube = RigidPrim(paths=cube_shape.paths)
            GeomPrim(paths=cube_shape.paths, apply_collision_apis=True)

        stage_utils.add_reference_to_stage(usd_path=str(robot_usd), path=ROBOT_PRIM_PATH)
        for _ in range(600):
            if not stage_utils.is_stage_loading():
                break
            simulation_app.update()
        if stage_utils.is_stage_loading():
            raise RuntimeError("Timed out while loading the local F1 USD package")

        stage = stage_utils.get_current_stage(backend="usd")
        if not stage.GetPrimAtPath(ROBOT_PRIM_PATH).IsValid():
            raise RuntimeError(f"Robot prim was not composed at {ROBOT_PRIM_PATH}")

        robot = Articulation(ROBOT_PRIM_PATH)
        dof_names = list(robot.dof_names)
        dof_index = {name: index for index, name in enumerate(dof_names)}
        required_control_dofs = (*BODY_JOINTS, *LEFT_ARM_JOINTS, *RIGHT_ARM_JOINTS)
        missing_control_dofs = [name for name in required_control_dofs if name not in dof_index]
        if missing_control_dofs:
            raise RuntimeError(
                "F1 articulation is missing expected control DOFs: " + ", ".join(missing_control_dofs)
            )

        gripper_dofs = [
            name
            for name in dof_names
            if any(token in name.lower() for token in ("gripper", "finger", "jaw", "knuckle"))
        ]
        if not gripper_dofs:
            print(
                "[f1-smoke] NOTE: no gripper/finger DOFs are present; "
                "physical cube pickup is unavailable with this URDF."
            )

        initial_positions = np.zeros(len(dof_names), dtype=np.float32)
        robot.set_default_state(
            # The lowest F1 geometry is about 0.217 m below base_link.
            positions=[0.0, 0.0, 0.22],
            orientations=[1.0, 0.0, 0.0, 0.0],
            dof_positions=initial_positions,
        )

        moving_joint_names = (*BODY_JOINTS, *HEAD_JOINTS, *LEFT_ARM_JOINTS, *RIGHT_ARM_JOINTS)
        moving_indices = [dof_index[name] for name in moving_joint_names if name in dof_index]
        right_arm_indices = [dof_index[name] for name in RIGHT_ARM_JOINTS]
        wheel_indices = [dof_index[name] for name in WHEEL_JOINTS]

        SimulationManager.set_physics_dt(1.0 / 60.0)
        set_camera_view(
            eye=[2.2, 2.2, 1.8],
            target=[0.15, 0.0, 0.75],
            camera_prim_path="/OmniverseKit_Persp",
        )

        app_utils.play()
        simulation_app.update()
        robot.reset_to_default_state()

        # arm_R_joint1, arm_L_joint1 and all head joints declare both effort
        # and velocity as zero in the source URDF.  Override the runtime
        # limits for every movable body/head/arm DOF so position targets can
        # actually be exercised; the source file in Downloads is untouched.
        robot.set_dof_max_efforts(100.0, dof_indices=moving_indices)
        robot.set_dof_max_velocities(3.14, dof_indices=moving_indices)
        robot.set_dof_gains(stiffnesses=1000.0, dampings=100.0, dof_indices=moving_indices)

        # The source URDF models all eight steering/rolling wheel DOFs as
        # revolute joints but gives them lower=upper=effort=velocity=0.  The
        # importer consequently leaves weak angular drives whose unbounded
        # response can numerically explode on ground contact.  This arm-only
        # scene parks those DOFs explicitly without modifying the source USD.
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

        for _ in range(10):
            SimulationManager.step()
            simulation_app.update()

        print(f"[f1-smoke] Loaded {len(dof_names)} DOFs")
        print("[f1-smoke] DOFs: " + ", ".join(dof_names))
        if args.with_table:
            print("[f1-smoke] Scene ready: /F1, /World/Table, /World/GraspCube")
        else:
            print("[f1-smoke] Scene ready: /F1 and /World/GroundPlane (table disabled by default)")
        if args.no_motion:
            print("[f1-smoke] Motion demo disabled")
        else:
            print("[f1-smoke] Motion cycle: right arm inspect -> home")

        frame = 0
        while simulation_app.is_running() and (args.steps is None or frame < args.steps):
            phase = frame % 360
            if not args.no_motion:
                if phase == 60:
                    robot.set_dof_position_targets(RIGHT_ARM_INSPECT, dof_indices=right_arm_indices)
                    print("[f1-smoke] Right arm -> inspect pose")
                elif phase == 240:
                    robot.set_dof_position_targets(RIGHT_ARM_HOME, dof_indices=right_arm_indices)
                    print("[f1-smoke] Right arm -> home")

            SimulationManager.step()
            if not args.no_render:
                RenderingManager.render()
            simulation_app.update()
            frame += 1

        final_positions = robot.get_dof_positions()
        if hasattr(final_positions, "numpy"):
            final_positions = final_positions.numpy()
        final_positions = np.asarray(final_positions).reshape(-1)
        final_velocities = robot.get_dof_velocities()
        if hasattr(final_velocities, "numpy"):
            final_velocities = final_velocities.numpy()
        final_velocities = np.asarray(final_velocities).reshape(-1)
        print(
            "[f1-smoke] Final right-arm joints:",
            np.array2string(final_positions[right_arm_indices], precision=3, suppress_small=True),
        )
        print(
            "[f1-smoke] Final wheel positions:",
            np.array2string(final_positions[wheel_indices], precision=3, suppress_small=True),
        )
        print(
            "[f1-smoke] Final wheel velocities (rad/s):",
            np.array2string(final_velocities[wheel_indices], precision=3, suppress_small=True),
        )
        if cube is not None:
            cube_positions, _ = cube.get_world_poses()
            if hasattr(cube_positions, "numpy"):
                cube_positions = cube_positions.numpy()
            cube_position = np.asarray(cube_positions).reshape(-1, 3)[0]
            print(
                "[f1-smoke] Final cube position:",
                np.array2string(cube_position, precision=3, suppress_small=True),
            )
        print(f"[f1-smoke] Completed {frame} frames")

        if args.save_stage is not None:
            save_path = args.save_stage.expanduser().resolve()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if not stage_utils.save_stage(str(save_path)):
                raise RuntimeError(f"Failed to save stage to {save_path}")
            print(f"[f1-smoke] Saved stage: {save_path}")

        app_utils.stop()
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
