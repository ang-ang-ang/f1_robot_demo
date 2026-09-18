#!/usr/bin/env python3
"""Minimal standalone Isaac Sim scene for the Genie G1 OmniPicker.

The scene intentionally does not import ``geniesim_benchmark``.  It verifies
the smallest useful stack:

* load ``assets/robot/G1_omnipicker/robot.usda``;
* build a table and a graspable cube from Isaac Sim primitives;
* enable ground/table/cube collisions and cube rigid-body physics;
* initialize the G1 articulation from joint names present in the USD; and
* optionally cycle a small right-arm motion and the right OmniPicker drive.

This is a loading, physics, and low-level joint-control smoke test.  It is not
an IK planner and does not claim to complete a closed-loop grasp.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROBOT_PRIM_PATH = "/G1"
RIGHT_ARM_JOINTS = (
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
)
LEFT_ARM_JOINTS = (
    "idx21_arm_l_joint1",
    "idx22_arm_l_joint2",
    "idx23_arm_l_joint3",
    "idx24_arm_l_joint4",
    "idx25_arm_l_joint5",
    "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
)

# The pose comes from benchmark/config/robot_init_states.py.  Keeping the
# values here makes this script standalone and prevents the benchmark package
# (and its inference/ROS dependencies) from being imported.
LEFT_ARM_HOME = (-1.0751, 0.6109, 0.2793, -1.2846, 0.7295, 1.4957, -0.1868)
RIGHT_ARM_HOME = (1.0734, -0.6109, -0.2793, 1.2846, -0.7313, -1.4957, 0.1868)
RIGHT_GRIPPER_DRIVE = "idx81_gripper_r_outer_joint1"
GRIPPER_OPEN = 0.75
GRIPPER_CLOSED = 0.0


def _repo_root() -> Path:
    override = os.environ.get("GENIESIM_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def _default_assets_root() -> Path:
    override = os.environ.get("GENIESIM_ASSETS_PATH") or os.environ.get("SIM_ASSETS")
    if override:
        return Path(override).expanduser().resolve()
    return _repo_root() / "assets"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load G1 OmniPicker with a primitive table and dynamic cube in Isaac Sim 6.0.",
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=_default_assets_root(),
        help="GenieSimAssets root (default: <repo>/assets or GENIESIM_ASSETS_PATH).",
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
        help="Load and simulate the scene without cycling the right arm and gripper.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip explicit render calls (useful for the fastest headless validation).",
    )
    parser.add_argument(
        "--save-stage",
        type=Path,
        default=None,
        help="Optionally save the composed stage to this USD/USDA path before exit.",
    )
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.headless and args.steps is None:
        args.steps = 240
    return args


def _validate_assets(assets_root: Path) -> Path:
    robot_dir = assets_root / "robot" / "G1_omnipicker"
    robot_usd = robot_dir / "robot.usda"
    required = (
        robot_usd,
        robot_dir / "configuration" / "robot_base.usd",
        robot_dir / "configuration" / "robot_physics.usd",
        robot_dir / "configuration" / "robot_sensor.usd",
        robot_dir / "configuration" / "robot_robot.usd",
        robot_dir / "configuration" / "Cam.usd",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        joined = "\n  - ".join(str(path) for path in missing)
        raise FileNotFoundError(f"G1 asset closure is incomplete:\n  - {joined}")

    # robot.usda references ../../../sim/isaacsim/kit/resources/... relative
    # to assets/robot/G1_omnipicker.  Report the exact expected path before
    # Kit emits a much less obvious unresolved-reference warning.
    camera_asset = (
        robot_dir
        / "../../../sim/isaacsim/kit/resources/models/camera/camera.usd"
    ).resolve()
    if not camera_asset.is_file():
        raise FileNotFoundError(
            "G1 camera dependency is missing. Expected the repo's sim/isaacsim "
            f"link to resolve this file:\n  - {camera_asset}"
        )
    return robot_usd.resolve()


def main() -> int:
    args = _parse_args()
    assets_root = args.assets_root.expanduser().resolve()
    try:
        robot_usd = _validate_assets(assets_root)
    except FileNotFoundError as exc:
        print(f"[g1-smoke] ERROR: {exc}", file=sys.stderr)
        return 2

    # SimulationApp must be created before importing any Omniverse APIs.
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

        print(f"[g1-smoke] Isaac Python: {sys.executable}")
        print(f"[g1-smoke] Assets root: {assets_root}")
        print(f"[g1-smoke] Robot USD: {robot_usd}")

        stage_utils.create_new_stage()
        stage_utils.set_stage_units(meters_per_unit=1.0)
        GroundPlane("/World/GroundPlane", positions=[0.0, 0.0, 0.0])
        light = DistantLight("/World/DistantLight")
        light.set_intensities(500)

        # A primitive table: one top and four static collision legs.  The top
        # surface is z=0.82 m; the block starts 5 mm above it.
        add_static_cube("/World/Table/Top", [0.0, 0.0, 0.78], [0.80, 1.00, 0.08], [0.36, 0.20, 0.10])
        for name, x, y in (
            ("LegFrontLeft", 0.32, 0.42),
            ("LegFrontRight", 0.32, -0.42),
            ("LegBackLeft", -0.32, 0.42),
            ("LegBackRight", -0.32, -0.42),
        ):
            add_static_cube(f"/World/Table/{name}", [x, y, 0.375], [0.07, 0.07, 0.75], [0.28, 0.14, 0.07])

        block_shape = Cube(
            paths="/World/GraspCube",
            positions=[0.05, -0.18, 0.86],
            sizes=1.0,
            scales=[0.07, 0.07, 0.07],
            colors=[0.10, 0.65, 0.95],
        )
        block = RigidPrim(paths=block_shape.paths)
        GeomPrim(paths=block_shape.paths, apply_collision_apis=True)

        stage_utils.add_reference_to_stage(usd_path=str(robot_usd), path=ROBOT_PRIM_PATH)
        for _ in range(600):
            if not stage_utils.is_stage_loading():
                break
            simulation_app.update()
        if stage_utils.is_stage_loading():
            raise RuntimeError("Timed out while loading the local G1 USD stage")

        stage = stage_utils.get_current_stage(backend="usd")
        if not stage.GetPrimAtPath(ROBOT_PRIM_PATH).IsValid():
            raise RuntimeError(f"Robot prim was not composed at {ROBOT_PRIM_PATH}")

        robot = Articulation(ROBOT_PRIM_PATH)
        dof_names = list(robot.dof_names)
        dof_index = {name: index for index, name in enumerate(dof_names)}
        required_control_dofs = (*LEFT_ARM_JOINTS, *RIGHT_ARM_JOINTS, RIGHT_GRIPPER_DRIVE)
        missing_control_dofs = [name for name in required_control_dofs if name not in dof_index]
        if missing_control_dofs:
            raise RuntimeError(
                "G1 articulation is missing expected control DOFs: " + ", ".join(missing_control_dofs)
            )

        initial_positions = np.zeros(len(dof_names), dtype=np.float32)
        initial_by_name = {
            **dict(zip(LEFT_ARM_JOINTS, LEFT_ARM_HOME)),
            **dict(zip(RIGHT_ARM_JOINTS, RIGHT_ARM_HOME)),
            "idx01_body_joint1": 0.30,
            "idx02_body_joint2": 0.52359877,
            "idx11_head_joint1": 0.0,
            "idx12_head_joint2": 0.0,
            "idx41_gripper_l_outer_joint1": GRIPPER_OPEN,
            RIGHT_GRIPPER_DRIVE: GRIPPER_OPEN,
        }
        for name, value in initial_by_name.items():
            if name in dof_index:
                initial_positions[dof_index[name]] = value

        robot.set_default_state(
            positions=[-0.80, 0.0, -0.01],
            orientations=[1.0, 0.0, 0.0, 0.0],
            dof_positions=initial_positions,
        )

        right_arm_indices = [dof_index[name] for name in RIGHT_ARM_JOINTS]
        right_gripper_index = dof_index[RIGHT_GRIPPER_DRIVE]
        inspect_pose = np.asarray(RIGHT_ARM_HOME, dtype=np.float32).copy()
        inspect_pose[0] -= 0.12
        inspect_pose[3] -= 0.18

        SimulationManager.set_physics_dt(1.0 / 60.0)
        set_camera_view(
            eye=[2.2, 2.2, 1.8],
            target=[-0.1, 0.0, 0.85],
            camera_prim_path="/OmniverseKit_Persp",
        )

        app_utils.play()
        simulation_app.update()
        robot.reset_to_default_state()
        for _ in range(10):
            SimulationManager.step()
            simulation_app.update()

        print(f"[g1-smoke] Loaded {len(dof_names)} DOFs")
        print("[g1-smoke] Scene ready: /G1, /World/Table, /World/GraspCube")
        if args.no_motion:
            print("[g1-smoke] Motion demo disabled")
        else:
            print("[g1-smoke] Motion cycle: arm inspect -> gripper close -> gripper open -> arm home")

        frame = 0
        while simulation_app.is_running() and (args.steps is None or frame < args.steps):
            phase = frame % 480
            if not args.no_motion:
                if phase == 60:
                    robot.set_dof_position_targets(inspect_pose, dof_indices=right_arm_indices)
                    print("[g1-smoke] Right arm -> inspect pose")
                elif phase == 180:
                    robot.set_dof_position_targets(GRIPPER_CLOSED, dof_indices=[right_gripper_index])
                    print("[g1-smoke] Right gripper -> closed")
                elif phase == 300:
                    robot.set_dof_position_targets(GRIPPER_OPEN, dof_indices=[right_gripper_index])
                    print("[g1-smoke] Right gripper -> open")
                elif phase == 420:
                    robot.set_dof_position_targets(RIGHT_ARM_HOME, dof_indices=right_arm_indices)
                    print("[g1-smoke] Right arm -> home")

            SimulationManager.step()
            if not args.no_render:
                RenderingManager.render()
            simulation_app.update()
            frame += 1

        final_positions = robot.get_dof_positions()
        if hasattr(final_positions, "numpy"):
            final_positions = final_positions.numpy()
        final_positions = np.asarray(final_positions).reshape(-1)
        block_positions, _ = block.get_world_poses()
        if hasattr(block_positions, "numpy"):
            block_positions = block_positions.numpy()
        block_position = np.asarray(block_positions).reshape(-1, 3)[0]
        print(
            "[g1-smoke] Final right-arm joints:",
            np.array2string(final_positions[right_arm_indices], precision=3, suppress_small=True),
        )
        print(f"[g1-smoke] Final right-gripper drive: {final_positions[right_gripper_index]:.3f} rad")
        print(
            "[g1-smoke] Final cube position:",
            np.array2string(block_position, precision=3, suppress_small=True),
        )
        print(f"[g1-smoke] Completed {frame} frames")

        if args.save_stage is not None:
            save_path = args.save_stage.expanduser().resolve()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if not stage_utils.save_stage(str(save_path)):
                raise RuntimeError(f"Failed to save stage to {save_path}")
            print(f"[g1-smoke] Saved stage: {save_path}")

        app_utils.stop()
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
