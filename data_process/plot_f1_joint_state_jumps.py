"""Plot a detected F1 joint jump beside a normal motion segment.

The plot is deliberately generated from the original MCAP JointState stream.
It keeps the raw values intact and only colors detected discontinuity edges red;
all other samples and edges remain blue.  This makes the visual evidence easy
to compare with the numerical report produced by
``repair_f1_joint_state_jumps``.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from data_process import f1_schema
from data_process.mcap import ImageReference, find_episode_dirs, load_episode, read_image_bytes
from data_process.quality_operator_config import load_operator_config
from data_process.repair_f1_joint_state_jumps import _load_joint_series

BLUE = "#1769aa"
RED = "#d62728"
GRID = "#d9e2ec"


@dataclasses.dataclass(frozen=True)
class JointJumpEvent:
    edge_start: int
    edge_end: int
    sample_start: int
    sample_end: int
    start_time_s: float
    end_time_s: float
    max_step_deg: float
    max_velocity_deg_s: float
    joints: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class Window:
    start: int
    end: int

    @property
    def sample_count(self) -> int:
        return self.end - self.start + 1


@dataclasses.dataclass(frozen=True)
class HeadFrame:
    topic_index: int
    header_time_ns: int
    episode_time_s: float
    delta_ms: float
    path: str
    format: str
    width: int
    height: int

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _contiguous_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive runs from sorted integer indices."""

    if indices.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value != previous + 1:
            runs.append((start, previous))
            start = value
        previous = value
    runs.append((start, previous))
    return runs


def _detect_events(
    values_deg: np.ndarray,
    timestamps_ns: np.ndarray,
    names: tuple[str, ...],
    *,
    max_step_deg: float,
    max_velocity_deg_s: float,
) -> tuple[np.ndarray, np.ndarray, list[JointJumpEvent]]:
    if any(not np.isfinite(value) or value <= 0 for value in (max_step_deg, max_velocity_deg_s)):
        raise ValueError("Step and velocity thresholds must be finite and positive")
    if values_deg.ndim != 2 or values_deg.shape != (len(timestamps_ns), len(names)) or len(names) == 0:
        raise ValueError("Positions must have one row per timestamp and one column per joint name")
    if len(timestamps_ns) < 3 or not np.isfinite(values_deg).all():
        raise ValueError("At least three finite JointState samples are required")
    # Subtract the integer origin before conversion, preserving sub-microsecond
    # timestamp differences even when the recording uses Unix epoch nanoseconds.
    times_s = (timestamps_ns - timestamps_ns[0]).astype(np.float64) / 1e9
    dt_s = np.diff(times_s)
    if np.any(dt_s <= 0):
        raise ValueError("JointState Header timestamps must be strictly increasing")
    steps_deg = np.diff(values_deg, axis=0)
    velocities_deg_s = steps_deg / dt_s[:, None]
    bad_by_joint = (np.abs(steps_deg) > max_step_deg) | (np.abs(velocities_deg_s) > max_velocity_deg_s)
    bad_edges = np.any(bad_by_joint, axis=1)
    events: list[JointJumpEvent] = []
    for edge_start, edge_end in _contiguous_runs(np.flatnonzero(bad_edges)):
        joint_indices = np.flatnonzero(np.any(bad_by_joint[edge_start : edge_end + 1], axis=0))
        events.append(
            JointJumpEvent(
                edge_start=edge_start,
                edge_end=edge_end,
                sample_start=edge_start,
                sample_end=edge_end + 1,
                start_time_s=float(times_s[edge_start] - times_s[0]),
                end_time_s=float(times_s[edge_end + 1] - times_s[0]),
                max_step_deg=float(np.max(np.abs(steps_deg[edge_start : edge_end + 1]))),
                max_velocity_deg_s=float(np.max(np.abs(velocities_deg_s[edge_start : edge_end + 1]))),
                joints=tuple(names[index] for index in joint_indices),
            )
        )
    return times_s, bad_by_joint, events


def _window_for_time_range(times_s: np.ndarray, start_s: float, end_s: float) -> Window:
    start = int(np.searchsorted(times_s - times_s[0], start_s, side="left"))
    end = int(np.searchsorted(times_s - times_s[0], end_s, side="right")) - 1
    start = max(0, min(start, len(times_s) - 1))
    end = max(start, min(end, len(times_s) - 1))
    return Window(start, end)


def _select_abnormal_window(times_s: np.ndarray, event: JointJumpEvent, width_s: float) -> Window:
    total_s = float(times_s[-1] - times_s[0])
    width_s = min(max(width_s, event.end_time_s - event.start_time_s), total_s)
    event_center_s = (event.start_time_s + event.end_time_s) / 2.0
    start_s = event_center_s - width_s / 2.0
    start_s = max(0.0, min(start_s, total_s - width_s))
    window = _window_for_time_range(times_s, start_s, start_s + width_s)
    return Window(min(window.start, event.sample_start), max(window.end, event.sample_end))


def _select_normal_window(times_s: np.ndarray, bad_edges: np.ndarray, width_s: float, values_deg: np.ndarray) -> Window:
    """Center in the longest clean run with motion; reject idle-only references."""

    bad_samples = np.zeros(len(times_s), dtype=bool)
    bad_samples[np.flatnonzero(bad_edges)] = True
    bad_samples[np.flatnonzero(bad_edges) + 1] = True
    clean_samples = np.flatnonzero(~bad_samples)
    clean_runs = [
        (start, end)
        for start, end in _contiguous_runs(clean_samples)
        if end - start >= 2 and float(np.max(np.ptp(values_deg[start : end + 1], axis=0))) >= 1.0
    ]
    if not clean_runs:
        raise ValueError("No clean motion interval with at least three samples and 1 degree of motion is available")

    best = max(clean_runs, key=lambda run: times_s[run[1]] - times_s[run[0]])
    run_start, run_end = best
    run_start_s = float(times_s[run_start] - times_s[0])
    run_end_s = float(times_s[run_end] - times_s[0])
    requested = min(width_s, run_end_s - run_start_s)
    start_s = run_start_s + (run_end_s - run_start_s - requested) / 2.0
    window = _window_for_time_range(times_s, start_s, start_s + requested)
    window = Window(max(run_start, window.start), min(run_end, window.end))
    if window.sample_count >= 3 and np.max(np.ptp(values_deg[window.start : window.end + 1], axis=0)) >= 1.0:
        return window
    # A long clean interval may contain idle time in its middle. Select the
    # most active eligible window instead of presenting a flat line as motion.
    candidates = []
    for start in range(run_start, run_end - 1):
        end = min(int(np.searchsorted(times_s, times_s[start] + width_s, side="right")) - 1, run_end)
        if end - start >= 2:
            motion = float(np.max(np.ptp(values_deg[start : end + 1], axis=0)))
            if motion >= 1.0:
                candidates.append((motion, start, end))
    if not candidates:
        raise ValueError("The requested normal window is too short to show at least 1 degree of motion")
    _, start, end = max(candidates)
    return Window(start, end)


def _image_bytes(reference: ImageReference) -> bytes:
    with reference.path.open("rb") as handle:
        return read_image_bytes(reference, {reference.path: handle})


def _decode_head_frame(
    sample: Any, topic_index: int, topic_origin_ns: int, target_ns: int
) -> tuple[HeadFrame, np.ndarray]:
    if sample.image is None or sample.header_time_ns is None:
        raise ValueError("Head image sample has no decodable image reference or Header timestamp")
    encoded = _image_bytes(sample.image)
    image = _decode_image_array(encoded)
    header_time_ns = int(sample.header_time_ns)
    return (
        HeadFrame(
            topic_index=topic_index,
            header_time_ns=header_time_ns,
            episode_time_s=(header_time_ns - topic_origin_ns) / 1e9,
            delta_ms=(header_time_ns - target_ns) / 1e6,
            path=str(sample.source_path.resolve()),
            format=sample.image.format,
            width=int(image.shape[1]),
            height=int(image.shape[0]),
        ),
        image,
    )


def _write_head_image(path: Path, image: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"OpenCV could not write head-camera screenshot: {path}")


def _decode_image_array(encoded: bytes) -> np.ndarray:
    import cv2

    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV could not decode a head-camera JPEG payload")
    return image


def _nearest_head_frame(head_samples: list[Any], head_times_ns: np.ndarray, target_ns: int) -> tuple[int, Any]:
    if not head_samples:
        raise ValueError(f"Topic {f1_schema.HEAD_IMAGE_TOPIC} has no messages")
    index = int(np.argmin(np.abs(head_times_ns - target_ns)))
    return index, head_samples[index]


def _write_events(path: Path, events: list[JointJumpEvent]) -> None:
    fieldnames = [
        "edge_start",
        "edge_end",
        "sample_start",
        "sample_end",
        "start_time_s",
        "end_time_s",
        "max_step_deg",
        "max_velocity_deg_s",
        "joints",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            row = event.as_dict()
            row["joints"] = ",".join(event.joints)
            writer.writerow(row)


def _write_samples(
    path: Path,
    times_s: np.ndarray,
    timestamps_ns: np.ndarray,
    values_deg: np.ndarray,
    names: tuple[str, ...],
    bad_by_joint: np.ndarray,
    windows: dict[str, Window],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["window", "frame_index", "header_time_ns", "episode_time_s", "window_time_s"]
            + [f"{name}_deg" for name in names]
            + [f"{name}_incoming_jump" for name in names]
        )
        for label, window in windows.items():
            for index in range(window.start, window.end + 1):
                flags = bad_by_joint[index - 1] if index else np.zeros(len(names), dtype=bool)
                writer.writerow(
                    [label, index, int(timestamps_ns[index]), times_s[index], times_s[index] - times_s[window.start]]
                    + values_deg[index].tolist()
                    + flags.astype(int).tolist()
                )


def _plot(
    *,
    output_path: Path,
    times_s: np.ndarray,
    values_deg: np.ndarray,
    names: tuple[str, ...],
    normal_window: Window,
    abnormal_window: Window,
    bad_by_joint: np.ndarray,
    event: JointJumpEvent,
    topic: str,
    episode_name: str,
    max_step_deg: float,
    max_velocity_deg_s: float,
    normal_head: tuple[HeadFrame, np.ndarray],
    abnormal_head: tuple[HeadFrame, np.ndarray],
) -> None:
    # Delay the plotting import so --help and numerical validation also work
    # in a minimal NumPy environment.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    joint_count = values_deg.shape[1]
    figure_height = max(18.0, 1.55 * joint_count + 4.0)
    figure = plt.figure(figsize=(15.5, figure_height))
    grid = figure.add_gridspec(
        joint_count + 1,
        2,
        height_ratios=[3.6, *([1.0] * joint_count)],
        wspace=0.22,
        hspace=0.42,
    )
    image_axes = [figure.add_subplot(grid[0, column]) for column in range(2)]
    axes = np.asarray(
        [[figure.add_subplot(grid[row, column]) for column in range(2)] for row in range(1, joint_count + 1)]
    )
    for row in range(1, joint_count):
        for column in range(2):
            axes[row, column].sharex(axes[0, column])
    for axis in image_axes:
        axis.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        axis.set_xticks([])
        axis.set_yticks([])
    normal_x = times_s[normal_window.start : normal_window.end + 1] - times_s[normal_window.start]
    normal_slice = slice(normal_window.start, normal_window.end + 1)

    for column, (axis, payload, title) in enumerate(
        (
            (image_axes[0], normal_head, "Head camera at normal reference"),
            (image_axes[1], abnormal_head, "Head camera at detected jump"),
        )
    ):
        head_info, image = payload
        axis.imshow(image[:, :, ::-1])
        axis.set_title(
            f"{title}\nframe {head_info.topic_index}, episode time {head_info.episode_time_s:.3f} s "
            f"(nearest Δ {head_info.delta_ms:+.1f} ms)",
            fontsize=10,
            pad=8,
        )
    for joint_index, name in enumerate(names):
        normal_axis, abnormal_axis = axes[joint_index]
        normal_axis.plot(normal_x, values_deg[normal_slice, joint_index], color=BLUE, linewidth=1.2)
        normal_axis.set_ylabel(f"{name}\n(deg)", fontsize=8, rotation=0, labelpad=39, va="center")
        normal_axis.grid(True, color=GRID, linewidth=0.5, alpha=0.8)
        normal_axis.tick_params(labelsize=7)

        indices = np.arange(abnormal_window.start, abnormal_window.end + 1)
        points = np.column_stack((times_s[indices] - times_s[abnormal_window.start], values_deg[indices, joint_index]))
        segments = np.stack((points[:-1], points[1:]), axis=1)
        flags = bad_by_joint[abnormal_window.start : abnormal_window.end, joint_index]
        abnormal_axis.add_collection(
            LineCollection(segments, colors=np.where(flags, RED, BLUE), linewidths=np.where(flags, 2.2, 1.2))
        )
        abnormal_axis.autoscale_view()
        # Explicitly mark the endpoints of abnormal edges for a clear point-level cue.
        edge_indices = (
            np.flatnonzero(bad_by_joint[abnormal_window.start : abnormal_window.end, joint_index])
            + abnormal_window.start
        )
        sample_indices = np.unique(np.concatenate((edge_indices, edge_indices + 1)))
        sample_indices = sample_indices[
            (sample_indices >= abnormal_window.start) & (sample_indices <= abnormal_window.end)
        ]
        if sample_indices.size:
            abnormal_axis.scatter(
                times_s[sample_indices] - times_s[abnormal_window.start],
                values_deg[sample_indices, joint_index],
                color=RED,
                s=10,
                zorder=3,
            )
        abnormal_axis.set_ylabel(f"{name}\n(deg)", fontsize=8, rotation=0, labelpad=39, va="center")
        abnormal_axis.grid(True, color=GRID, linewidth=0.5, alpha=0.8)
        abnormal_axis.tick_params(labelsize=7)

    axes[0, 0].set_title(
        f"Normal reference (no threshold crossing)\n"
        f"Episode {times_s[normal_window.start]:.3f}-{times_s[normal_window.end]:.3f} s, "
        f"frames {normal_window.start}-{normal_window.end}",
        fontsize=10,
        pad=12,
    )
    axes[0, 1].set_title(
        f"Detected jump with original context\n"
        f"Episode {times_s[abnormal_window.start]:.3f}-{times_s[abnormal_window.end]:.3f} s, "
        f"frames {abnormal_window.start}-{abnormal_window.end}",
        fontsize=10,
        pad=12,
    )
    axes[-1, 0].set_xlabel("Time in normal window (s)")
    axes[-1, 1].set_xlabel("Time in jump window (s)")
    for axis in axes[-1]:
        axis.tick_params(labelsize=8)
    figure.suptitle(
        f"F1 joint angles | {episode_name}\n"
        f"topic={topic} | jump frames {event.sample_start}-{event.sample_end} | "
        f"max step {event.max_step_deg:.2f}° | max velocity {event.max_velocity_deg_s:.1f}°/s\n"
        f"thresholds: {max_step_deg:.1f}°/sample, {max_velocity_deg_s:.1f}°/s",
        fontsize=12,
        y=0.993,
    )
    figure.legend(
        handles=[
            Line2D([0], [0], color=BLUE, lw=2, label="Below jump thresholds"),
            Line2D([0], [0], color=RED, marker="o", lw=2, label="Threshold exceeded (review required)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.55, 0.965),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    figure.text(
        0.55,
        0.013,
        "Original degree readings; zero-based topic frames. Blue means no detected jump, not full quality approval.\n"
        "Each subplot has its own angle scale. /lead = commanded targets; /hal = measured follower state.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    figure.subplots_adjust(top=0.91, left=0.12, right=0.985, bottom=0.045)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def generate_plot(
    raw_dir: Path,
    output_dir: Path,
    *,
    topic: str = "/lead/joint_states",
    joint_prefix: str = "arm_",
    max_step_deg: float = 25.0,
    max_velocity_deg_s: float = 500.0,
    normal_window_s: float = 3.0,
    abnormal_window_s: float = 1.4,
) -> dict[str, Any]:
    if any(not np.isfinite(value) or value <= 0 for value in (normal_window_s, abnormal_window_s)):
        raise ValueError("Window durations must be finite and positive")
    episode_dirs = find_episode_dirs(raw_dir)
    if len(episode_dirs) != 1:
        raise ValueError(
            f"Expected one Episode directory or a directory containing one MCAP, found {len(episode_dirs)} under {raw_dir}; "
            "pass the exact Episode directory with --raw-dir"
        )
    episode_dir = episode_dirs[0]
    episode = load_episode(episode_dir)
    samples, names, values, timestamps_ns, selected_indices = _load_joint_series(episode, topic, joint_prefix)
    if any(sample.header_time_ns is None for sample in samples):
        raise ValueError("Every JointState sample must have a Header timestamp")
    selected_names = tuple(names[index] for index in selected_indices)
    values_deg = values[:, selected_indices]
    times_s, bad_by_joint, events = _detect_events(
        values_deg,
        timestamps_ns,
        selected_names,
        max_step_deg=max_step_deg,
        max_velocity_deg_s=max_velocity_deg_s,
    )
    if not events:
        raise ValueError(f"No joint jump was detected in {episode_dir} on {topic}")
    event = max(events, key=lambda item: (item.max_step_deg, item.max_velocity_deg_s))
    bad_edges = np.any(bad_by_joint, axis=1)
    normal_window = _select_normal_window(times_s, bad_edges, normal_window_s, values_deg)
    abnormal_window = _select_abnormal_window(times_s, event, abnormal_window_s)

    head_samples = episode.topic(f1_schema.HEAD_IMAGE_TOPIC)
    head_times_ns = np.asarray([sample.header_time_ns for sample in head_samples], dtype=np.int64)
    if len(head_samples) == 0 or np.any(np.diff(head_times_ns) < 0):
        raise ValueError(f"Head camera topic {f1_schema.HEAD_IMAGE_TOPIC} must have non-decreasing Header timestamps")
    topic_origin_ns = int(timestamps_ns[0])
    normal_target_ns = int(timestamps_ns[normal_window.start])
    abnormal_target_ns = int(timestamps_ns[event.sample_start])
    normal_head_index, normal_head_sample = _nearest_head_frame(head_samples, head_times_ns, normal_target_ns)
    abnormal_head_index, abnormal_head_sample = _nearest_head_frame(head_samples, head_times_ns, abnormal_target_ns)
    normal_head = _decode_head_frame(normal_head_sample, normal_head_index, topic_origin_ns, normal_target_ns)
    abnormal_head = _decode_head_frame(abnormal_head_sample, abnormal_head_index, topic_origin_ns, abnormal_target_ns)

    output_dir.mkdir(parents=True, exist_ok=True)
    normal_head_image_path = output_dir / "head_camera_normal_reference.jpg"
    abnormal_head_image_path = output_dir / "head_camera_jump_reference.jpg"
    _write_head_image(normal_head_image_path, normal_head[1])
    _write_head_image(abnormal_head_image_path, abnormal_head[1])
    image_path = output_dir / "joint_jump_vs_normal.png"
    _plot(
        output_path=image_path,
        times_s=times_s,
        values_deg=values_deg,
        names=selected_names,
        normal_window=normal_window,
        abnormal_window=abnormal_window,
        bad_by_joint=bad_by_joint,
        event=event,
        topic=topic,
        episode_name=episode.name,
        max_step_deg=max_step_deg,
        max_velocity_deg_s=max_velocity_deg_s,
        normal_head=normal_head,
        abnormal_head=abnormal_head,
    )
    _write_events(output_dir / "joint_jump_events.csv", events)
    _write_samples(
        output_dir / "joint_angle_samples.csv",
        times_s,
        timestamps_ns,
        values_deg,
        selected_names,
        bad_by_joint,
        {"normal": normal_window, "jump": abnormal_window},
    )
    normal_steps = np.diff(values_deg[normal_window.start : normal_window.end + 1], axis=0)
    normal_dt = np.diff(times_s[normal_window.start : normal_window.end + 1])
    normal_range = float(np.max(np.ptp(values_deg[normal_window.start : normal_window.end + 1], axis=0)))
    event_steps = np.diff(values_deg[event.sample_start : event.sample_end + 1], axis=0)
    largest_edge, largest_joint = np.unravel_index(np.argmax(np.abs(event_steps)), event_steps.shape)
    largest_frame = event.sample_start + int(largest_edge)
    metadata = {
        "format": "f1_joint_jump_plot_v1",
        "episode": episode.name,
        "episode_dir": str(episode_dir.resolve()),
        "topic": topic,
        "units": "Raw F1 JointState.position in degrees; no interpolation or resampling",
        "time_source": "Header nanoseconds; episode_time_s relative to first sample of selected topic",
        "frame_indexing": "zero-based, selected topic in original MCAP record order",
        "sample_count": len(samples),
        "source_files": [
            {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in episode.source_files
        ],
        "joint_names": list(selected_names),
        "head_camera": {
            "topic": f1_schema.HEAD_IMAGE_TOPIC,
            "normal_reference": normal_head[0].as_dict(),
            "abnormal_reference": abnormal_head[0].as_dict(),
            "selection": "nearest Header timestamp to the first sample of each plotted JointState window",
        },
        "thresholds": {
            "max_step_deg": max_step_deg,
            "max_velocity_deg_s": max_velocity_deg_s,
        },
        "event_count": len(events),
        "event_count_definition": "Contiguous abnormal time intervals merged across joints",
        "flagged_joint_edge_count": int(np.count_nonzero(bad_by_joint)),
        "selected_event": event.as_dict(),
        "largest_jump": {
            "joint": selected_names[largest_joint],
            "from_frame": largest_frame,
            "to_frame": largest_frame + 1,
            "from_deg": float(values_deg[largest_frame, largest_joint]),
            "to_deg": float(values_deg[largest_frame + 1, largest_joint]),
            "step_deg": float(event_steps[largest_edge, largest_joint]),
            "interval_ms": float((times_s[largest_frame + 1] - times_s[largest_frame]) * 1000),
        },
        "normal_window": {
            "start_frame": normal_window.start,
            "end_frame": normal_window.end,
            "start_time_s": float(times_s[normal_window.start] - times_s[0]),
            "end_time_s": float(times_s[normal_window.end] - times_s[0]),
            "sample_count": normal_window.sample_count,
            "max_step_deg": float(np.max(np.abs(normal_steps))),
            "max_velocity_deg_s": float(np.max(np.abs(normal_steps / normal_dt[:, None]))),
            "max_joint_range_deg": normal_range,
        },
        "abnormal_window": {
            "start_frame": abnormal_window.start,
            "end_frame": abnormal_window.end,
            "start_time_s": float(times_s[abnormal_window.start] - times_s[0]),
            "end_time_s": float(times_s[abnormal_window.end] - times_s[0]),
            "sample_count": abnormal_window.sample_count,
        },
        "artifacts": {
            "plot": str(image_path),
            "events_csv": str(output_dir / "joint_jump_events.csv"),
            "samples_csv": str(output_dir / "joint_angle_samples.csv"),
            "normal_head_image": str(normal_head_image_path),
            "abnormal_head_image": str(abnormal_head_image_path),
        },
    }
    (output_dir / "joint_jump_vs_normal.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True, help="One Episode directory containing an MCAP")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/joint_jump_visualization"),
        help="Directory for PNG, JSON and CSV outputs",
    )
    parser.add_argument(
        "--topic", choices=(f1_schema.HAL_JOINT_TOPIC, f1_schema.LEAD_JOINT_TOPIC), default=f1_schema.LEAD_JOINT_TOPIC
    )
    parser.add_argument("--joint-prefix", default="arm_")
    parser.add_argument("--config", type=Path, help="Optional operator TOML override")
    parser.add_argument("--max-step-deg", type=float, help="Override the topic-specific operator threshold")
    parser.add_argument("--max-velocity-deg-s", type=float, help="Override the topic-specific operator threshold")
    parser.add_argument("--normal-window-s", type=float, default=3.0)
    parser.add_argument("--abnormal-window-s", type=float, default=1.4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = load_operator_config(args.config).joint_motion
    signal = config.state if args.topic == f1_schema.HAL_JOINT_TOPIC else config.action
    metadata = generate_plot(
        args.raw_dir,
        args.output_dir,
        topic=args.topic,
        joint_prefix=args.joint_prefix,
        max_step_deg=args.max_step_deg if args.max_step_deg is not None else signal.max_step_deg,
        max_velocity_deg_s=(
            args.max_velocity_deg_s if args.max_velocity_deg_s is not None else signal.fallback_max_velocity_deg_s
        ),
        normal_window_s=args.normal_window_s,
        abnormal_window_s=args.abnormal_window_s,
    )
    selected = metadata["selected_event"]
    normal = metadata["normal_window"]
    abnormal = metadata["abnormal_window"]
    print(
        f"Wrote {metadata['artifacts']['plot']}\n"
        f"selected jump: frames {selected['sample_start']}-{selected['sample_end']}, "
        f"max step {selected['max_step_deg']:.3f} deg, max velocity {selected['max_velocity_deg_s']:.3f} deg/s\n"
        f"normal window: frames {normal['start_frame']}-{normal['end_frame']}\n"
        f"abnormal window: frames {abnormal['start_frame']}-{abnormal['end_frame']}"
    )


if __name__ == "__main__":
    main()
