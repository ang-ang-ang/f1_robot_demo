from __future__ import annotations

HEAD_IMAGE_TOPIC = "/camera/head/color/image_raw/compressed"
LEFT_WRIST_IMAGE_TOPIC = "/camera/left_wrist/color/image_raw/compressed"
RIGHT_WRIST_IMAGE_TOPIC = "/camera/right_wrist/color/image_raw/compressed"
HAL_JOINT_TOPIC = "/hal/joint_states"
LEAD_JOINT_TOPIC = "/lead/joint_states"
LEFT_GRIPPER_COMMAND_TOPIC = "/motion_ctl/gripper/left"
RIGHT_GRIPPER_COMMAND_TOPIC = "/motion_ctl/gripper/right"
LEFT_GRIPPER_STATE_TOPIC = "/motion_ctl/gripper/left/state"
RIGHT_GRIPPER_STATE_TOPIC = "/motion_ctl/gripper/right/state"
LEFT_TCP_STATE_TOPIC = "/state/left_arm/tcp_pos"
RIGHT_TCP_STATE_TOPIC = "/state/right_arm/tcp_pos"
LEFT_LEAD_TCP_TOPIC = "/lead/left_tcp"
RIGHT_LEAD_TCP_TOPIC = "/lead/right_tcp"

IMAGE_TOPICS = (
    HEAD_IMAGE_TOPIC,
    LEFT_WRIST_IMAGE_TOPIC,
    RIGHT_WRIST_IMAGE_TOPIC,
)
STATE_TOPICS = (
    HAL_JOINT_TOPIC,
    LEFT_GRIPPER_STATE_TOPIC,
    RIGHT_GRIPPER_STATE_TOPIC,
    LEFT_TCP_STATE_TOPIC,
    RIGHT_TCP_STATE_TOPIC,
)
ACTION_TOPICS = (
    LEAD_JOINT_TOPIC,
    LEFT_GRIPPER_COMMAND_TOPIC,
    RIGHT_GRIPPER_COMMAND_TOPIC,
    LEFT_LEAD_TCP_TOPIC,
    RIGHT_LEAD_TCP_TOPIC,
)
REQUIRED_TRAINING_TOPICS = (
    *IMAGE_TOPICS,
    HAL_JOINT_TOPIC,
    LEAD_JOINT_TOPIC,
    LEFT_GRIPPER_COMMAND_TOPIC,
    RIGHT_GRIPPER_COMMAND_TOPIC,
    LEFT_GRIPPER_STATE_TOPIC,
    RIGHT_GRIPPER_STATE_TOPIC,
)

STATE_JOINT_NAMES = tuple([f"arm_l_j{index}" for index in range(1, 8)] + [f"arm_r_j{index}" for index in range(1, 8)])
ACTION_JOINT_NAMES = tuple(
    [f"arm_L_joint{index}" for index in range(1, 8)] + [f"arm_R_joint{index}" for index in range(1, 8)]
)


def topic_category(topic: str) -> str:
    if topic in IMAGE_TOPICS:
        return "image"
    if topic in ACTION_TOPICS:
        return "action"
    return "state"
