"""
CoMaD Person_1 joint order matches official mapping.json indices 0..24:

 0 BackLeft, 1 BackRight, 2 BackTop, 3 Chest, 4 HeadFront, 5 HeadSide, 6 HeadTop,
 7 LElbowOut, 8 LHandOut, 9 LShoulderBack, 10 LShoulderTop, 11 LUArmHigh,
 12 LWristIn, 13 LWristOut, 14 RElbowOut, 15 RHandOut, 16 RShoulderBack,
 17 RShoulderTop, 18 RUArmHigh, 19 RWristIn, 20 RWristOut,
 21 WaistLBack, 22 WaistLFront, 23 WaistRBack, 24 WaistRFront

Parent array for visualization / Skeleton.links (root = index 0, consistent with root-relative code).
"""

import numpy as np
from data_loader.skeleton import Skeleton

# parent[i] = parent joint index of i, or -1 for root
COMAD_P1_PARENTS = [
    -1,  # 0 BackLeft (dataset root-relative anchor)
    0,  # 1 BackRight
    0,  # 2 BackTop
    2,  # 3 Chest
    6,  # 4 HeadFront
    6,  # 5 HeadSide
    3,  # 6 HeadTop
    11,  # 7 LElbowOut
    7,  # 8 LHandOut
    3,  # 9 LShoulderBack
    9,  # 10 LShoulderTop
    10,  # 11 LUArmHigh
    11,  # 12 LWristIn
    11,  # 13 LWristOut
    18,  # 14 RElbowOut
    14,  # 15 RHandOut
    3,  # 16 RShoulderBack
    16,  # 17 RShoulderTop
    17,  # 18 RUArmHigh
    18,  # 19 RWristIn
    18,  # 20 RWristOut
    3,  # 21 WaistLBack
    21,  # 22 WaistLFront
    3,  # 23 WaistRBack
    23,  # 24 WaistRFront
]

# Left / right coloring in render_animation (anatomical left vs right arm chains)
COMAD_P1_JOINTS_LEFT = [7, 8, 9, 10, 11, 12, 13]
COMAD_P1_JOINTS_RIGHT = [14, 15, 16, 17, 18, 19, 20]

# Official InteRACT loaders do not use all 25 CoMaD markers for forecasting.
# HR uses a compact Alice/P1 marker set; HH uses a compact upper-body/arms set.
COMAD_HR_VIS_JOINTS = [0, 1, 2, 3, 4, 5, 6, 9, 10]
COMAD_HH_VIS_JOINTS = [2, 9, 16, 7, 14, 13, 20, 8, 15]

COMAD_HR_VIS_LINKS = [
    (1, 0),  # BackRight - BackLeft
    (2, 0),  # BackTop - BackLeft
    (2, 1),  # BackTop - BackRight
    (3, 2),  # Chest - BackTop
    (6, 3),  # HeadTop - Chest
    (4, 6),  # HeadFront - HeadTop
    (5, 6),  # HeadSide - HeadTop
    (7, 3),  # LShoulderBack - Chest
    (8, 7),  # LShoulderTop - LShoulderBack
]

COMAD_HH_VIS_LINKS = [
    (1, 0),  # LShoulderBack - BackTop
    (3, 1),  # LElbowOut - LShoulderBack
    (5, 3),  # LWristOut - LElbowOut
    (7, 5),  # LHandOut - LWristOut
    (2, 0),  # RShoulderBack - BackTop
    (4, 2),  # RElbowOut - RShoulderBack
    (6, 4),  # RWristOut - RElbowOut
    (8, 6),  # RHandOut - RWristOut
]


def comad_p1_links():
    return [(j, p) for j, p in enumerate(COMAD_P1_PARENTS) if p >= 0]


def comad_visual_mode(cfg):
    mode = str(getattr(cfg, "comad_vis_joint_set", "auto")).lower()
    if mode in {"hr", "hh"}:
        return mode

    interactions = getattr(cfg, "comad_test_interactions", None)
    if interactions == {"HH"}:
        return "hh"
    return "hr"


def comad_visual_joint_indices(cfg):
    mode = comad_visual_mode(cfg)
    if mode == "hh":
        return COMAD_HH_VIS_JOINTS
    return COMAD_HR_VIS_JOINTS


def comad_visual_skeleton(cfg):
    mode = comad_visual_mode(cfg)
    if mode == "hh":
        parents = [-1, 0, 0, 1, 2, 3, 4, 5, 6]
        return Skeleton(
            parents=parents,
            joints_left=[1, 3, 5, 7],
            joints_right=[2, 4, 6, 8],
            links=COMAD_HH_VIS_LINKS,
        )

    parents = [-1, 0, 0, 2, 6, 6, 3, 3, 7]
    return Skeleton(
        parents=parents,
        joints_left=[],
        joints_right=[],
        links=COMAD_HR_VIS_LINKS,
    )


def comad_fix_orientation_motive_to_interact(x):
    """
    Match InteRACT interact/utils/comad_hr.py::fix_orientation (Motive -> plotting frame).

    https://github.com/portal-cornell/interact/blob/release/interact/utils/comad_hr.py
    tensor[:, :, [0, 1, 2]] = tensor[:, :, [0, 2, 1]]; tensor[:, :, 0] *= -1

    Same as interact/utils/read_json_data.py::transform_coords for Motive (x,y,z).
    Supports any leading batch dims; last dim is xyz.
    """
    a = np.asarray(x, dtype=np.float32)
    y = np.empty_like(a, dtype=np.float32)
    y[..., 0] = -a[..., 0]
    y[..., 1] = a[..., 2]
    y[..., 2] = a[..., 1]
    return y
