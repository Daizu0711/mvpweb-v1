import numpy as np

# Limb segment definitions: (start_keypoint_index, end_keypoint_index)
LIMB_SEGMENTS = {
    'left_upper_arm': (5, 7),    # left_shoulder -> left_elbow
    'left_forearm': (7, 9),      # left_elbow -> left_wrist
    'right_upper_arm': (6, 8),   # right_shoulder -> right_elbow
    'right_forearm': (8, 10),    # right_elbow -> right_wrist
    'left_thigh': (11, 13),      # left_hip -> left_knee
    'left_shin': (13, 15),       # left_knee -> left_ankle
    'right_thigh': (12, 14),     # right_hip -> right_knee
    'right_shin': (14, 16),      # right_knee -> right_ankle
}

LIMB_SEGMENT_NAMES_JA = {
    'left_upper_arm': '左上腕',
    'left_forearm': '左前腕',
    'right_upper_arm': '右上腕',
    'right_forearm': '右前腕',
    'left_thigh': '左太もも',
    'left_shin': '左すね',
    'right_thigh': '右太もも',
    'right_shin': '右すね',
}


def calculate_body_ratios(pose):
    """Calculate limb-length ratios relative to torso height from a pose."""
    if pose is None or 'keypoints' not in pose:
        return None
    kp = pose['keypoints']

    if kp[5][2] < 0.3 or kp[6][2] < 0.3 or kp[11][2] < 0.3 or kp[12][2] < 0.3:
        return None

    shoulder_center = (np.array(kp[5][:2]) + np.array(kp[6][:2])) / 2
    hip_center = (np.array(kp[11][:2]) + np.array(kp[12][:2])) / 2
    torso_height = np.linalg.norm(shoulder_center - hip_center)
    if torso_height < 1:
        return None

    ratios = {}
    for seg_name, (start_idx, end_idx) in LIMB_SEGMENTS.items():
        if kp[start_idx][2] >= 0.3 and kp[end_idx][2] >= 0.3:
            length = np.linalg.norm(np.array(kp[start_idx][:2]) - np.array(kp[end_idx][:2]))
            ratios[seg_name] = length / torso_height
        else:
            ratios[seg_name] = None

    return ratios


def detect_deficiency(current_ratios, registered_ratios, threshold=0.5):
    """Detect limb deficiency by comparing current ratios against registered T-pose ratios."""
    deficiencies = []
    if not current_ratios or not registered_ratios:
        return deficiencies

    for seg_name, reg_ratio in registered_ratios.items():
        if reg_ratio is None:
            continue

        cur_ratio = current_ratios.get(seg_name)
        if cur_ratio is None:
            deficiencies.append({
                'segment': seg_name,
                'segment_ja': LIMB_SEGMENT_NAMES_JA.get(seg_name, seg_name),
                'reason': 'not_detected',
                'registered_ratio': reg_ratio,
                'current_ratio': None
            })
        elif cur_ratio < reg_ratio * (1 - threshold):
            deficiencies.append({
                'segment': seg_name,
                'segment_ja': LIMB_SEGMENT_NAMES_JA.get(seg_name, seg_name),
                'reason': 'ratio_deviation',
                'registered_ratio': reg_ratio,
                'current_ratio': cur_ratio,
                'deviation': (reg_ratio - cur_ratio) / reg_ratio * 100
            })

    return deficiencies


def average_ratios_from_poses(poses):
    """Compute average body ratios across a pose sequence."""
    frame_ratios_list = []
    for pose in poses:
        if pose is None:
            continue
        frame_ratios = calculate_body_ratios(pose)
        if frame_ratios:
            frame_ratios_list.append(frame_ratios)

    if not frame_ratios_list:
        return None

    avg_ratios = {}
    for seg_name in LIMB_SEGMENTS:
        values = [frame_ratio[seg_name] for frame_ratio in frame_ratios_list if frame_ratio.get(seg_name) is not None]
        avg_ratios[seg_name] = sum(values) / len(values) if values else None

    return avg_ratios
