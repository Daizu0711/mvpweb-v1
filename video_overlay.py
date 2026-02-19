"""
Advanced Video Overlay Module
Overlay pose keypoints with score-based coloring
"""

import cv2
import numpy as np
from pathlib import Path
import json
import subprocess
import shutil
import os


def find_ffmpeg():
    """Find ffmpeg binary path"""
    # Check PATH first
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        return ffmpeg
    # Common Homebrew paths on macOS
    for path in [
        '/opt/homebrew/bin/ffmpeg',
        '/usr/local/bin/ffmpeg',
        '/opt/homebrew/Cellar/ffmpeg/8.0_1/bin/ffmpeg',
        '/opt/homebrew/Cellar/ffmpeg/7.1.1_3/bin/ffmpeg',
    ]:
        if os.path.isfile(path):
            return path
    return None


def convert_to_h264(input_path):
    """Convert video to H.264 codec for browser compatibility"""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("⚠ ffmpeg not found, video may not play in browser")
        return input_path

    tmp_path = input_path + '.tmp.mp4'
    try:
        result = subprocess.run(
            [
                ffmpeg, '-y', '-i', input_path,
                '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '23', '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p',
                tmp_path
            ],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0 and os.path.exists(tmp_path):
            os.replace(tmp_path, input_path)
            print(f"✓ Converted to H.264: {input_path}")
        else:
            print(f"⚠ ffmpeg conversion failed: {result.stderr[:200]}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception as e:
        print(f"⚠ ffmpeg conversion error: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return input_path

# COCO skeleton connections
SKELETON_CONNECTIONS = [
    [0, 1], [0, 2], [1, 3], [2, 4],  # Head
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # Arms
    [5, 11], [6, 12], [11, 12],  # Torso
    [11, 13], [13, 15], [12, 14], [14, 16]  # Legs
]

# Joint names (COCO format)
JOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Joint groups for coloring
JOINT_GROUPS = {
    'head': [0, 1, 2, 3, 4],
    'torso': [5, 6, 11, 12],
    'left_arm': [5, 7, 9],
    'right_arm': [6, 8, 10],
    'left_leg': [11, 13, 15],
    'right_leg': [12, 14, 16]
}


class PoseOverlayRenderer:
    """
    Render pose overlay on video with score-based coloring
    """
    
    def __init__(self, color_mode='score', show_labels=True, show_confidence=True):
        """
        Initialize renderer
        
        Args:
            color_mode: 'score' (based on joint scores), 'default' (single color), 'group' (by body part)
            show_labels: Show joint names
            show_confidence: Show confidence values
        """
        self.color_mode = color_mode
        self.show_labels = show_labels
        self.show_confidence = show_confidence
        
        # Color schemes
        self.default_colors = {
            'skeleton': (0, 255, 0),      # Green
            'joint': (255, 0, 0),         # Blue
            'text': (255, 255, 255),      # White
            'background': (0, 0, 0)       # Black
        }
        
        self.group_colors = {
            'head': (255, 200, 100),      # Light blue
            'torso': (100, 255, 100),     # Green
            'left_arm': (255, 100, 255),  # Magenta
            'right_arm': (255, 255, 100), # Cyan
            'left_leg': (100, 100, 255),  # Red
            'right_leg': (100, 255, 255)  # Yellow
        }
    
    def get_score_color(self, score):
        """
        Get color based on score (0-100)
        
        Args:
            score: Joint score (0-100)
        
        Returns:
            BGR color tuple
        """
        if score >= 80:
            # Green (good)
            return (0, 255, 0)
        elif score >= 60:
            # Yellow (medium)
            return (0, 255, 255)
        elif score >= 40:
            # Orange
            return (0, 165, 255)
        else:
            # Red (poor)
            return (0, 0, 255)
    
    def get_joint_color(self, joint_idx, score=None):
        """
        Get color for joint based on mode
        
        Args:
            joint_idx: Joint index (0-16)
            score: Joint score (if color_mode='score')
        
        Returns:
            BGR color tuple
        """
        if self.color_mode == 'score' and score is not None:
            return self.get_score_color(score)
        
        elif self.color_mode == 'group':
            # Find which group this joint belongs to
            for group_name, joints in JOINT_GROUPS.items():
                if joint_idx in joints:
                    return self.group_colors[group_name]
            return self.default_colors['joint']
        
        else:  # default
            return self.default_colors['joint']
    
    def draw_pose_on_frame(self, frame, pose, joint_scores=None, alpha=0.7):
        """
        Draw pose overlay on frame
        
        Args:
            frame: Video frame (numpy array)
            pose: Pose data dict with 'keypoints'
            joint_scores: Dict of joint scores (0-100) or None
            alpha: Overlay transparency (0-1)
        
        Returns:
            Frame with overlay
        """
        if pose is None or 'keypoints' not in pose:
            return frame
        
        overlay = frame.copy()
        keypoints = pose['keypoints']
        
        # Draw skeleton connections
        for connection in SKELETON_CONNECTIONS:
            start_idx, end_idx = connection
            
            if start_idx >= len(keypoints) or end_idx >= len(keypoints):
                continue
            
            start_kp = keypoints[start_idx]
            end_kp = keypoints[end_idx]
            
            # Check confidence
            if start_kp[2] < 0.3 or end_kp[2] < 0.3:
                continue
            
            start_pt = (int(start_kp[0]), int(start_kp[1]))
            end_pt = (int(end_kp[0]), int(end_kp[1]))
            
            # Get color based on average score of connected joints
            if joint_scores and self.color_mode == 'score':
                start_name = JOINT_NAMES[start_idx]
                end_name = JOINT_NAMES[end_idx]
                avg_score = (joint_scores.get(start_name, 50) + 
                           joint_scores.get(end_name, 50)) / 2
                color = self.get_score_color(avg_score)
            else:
                color = self.get_joint_color(start_idx)
            
            # Draw line
            cv2.line(overlay, start_pt, end_pt, color, 3)
        
        # Draw joints
        for idx, kp in enumerate(keypoints):
            if kp[2] < 0.3:  # Low confidence
                continue
            
            center = (int(kp[0]), int(kp[1]))
            
            # Get color
            if joint_scores and self.color_mode == 'score':
                joint_name = JOINT_NAMES[idx]
                score = joint_scores.get(joint_name, 50)
                color = self.get_score_color(score)
            else:
                color = self.get_joint_color(idx)
            
            # Draw outer circle (border)
            cv2.circle(overlay, center, 7, (0, 0, 0), -1)
            # Draw inner circle
            cv2.circle(overlay, center, 5, color, -1)
            
            # Draw label and score
            if self.show_labels or (self.show_confidence and joint_scores):
                label_parts = []
                
                if self.show_labels:
                    label_parts.append(JOINT_NAMES[idx])
                
                if self.show_confidence and joint_scores:
                    joint_name = JOINT_NAMES[idx]
                    if joint_name in joint_scores:
                        label_parts.append(f"{joint_scores[joint_name]:.0f}")
                
                if label_parts:
                    label = ": ".join(label_parts)
                    
                    # Text background
                    (w, h), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                    )
                    cv2.rectangle(
                        overlay,
                        (center[0] + 8, center[1] - h - 5),
                        (center[0] + w + 12, center[1] + 5),
                        (0, 0, 0),
                        -1
                    )
                    
                    # Text
                    cv2.putText(
                        overlay,
                        label,
                        (center[0] + 10, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1
                    )
        
        # Blend with original frame
        result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        return result
    
    def add_score_legend(self, frame, position='top-right'):
        """
        Add color legend to frame
        
        Args:
            frame: Video frame
            position: 'top-right', 'top-left', 'bottom-right', 'bottom-left'
        
        Returns:
            Frame with legend
        """
        if self.color_mode != 'score':
            return frame
        
        h, w = frame.shape[:2]
        legend_items = [
            ("Excellent (80+)", (0, 255, 0)),
            ("Good (60-79)", (0, 255, 255)),
            ("Fair (40-59)", (0, 165, 255)),
            ("Poor (<40)", (0, 0, 255))
        ]
        
        # Calculate legend dimensions
        item_height = 25
        legend_height = len(legend_items) * item_height + 20
        legend_width = 180
        padding = 10
        
        # Calculate position
        if position == 'top-right':
            x = w - legend_width - padding
            y = padding
        elif position == 'top-left':
            x = padding
            y = padding
        elif position == 'bottom-right':
            x = w - legend_width - padding
            y = h - legend_height - padding
        else:  # bottom-left
            x = padding
            y = h - legend_height - padding
        
        # Draw background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x, y),
            (x + legend_width, y + legend_height),
            (0, 0, 0),
            -1
        )
        cv2.rectangle(
            overlay,
            (x, y),
            (x + legend_width, y + legend_height),
            (255, 255, 255),
            2
        )
        
        # Draw items
        for i, (label, color) in enumerate(legend_items):
            item_y = y + 10 + i * item_height
            
            # Color circle
            cv2.circle(overlay, (x + 15, item_y + 10), 8, color, -1)
            cv2.circle(overlay, (x + 15, item_y + 10), 8, (255, 255, 255), 1)
            
            # Text
            cv2.putText(
                overlay,
                label,
                (x + 30, item_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )
        
        # Blend
        result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        return result
    
    def add_overall_score(self, frame, score, position='top-left'):
        """
        Add overall score display to frame
        
        Args:
            frame: Video frame
            score: Overall score (0-100)
            position: 'top-left', 'top-center', 'top-right'
        
        Returns:
            Frame with score
        """
        h, w = frame.shape[:2]
        
        # Create score text
        score_text = f"Score: {score:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX  # Use SIMPLEX instead of BOLD
        font_scale = 1.2
        thickness = 3  # Increase thickness for bold effect
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(
            score_text,
            font,
            font_scale,
            thickness
        )
        
        # Calculate position
        padding = 15
        if position == 'top-left':
            x = padding
        elif position == 'top-center':
            x = (w - text_w) // 2
        else:  # top-right
            x = w - text_w - padding
        
        y = padding + text_h
        
        # Draw background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x - 10, y - text_h - 10),
            (x + text_w + 10, y + baseline + 10),
            (0, 0, 0),
            -1
        )
        
        # Draw border with score color
        border_color = self.get_score_color(score)
        cv2.rectangle(
            overlay,
            (x - 10, y - text_h - 10),
            (x + text_w + 10, y + baseline + 10),
            border_color,
            3
        )
        
        # Draw text
        cv2.putText(
            overlay,
            score_text,
            (x, y),
            font,
            font_scale,
            border_color,
            thickness
        )
        
        # Blend
        result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        return result


def create_overlay_video(
    video_path,
    poses,
    joint_scores,
    output_path,
    overall_score=None,
    color_mode='score',
    show_labels=True,
    show_confidence=True,
    show_legend=True
):
    """
    Create video with pose overlay
    
    Args:
        video_path: Input video path
        poses: List of pose dictionaries
        joint_scores: Dict of joint scores or list of dicts (per frame)
        output_path: Output video path
        overall_score: Overall score to display
        color_mode: 'score', 'default', or 'group'
        show_labels: Show joint labels
        show_confidence: Show confidence scores
        show_legend: Show color legend
    
    Returns:
        Output video path
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"✗ Failed to create video writer for {output_path}")
        return None
    
    # Initialize renderer
    renderer = PoseOverlayRenderer(
        color_mode=color_mode,
        show_labels=show_labels,
        show_confidence=show_confidence
    )
    
    frame_idx = 0
    
    print(f"Creating overlay video: {output_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get pose for this frame
        if frame_idx < len(poses):
            pose = poses[frame_idx]
            
            # Get joint scores for this frame
            if isinstance(joint_scores, list):
                # Per-frame scores
                frame_joint_scores = joint_scores[frame_idx] if frame_idx < len(joint_scores) else None
            else:
                # Overall joint scores
                frame_joint_scores = joint_scores
            
            # Draw pose overlay
            frame = renderer.draw_pose_on_frame(
                frame,
                pose,
                frame_joint_scores,
                alpha=0.6
            )
        
        # Add legend
        if show_legend and color_mode == 'score':
            frame = renderer.add_score_legend(frame, position='top-right')
        
        # Add overall score
        if overall_score is not None:
            frame = renderer.add_overall_score(
                frame,
                overall_score,
                position='top-left'
            )
        
        out.write(frame)
        
        # Progress
        if frame_idx % 30 == 0:
            progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            print(f"\rProgress: {progress:.1f}%", end='')
        
        frame_idx += 1
    
    cap.release()
    out.release()

    # Convert to H.264 for browser compatibility
    convert_to_h264(output_path)

    print(f"\n✓ Created overlay video: {output_path}")

    return output_path


def create_comparison_overlay_video(
    ref_video_path,
    comp_video_path,
    ref_poses,
    comp_poses,
    ref_joint_scores,
    comp_joint_scores,
    output_path,
    ref_overall_score=None,
    comp_overall_score=None
):
    """
    Create side-by-side comparison video with overlays
    
    Args:
        ref_video_path: Reference video path
        comp_video_path: Comparison video path
        ref_poses: Reference poses
        comp_poses: Comparison poses
        ref_joint_scores: Reference joint scores
        comp_joint_scores: Comparison joint scores
        output_path: Output video path
        ref_overall_score: Reference overall score
        comp_overall_score: Comparison overall score
    
    Returns:
        Output video path
    """
    cap_ref = cv2.VideoCapture(ref_video_path)
    cap_comp = cv2.VideoCapture(comp_video_path)
    
    # Get video properties
    fps_ref = int(cap_ref.get(cv2.CAP_PROP_FPS))
    fps_comp = int(cap_comp.get(cv2.CAP_PROP_FPS))
    fps = min(fps_ref, fps_comp)  # Use lower fps
    
    width_ref = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_ref = int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    width_comp = int(cap_comp.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_comp = int(cap_comp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use maximum dimensions for consistency
    target_height = max(height_ref, height_comp)
    target_width = max(width_ref, width_comp)
    
    print(f"Reference video: {width_ref}x{height_ref} @ {fps_ref}fps")
    print(f"Comparison video: {width_comp}x{height_comp} @ {fps_comp}fps")
    print(f"Output video: {target_width*2}x{target_height} @ {fps}fps")
    
    # Create video writer (side-by-side)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width * 2, target_height))
    
    # Initialize renderer
    renderer = PoseOverlayRenderer(
        color_mode='score',
        show_labels=False,
        show_confidence=True
    )
    
    frame_idx = 0
    
    print(f"Creating comparison overlay video: {output_path}")
    
    while True:
        ret_ref, frame_ref = cap_ref.read()
        ret_comp, frame_comp = cap_comp.read()
        
        if not ret_ref or not ret_comp:
            break
        
        # Resize frames to target dimensions if needed
        if frame_ref.shape[0] != target_height or frame_ref.shape[1] != target_width:
            frame_ref = cv2.resize(frame_ref, (target_width, target_height))
        
        if frame_comp.shape[0] != target_height or frame_comp.shape[1] != target_width:
            frame_comp = cv2.resize(frame_comp, (target_width, target_height))
        
        # Process reference frame
        if frame_idx < len(ref_poses):
            pose = ref_poses[frame_idx]
            
            # Adjust pose keypoints if frame was resized
            if pose and (width_ref != target_width or height_ref != target_height):
                pose = adjust_pose_for_resize(pose, width_ref, height_ref, target_width, target_height)
            
            frame_ref = renderer.draw_pose_on_frame(
                frame_ref,
                pose,
                ref_joint_scores,
                alpha=0.6
            )
        
        # Add reference label and score
        cv2.putText(
            frame_ref,
            "REFERENCE",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )
        
        if ref_overall_score is not None:
            frame_ref = renderer.add_overall_score(
                frame_ref,
                ref_overall_score,
                position='top-center'
            )
        
        # Process comparison frame
        if frame_idx < len(comp_poses):
            pose = comp_poses[frame_idx]
            
            # Adjust pose keypoints if frame was resized
            if pose and (width_comp != target_width or height_comp != target_height):
                pose = adjust_pose_for_resize(pose, width_comp, height_comp, target_width, target_height)
            
            frame_comp = renderer.draw_pose_on_frame(
                frame_comp,
                pose,
                comp_joint_scores,
                alpha=0.6
            )
        
        # Add comparison label and score
        cv2.putText(
            frame_comp,
            "YOUR FORM",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 100, 100),
            3
        )
        
        if comp_overall_score is not None:
            frame_comp = renderer.add_overall_score(
                frame_comp,
                comp_overall_score,
                position='top-center'
            )
        
        # Add legend to comparison side
        frame_comp = renderer.add_score_legend(frame_comp, position='top-right')
        
        # Combine frames
        combined = np.hstack([frame_ref, frame_comp])
        out.write(combined)
        
        frame_idx += 1
    
    cap_ref.release()
    cap_comp.release()
    out.release()

    # Convert to H.264 for browser compatibility
    convert_to_h264(output_path)

    print(f"\n✓ Created comparison overlay video: {output_path}")
    
    return output_path


def adjust_pose_for_resize(pose, orig_width, orig_height, new_width, new_height):
    """
    Adjust pose keypoints after frame resize
    
    Args:
        pose: Original pose dict
        orig_width: Original frame width
        orig_height: Original frame height
        new_width: New frame width
        new_height: New frame height
    
    Returns:
        Adjusted pose dict
    """
    if pose is None or 'keypoints' not in pose:
        return pose
    
    adjusted_pose = pose.copy()
    keypoints = pose['keypoints'].copy()
    
    # Calculate scale factors
    scale_x = new_width / orig_width
    scale_y = new_height / orig_height
    
    # Adjust x, y coordinates
    keypoints[:, 0] *= scale_x
    keypoints[:, 1] *= scale_y
    # Keep confidence scores unchanged (column 2)
    
    adjusted_pose['keypoints'] = keypoints
    
    return adjusted_pose