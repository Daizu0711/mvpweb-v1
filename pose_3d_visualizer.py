"""
3D Pose Visualization Module
Supports interactive 3D rendering and animation export
"""

import numpy as np
import cv2
import json
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
import plotly.graph_objects as go
import plotly.express as px

# H36M skeleton definition for 17 joints
H36M_SKELETON = [
    [0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
    [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
    [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]
]

# COCO skeleton definition
COCO_SKELETON = [
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

# Color scheme for different body parts
BODY_PART_COLORS = {
    'head': '#FF6B6B',
    'torso': '#4ECDC4',
    'left_arm': '#45B7D1',
    'right_arm': '#96CEB4',
    'left_leg': '#FFEAA7',
    'right_leg': '#DFE6E9'
}


class Pose3DVisualizer:
    """
    3D Pose Visualization with multiple rendering backends
    """
    
    def __init__(self, skeleton_type='coco'):
        self.skeleton_type = skeleton_type
        self.skeleton = COCO_SKELETON if skeleton_type == 'coco' else H36M_SKELETON
    
    def create_interactive_3d_html(self, poses_3d, output_path, title="3D Pose Sequence"):
        """
        Create interactive 3D visualization using Plotly
        
        Args:
            poses_3d: List of 3D poses [T, 17, 3]
            output_path: Path to save HTML file
        """
        if isinstance(poses_3d, list):
            poses_3d = np.array([p['keypoints_3d'][:, :3] if p else np.zeros((17, 3)) 
                                for p in poses_3d])
        
        num_frames = len(poses_3d)
        
        # Create frames for animation
        frames = []
        
        for frame_idx in range(num_frames):
            pose = poses_3d[frame_idx]
            
            # Skeleton lines
            skeleton_x, skeleton_y, skeleton_z = [], [], []
            for connection in self.skeleton:
                start_idx, end_idx = connection
                if start_idx < len(pose) and end_idx < len(pose):
                    skeleton_x.extend([pose[start_idx, 0], pose[end_idx, 0], None])
                    skeleton_y.extend([pose[start_idx, 1], pose[end_idx, 1], None])
                    skeleton_z.extend([pose[start_idx, 2], pose[end_idx, 2], None])
            
            # Joint points
            joint_x = pose[:, 0]
            joint_y = pose[:, 1]
            joint_z = pose[:, 2]
            
            frame_data = [
                go.Scatter3d(
                    x=skeleton_x,
                    y=skeleton_y,
                    z=skeleton_z,
                    mode='lines',
                    line=dict(color='#4ECDC4', width=4),
                    name='Skeleton',
                    showlegend=False
                ),
                go.Scatter3d(
                    x=joint_x,
                    y=joint_y,
                    z=joint_z,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=list(range(17)),
                        colorscale='Viridis',
                        showscale=False
                    ),
                    text=JOINT_NAMES,
                    name='Joints',
                    showlegend=False
                )
            ]
            
            frames.append(go.Frame(data=frame_data, name=str(frame_idx)))
        
        # Initial frame
        fig = go.Figure(
            data=frames[0].data,
            frames=frames
        )
        
        # Layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(range=[-1, 1], title='X'),
                yaxis=dict(range=[-1, 1], title='Y'),
                zaxis=dict(range=[-1, 1], title='Z'),
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    buttons=[
                        dict(
                            label='Play',
                            method='animate',
                            args=[None, dict(
                                frame=dict(duration=50, redraw=True),
                                fromcurrent=True,
                                mode='immediate'
                            )]
                        ),
                        dict(
                            label='Pause',
                            method='animate',
                            args=[[None], dict(
                                frame=dict(duration=0, redraw=False),
                                mode='immediate'
                            )]
                        )
                    ],
                    x=0.1,
                    y=0
                )
            ],
            sliders=[
                dict(
                    active=0,
                    yanchor='top',
                    y=0,
                    xanchor='left',
                    currentvalue=dict(
                        prefix='Frame: ',
                        visible=True,
                        xanchor='right'
                    ),
                    steps=[
                        dict(
                            args=[[f.name], dict(
                                frame=dict(duration=0, redraw=True),
                                mode='immediate'
                            )],
                            label=str(k),
                            method='animate'
                        )
                        for k, f in enumerate(fig.frames)
                    ]
                )
            ]
        )
        
        # Save
        fig.write_html(output_path)
        print(f"Saved interactive 3D visualization to {output_path}")
        
        return output_path
    
    def create_comparison_3d_html(self, ref_poses_3d, comp_poses_3d, output_path):
        """
        Create side-by-side 3D comparison
        """
        if isinstance(ref_poses_3d, list):
            ref_poses_3d = np.array([p['keypoints_3d'][:, :3] if p else np.zeros((17, 3)) 
                                     for p in ref_poses_3d])
        if isinstance(comp_poses_3d, list):
            comp_poses_3d = np.array([p['keypoints_3d'][:, :3] if p else np.zeros((17, 3)) 
                                      for p in comp_poses_3d])
        
        from plotly.subplots import make_subplots
        
        num_frames = min(len(ref_poses_3d), len(comp_poses_3d))
        
        # Create subplot
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=('Reference', 'Comparison')
        )
        
        # Create frames
        frames = []
        for frame_idx in range(num_frames):
            ref_pose = ref_poses_3d[frame_idx]
            comp_pose = comp_poses_3d[frame_idx]
            
            # Reference skeleton
            ref_skel_x, ref_skel_y, ref_skel_z = [], [], []
            for connection in self.skeleton:
                start_idx, end_idx = connection
                if start_idx < len(ref_pose) and end_idx < len(ref_pose):
                    ref_skel_x.extend([ref_pose[start_idx, 0], ref_pose[end_idx, 0], None])
                    ref_skel_y.extend([ref_pose[start_idx, 1], ref_pose[end_idx, 1], None])
                    ref_skel_z.extend([ref_pose[start_idx, 2], ref_pose[end_idx, 2], None])
            
            # Comparison skeleton
            comp_skel_x, comp_skel_y, comp_skel_z = [], [], []
            for connection in self.skeleton:
                start_idx, end_idx = connection
                if start_idx < len(comp_pose) and end_idx < len(comp_pose):
                    comp_skel_x.extend([comp_pose[start_idx, 0], comp_pose[end_idx, 0], None])
                    comp_skel_y.extend([comp_pose[start_idx, 1], comp_pose[end_idx, 1], None])
                    comp_skel_z.extend([comp_pose[start_idx, 2], comp_pose[end_idx, 2], None])
            
            frame_data = [
                # Reference
                go.Scatter3d(
                    x=ref_skel_x, y=ref_skel_y, z=ref_skel_z,
                    mode='lines', line=dict(color='green', width=4),
                    showlegend=False
                ),
                go.Scatter3d(
                    x=ref_pose[:, 0], y=ref_pose[:, 1], z=ref_pose[:, 2],
                    mode='markers', marker=dict(size=8, color='green'),
                    showlegend=False
                ),
                # Comparison
                go.Scatter3d(
                    x=comp_skel_x, y=comp_skel_y, z=comp_skel_z,
                    mode='lines', line=dict(color='red', width=4),
                    showlegend=False
                ),
                go.Scatter3d(
                    x=comp_pose[:, 0], y=comp_pose[:, 1], z=comp_pose[:, 2],
                    mode='markers', marker=dict(size=8, color='red'),
                    showlegend=False
                )
            ]
            
            frames.append(go.Frame(data=frame_data, name=str(frame_idx)))
        
        # Add initial data
        fig.add_traces(frames[0].data[:2], rows=1, cols=1)
        fig.add_traces(frames[0].data[2:], rows=1, cols=2)
        fig.frames = frames
        
        # Update layout
        fig.update_layout(
            title="3D Pose Comparison",
            scene=dict(
                xaxis=dict(range=[-1, 1]), yaxis=dict(range=[-1, 1]), zaxis=dict(range=[-1, 1]),
                aspectmode='cube'
            ),
            scene2=dict(
                xaxis=dict(range=[-1, 1]), yaxis=dict(range=[-1, 1]), zaxis=dict(range=[-1, 1]),
                aspectmode='cube'
            ),
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    buttons=[
                        dict(label='Play', method='animate',
                             args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]),
                        dict(label='Pause', method='animate',
                             args=[[None], dict(frame=dict(duration=0, redraw=False))])
                    ]
                )
            ],
            sliders=[dict(
                active=0,
                steps=[dict(args=[[f.name]], label=str(k), method='animate')
                       for k, f in enumerate(fig.frames)]
            )]
        )
        
        fig.write_html(output_path)
        print(f"Saved 3D comparison to {output_path}")
        
        return output_path
    
    def create_matplotlib_animation(self, poses_3d, output_path, fps=10):
        """
        Create 3D animation video using matplotlib
        """
        if isinstance(poses_3d, list):
            poses_3d = np.array([p['keypoints_3d'][:, :3] if p else np.zeros((17, 3)) 
                                for p in poses_3d])
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        def update(frame_idx):
            ax.clear()
            pose = poses_3d[frame_idx]
            
            # Plot skeleton
            for connection in self.skeleton:
                start_idx, end_idx = connection
                if start_idx < len(pose) and end_idx < len(pose):
                    ax.plot(
                        [pose[start_idx, 0], pose[end_idx, 0]],
                        [pose[start_idx, 1], pose[end_idx, 1]],
                        [pose[start_idx, 2], pose[end_idx, 2]],
                        'b-', linewidth=2
                    )
            
            # Plot joints
            ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], 
                      c='red', s=50, marker='o')
            
            # Set limits
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Frame {frame_idx}/{len(poses_3d)}')
            
            # Set viewing angle
            ax.view_init(elev=20, azim=frame_idx * 2)
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=len(poses_3d), interval=1000/fps)
        
        # Save
        writer = FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(output_path, writer=writer)
        plt.close()
        
        print(f"Saved 3D animation to {output_path}")
        return output_path
    
    def export_3d_json(self, poses_3d, output_path):
        """
        Export 3D poses to JSON format
        """
        if isinstance(poses_3d, list):
            poses_data = []
            for i, pose in enumerate(poses_3d):
                if pose and 'keypoints_3d' in pose:
                    poses_data.append({
                        'frame': i,
                        'keypoints': pose['keypoints_3d'].tolist(),
                        'joint_names': JOINT_NAMES
                    })
        else:
            poses_data = []
            for i, pose in enumerate(poses_3d):
                poses_data.append({
                    'frame': i,
                    'keypoints': pose.tolist(),
                    'joint_names': JOINT_NAMES
                })
        
        data = {
            'skeleton': self.skeleton,
            'poses': poses_data,
            'num_frames': len(poses_data),
            'num_joints': 17
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved 3D pose data to {output_path}")
        return output_path


def calculate_3d_metrics(ref_poses_3d, comp_poses_3d):
    """
    Calculate 3D-specific metrics
    """
    if isinstance(ref_poses_3d, list):
        ref_poses_3d = np.array([p['keypoints_3d'][:, :3] if p else np.zeros((17, 3)) 
                                for p in ref_poses_3d])
    if isinstance(comp_poses_3d, list):
        comp_poses_3d = np.array([p['keypoints_3d'][:, :3] if p else np.zeros((17, 3)) 
                                  for p in comp_poses_3d])
    
    metrics = {
        'mpjpe': [],  # Mean Per Joint Position Error
        'pa_mpjpe': [],  # Procrustes Aligned MPJPE
        'joint_errors': {name: [] for name in JOINT_NAMES}
    }
    
    num_frames = min(len(ref_poses_3d), len(comp_poses_3d))
    
    for i in range(num_frames):
        ref_pose = ref_poses_3d[i]
        comp_pose = comp_poses_3d[i]
        
        # MPJPE
        errors = np.linalg.norm(ref_pose - comp_pose, axis=1)
        metrics['mpjpe'].append(errors.mean())
        
        # Per-joint errors
        for j, joint_name in enumerate(JOINT_NAMES):
            if j < len(errors):
                metrics['joint_errors'][joint_name].append(errors[j])
    
    # Average metrics
    metrics['mean_mpjpe'] = np.mean(metrics['mpjpe'])
    metrics['mean_joint_errors'] = {
        name: np.mean(errors) if errors else 0
        for name, errors in metrics['joint_errors'].items()
    }
    
    return metrics
