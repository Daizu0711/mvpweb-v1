import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.transform import Rotation
import torch
import traceback

class PoseAnalyzer:
    """VitPose-based pose extraction and analysis"""
    
    def __init__(self, use_vitpose=True, model_variant='vitpose-b', force_mediapipe=False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_vitpose = use_vitpose
        self.model_variant = model_variant
        self.force_mediapipe = force_mediapipe
        
        print(f"Using device: {self.device}")
        
        if use_vitpose:
            self.init_vitpose_model()
        else:
            self.init_pose_model()
    
    def init_vitpose_model(self):
        """Initialize VitPose model"""
        try:
            from vitpose_detector import VitPoseDetector
            
            self.vitpose = VitPoseDetector(
                model_name=self.model_variant,
                device=self.device,
                force_mediapipe=self.force_mediapipe
            )
            self.detector_ready = True
            print(f"✓ VitPose ({self.model_variant}) initialized successfully")
            
        except Exception as e:
            print(f"⚠ VitPose initialization failed: {e}")
            print("Falling back to basic detector")
            self.use_vitpose = False
            self.init_pose_model()
    
    def init_pose_model(self):
        """Initialize pose estimation model"""
        try:
            # Using OpenPose model as fallback (VitPose would require specific setup)
            # Download model files if needed
            prototxt = "pose/coco/pose_deploy_linevec.prototxt"
            weights = "pose/coco/pose_iter_440000.caffemodel"
            
            # For demo purposes, we'll use a simple keypoint detector
            # In production, integrate actual VitPose model
            self.net = None
            self.detector_ready = True
            print("Pose model initialized (using fallback detector)")
            
        except Exception as e:
            print(f"Model initialization error: {e}")
            self.net = None
            self.detector_ready = True
    
    def extract_poses_from_video(self, video_path, sample_rate=1):
        """Extract pose sequences from video"""
        if self.use_vitpose and hasattr(self, 'vitpose'):
            print(f"Using VitPose detector for {video_path}")
            return self.vitpose.detect_video(video_path, sample_rate)
        else:
            print(f"Using fallback detector for {video_path}")
            return self._extract_poses_fallback(video_path, sample_rate)
    
    def _extract_poses_fallback(self, video_path, sample_rate=1):
        """Fallback pose extraction"""
        cap = cv2.VideoCapture(video_path)
        poses = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                pose = self.detect_pose(frame)
                poses.append(pose)
            
            frame_idx += 1
        
        cap.release()
        return poses
    
    def detect_pose(self, frame):
        """Detect pose in a single frame"""
        if self.use_vitpose and hasattr(self, 'vitpose'):
            return self.vitpose.detect(frame)
        else:
            return self._detect_pose_fallback(frame)
    
    def _detect_pose_fallback(self, frame):
        """Basic fallback pose detection"""
        try:
            # Simplified pose detection - in production use VitPose
            # This is a placeholder that generates realistic keypoint structure
            height, width = frame.shape[:2]
            
            # Use simple color-based or contour detection as fallback
            # For demo, create structure compatible with pose comparison
            keypoints = self.simple_pose_detection(frame)
            
            return {
                'keypoints': keypoints,
                'bbox': [0, 0, width, height],
                'score': 0.9
            }
            
        except Exception as e:
            print(f"Pose detection error: {e}")
            return None
    
    def simple_pose_detection(self, frame):
        """Simple pose detection using image processing"""
        height, width = frame.shape[:2]
        
        # Convert to grayscale and detect contours
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest contour (assume it's the person)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Estimate keypoints based on body proportions (17 keypoints - COCO format)
            keypoints = []
            
            # Head (nose, eyes, ears)
            keypoints.append([x + w/2, y + h*0.1, 0.9])  # nose
            keypoints.append([x + w*0.4, y + h*0.08, 0.85])  # left eye
            keypoints.append([x + w*0.6, y + h*0.08, 0.85])  # right eye
            keypoints.append([x + w*0.35, y + h*0.12, 0.8])  # left ear
            keypoints.append([x + w*0.65, y + h*0.12, 0.8])  # right ear
            
            # Upper body (shoulders, elbows, wrists)
            keypoints.append([x + w*0.3, y + h*0.25, 0.9])  # left shoulder
            keypoints.append([x + w*0.7, y + h*0.25, 0.9])  # right shoulder
            keypoints.append([x + w*0.2, y + h*0.45, 0.85])  # left elbow
            keypoints.append([x + w*0.8, y + h*0.45, 0.85])  # right elbow
            keypoints.append([x + w*0.15, y + h*0.6, 0.8])  # left wrist
            keypoints.append([x + w*0.85, y + h*0.6, 0.8])  # right wrist
            
            # Lower body (hips, knees, ankles)
            keypoints.append([x + w*0.35, y + h*0.55, 0.9])  # left hip
            keypoints.append([x + w*0.65, y + h*0.55, 0.9])  # right hip
            keypoints.append([x + w*0.3, y + h*0.75, 0.85])  # left knee
            keypoints.append([x + w*0.7, y + h*0.75, 0.85])  # right knee
            keypoints.append([x + w*0.3, y + h*0.95, 0.8])  # left ankle
            keypoints.append([x + w*0.7, y + h*0.95, 0.8])  # right ankle
            
            return np.array(keypoints)
        
        # Default pose if no contour found
        return np.array([[width/2, height/2, 0.5] for _ in range(17)])


class PoseComparator:
    """Advanced pose comparison with normalization"""
    
    def __init__(self):
        # Define important joint groups
        self.joint_groups = {
            'head': [0, 1, 2, 3, 4],
            'torso': [5, 6, 11, 12],
            'left_arm': [5, 7, 9],
            'right_arm': [6, 8, 10],
            'left_leg': [11, 13, 15],
            'right_leg': [12, 14, 16]
        }
        
        # Joint names for COCO format
        self.joint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def normalize_pose(self, pose_data):
        """Normalize pose to be scale and position invariant"""
        if pose_data is None:
            return None
        
        keypoints = pose_data['keypoints'].copy()
        
        # Extract coordinates and confidence
        coords = keypoints[:, :2]
        confidence = keypoints[:, 2]
        
        # Filter out low confidence points
        valid_mask = confidence > 0.3
        if not np.any(valid_mask):
            return None
        
        valid_coords = coords[valid_mask]
        
        # Calculate center (hip center)
        if len(keypoints) > 12:
            center = (coords[11] + coords[12]) / 2  # Average of hips
        else:
            center = np.mean(valid_coords, axis=0)
        
        # Translate to origin
        normalized_coords = coords - center
        
        # Calculate scale (shoulder-to-hip distance)
        if len(keypoints) > 12:
            shoulder_center = (coords[5] + coords[6]) / 2
            hip_center = center
            scale = np.linalg.norm(shoulder_center - hip_center)
            
            if scale > 0:
                normalized_coords = normalized_coords / scale
        
        # Combine with confidence scores
        normalized_keypoints = np.column_stack([normalized_coords, confidence])
        
        return {
            'keypoints': normalized_keypoints,
            'center': center,
            'scale': scale if 'scale' in locals() else 1.0
        }
    
    def align_temporal_sequences(self, seq1, seq2):
        """Align two sequences using Dynamic Time Warping"""
        n, m = len(seq1), len(seq2)
        
        # DTW matrix
        dtw = np.zeros((n + 1, m + 1))
        dtw[0, :] = np.inf
        dtw[:, 0] = np.inf
        dtw[0, 0] = 0
        
        # Fill DTW matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if seq1[i-1] is not None and seq2[j-1] is not None:
                    cost = self.pose_distance(seq1[i-1], seq2[j-1])
                    dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
                else:
                    dtw[i, j] = np.inf
        
        # Backtrack to find alignment path
        path = []
        i, j = n, m
        while i > 0 and j > 0:
            path.append((i-1, j-1))
            min_idx = np.argmin([dtw[i-1, j-1], dtw[i-1, j], dtw[i, j-1]])
            if min_idx == 0:
                i, j = i-1, j-1
            elif min_idx == 1:
                i = i-1
            else:
                j = j-1
        
        path.reverse()
        return path, dtw[n, m] / len(path) if path else float('inf')
    
    def pose_distance(self, pose1, pose2):
        """Calculate distance between two normalized poses"""
        if pose1 is None or pose2 is None:
            return float('inf')
        
        kp1 = pose1['keypoints']
        kp2 = pose2['keypoints']
        
        # Calculate weighted distance
        total_distance = 0
        total_weight = 0
        
        for i in range(min(len(kp1), len(kp2))):
            # Weight by confidence
            weight = kp1[i, 2] * kp2[i, 2]
            if weight > 0.09:  # 0.3 * 0.3
                dist = euclidean(kp1[i, :2], kp2[i, :2])
                total_distance += dist * weight
                total_weight += weight
        
        return total_distance / total_weight if total_weight > 0 else float('inf')
    
    def calculate_joint_angles(self, pose):
        """Calculate angles at each joint"""
        if pose is None:
            return {}
        
        keypoints = pose['keypoints']
        angles = {}
        
        # Define angle triplets (point1, vertex, point2)
        angle_defs = {
            'left_elbow': (5, 7, 9),
            'right_elbow': (6, 8, 10),
            'left_knee': (11, 13, 15),
            'right_knee': (12, 14, 16),
            'left_shoulder': (7, 5, 11),
            'right_shoulder': (8, 6, 12),
            'left_hip': (5, 11, 13),
            'right_hip': (6, 12, 14)
        }
        
        for joint_name, (p1, vertex, p2) in angle_defs.items():
            if p1 < len(keypoints) and vertex < len(keypoints) and p2 < len(keypoints):
                if keypoints[p1, 2] > 0.3 and keypoints[vertex, 2] > 0.3 and keypoints[p2, 2] > 0.3:
                    v1 = keypoints[p1, :2] - keypoints[vertex, :2]
                    v2 = keypoints[p2, :2] - keypoints[vertex, :2]
                    
                    angle = np.arccos(np.clip(np.dot(v1, v2) / 
                                             (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
                    angles[joint_name] = np.degrees(angle)
        
        return angles
    
    def compare_pose_sequences(self, ref_poses, comp_poses, use_3d=False):
        """Compare two pose sequences comprehensively"""
        # Normalize all poses
        ref_normalized = [self.normalize_pose(p) for p in ref_poses]
        comp_normalized = [self.normalize_pose(p) for p in comp_poses]
        
        # Temporal alignment
        alignment_path, dtw_cost = self.align_temporal_sequences(ref_normalized, comp_normalized)
        
        # Calculate frame-by-frame scores
        frame_scores = []
        joint_score_accumulator = {name: [] for name in self.joint_names}
        
        for ref_idx, comp_idx in alignment_path:
            ref_pose = ref_normalized[ref_idx]
            comp_pose = comp_normalized[comp_idx]
            
            if ref_pose and comp_pose:
                # Frame distance
                frame_dist = self.pose_distance(ref_pose, comp_pose)
                frame_score = max(0, 100 - frame_dist * 50)
                frame_scores.append(frame_score)
                
                # Per-joint scores
                ref_kp = ref_pose['keypoints']
                comp_kp = comp_pose['keypoints']
                
                for i, joint_name in enumerate(self.joint_names):
                    if i < len(ref_kp) and i < len(comp_kp):
                        if ref_kp[i, 2] > 0.3 and comp_kp[i, 2] > 0.3:
                            joint_dist = euclidean(ref_kp[i, :2], comp_kp[i, :2])
                            joint_score = max(0, 100 - joint_dist * 100)
                            joint_score_accumulator[joint_name].append(joint_score)
        
        # Calculate overall scores
        overall_score = np.mean(frame_scores) if frame_scores else 0
        
        joint_scores = {}
        for joint_name, scores in joint_score_accumulator.items():
            joint_scores[joint_name] = np.mean(scores) if scores else 0
        
        # Temporal alignment score
        temporal_score = max(0, 100 - dtw_cost * 10)
        
        return {
            'overall_score': overall_score,
            'joint_scores': joint_scores,
            'temporal_alignment': temporal_score,
            'frame_scores': frame_scores,
            'alignment_quality': 1.0 - (dtw_cost / max(len(ref_poses), len(comp_poses)))
        }


class Pose3DEstimator:
    """3D pose estimation using PoseFormer"""
    
    def __init__(self, device='cuda'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.initialized = False
        print(f"Pose3DEstimator using device: {self.device}")
    
    def initialize_model(self, checkpoint_path=None):
        """Initialize PoseFormer model"""
        try:
            from poseformer_model import load_poseformer_model
            self.model = load_poseformer_model(checkpoint_path, device=self.device)
            self.initialized = True
            print("PoseFormer model initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize PoseFormer model: {e}")
            print("Will use simplified 3D estimation")
            self.initialized = False
    
    def lift_to_3d(self, pose_2d_sequence):
        """Lift 2D poses to 3D using PoseFormer"""
        if not self.initialized:
            # Try to initialize
            self.initialize_model()
        
        if self.initialized and self.model is not None:
            try:
                return self._lift_with_poseformer(pose_2d_sequence)
            except Exception as e:
                print(f"PoseFormer inference failed: {e}, falling back to simple estimation")
                return self._lift_simple(pose_2d_sequence)
        else:
            return self._lift_simple(pose_2d_sequence)
    
    def _lift_with_poseformer(self, pose_2d_sequence):
        """Use actual PoseFormer model for 3D lifting"""
        from poseformer_model import prepare_2d_poses_for_poseformer, normalize_2d_poses
        
        # Normalize 2D poses
        normalized_poses = normalize_2d_poses(pose_2d_sequence)
        
        # Prepare input tensor
        input_tensor = prepare_2d_poses_for_poseformer(normalized_poses)
        input_tensor = input_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            output_3d = self.model(input_tensor)  # [1, T, 17, 3]
        
        # Convert to numpy
        output_3d = output_3d.cpu().numpy()[0]  # [T, 17, 3]
        
        # Create output format
        poses_3d = []
        for i, pose_2d in enumerate(pose_2d_sequence):
            if pose_2d is None:
                poses_3d.append(None)
                continue
            
            if i < len(output_3d):
                # Combine 3D coords with 2D confidence
                keypoints_3d = np.column_stack([
                    output_3d[i],  # [17, 3]
                    pose_2d['keypoints'][:, 2]  # confidence
                ])
                
                poses_3d.append({
                    'keypoints_3d': keypoints_3d,  # [17, 4] (x, y, z, conf)
                    'bbox': pose_2d.get('bbox', [0, 0, 0, 0])
                })
            else:
                poses_3d.append(None)
        
        return poses_3d
    
    def _lift_simple(self, pose_2d_sequence):
        """Simple depth estimation based on body proportions"""
        poses_3d = []
        
        for pose_2d in pose_2d_sequence:
            if pose_2d is None:
                poses_3d.append(None)
                continue
            
            keypoints_2d = pose_2d['keypoints']
            
            # Estimate depth based on vertical position and body part
            depths = np.zeros(len(keypoints_2d))
            
            # Head further forward
            depths[:5] = 0.1
            
            # Torso neutral
            depths[5:13] = 0.0
            
            # Estimate limb depth from angles
            if len(keypoints_2d) > 16:
                # Left arm
                if keypoints_2d[5, 2] > 0.3 and keypoints_2d[7, 2] > 0.3:
                    shoulder_elbow = keypoints_2d[7, :2] - keypoints_2d[5, :2]
                    angle = np.arctan2(shoulder_elbow[1], shoulder_elbow[0])
                    depths[7] = np.sin(angle) * 0.2
                    depths[9] = np.sin(angle) * 0.3
                
                # Right arm
                if keypoints_2d[6, 2] > 0.3 and keypoints_2d[8, 2] > 0.3:
                    shoulder_elbow = keypoints_2d[8, :2] - keypoints_2d[6, :2]
                    angle = np.arctan2(shoulder_elbow[1], shoulder_elbow[0])
                    depths[8] = np.sin(angle) * 0.2
                    depths[10] = np.sin(angle) * 0.3
                
                # Legs
                depths[13:17] = -0.05
            
            # Combine with 2D coordinates
            keypoints_3d = np.column_stack([
                keypoints_2d[:, :2],
                depths,
                keypoints_2d[:, 2]  # Confidence
            ])
            
            poses_3d.append({
                'keypoints_3d': keypoints_3d,
                'bbox': pose_2d.get('bbox', [0, 0, 0, 0])
            })
        
        return poses_3d
    
    def create_3d_visualization(self, pose_3d_sequence, output_path, viz_type='interactive'):
        """Create 3D visualization of pose sequence"""
        from pose_3d_visualizer import Pose3DVisualizer
        
        visualizer = Pose3DVisualizer(skeleton_type='coco')
        
        if viz_type == 'interactive':
            return visualizer.create_interactive_3d_html(
                pose_3d_sequence, 
                output_path
            )
        elif viz_type == 'video':
            return visualizer.create_matplotlib_animation(
                pose_3d_sequence,
                output_path
            )
        else:
            return visualizer.create_interactive_3d_html(
                pose_3d_sequence,
                output_path
            )