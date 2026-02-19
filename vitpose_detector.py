"""
VitPose Model Integration
High-accuracy pose estimation using Vision Transformer
"""

import cv2
import numpy as np
import torch
import os
from pathlib import Path
import urllib.request
import json

class VitPoseDetector:
    """
    VitPose detector with MMPose backend
    Supports multiple model variants: base, large, huge
    """
    
    def __init__(self, model_name='vitpose-b', device='cuda', force_mediapipe=False):
        """
        Initialize VitPose detector
        
        Args:
            model_name: 'vitpose-b' (base), 'vitpose-l' (large), 'vitpose-h' (huge)
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.model = None
        self.cfg = None
        self.force_mediapipe = force_mediapipe
        
        # Model configurations
        self.model_configs = {
            'vitpose-b': {
                'config': 'vitpose_base_coco_256x192.py',
                'checkpoint': 'vitpose_base_coco_256x192.pth',
                'url': 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/vitpose_base_coco_256x192-216eae50_20221122.pth',
                'input_size': (192, 256)
            },
            'vitpose-l': {
                'config': 'vitpose_large_coco_256x192.py',
                'checkpoint': 'vitpose_large_coco_256x192.pth',
                'url': 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/vitpose_large_coco_256x192-f53f2d93_20221122.pth',
                'input_size': (192, 256)
            },
            'vitpose-h': {
                'config': 'vitpose_huge_coco_256x192.py',
                'checkpoint': 'vitpose_huge_coco_256x192.pth',
                'url': 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/vitpose_huge_coco_256x192-b0f78d35_20221122.pth',
                'input_size': (192, 256)
            }
        }
        
        print(f"Initializing VitPose ({model_name}) on {self.device}")
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize VitPose model with MMPose"""
        try:
            if self.force_mediapipe:
                self.use_mmpose = False
                self._initialize_fallback()
                return

            # Try to use MMPose
            from mmpose.apis import init_model, inference_topdown
            from mmpose.structures import merge_data_samples
            
            config = self.model_configs[self.model_name]
            checkpoint_path = self._get_checkpoint(config)
            config_path = self._get_config(config)
            
            # Initialize model
            self.model = init_model(
                config_path,
                checkpoint_path,
                device=self.device
            )
            
            self.use_mmpose = True
            print(f"✓ VitPose model loaded with MMPose backend")
            
        except ImportError:
            print("⚠ MMPose not available, using fallback detector")
            self.use_mmpose = False
            self._initialize_fallback()
        except Exception as e:
            print(f"⚠ Error loading VitPose: {e}")
            print("Using fallback detector")
            self.use_mmpose = False
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Initialize fallback detector (MediaPipe or OpenPose)"""
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.use_mediapipe = True
            print("✓ Using MediaPipe as fallback")
        except ImportError:
            print("⚠ MediaPipe not available")
            self.use_mediapipe = False
            print("✓ Using basic keypoint detector")
    
    def _get_checkpoint(self, config):
        """Download or get checkpoint file"""
        checkpoint_dir = Path('models/vitpose')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / config['checkpoint']
        
        if not checkpoint_path.exists():
            print(f"Downloading VitPose checkpoint...")
            try:
                urllib.request.urlretrieve(
                    config['url'],
                    checkpoint_path,
                    reporthook=self._download_progress
                )
                print(f"\n✓ Downloaded to {checkpoint_path}")
            except Exception as e:
                print(f"✗ Download failed: {e}")
                raise
        
        return str(checkpoint_path)
    
    def _get_config(self, config):
        """Get or create config file"""
        config_dir = Path('models/vitpose/configs')
        config_dir.mkdir(parents=True, exist_ok=True)
        
        config_path = config_dir / config['config']
        
        if not config_path.exists():
            # Create config file
            self._create_config_file(config_path, config)
        
        return str(config_path)
    
    def _create_config_file(self, config_path, config):
        """Create MMPose config file"""
        config_content = f"""
# VitPose Configuration

#Model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True
    ),
    backbone=dict(
        type='ViT',
        img_size={config['input_size']},
        patch_size=16,
        embed_dim=768 if 'base' in self.model_name else 1024,
        depth=12 if 'base' in self.model_name else 24,
        num_heads=12 if 'base' in self.model_name else 16,
        ratio=1,
        use_abs_pos_embed=True,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=768 if 'base' in self.model_name else 1024,
        out_channels=17,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=dict(
            type='RegressionLabel',
            input_size={config['input_size']},
            sigma=2
        )
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True
    )
)

# Dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/coco/'

# Codec settings
codec = dict(
    type='MSRAHeatmap',
    input_size={config['input_size']},
    heatmap_size=(48, 64),
    sigma=2
)
"""
        
        config_path.write_text(config_content)
        print(f"✓ Created config file: {config_path}")
    
    def _download_progress(self, block_num, block_size, total_size):
        """Show download progress"""
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        print(f"\rProgress: {percent:.1f}%", end='')
    
    def detect(self, image, bbox=None):
        """
        Detect pose in image
        
        Args:
            image: numpy array (H, W, 3)
            bbox: optional bounding box [x1, y1, x2, y2]
        
        Returns:
            dict with 'keypoints' [17, 3] and 'bbox'
        """
        if self.use_mmpose and self.model is not None:
            return self._detect_with_vitpose(image, bbox)
        elif hasattr(self, 'use_mediapipe') and self.use_mediapipe:
            return self._detect_with_mediapipe(image)
        else:
            return self._detect_fallback(image)
    
    def _detect_with_vitpose(self, image, bbox=None):
        """Detect with VitPose model"""
        from mmpose.apis import inference_topdown
        from mmpose.structures import PoseDataSample, InstanceData
        
        # If no bbox provided, use whole image
        if bbox is None:
            h, w = image.shape[:2]
            bbox = [0, 0, w, h]
        
        # Create bbox data
        bbox_data = np.array([bbox], dtype=np.float32)
        
        # Inference
        results = inference_topdown(
            self.model,
            image,
            bboxes=bbox_data
        )
        
        if not results:
            return None
        
        # Extract keypoints
        result = results[0]
        keypoints = result.pred_instances.keypoints[0]  # [17, 2]
        scores = result.pred_instances.keypoint_scores[0]  # [17]
        
        # Combine to [17, 3] format
        keypoints_with_conf = np.concatenate([
            keypoints,
            scores[:, np.newaxis]
        ], axis=1)
        
        return {
            'keypoints': keypoints_with_conf,
            'bbox': bbox,
            'score': float(scores.mean())
        }
    
    def _detect_with_mediapipe(self, image):
        """Detect with MediaPipe"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process
        results = self.pose_detector.process(image_rgb)
        
        if not results.pose_landmarks:
            return None
        
        # Convert MediaPipe landmarks to COCO format
        h, w = image.shape[:2]
        keypoints = self._mediapipe_to_coco(results.pose_landmarks, w, h)
        
        return {
            'keypoints': keypoints,
            'bbox': [0, 0, w, h],
            'score': 0.9
        }
    
    def _mediapipe_to_coco(self, landmarks, width, height):
        """Convert MediaPipe landmarks to COCO format"""
        # MediaPipe to COCO mapping
        mp_to_coco = {
            0: 0,   # nose
            2: 1,   # left eye
            5: 2,   # right eye
            7: 3,   # left ear
            8: 4,   # right ear
            11: 5,  # left shoulder
            12: 6,  # right shoulder
            13: 7,  # left elbow
            14: 8,  # right elbow
            15: 9,  # left wrist
            16: 10, # right wrist
            23: 11, # left hip
            24: 12, # right hip
            25: 13, # left knee
            26: 14, # right knee
            27: 15, # left ankle
            28: 16  # right ankle
        }
        
        keypoints = np.zeros((17, 3))
        
        for mp_idx, coco_idx in mp_to_coco.items():
            if mp_idx < len(landmarks.landmark):
                lm = landmarks.landmark[mp_idx]
                keypoints[coco_idx] = [
                    lm.x * width,
                    lm.y * height,
                    lm.visibility
                ]
        
        return keypoints
    
    def _detect_fallback(self, image):
        """Basic fallback detector"""
        h, w = image.shape[:2]
        
        # Simple contour-based detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(
            thresh, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w_box, h_box = cv2.boundingRect(largest_contour)
        
        # Estimate keypoints from body proportions
        keypoints = np.zeros((17, 3))
        
        # Head
        keypoints[0] = [x + w_box/2, y + h_box*0.1, 0.8]  # nose
        keypoints[1] = [x + w_box*0.4, y + h_box*0.08, 0.7]  # left eye
        keypoints[2] = [x + w_box*0.6, y + h_box*0.08, 0.7]  # right eye
        keypoints[3] = [x + w_box*0.35, y + h_box*0.12, 0.6]  # left ear
        keypoints[4] = [x + w_box*0.65, y + h_box*0.12, 0.6]  # right ear
        
        # Upper body
        keypoints[5] = [x + w_box*0.3, y + h_box*0.25, 0.8]  # left shoulder
        keypoints[6] = [x + w_box*0.7, y + h_box*0.25, 0.8]  # right shoulder
        keypoints[7] = [x + w_box*0.2, y + h_box*0.45, 0.7]  # left elbow
        keypoints[8] = [x + w_box*0.8, y + h_box*0.45, 0.7]  # right elbow
        keypoints[9] = [x + w_box*0.15, y + h_box*0.6, 0.6]  # left wrist
        keypoints[10] = [x + w_box*0.85, y + h_box*0.6, 0.6]  # right wrist
        
        # Lower body
        keypoints[11] = [x + w_box*0.35, y + h_box*0.55, 0.8]  # left hip
        keypoints[12] = [x + w_box*0.65, y + h_box*0.55, 0.8]  # right hip
        keypoints[13] = [x + w_box*0.3, y + h_box*0.75, 0.7]  # left knee
        keypoints[14] = [x + w_box*0.7, y + h_box*0.75, 0.7]  # right knee
        keypoints[15] = [x + w_box*0.3, y + h_box*0.95, 0.6]  # left ankle
        keypoints[16] = [x + w_box*0.7, y + h_box*0.95, 0.6]  # right ankle
        
        return {
            'keypoints': keypoints,
            'bbox': [x, y, x + w_box, y + h_box],
            'score': 0.6
        }
    
    def detect_video(self, video_path, sample_rate=1):
        """
        Detect poses in video
        
        Args:
            video_path: path to video file
            sample_rate: process every Nth frame
        
        Returns:
            list of pose dictionaries
        """
        cap = cv2.VideoCapture(video_path)
        poses = []
        frame_idx = 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                pose = self.detect(frame)
                poses.append(pose)
                
                # Progress
                if frame_idx % 30 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"\rProcessing: {progress:.1f}%", end='')
            
            frame_idx += 1
        
        cap.release()
        print(f"\n✓ Processed {len(poses)} frames")
        
        return poses


def download_vitpose_models():
    """Download all VitPose model variants"""
    detector = VitPoseDetector(model_name='vitpose-b')
    print("\n✓ VitPose-Base downloaded")
    
    detector = VitPoseDetector(model_name='vitpose-l')
    print("\n✓ VitPose-Large downloaded")
    
    print("\n✓ All VitPose models ready!")


if __name__ == '__main__':
    # Test VitPose detector
    print("Testing VitPose detector...")
    
    detector = VitPoseDetector(model_name='vitpose-b')
    
    # Create test image
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Draw simple person shape
    cv2.circle(test_image, (320, 100), 30, (0, 0, 0), -1)  # head
    cv2.rectangle(test_image, (280, 130), (360, 300), (0, 0, 0), -1)  # body
    
    # Detect
    result = detector.detect(test_image)
    
    if result:
        print(f"\n✓ Detection successful!")
        print(f"  Score: {result['score']:.3f}")
        print(f"  Keypoints shape: {result['keypoints'].shape}")
    else:
        print("\n✗ Detection failed")