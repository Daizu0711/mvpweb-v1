from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import os
import json
import requests
from datetime import datetime
import traceback
from pose_analyzer import PoseAnalyzer, PoseComparator

app = Flask(__name__, static_folder='static')
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
DATA_FOLDER = 'data'
DATA_FILE = os.path.join(DATA_FOLDER, 'training_records.json')
USERS_FILE = os.path.join(DATA_FOLDER, 'users.json')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

analyzer = None

# --- User data management ---

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠ Failed to load users: {e}")
    return {}

def save_users(users):
    try:
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠ Failed to save users: {e}")

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

def calculate_body_ratios(pose):
    """Calculate limb-length ratios relative to torso height from a pose."""
    if pose is None or 'keypoints' not in pose:
        return None
    kp = pose['keypoints']
    # Torso height = distance from shoulder_center to hip_center
    if kp[5][2] < 0.3 or kp[6][2] < 0.3 or kp[11][2] < 0.3 or kp[12][2] < 0.3:
        return None
    shoulder_center = (np.array(kp[5][:2]) + np.array(kp[6][:2])) / 2
    hip_center = (np.array(kp[11][:2]) + np.array(kp[12][:2])) / 2
    torso_height = np.linalg.norm(shoulder_center - hip_center)
    if torso_height < 1:
        return None

    ratios = {}
    for seg_name, (i, j) in LIMB_SEGMENTS.items():
        if kp[i][2] >= 0.3 and kp[j][2] >= 0.3:
            length = np.linalg.norm(np.array(kp[i][:2]) - np.array(kp[j][:2]))
            ratios[seg_name] = length / torso_height
        else:
            ratios[seg_name] = None
    return ratios

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

def detect_deficiency(current_ratios, registered_ratios, threshold=0.5):
    """Detect limb deficiency by comparing current ratios against registered T-pose ratios.

    If a limb ratio is less than (1 - threshold) of the registered ratio,
    it is flagged as deficient (potential missing/occluded limb).
    """
    deficiencies = []
    if not current_ratios or not registered_ratios:
        return deficiencies
    for seg_name, reg_ratio in registered_ratios.items():
        if reg_ratio is None:
            continue
        cur_ratio = current_ratios.get(seg_name)
        if cur_ratio is None:
            # Not detected at all -> deficiency
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

def load_training_records():
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception as e:
        print(f"⚠ Failed to load training records: {e}")
    return []

def save_training_records(records):
    try:
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠ Failed to save training records: {e}")

def append_training_record(record):
    records = load_training_records()
    records.append(record)
    save_training_records(records)

def aggregate_joint_scores(records):
    sums = {}
    counts = {}
    for rec in records:
        for joint, score in rec.get('joint_scores', {}).items():
            sums[joint] = sums.get(joint, 0.0) + float(score)
            counts[joint] = counts.get(joint, 0) + 1
    return {joint: sums[joint] / counts[joint] for joint in sums}

def average_value(records, key):
    values = [float(r.get(key, 0)) for r in records if r.get(key) is not None]
    if not values:
        return 0.0
    return sum(values) / len(values)

def generate_training_summary(records):
    self_records = [r for r in records if r.get('compare_mode') == 'self_vs_self']
    improved_records = [r for r in self_records if r.get('self_improved') is True]
    not_improved_records = [r for r in self_records if r.get('self_improved') is False]

    summary = {
        'total_records': len(records),
        'self_vs_self_records': len(self_records),
        'improved_count': len(improved_records),
        'not_improved_count': len(not_improved_records),
        'averages': {
            'improved': {
                'overall_score': average_value(improved_records, 'overall_score'),
                'temporal_alignment': average_value(improved_records, 'temporal_alignment'),
                'joint_scores': aggregate_joint_scores(improved_records)
            },
            'not_improved': {
                'overall_score': average_value(not_improved_records, 'overall_score'),
                'temporal_alignment': average_value(not_improved_records, 'temporal_alignment'),
                'joint_scores': aggregate_joint_scores(not_improved_records)
            }
        }
    }

    joint_differences = {}
    for joint, score in summary['averages']['improved']['joint_scores'].items():
        if joint in summary['averages']['not_improved']['joint_scores']:
            joint_differences[joint] = (
                score - summary['averages']['not_improved']['joint_scores'][joint]
            )

    summary['differences'] = {
        'joint_differences': joint_differences,
        'top_positive': sorted(joint_differences.items(), key=lambda x: x[1], reverse=True)[:5],
        'top_negative': sorted(joint_differences.items(), key=lambda x: x[1])[:5]
    }

    return summary

def init_analyzer(use_vitpose=True, model_variant='vitpose-b'):
    global analyzer
    if analyzer is None:
        analyzer = PoseAnalyzer(
            use_vitpose=use_vitpose,
            model_variant=model_variant
        )
    return analyzer

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video = request.files['video']
        video_type = request.form.get('type', 'reference')
        
        if video.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Save video
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{video_type}_{timestamp}_{video.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        video.save(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath
        })
    
    except Exception as e:
        print(f"Upload error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_videos():
    try:
        data = request.json
        reference_path = data.get('reference_video')
        comparison_path = data.get('comparison_video')
        use_3d = data.get('use_3d', False)
        use_vitpose = data.get('use_vitpose', True)
        model_variant = data.get('model_variant', 'vitpose-b')
        compare_mode = data.get('compare_mode', 'pro_vs_self')
        self_improved = data.get('self_improved', None)
        username = data.get('username', '').strip()
        
        if not reference_path or not comparison_path:
            return jsonify({'error': 'Both videos are required'}), 400

        if compare_mode == 'self_vs_self' and self_improved is None:
            return jsonify({'error': 'Self vs Self mode requires self_improved flag'}), 400
        
        # Initialize analyzer with VitPose options
        pose_analyzer = init_analyzer(
            use_vitpose=use_vitpose,
            model_variant=model_variant
        )
        
        # Extract poses from both videos
        print("Extracting poses from reference video...")
        reference_poses = pose_analyzer.extract_poses_from_video(reference_path)
        
        print("Extracting poses from comparison video...")
        comparison_poses = pose_analyzer.extract_poses_from_video(comparison_path)
        
        if not reference_poses or not comparison_poses:
            return jsonify({'error': 'Failed to extract poses from videos'}), 500
        
        # 3D pose estimation if requested
        reference_poses_3d = None
        comparison_poses_3d = None
        visualization_3d_paths = {}
        
        if use_3d:
            print("Performing 3D pose estimation with PoseFormer...")
            from pose_analyzer import Pose3DEstimator
            
            estimator_3d = Pose3DEstimator()
            
            # Lift to 3D
            reference_poses_3d = estimator_3d.lift_to_3d(reference_poses)
            comparison_poses_3d = estimator_3d.lift_to_3d(comparison_poses)
            
            # Create 3D visualizations
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Individual 3D visualizations
            ref_3d_path = os.path.join(OUTPUT_FOLDER, f'reference_3d_{timestamp}.html')
            comp_3d_path = os.path.join(OUTPUT_FOLDER, f'comparison_3d_{timestamp}.html')
            
            estimator_3d.create_3d_visualization(
                reference_poses_3d, 
                ref_3d_path,
                viz_type='interactive'
            )
            estimator_3d.create_3d_visualization(
                comparison_poses_3d,
                comp_3d_path,
                viz_type='interactive'
            )
            
            # 3D comparison visualization
            from pose_3d_visualizer import Pose3DVisualizer, calculate_3d_metrics
            
            visualizer_3d = Pose3DVisualizer(skeleton_type='coco')
            comparison_3d_path = os.path.join(OUTPUT_FOLDER, f'comparison_3d_{timestamp}.html')
            visualizer_3d.create_comparison_3d_html(
                reference_poses_3d,
                comparison_poses_3d,
                comparison_3d_path
            )
            
            # Export 3D data to JSON
            json_3d_path = os.path.join(OUTPUT_FOLDER, f'poses_3d_{timestamp}.json')
            visualizer_3d.export_3d_json(
                comparison_poses_3d,
                json_3d_path
            )
            
            # Calculate 3D metrics
            metrics_3d = calculate_3d_metrics(reference_poses_3d, comparison_poses_3d)
            
            visualization_3d_paths = {
                'reference_3d': os.path.basename(ref_3d_path),
                'comparison_3d': os.path.basename(comp_3d_path),
                'comparison_3d_sidebyside': os.path.basename(comparison_3d_path),
                'json_3d': os.path.basename(json_3d_path),
                'metrics_3d': metrics_3d
            }
            
            print("3D visualization complete")
        
        # Compare poses (2D)
        print("Comparing poses...")
        comparator = PoseComparator()
        comparison_result = comparator.compare_pose_sequences(
            reference_poses, 
            comparison_poses,
            use_3d=use_3d
        )

        # Save training record
        try:
            record = {
                'timestamp': datetime.now().isoformat(),
                'compare_mode': compare_mode,
                'self_improved': self_improved if compare_mode == 'self_vs_self' else None,
                'overall_score': comparison_result['overall_score'],
                'joint_scores': comparison_result['joint_scores'],
                'temporal_alignment': comparison_result['temporal_alignment'],
                'use_3d': use_3d
            }

            if use_3d and visualization_3d_paths and 'metrics_3d' in visualization_3d_paths:
                record['metrics_3d'] = visualization_3d_paths['metrics_3d']

            append_training_record(record)
        except Exception as e:
            print(f"⚠ Failed to save training record: {e}")
        
        # Generate analysis with Ollama
        print("Generating analysis with Ollama...")
        analysis = generate_ollama_analysis(comparison_result, use_3d, visualization_3d_paths)
        
        # Create visualization with overlays
        print("Creating comparison video with pose overlays...")
        output_filename = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Also create individual overlay videos
        ref_overlay_filename = f"reference_overlay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        ref_overlay_path = os.path.join(OUTPUT_FOLDER, ref_overlay_filename)
        
        comp_overlay_filename = f"comparison_overlay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        comp_overlay_path = os.path.join(OUTPUT_FOLDER, comp_overlay_filename)
        
        from video_overlay import (
            create_comparison_overlay_video,
            create_overlay_video
        )
        
        try:
            # Create individual overlays
            print("Creating reference video overlay...")
            create_overlay_video(
                reference_path,
                reference_poses,
                {name: 100 for name in comparison_result['joint_scores'].keys()},
                ref_overlay_path,
                overall_score=100,
                color_mode='default',
                show_labels=False,
                show_confidence=False,
                show_legend=False
            )
            
            print("Creating comparison video overlay...")
            create_overlay_video(
                comparison_path,
                comparison_poses,
                comparison_result['joint_scores'],
                comp_overlay_path,
                overall_score=comparison_result['overall_score'],
                color_mode='score',
                show_labels=True,
                show_confidence=True,
                show_legend=True
            )
            
            # Create side-by-side comparison
            print("Creating side-by-side comparison...")
            create_comparison_overlay_video(
                reference_path,
                comparison_path,
                reference_poses,
                comparison_poses,
                {name: 100 for name in comparison_result['joint_scores'].keys()},
                comparison_result['joint_scores'],
                output_path,
                ref_overall_score=100,
                comp_overall_score=comparison_result['overall_score']
            )
            
        except Exception as e:
            print(f"⚠ Warning: Video overlay creation failed: {e}")
            print("Continuing with analysis results...")
            # Set to None if creation failed
            if not os.path.exists(ref_overlay_path):
                ref_overlay_filename = None
            if not os.path.exists(comp_overlay_path):
                comp_overlay_filename = None
            if not os.path.exists(output_path):
                output_filename = None
        
        response_data = {
            'success': True,
            'score': comparison_result['overall_score'],
            'joint_scores': comparison_result['joint_scores'],
            'temporal_alignment': comparison_result['temporal_alignment'],
            'analysis': analysis,
            'output_video': output_filename,
            'reference_overlay': ref_overlay_filename,
            'comparison_overlay': comp_overlay_filename,
            'frame_scores': comparison_result.get('frame_scores', []),
            'use_3d': use_3d,
            'username': username
        }

        # Add 3D data if available
        if use_3d and visualization_3d_paths:
            response_data['visualization_3d'] = visualization_3d_paths

        # Deficiency detection if user has registered T-pose
        if username:
            users = load_users()
            user_data = users.get(username)
            if user_data and user_data.get('body_ratios'):
                registered_ratios = user_data['body_ratios']
                # Calculate average ratios from comparison video frames
                frame_ratios_list = []
                for pose in comparison_poses:
                    if pose is not None:
                        fr = calculate_body_ratios(pose)
                        if fr:
                            frame_ratios_list.append(fr)
                if frame_ratios_list:
                    # Average ratios across all frames
                    avg_ratios = {}
                    for seg_name in LIMB_SEGMENTS:
                        vals = [fr[seg_name] for fr in frame_ratios_list if fr.get(seg_name) is not None]
                        avg_ratios[seg_name] = sum(vals) / len(vals) if vals else None
                    deficiencies = detect_deficiency(avg_ratios, registered_ratios)
                    if deficiencies:
                        response_data['deficiencies'] = deficiencies

        return jsonify(response_data)
    
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def generate_ollama_analysis(comparison_result, use_3d=False, viz_3d_paths=None):
    """Generate detailed analysis using local Ollama"""
    try:
        # Prepare prompt for Ollama
        prompt = f"""あなたはプロのスポーツコーチです。2つの動作を比較分析した結果を基に、改善点を日本語で提案してください。

分析結果:
- 全体スコア: {comparison_result['overall_score']:.2f}/100
- 各関節のスコア:
"""
        for joint, score in comparison_result['joint_scores'].items():
            prompt += f"  - {joint}: {score:.2f}\n"
        
        prompt += f"\n- 時間的整合性: {comparison_result['temporal_alignment']:.2f}\n"
        
        if use_3d and viz_3d_paths and 'metrics_3d' in viz_3d_paths:
            metrics_3d = viz_3d_paths['metrics_3d']
            prompt += f"\n3D分析結果:\n"
            prompt += f"- 平均3D誤差 (MPJPE): {metrics_3d['mean_mpjpe']:.4f}\n"
            prompt += f"- 3D関節誤差トップ3:\n"
            sorted_3d_errors = sorted(
                metrics_3d['mean_joint_errors'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            for joint, error in sorted_3d_errors:
                prompt += f"  - {joint}: {error:.4f}\n"
        
        prompt += "\n以下の形式で回答してください:\n"
        prompt += "1. 全体的な評価\n"
        prompt += "2. 良い点（3つ）\n"
        prompt += "3. 改善が必要な点（3つ）\n"
        if use_3d:
            prompt += "4. 3D分析からわかる深度方向の改善点\n"
            prompt += "5. 具体的な修正アドバイス\n"
        else:
            prompt += "4. 具体的な修正アドバイス\n"
        
        # Call Ollama API with longer timeout
        print("Connecting to Ollama...")
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama3.2',
                'prompt': prompt,
                'stream': False
            },
            timeout=180  # Increase timeout to 3 minutes
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Ollama analysis complete")
            return result.get('response', 'Analysis generated')
        else:
            print(f"⚠ Ollama returned status {response.status_code}")
            return generate_fallback_analysis(comparison_result, use_3d, viz_3d_paths)
    
    except requests.exceptions.Timeout:
        print("⚠ Ollama request timed out - using fallback analysis")
        return generate_fallback_analysis(comparison_result, use_3d, viz_3d_paths)
    
    except requests.exceptions.ConnectionError:
        print("⚠ Cannot connect to Ollama - is it running? Using fallback analysis")
        return generate_fallback_analysis(comparison_result, use_3d, viz_3d_paths)
    
    except Exception as e:
        print(f"⚠ Ollama error: {str(e)} - using fallback analysis")
        return generate_fallback_analysis(comparison_result, use_3d, viz_3d_paths)

def generate_fallback_analysis(comparison_result, use_3d, viz_3d_paths):
    """Fallback analysis without Ollama"""
    analysis = "=" * 50 + "\n"
    analysis += "📊 分析結果レポート\n"
    analysis += "=" * 50 + "\n\n"
    
    # Overall score
    score = comparison_result['overall_score']
    analysis += f"【総合評価】\n"
    analysis += f"スコア: {score:.1f}/100\n"
    
    if score >= 80:
        analysis += "評価: 優秀 - 素晴らしいフォームです！\n\n"
    elif score >= 60:
        analysis += "評価: 良好 - あと一歩で完璧です\n\n"
    elif score >= 40:
        analysis += "評価: 普通 - 改善の余地があります\n\n"
    else:
        analysis += "評価: 要改善 - 基本から見直しましょう\n\n"
    
    # Top performing joints
    sorted_joints = sorted(
        comparison_result['joint_scores'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    analysis += "【良好な部位 TOP 3】\n"
    for i, (joint, score) in enumerate(sorted_joints[:3], 1):
        joint_ja = get_joint_japanese_name(joint)
        analysis += f"{i}. {joint_ja}: {score:.1f}点 ✓\n"
    
    analysis += "\n【改善が必要な部位 TOP 3】\n"
    for i, (joint, score) in enumerate(sorted_joints[-3:], 1):
        joint_ja = get_joint_japanese_name(joint)
        analysis += f"{i}. {joint_ja}: {score:.1f}点 ✗\n"
        
        # Add specific advice for low-scoring joints
        if score < 40:
            analysis += f"   → 重点的な改善が必要です\n"
        elif score < 60:
            analysis += f"   → フォームの見直しを推奨\n"
    
    # Temporal alignment
    temporal = comparison_result['temporal_alignment']
    analysis += f"\n【動作のタイミング】\n"
    analysis += f"整合性: {temporal:.1f}/100\n"
    if temporal >= 80:
        analysis += "動作のリズムが良好です\n"
    elif temporal >= 60:
        analysis += "タイミングに若干のずれがあります\n"
    else:
        analysis += "動作速度の調整が必要です\n"
    
    # 3D analysis if available
    if use_3d and viz_3d_paths and 'metrics_3d' in viz_3d_paths:
        metrics_3d = viz_3d_paths['metrics_3d']
        analysis += f"\n【3D分析】\n"
        analysis += f"平均3D誤差 (MPJPE): {metrics_3d['mean_mpjpe']:.4f}\n"
        
        analysis += "\n深度方向の誤差が大きい部位:\n"
        sorted_3d = sorted(
            metrics_3d['mean_joint_errors'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        for joint, error in sorted_3d:
            joint_ja = get_joint_japanese_name(joint)
            analysis += f"- {joint_ja}: {error:.4f}\n"
    
    # General advice
    analysis += "\n【改善アドバイス】\n"
    worst_joint = sorted_joints[-1][0]
    worst_score = sorted_joints[-1][1]
    worst_ja = get_joint_japanese_name(worst_joint)
    
    analysis += f"1. 最優先: {worst_ja}の修正 (現在{worst_score:.1f}点)\n"
    analysis += f"2. 参照動画を繰り返し確認し、正しい動きを理解する\n"
    analysis += f"3. ゆっくりとした動作で練習し、徐々にスピードアップ\n"
    analysis += f"4. 定期的に撮影して進捗を確認\n"
    
    analysis += "\n" + "=" * 50 + "\n"
    analysis += "※ より詳細な分析にはOllamaを起動してください\n"
    analysis += "  コマンド: ollama serve\n"
    analysis += "=" * 50 + "\n"
    
    return analysis

def get_joint_japanese_name(joint_name):
    """Convert joint name to Japanese"""
    names = {
        'nose': '鼻',
        'left_eye': '左目',
        'right_eye': '右目',
        'left_ear': '左耳',
        'right_ear': '右耳',
        'left_shoulder': '左肩',
        'right_shoulder': '右肩',
        'left_elbow': '左肘',
        'right_elbow': '右肘',
        'left_wrist': '左手首',
        'right_wrist': '右手首',
        'left_hip': '左腰',
        'right_hip': '右腰',
        'left_knee': '左膝',
        'right_knee': '右膝',
        'left_ankle': '左足首',
        'right_ankle': '右足首'
    }
    return names.get(joint_name, joint_name)

def create_comparison_video(ref_path, comp_path, ref_poses, comp_poses, output_path):
    """Create side-by-side comparison video with pose overlay"""
    cap_ref = cv2.VideoCapture(ref_path)
    cap_comp = cv2.VideoCapture(comp_path)
    
    fps = int(cap_ref.get(cv2.CAP_PROP_FPS))
    width = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    
    frame_idx = 0
    while True:
        ret_ref, frame_ref = cap_ref.read()
        ret_comp, frame_comp = cap_comp.read()
        
        if not ret_ref or not ret_comp:
            break
        
        # Draw poses
        if frame_idx < len(ref_poses) and ref_poses[frame_idx] is not None:
            frame_ref = draw_pose(frame_ref, ref_poses[frame_idx], color=(0, 255, 0))
        
        if frame_idx < len(comp_poses) and comp_poses[frame_idx] is not None:
            frame_comp = draw_pose(frame_comp, comp_poses[frame_idx], color=(255, 0, 0))
        
        # Combine frames
        combined = np.hstack([frame_ref, frame_comp])
        out.write(combined)
        frame_idx += 1
    
    cap_ref.release()
    cap_comp.release()
    out.release()

def draw_pose(image, pose_data, color=(0, 255, 0)):
    """Draw pose keypoints and skeleton on image"""
    keypoints = pose_data['keypoints']
    
    # COCO skeleton connections
    skeleton = [
        [0, 1], [0, 2], [1, 3], [2, 4],  # Head
        [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # Arms
        [5, 11], [6, 12], [11, 12],  # Torso
        [11, 13], [13, 15], [12, 14], [14, 16]  # Legs
    ]
    
    # Draw skeleton
    for connection in skeleton:
        pt1_idx, pt2_idx = connection
        if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
            pt1 = tuple(map(int, keypoints[pt1_idx][:2]))
            pt2 = tuple(map(int, keypoints[pt2_idx][:2]))
            
            if keypoints[pt1_idx][2] > 0.3 and keypoints[pt2_idx][2] > 0.3:
                cv2.line(image, pt1, pt2, color, 2)
    
    # Draw keypoints
    for kp in keypoints:
        if kp[2] > 0.3:  # confidence threshold
            pt = tuple(map(int, kp[:2]))
            cv2.circle(image, pt, 4, color, -1)
    
    return image

@app.route('/api/outputs/<filename>')
def get_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/api/training/summary', methods=['GET'])
def get_training_summary():
    records = load_training_records()
    if not records:
        return jsonify({
            'success': False,
            'message': 'まだ記録がありません。分析後にデータが保存されます。'
        }), 200

    summary = generate_training_summary(records)

    if summary['improved_count'] == 0 or summary['not_improved_count'] == 0:
        return jsonify({
            'success': False,
            'summary': summary,
            'message': '上達・未上達の両方のデータが必要です。'
        }), 200

    # Build human-readable message
    msg = "".join([
        "📊 上達/未上達の違い（自分×自分）\n",
        f"- 上達あり: {summary['improved_count']}件\n",
        f"- 上達なし: {summary['not_improved_count']}件\n\n",
        f"【平均スコア】\n",
        f"上達あり: {summary['averages']['improved']['overall_score']:.2f}\n",
        f"上達なし: {summary['averages']['not_improved']['overall_score']:.2f}\n\n",
        f"【時間的整合性(平均)】\n",
        f"上達あり: {summary['averages']['improved']['temporal_alignment']:.2f}\n",
        f"上達なし: {summary['averages']['not_improved']['temporal_alignment']:.2f}\n\n",
        "【差分が大きい関節(上達あり - 上達なし)】\n"
    ])

    top_positive = summary['differences']['top_positive']
    for joint, diff in top_positive:
        joint_ja = get_joint_japanese_name(joint)
        msg += f"- {joint_ja}: {diff:+.2f}\n"

    return jsonify({
        'success': True,
        'summary': summary,
        'message': msg
    }), 200

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'analyzer': 'ready'})

# --- User & T-pose endpoints ---

@app.route('/api/users', methods=['GET'])
def get_users():
    users = load_users()
    return jsonify({'success': True, 'users': list(users.keys())})

@app.route('/api/user/<username>', methods=['GET'])
def get_user(username):
    users = load_users()
    if username not in users:
        return jsonify({'success': False, 'error': 'User not found'}), 404
    return jsonify({'success': True, 'user': users[username]})

@app.route('/api/user/tpose', methods=['POST'])
def register_tpose():
    """Upload T-pose photo and calculate body ratios for a user."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        username = request.form.get('username', '').strip()
        if not username:
            return jsonify({'error': 'Username is required'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ext = os.path.splitext(image_file.filename)[1] or '.jpg'
        img_filename = f"tpose_{username}_{timestamp}{ext}"
        img_path = os.path.join(UPLOAD_FOLDER, img_filename)
        image_file.save(img_path)

        # Read image and detect pose
        pose_analyzer = init_analyzer()
        frame = cv2.imread(img_path)
        if frame is None:
            return jsonify({'error': 'Failed to read image'}), 400

        pose = pose_analyzer.detect_pose(frame)
        if pose is None:
            return jsonify({'error': 'Failed to detect pose in image'}), 400

        # Convert numpy arrays to lists for JSON serialization
        pose_serializable = {
            'keypoints': pose['keypoints'].tolist() if hasattr(pose['keypoints'], 'tolist') else pose['keypoints']
        }

        # Calculate body ratios
        ratios = calculate_body_ratios(pose)
        if ratios is None:
            return jsonify({'error': 'Could not calculate body ratios. Ensure full body is visible in T-pose.'}), 400

        # Save user data
        users = load_users()
        users[username] = {
            'registered_at': datetime.now().isoformat(),
            'tpose_image': img_filename,
            'body_ratios': ratios,
            'tpose_keypoints': pose_serializable['keypoints']
        }
        save_users(users)

        # Build ratio display
        ratio_display = {}
        for seg, val in ratios.items():
            ja_name = LIMB_SEGMENT_NAMES_JA.get(seg, seg)
            ratio_display[ja_name] = round(val, 3) if val is not None else None

        return jsonify({
            'success': True,
            'username': username,
            'body_ratios': ratios,
            'ratio_display': ratio_display,
            'message': f'{username} のT-pose比率を登録しました'
        })

    except Exception as e:
        print(f"T-pose registration error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Initializing pose analyzer...")
    init_analyzer()
    print("Server starting on http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)