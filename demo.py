#!/usr/bin/env python3
"""
デモスクリプト - システムの基本機能をテスト
"""

import numpy as np
import cv2
import os
from pose_analyzer import PoseAnalyzer, PoseComparator

def create_sample_video(filename, frames=30):
    """サンプル動画を生成"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 10.0, (640, 480))
    
    for i in range(frames):
        # 白い背景
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # 簡単な図形を描画（人物のシミュレーション）
        # 頭
        cv2.circle(frame, (320, 100), 30, (0, 0, 0), -1)
        
        # 体
        cv2.line(frame, (320, 130), (320, 300), (0, 0, 0), 20)
        
        # 腕（アニメーション）
        angle = i * 12  # 度数
        arm_length = 80
        left_arm_x = int(320 - arm_length * np.cos(np.radians(angle)))
        left_arm_y = int(200 + arm_length * np.sin(np.radians(angle)))
        right_arm_x = int(320 + arm_length * np.cos(np.radians(angle)))
        right_arm_y = int(200 + arm_length * np.sin(np.radians(angle)))
        
        cv2.line(frame, (320, 200), (left_arm_x, left_arm_y), (0, 0, 255), 10)
        cv2.line(frame, (320, 200), (right_arm_x, right_arm_y), (0, 0, 255), 10)
        
        # 脚
        cv2.line(frame, (320, 300), (280, 450), (0, 0, 0), 15)
        cv2.line(frame, (320, 300), (360, 450), (0, 0, 0), 15)
        
        out.write(frame)
    
    out.release()
    print(f"✓ サンプル動画を作成: {filename}")

def test_pose_extraction():
    """姿勢推定のテスト"""
    print("\n=== 姿勢推定テスト ===")
    
    # サンプル動画を作成
    os.makedirs('test_videos', exist_ok=True)
    ref_video = 'test_videos/reference.mp4'
    comp_video = 'test_videos/comparison.mp4'
    
    create_sample_video(ref_video, 30)
    create_sample_video(comp_video, 30)
    
    # 姿勢推定
    analyzer = PoseAnalyzer()
    
    print("参照動画から姿勢を抽出中...")
    ref_poses = analyzer.extract_poses_from_video(ref_video)
    print(f"✓ {len(ref_poses)} フレームの姿勢を抽出")
    
    print("比較動画から姿勢を抽出中...")
    comp_poses = analyzer.extract_poses_from_video(comp_video)
    print(f"✓ {len(comp_poses)} フレームの姿勢を抽出")
    
    return ref_poses, comp_poses

def test_pose_comparison(ref_poses, comp_poses):
    """姿勢比較のテスト"""
    print("\n=== 姿勢比較テスト ===")
    
    comparator = PoseComparator()
    
    print("姿勢を比較中...")
    result = comparator.compare_pose_sequences(ref_poses, comp_poses)
    
    print(f"\n総合スコア: {result['overall_score']:.2f}/100")
    print(f"時間的整合性: {result['temporal_alignment']:.2f}/100")
    
    print("\n部位別スコア (上位5つ):")
    sorted_joints = sorted(
        result['joint_scores'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    for joint, score in sorted_joints[:5]:
        print(f"  {joint}: {score:.2f}")
    
    print("\n改善が必要な部位 (下位5つ):")
    for joint, score in sorted_joints[-5:]:
        print(f"  {joint}: {score:.2f}")
    
    return result

def test_normalization():
    """正規化処理のテスト"""
    print("\n=== 正規化処理テスト ===")
    
    # サンプル姿勢データ
    sample_pose = {
        'keypoints': np.array([
            [320, 100, 0.9],  # nose
            [300, 95, 0.85],  # left_eye
            [340, 95, 0.85],  # right_eye
            [290, 100, 0.8],  # left_ear
            [350, 100, 0.8],  # right_ear
            [280, 150, 0.9],  # left_shoulder
            [360, 150, 0.9],  # right_shoulder
            [250, 220, 0.85], # left_elbow
            [390, 220, 0.85], # right_elbow
            [240, 280, 0.8],  # left_wrist
            [400, 280, 0.8],  # right_wrist
            [290, 250, 0.9],  # left_hip
            [350, 250, 0.9],  # right_hip
            [280, 350, 0.85], # left_knee
            [360, 350, 0.85], # right_knee
            [280, 450, 0.8],  # left_ankle
            [360, 450, 0.8],  # right_ankle
        ]),
        'bbox': [0, 0, 640, 480]
    }
    
    comparator = PoseComparator()
    normalized = comparator.normalize_pose(sample_pose)
    
    print("元の姿勢データ:")
    print(f"  重心: ({np.mean(sample_pose['keypoints'][:, 0]):.2f}, "
          f"{np.mean(sample_pose['keypoints'][:, 1]):.2f})")
    
    print("\n正規化後の姿勢データ:")
    print(f"  重心: ({np.mean(normalized['keypoints'][:, 0]):.2f}, "
          f"{np.mean(normalized['keypoints'][:, 1]):.2f})")
    print(f"  スケール: {normalized['scale']:.2f}")
    
    print("✓ 正規化処理が正常に動作しました")

def test_angle_calculation():
    """関節角度計算のテスト"""
    print("\n=== 関節角度計算テスト ===")
    
    sample_pose = {
        'keypoints': np.array([
            [320, 100, 0.9],  # 0: nose
            [300, 95, 0.85],  # 1: left_eye
            [340, 95, 0.85],  # 2: right_eye
            [290, 100, 0.8],  # 3: left_ear
            [350, 100, 0.8],  # 4: right_ear
            [280, 150, 0.9],  # 5: left_shoulder
            [360, 150, 0.9],  # 6: right_shoulder
            [250, 220, 0.85], # 7: left_elbow
            [390, 220, 0.85], # 8: right_elbow
            [240, 280, 0.8],  # 9: left_wrist
            [400, 280, 0.8],  # 10: right_wrist
            [290, 250, 0.9],  # 11: left_hip
            [350, 250, 0.9],  # 12: right_hip
            [280, 350, 0.85], # 13: left_knee
            [360, 350, 0.85], # 14: right_knee
            [280, 450, 0.8],  # 15: left_ankle
            [360, 450, 0.8],  # 16: right_ankle
        ])
    }
    
    comparator = PoseComparator()
    angles = comparator.calculate_joint_angles(sample_pose)
    
    print("計算された関節角度:")
    for joint, angle in angles.items():
        print(f"  {joint}: {angle:.2f}°")
    
    print("✓ 関節角度計算が正常に動作しました")

def main():
    """メインテスト実行"""
    print("="*50)
    print("動画フォーム比較分析システム - デモテスト")
    print("="*50)
    
    try:
        # 1. 正規化テスト
        test_normalization()
        
        # 2. 関節角度計算テスト
        test_angle_calculation()
        
        # 3. 姿勢推定テスト
        ref_poses, comp_poses = test_pose_extraction()
        
        # 4. 姿勢比較テスト
        result = test_pose_comparison(ref_poses, comp_poses)
        
        print("\n" + "="*50)
        print("✓ すべてのテストが完了しました！")
        print("="*50)
        print("\n次のステップ:")
        print("1. python app.py でサーバーを起動")
        print("2. http://localhost:5001 にアクセス")
        print("3. 実際の動画をアップロードして分析")
        
    except Exception as e:
        print(f"\n✗ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
