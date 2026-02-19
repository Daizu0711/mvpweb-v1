# 動画フォーム比較分析システム - 設定ファイル

# サーバー設定
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 5001
DEBUG = True

# Ollama設定
OLLAMA_URL = 'http://localhost:11434'
OLLAMA_MODEL = 'llama3.2'  # 使用するモデル名
OLLAMA_TIMEOUT = 60  # タイムアウト（秒）

# Render API -> Colab 推論設定
INFERENCE_SERVER_URL = ''  # 例: https://xxxx.ngrok-free.app
INFERENCE_TIMEOUT = 240
INFERENCE_MODE_DEFAULT = 'auto'  # auto / local / remote

# 姿勢推定設定
POSE_CONFIDENCE_THRESHOLD = 0.3  # 信頼度の閾値
SAMPLE_RATE = 1  # フレームサンプリングレート（1=全フレーム）
USE_GPU = True  # GPU使用の有無（自動検出）

# スコアリング設定
SCORE_WEIGHTS = {
    'frame_distance_coefficient': 50,  # フレーム距離係数
    'joint_distance_coefficient': 100,  # 関節距離係数
    'temporal_alignment_coefficient': 10  # 時間的整合性係数
}

# 関節グループの重要度（合計100）
JOINT_IMPORTANCE = {
    'head': 10,
    'torso': 20,
    'left_arm': 15,
    'right_arm': 15,
    'left_leg': 20,
    'right_leg': 20
}

# ファイル設定
MAX_VIDEO_SIZE_MB = 100  # 最大ファイルサイズ（MB）
ALLOWED_EXTENSIONS = ['mp4', 'avi', 'mov', 'mkv']
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'

# VitPose設定
VITPOSE_CONFIG = {
    'model_name': 'vitpose_base',
    'checkpoint': 'models/vitpose.pth',
    'config_file': 'configs/vitpose_base_coco.py',
    'input_size': (256, 192),  # (height, width)
}

# PoseFormer設定（3D姿勢推定）
POSEFORMER_CONFIG = {
    'enabled': False,  # デフォルトで無効
    'model_path': 'models/poseformer.pth',
    'num_joints': 17,
    'in_chans': 2,
    'embed_dim': 512
}

# 動画出力設定
VIDEO_OUTPUT = {
    'fps': 30,
    'codec': 'mp4v',
    'quality': 90,
    'show_skeleton': True,
    'show_confidence': True,
    'overlay_alpha': 0.6
}

# DTW（Dynamic Time Warping）設定
DTW_CONFIG = {
    'window_size': None,  # None = 制限なし
    'distance_metric': 'euclidean',  # 'euclidean', 'manhattan', 'cosine'
}

# 正規化設定
NORMALIZATION = {
    'method': 'scale_and_translate',  # 'scale_and_translate', 'procrustes'
    'reference_points': ['left_hip', 'right_hip'],  # 中心点の計算に使用
    'scale_points': ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
}

# ログ設定
LOGGING = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'app.log'
}

# キャッシュ設定
CACHE = {
    'enabled': True,
    'directory': '.cache',
    'max_size_mb': 500
}

# 分析プロンプトテンプレート（Ollama用）
ANALYSIS_PROMPT_TEMPLATE = """あなたはプロのスポーツコーチです。2つの動作を比較分析した結果を基に、改善点を日本語で提案してください。

分析結果:
- 全体スコア: {overall_score:.2f}/100
- 各関節のスコア:
{joint_scores}

- 時間的整合性: {temporal_alignment:.2f}

以下の形式で回答してください:
1. 全体的な評価（2-3文）
2. 良い点（3つ、それぞれ1文）
3. 改善が必要な点（3つ、それぞれ1文）
4. 具体的な修正アドバイス（3つ、それぞれ2-3文）

専門的でありながら、初心者にも理解できる言葉で説明してください。
"""

# 3D可視化設定
VISUALIZATION_3D = {
    'enabled': False,
    'camera_distance': 5.0,
    'camera_elevation': 30,
    'camera_azimuth': 45,
    'background_color': (1, 1, 1),
    'skeleton_color': (0, 0.5, 1),
    'joint_size': 0.05
}

# パフォーマンス設定
PERFORMANCE = {
    'max_frames_per_video': 1000,  # 処理する最大フレーム数
    'resize_input': True,  # 入力動画のリサイズ
    'target_resolution': (640, 480),  # リサイズ先の解像度
    'parallel_processing': False  # 並列処理（実験的）
}
