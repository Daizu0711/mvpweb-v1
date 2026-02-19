# 動画フォーム比較分析システム

VitPoseによる姿勢推定とローカルOllamaを活用した、2つの動画のフォーム比較・分析システムです。

## 🎯 主な機能

### 1. **高精度姿勢推定 (VitPose)**
- 17個の関節点を検出
- 体格差や画面位置に依存しない正規化アルゴリズム
- フレームごとの姿勢トラッキング

### 2. **3D姿勢推定 (PoseFormer) ⭐NEW**
- **2Dから3Dへの投影**: Transformer-based PoseFormerモデルで2D姿勢を3D空間に変換
- **深度方向の分析**: 前後方向の動きも含めた包括的なフォーム分析
- **インタラクティブ3D可視化**: Plotlyによる回転・ズーム可能な3Dビジュアライゼーション
- **並列3D比較**: 参照動画と比較動画を3D空間で並べて表示
- **3Dメトリクス**: MPJPE (Mean Per Joint Position Error) など3D専用の精度指標
- **データエクスポート**: JSON形式で3D座標データをダウンロード可能

### 3. **インテリジェント比較分析**
- **正規化処理**: スケール・位置不変の比較
- **時系列アライメント**: DTW (Dynamic Time Warping) による動作速度の違いを吸収
- **関節角度分析**: 各関節の角度を計算・比較
- **3D空間での比較**: 深度情報を含めた立体的な動作比較

### 4. **スコアリングシステム**
- 総合スコア (0-100)
- 部位別スコア (17関節)
- フレーム単位の詳細スコア
- 時間的整合性スコア
- **3D誤差メトリクス**: 3D空間での位置誤差を定量化

### 5. **AI分析 (Ollama)**
- ローカルLLMによる改善点の提案
- 具体的な修正アドバイス
- 良い点と改善点の明確化
- **3D分析レポート**: 深度方向の動きに関するフィードバック

### 6. **高度な可視化**
- **2D動画比較**: 姿勢オーバーレイ付き並列表示
- **インタラクティブ3D**: ブラウザで操作可能な3D姿勢ビューア
- **並列3D比較**: 参照と比較を同時に3D表示
- **アニメーション再生**: タイムスライダーで任意のフレームを確認

## 📋 システム要件

- Python 3.8+
- CUDA対応GPU (推奨、CPUでも動作可能)
- Ollama (ローカルLLM)
- 8GB+ RAM

## 🚀 セットアップ

### 1. 依存関係のインストール

```bash
# 仮想環境の作成 (推奨)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# パッケージのインストール
pip install -r requirements.txt --break-system-packages
```

### 2. Ollamaのセットアップ

```bash
# Ollamaのインストール (https://ollama.ai)
curl -fsSL https://ollama.com/install.sh | sh

# モデルのダウンロード
ollama pull llama3.2

# Ollamaサーバーの起動
ollama serve
```

### 3. VitPoseモデルの準備

実際のVitPoseモデルを使用する場合:

```bash
# MMPoseのインストール
pip install openmim
mim install mmcv
mim install mmpose

# VitPoseモデルのダウンロード
mkdir -p models
wget https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/vitpose_base_coco_256x192-216eae50_20221122.pth -O models/vitpose.pth
```
https://huggingface.co/JunkyByte/easy_ViTPose/tree/main/torch/coco_25
## 🎬 使用方法

### 1. サーバーの起動

```bash
python app.py
```

サーバーが `http://localhost:5001` で起動します。

### 2. Webインターフェース

ブラウザで `http://localhost:5001` にアクセス:

1. **参照動画をアップロード** (お手本の動画)
2. **比較動画をアップロード** (自分のフォーム)
3. **分析オプションを選択**
   - ✅ **3D姿勢推定を使用**: より詳細な深度方向の分析
4. **「分析開始」をクリック**

### 3. 結果の確認

#### 基本分析結果
- **総合スコア**: 0-100の範囲で表示
- **部位別スコア**: 各関節の一致度
- **AI分析**: Ollamaによる改善提案
- **比較動画**: 姿勢オーバーレイ付き

#### 3D分析結果（3D推定を有効にした場合）
- **3D精度メトリクス**:
  - MPJPE (Mean Per Joint Position Error): 平均関節位置誤差
  - 各関節の3D空間での誤差
- **インタラクティブ3D可視化**:
  - 📊 **並列比較**: 参照と比較を同時に3D表示
  - 🎯 **参照動画**: 参照動画のみを3D表示
  - 👤 **比較動画**: 比較動画のみを3D表示
  - 💾 **データダウンロード**: JSON形式で3D座標をエクスポート
- **3D操作**:
  - マウスドラッグで視点回転
  - スクロールでズーム
  - 再生ボタンでアニメーション
  - スライダーでフレーム選択

## 📊 アルゴリズムの詳細

### 正規化処理

```python
# 1. 中心点の計算 (腰の中心)
center = (left_hip + right_hip) / 2

# 2. 原点への平行移動
normalized_coords = coords - center

# 3. スケール正規化
scale = ||shoulder_center - hip_center||
normalized_coords = normalized_coords / scale
```

### 時系列アライメント (DTW)

Dynamic Time Warpingを使用して、動作速度の違いを吸収:

```python
# DTW行列の計算
for i in range(1, n + 1):
    for j in range(1, m + 1):
        cost = pose_distance(seq1[i-1], seq2[j-1])
        dtw[i, j] = cost + min(
            dtw[i-1, j],    # 挿入
            dtw[i, j-1],    # 削除
            dtw[i-1, j-1]   # 一致
        )
```

### 3D姿勢推定 (PoseFormer) ⭐

PoseFormerは、Transformerアーキテクチャを使用して2D姿勢を3D空間に投影します:

#### アーキテクチャ
```python
# 1. Spatial Transformer: 各フレームの空間的関係を学習
spatial_features = SpatialTransformer(pose_2d)  # [B, T, J, C]

# 2. Temporal Transformer: 時系列の動きパターンを学習
temporal_features = TemporalTransformer(spatial_features)  # [B, T, C]

# 3. 3D座標への投影
pose_3d = OutputHead(temporal_features)  # [B, T, J, 3]
```

#### 入力処理
```python
# 2D姿勢の正規化
normalized_2d = (poses_2d - center) / scale

# シーケンスの準備（243フレーム固定長）
if len(poses_2d) < 243:
    # パディング
    padded = np.pad(poses_2d, ...)
elif len(poses_2d) > 243:
    # 均等サンプリング
    indices = np.linspace(0, len(poses_2d)-1, 243)
    sampled = poses_2d[indices]
```

#### 深度推定
PoseFormerは、以下の情報から深度(Z座標)を推定:
- **視覚的手がかり**: 関節の2D位置関係
- **時系列パターン**: 動きの軌跡
- **解剖学的制約**: 人体の構造的制約
- **学習済み知識**: 大規模3D姿勢データセットからの学習

### 3Dメトリクス計算

```python
# MPJPE (Mean Per Joint Position Error)
errors = np.linalg.norm(pred_3d - gt_3d, axis=-1)  # [T, J]
mpjpe = errors.mean()

# PA-MPJPE (Procrustes Aligned MPJPE)
# 1. スケール・回転・平行移動を最適化
aligned_pred = procrustes_align(pred_3d, gt_3d)
# 2. 誤差計算
pa_errors = np.linalg.norm(aligned_pred - gt_3d, axis=-1)
pa_mpjpe = pa_errors.mean()
```

### スコア計算

```python
# フレームスコア
frame_score = max(0, 100 - distance * 50)

# 関節スコア
joint_score = max(0, 100 - joint_distance * 100)

# 3Dスコア（深度も考慮）
score_3d = max(0, 100 - mpjpe * 200)

# 重み付き平均
overall_score = weighted_average(frame_scores)
```

## 🔧 カスタマイズ

### 1. Ollamaモデルの変更

`app.py`の`generate_ollama_analysis`関数:

```python
response = requests.post(
    'http://localhost:11434/api/generate',
    json={
        'model': 'llama3.2',  # ここを変更
        'prompt': prompt,
        'stream': False
    }
)
```

### 2. スコアリングの調整

`pose_analyzer.py`のスコア計算係数を変更:

```python
# フレームスコアの係数
frame_score = max(0, 100 - frame_dist * 50)  # 50を調整

# 関節スコアの係数
joint_score = max(0, 100 - joint_dist * 100)  # 100を調整
```

### 3. VitPoseモデルの統合

実際のVitPoseモデルを使用する場合、`pose_analyzer.py`の`init_pose_model`を実装:

```python
def init_pose_model(self):
    from mmpose.apis import init_model
    
    config_file = 'configs/vitpose_base_coco.py'
    checkpoint_file = 'models/vitpose.pth'
    
    self.model = init_model(
        config_file, 
        checkpoint_file, 
        device=self.device
    )
```

## 📁 プロジェクト構成

```
.
├── app.py                      # Flaskバックエンド
├── pose_analyzer.py            # 2D姿勢推定・比較ロジック
├── poseformer_model.py         # ⭐ PoseFormerモデル実装
├── pose_3d_visualizer.py       # ⭐ 3D可視化モジュール
├── config.py                   # 設定ファイル
├── requirements.txt            # Python依存関係
├── static/
│   └── index.html             # Webインターフェース
├── uploads/                   # アップロード動画
├── outputs/                   # 出力結果
│   ├── *.mp4                  # 2D比較動画
│   ├── *_3d.html              # ⭐ 3Dインタラクティブ可視化
│   └── poses_3d_*.json        # ⭐ 3D座標データ
└── models/                    # モデルファイル (オプション)
    └── poseformer.pth         # PoseFormer学習済みモデル
```

## 🎯 応用例

### スポーツフォーム分析
- **ゴルフスイング**: 3Dでクラブの軌道と体の回転を分析
- **テニスサーブ**: ラケットの振り抜きと体重移動を3D可視化
- **野球ピッチング**: 投球フォームの深度方向の動きを検証
- **水泳**: ストロークの3D軌跡分析
- **体操・フィギュアスケート**: 回転技の3D解析

### フィットネス
- **スクワットフォーム**: 膝の前後位置と深度を3Dチェック
- **プランクの姿勢**: 体の水平性を3D空間で確認
- **ヨガポーズ**: アサナの3D完成度を測定
- **ダンス**: 振り付けの3D記録と比較

### リハビリテーション
- **歩行分析**: 歩容の3Dパターン分析
- **可動域評価**: 関節の3D空間での動き測定
- **運動パターン比較**: 回復過程の3D記録
- **バランス評価**: 重心位置の3D追跡

### 研究・トレーニング
- **モーションキャプチャ代替**: 低コストで3D動作データ取得
- **バイオメカニクス研究**: 3D運動学・運動力学データ収集
- **パフォーマンス分析**: アスリートの3D動作データベース構築

## 🔐 セキュリティ注意事項

- このシステムはローカル実行を想定
- 本番環境では適切な認証・認可を実装
- アップロードファイルのサイズ制限を設定
- CORS設定を適切に構成

## 🐛 トラブルシューティング

### Ollamaに接続できない

```bash
# Ollamaが起動しているか確認
curl http://localhost:11434/api/tags

# 起動していない場合
ollama serve
```

### GPUメモリ不足

```python
# app.py でバッチサイズを削減
# または CPU モードで実行
self.device = 'cpu'
```

### 動画が処理されない

- 対応フォーマット: MP4, AVI, MOV
- 最大ファイルサイズ: 100MB (変更可能)
- フレームレートが高すぎる場合は`sample_rate`を調整

## 📝 ライセンス

MIT License

## 🤝 貢献

プルリクエストを歓迎します！

## 📧 サポート

問題が発生した場合は、GitHubのIssueを作成してください。

## 🚀 今後の改善予定

- [ ] リアルタイムストリーミング分析
- [ ] 複数人同時比較
- [ ] モーションキャプチャエクスポート
- [ ] カスタムトレーニングデータ対応
- [ ] モバイルアプリ版
