import torch
import torch.nn as nn
import numpy as np


class BiGRUCompletionModel(nn.Module):
    """
    BiGRU モデルで欠損キーポイント（confidence < 0.3）を補完。
    入力: [batch, seq_len, 34] (17 joints × (x, y))
    出力: [batch, seq_len, 34] (補完済み座標)
    """

    def __init__(self, input_size=34, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Decoder (bidir: hidden_size * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, 34] 入力キーポイント（欠損部分は 0）
            mask: [batch, seq_len, 34] または [batch, seq_len, 17]
                  mask=1 で欠損フレーム・関節を示す
        Returns:
            output: [batch, seq_len, 34] 補完後のキーポイント
        """
        batch_size, seq_len, _ = x.size()

        # GRU エンコーディング
        gru_out, _ = self.gru(x)  # [batch, seq_len, hidden_size*2]

        # デコーディング
        output = self.fc(gru_out)  # [batch, seq_len, 34]

        # マスク適用：欠損部分のみ補完値を使用
        if mask is not None:
            # mask が [batch, seq_len, 17] の場合、[batch, seq_len, 34] に展開
            if mask.size(-1) == 17:
                mask_expanded = mask.unsqueeze(-1).expand(-1, -1, -1, 2).reshape(batch_size, seq_len, 34)
            else:
                mask_expanded = mask

            # 欠損: mask=1, 検出済み: mask=0
            # output = mask * output + (1 - mask) * x
            output = mask_expanded * output + (1 - mask_expanded) * x

        return output

    @classmethod
    def load_pretrained(cls, checkpoint_path, device='cpu'):
        """学習済みモデルを読み込み"""
        model = cls()
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model


def create_completion_mask_from_confidence(keypoints, confidence_threshold=0.3):
    """
    keypoints: [seq_len, 17, 3] (x, y, confidence)
    返り値: [seq_len, 17] マスク (欠損=1, 検出済み=0)
    """
    confidences = keypoints[:, :, 2]  # [seq_len, 17]
    mask = (confidences < confidence_threshold).float()
    return mask


def apply_bigru_completion(keypoints, model, device='cpu', confidence_threshold=0.3):
    """
    BiGRU で欠損キーポイントを補完。

    Args:
        keypoints: [seq_len, 17, 3] (x, y, confidence)
        model: BiGRUCompletionModel インスタンス
        device: 'cpu' or 'cuda'
        confidence_threshold: これ以下の信頼度は欠損と判定

    Returns:
        completed_keypoints: [seq_len, 17, 3] 補完後のキーポイント
    """
    seq_len, num_joints, _ = keypoints.shape

    # confidence < threshold の部分をマスク
    mask = create_completion_mask_from_confidence(keypoints, confidence_threshold)  # [seq_len, 17]

    # 入力: x, y のみ（confidence は除外）
    keypoints_xy = keypoints[:, :, :2]  # [seq_len, 17, 2]
    keypoints_xy_flat = keypoints_xy.reshape(seq_len, -1)  # [seq_len, 34]

    # 欠損部分を 0 にマスク
    keypoints_xy_masked = keypoints_xy_flat.clone()
    for t in range(seq_len):
        for j in range(num_joints):
            if mask[t, j] > 0.5:  # 欠損フレーム
                keypoints_xy_masked[t, j*2:(j+1)*2] = 0

    # モデル入力
    x_input = torch.FloatTensor(keypoints_xy_masked).unsqueeze(0).to(device)  # [1, seq_len, 34]
    mask_input = torch.FloatTensor(mask).unsqueeze(0).to(device)  # [1, seq_len, 17]

    with torch.no_grad():
        output = model(x_input, mask=mask_input)  # [1, seq_len, 34]

    completed_xy = output[0].cpu().numpy()  # [seq_len, 34]
    completed_xy = completed_xy.reshape(seq_len, num_joints, 2)  # [seq_len, 17, 2]

    # 元の confidence を保持
    completed_keypoints = np.zeros_like(keypoints)
    completed_keypoints[:, :, :2] = completed_xy
    completed_keypoints[:, :, 2] = keypoints[:, :, 2]  # confidence は変更なし

    return completed_keypoints
