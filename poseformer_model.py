"""
PoseFormer: 2D to 3D Pose Estimation
Based on "3D Human Pose Estimation with Spatial and Temporal Transformers"
"""

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat

class PoseFormer(nn.Module):
    """
    PoseFormer model for lifting 2D poses to 3D
    """
    def __init__(
        self,
        num_joints=17,
        in_chans=2,
        embed_dim=512,
        depth=8,
        num_heads=8,
        mlp_ratio=2.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        num_frames=243
    ):
        super().__init__()
        
        self.num_joints = num_joints
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        
        # Spatial patch embedding
        self.spatial_patch_embedding = nn.Linear(in_chans, embed_dim)
        
        # Temporal patch embedding
        self.temporal_patch_embedding = nn.Linear(embed_dim, embed_dim)
        
        # Positional embeddings
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, num_joints, embed_dim)
        )
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, num_frames, embed_dim)
        )
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.spatial_blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i]
            )
            for i in range(depth)
        ])
        
        self.temporal_blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i]
            )
            for i in range(depth)
        ])
        
        # Output head
        self.head = nn.Linear(embed_dim, 3)  # Output 3D coordinates
        
        # Initialize weights
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def spatial_forward(self, x):
        """
        Spatial transformer for each frame
        x: [B, T, J, C] where B=batch, T=time, J=joints, C=channels
        """
        B, T, J, C = x.shape
        
        # Reshape for spatial processing
        x = rearrange(x, 'b t j c -> (b t) j c')
        
        # Patch embedding
        x = self.spatial_patch_embedding(x)
        
        # Add positional embedding
        x = x + self.spatial_pos_embed
        
        # Apply spatial transformer blocks
        for blk in self.spatial_blocks:
            x = blk(x)
        
        # Reshape back
        x = rearrange(x, '(b t) j c -> b t j c', b=B, t=T)
        
        return x
    
    def temporal_forward(self, x):
        """
        Temporal transformer across frames
        x: [B, T, J, C]
        """
        B, T, J, C = x.shape
        
        # Average across joints for temporal modeling
        x = x.mean(dim=2)  # [B, T, C]
        
        # Temporal patch embedding
        x = self.temporal_patch_embedding(x)
        
        # Add positional embedding
        x = x + self.temporal_pos_embed[:, :T, :]
        
        # Apply temporal transformer blocks
        for blk in self.temporal_blocks:
            x = blk(x)
        
        # Expand back to joints
        x = repeat(x, 'b t c -> b t j c', j=J)
        
        return x
    
    def forward(self, x):
        """
        Forward pass
        x: [B, T, J, 2] - 2D pose sequence
        returns: [B, T, J, 3] - 3D pose sequence
        """
        # Spatial-temporal processing
        x = self.spatial_forward(x)
        x = self.temporal_forward(x)
        
        # Output head
        x = self.head(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and MLP
    """
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        drop=0.,
        attn_drop=0.,
        drop_path=0.
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop
        )
    
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Attention(nn.Module):
    """
    Multi-head self-attention
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.
    ):
        super().__init__()
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """
    MLP with GELU activation
    """
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.
    ):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample
    """
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        
        return output


def load_poseformer_model(checkpoint_path=None, device='cuda'):
    """
    Load PoseFormer model
    """
    model = PoseFormer(
        num_joints=17,
        in_chans=2,
        embed_dim=512,
        depth=8,
        num_heads=8,
        num_frames=243
    )
    
    if checkpoint_path and torch.cuda.is_available():
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded PoseFormer checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            print("Using randomly initialized model")
    
    model = model.to(device)
    model.eval()
    
    return model


def prepare_2d_poses_for_poseformer(poses_2d, num_frames=243):
    """
    Prepare 2D pose sequence for PoseFormer input
    
    Args:
        poses_2d: List of pose dictionaries with 'keypoints' [N, 17, 3]
        num_frames: Target sequence length
    
    Returns:
        torch.Tensor: [1, num_frames, 17, 2]
    """
    # Extract keypoints
    keypoints_list = []
    for pose in poses_2d:
        if pose is not None:
            kp = pose['keypoints'][:, :2]  # Take only x, y
            keypoints_list.append(kp)
        else:
            # Use zeros for missing frames
            keypoints_list.append(np.zeros((17, 2)))
    
    keypoints_array = np.array(keypoints_list)  # [T, 17, 2]
    
    # Pad or truncate to num_frames
    T = len(keypoints_array)
    if T < num_frames:
        # Pad with last frame
        padding = np.repeat(keypoints_array[-1:], num_frames - T, axis=0)
        keypoints_array = np.concatenate([keypoints_array, padding], axis=0)
    elif T > num_frames:
        # Sample uniformly
        indices = np.linspace(0, T - 1, num_frames, dtype=int)
        keypoints_array = keypoints_array[indices]
    
    # Convert to tensor and add batch dimension
    tensor = torch.from_numpy(keypoints_array).float()
    tensor = tensor.unsqueeze(0)  # [1, num_frames, 17, 2]
    
    return tensor


def normalize_2d_poses(poses_2d):
    """
    Normalize 2D poses to [-1, 1] range
    """
    # Find bounding box
    all_points = []
    for pose in poses_2d:
        if pose is not None and 'keypoints' in pose:
            valid_kp = pose['keypoints'][pose['keypoints'][:, 2] > 0.3]
            if len(valid_kp) > 0:
                all_points.append(valid_kp[:, :2])
    
    if not all_points:
        return poses_2d
    
    all_points = np.concatenate(all_points, axis=0)
    min_xy = all_points.min(axis=0)
    max_xy = all_points.max(axis=0)
    center = (min_xy + max_xy) / 2
    scale = (max_xy - min_xy).max()
    
    # Normalize
    normalized_poses = []
    for pose in poses_2d:
        if pose is None:
            normalized_poses.append(None)
            continue
        
        norm_pose = pose.copy()
        norm_pose['keypoints'][:, :2] = (pose['keypoints'][:, :2] - center) / (scale / 2)
        normalized_poses.append(norm_pose)
    
    return normalized_poses
