import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionEncodingLayer(nn.Module):
    """Implementation of the positional encoding function."""

    def __init__(self, feature_size, time_length, num_joints, attention_domain):
        super(PositionEncodingLayer, self).__init__()
        self.num_joints = num_joints
        self.time_length = time_length
        self.attention_domain = attention_domain

        positions = self._calculate_positions()
        positional_encoding = self._compute_positional_encoding(positions, feature_size)
        self.register_buffer('positional_encoding', positional_encoding)

    def _calculate_positions(self):
        if self.attention_domain in ["temporal", "mask_t"]:
            positions = list(range(self.num_joints * self.time_length))
        elif self.attention_domain in ["spatial", "mask_s"]:
            positions = []
            for _ in range(self.time_length):
                positions.extend([joint_id for joint_id in range(self.num_joints)])
        else:
            raise ValueError("Unsupported attention domain")

        positions = torch.from_numpy(np.array(positions)).unsqueeze(1).float()
        return positions

    def _compute_positional_encoding(self, positions, feature_size):
        pe = torch.zeros(self.time_length * self.num_joints, feature_size)
        div_term = torch.exp(torch.arange(0, feature_size, 2).float() *
                             -(math.log(10000.0) / feature_size))
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x):
        x = x + self.positional_encoding[:, :x.size(1)]
        return x


class LayerNormalization(nn.Module):
    """Implementation of the Layer Normalization module."""

    def __init__(self, feature_dim, epsilon=1e-6):
        super(LayerNormalization, self).__init__()
        self.weight = nn.Parameter(torch.ones(feature_dim))
        self.bias = nn.Parameter(torch.zeros(feature_dim))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.epsilon) + self.bias


class MultiHeadAttention(nn.Module):
    """Implementation of the Multi-Head Attention mechanism."""

    def __init__(self, num_heads, head_dim, input_dim, num_frames, num_joints, dropout_rate, attention_domain):
        super(MultiHeadAttention, self).__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.attention_domain = attention_domain
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.attention_weights = None

        temporal_mask, spatial_mask = self._generate_attention_masks()
        self.register_buffer('temporal_mask', temporal_mask)
        self.register_buffer('spatial_mask', spatial_mask)

        self.query_linear = nn.Sequential(
            nn.Linear(input_dim, self.head_dim * self.num_heads),
            nn.Dropout(dropout_rate),
        )
        self.key_linear = nn.Sequential(
            nn.Linear(input_dim, self.head_dim * self.num_heads),
            nn.Dropout(dropout_rate),
        )
        self.value_linear = nn.Sequential(
            nn.Linear(input_dim, self.head_dim * self.num_heads),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

    def _generate_attention_masks(self):
        total_size = self.num_frames * self.num_joints
        temporal_mask = torch.ones(total_size, total_size)
        filtered_area = torch.zeros(self.num_joints, self.num_joints)

        for i in range(self.num_frames):
            start = i * self.num_joints
            temporal_mask[start:start + self.num_joints, start:start + self.num_joints] *= filtered_area

        identity = torch.eye(total_size)
        spatial_mask = 1 - temporal_mask
        temporal_mask = temporal_mask + identity
        return temporal_mask, spatial_mask

    def _scaled_dot_product_attention(self, query, key, value):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if self.attention_domain:
            if self.attention_domain == "temporal":
                scores *= self.temporal_mask
                scores += (1 - self.temporal_mask) * (-9e15)
            elif self.attention_domain == "spatial":
                scores *= self.spatial_mask
                scores += (1 - self.spatial_mask) * (-9e15)

        attention_probs = F.softmax(scores, dim=-1)
        return torch.matmul(attention_probs, value), attention_probs

    def forward(self, x):
        batch_size = x.size(0)

        # Linear projections
        query = self.query_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply attention
        x, self.attention_weights = self._scaled_dot_product_attention(query, key, value)

        # Concatenate heads
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head_dim * self.num_heads)

        return x, self.attention_weights


class SpatioTemporalAttentionLayer(nn.Module):
    """Implementation of the Spatio-Temporal Attention Layer."""

    def __init__(self, input_dim, output_dim, num_heads, head_dim, dropout_rate, num_frames, num_joints, attention_domain):
        super(SpatioTemporalAttentionLayer, self).__init__()

        self.positional_encoding = PositionEncodingLayer(input_dim, num_frames, num_joints, attention_domain)

        self.attention = MultiHeadAttention(num_heads, head_dim, input_dim, num_frames, num_joints, dropout_rate, attention_domain)

        self.feature_mapping = nn.Sequential(
            nn.Linear(num_heads * head_dim, output_dim),
            nn.ReLU(),
            LayerNormalization(output_dim),
            nn.Dropout(dropout_rate),
        )
        self.attention_weights = None
        self._initialize_parameters()

    def forward(self, x):
        x = self.positional_encoding(x)
        x, self.attention_weights = self.attention(x)
        x = self.feature_mapping(x)
        return x, self.attention_weights

    def _initialize_parameters(self):
        modules = [self.attention, self.feature_mapping]
        for module in modules:
            for param in module.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
