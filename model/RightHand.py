import os
import seaborn as sns
import torch.nn as nn
from .modules import *


class SpatioTemporalDualNetwork(nn.Module):
    """Spatio-Temporal Attention Network with Two Attention Heads.
    """

    def __init__(self, num_classes, num_states, num_joints, input_dim, sequence_length, dropout_rate, save_attention=False):
        super().__init__()

        head_dim = 64
        num_heads = 8
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.num_states = num_states
        self.num_joints = num_joints
        self.save_attention = save_attention
        self.temperature = 0.1

        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            LayerNormalization(256),
            nn.Dropout(0.2),
        )

        self.spatial_attention_weights = None
        self.temporal_attention_weights = None
        self.attention_layer3 = None

        self.spatial_attention = SpatioTemporalAttentionLayer(
            input_dim=256, output_dim=256, num_heads=num_heads, head_dim=head_dim, dropout_rate=dropout_rate,
            num_frames=self.sequence_length, num_joints=self.num_joints, attention_domain="spatial")

        self.temporal_attention = SpatioTemporalAttentionLayer(
            input_dim=256, output_dim=256, num_heads=num_heads, head_dim=head_dim, dropout_rate=dropout_rate,
            num_frames=self.sequence_length, num_joints=self.num_joints, attention_domain="temporal")

        # Second temporal attention layer for gesture state classification
        self.temporal_attention_2 = SpatioTemporalAttentionLayer(
            input_dim=256, output_dim=256, num_heads=num_heads, head_dim=head_dim, dropout_rate=dropout_rate,
            num_frames=self.sequence_length, num_joints=self.num_joints, attention_domain="temporal")

        self.state_classifier = nn.Linear(256 + num_classes, self.num_states)
        self.classifier = nn.Linear(256, self.num_classes)

    def forward(self, input_data, contrastive=False, positive_pair=None):
        # input_data shape: [batch_size, sequence_length, num_joints, dimension]
        seq_length = input_data.shape[1]
        num_joints = input_data.shape[2]
        dimension = input_data.shape[-1]
        if seq_length * num_joints * dimension == 0:
            raise ValueError("The input tensor has a zero dimension, cannot reshape")

        # Compute embeddings
        x, embeddings = self.compute_embeddings(input_data, seq_length, num_joints, dimension)
        class_output = self.classifier(embeddings)

        x2, self.attention_layer3 = self.temporal_attention_2(x)

        # Reshape for state prediction
        x2 = x2.view(-1, seq_length, num_joints, x2.shape[-1])
        x2 = torch.mean(x2, dim=2)  # [batch_size, sequence_length, feature_dim]

        # Concatenate class predictions with features for state prediction
        class_expanded = class_output.unsqueeze(1).expand(-1, x2.shape[1], -1)
        x2 = torch.cat((x2, class_expanded), dim=2)

        x2 = self.state_classifier(x2)
        state_output = x2.transpose(1, 2)  # Reshape output for loss calculation

        if contrastive and positive_pair is not None:
            # Contrastive learning
            _, positive_embeddings = self.compute_embeddings(positive_pair, seq_length, num_joints, dimension)
            # Compute contrastive loss
            loss = self.contrastive_loss(embeddings, positive_embeddings)
            return class_output, state_output, loss

        return class_output, state_output

    def compute_embeddings(self, input_data, seq_length, num_joints, dimension):
        x = input_data.view(-1, seq_length * num_joints, dimension)
        x = self.input_projection(x)
        x, self.spatial_attention_weights = self.spatial_attention(x)
        embeddings, self.temporal_attention_weights = self.temporal_attention(x)
        embeddings = torch.mean(embeddings, dim=1)
        return x, embeddings  # Return intermediate features and embeddings

    def contrastive_loss(self, embeddings1, embeddings2):
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, dim=1)
        embeddings2 = F.normalize(embeddings2, dim=1)

        similarity_matrix = torch.matmul(embeddings1, embeddings2.T) / self.temperature
        labels = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)

        # Compute contrastive loss
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

    def get_sequence_length(self):
        return self.sequence_length

    def get_num_classes(self):
        return self.num_classes

    def get_input_dimension(self):
        return self.input_dim

    def get_num_joints(self):
        return self.num_joints

    def save_attention_weights(self, save_path):
        spatial_att_dir = os.path.join(save_path, 'spatial_attention')
        temporal_att_dir = os.path.join(save_path, 'temporal_attention')

        # Save spatial attention weights
        if self.spatial_attention_weights is not None:
            os.makedirs(spatial_att_dir, exist_ok=True)
            for batch_idx in range(self.spatial_attention_weights.shape[0]):
                for head_idx in range(self.spatial_attention_weights.shape[1]):
                    fig = sns.heatmap(self.spatial_attention_weights[batch_idx, head_idx].cpu())
                    fig_path = os.path.join(spatial_att_dir, f'batch{batch_idx}_head{head_idx}.png')
                    fig.get_figure().savefig(fig_path)
                    fig.get_figure().clf()

        # Save temporal attention weights
        if self.temporal_attention_weights is not None:
            os.makedirs(temporal_att_dir, exist_ok=True)
            for batch_idx in range(self.temporal_attention_weights.shape[0]):
                for head_idx in range(self.temporal_attention_weights.shape[1]):
                    fig = sns.heatmap(self.temporal_attention_weights[batch_idx, head_idx].cpu())
                    fig_path = os.path.join(temporal_att_dir, f'batch{batch_idx}_head{head_idx}.png')
                    fig.get_figure().savefig(fig_path)
                    fig.get_figure().clf()
