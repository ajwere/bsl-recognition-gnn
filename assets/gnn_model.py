"""
Graph Neural Network Model for BSL Recognition

This file defines the main model architecture used for British Sign Language
recognition. The model combines spatial graph learning, temporal modelling,
and optional multimodal feature fusion.

Overview:
- Encodes landmark-based graph features (hands, mouth, upper body)
- Applies stacked graph attention layers (TransformerConv)
- Pools spatial features into region-level representations
- Models temporal dynamics using a bidirectional GRU with attention
- Optionally fuses deep VideoMAE features (upper body, hands, mouth)
  using a gated residual attention mechanism

Key Components:
- ResidualAttentionBlock: graph attention layers with residual connections
- Temporal GRU + attention pooling: captures motion over time
- GatedResidualFusion: combines landmark and deep features
- Classifier: predicts sign class labels

Outputs:
- Main classification logits
- Auxiliary landmark-only logits (for regularisation)
- Optional attention weights (for analysis/interpretability)

Purpose:
This model forms the core of the sign language recognition system, enabling
learning of both spatial structure and temporal dynamics, with optional
multimodal fusion for improved performance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GraphNorm
import math
from assets.utils.utils import *
from assets.utils.SSL_utils import *
from assets.dataset_loader import *
scaler = torch.amp.GradScaler("cuda")

class GatedResidualFusion(nn.Module):
    def __init__(
        self,
        query_dim: int,
        token_dims: list[int],
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.query_proj = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.token_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            for d in token_dims
        ])

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid(),
        )

        self.alpha = nn.Parameter(torch.tensor(0.1))#-2.0

        self.query_out_proj = (
            nn.Identity() if query_dim == hidden_dim else nn.Linear(query_dim, hidden_dim)
        )

    def forward(self, query_feat, token_feats):
        q = self.query_proj(query_feat)          # [B, H]
        q_seq = q.unsqueeze(1)                   # [B, 1, H]

        tokens = [proj(feat) for feat, proj in zip(token_feats, self.token_projs)]
        kv = torch.stack(tokens, dim=1)          # [B, N, H]

        attn_out, attn_weights = self.attn(
            query=q_seq,
            key=kv,
            value=kv,
            need_weights=True,
        )
        attn_out = attn_out.squeeze(1)           # [B, H]

        fused = self.fuse(torch.cat([q, attn_out], dim=-1))   # [B, H]
        gate = self.gate(torch.cat([q, fused], dim=-1))       # [B, H]

        base = self.query_out_proj(query_feat)                 # [B, H]
        alpha = torch.sigmoid(self.alpha)
        out = base + alpha * gate * fused

        return out, attn_weights.squeeze(1)

class CustomGNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        heads: int = 4,
        attn_dropout: float = 0.20,
        feat_dropout: float = 0.25,
        gru_hidden: int = 128,
        T: int = 32,
        V: int = 89,
        upper_dim: int = 768,
        mouth_dim: int = 768,
        hand_dim: int = 768,
        fusion_hidden_dim: int = 256,
        return_attention: bool = True,
        temporal_model="gru",
        graph_type="fixed",
        use_fuse=False, 
        use_upper_mouth_fusion=False,
        
    ):
        super().__init__()

        if 256 % heads != 0 or 128 % heads != 0 or 64 % heads != 0:
            raise ValueError("256, 128, and 64 must all be divisible by heads")

        self.T = T
        self.V = V
        self.final_gnn_dim = 128

       
        self.return_attention = return_attention

        self.temporal_out_dim = gru_hidden * 2 # could just use 256 instead of gru_hidden * 2
        self.fusion_hidden_dim = fusion_hidden_dim

        # temproal transfrmer | BI-GRU
        self.temporal_model = temporal_model

        # adaptive graph leanring choice
        self.graph_type = graph_type
        self.use_fuse = use_fuse


        # or if need adjacency:
        # self.base_fixed_adj = edge_index_to_adj(
        #     build_spatiotemporal_edge_index_cached(self.T, self.V),
        #     self.T * self.V
        # )
        self.base_fixed_adj = torch.zeros(V, V)

        

        # classifier input dim depends on final stage used
        if use_fuse:
            cls_in = fusion_hidden_dim
        else:
            cls_in = self.temporal_out_dim

        # region splits
        self.lh_slice = slice(0, 21)
        self.rh_slice = slice(21, 42)
        self.mouth_slice = slice(42, 82)
        self.pose_slice = slice(82, 89)
        # weight learning pooling 
        self.pool_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
        self.region_pool_weights = nn.Parameter(torch.tensor([1.0, 1.0]))


         # Optional fusion layer 1 & 2: temporal -> [upper, mouth]
        if self.use_fuse:
            self.mouth_fusion = GatedResidualFusion(
                query_dim=self.temporal_out_dim,
                token_dims=[mouth_dim],
                hidden_dim=fusion_hidden_dim,
                num_heads=heads,
                dropout=feat_dropout,
            )
            self.upper_fusion = GatedResidualFusion(
                query_dim=fusion_hidden_dim,
                token_dims=[upper_dim],
                hidden_dim=fusion_hidden_dim,
                num_heads=heads,
                dropout=feat_dropout,
            )
            fusion2_query_dim = fusion_hidden_dim
            self.fusion2 = GatedResidualFusion(
                query_dim=fusion2_query_dim,
                token_dims=[hand_dim],
                hidden_dim=fusion_hidden_dim,
                num_heads=heads,
                dropout=feat_dropout,
            )

            



        # node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(feat_dropout),
        )

        # # node decoder
        # self.ssl_decoder = nn.Sequential(
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, input_dim)  # reconstruct original landmarks
        # )
        
        # Adaptive graph leanring 
        # self.adaptive_block = AdaptiveGraphBlock(
        #     hidden_dim=256,
        #     fixed_adj=self.base_fixed_adj,
        #     alpha=0.7
        # )
        self.adaptive_block = TopKGraphLearner(
            hidden_dim=256,
            k=8,
        )

        # graph layers
        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(256, 256, heads, attn_dropout, feat_dropout),
            #ResidualAttentionBlock(256, 256, heads, attn_dropout, feat_dropout),
            ResidualAttentionBlock(256, 128, heads, attn_dropout, feat_dropout),
            ResidualAttentionBlock(128, 128, heads, attn_dropout, feat_dropout),
        ])

        self.ssl_projection = nn.Sequential(
            nn.Linear(self.final_gnn_dim, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )

        self.region_proj = nn.Sequential(
            nn.Linear(self.final_gnn_dim * 4, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(feat_dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(cls_in, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(feat_dropout),
            nn.Linear(128, num_classes),
        )
        self.landmark_classifier = nn.Sequential(
            nn.Linear(self.temporal_out_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(feat_dropout),
            nn.Linear(128, num_classes),
        )
        self.temporal_attn = nn.Sequential(
            nn.Linear(self.temporal_out_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

       

        self.temporal_gru = nn.GRU(
            input_size=256,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            dropout=feat_dropout,
            bidirectional=True,
        )

       

        # Project frame features for transformer-based models
        self.pos_encoding = PositionalEncoding(256)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=512,
            dropout=feat_dropout,
            activation="gelu",
            batch_first=True,
        )

        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4
        )

        self.temporal_norm = nn.LayerNorm(gru_hidden * 2)

    def _pool_region(self, x_region):
        mean_pool = x_region.mean(dim=2)
        max_pool = x_region.max(dim=2).values
        #return 0.5 * (mean_pool + max_pool)

        weights = torch.softmax(self.region_pool_weights, dim=0)

        return (
            weights[0] * mean_pool +
            weights[1] * max_pool
        )

    def encode(self, data, debug=False):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.node_encoder(x)
        num_graphs = int(batch.max().item()) + 1
        x = x.view(num_graphs, self.T, self.V, 256)
        
        edge_weight = None
        if self.graph_type in ["adaptive", "hybrid"]:
            adaptive_edge_list, adaptive_weight_list = self.adaptive_block(x, debug=debug)

            adaptive_edge_index = batch_time_to_edge_index(
                adaptive_edge_list,
                B=x.shape[0],
                T=x.shape[1],
                V=x.shape[2]
            )
            #print("Nodes:",x.shape,"Edges:",edge_index.shape)

            adaptive_edge_weight = torch.cat(adaptive_weight_list, dim=0).unsqueeze(-1)

            if self.graph_type == "adaptive":
                # AGL only
                edge_index = adaptive_edge_index
                edge_weight = adaptive_edge_weight

            elif self.graph_type == "hybrid":
                edge_index = torch.cat([edge_index,adaptive_edge_index],dim=1)
                # fixed edges have no learned weight
                fixed_weights = torch.ones(edge_index.shape[1] - adaptive_edge_index.shape[1],device=edge_index.device).unsqueeze(-1)
                edge_weight = torch.cat([fixed_weights,adaptive_edge_weight], dim=0)
        

        
        x = x.view(num_graphs * self.T * self.V, 256)

        #print("SHAPE :  ", x.shape) # [128, 89, 256]
        if edge_weight is None:
            edge_weight = torch.ones(
                edge_index.size(1),
                device=x.device
            )
        edge_weight = edge_weight.unsqueeze(-1)

        for block in self.blocks:
            #x = block(x, edge_index)
            x = block(x, edge_index, edge_attr=edge_weight)

        x = x.view(num_graphs, self.T, self.V, self.final_gnn_dim)
        x = self.ssl_projection(x)

        return x


        
    def forward(self, data, ssl=False, debug=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        with torch.autocast("cuda", enabled=False):
            x = self.node_encoder(x)
        num_graphs = int(batch.max().item()) + 1
        x = x.view(num_graphs, self.T, self.V, 256)

        edge_weight = None
        if self.graph_type in ["adaptive", "hybrid"]:
            adaptive_edge_list, adaptive_weight_list = self.adaptive_block(x, debug=debug)

            adaptive_edge_index = batch_time_to_edge_index(
                adaptive_edge_list,
                B=x.shape[0],
                T=x.shape[1],
                V=x.shape[2]
            )
            #print("Nodes:",x.shape,"Edges:",edge_index.shape)

            adaptive_edge_weight = torch.cat(adaptive_weight_list, dim=0).unsqueeze(-1)

            if self.graph_type == "adaptive":
                # AGL only
                edge_index = adaptive_edge_index
                edge_weight = adaptive_edge_weight

            elif self.graph_type == "hybrid":
                edge_index = torch.cat([edge_index,adaptive_edge_index],dim=1)
                # fixed edges have no learned weight
                fixed_weights = torch.ones(edge_index.shape[1] - adaptive_edge_index.shape[1],device=edge_index.device).unsqueeze(-1)
                edge_weight = torch.cat([fixed_weights,adaptive_edge_weight], dim=0)
        
        
        x = x.view(num_graphs * self.T * self.V, 256)

        for block in self.blocks:
            #x = block(x, edge_index)
            x = block(x, edge_index, edge_attr=edge_weight)

        x = x.view(num_graphs, self.T, self.V, self.final_gnn_dim)


        # region pooling
        lh = self._pool_region(x[:, :, self.lh_slice, :])
        rh = self._pool_region(x[:, :, self.rh_slice, :])
        mouth = self._pool_region(x[:, :, self.mouth_slice, :])
        pose = self._pool_region(x[:, :, self.pose_slice, :])

        graph_tokens = torch.stack(
            [lh, rh, mouth, pose],
            dim=2,
        )
        # Combine the 4 region representations into a frame-level feature
        frame_feat = graph_tokens.reshape(
            num_graphs, self.T, self.final_gnn_dim * 4
        )

        frame_feat = self.region_proj(frame_feat)


        # defining is you want transformer | GRU | hybrid
        if self.temporal_model == "gru":
            temporal_out, _ = self.temporal_gru(frame_feat)

        elif self.temporal_model == "transformer":
            temporal_out = self.pos_encoding(frame_feat)
            temporal_out = self.temporal_transformer(temporal_out)

        elif self.temporal_model == "hybrid":
            temporal_out, _ = self.temporal_gru(frame_feat)
            temporal_out = self.pos_encoding(temporal_out)
            temporal_out = self.temporal_transformer(temporal_out)


        attn_scores = self.temporal_attn(temporal_out)           
        #attn_weights = torch.softmax(attn_scores, dim=1)
        attn_weights = torch.softmax(attn_scores.float(), dim=1).to(attn_scores.dtype)
        temporal_attn_pool = (temporal_out * attn_weights).sum(dim=1)

        temporal_mean = temporal_out.mean(dim=1)
        temporal_max = temporal_out.max(dim=1).values
        weights = torch.softmax(self.pool_weights, dim=0)
        temporal_feat = (weights[0] * temporal_attn_pool + weights[1] * temporal_mean + weights[2] * temporal_max)
        temporal_feat = self.temporal_norm(temporal_feat)


        attn_dict = {}
        feat_for_cls = temporal_feat

        # -------------------------
        # Case 1: upper + mouth
        # -------------------------
        feat_for_cls = temporal_feat

        if self.use_fuse:
            mouth_feat = data.mouth_feature.view(num_graphs, -1).float()
            upper_feat = data.upper_body_feature.view(num_graphs, -1).float()
            hand_feat = data.hands_feature.view(num_graphs, -1).float()

            feat_for_cls, attn_mouth = self.mouth_fusion(
                query_feat=feat_for_cls,
                token_feats=[mouth_feat],
            )
            attn_dict["mouth_attn"] = attn_mouth

            feat_for_cls, attn_upper = self.upper_fusion(
                query_feat=feat_for_cls,
                token_feats=[upper_feat],
            )
            attn_dict["upper_attn"] = attn_upper
       
            if not hasattr(data, "hands_feature"):
                raise ValueError("Data object missing hands_feature")

            feat_for_cls, attn_weights_2 = self.fusion2(
                query_feat=feat_for_cls,
                token_feats=[hand_feat],
            )

            attn_dict["fusion2_attn"] = attn_weights_2  # [B, 1]


    
        landmark_logits = self.landmark_classifier(temporal_feat)

        logits = self.classifier(feat_for_cls)



        if self.return_attention:
            return logits, landmark_logits, edge_index, edge_weight, attn_dict
        else: 
            return logits, landmark_logits, edge_index, edge_weight,


        
