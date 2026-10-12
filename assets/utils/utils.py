import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GraphNorm
import math
from timm.layers import DropPath

# =========================
# ADAPTIVE GRAPH LEARNING
# =========================


#keep the original fixed graph and add a learnable graph attention layer on top of it
class AdaptiveGraphLearner(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_features):
        """
        node_features: [B, T, V, D]
        B = batch size
        T = number of frames
        V = number of landmarks
        D = feature dimension
        """
        B, T, V, D = node_features.shape

        h_i = node_features.unsqueeze(3).repeat(1, 1, 1, V, 1)
        h_j = node_features.unsqueeze(2).repeat(1, 1, V, 1, 1)

        edge_input = torch.cat([h_i, h_j], dim=-1)  # [B, T, V, V, 2D]
        edge_scores = self.edge_mlp(edge_input).squeeze(-1)  # [B, T, V, V]

        adaptive_adj = F.softmax(edge_scores, dim=-1)

        return adaptive_adj



# use this adaptive adjacency matrix to aggregate node information:

class AdaptiveGraphBlock(nn.Module):
    def __init__(self, hidden_dim, fixed_adj, alpha=0.7):
        super().__init__()
        self.graph_learner = AdaptiveGraphLearner(hidden_dim)

        self.alpha = alpha
        self.register_buffer("fixed_adj", fixed_adj)

        self.update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x):
        """
        x: [B, T, V, D]
        """
        B, T, V, D = x.shape
        adaptive_adj = self.graph_learner(x)  # [B, T, V, V]

        fixed = self.fixed_adj.unsqueeze(0).unsqueeze(0)  # [1,1,V,V]
        fixed = fixed.expand(B, T, V, V)

        # hybrid fusion
        adj = self.alpha * fixed + (1 - self.alpha) * adaptive_adj
        adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-6)


        message = torch.matmul(adj, x)  # [B, T, V, D]
        out = self.update(message + x)

        return out, adj

class TopKGraphLearner(nn.Module):
    def __init__(self, hidden_dim, k=8):
        super().__init__()
        self.k = k

        self.score_fn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )


    def forward(self, x, debug=False):

        B,T,V,D = x.shape

        # feature transform
        x = self.score_fn(x)

        # merge batch/time
        x = x.reshape(B*T, V, D)


        # similarity for every graph at once
        sim = torch.bmm(x, x.transpose(1,2))
        sim = sim / (D ** 0.5)


        # make symmetric
        sim = (sim + sim.transpose(1,2))/2


        # remove self connections
        diag = torch.eye(V, device=x.device).bool()
        sim[:, diag] = torch.finfo(sim.dtype).min


        # top-k neighbours
        topk_vals, topk_idx = torch.topk(
            sim,
            self.k,
            dim=-1
        )


        # create source nodes once
        src = torch.arange(
            V,
            device=x.device
        ).view(1,V,1).expand(B*T,V,self.k)


        dst = topk_idx


        edge_index = torch.stack(
            [
                src.reshape(B*T,-1),
                dst.reshape(B*T,-1)
            ],
            dim=1
        )


        edge_weight = topk_vals.reshape(B*T,-1)


        if debug:
            print("sim:", sim.mean(), sim.std())
            print("edges:", edge_index.shape)


        return edge_index, edge_weight

class TopKGraphLearner__(nn.Module):
    def __init__(self, hidden_dim, k=8):
        super().__init__()
        self.k = k

        self.score_fn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, debug=False):
        """
        x: [B, T, V, D]
        returns: edge_index + edge_weight per batch/time
        """
        B, T, V, D = x.shape

        with torch.autocast(device_type="cuda"):

            x = self.score_fn(x)  # feature transform
            sim = torch.matmul(x, x.transpose(-1, -2)) / math.sqrt(D)
            printed = False

            idx = torch.arange(V, device=x.device)
            sim[...,idx,idx] = torch.finfo(sim.dtype).min

            topk_vals, topk_idx = torch.topk(sim, self.k, dim=-1)

            #topk_vals, topk_idx = sim_sym.topk(self.k, dim=-1)
            
            if debug and not printed and b == 0 and t == 0:
                print("sim stats:", sim.mean().item(), sim.std().item())
                print("topk sample:", topk_idx[0][:self.k])
                printed = True
                
            #src = torch.arange(V, device=x.device).unsqueeze(1).expand(V, self.k)
            src = torch.arange(V, device=x.device)
            src = (src[None,None,:,None].expand(B,T,V,self.k))
            src = src.reshape(-1)
            dst = topk_idx.reshape(-1)

            edge_index = torch.stack([src,dst], dim=0)  
            edge_weight = topk_vals.reshape(-1)
        
    
            return edge_index, edge_weight


def batch_time_to_edge_index(edge_index_list, B, T, V):
    """
    Converts per-frame edges → global graph edges
    """
    all_edges = []

    for i, edge_index in enumerate(edge_index_list):
        bt = i
        base = bt * V

        edge_index = edge_index + base
        all_edges.append(edge_index)

    return torch.cat(all_edges, dim=1)

# =========================
# TEMPROAL TRANSFORMER
# =========================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)

        div = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000.0) / d_model)
        )

        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self,x):
        return x + self.pe[:,:x.size(1)]




# =========================
# DEEP FEAT FUSION CLASS
# =========================
class CrossRegionSwinTFusion(nn.Module):
    def __init__(self, region_dim=64, swin_dim=37632, hidden=256, heads=4):
        super().__init__()

        self.region_proj = nn.Linear(region_dim, hidden)

        self.swin_proj = nn.Linear(swin_dim, hidden)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=heads,
            batch_first=True
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Sigmoid()
        )

    def forward(self, region_tokens, swin_tokens):
        """
        region_tokens: [B, T, R, 64]
        swin_tokens:   [B, R, D] OR [B, D]
        """

        B, T, R, _ = region_tokens.shape

        region = self.region_proj(region_tokens)   # [B,T,R,H]

        # convert SwinT into region tokens
        swin = self.swin_proj(swin_tokens).unsqueeze(1).expand(-1, T, -1, -1)

        fused_regions = []

        for r in range(R):
            q = region[:, :, r, :]          # [B,T,H]
            kv = swin[:, :, r, :]           # [B,T,H]
            query = (B*T,4,256)

            key = (B*T,4,256)

            attn_out, _ = self.attn(
                query=q,
                key=kv,
                value=kv
            )

            gate = self.gate(torch.cat([q, attn_out], dim=-1))

            fused_regions.append(q + gate * attn_out)

        return torch.stack(fused_regions, dim=2)  # [B,T,R,H]

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


# ======================================
# ATTENTION TRANSFORMERCONV BLOCK CLASS
# ======================================

class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        attn_dropout: float = 0.20,
        feat_dropout: float = 0.25,
    ):
        super().__init__()
        if out_dim % heads != 0:
            raise ValueError("out_dim must be divisible by heads")

        self.norm = GraphNorm(in_dim)
        self.conv = TransformerConv(
            in_channels=in_dim,
            out_channels=out_dim // heads,
            heads=heads,
            concat=True,
            dropout=attn_dropout,
            beta=True,
        )
        self.drop_path = DropPath(0.1)
        self.feat_dropout = feat_dropout
        self.res_proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(feat_dropout),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, x, edge_index, edge_attr=None):
        residual = self.res_proj(x)

        x = self.norm(x)
        if edge_attr is not None:
            x = self.conv(x, edge_index, edge_attr=edge_attr)
        else:
            x = self.conv(x, edge_index)
        x = F.gelu(x)
        x = F.dropout(x, p=self.feat_dropout, training=self.training)
        x = residual + self.drop_path(x)

        residual = x
        x = self.ffn(x)
        x = F.dropout(x, p=self.feat_dropout, training=self.training)
        x = residual + self.drop_path(x)
        return x













