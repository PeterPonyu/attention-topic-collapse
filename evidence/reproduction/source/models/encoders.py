"""
Efficient Encoder Modules for Single-Cell Data.

This module provides encoder architectures for single-cell representation learning:

1. MLP ENCODER (Baseline):
   - Simple and fast
   - Good baseline performance
   - Optional residual connections

2. MULTI-HEAD PROJECTION TRANSFORMER (Recommended for attention):
   - Projects gene vector into multiple token embeddings
   - Self-attention learns relationships between projections  
   - Prevents representation collapse through diverse projections
   - O(num_tokens²) complexity where num_tokens << n_genes

3. HYBRID MLP-ATTENTION (Balanced):
   - MLP backbone with multi-head attention refinement
   - Best of both worlds: MLP stability + attention expressiveness
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Literal

from .shared_modules import weight_init, MLP, ResidualMLP


# =============================================================================
# MLP ENCODERS (Baseline)
# =============================================================================

class MLPEncoder(nn.Module):
    """Simple MLP encoder (fastest baseline)."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_vae: bool = False,
        var_eps: float = 1e-4,
        use_residual: bool = False):
        super().__init__()
        self.use_vae = use_vae
        self.var_eps = var_eps
        
        if use_residual:
            self.encoder = ResidualMLP(input_dim, hidden_dim, hidden_dim, num_layers, dropout)
        else:
            layers = [input_dim] + [hidden_dim] * num_layers
            self.encoder = MLP(layers, dropout)
        
        if use_vae:
            self.mu_head = nn.Linear(hidden_dim, output_dim)
            self.var_head = nn.Linear(hidden_dim, output_dim)
            nn.init.constant_(self.var_head.bias, 0.5)
        else:
            self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        self.apply(weight_init)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        h = self.encoder(x)
        
        if self.use_vae:
            mu = self.mu_head(h)
            var = F.softplus(self.var_head(h)) + self.var_eps
            std = torch.sqrt(var)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z, mu, var
        else:
            z = self.output_proj(h)
            return (z)


class MLPLogisticNormalEncoder(nn.Module):
    """MLP encoder for Topic Model (outputs mu, var for logistic-normal)."""
    def __init__(
        self,
        input_dim: int,
        n_topics: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        var_eps: float = 1e-4,
        use_residual: bool = False):
        super().__init__()
        self.var_eps = var_eps
        
        if use_residual:
            self.encoder = ResidualMLP(input_dim, hidden_dim, hidden_dim, num_layers, dropout)
        else:
            layers = [input_dim] + [hidden_dim] * num_layers
            self.encoder = MLP(layers, dropout)
        
        self.mu_head = nn.Linear(hidden_dim, n_topics)
        self.var_head = nn.Linear(hidden_dim, n_topics)
        nn.init.constant_(self.var_head.bias, 0.5)
        
        self.apply(weight_init)
    
    def forward(self, x: torch.Tensor, x_norm: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x_input = x_norm if x_norm is not None else torch.log1p(x)
        h = self.encoder(x_input)
        
        mu = self.mu_head(h)
        var = F.softplus(self.var_head(h)) + self.var_eps
        
        return mu, var


# =============================================================================
# MULTI-HEAD PROJECTION TRANSFORMER
# =============================================================================

class MultiHeadProjectionEncoder(nn.Module):
    """
    Multi-Head Projection Transformer Encoder.
    
    Architecture:
    1. K parallel projections: gene vector -> K x d_model tokens
    2. Transformer self-attention across K tokens
    3. Learnable aggregation -> latent space
    
    This design prevents the single-token collapse problem.
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        output_dim: int = 32,
        num_tokens: int = 8,
        use_vae: bool = False,
        var_eps: float = 1e-4):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        self.use_vae = use_vae
        self.var_eps = var_eps
        
        # Multiple projection heads
        self.projection_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout))
            for _ in range(num_tokens)
        ])
        
        # Token type embeddings
        self.token_embeddings = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Aggregation
        self.aggregation_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Output
        self.final_norm = nn.LayerNorm(d_model)
        if use_vae:
            self.mu_head = nn.Linear(d_model, output_dim)
            self.var_head = nn.Linear(d_model, output_dim)
            nn.init.constant_(self.var_head.bias, 0.5)
        else:
            self.output_proj = nn.Linear(d_model, output_dim)
        
        self.apply(weight_init)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        batch_size = x.size(0)
        
        # Create K tokens
        tokens = torch.stack([proj(x) for proj in self.projection_heads], dim=1)
        tokens = tokens + self.token_embeddings.expand(batch_size, -1, -1)
        
        # Transform
        tokens = self.transformer(tokens)
        
        # Aggregate
        query = self.aggregation_query.expand(batch_size, -1, -1)
        h, _ = self.cross_attention(query, tokens, tokens)
        h = h.squeeze(1)
        h = self.final_norm(h)
        
        if self.use_vae:
            mu = self.mu_head(h)
            var = F.softplus(self.var_head(h)) + self.var_eps
            std = torch.sqrt(var)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z, mu, var
        else:
            z = self.output_proj(h)
            return (z)


class MultiHeadProjectionLogisticNormalEncoder(nn.Module):
    """Multi-Head Projection Encoder for Topic Models."""
    def __init__(
        self,
        input_dim: int,
        n_topics: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_tokens: int = 8,
        var_eps: float = 1e-4):
        super().__init__()
        self.var_eps = var_eps
        self.num_tokens = num_tokens
        
        # Multiple projection heads
        self.projection_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout))
            for _ in range(num_tokens)
        ])
        
        self.token_embeddings = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.aggregation_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.final_norm = nn.LayerNorm(d_model)
        self.mu_head = nn.Linear(d_model, n_topics)
        self.var_head = nn.Linear(d_model, n_topics)
        nn.init.constant_(self.var_head.bias, 0.5)
        
        self.apply(weight_init)
    
    def forward(self, x: torch.Tensor, x_norm: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x_input = x_norm if x_norm is not None else torch.log1p(x)
        batch_size = x_input.size(0)
        
        tokens = torch.stack([proj(x_input) for proj in self.projection_heads], dim=1)
        tokens = tokens + self.token_embeddings.expand(batch_size, -1, -1)
        tokens = self.transformer(tokens)
        
        query = self.aggregation_query.expand(batch_size, -1, -1)
        h, _ = self.cross_attention(query, tokens, tokens)
        h = h.squeeze(1)
        h = self.final_norm(h)
        
        mu = self.mu_head(h)
        var = F.softplus(self.var_head(h)) + self.var_eps
        
        return mu, var


# =============================================================================
# HYBRID MLP-ATTENTION ENCODER
# =============================================================================

class HybridMLPAttentionEncoder(nn.Module):
    """
    Hybrid MLP + Attention Encoder.
    
    Architecture:
    1. MLP backbone produces base embedding
    2. Multi-head attention refines across batch
    3. Residual connection preserves MLP stability
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 32,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
        use_vae: bool = False,
        var_eps: float = 1e-4,
        attention_weight: float = 0.3):
        super().__init__()
        self.use_vae = use_vae
        self.var_eps = var_eps
        self.attention_weight = attention_weight
        
        # MLP backbone
        self.mlp_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout))
        
        # Attention refinement
        self.attention = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # Output
        if use_vae:
            self.mu_head = nn.Linear(hidden_dim, output_dim)
            self.var_head = nn.Linear(hidden_dim, output_dim)
            nn.init.constant_(self.var_head.bias, 0.5)
        else:
            self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        self.apply(weight_init)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        h_mlp = self.mlp_encoder(x)
        
        h_seq = h_mlp.unsqueeze(0)
        h_attn, _ = self.attention(h_seq, h_seq, h_seq)
        h_attn = h_attn.squeeze(0)
        
        h = h_mlp + self.attention_weight * (h_attn - h_mlp)
        h = self.attn_norm(h)
        
        if self.use_vae:
            mu = self.mu_head(h)
            var = F.softplus(self.var_head(h)) + self.var_eps
            std = torch.sqrt(var)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z, mu, var
        else:
            z = self.output_proj(h)
            return (z)


class HybridMLPAttentionLogisticNormalEncoder(nn.Module):
    """Hybrid encoder for topic models."""
    def __init__(
        self,
        input_dim: int,
        n_topics: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
        var_eps: float = 1e-4,
        attention_weight: float = 0.3):
        super().__init__()
        self.var_eps = var_eps
        self.attention_weight = attention_weight
        
        self.mlp_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout))
        
        self.attention = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        self.mu_head = nn.Linear(hidden_dim, n_topics)
        self.var_head = nn.Linear(hidden_dim, n_topics)
        nn.init.constant_(self.var_head.bias, 0.5)
        
        self.apply(weight_init)
    
    def forward(self, x: torch.Tensor, x_norm: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x_input = x_norm if x_norm is not None else torch.log1p(x)
        
        h_mlp = self.mlp_encoder(x_input)
        h_seq = h_mlp.unsqueeze(0)
        h_attn, _ = self.attention(h_seq, h_seq, h_seq)
        h_attn = h_attn.squeeze(0)
        
        h = h_mlp + self.attention_weight * (h_attn - h_mlp)
        h = self.attn_norm(h)
        
        mu = self.mu_head(h)
        var = F.softplus(self.var_head(h)) + self.var_eps
        
        return mu, var


# =============================================================================
# GAT (GRAPH ATTENTION NETWORK) ENCODER
# =============================================================================

try:
    from torch_geometric.nn import GATConv
    _HAS_PYGEOMETRIC = True
except ImportError:
    _HAS_PYGEOMETRIC = False


class GATLogisticNormalEncoder(nn.Module):
    """Graph Attention Network encoder for topic models.

    Uses GATConv layers over a precomputed k-NN graph.
    """
    def __init__(self, input_dim, n_topics, hidden_dim=128, num_layers=2,
                 nhead=4, dropout=0.1, var_eps=1e-4):
        super().__init__()
        if not _HAS_PYGEOMETRIC:
            raise ImportError("torch_geometric is required for GAT encoder. Install it with: pip install torch-geometric")
        self.var_eps = var_eps
        self.dropout = dropout

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        for _ in range(num_layers):
            out_channels = hidden_dim // nhead
            self.gat_layers.append(
                GATConv(hidden_dim, out_channels, heads=nhead, dropout=dropout, concat=True))
            self.gat_norms.append(nn.LayerNorm(hidden_dim))

        self.mu_head = nn.Linear(hidden_dim, n_topics)
        self.var_head = nn.Linear(hidden_dim, n_topics)
        nn.init.constant_(self.var_head.bias, 0.5)
        self.apply(weight_init)

    def forward(self, x, x_norm=None, edge_index=None, edge_weight=None):
        x_input = x_norm if x_norm is not None else torch.log1p(x)
        h = self.input_proj(x_input)

        if edge_index is None:
            raise ValueError("GAT encoder requires edge_index. Precompute k-NN graph first.")

        for gat_layer, norm in zip(self.gat_layers, self.gat_norms):
            residual = h
            h = gat_layer(h, edge_index)
            h = norm(h)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + residual

        mu = self.mu_head(h)
        var = F.softplus(self.var_head(h)) + self.var_eps
        return mu, var


# =============================================================================
# ENCODER FACTORY
# =============================================================================

def create_encoder(
    encoder_type: Literal['mlp', 'transformer', 'hybrid'],
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 128,
    use_vae: bool = False,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    var_eps: float = 1e-4,
    num_tokens: int = 8,
    attention_weight: float = 0.3) -> nn.Module:
    """
    Factory function to create encoder.
    
    Args:
        encoder_type: 
            - 'mlp': Simple MLP (fastest)
            - 'transformer': Multi-head projection transformer
            - 'hybrid': Hybrid MLP + attention
        input_dim: Number of input features (genes)
        output_dim: Latent dimension
        hidden_dim: Hidden dimension for MLP/hybrid
        use_vae: Whether to output (z, mu, var) or (z)
        d_model: Transformer model dimension
        nhead: Number of attention heads
        num_layers: Number of layers
        dim_feedforward: Transformer feedforward dimension
        dropout: Dropout rate
        var_eps: Variance epsilon
        num_tokens: Number of projection heads (transformer)
        attention_weight: Attention blend ratio (hybrid)
    
    Returns:
        Encoder module
    """
    if encoder_type == 'mlp':
        return MLPEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_vae=use_vae,
            var_eps=var_eps)
    elif encoder_type == 'transformer':
        return MultiHeadProjectionEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            output_dim=output_dim,
            num_tokens=num_tokens,
            use_vae=use_vae,
            var_eps=var_eps)
    elif encoder_type == 'hybrid':
        return HybridMLPAttentionEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout,
            use_vae=use_vae,
            var_eps=var_eps,
            attention_weight=attention_weight)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Use 'mlp', 'transformer', or 'hybrid'.")


def create_topic_encoder(
    encoder_type: Literal['mlp', 'transformer', 'hybrid', 'gat'],
    input_dim: int,
    n_topics: int,
    hidden_dim: int = 128,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    var_eps: float = 1e-4,
    num_tokens: int = 8,
    attention_weight: float = 0.3) -> nn.Module:
    """
    Factory function to create encoder for topic models.
    
    Returns encoder outputting (mu, var) for logistic-normal distribution.
    """
    if encoder_type == 'mlp':
        return MLPLogisticNormalEncoder(
            input_dim=input_dim,
            n_topics=n_topics,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            var_eps=var_eps)
    elif encoder_type == 'transformer':
        return MultiHeadProjectionLogisticNormalEncoder(
            input_dim=input_dim,
            n_topics=n_topics,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_tokens=num_tokens,
            var_eps=var_eps)
    elif encoder_type == 'hybrid':
        return HybridMLPAttentionLogisticNormalEncoder(
            input_dim=input_dim,
            n_topics=n_topics,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout,
            var_eps=var_eps,
            attention_weight=attention_weight)
    elif encoder_type == 'gat':
        return GATLogisticNormalEncoder(
            input_dim=input_dim, n_topics=n_topics,
            hidden_dim=hidden_dim or d_model, num_layers=num_layers,
            nhead=nhead, dropout=dropout, var_eps=var_eps)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Use 'mlp', 'transformer', 'hybrid', or 'gat'.")
