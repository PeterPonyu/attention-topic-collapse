"""
Shared Modules for PanODE-Topic Models.

Reusable components shared across Topic-FM and Pure-VAE models:
- Weight initialization
- Common layers (MLP, ResidualMLP, Bottleneck)
- Decoder architectures (TopicDecoder, MLPDecoder)
- Graph utilities (SubgraphDataset for efficient graph sampling)
- Reparameterization utilities
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from torch.utils.data import Dataset


# =============================================================================
# Weight Initialization
# =============================================================================

def weight_init(m):
    """Xavier normal initialization for linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=0.02)


# =============================================================================
# Common MLP and Bottleneck
# =============================================================================

class MLP(nn.Module):
    """Simple MLP with LayerNorm and GELU activation."""
    def __init__(self, features: list, dropout: float = 0.1, use_layer_norm: bool = True):
        super().__init__()
        layers = []
        for i in range(1, len(features)):
            layers.append(nn.Linear(features[i-1], features[i]))
            if i < len(features) - 1:
                if use_layer_norm:
                    layers.append(nn.LayerNorm(features[i]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.apply(weight_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualMLP(nn.Module):
    """MLP with residual connections (iAODE-style)."""
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        num_layers: int = 2,
        dropout: float = 0.1,
        use_batch_norm: bool = True):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            else:
                self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.apply(weight_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.input_proj(x))
        
        for layer, norm in zip(self.layers, self.norms):
            residual = h
            h = layer(h)
            h = norm(h)
            h = self.activation(h)
            h = self.dropout(h)
            h = h + residual  # Residual connection
        
        return self.output_proj(h)


class InformationBottleneck(nn.Module):
    """Compress and reconstruct latent space."""
    def __init__(self, latent_dim: int, bottleneck_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = max(latent_dim // 2, bottleneck_dim)
        self.compress = MLP([latent_dim, hidden_dim, bottleneck_dim], dropout=dropout)
        self.expand = MLP([bottleneck_dim, hidden_dim, latent_dim], dropout=dropout)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_bottleneck = self.compress(z)
        z_reconstructed = self.expand(z_bottleneck)
        return z_bottleneck, z_reconstructed


class MLPDecoder(nn.Module):
    """Standard MLP decoder for reconstruction."""
    def __init__(
        self, 
        latent_dim: int, 
        hidden_dim: int, 
        output_dim: int,
        dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim))
        self.apply(weight_init)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TopicDecoder(nn.Module):
    """Decoder: theta (topic distribution) -> word distribution."""
    def __init__(self, n_topics: int, n_words: int):
        super().__init__()
        self.n_topics = n_topics
        self.n_words = n_words
        self.beta_logit = nn.Parameter(torch.randn(n_topics, n_words))
        self.apply(weight_init)
    
    @property
    def beta(self) -> torch.Tensor:
        """Topic-word probability matrix."""
        return F.softmax(self.beta_logit, dim=1)
    
    @property
    def log_beta(self) -> torch.Tensor:
        """Log topic-word probability matrix."""
        return F.log_softmax(self.beta_logit, dim=1)
    
    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """Reconstruct word distribution from topic distribution."""
        log_theta = torch.log(theta + 1e-10)
        log_beta = self.log_beta.T.unsqueeze(0)
        log_prob = torch.logsumexp(log_theta.unsqueeze(1) + log_beta, dim=2)
        return torch.exp(log_prob)


# =============================================================================
# Graph Utilities (CCVGAE-style)
# =============================================================================

class SubgraphDataset(Dataset):
    """
    Dataset for subgraph sampling from precomputed k-NN graph (CCVGAE-style).
    
    This enables efficient training on large graphs by:
    1. Building the k-NN graph ONCE for the entire dataset
    2. Sampling smaller subgraphs during training
    
    Much faster than computing k-NN graph per batch!
    """
    def __init__(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edge_weight: np.ndarray,
        device: torch.device,
        subgraph_size: int = 512):
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.device = device
        self.subgraph_size = subgraph_size
        self.num_nodes = node_features.shape[0]
        self.neighbors = self._compute_neighbors()

    def _compute_neighbors(self) -> List[List[int]]:
        """Precompute neighbor lists for each node."""
        neighbors = [[] for _ in range(self.num_nodes)]
        for i, j in self.edge_index.T:
            neighbors[i].append(j)
            if i != j:
                neighbors[j].append(i)
        return neighbors

    def __len__(self) -> int:
        return max(1, self.num_nodes // self.subgraph_size * 2)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Generate a subgraph sample."""
        selected_nodes = self._random_node_sampling()
        return self._create_data_object(selected_nodes)

    def _random_node_sampling(self) -> np.ndarray:
        """Randomly sample nodes for subgraph creation."""
        num_sample = min(self.subgraph_size, self.num_nodes)
        return np.random.choice(self.num_nodes, size=num_sample, replace=False)

    def _create_data_object(self, selected_nodes: np.ndarray) -> Dict[str, torch.Tensor]:
        """Create data dict from selected nodes."""
        node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_nodes)}
        
        # Filter edges within subgraph
        edge_mask = np.isin(self.edge_index[0], selected_nodes) & np.isin(self.edge_index[1], selected_nodes)
        subgraph_edges = self.edge_index[:, edge_mask]
        subgraph_weights = self.edge_weight[edge_mask]
        
        # Remap edge indices
        new_edge_index = np.array([
            [node_map[i] for i in subgraph_edges[0]],
            [node_map[i] for i in subgraph_edges[1]]
        ]) if subgraph_edges.size > 0 else np.array([[], []], dtype=np.int64)
        
        # Build adjacency matrix
        n_sub = len(selected_nodes)
        adj = np.zeros((n_sub, n_sub), dtype=np.float32)
        if new_edge_index.size > 0:
            adj[new_edge_index[0], new_edge_index[1]] = subgraph_weights
        np.fill_diagonal(adj, 1.0)
        adj = np.maximum(adj, adj.T)
        
        return {
            'x': torch.tensor(self.node_features[selected_nodes], dtype=torch.float, device=self.device),
            'adj': torch.tensor(adj, dtype=torch.float, device=self.device),
            'original_idx': torch.tensor(selected_nodes, dtype=torch.long, device=self.device),
        }


def precompute_knn_graph(
    x_full: np.ndarray, 
    k: int = 10, 
    metric: str = 'euclidean',
    pca_dim: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute k-NN graph for entire dataset ONCE.
    
    Args:
        x_full: Features [n_samples, n_features]
        k: Number of neighbors
        metric: Distance metric
        pca_dim: Optional PCA reduction before computing k-NN
        
    Returns:
        edge_index: [2, n_edges] COO format
        edge_weight: [n_edges] edge weights
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.decomposition import PCA
    
    n_samples = x_full.shape[0]
    k_actual = min(k, n_samples - 1)
    
    # Optional PCA reduction
    x_for_graph = x_full
    if pca_dim is not None and x_full.shape[1] > pca_dim:
        max_pca_dim = min(pca_dim, n_samples - 1, x_full.shape[1])
        if max_pca_dim > 0:
            x_for_graph = PCA(n_components=max_pca_dim, random_state=42).fit_transform(x_full)
    
    # Compute k-NN
    nn = NearestNeighbors(n_neighbors=k_actual + 1, metric=metric, n_jobs=-1)
    nn.fit(x_for_graph)
    distances, indices = nn.kneighbors(x_for_graph)
    
    # Build edge_index and edge_weight
    rows, cols, weights = [], [], []
    for i in range(n_samples):
        for j_idx, j in enumerate(indices[i]):
            if i != j:
                rows.append(i)
                cols.append(j)
                weights.append(1.0)
    
    edge_index = np.array([rows, cols], dtype=np.int64)
    edge_weight = np.array(weights, dtype=np.float32)
    
    return edge_index, edge_weight


# =============================================================================
# Reparameterization Utilities
# =============================================================================

def reparameterize(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Reparameterization trick for VAE."""
    std = torch.sqrt(var)
    eps = torch.randn_like(std)
    return mu + eps * std


def log_to_simplex(log_theta: torch.Tensor) -> torch.Tensor:
    """Convert log-space to simplex (probability distribution)."""
    return F.softmax(log_theta, dim=-1)
