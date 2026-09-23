"""
LDAODETransformer: Topic model with Transformer encoder.

Supports TWO transformer approaches:
1. CELL-AS-TOKEN (default, iAODE-style): Efficient O(batch_size) attention
2. GENE-AS-TOKEN (Geneformer-style): O(n_genes²) but interpretable

Single-phase training:
- Phase 1: Train Transformer VAE-based topic model (fit method)


Uses shared modules from:
- shared_modules.py: InformationBottleneck, TopicDecoder
- encoders.py: CellAsTokenLogisticNormalEncoder, GeneAsTokenLogisticNormalEncoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, Literal
from torch.distributions import Normal
import math
import numpy as np

try:
    from .base_model import BaseModel
    from .mixins import PriorMixin, ReconstructionLossMixin
    from .shared_modules import (
        weight_init, InformationBottleneck, TopicDecoder,
        reparameterize, log_to_simplex)
    from .encoders import (
        MultiHeadProjectionLogisticNormalEncoder, HybridMLPAttentionLogisticNormalEncoder,
        MLPLogisticNormalEncoder, create_topic_encoder)
except ImportError:
    from utils.base_model import BaseModel
    from utils.mixins import PriorMixin, ReconstructionLossMixin
    from models.shared_modules import (
        weight_init, InformationBottleneck, TopicDecoder,
        reparameterize, log_to_simplex)
    from models.encoders import (
        MultiHeadProjectionLogisticNormalEncoder, HybridMLPAttentionLogisticNormalEncoder,
        MLPLogisticNormalEncoder, create_topic_encoder)


class TopicTransformerAutoEncoder(nn.Module):
    """Transformer-based topic model autoencoder with optional ODE."""
    def __init__(
        self,
        n_words: int,
        n_topics: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        use_bottleneck: bool = False,
        bottleneck_dim: Optional[int] = None,
        encoder_type: Literal['transformer', 'hybrid', 'mlp'] = 'transformer'):
        super().__init__()
        self.n_words = n_words
        self.n_topics = n_topics
        self.use_bottleneck = use_bottleneck
        self.encoder_type = encoder_type
        
        # Create encoder using factory
        self.encoder = create_topic_encoder(
            encoder_type=encoder_type,
            input_dim=n_words,
            n_topics=n_topics,
            hidden_dim=d_model,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout)
        
        # Decoder
        self.decoder = TopicDecoder(n_topics, n_words)

        if use_bottleneck:
            if bottleneck_dim is None:
                bottleneck_dim = max(n_topics // 2, 8)
            self.bottleneck = InformationBottleneck(n_topics, bottleneck_dim, drop=dropout)
        
        self.apply(weight_init)
    
    def log_to_simplex(self, log_theta: torch.Tensor) -> torch.Tensor:
        return F.softmax(log_theta, dim=-1)
    
    def forward(self, x: torch.Tensor, x_norm: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        return self._forward_normal(x, x_norm)
    
    def _forward_normal(self, x: torch.Tensor, x_norm: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        mu, var = self.encoder(x, x_norm)
        log_theta = reparameterize(mu, var)
        theta = self.log_to_simplex(log_theta)
        x_recon = self.decoder(theta)

        result = {
            "x": x,
            "x_recon": x_recon,
            "mu": mu,
            "var": var,
            "log_theta": log_theta,
            "theta": theta,
        }

        if self.use_bottleneck:
            log_theta_le, log_theta_ld = self.bottleneck(log_theta)
            theta_ld = self.log_to_simplex(log_theta_ld)
            x_recon_bottleneck = self.decoder(theta_ld)
            result.update({
                "log_theta_le": log_theta_le,
                "log_theta_bottleneck": log_theta_ld,
                "x_recon_bottleneck": x_recon_bottleneck,
            })

        return result
    

class TopicODETransformerModel(PriorMixin, ReconstructionLossMixin, BaseModel):
    """LDAODETransformer: Transformer-based topic model with optional ODE.
    
    Uses efficient Cell-as-Token transformer (iAODE-style) with O(batch_size) attention.
    """
    def __init__(
        self,
        input_dim: int,
        n_topics: int = 10,
        latent_dim: Optional[int] = None,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        encoder_type: Literal['transformer', 'hybrid', 'mlp'] = 'transformer',
        use_bottleneck: bool = False,
        bottleneck_dim: Optional[int] = None,
        sparsity_strength: float = 10.0,
        cell_topic_prior: Optional[torch.Tensor] = None,
        topic_word_prior: Optional[torch.Tensor] = None,
        kl_weight: float = 0.01,
        reconstruction_loss: str = "multinomial",
        model_name: str = "LDAODETransformer",
        use_raw_counts: bool = True,
):
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim or n_topics,
            hidden_dims=[d_model],
            model_name=model_name,
            use_raw_counts=use_raw_counts)
        
        self.n_words = input_dim
        self.n_topics = n_topics
        self.kl_weight = kl_weight
        self.reconstruction_loss = reconstruction_loss
        self.encoder_type = encoder_type
        
        # Initialize autoencoder
        self.ae = TopicTransformerAutoEncoder(
            n_words=input_dim,
            n_topics=n_topics,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_bottleneck=use_bottleneck,
            bottleneck_dim=bottleneck_dim,
            encoder_type=encoder_type)
        
        # Priors
        if cell_topic_prior is None:
            cell_topic_prior = torch.ones(n_topics) / n_topics * sparsity_strength
        if topic_word_prior is None:
            topic_word_prior = torch.ones(input_dim) / n_topics * sparsity_strength
        
        prior_mu_topics, prior_sigma_topics = self._dirichlet_to_logistic_normal(cell_topic_prior)
        self.register_buffer("prior_mu_topics", prior_mu_topics)
        self.register_buffer("prior_sigma_topics", prior_sigma_topics)
        
        prior_mu_words, prior_sigma_words = self._dirichlet_to_logistic_normal(topic_word_prior)
        self.register_buffer("prior_mu_words", prior_mu_words)
        self.register_buffer("prior_sigma_words", prior_sigma_words)
    
    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode to topic distribution (theta)."""
        x_norm = kwargs.get('x_norm', None)
        mu, var = self.ae.encoder(x, x_norm)
        log_theta = mu
        theta = self.ae.log_to_simplex(log_theta)
        return theta
    
    def encode_log_theta(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x_norm = kwargs.get('x_norm', None)
        mu, var = self.ae.encoder(x, x_norm)
        return mu
    

    def extract_latent(self, data_loader, device='cuda', return_reconstructions: bool = False,
                       return_log_theta: bool = False):
        self.eval()
        self.to(device)
        latents, recons = [], []
        
        with torch.no_grad():
            for batch_data in data_loader:
                x, batch_kwargs = self._prepare_batch(batch_data, device)
                
                if return_log_theta:
                    z = self.encode_log_theta(x, **batch_kwargs)
                else:
                    z = self.encode(x, **batch_kwargs)
                
                latents.append(z.cpu().numpy())
                
                if return_reconstructions:
                    theta = z if not return_log_theta else self.ae.log_to_simplex(z)
                    recons.append(self.decode(theta).cpu().numpy())
        
        result = {"latent": np.concatenate(latents, axis=0)}
        if return_reconstructions:
            result["reconstruction"] = np.concatenate(recons, axis=0)
        return result
    
    def decode(self, theta: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.ae.decoder(theta)
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        x_norm = kwargs.get('x_norm', None)
        return self.ae(x, x_norm)
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        if self.reconstruction_loss == "multinomial":
            recon_main = self._multinomial_nll(outputs["x"], outputs["x_recon"])
            if "x_recon_bottleneck" in outputs:
                recon_bottleneck = self._multinomial_nll(outputs["x"], outputs["x_recon_bottleneck"])
                recon_loss = 0.5 * (recon_main + recon_bottleneck)
            else:
                recon_loss = recon_main
        elif self.reconstruction_loss == "kl":
            recon_main = self._kl_divergence(outputs["x"], outputs["x_recon"])
            if "x_recon_bottleneck" in outputs:
                recon_bottleneck = self._kl_divergence(outputs["x"], outputs["x_recon_bottleneck"])
                recon_loss = 0.5 * (recon_main + recon_bottleneck)
            else:
                recon_loss = recon_main
        else:
            raise ValueError(f"Unknown reconstruction_loss: {self.reconstruction_loss}")

        # KL divergence with free bits (anti-collapse)
        kl_loss = self._kl_logistic_normal_free_bits(
            outputs["mu"], outputs["var"],
            self.prior_mu_topics, self.prior_sigma_topics ** 2,
            free_bits=0.1)

        total_loss = recon_loss + self.kl_weight * kl_loss

        loss_dict = {"total_loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}
        if "x_recon_bottleneck" in outputs:
            loss_dict["recon_bottleneck"] = recon_bottleneck
        return loss_dict

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 1000,
        lr: float = 0.001,
        device: str = "cuda",
        save_path: Optional[str] = None,
        patience: int = 50,
        verbose: int = 1,
        verbose_every: int = 1,
        weight_decay: float = 1e-3,
        **kwargs):
        """Phase 1: Train Transformer VAE-based LDA.
        
        Uses fixed KL weight (no annealing) and fixed learning rate (no scheduler).
        
        Args:
            weight_decay: AdamW weight decay (L2 regularization).
        """
        self.to(device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        best_loss = float('inf')
        patience_counter = 0
        train_losses, recon_losses, kl_losses = [], [], []

        if verbose_every is None or verbose_every < 1:
            verbose_every = 1

        for epoch in range(epochs):
            self.train()
            
            epoch_loss, epoch_recon, epoch_kl = 0.0, 0.0, 0.0
            n_batches = 0

            for batch in train_loader:
                x, batch_kwargs = self._prepare_batch(batch, device)
                optimizer.zero_grad()
                out = self.forward(x, **batch_kwargs, **kwargs)
                loss_dict = self.compute_loss(out, **batch_kwargs, **kwargs)
                loss = loss_dict["total_loss"]
                
                if not torch.isfinite(loss):
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_recon += loss_dict["recon_loss"].item()
                epoch_kl += loss_dict["kl_loss"].item()
                n_batches += 1

            if n_batches == 0:
                continue

            avg_loss = epoch_loss / n_batches
            avg_recon = epoch_recon / n_batches
            avg_kl = epoch_kl / n_batches
            
            train_losses.append(avg_loss)
            recon_losses.append(avg_recon)
            kl_losses.append(avg_kl)

            do_print = (verbose >= 1) and (((epoch + 1) % verbose_every == 0) or (epoch == 0) or (epoch + 1 == epochs))

            if do_print:
                print(f"Epoch {epoch+1:3d}/{epochs} [Phase1-Transformer-LDA] | "
                      f"Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                if save_path:
                    torch.save(self.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose >= 1:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

        return {"train_loss": train_losses, "recon_loss": recon_losses, "kl_loss": kl_losses}
    
    def _compute_loss_with_kl_weight(self, outputs: Dict[str, torch.Tensor], kl_weight: float, **kwargs) -> Dict[str, torch.Tensor]:
        """Compute loss with a specific KL weight (for KL annealing)."""
        if self.reconstruction_loss == "multinomial":
            recon_main = self._multinomial_nll(outputs["x"], outputs["x_recon"])
            if "x_recon_bottleneck" in outputs:
                recon_bottleneck = self._multinomial_nll(outputs["x"], outputs["x_recon_bottleneck"])
                recon_loss = 0.5 * (recon_main + recon_bottleneck)
            else:
                recon_loss = recon_main
        elif self.reconstruction_loss == "kl":
            recon_main = self._kl_divergence(outputs["x"], outputs["x_recon"])
            if "x_recon_bottleneck" in outputs:
                recon_bottleneck = self._kl_divergence(outputs["x"], outputs["x_recon_bottleneck"])
                recon_loss = 0.5 * (recon_main + recon_bottleneck)
            else:
                recon_loss = recon_main
        else:
            raise ValueError(f"Unknown reconstruction_loss: {self.reconstruction_loss}")
        
        # Use free bits KL to prevent posterior collapse
        # Each dimension must contribute at least 0.1 nats to KL
        kl_loss = self._kl_logistic_normal_free_bits(
            outputs["mu"], outputs["var"],
            self.prior_mu_topics, self.prior_sigma_topics ** 2,
            free_bits=0.1
        )
        
        total_loss = recon_loss + kl_weight * kl_loss
        
        loss_dict = {"total_loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}
        if "x_recon_bottleneck" in outputs:
            loss_dict["recon_bottleneck"] = recon_bottleneck
        return loss_dict


    def get_topic_word_distribution(self) -> torch.Tensor:
        return self.ae.decoder.beta.detach()
    
    def get_top_words_per_topic(self, vocab: list, n_words: int = 10) -> Dict[int, list]:
        beta = self.get_topic_word_distribution().cpu().numpy()
        result = {}
        for topic_id in range(self.n_topics):
            top_idx = np.argsort(beta[topic_id])[::-1][:n_words]
            result[topic_id] = [(vocab[idx], beta[topic_id, idx]) for idx in top_idx]
        return result


def create_topic_ode_transformer_model(
    n_words: int,
    n_topics: int = 10,
    **kwargs
) -> TopicODETransformerModel:
    """
    Create TopicODETransformer (LDAODETransformer) model.
    
    Args:
        n_words: Vocabulary size (input dimension)
        n_topics: Number of topics (latent dimension)
    
    Critical Parameters:
        d_model: Transformer model dimension (default: 128)
        nhead: Number of attention heads (default: 4)
        num_encoder_layers: Transformer layers (default: 2)
        kl_weight: KL divergence weight (0.1-2.0)
    """
    return TopicODETransformerModel(
        input_dim=n_words,
        n_topics=n_topics,
        **kwargs
    )
