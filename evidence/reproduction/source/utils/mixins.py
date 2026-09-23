import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from sklearn.mixture import BayesianGaussianMixture
import math


class PriorMixin:
    """Logistic-normal prior for topic models"""

    @staticmethod
    def _dirichlet_to_logistic_normal(alpha: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert Dirichlet prior to logistic-normal prior parameters.

        Uses the digamma/trigamma moment-matching approximation:
            mu_k = psi(alpha_k) - (1/K) * sum_j psi(alpha_j)
            sigma_k = sqrt( psi'(alpha_k) - (1/K^2) * sum_j psi'(alpha_j) )

        where psi is the digamma function and psi' is the trigamma function.
        This is more accurate than the log-based approximation, especially
        for small concentration parameters.
        """
        K = alpha.shape[0]
        # Digamma: psi(alpha)
        psi = torch.digamma(alpha)
        mu = psi - psi.sum() / K

        # Trigamma: psi'(alpha) — use torch.special.polygamma(1, x)
        trigamma = torch.special.polygamma(1, alpha)
        var_k = trigamma - trigamma.sum() / (K ** 2)
        # Clamp variance to be positive (numerical safety)
        var_k = torch.clamp(var_k, min=1e-6)
        sigma = torch.sqrt(var_k)
        return mu, sigma
    
    def _kl_logistic_normal(self, mu_q, var_q, mu_p, var_p) -> torch.Tensor:
        """KL divergence between two logistic-normal distributions"""
        if mu_p.dim() == 1:
            mu_p = mu_p.unsqueeze(0)
            var_p = var_p.unsqueeze(0)
        
        # Ensure same device
        mu_p = mu_p.to(mu_q.device)
        var_p = var_p.to(mu_q.device)
        
        kl = 0.5 * torch.sum(
            var_q / (var_p + 1e-10) + (mu_p - mu_q) ** 2 / (var_p + 1e-10) - 1 +
            torch.log(var_p + 1e-10) - torch.log(var_q + 1e-10), dim=1
        )
        return kl.mean()
    
    def _kl_logistic_normal_free_bits(self, mu_q, var_q, mu_p, var_p, free_bits: float = 0.5) -> torch.Tensor:
        """KL divergence with anti-collapse regularization.
        
        Uses a penalty when KL per dimension is below threshold to prevent collapse.
        
        Args:
            mu_q, var_q: Mean and variance of posterior
            mu_p, var_p: Mean and variance of prior
            free_bits: Target minimum KL per dimension (default 0.5 nats)
        
        Returns:
            KL divergence with anti-collapse regularization
        """
        if mu_p.dim() == 1:
            mu_p = mu_p.unsqueeze(0)
            var_p = var_p.unsqueeze(0)
        
        # Ensure same device
        mu_p = mu_p.to(mu_q.device)
        var_p = var_p.to(mu_q.device)
        
        # Per-dimension KL: [batch, latent_dim]
        kl_per_dim = 0.5 * (
            var_q / (var_p + 1e-10) + (mu_p - mu_q) ** 2 / (var_p + 1e-10) - 1 +
            torch.log(var_p + 1e-10) - torch.log(var_q + 1e-10)
        )
        
        # Average over batch: [latent_dim]
        kl_per_dim_avg = kl_per_dim.mean(dim=0)
        
        # Strong penalty for collapsed dimensions (KL < free_bits)
        # Use linear penalty with high weight
        penalty_weight = 50.0
        shortfall = torch.clamp(free_bits - kl_per_dim_avg, min=0)
        penalty = penalty_weight * shortfall.sum()
        
        return kl_per_dim_avg.sum() + penalty

    @staticmethod
    def resolve_dirichlet_alpha(
        n_topics: int,
        sparsity_strength: float,
        dirichlet_concentration: float | None = None,
        cell_topic_prior: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, str]:
        """Return the per-topic Dirichlet concentration and how it was obtained.

        The stock constructor used ``ones(K)/K * sparsity_strength``, which
        couples prior strength to capacity and pins the shipped ``K=10, s=10``
        configuration at the uniform simplex.  Passing
        ``dirichlet_concentration`` (or an explicit ``cell_topic_prior``)
        decouples the two axes.  The regime tag is recorded so a sweep cannot
        silently fall back to the legacy formula.
        """

        if cell_topic_prior is not None:
            return cell_topic_prior, "explicit"
        if dirichlet_concentration is not None:
            return (
                torch.ones(n_topics) * float(dirichlet_concentration),
                "decoupled",
            )
        return (
            torch.ones(n_topics) / n_topics * float(sparsity_strength),
            "legacy_s_over_k",
        )

    def topic_word_dirichlet_nll(self, log_beta: torch.Tensor) -> torch.Tensor:
        """Mean Dirichlet negative log-density of the topic-word rows.

        The corresponding prior buffer used to be registered and then ignored.
        This term is the missing piece: it is the only place the topic-word
        concentration actually enters the objective.  The constant
        ``log B(alpha)`` is omitted because it does not depend on parameters.
        """

        alpha = getattr(self, "topic_word_alpha", None)
        if alpha is None:
            raise RuntimeError("topic_word_alpha buffer is not registered")
        alpha = alpha.to(log_beta.device)
        return -((alpha - 1.0) * log_beta).sum(dim=1).mean()
    
    @staticmethod
    def _kl_gaussian_free_bits(mu: torch.Tensor, var: torch.Tensor, free_bits: float = 0.1) -> torch.Tensor:
        """KL divergence for standard Gaussian prior N(0,1) with soft penalty for low KL.
        
        Uses a soft penalty approach to prevent posterior collapse by encouraging
        each latent dimension to have KL above free_bits threshold.
        
        Args:
            mu: Mean of posterior [batch, latent_dim]
            var: Variance of posterior [batch, latent_dim]
            free_bits: Target minimum KL per dimension (default 0.1 nats)
        
        Returns:
            KL divergence with soft penalty for low-KL dimensions
        """
        # Per-dimension KL for N(mu, var) vs N(0, 1): [batch, latent_dim]
        kl_per_dim = 0.5 * (var + mu ** 2 - 1.0 - torch.log(var + 1e-10))
        
        # Average over batch: [latent_dim]
        kl_per_dim_avg = kl_per_dim.mean(dim=0)
        
        # Soft penalty approach: encourage KL to be at least free_bits
        # When KL < free_bits, add a penalty proportional to the shortfall squared
        penalty_weight = 10.0  # Strength of the penalty
        shortfall = torch.clamp(free_bits - kl_per_dim_avg, min=0)
        penalty = penalty_weight * (shortfall ** 2).sum()
        
        return kl_per_dim_avg.sum() + penalty


class ReconstructionLossMixin:
    """Reconstruction losses for count data"""
    
    def _multinomial_nll(self, x, x_recon) -> torch.Tensor:
        """Multinomial negative log-likelihood (categorical cross-entropy)"""
        eps = 1e-10
        x_recon = torch.clamp(x_recon, min=eps, max=1.0 - eps)
        x_sum = x.sum(dim=1, keepdim=True)
        x_norm = x / (x_sum + eps)
        nll = -torch.sum(x_norm * torch.log(x_recon + eps), dim=1)
        return nll.mean()
    
    def _kl_divergence(self, x, x_recon) -> torch.Tensor:
        """Kullback-Leibler divergence"""
        eps = 1e-10
        x_sum = x.sum(dim=1, keepdim=True)
        x_norm = x / (x_sum + eps)
        x_recon = torch.clamp(x_recon, min=eps, max=1.0 - eps)
        kl = torch.sum(x_norm * (torch.log(x_norm + eps) - torch.log(x_recon + eps)), dim=1)
        return kl.mean()
    


class DPMMLossMixin:
    """DPMM clustering losses - ALIGNED WITH DPMMODE"""
    
    def _check_dpmm_fitted(self, loss_name: str) -> bool:
        """Safety check: DPMM must be fitted before computing DPMM loss"""
        if not self.use_dpmm:
            return False
        if not self.dpmm_fitted or self.dpmm_params is None:
            if hasattr(self, '_dpmm_warning_count'):
                self._dpmm_warning_count = getattr(self, '_dpmm_warning_count', 0) + 1
                if self._dpmm_warning_count <= 1:
                    print(f"Warning: {loss_name} called but DPMM not fitted. Skipping DPMM loss.")
            return False
        return True
    
    def _dpmm_loss_kl(self, z: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood under DPMM (Gaussian mixture)
        
        CRITICAL: Only uses weights and precisions_cholesky (diagonal case)
        Matches DPMMODE implementation exactly.
        """
        if not self._check_dpmm_fitted("_dpmm_loss_kl"):
            return torch.tensor(0.0, device=z.device)
        
        dp = self.dpmm_params
        n_features = z.size(1)
        eps = 1e-10

        # Get mixture weights and log weights (BUG #16 FIX: clamp weights)
        weights = torch.clamp(dp["weights"], min=eps, max=1.0)
        log_weights = torch.log(weights)

        # Compute log determinant of precision matrix (diagonal case)
        # BUG #16 FIX: Add clamping to precisions_cholesky
        prec_chol = torch.clamp(dp["precisions_cholesky"], min=eps, max=1e6)
        log_det = torch.sum(torch.log(prec_chol), dim=1)
        precisions = prec_chol ** 2

        # Compute Mahalanobis distance
        diff = z.unsqueeze(1) - dp["means"].unsqueeze(0)
        mahalanobis = torch.sum(diff ** 2 * precisions.unsqueeze(0), dim=2)
        # BUG #16 FIX: Clamp mahalanobis to prevent overflow
        mahalanobis = torch.clamp(mahalanobis, max=1e6)

        # Log probability per component (Gaussian)
        log_2pi = math.log(2.0 * math.pi)
        log_gauss = -0.5 * (n_features * log_2pi + mahalanobis) + log_det
        
        # Weighted log probability (mixture model)
        log_prob = torch.logsumexp(log_gauss + log_weights, dim=1)
        
        # Negative log-likelihood (BUG #16 FIX: clamp before AND after mean)
        nll = -log_prob
        nll = torch.clamp(nll, min=0.0, max=1e6)
        return nll.mean()
    
    def _dpmm_loss_nll(self, z: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood with stick-breaking weights (variational)"""
        if not self._check_dpmm_fitted("_dpmm_loss_nll"):
            return torch.tensor(0.0, device=z.device)
        
        dp = self.dpmm_params
        
        if "weight_concentration" not in dp:
            return self._dpmm_loss_kl(z)
        
        n_features = z.size(1)
        eps = 1e-10

        weight_conc = dp["weight_concentration"]
        
        if weight_conc.dim() == 1 and len(weight_conc) == 2:
            alpha0, alpha1 = weight_conc[0], weight_conc[1]
        else:
            return self._dpmm_loss_kl(z)
        
        # BUG #16 FIX: Clamp concentration parameters
        alpha0 = torch.clamp(alpha0, min=eps)
        alpha1 = torch.clamp(alpha1, min=eps)
        
        digamma_sum = torch.special.digamma(alpha0 + alpha1)
        digamma_a = torch.special.digamma(alpha0)
        digamma_b = torch.special.digamma(alpha1)

        log_weights_b = torch.cat(
            [torch.zeros(1, device=z.device), torch.cumsum(digamma_b - digamma_sum, dim=0)[:-1]]
        )
        log_weights = digamma_a - digamma_sum + log_weights_b

        # Compute log determinant (BUG #16 FIX: clamp precisions)
        prec_chol = torch.clamp(dp["precisions_cholesky"], min=eps, max=1e6)
        log_det = torch.sum(torch.log(prec_chol), dim=1)
        precisions = prec_chol ** 2

        # Mahalanobis distance
        diff = z.unsqueeze(1) - dp["means"].unsqueeze(0)
        mahalanobis = torch.sum(diff ** 2 * precisions.unsqueeze(0), dim=2)
        # BUG #16 FIX: Clamp mahalanobis
        mahalanobis = torch.clamp(mahalanobis, max=1e6)

        # Log Gaussian
        log_2pi = math.log(2.0 * math.pi)
        log_gauss = -0.5 * (n_features * log_2pi + mahalanobis) + log_det
        log_likelihood = torch.logsumexp(log_gauss + log_weights, dim=1)
        
        nll = -log_likelihood
        # BUG #16 FIX: Clamp before mean
        nll = torch.clamp(nll, min=0.0, max=1e6)
        return nll.mean()
    
    def _dpmm_loss_energy(self, z: torch.Tensor) -> torch.Tensor:
        """Energy-based distance to clusters"""
        if not self._check_dpmm_fitted("_dpmm_loss_energy"):
            return torch.tensor(0.0, device=z.device)
        
        dp = self.dpmm_params
        eps = 1e-10
        
        diff = z.unsqueeze(1) - dp["means"].unsqueeze(0)
        # BUG #16 FIX: Clamp precisions
        prec_chol = torch.clamp(dp["precisions_cholesky"], min=eps, max=1e6)
        precisions = prec_chol ** 2
        weighted_dist = torch.sum(diff ** 2 * precisions.unsqueeze(0), dim=2)
        # BUG #16 FIX: Clamp weighted_dist
        weighted_dist = torch.clamp(weighted_dist, min=0.0, max=1e6)
        
        temperature = 1.0
        soft_assign = torch.softmax(-weighted_dist / temperature, dim=1)
        energy = torch.sum(soft_assign * weighted_dist, dim=1).mean()
        
        return torch.clamp(energy, min=0.0, max=1e6)
    
    def _dpmm_loss_student_t(self, z: torch.Tensor) -> torch.Tensor:
        """Student's t-distribution loss"""
        if not self._check_dpmm_fitted("_dpmm_loss_student_t"):
            return torch.tensor(0.0, device=z.device)
        
        dp = self.dpmm_params
        n_features = z.size(1)
        df = self.student_t_df
        eps = 1e-10

        # BUG #16 FIX: Clamp weights
        weights = torch.clamp(dp["weights"], min=eps, max=1.0)
        log_weights = torch.log(weights)

        diff = z.unsqueeze(1) - dp["means"].unsqueeze(0)
        # BUG #16 FIX: Clamp precisions
        prec_chol = torch.clamp(dp["precisions_cholesky"], min=eps, max=1e6)
        precisions = prec_chol ** 2
        mahalanobis = torch.sum(diff ** 2 * precisions.unsqueeze(0), dim=2)
        # BUG #16 FIX: Clamp mahalanobis
        mahalanobis = torch.clamp(mahalanobis, min=0.0, max=1e6)

        log_det = torch.sum(torch.log(prec_chol), dim=1)
        
        log_student_t = (
            torch.lgamma(torch.tensor((df + n_features) / 2.0, device=z.device))
            - torch.lgamma(torch.tensor(df / 2.0, device=z.device))
            - (n_features / 2.0) * math.log(df * math.pi)
            + log_det
            - ((df + n_features) / 2.0) * torch.log(1.0 + mahalanobis / df)
        )
        
        log_prob = torch.logsumexp(log_student_t + log_weights, dim=1)
        nll = -log_prob
        # BUG #16 FIX: Clamp before mean
        nll = torch.clamp(nll, min=0.0, max=1e6)
        return nll.mean()
    
    def _dpmm_loss_mmd(self, z: torch.Tensor) -> torch.Tensor:
        """Maximum Mean Discrepancy"""
        if not self._check_dpmm_fitted("_dpmm_loss_mmd"):
            return torch.tensor(0.0, device=z.device)
        
        dp = self.dpmm_params
        n_samples = z.size(0)
        
        component_samples = torch.multinomial(dp["weights"], n_samples, replacement=True)
        means = dp["means"][component_samples]
        stds = torch.sqrt(1.0 / (dp["precisions_cholesky"][component_samples] ** 2 + 1e-10))
        dpmm_samples = means + stds * torch.randn_like(means)
        
        def rbf_kernel(x, y, bandwidth):
            xx = torch.sum(x ** 2, dim=1, keepdim=True)
            yy = torch.sum(y ** 2, dim=1, keepdim=True)
            xy = torch.mm(x, y.T)
            dists = xx + yy.T - 2 * xy
            return torch.exp(-dists / (2 * bandwidth ** 2))
        
        bandwidth = self.mmd_bandwidth
        k_xx = rbf_kernel(z, z, bandwidth).mean()
        k_yy = rbf_kernel(dpmm_samples, dpmm_samples, bandwidth).mean()
        k_xy = rbf_kernel(z, dpmm_samples, bandwidth).mean()
        
        mmd = k_xx - 2 * k_xy + k_yy
        return torch.relu(mmd)
    
    def _dpmm_loss_soft_nll(self, z: torch.Tensor) -> torch.Tensor:
        """Softened NLL (Brier score style)"""
        if not self._check_dpmm_fitted("_dpmm_loss_soft_nll"):
            return torch.tensor(0.0, device=z.device)
        
        dp = self.dpmm_params
        n_features = z.size(1)
        eps = 1e-10

        weights = dp["weights"]
        log_weights = torch.log(weights + eps)

        log_det = torch.sum(torch.log(dp["precisions_cholesky"] + eps), dim=1)
        precisions = dp["precisions_cholesky"] ** 2

        diff = z.unsqueeze(1) - dp["means"].unsqueeze(0)
        mahalanobis = torch.sum(diff ** 2 * precisions.unsqueeze(0), dim=2)

        log_2pi = math.log(2.0 * math.pi)
        log_gauss = -0.5 * (n_features * log_2pi + mahalanobis) + log_det
        log_prob = torch.logsumexp(log_gauss + log_weights, dim=1)
        
        temperature = 2.0
        prob = torch.exp(log_prob / temperature)
        loss = ((1.0 - prob) ** 2).mean()
        
        return loss


class DPMMManagementMixin:
    """DPMM parameter extraction and device management"""
    
    def _update_dpmm_params(self, bgm: BayesianGaussianMixture, device: torch.device):
        """Extract DPMM parameters from fitted sklearn model
        
        CRITICAL: Handles diagonal covariance case exactly like DPMMODE
        Ensures all tensors are on correct device.
        """
        means = torch.as_tensor(bgm.means_, dtype=torch.float32, device=device)
        weight_concentration = torch.as_tensor(bgm.weight_concentration_, dtype=torch.float32, device=device)
        weights = torch.as_tensor(bgm.weights_, dtype=torch.float32, device=device)
        
        precisions = torch.as_tensor(bgm.precisions_, dtype=torch.float32, device=device)
        precisions_cholesky = torch.sqrt(precisions)
        
        precisions_cholesky = torch.clamp(precisions_cholesky, min=1e-6, max=1e6)
        
        self.dpmm_params = {
            "means": means,
            "weight_concentration": weight_concentration,
            "precisions_cholesky": precisions_cholesky,
            "weights": weights,
            "covariances": 1.0 / (precisions_cholesky ** 2 + 1e-10),
        }
        self.dpmm_fitted = True
    
    def _move_dpmm_params_to_device(self, device: torch.device):
        """Move DPMM parameters to device (needed after model.to(device))"""
        if self.dpmm_params is None:
            return
        
        for key in self.dpmm_params:
            if torch.is_tensor(self.dpmm_params[key]):
                self.dpmm_params[key] = self.dpmm_params[key].to(device)