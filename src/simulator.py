"""
Geometric Brownian Motion Simulator and Advanced Stochastic Models
Optimized with bug fixes and Numba acceleration
"""

import numpy as np
from numba import jit, prange
from typing import Tuple, Optional

# ============================================================================
# NUMBA JIT-COMPILED CORE FUNCTIONS (10-100x faster)
# ============================================================================

@jit(nopython=True, parallel=True, cache=True)
def _gbm_paths_numba(S0: float, r: float, sigma: float, T: float, 
                     n_steps: int, n_sims: int, seed: int) -> np.ndarray:
    """Fast GBM path generation using Numba."""
    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    paths = np.zeros((n_sims, n_steps + 1))
    paths[:, 0] = S0
    
    np.random.seed(seed)
    
    for i in prange(n_sims):
        Z = np.random.standard_normal(n_steps)
        for t in range(n_steps):
            paths[i, t+1] = paths[i, t] * np.exp(drift + diffusion * Z[t])
    
    return paths

@jit(nopython=True, parallel=True, cache=True)
def _gbm_paths_antithetic_numba(S0: float, r: float, sigma: float, T: float, 
                                 n_steps: int, n_sims: int, seed: int) -> np.ndarray:
    """Fast GBM with antithetic variates."""
    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    n_half = n_sims // 2
    
    paths = np.zeros((n_sims, n_steps + 1))
    paths[:, 0] = S0
    
    np.random.seed(seed)
    
    for i in prange(n_half):
        Z = np.random.standard_normal(n_steps)
        for t in range(n_steps):
            paths[i, t+1] = paths[i, t] * np.exp(drift + diffusion * Z[t])
            paths[i + n_half, t+1] = paths[i + n_half, t] * np.exp(drift - diffusion * Z[t])
    
    return paths

@jit(nopython=True, cache=True)
def _terminal_values_numba(S0: float, r: float, sigma: float, T: float, 
                           n_sims: int, seed: int) -> np.ndarray:
    """Fast terminal value generation."""
    np.random.seed(seed)
    Z = np.random.standard_normal(n_sims)
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T)
    return S0 * np.exp(drift + diffusion * Z)

# ============================================================================
# SIMULATOR CLASSES
# ============================================================================

class GBMSimulator:
    """Geometric Brownian Motion Simulator for option pricing."""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.seed = seed
    
    def generate_paths(self, S0: float, r: float, sigma: float, T: float, 
                       n_steps: int, n_sims: int, antithetic: bool = False,
                       use_numba: bool = True) -> np.ndarray:
        """
        Generate GBM price paths.
        
        Args:
            use_numba: If True and seed is set, uses JIT-compiled version (much faster)
        """
        # Use Numba version for speed if possible
        if use_numba and self.seed is not None:
            if antithetic:
                return _gbm_paths_antithetic_numba(S0, r, sigma, T, n_steps, n_sims, self.seed)
            else:
                return _gbm_paths_numba(S0, r, sigma, T, n_steps, n_sims, self.seed)
        
        # Fallback to Python version
        dt = T / n_steps
        paths = np.zeros((n_sims, n_steps + 1))
        paths[:, 0] = S0
        
        if antithetic:
            n_half = n_sims // 2
            for t in range(1, n_steps + 1):
                Z = self.rng.standard_normal(n_half)
                dW_pos = np.sqrt(dt) * Z
                dW_neg = -dW_pos
                paths[:n_half, t] = paths[:n_half, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * dW_pos)
                paths[n_half:2*n_half, t] = paths[n_half:2*n_half, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * dW_neg)
        else:
            for t in range(1, n_steps + 1):
                Z = self.rng.standard_normal(n_sims)
                paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        
        return paths
    
    def generate_terminal_values(self, S0: float, r: float, sigma: float, T: float, 
                                  n_sims: int, antithetic: bool = False,
                                  use_numba: bool = True) -> np.ndarray:
        """Generate terminal stock prices."""
        # Use Numba for speed
        if use_numba and self.seed is not None and not antithetic:
            return _terminal_values_numba(S0, r, sigma, T, n_sims, self.seed)
        
        # Python version with antithetic support
        drift = (r - 0.5 * sigma**2) * T
        diffusion = sigma * np.sqrt(T)
        
        if antithetic:
            n_half = n_sims // 2
            Z = self.rng.standard_normal(n_half)
            ST_pos = S0 * np.exp(drift + diffusion * Z)
            ST_neg = S0 * np.exp(drift - diffusion * Z)
            result = np.concatenate([ST_pos, ST_neg])
            
            # Handle odd n_sims
            if len(result) < n_sims:
                extra_Z = self.rng.standard_normal(1)
                extra_ST = S0 * np.exp(drift + diffusion * extra_Z)
                result = np.concatenate([result, extra_ST])
            
            return result[:n_sims]
        else:
            Z = self.rng.standard_normal(n_sims)
            return S0 * np.exp(drift + diffusion * Z)


class HestonSimulator:
    """Heston Stochastic Volatility Model with full truncation scheme."""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.seed = seed
    
    def generate_paths(self, S0: float, v0: float, r: float, kappa: float, theta: float, 
                       xi: float, rho: float, T: float, n_steps: int, n_sims: int, 
                       scheme: str = 'full_truncation') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Heston model paths.
        
        scheme: 'full_truncation' (recommended), 'reflection', or 'absorption'
        """
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1))
        v = np.zeros((n_sims, n_steps + 1))
        S[:, 0] = S0
        v[:, 0] = v0
        
        for t in range(n_steps):
            # Correlated Brownian motions
            Z1 = self.rng.standard_normal((n_sims, 2))
            Z_S = Z1[:, 0]
            Z_v = rho * Z1[:, 0] + np.sqrt(1 - rho**2) * Z1[:, 1]
            
            # Full truncation: use max(v, 0) for calculations
            v_pos = np.maximum(v[:, t], 0)
            
            # Variance process (Euler-Maruyama with truncation)
            v[:, t+1] = v[:, t] + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos * dt) * Z_v
            
            # Milstein correction for variance
            if scheme == 'milstein':
                v[:, t+1] += 0.25 * xi**2 * dt * (Z_v**2 - 1)
            
            # Keep variance non-negative
            if scheme == 'reflection':
                v[:, t+1] = np.abs(v[:, t+1])
            else:  # full_truncation or absorption
                v[:, t+1] = np.maximum(v[:, t+1], 0)
            
            # Stock price process (use truncated variance)
            S[:, t+1] = S[:, t] * np.exp((r - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * Z_S)
        
        return S, v


class MertonJumpDiffusion:
    """Merton Jump-Diffusion Model (optimized)."""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.seed = seed
    
    def generate_paths(self, S0: float, r: float, sigma: float, T: float, n_steps: int, 
                       n_sims: int, lam: float, mu_j: float, delta: float,
                       max_jumps_per_step: int = 10) -> np.ndarray:
        """
        Generate Merton jump-diffusion paths.
        
        Optimized using vectorized operations instead of Python loops.
        """
        dt = T / n_steps
        paths = np.zeros((n_sims, n_steps + 1))
        paths[:, 0] = S0
        
        # Risk-neutral drift adjustment
        jump_drift = lam * (np.exp(mu_j + 0.5 * delta**2) - 1)
        r_adj = r - jump_drift
        
        for t in range(1, n_steps + 1):
            # Diffusion component
            Z = self.rng.standard_normal(n_sims)
            diffusion = np.exp((r_adj - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
            
            # Jump component (vectorized)
            N = self.rng.poisson(lam * dt, n_sims)
            jumps = np.ones(n_sims)
            
            # Handle paths with jumps
            jump_mask = N > 0
            if np.any(jump_mask):
                max_n = min(np.max(N[jump_mask]), max_jumps_per_step)
                
                # Generate jump sizes for all possible jumps
                for j in range(1, max_n + 1):
                    # Paths that have at least j jumps
                    j_mask = N >= j
                    if np.any(j_mask):
                        J = self.rng.lognormal(mu_j, delta, np.sum(j_mask))
                        jumps[j_mask] *= J
            
            paths[:, t] = paths[:, t-1] * diffusion * jumps
        
        return paths
    
    def generate_paths_simple(self, S0: float, r: float, sigma: float, T: float, n_steps: int, 
                              n_sims: int, lam: float, mu_j: float, delta: float) -> np.ndarray:
        """
        Simpler but slower version - for testing/verification only.
        """
        dt = T / n_steps
        paths = np.zeros((n_sims, n_steps + 1))
        paths[:, 0] = S0
        r_adj = r - lam * (np.exp(mu_j + 0.5 * delta**2) - 1)
        
        for t in range(1, n_steps + 1):
            Z = self.rng.standard_normal(n_sims)
            paths[:, t] = paths[:, t-1] * np.exp((r_adj - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
            N = self.rng.poisson(lam * dt, n_sims)
            
            for i in range(n_sims):
                if N[i] > 0:
                    for _ in range(N[i]):
                        J = np.exp(self.rng.normal(mu_j, delta))
                        paths[i, t] *= J
        
        return paths


class MultiAssetSimulator:
    """Multi-asset simulator with correlated Brownian motions."""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.seed = seed
    
    def generate_paths(self, S0: np.ndarray, r: float, sigma: np.ndarray, corr: np.ndarray, 
                       T: float, n_steps: int, n_sims: int) -> np.ndarray:
        """
        Generate correlated asset paths.
        
        Args:
            S0: Initial prices (n_assets,)
            sigma: Volatilities (n_assets,)
            corr: Correlation matrix (n_assets, n_assets)
        """
        n_assets = len(S0)
        dt = T / n_steps
        
        # Cholesky decomposition for correlation
        L = np.linalg.cholesky(corr)
        
        paths = np.zeros((n_sims, n_assets, n_steps + 1))
        paths[:, :, 0] = S0
        
        for t in range(1, n_steps + 1):
            # Generate independent normals and correlate them
            Z = self.rng.standard_normal((n_sims, n_assets))
            dW = Z @ L.T * np.sqrt(dt)  # Correlated Brownian increments
            
            for i in range(n_assets):
                paths[:, i, t] = paths[:, i, t-1] * np.exp(
                    (r - 0.5 * sigma[i]**2) * dt + sigma[i] * dW[:, i]
                )
        
        return paths