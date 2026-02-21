"""
Visualization Tools for Monte Carlo Option Pricing
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from typing import Optional, List, Dict
import warnings

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class OptionVisualizer:
    """Comprehensive visualization toolkit for option pricing analysis."""
    
class OptionVisualizer:
    def __init__(self, figsize=(12, 8), seed: int = None):
        self.figsize = figsize
        # Create your OWN random number generator with fixed seed
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random  # Fall back to global (old behavior)
       
    def plot_paths(self, paths: np.ndarray, T: float, n_plot: int = 100,
                   title: str = "Simulated Asset Paths", figsize: Optional[tuple] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize or self.figsize)
        n_sims, n_steps = paths.shape
        n_steps -= 1
        time_grid = np.linspace(0, T, n_steps + 1)
        indices = self.rng.choice(n_sims, min(n_plot, n_sims), replace=False)
        
        for idx in indices:
            ax.plot(time_grid, paths[idx], alpha=0.6, linewidth=0.8)
        
        mean_path = np.mean(paths, axis=0)
        ax.plot(time_grid, mean_path, 'k--', linewidth=2, label='Mean Path')
        
        p05 = np.percentile(paths, 5, axis=0)
        p95 = np.percentile(paths, 95, axis=0)
        ax.fill_between(time_grid, p05, p95, alpha=0.2, color='blue', label='90% CI')
        
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Asset Price', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig
    
    def plot_terminal_distribution(self, ST: np.ndarray, S0: float, K: Optional[float] = None,
                                   r: Optional[float] = None, T: Optional[float] = None,
                                   sigma: Optional[float] = None, option_type: str = 'call') -> plt.Figure:
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.hist(ST, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Simulated')
        
        if all(v is not None for v in [S0, r, T, sigma]):
            mu = np.log(S0) + (r - 0.5 * sigma**2) * T
            s = sigma * np.sqrt(T)
            x = np.linspace(ST.min(), ST.max(), 100)
            from scipy.stats import lognorm
            pdf = lognorm.pdf(x, s, scale=np.exp(mu))
            ax.plot(x, pdf, 'r-', linewidth=2, label='Theoretical Log-Normal')
        
        if K is not None:
            ax.axvline(K, color='green', linestyle='--', linewidth=2, label=f'Strike K={K}')
            if option_type == 'call':
                ax.axvspan(K, ST.max(), alpha=0.2, color='green', label='ITM Region')
            else:
                ax.axvspan(ST.min(), K, alpha=0.2, color='green', label='ITM Region')
        
        ax.set_xlabel('Terminal Asset Price', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title('Distribution of Terminal Prices', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig
    
    def plot_convergence(self, prices: List[float], std_errors: List[float], n_sims_list: List[int],
                         theoretical_price: Optional[float] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=self.figsize)
        prices = np.array(prices)
        std_errors = np.array(std_errors)
        
        ax.plot(n_sims_list, prices, 'b-o', linewidth=2, markersize=6, label='MC Estimate')
        ax.fill_between(n_sims_list, prices - 1.96 * std_errors, prices + 1.96 * std_errors,
                        alpha=0.3, color='blue', label='95% CI')
        
        if theoretical_price is not None:
            ax.axhline(theoretical_price, color='red', linestyle='--', linewidth=2,
                      label=f'Theoretical: {theoretical_price:.4f}')
        
        ax.set_xscale('log')
        ax.set_xlabel('Number of Simulations', fontsize=12)
        ax.set_ylabel('Option Price', fontsize=12)
        ax.set_title('Monte Carlo Convergence', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig
    
    def plot_greeks_surface(self, S_range: np.ndarray, sigma_range: np.ndarray, greek_values: np.ndarray,
                            greek_name: str, K: float, T: float) -> plt.Figure:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        S_grid, sigma_grid = np.meshgrid(S_range, sigma_range)
        surf = ax.plot_surface(S_grid, sigma_grid, greek_values.T, cmap='viridis',
                               alpha=0.8, edgecolor='none')
        ax.set_xlabel('Spot Price', fontsize=11)
        ax.set_ylabel('Volatility', fontsize=11)
        ax.set_zlabel(greek_name.capitalize(), fontsize=11)
        ax.set_title(f'{greek_name.capitalize()} Surface (K={K}, T={T})', fontsize=14, fontweight='bold')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        return fig
    
    def plot_greeks_2d(self, S_range: np.ndarray, greeks_dict: Dict[str, np.ndarray],
                       K: float, title: str = "Greeks Analysis") -> plt.Figure:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        greek_names = ['delta', 'gamma', 'vega', 'theta', 'rho']
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        for idx, (greek, color) in enumerate(zip(greek_names, colors)):
            if greek in greeks_dict:
                ax = axes[idx]
                values = greeks_dict[greek]
                ax.plot(S_range, values, color=color, linewidth=2)
                ax.axvline(K, color='black', linestyle='--', alpha=0.5, label=f'Strike K={K}')
                ax.set_xlabel('Spot Price', fontsize=10)
                ax.set_ylabel(greek.capitalize(), fontsize=10)
                ax.set_title(f'{greek.capitalize()} vs Spot', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        fig.delaxes(axes[5])
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_comparison_bar(self, labels: List[str], mc_prices: List[float],
                            bs_prices: Optional[List[float]] = None, title: str = "Option Price Comparison") -> plt.Figure:
        fig, ax = plt.subplots(figsize=self.figsize)
        x = np.arange(len(labels))
        width = 0.35
        bars1 = ax.bar(x - width/2, mc_prices, width, label='Monte Carlo', color='skyblue')
        
        if bs_prices is not None:
            bars2 = ax.bar(x + width/2, bs_prices, width, label='Black-Scholes', color='lightcoral')
        
        ax.set_xlabel('Option Type', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        for bars in [bars1]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        return fig
    
    def plot_variance_reduction(self, methods: List[str], variances: List[float],
                                baseline_var: float) -> plt.Figure:
        fig, ax = plt.subplots(figsize=self.figsize)
        reductions = [(1 - v / baseline_var) * 100 for v in variances]
        colors = ['green' if r > 0 else 'red' for r in reductions]
        bars = ax.bar(methods, reductions, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Variance Reduction (%)', fontsize=12)
        ax.set_title('Variance Reduction Techniques Comparison', fontsize=14, fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height > 0 else -15), textcoords="offset points",
                       ha='center', va='bottom' if height > 0 else 'top', fontsize=10)
        return fig


def save_figure(fig: plt.Figure, filename: str, dpi: int = 150):
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)