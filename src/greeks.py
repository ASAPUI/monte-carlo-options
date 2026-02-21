"""
Greeks Calculation using Finite Differences and Pathwise Methods
"""

import numpy as np
from typing import Dict, Callable, Optional
from simulator import GBMSimulator
from option_pricer import EuropeanPricer, AmericanPricer, ExoticPricer


class GreeksCalculator:
    """Calculate option Greeks using various methods."""
    
    def __init__(self, pricer=None, method: str = 'finite_difference'):
        self.pricer = pricer or EuropeanPricer()
        self.method = method
        self.bump_size = 0.01
    
    def calculate_european_greeks(self, S0: float, K: float, T: float, r: float, sigma: float,
                                  option_type: str = 'call', n_sims: int = 100000) -> Dict:
        if self.method == 'finite_difference':
            return self._finite_difference_greeks(S0, K, T, r, sigma, option_type, n_sims)
        elif self.method == 'pathwise':
            return self._pathwise_greeks(S0, K, T, r, sigma, option_type, n_sims)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _finite_difference_greeks(self, S0: float, K: float, T: float, r: float, sigma: float,
                                  option_type: str, n_sims: int) -> Dict:
        h_S = S0 * self.bump_size
        h_sigma = sigma * self.bump_size
        h_T = T * 0.01
        h_r = 0.0001
        
        def price(S, sig, t, rate):
            if option_type == 'call':
                return self.pricer.price_call(S, K, t, rate, sig, n_sims)['price']
            else:
                return self.pricer.price_put(S, K, t, rate, sig, n_sims)['price']
        
        base_price = price(S0, sigma, T, r)
        price_up = price(S0 + h_S, sigma, T, r)
        price_down = price(S0 - h_S, sigma, T, r)
        delta = (price_up - price_down) / (2 * h_S)
        gamma = (price_up - 2 * base_price + price_down) / (h_S ** 2)
        
        price_vol_up = price(S0, sigma + h_sigma, T, r)
        price_vol_down = price(S0, sigma - h_sigma, T, r)
        vega = (price_vol_up - price_vol_down) / (2 * h_sigma)
        
        if T > h_T:
            price_time_up = price(S0, sigma, T + h_T, r)
            price_time_down = price(S0, sigma, T - h_T, r)
            theta = -(price_time_up - price_time_down) / (2 * h_T)
        else:
            theta = 0.0
        
        price_rate_up = price(S0, sigma, T, r + h_r)
        price_rate_down = price(S0, sigma, T, r - h_r)
        rho = (price_rate_up - price_rate_down) / (2 * h_r)
        
        return {
            'price': base_price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho,
            'method': 'finite_difference',
            'bump_size': self.bump_size
        }
    
    def _pathwise_greeks(self, S0: float, K: float, T: float, r: float, sigma: float,
                         option_type: str, n_sims: int) -> Dict:
        simulator = GBMSimulator()
        ST = simulator.generate_terminal_values(S0, r, sigma, T, n_sims)
        df = np.exp(-r * T)
        
        if option_type == 'call':
            indicator = (ST > K).astype(float)
            payoff = np.maximum(ST - K, 0)
        else:
            indicator = (ST < K).astype(float)
            payoff = np.maximum(K - ST, 0)
        
        price = np.mean(payoff * df)
        delta = np.mean(indicator * (ST / S0) * df)
        log_return = np.log(ST / S0)
        vega_term = (log_return - (r + 0.5 * sigma**2) * T) / sigma
        vega = np.mean(indicator * ST * vega_term * df)
        
        h_S = S0 * self.bump_size
        delta_up = self._pathwise_delta(S0 + h_S, K, T, r, sigma, option_type, n_sims//2)
        delta_down = self._pathwise_delta(S0 - h_S, K, T, r, sigma, option_type, n_sims//2)
        gamma = (delta_up - delta_down) / (2 * h_S)
        
        greeks_fd = self._finite_difference_greeks(S0, K, T, r, sigma, option_type, n_sims//2)
        
        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': greeks_fd['theta'],
            'rho': greeks_fd['rho'],
            'method': 'pathwise'
        }
    
    def _pathwise_delta(self, S0, K, T, r, sigma, option_type, n_sims):
        simulator = GBMSimulator()
        ST = simulator.generate_terminal_values(S0, r, sigma, T, n_sims)
        df = np.exp(-r * T)
        if option_type == 'call':
            indicator = (ST > K).astype(float)
        else:
            indicator = (ST < K).astype(float)
        return np.mean(indicator * (ST / S0) * df)
    
    def calculate_american_greeks(self, S0: float, K: float, T: float, r: float, sigma: float,
                                  option_type: str = 'put', n_sims: int = 50000, n_steps: int = 50) -> Dict:
        american_pricer = AmericanPricer()
        h_S = S0 * self.bump_size
        h_sigma = sigma * self.bump_size
        h_T = max(T * 0.01, 0.001)
        h_r = 0.0001
        
        def price(S, sig, t, rate):
            if option_type == 'put':
                return american_pricer.price_put(S, K, t, rate, sig, n_sims, n_steps)['price']
            else:
                return american_pricer.price_call(S, K, t, rate, sig, n_sims, n_steps)['price']
        
        base_price = price(S0, sigma, T, r)
        price_up = price(S0 + h_S, sigma, T, r)
        price_down = price(S0 - h_S, sigma, T, r)
        delta = (price_up - price_down) / (2 * h_S)
        gamma = (price_up - 2 * base_price + price_down) / (h_S ** 2)
        
        price_vol_up = price(S0, sigma + h_sigma, T, r)
        price_vol_down = price(S0, sigma - h_sigma, T, r)
        vega = (price_vol_up - price_vol_down) / (2 * h_sigma)
        
        if T > h_T:
            price_time_up = price(S0, sigma, T + h_T, r)
            price_time_down = price(S0, sigma, T - h_T, r)
            theta = -(price_time_up - price_time_down) / (2 * h_T)
        else:
            theta = 0.0
        
        price_rate_up = price(S0, sigma, T, r + h_r)
        price_rate_down = price(S0, sigma, T, r - h_r)
        rho = (price_rate_up - price_rate_down) / (2 * h_r)
        
        return {
            'price': base_price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho,
            'option_type': 'american',
            'method': 'finite_difference'
        }
    
    def greek_surface(self, greek: str, S_range: np.ndarray, sigma_range: np.ndarray,
                      K: float, T: float, r: float, option_type: str = 'call', n_sims: int = 50000) -> np.ndarray:
        surface = np.zeros((len(S_range), len(sigma_range)))
        for i, S in enumerate(S_range):
            for j, sig in enumerate(sigma_range):
                greeks = self.calculate_european_greeks(S, K, T, r, sig, option_type, n_sims)
                surface[i, j] = greeks.get(greek, 0)
        return surface