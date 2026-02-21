"""
Black-Scholes Analytical Solutions
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict

class BlackScholes:
    """Analytical Black-Scholes formulas for European options."""
    
    @staticmethod
    def d1(S0: float, K: float, T: float, r: float, sigma: float) -> float:
        return (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S0: float, K: float, T: float, r: float, sigma: float) -> float:
        return (np.log(S0 / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @classmethod
    def call_price(cls, S0: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return max(S0 - K, 0)
        d_1 = cls.d1(S0, K, T, r, sigma)
        d_2 = cls.d2(S0, K, T, r, sigma)
        return S0 * stats.norm.cdf(d_1) - K * np.exp(-r * T) * stats.norm.cdf(d_2)
    
    @classmethod
    def put_price(cls, S0: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return max(K - S0, 0)
        d_1 = cls.d1(S0, K, T, r, sigma)
        d_2 = cls.d2(S0, K, T, r, sigma)
        return K * np.exp(-r * T) * stats.norm.cdf(-d_2) - S0 * stats.norm.cdf(-d_1)
    
    @classmethod
    def call_delta(cls, S0: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return 1.0 if S0 > K else 0.0
        return stats.norm.cdf(cls.d1(S0, K, T, r, sigma))
    
    @classmethod
    def put_delta(cls, S0: float, K: float, T: float, r: float, sigma: float) -> float:
        return cls.call_delta(S0, K, T, r, sigma) - 1.0
    
    @classmethod
    def gamma(cls, S0: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return 0.0
        d_1 = cls.d1(S0, K, T, r, sigma)
        return stats.norm.pdf(d_1) / (S0 * sigma * np.sqrt(T))
    
    @classmethod
    def vega(cls, S0: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return 0.0
        d_1 = cls.d1(S0, K, T, r, sigma)
        return S0 * stats.norm.pdf(d_1) * np.sqrt(T)
    
    @classmethod
    def theta_call(cls, S0: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return 0.0
        d_1 = cls.d1(S0, K, T, r, sigma)
        d_2 = cls.d2(S0, K, T, r, sigma)
        theta = -(S0 * stats.norm.pdf(d_1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * stats.norm.cdf(d_2)
        return theta
    
    @classmethod
    def theta_put(cls, S0: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return 0.0
        d_1 = cls.d1(S0, K, T, r, sigma)
        d_2 = cls.d2(S0, K, T, r, sigma)
        theta = -(S0 * stats.norm.pdf(d_1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * stats.norm.cdf(-d_2)
        return theta
    
    @classmethod
    def rho_call(cls, S0: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return 0.0
        d_2 = cls.d2(S0, K, T, r, sigma)
        return K * T * np.exp(-r * T) * stats.norm.cdf(d_2)
    
    @classmethod
    def rho_put(cls, S0: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return 0.0
        d_2 = cls.d2(S0, K, T, r, sigma)
        return -K * T * np.exp(-r * T) * stats.norm.cdf(-d_2)
    
    @classmethod
    def implied_volatility(cls, price: float, S0: float, K: float, T: float, r: float, 
                          option_type: str = 'call', precision: float = 1e-6, max_iter: int = 100) -> float:
        from scipy.optimize import brentq
        
        def objective(sigma):
            if option_type == 'call':
                return cls.call_price(S0, K, T, r, sigma) - price
            else:
                return cls.put_price(S0, K, T, r, sigma) - price
        
        try:
            return brentq(objective, 0.0001, 5.0, xtol=precision)
        except:
            return np.nan
    
    @classmethod
    def get_all_greeks(cls, S0: float, K: float, T: float, r: float, sigma: float) -> Dict[str, float]:
        return {
            'call_price': cls.call_price(S0, K, T, r, sigma),
            'put_price': cls.put_price(S0, K, T, r, sigma),
            'call_delta': cls.call_delta(S0, K, T, r, sigma),
            'put_delta': cls.put_delta(S0, K, T, r, sigma),
            'gamma': cls.gamma(S0, K, T, r, sigma),
            'vega': cls.vega(S0, K, T, r, sigma),
            'call_theta': cls.theta_call(S0, K, T, r, sigma),
            'put_theta': cls.theta_put(S0, K, T, r, sigma),
            'call_rho': cls.rho_call(S0, K, T, r, sigma),
            'put_rho': cls.rho_put(S0, K, T, r, sigma)
        }