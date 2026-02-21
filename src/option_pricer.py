"""
Option Pricers using Monte Carlo Methods
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from scipy import stats

from simulator import GBMSimulator
from black_scholes import BlackScholes


class EuropeanPricer:
    """European Option Pricer with variance reduction techniques."""
    
    def __init__(self, simulator: Optional[GBMSimulator] = None, seed: Optional[int] = None):
        self.simulator = simulator or GBMSimulator(seed=seed)
        self.bs = BlackScholes()
    
    def price_call(self, S0: float, K: float, T: float, r: float, sigma: float, 
                   n_sims: int = 100000, antithetic: bool = True, control_variate: bool = True) -> Dict:
        if control_variate:
            return self._price_with_control_variate(S0, K, T, r, sigma, n_sims, 'call', antithetic)
        
        ST = self.simulator.generate_terminal_values(S0, r, sigma, T, n_sims, antithetic)
        payoffs = np.maximum(ST - K, 0) * np.exp(-r * T)
        price = np.mean(payoffs)
        std_err = np.std(payoffs, ddof=1) / np.sqrt(n_sims)
        
        return {
            'price': price,
            'std_error': std_err,
            'confidence_interval': (price - 1.96*std_err, price + 1.96*std_err),
            'n_sims': n_sims
        }
    
    def price_put(self, S0: float, K: float, T: float, r: float, sigma: float, 
                  n_sims: int = 100000, antithetic: bool = True, control_variate: bool = True) -> Dict:
        if control_variate:
            return self._price_with_control_variate(S0, K, T, r, sigma, n_sims, 'put', antithetic)
        
        ST = self.simulator.generate_terminal_values(S0, r, sigma, T, n_sims, antithetic)
        payoffs = np.maximum(K - ST, 0) * np.exp(-r * T)
        price = np.mean(payoffs)
        std_err = np.std(payoffs, ddof=1) / np.sqrt(n_sims)
        
        return {
            'price': price,
            'std_error': std_err,
            'confidence_interval': (price - 1.96*std_err, price + 1.96*std_err),
            'n_sims': n_sims
        }
    
    def _price_with_control_variate(self, S0: float, K: float, T: float, r: float, sigma: float, 
                                    n_sims: int, option_type: str, antithetic: bool) -> Dict:
        n_steps = 50
        paths = self.simulator.generate_paths(S0, r, sigma, T, n_steps, n_sims, antithetic)
        ST = paths[:, -1]
        
        if option_type == 'call':
            payoffs = np.maximum(ST - K, 0) * np.exp(-r * T)
        else:
            payoffs = np.maximum(K - ST, 0) * np.exp(-r * T)
        
        log_paths = np.log(paths[:, 1:])
        geom_mean = np.exp(np.mean(log_paths, axis=1))
        
        if option_type == 'call':
            control_payoffs = np.maximum(geom_mean - K, 0) * np.exp(-r * T)
            control_theory = self._geometric_asian_call(S0, K, T, r, sigma, n_steps)
        else:
            control_payoffs = np.maximum(K - geom_mean, 0) * np.exp(-r * T)
            control_theory = self._geometric_asian_put(S0, K, T, r, sigma, n_steps)
        
        cov = np.cov(payoffs, control_payoffs)
        c_star = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 1.0
        adjusted_payoffs = payoffs - c_star * (control_payoffs - control_theory)
        
        price = np.mean(adjusted_payoffs)
        std_err = np.std(adjusted_payoffs, ddof=1) / np.sqrt(n_sims)
        var_reduction = 1 - (np.var(adjusted_payoffs) / np.var(payoffs))
        
        return {
            'price': price,
            'std_error': std_err,
            'confidence_interval': (price - 1.96*std_err, price + 1.96*std_err),
            'variance_reduction': var_reduction,
            'control_coefficient': c_star,
            'n_sims': n_sims
        }
    
    def _geometric_asian_call(self, S0, K, T, r, sigma, n):
        dt = T / n
        nu = r - 0.5 * sigma**2
        a = np.log(S0) + nu * T + 0.5 * nu * dt * (n - 1)
        b = sigma**2 * dt + (sigma**2 * dt * (n-1) * (2*n - 1)) / (6 * n)
        b = max(b, 1e-10)
        d1 = (a - np.log(K) + b) / np.sqrt(b)
        d2 = d1 - np.sqrt(b)
        return np.exp(a + 0.5 * b) * stats.norm.cdf(d1) - K * stats.norm.cdf(d2)
    
    def _geometric_asian_put(self, S0, K, T, r, sigma, n):
        call = self._geometric_asian_call(S0, K, T, r, sigma, n)
        dt = T / n
        nu = r - 0.5 * sigma**2
        fwd = np.exp(np.log(S0) + nu * T + 0.5 * nu * dt * (n - 1) + 
                     0.5 * sigma**2 * dt * (n+1) * (2*n+1) / (6*n**2))
        return call - (fwd - K) * np.exp(-r * T)
    
    def verify_put_call_parity(self, S0, K, T, r, sigma, n_sims=100000):
        call_res = self.price_call(S0, K, T, r, sigma, n_sims, control_variate=False)
        put_res = self.price_put(S0, K, T, r, sigma, n_sims, control_variate=False)
        lhs = call_res['price'] - put_res['price']
        rhs = S0 - K * np.exp(-r * T)
        return {
            'call_price': call_res['price'],
            'put_price': put_res['price'],
            'difference': lhs,
            'theoretical': rhs,
            'error': abs(lhs - rhs),
            'parity_holds': abs(lhs - rhs) < 3 * (call_res['std_error'] + put_res['std_error'])
        }


class AmericanPricer:
    """American Option Pricer using Longstaff-Schwartz Least Squares Monte Carlo."""
    
    def __init__(self, simulator: Optional[GBMSimulator] = None, seed: Optional[int] = None):
        self.simulator = simulator or GBMSimulator(seed=seed)
        self.poly_degree = 3
    
    def price_put(self, S0: float, K: float, T: float, r: float, sigma: float, 
                  n_sims: int = 100000, n_steps: int = 50) -> Dict:
        dt = T / n_steps
        discount = np.exp(-r * dt)
        paths = self.simulator.generate_paths(S0, r, sigma, T, n_steps, n_sims)
        cashflows = np.maximum(K - paths[:, -1], 0)
        
        for t in range(n_steps - 1, 0, -1):
            S_t = paths[:, t]
            itm = S_t < K
            if np.sum(itm) > 10:
                X = S_t[itm]
                Y = cashflows[itm] * discount
                X_poly = np.column_stack([np.ones(len(X)), X, X**2, X**3])
                try:
                    beta = np.linalg.lstsq(X_poly, Y, rcond=None)[0]
                    continuation = X_poly @ beta
                except:
                    continuation = np.zeros(len(X))
                exercise_value = K - X
                exercise = exercise_value > continuation
                cashflows[itm] = np.where(exercise, exercise_value, Y)
                cashflows[~itm] *= discount
            else:
                cashflows *= discount
        
        price = np.mean(cashflows * discount)
        std_err = np.std(cashflows * discount, ddof=1) / np.sqrt(n_sims)
        
        return {
            'price': price,
            'std_error': std_err,
            'confidence_interval': (price - 1.96*std_err, price + 1.96*std_err),
            'n_sims': n_sims,
            'n_steps': n_steps
        }
    
    def price_call(self, S0: float, K: float, T: float, r: float, sigma: float, 
                   n_sims: int = 100000, n_steps: int = 50, dividend_yield: float = 0.0) -> Dict:
        if dividend_yield == 0:
            european = EuropeanPricer(self.simulator)
            result = european.price_call(S0, K, T, r, sigma, n_sims)
            result['early_exercise_premium'] = 0.0
            return result
        
        dt = T / n_steps
        discount = np.exp(-r * dt)
        paths = self.simulator.generate_paths(S0, r - dividend_yield, sigma, T, n_steps, n_sims)
        cashflows = np.maximum(paths[:, -1] - K, 0)
        
        for t in range(n_steps - 1, 0, -1):
            S_t = paths[:, t]
            itm = S_t > K
            if np.sum(itm) > 10:
                X = S_t[itm]
                Y = cashflows[itm] * discount
                X_poly = np.column_stack([np.ones(len(X)), X, X**2, X**3])
                try:
                    beta = np.linalg.lstsq(X_poly, Y, rcond=None)[0]
                    continuation = X_poly @ beta
                except:
                    continuation = np.zeros(len(X))
                exercise = (X - K) > continuation
                cashflows[itm] = np.where(exercise, X - K, Y)
                cashflows[~itm] *= discount
            else:
                cashflows *= discount
        
        price = np.mean(cashflows * discount)
        std_err = np.std(cashflows * discount, ddof=1) / np.sqrt(n_sims)
        
        return {
            'price': price,
            'std_error': std_err,
            'confidence_interval': (price - 1.96*std_err, price + 1.96*std_err),
            'n_sims': n_sims,
            'n_steps': n_steps
        }


class ExoticPricer:
    """Exotic Option Pricers: Asian, Barrier, Lookback, Digital"""
    
    def __init__(self, simulator: Optional[GBMSimulator] = None, seed: Optional[int] = None):
        self.simulator = simulator or GBMSimulator(seed=seed)
    
    def asian_call(self, S0: float, K: float, T: float, r: float, sigma: float, 
                   n_sims: int = 100000, n_steps: int = 100, averaging_type: str = 'arithmetic') -> Dict:
        paths = self.simulator.generate_paths(S0, r, sigma, T, n_steps, n_sims)
        if averaging_type == 'arithmetic':
            avg_prices = np.mean(paths[:, 1:], axis=1)
        else:
            avg_prices = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
        payoffs = np.maximum(avg_prices - K, 0) * np.exp(-r * T)
        price = np.mean(payoffs)
        std_err = np.std(payoffs, ddof=1) / np.sqrt(n_sims)
        return {
            'price': price,
            'std_error': std_err,
            'confidence_interval': (price - 1.96*std_err, price + 1.96*std_err),
            'averaging_type': averaging_type,
            'n_sims': n_sims
        }
    
    def asian_put(self, S0: float, K: float, T: float, r: float, sigma: float, 
                  n_sims: int = 100000, n_steps: int = 100, averaging_type: str = 'arithmetic') -> Dict:
        paths = self.simulator.generate_paths(S0, r, sigma, T, n_steps, n_sims)
        if averaging_type == 'arithmetic':
            avg_prices = np.mean(paths[:, 1:], axis=1)
        else:
            avg_prices = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
        payoffs = np.maximum(K - avg_prices, 0) * np.exp(-r * T)
        price = np.mean(payoffs)
        std_err = np.std(payoffs, ddof=1) / np.sqrt(n_sims)
        return {
            'price': price,
            'std_error': std_err,
            'confidence_interval': (price - 1.96*std_err, price + 1.96*std_err),
            'averaging_type': averaging_type,
            'n_sims': n_sims
        }
    
    def barrier_call_up_out(self, S0: float, K: float, B: float, T: float, r: float, sigma: float, 
                            n_sims: int = 100000, n_steps: int = 100, monitoring: str = 'discrete') -> Dict:
        paths = self.simulator.generate_paths(S0, r, sigma, T, n_steps, n_sims)
        barrier_hit = np.any(paths >= B, axis=1)
        ST = paths[:, -1]
        payoffs = np.where(~barrier_hit, np.maximum(ST - K, 0), 0) * np.exp(-r * T)
        price = np.mean(payoffs)
        std_err = np.std(payoffs, ddof=1) / np.sqrt(n_sims)
        knock_out_prob = np.mean(barrier_hit)
        return {
            'price': price,
            'std_error': std_err,
            'confidence_interval': (price - 1.96*std_err, price + 1.96*std_err),
            'knock_out_probability': knock_out_prob,
            'n_sims': n_sims
        }
    
    def lookback_call_fixed_strike(self, S0: float, K: float, T: float, r: float, sigma: float, 
                                   n_sims: int = 100000, n_steps: int = 100) -> Dict:
        paths = self.simulator.generate_paths(S0, r, sigma, T, n_steps, n_sims)
        S_max = np.max(paths, axis=1)
        payoffs = np.maximum(S_max - K, 0) * np.exp(-r * T)
        price = np.mean(payoffs)
        std_err = np.std(payoffs, ddof=1) / np.sqrt(n_sims)
        return {
            'price': price,
            'std_error': std_err,
            'confidence_interval': (price - 1.96*std_err, price + 1.96*std_err),
            'n_sims': n_sims
        }
    
    def lookback_put_fixed_strike(self, S0: float, K: float, T: float, r: float, sigma: float, 
                                  n_sims: int = 100000, n_steps: int = 100) -> Dict:
        paths = self.simulator.generate_paths(S0, r, sigma, T, n_steps, n_sims)
        S_min = np.min(paths, axis=1)
        payoffs = np.maximum(K - S_min, 0) * np.exp(-r * T)
        price = np.mean(payoffs)
        std_err = np.std(payoffs, ddof=1) / np.sqrt(n_sims)
        return {
            'price': price,
            'std_error': std_err,
            'confidence_interval': (price - 1.96*std_err, price + 1.96*std_err),
            'n_sims': n_sims
        }
    
    def digital_call(self, S0: float, K: float, T: float, r: float, sigma: float, 
                     payout: float = 1.0, n_sims: int = 100000) -> Dict:
        ST = self.simulator.generate_terminal_values(S0, r, sigma, T, n_sims)
        payoffs = (ST > K) * payout * np.exp(-r * T)
        price = np.mean(payoffs)
        std_err = np.std(payoffs, ddof=1) / np.sqrt(n_sims)
        from .black_scholes import BlackScholes
        bs = BlackScholes()
        d2 = bs.d2(S0, K, T, r, sigma)
        theoretical = payout * np.exp(-r * T) * stats.norm.cdf(d2)
        return {
            'price': price,
            'theoretical': theoretical,
            'std_error': std_err,
            'confidence_interval': (price - 1.96*std_err, price + 1.96*std_err),
            'n_sims': n_sims
        }
    
    def basket_call(self, S0: np.ndarray, weights: np.ndarray, K: float, T: float, r: float, 
                    sigma: np.ndarray, corr: np.ndarray, n_sims: int = 100000, n_steps: int = 50) -> Dict:
        from .simulator import MultiAssetSimulator
        simulator = MultiAssetSimulator()
        paths = simulator.generate_paths(S0, r, sigma, corr, T, n_steps, n_sims)
        ST = paths[:, :, -1]
        basket_values = np.sum(ST * weights, axis=1)
        payoffs = np.maximum(basket_values - K, 0) * np.exp(-r * T)
        price = np.mean(payoffs)
        std_err = np.std(payoffs, ddof=1) / np.sqrt(n_sims)
        return {
            'price': price,
            'std_error': std_err,
            'confidence_interval': (price - 1.96*std_err, price + 1.96*std_err),
            'n_sims': n_sims
        }