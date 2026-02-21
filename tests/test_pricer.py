"""
Comprehensive Test Suite for Monte Carlo Options Pricing
"""

import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.black_scholes import BlackScholes
from src.simulator import GBMSimulator, HestonSimulator, MertonJumpDiffusion, MultiAssetSimulator
from src.option_pricer import EuropeanPricer, AmericanPricer, ExoticPricer
from src.greeks import GreeksCalculator


class TestBlackScholes:
    """Test analytical Black-Scholes formulas."""
    
    def test_call_price(self):
        price = BlackScholes.call_price(S0=100, K=100, T=1.0, r=0.05, sigma=0.2)
        assert 10.0 < price < 11.0
        
    def test_put_price(self):
        price = BlackScholes.put_price(S0=100, K=100, T=1.0, r=0.05, sigma=0.2)
        assert 5.0 < price < 6.0
        
    def test_put_call_parity(self):
        S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        call = BlackScholes.call_price(S0, K, T, r, sigma)
        put = BlackScholes.put_price(S0, K, T, r, sigma)
        lhs = call - put
        rhs = S0 - K * np.exp(-r * T)
        assert abs(lhs - rhs) < 1e-10
        
    def test_greeks(self):
        S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        delta = BlackScholes.call_delta(S0, K, T, r, sigma)
        assert 0.5 < delta < 0.7
        gamma = BlackScholes.gamma(S0, K, T, r, sigma)
        assert gamma > 0
        vega = BlackScholes.vega(S0, K, T, r, sigma)
        assert vega > 0
        
    def test_implied_volatility(self):
        S0, K, T, r, sigma_true = 100, 100, 1.0, 0.05, 0.2
        price = BlackScholes.call_price(S0, K, T, r, sigma_true)
        sigma_iv = BlackScholes.implied_volatility(price, S0, K, T, r, 'call')
        assert abs(sigma_iv - sigma_true) < 1e-4


class TestGBMSimulator:
    """Test Geometric Brownian Motion simulator."""
    
    def test_terminal_distribution(self):
        sim = GBMSimulator(seed=42)
        S0, r, sigma, T = 100, 0.05, 0.2, 1.0
        n_sims = 100000
        ST = sim.generate_terminal_values(S0, r, sigma, T, n_sims)
        expected_mean = S0 * np.exp(r * T)
        actual_mean = np.mean(ST)
        assert abs(actual_mean - expected_mean) / expected_mean < 0.05
        
    def test_path_properties(self):
        sim = GBMSimulator(seed=42)
        paths = sim.generate_paths(100, 0.05, 0.2, 1.0, 252, 1000)
        assert np.all(paths > 0)
        assert np.allclose(paths[:, 0], 100)
        assert paths.shape == (1000, 253)


class TestEuropeanPricer:
    """Test European option pricing."""
    
    def test_call_convergence(self):
        pricer = EuropeanPricer(seed=42)
        S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        bs_price = BlackScholes.call_price(S0, K, T, r, sigma)
        mc_result = pricer.price_call(S0, K, T, r, sigma, n_sims=200000)
        assert abs(mc_result['price'] - bs_price) < 2 * mc_result['std_error']
        
    def test_put_convergence(self):
        pricer = EuropeanPricer(seed=42)
        S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        bs_price = BlackScholes.put_price(S0, K, T, r, sigma)
        mc_result = pricer.price_put(S0, K, T, r, sigma, n_sims=200000)
        assert abs(mc_result['price'] - bs_price) < 2 * mc_result['std_error']
        
    def test_put_call_parity_mc(self):
        pricer = EuropeanPricer(seed=42)
        S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        result = pricer.verify_put_call_parity(S0, K, T, r, sigma, n_sims=100000)
        assert result['parity_holds']


class TestAmericanPricer:
    """Test American option pricing with LSM."""
    
    def test_american_put_premium(self):
        american_pricer = AmericanPricer(seed=42)
        european_pricer = EuropeanPricer(seed=42)
        S0_itm, K_itm = 100, 110
        am_put = american_pricer.price_put(S0_itm, K_itm, 1.0, 0.05, 0.2, n_sims=50000, n_steps=50)
        eu_put = european_pricer.price_put(S0_itm, K_itm, 1.0, 0.05, 0.2, n_sims=50000)
        assert am_put['price'] >= eu_put['price'] * 0.95
        
    def test_american_call_no_dividend(self):
        american_pricer = AmericanPricer(seed=42)
        european_pricer = EuropeanPricer(seed=42)
        S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        am_price = american_pricer.price_call(S0, K, T, r, sigma, n_sims=30000, n_steps=50)['price']
        eu_price = european_pricer.price_call(S0, K, T, r, sigma, n_sims=30000)['price']
        assert abs(am_price - eu_price) < 0.5


class TestExoticPricer:
    """Test exotic option pricing."""
    
    def test_asian_price_range(self):
        pricer = ExoticPricer(seed=42)
        S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        asian = pricer.asian_call(S0, K, T, r, sigma, n_sims=50000)
        european = BlackScholes.call_price(S0, K, T, r, sigma)
        assert asian['price'] < european
        assert asian['price'] > 0
        
    def test_barrier_knockout(self):
        pricer = ExoticPricer(seed=42)
        S0, K, B, T, r, sigma = 100, 100, 105, 1.0, 0.05, 0.2
        result = pricer.barrier_call_up_out(S0, K, B, T, r, sigma, n_sims=50000)
        vanilla = BlackScholes.call_price(S0, K, T, r, sigma)
        assert result['price'] < vanilla
        assert result['knock_out_probability'] > 0
        
    def test_digital_option(self):
        pricer = ExoticPricer(seed=42)
        S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        result = pricer.digital_call(S0, K, T, r, sigma, payout=1.0, n_sims=100000)
        assert abs(result['price'] - result['theoretical']) < 0.02


class TestGreeksCalculator:
    """Test Greeks calculation."""
    
    def test_delta_accuracy(self):
        calc = GreeksCalculator(method='finite_difference')
        S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        greeks = calc.calculate_european_greeks(S0, K, T, r, sigma, 'call', n_sims=100000)
        true_delta = BlackScholes.call_delta(S0, K, T, r, sigma)
        assert abs(greeks['delta'] - true_delta) < 0.02
        
    def test_vega_accuracy(self):
        calc = GreeksCalculator(method='finite_difference')
        S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        greeks = calc.calculate_european_greeks(S0, K, T, r, sigma, 'call', n_sims=100000)
        true_vega = BlackScholes.vega(S0, K, T, r, sigma)
        assert abs(greeks['vega'] - true_vega) < 0.5


class TestMultiAssetSimulator:
    """Test multi-asset simulation."""
    
    def test_correlation_structure(self):
        sim = MultiAssetSimulator(seed=42)
        S0 = np.array([100, 100])
        sigma = np.array([0.2, 0.3])
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        paths = sim.generate_paths(S0, 0.05, sigma, corr, 1.0, 252, 10000)
        returns1 = np.log(paths[:, 0, 1:] / paths[:, 0, :-1])
        returns2 = np.log(paths[:, 1, 1:] / paths[:, 1, :-1])
        realized_corr = np.corrcoef(returns1.flatten(), returns2.flatten())[0, 1]
        assert abs(realized_corr - 0.5) < 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])