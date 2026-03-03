"""
tests/test_accuracy.py
Financial Accuracy Test Suite for Monte Carlo Options Pricing Engine

This module contains critical financial accuracy tests ensuring the Monte Carlo
implementation adheres to theoretical no-arbitrage bounds and convergence properties.
(wlh hta msali ana hhhhhhhhhhh)
"""

import numpy as np
import pytest
from scipy.stats import norm
from typing import Dict, List, Tuple
import warnings

# Import the pricing modules (adjust paths based on actual project structure)
try:
    from src.pricing.european import EuropeanOptionPricer
    from src.pricing.american import LongstaffSchwartzPricer
    from src.pricing.greeks import GreeksCalculator
    from src.models.black_scholes import BlackScholesModel
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from pricing.european import EuropeanOptionPricer
    from pricing.american import LongstaffSchwartzPricer
    from pricing.greeks import GreeksCalculator
    from models.black_scholes import BlackScholesModel


# ============================================================================
# Shared Utilities (avoid instantiating test classes inside tests)
# ============================================================================

def black_scholes_call(S: float, K: float, T: float, 
                       r: float, sigma: float, q: float = 0) -> float:
    """
    Closed-form Black-Scholes formula for European call option.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity
        r: Risk-free interest rate
        sigma: Volatility
        q: Dividend yield
        
    Returns:
        Theoretical call option price
    """
    if T <= 0:
        return max(S - K, 0)
    
    if sigma < 0:
        raise ValueError("Volatility cannot be negative")
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = (S * np.exp(-q * T) * norm.cdf(d1) - 
                  K * np.exp(-r * T) * norm.cdf(d2))
    return call_price


def black_scholes_put(S: float, K: float, T: float,
                      r: float, sigma: float, q: float = 0) -> float:
    """Closed-form Black-Scholes for European put using put-call parity."""
    call = black_scholes_call(S, K, T, r, sigma, q)
    return call - S * np.exp(-q * T) + K * np.exp(-r * T)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def standard_params() -> Dict:
    """Standard test parameters from Hull/Black-Scholes literature."""
    return {
        'S0': 100.0,
        'K': 100.0,
        'T': 1.0,
        'r': 0.05,
        'sigma': 0.20,
        'q': 0.0
    }


@pytest.fixture
def parity_params() -> List[Dict]:
    """Various parameter sets to test parity under different conditions."""
    return [
        {'S0': 100, 'K': 100, 'T': 1.0, 'r': 0.05, 'sigma': 0.2, 'q': 0.0, 'name': 'ATM'},
        {'S0': 120, 'K': 100, 'T': 1.0, 'r': 0.05, 'sigma': 0.2, 'q': 0.0, 'name': 'ITM_Call'},
        {'S0': 80, 'K': 100, 'T': 1.0, 'r': 0.05, 'sigma': 0.2, 'q': 0.0, 'name': 'OTM_Call'},
        {'S0': 100, 'K': 100, 'T': 0.1, 'r': 0.05, 'sigma': 0.2, 'q': 0.0, 'name': 'Short_Maturity'},
        {'S0': 100, 'K': 100, 'T': 1.0, 'r': 0.05, 'sigma': 0.5, 'q': 0.0, 'name': 'High_Vol'},
        {'S0': 100, 'K': 100, 'T': 1.0, 'r': 0.05, 'sigma': 0.2, 'q': 0.03, 'name': 'With_Dividends'},
        {'S0': 100, 'K': 100, 'T': 1.0, 'r': 0.10, 'sigma': 0.2, 'q': 0.0, 'name': 'High_Rates'},
    ]


@pytest.fixture
def american_test_params() -> List[Dict]:
    """Parameters for American option testing."""
    return [
        {'S0': 100, 'K': 100, 'T': 1.0, 'r': 0.05, 'sigma': 0.2, 'name': 'ATM'},
        {'S0': 80, 'K': 100, 'T': 1.0, 'r': 0.05, 'sigma': 0.2, 'name': 'ITM_Put'},
        {'S0': 120, 'K': 100, 'T': 1.0, 'r': 0.05, 'sigma': 0.2, 'name': 'ITM_Call'},
        {'S0': 100, 'K': 100, 'T': 0.25, 'r': 0.05, 'sigma': 0.2, 'name': 'Short'},
        {'S0': 100, 'K': 100, 'T': 1.0, 'r': 0.10, 'sigma': 0.2, 'name': 'High_Rates'},
    ]


# ============================================================================
# Test Classes
# ============================================================================

class TestBSConvergence:
    """
    Test suite for Black-Scholes convergence properties.
    
    As N → ∞, Monte Carlo estimates must converge to closed-form solutions
    by the Law of Large Numbers and Central Limit Theorem.
    """
    
    def test_european_call_convergence(self, standard_params):
        """
        BS Convergence Test: European call price must be within 0.5% of 
        Black-Scholes closed-form at N=100,000 paths.
        
        Validates the Monte Carlo implementation against the analytical
        benchmark, ensuring path generation and discounting are correct.
        """
        params = standard_params
        N_paths = 100_000
        tolerance = 0.005  # 0.5% relative error
        
        # Calculate theoretical price using shared utility
        bs_price = black_scholes_call(
            params['S0'], params['K'], params['T'],
            params['r'], params['sigma'], params['q']
        )
        
        # Single seed for this test only - not reused across parameters
        np.random.seed(2024)
        pricer = EuropeanOptionPricer(**params)
        
        # Monte Carlo simulation
        mc_price = pricer.price_call(n_paths=N_paths, n_steps=50)
        
        # Calculate relative error
        relative_error = abs(mc_price - bs_price) / bs_price
        
        assert relative_error < tolerance, (
            f"Monte Carlo price {mc_price:.6f} deviates from BS price {bs_price:.6f} "
            f"by {relative_error:.4%} (tolerance: {tolerance:.1%}). "
            f"Convergence failed at N={N_paths:,} paths."
        )
        
        # Additional check: Price must be positive and reasonable
        assert mc_price > 0, "Option price must be positive"
        assert mc_price < params['S0'], "Call price cannot exceed spot price"


class TestPutCallParity:
    """
    Test suite for Put-Call Parity relationships.
    
    Put-Call Parity is a fundamental no-arbitrage relationship:
    C - P = S*e^(-qT) - K*e^(-rT)
    
    Uses paired simulations with antithetic variates for true consistency.
    """
    
    def test_put_call_parity_antithetic(self, parity_params):
        """
        Put-Call Parity Test using antithetic variates for path consistency.
        
        Instead of resetting seeds (which creates false correlations across params),
        we use antithetic variates within each parameter set to ensure the same
        random shocks drive both call and put valuations.
        """
        tolerance = 0.01  # 1% relative tolerance
        
        for param_set in parity_params:
            # Create fresh pricer with unique seed derived from parameter hash
            # This ensures different params get different paths, same param gets same paths
            seed = hash(frozenset(param_set.items())) % (2**32)
            np.random.seed(seed)
            
            # Create pricer with antithetic variates enabled for path consistency
            pricer = EuropeanOptionPricer(**{k: v for k, v in param_set.items() 
                                              if k != 'name'})
            
            # Generate paths once and price both derivatives on same paths
            # This is the architecturally correct way to test parity
            paths = pricer.generate_paths(n_paths=50_000, n_steps=50, 
                                          antithetic=True)
            call_price = pricer.price_call_paths(paths)
            put_price = pricer.price_put_paths(paths)
            
            # Left side: C - P
            lhs = call_price - put_price
            
            # Right side: S*e^(-qT) - K*e^(-rT)
            rhs = (param_set['S0'] * np.exp(-param_set['q'] * param_set['T']) - 
                   param_set['K'] * np.exp(-param_set['r'] * param_set['T']))
            
            # Calculate relative deviation
            if abs(rhs) > 1e-10:
                relative_error = abs(lhs - rhs) / abs(rhs)
            else:
                relative_error = abs(lhs - rhs)
            
            assert relative_error < tolerance, (
                f"Put-Call Parity violated for {param_set.get('name', 'unknown')}: "
                f"C-P = {lhs:.6f}, S*e^(-qT) - K*e^(-rT) = {rhs:.6f}, "
                f"error = {relative_error:.4%}"
            )


class TestGreekSigns:
    """
    Test suite for Greeks monotonicity and sign constraints.
    """
    
    @pytest.fixture
    def greek_params(self) -> Dict:
        """Parameters for Greek calculation tests."""
        return {
            'S0': 100.0,
            'K': 100.0,
            'T': 1.0,
            'r': 0.05,
            'sigma': 0.20,
            'q': 0.0
        }
    
    def test_delta_bounds_call(self, greek_params):
        """Delta must be in (0,1) for European calls."""
        calculator = GreeksCalculator(**greek_params)
        delta = calculator.delta_call(method='central', h=0.01)
        
        assert 0 < delta < 1, (
            f"Call Delta {delta:.6f} violates bounds (0,1)."
        )
    
    def test_delta_bounds_put(self, greek_params):
        """Delta must be in (-1,0) for European puts."""
        calculator = GreeksCalculator(**greek_params)
        delta = calculator.delta_put(method='central', h=0.01)
        
        assert -1 < delta < 0, (
            f"Put Delta {delta:.6f} violates bounds (-1,0)."
        )
    
    def test_gamma_positive(self, greek_params):
        """Gamma must be > 0 for long positions (convexity)."""
        calculator = GreeksCalculator(**greek_params)
        gamma = calculator.gamma(method='central', h=0.5)
        
        assert gamma > 0, (
            f"Gamma {gamma:.6f} is not positive. Long options must have positive Gamma."
        )
    
    def test_vega_positive(self, greek_params):
        """Vega must be > 0 for long positions."""
        calculator = GreeksCalculator(**greek_params)
        vega = calculator.vega(method='finite_diff', h=0.001)
        
        assert vega > 0, (
            f"Vega {vega:.6f} is not positive. Long options must have positive Vega."
        )
    
    def test_theta_negative(self, greek_params):
        """Theta must be < 0 for long European options (time decay)."""
        calculator = GreeksCalculator(**greek_params)
        theta = calculator.theta_call(method='finite_diff', h=1/365)
        
        assert theta < 0, (
            f"Theta {theta:.6f} is not negative. Long options must have negative Theta."
        )


class TestAmericanEuropeanRelationship:
    """
    American options must always be worth at least as much as European
    options with identical parameters due to early exercise premium.
    """
    
    def test_american_geq_european_call(self, american_test_params):
        """American call >= European call."""
        for param_set in american_test_params:
            # Unique seed per parameter set
            seed = hash(frozenset(param_set.items())) % (2**32)
            np.random.seed(seed)
            
            params = {k: v for k, v in param_set.items() if k != 'name'}
            
            european_pricer = EuropeanOptionPricer(**params, q=0.0)
            european_price = european_pricer.price_call(n_paths=20_000, n_steps=50)
            
            american_pricer = LongstaffSchwartzPricer(**params)
            american_price = american_pricer.price_call(n_paths=20_000, n_steps=50)
            
            tolerance = 0.01
            
            assert american_price >= european_price - tolerance, (
                f"American call {american_price:.6f} < European call {european_price:.6f} "
                f"for {param_set.get('name', 'unknown')}. American >= European required."
            )
    
    def test_american_geq_european_put(self, american_test_params):
        """American put >= European put."""
        for param_set in american_test_params:
            seed = hash(frozenset(param_set.items())) % (2**32)
            np.random.seed(seed)
            
            params = {k: v for k, v in param_set.items() if k != 'name'}
            
            european_pricer = EuropeanOptionPricer(**params, q=0.0)
            european_price = european_pricer.price_put(n_paths=20_000, n_steps=50)
            
            american_pricer = LongstaffSchwartzPricer(**params)
            american_price = american_pricer.price_put(n_paths=20_000, n_steps=50)
            
            tolerance = 0.01
            
            assert american_price >= european_price - tolerance, (
                f"American put {american_price:.6f} < European put {european_price:.6f} "
                f"for {param_set.get('name', 'unknown')}."
            )


class TestBoundaryConditions:
    """
    Test suite for boundary condition asymptotics.
    """
    
    def test_deep_itm_call_approaches_intrinsic(self):
        """Deep ITM call approaches S - K*e^(-rT)."""
        S0, K, T, r, sigma = 1000.0, 100.0, 1.0, 0.05, 0.20
        
        np.random.seed(42)
        pricer = EuropeanOptionPricer(S0=S0, K=K, T=T, r=r, sigma=sigma, q=0.0)
        mc_price = pricer.price_call(n_paths=50_000, n_steps=50)
        
        theoretical_limit = S0 - K * np.exp(-r * T)
        relative_diff = abs(mc_price - theoretical_limit) / theoretical_limit
        
        assert relative_diff < 0.01, (
            f"Deep ITM call price {mc_price:.6f} deviates from limit {theoretical_limit:.6f} "
            f"by {relative_diff:.4%}."
        )
    
    def test_deep_otm_call_approaches_zero(self):
        """Deep OTM call approaches 0."""
        S0, K, T, r, sigma = 10.0, 100.0, 1.0, 0.05, 0.20
        
        np.random.seed(42)
        pricer = EuropeanOptionPricer(S0=S0, K=K, T=T, r=r, sigma=sigma, q=0.0)
        mc_price = pricer.price_call(n_paths=50_000, n_steps=50)
        
        assert mc_price < 0.001 * K, (
            f"Deep OTM call price {mc_price:.6f} is not approaching zero."
        )
    
    def test_zero_vol_call_equals_intrinsic(self):
        """Zero volatility call = max(S - K*e^(-rT), 0)."""
        S0, K, T, r = 100.0, 100.0, 1.0, 0.05
        
        # Test with multiple step counts to catch step-related bugs
        for n_steps in [1, 50]:
            np.random.seed(42)
            pricer = EuropeanOptionPricer(S0=S0, K=K, T=T, r=r, sigma=0.0, q=0.0)
            mc_price = pricer.price_call(n_paths=10_000, n_steps=n_steps)
            
            theoretical = max(S0 - K * np.exp(-r * T), 0)
            
            assert abs(mc_price - theoretical) < 0.01, (
                f"Zero-vol call price {mc_price:.6f} != intrinsic {theoretical:.6f} "
                f"at n_steps={n_steps}."
            )
    
    def test_zero_vol_put_equals_intrinsic(self):
        """Zero volatility put = max(K*e^(-rT) - S, 0)."""
        S0, K, T, r = 100.0, 100.0, 1.0, 0.05
        
        for n_steps in [1, 50]:
            np.random.seed(42)
            pricer = EuropeanOptionPricer(S0=S0, K=K, T=T, r=r, sigma=0.0, q=0.0)
            mc_price = pricer.price_put(n_paths=10_000, n_steps=n_steps)
            
            theoretical = max(K * np.exp(-r * T) - S0, 0)
            
            assert abs(mc_price - theoretical) < 0.01, (
                f"Zero-vol put price {mc_price:.6f} != intrinsic {theoretical:.6f} "
                f"at n_steps={n_steps}."
            )
    
    def test_expired_option_equals_payoff(self):
        """At expiration (T=0), option = intrinsic value."""
        S0, K, r, sigma = 100.0, 100.0, 0.05, 0.20
        
        pricer = EuropeanOptionPricer(S0=S0, K=K, T=0.0, r=r, sigma=sigma, q=0.0)
        
        call_price = pricer.price_call(n_paths=1, n_steps=1)
        put_price = pricer.price_put(n_paths=1, n_steps=1)
        
        # Use pytest.approx for safe float comparison
        assert call_price == pytest.approx(max(S0 - K, 0), abs=1e-9), \
            "Expired call must equal intrinsic value"
        assert put_price == pytest.approx(max(K - S0, 0), abs=1e-9), \
            "Expired put must equal intrinsic value"


class TestStatisticalProperties:
    """
    Statistical tests for Monte Carlo convergence properties.
    """
    
    def test_monte_carlo_convergence_rate_robust(self):
        """
        Verify Monte Carlo converges at rate O(1/sqrt(N)) using median of trials.
        
        Uses multiple trials at each path count to reduce variance, then checks
        that the median error ratio follows the expected 1/sqrt(N) decay.
        """
        S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        
        pricer = EuropeanOptionPricer(S0=S0, K=K, T=T, r=r, sigma=sigma)
        bs_price = black_scholes_call(S0, K, T, r, sigma)
        
        # Use larger path counts to reduce flakiness
        path_counts = [10_000, 40_000, 160_000]
        n_trials = 5  # Multiple trials per path count
        
        median_errors = []
        
        for n_paths in path_counts:
            trial_errors = []
            for trial in range(n_trials):
                # Unique seed for each trial
                np.random.seed(1000 * n_paths + trial)
                mc_price = pricer.price_call(n_paths=n_paths, n_steps=50)
                error = abs(mc_price - bs_price)
                trial_errors.append(error)
            
            median_errors.append(np.median(trial_errors))
        
        # Check median ratios follow ~2.0 (sqrt(4))
        for i in range(1, len(median_errors)):
            ratio = median_errors[i-1] / median_errors[i]
            # More lenient bounds for median-based testing
            assert 1.3 < ratio < 2.5, (
                f"Convergence rate anomaly: median error ratio {ratio:.2f} between "
                f"N={path_counts[i-1]} and N={path_counts[i]}. "
                f"Expected ~2.0 for O(1/sqrt(N)) convergence."
            )


class TestInputValidation:
    """
    Negative tests for invalid inputs.
    """
    
    def test_negative_volatility_raises_error(self):
        """Negative volatility should raise ValueError."""
        with pytest.raises((ValueError, AssertionError)):
            pricer = EuropeanOptionPricer(S0=100, K=100, T=1.0, r=0.05, sigma=-0.2)
            pricer.price_call(n_paths=1000, n_steps=10)
    
    def test_negative_time_raises_error(self):
        """Negative time to maturity should raise ValueError."""
        with pytest.raises((ValueError, AssertionError)):
            pricer = EuropeanOptionPricer(S0=100, K=100, T=-1.0, r=0.05, sigma=0.2)
            pricer.price_call(n_paths=1000, n_steps=10)
    
    def test_zero_spot_raises_error(self):
        """Zero or negative spot price should raise ValueError."""
        with pytest.raises((ValueError, AssertionError)):
            pricer = EuropeanOptionPricer(S0=0, K=100, T=1.0, r=0.05, sigma=0.2)
            pricer.price_call(n_paths=1000, n_steps=10)
        
        with pytest.raises((ValueError, AssertionError)):
            pricer = EuropeanOptionPricer(S0=-50, K=100, T=1.0, r=0.05, sigma=0.2)
            pricer.price_call(n_paths=1000, n_steps=10)
    
    def test_negative_strike_raises_error(self):
        """Negative strike price should raise ValueError."""
        with pytest.raises((ValueError, AssertionError)):
            pricer = EuropeanOptionPricer(S0=100, K=-100, T=1.0, r=0.05, sigma=0.2)
            pricer.price_call(n_paths=1000, n_steps=10)
    
    def test_zero_paths_raises_error(self):
        """Zero paths should raise ValueError."""
        pricer = EuropeanOptionPricer(S0=100, K=100, T=1.0, r=0.05, sigma=0.2)
        with pytest.raises((ValueError, AssertionError)):
            pricer.price_call(n_paths=0, n_steps=10)
    
    def test_zero_steps_raises_error(self):
        """Zero time steps should raise ValueError."""
        pricer = EuropeanOptionPricer(S0=100, K=100, T=1.0, r=0.05, sigma=0.2)
        with pytest.raises((ValueError, AssertionError)):
            pricer.price_call(n_paths=1000, n_steps=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])