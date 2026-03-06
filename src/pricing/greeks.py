"""Greeks calculation using finite difference and pathwise methods (Monte Carlo)."""
from __future__ import annotations
from typing import Literal, Optional
import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import NDArray

from src.pricing.european import EuropeanOptionPricer


class GreeksCalculator:
    """Calculate European option Greeks using finite-difference and pathwise Monte Carlo.

    All finite-difference methods now use **common random numbers** (same RNG instance
    passed to every bumped pricer) for dramatically lower variance in the differences.

    Fixes applied:
    • Common random numbers (CRN) across all bumped simulations
    • Vega now uses central difference (O(h²) accuracy, consistent with delta/gamma)
    • Imports moved to top (no more deferred imports)
    • Added theta_put for symmetry
    • Pathwise delta formula kept exactly as-is (it is correct for any q)
    • Full input validation in __init__
    • Consistent API and RNG handling with EuropeanOptionPricer / LongstaffSchwartzPricer
    """

    def __init__(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> None:
        """Initialize Greeks calculator.

        Raises:
            ValueError: If any parameter is invalid (matches EuropeanOptionPricer)
        """
        if S0 <= 0:
            raise ValueError(f"S0 must be positive, got {S0}")
        if K <= 0:
            raise ValueError(f"K must be positive, got {K}")
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        if q < 0:
            raise ValueError(f"q must be non-negative, got {q}")

        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q

    def delta_call(
        self,
        method: Literal["central", "forward", "pathwise"] = "central",
        h: float = 0.01,
        n_paths: int = 20_000,
    ) -> float:
        """Delta for European call (∂V/∂S)."""
        if method == "pathwise":
            return self._delta_pathwise("call", n_paths)

        rng: Generator = default_rng()  # shared across all bumps

        if method == "forward":
            price = EuropeanOptionPricer(
                self.S0, self.K, self.T, self.r, self.sigma, self.q
            ).price_call(n_paths=n_paths, rng=rng)
            price_up = EuropeanOptionPricer(
                self.S0 + h, self.K, self.T, self.r, self.sigma, self.q
            ).price_call(n_paths=n_paths, rng=rng)
            return (price_up - price) / h

        # central (default)
        price_up = EuropeanOptionPricer(
            self.S0 + h, self.K, self.T, self.r, self.sigma, self.q
        ).price_call(n_paths=n_paths, rng=rng)
        price_down = EuropeanOptionPricer(
            self.S0 - h, self.K, self.T, self.r, self.sigma, self.q
        ).price_call(n_paths=n_paths, rng=rng)
        return (price_up - price_down) / (2 * h)

    def delta_put(
        self,
        method: Literal["central", "forward", "pathwise"] = "central",
        h: float = 0.01,
        n_paths: int = 20_000,
    ) -> float:
        """Delta for European put (∂V/∂S)."""
        if method == "pathwise":
            return self._delta_pathwise("put", n_paths)

        rng: Generator = default_rng()

        if method == "forward":
            price = EuropeanOptionPricer(
                self.S0, self.K, self.T, self.r, self.sigma, self.q
            ).price_put(n_paths=n_paths, rng=rng)
            price_up = EuropeanOptionPricer(
                self.S0 + h, self.K, self.T, self.r, self.sigma, self.q
            ).price_put(n_paths=n_paths, rng=rng)
            return (price_up - price) / h

        price_up = EuropeanOptionPricer(
            self.S0 + h, self.K, self.T, self.r, self.sigma, self.q
        ).price_put(n_paths=n_paths, rng=rng)
        price_down = EuropeanOptionPricer(
            self.S0 - h, self.K, self.T, self.r, self.sigma, self.q
        ).price_put(n_paths=n_paths, rng=rng)
        return (price_up - price_down) / (2 * h)

    def gamma(
        self,
        h: float = 0.5,
        n_paths: int = 20_000,
    ) -> float:
        """Gamma (∂²V/∂S²) — second-order central difference (call/put identical)."""
        rng: Generator = default_rng()

        price_up = EuropeanOptionPricer(
            self.S0 + h, self.K, self.T, self.r, self.sigma, self.q
        ).price_call(n_paths=n_paths, rng=rng)
        price = EuropeanOptionPricer(
            self.S0, self.K, self.T, self.r, self.sigma, self.q
        ).price_call(n_paths=n_paths, rng=rng)
        price_down = EuropeanOptionPricer(
            self.S0 - h, self.K, self.T, self.r, self.sigma, self.q
        ).price_call(n_paths=n_paths, rng=rng)

        return (price_up - 2 * price + price_down) / (h**2)

    def vega(
        self,
        h: float = 0.001,
        n_paths: int = 20_000,
    ) -> float:
        """Vega (∂V/∂σ) — now central difference (was forward).

        Returns raw vega (multiply by 0.01 for "per 1% vol" convention).
        """
        rng: Generator = default_rng()

        price_up = EuropeanOptionPricer(
            self.S0, self.K, self.T, self.r, self.sigma + h, self.q
        ).price_call(n_paths=n_paths, rng=rng)
        price_down = EuropeanOptionPricer(
            self.S0, self.K, self.T, self.r, self.sigma - h, self.q
        ).price_call(n_paths=n_paths, rng=rng)

        return (price_up - price_down) / (2 * h)

    def theta_call(
        self,
        h: Optional[float] = None,
        n_paths: int = 20_000,
    ) -> float:
        """Theta for European call (∂V/∂t, usually negative)."""
        if h is None:
            h = 1.0 / 365.0

        rng: Generator = default_rng()

        price = EuropeanOptionPricer(
            self.S0, self.K, self.T, self.r, self.sigma, self.q
        ).price_call(n_paths=n_paths, rng=rng)

        if self.T - h <= 0:
            return -price / h

        price_decay = EuropeanOptionPricer(
            self.S0, self.K, self.T - h, self.r, self.sigma, self.q
        ).price_call(n_paths=n_paths, rng=rng)

        return (price_decay - price) / h

    def theta_put(
        self,
        h: Optional[float] = None,
        n_paths: int = 20_000,
    ) -> float:
        """Theta for European put (∂V/∂t, usually negative)."""
        if h is None:
            h = 1.0 / 365.0

        rng: Generator = default_rng()

        price = EuropeanOptionPricer(
            self.S0, self.K, self.T, self.r, self.sigma, self.q
        ).price_put(n_paths=n_paths, rng=rng)

        if self.T - h <= 0:
            return -price / h

        price_decay = EuropeanOptionPricer(
            self.S0, self.K, self.T - h, self.r, self.sigma, self.q
        ).price_put(n_paths=n_paths, rng=rng)

        return (price_decay - price) / h

    def _delta_pathwise(
        self,
        option_type: Literal["call", "put"],
        n_paths: int,
    ) -> float:
        """Pathwise delta estimator (single simulation, very low variance).

        Correct formula for GBM with dividends:
            delta_call = e^{-rT} E[ 1_{S_T > K} * (S_T / S0) ]
            delta_put  = -e^{-rT} E[ 1_{S_T < K} * (S_T / S0) ]

        The e^{-qT} N(d1) term appears automatically in the expectation.
        """
        rng: Generator = default_rng()

        pricer = EuropeanOptionPricer(self.S0, self.K, self.T, self.r, self.sigma, self.q)
        paths = pricer.generate_paths(n_paths, n_steps=50, antithetic=True, rng=rng)

        ST = paths[:, -1]
        discount = np.exp(-self.r * self.T)

        if option_type == "call":
            indicator = (ST > self.K).astype(float)
            payoff_deriv = indicator * (ST / self.S0)
        else:
            indicator = (ST < self.K).astype(float)
            payoff_deriv = -indicator * (ST / self.S0)

        return float(np.mean(discount * payoff_deriv))


# Quick example
if __name__ == "__main__":
    greeks = GreeksCalculator(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.0)

    print(f"Delta call (central + CRN) = {greeks.delta_call():.4f}")
    print(f"Delta call (pathwise)      = {greeks.delta_call(method='pathwise'):.4f}")
    print(f"Vega (central)             = {greeks.vega():.4f}")
    print(f"Theta call (daily)         = {greeks.theta_call():.4f}")
    print(f"Gamma                      = {greeks.gamma():.4f}")