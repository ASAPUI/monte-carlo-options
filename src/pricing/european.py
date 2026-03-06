"""European option pricing using Monte Carlo simulation with geometric Brownian motion."""
from __future__ import annotations
from typing import Literal, Optional, Tuple
import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import NDArray



class EuropeanOptionPricer:
    """Monte Carlo pricer for European call and put options under Black-Scholes (GBM)."""

    def __init__(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        validate: bool = True,
    ) -> None:
        """Initialize with optional validation.

        Args:
            S0, K, T, r, sigma, q: Standard BS parameters
            validate: Whether to validate immediately (default: True)
        """
        if validate:
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

        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.q = float(q)
        
    def price(
        self,
        option_type: Literal["call", "put"] = "call",
        n_paths: int = 10_000,
        n_steps: int = 50,
        antithetic: bool = True,
        seed: Optional[int] = None,
        rng: Optional[Generator] = None,
    ) -> Tuple[float, float]:
        """Price a European option using Monte Carlo simulation.

        Returns:
            Tuple of (price, standard_error)
        """
        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")

        local_rng = rng if rng is not None else default_rng(seed)

        paths = self.generate_paths(n_paths, n_steps, antithetic=antithetic, rng=local_rng)

        discount = np.exp(-self.r * self.T)
        ST = paths[:, -1]

        if option_type == "call":
            payoffs = np.maximum(ST - self.K, 0.0)
        else:
            payoffs = np.maximum(self.K - ST, 0.0)

        discounted = discount * payoffs
        price = float(np.mean(discounted))
        stderr = float(np.std(discounted, ddof=1) / np.sqrt(n_paths))

        return price, stderr

    def price_call(
        self,
        n_paths: int = 10_000,
        n_steps: int = 100,
        antithetic: bool = True,
        seed: Optional[int] = None,
        rng: Optional[Generator] = None,
    ) -> float:
        """Convenience method: price of a European call."""
        price, _ = self.price("call", n_paths, n_steps, antithetic, seed, rng)
        return price

    def price_put(
        self,
        n_paths: int = 10_000,
        n_steps: int = 100,
        antithetic: bool = True,
        seed: Optional[int] = None,
        rng: Optional[Generator] = None,
    ) -> float:
        """Convenience method: price of a European put."""
        price, _ = self.price("put", n_paths, n_steps, antithetic, seed, rng)
        return price

    def generate_paths(
        self,
        n_paths: int,
        n_steps: int,
        antithetic: bool = True,
        rng: Optional[Generator] = None,
    ) -> NDArray[np.float64]:
        """Generate GBM price paths.

        Args:
            n_paths: Target number of paths
            n_steps: Number of time steps
            antithetic: Use antithetic variates (requires even n_paths)
            rng: numpy Generator to use for random numbers

        Returns:
            Array of shape (n_paths, n_steps + 1) with simulated price paths

        Raises:
            ValueError: If antithetic=True and n_paths is odd
        """
        if rng is None:
            rng = default_rng()

        dt = self.T / n_steps
        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)

        if antithetic:
            if n_paths % 2 != 0:
                raise ValueError(
                    f"antithetic=True requires even number of paths (got {n_paths}). "
                    "Either use an even value or set antithetic=False."
                )
            n_sim = n_paths // 2
        else:
            n_sim = n_paths

        # Generate normal random variables
        Z = rng.standard_normal(size=(n_sim, n_steps))

        if antithetic:
            Z = np.concatenate([Z, -Z], axis=0)

        # Vectorized GBM simulation
        log_increments = drift + diffusion * Z
        log_paths = np.cumsum(log_increments, axis=1)
        paths = self.S0 * np.exp(log_paths)
        paths = np.insert(paths, 0, self.S0, axis=1)  # insert S0 at t=0

        return paths

    def price_call_paths(self, paths: NDArray[np.float64]) -> float:
        """Compute call price from pre-generated paths."""
        payoffs = np.maximum(paths[:, -1] - self.K, 0.0)
        return float(np.mean(np.exp(-self.r * self.T) * payoffs))

    def price_put_paths(self, paths: NDArray[np.float64]) -> float:
        """Compute put price from pre-generated paths."""
        payoffs = np.maximum(self.K - paths[:, -1], 0.0)
        return float(np.mean(np.exp(-self.r * self.T) * payoffs))


# Quick example usage:
if __name__ == "__main__":
    pricer = EuropeanOptionPricer(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.0)

    price, se = pricer.price("call", n_paths=100_000, n_steps=200, antithetic=True)
    print(f"Call price = {price:.4f} ± {se:.4f}")

    # Or with custom RNG (useful for parallel / reproducible sub-simulations)
    rng = default_rng(12345)
    put_price = pricer.price_put(n_paths=50_000, rng=rng)
    print(f"Put price = {put_price:.4f}")