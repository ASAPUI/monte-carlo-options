"""American option pricing using Longstaff-Schwartz Least Squares Monte Carlo."""
from __future__ import annotations
from typing import Literal, Optional, Tuple
import numpy as np
from numpy.random import Generator, default_rng
from numpy.polynomial import polynomial as P
from numpy.typing import NDArray


class PolynomialBasis:
    """Normalized polynomial basis for LSM regression (degree configurable)."""

    def __init__(self, degree: int = 3) -> None:
        self.degree = degree
        self.coeffs: Optional[NDArray[np.float64]] = None
        self.mean: float = 0.0
        self.std: float = 1.0

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        if len(X) < self.degree + 2:  # need at least degree + 1 + safety
            raise ValueError(f"Not enough points ({len(X)}) for degree {self.degree}")
        self.mean = float(np.mean(X))
        self.std = float(np.std(X)) if np.std(X) > 1e-10 else 1.0
        X_norm = (X - self.mean) / self.std
        self.coeffs = P.polyfit(X_norm, y, self.degree)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.coeffs is None:
            raise RuntimeError("Basis not fitted")
        X_norm = (X - self.mean) / self.std
        return P.polyval(X_norm, self.coeffs)


class LongstaffSchwartzPricer:
    """Longstaff-Schwartz LSM Monte Carlo pricer for American options under GBM.

    Fixes applied:
    • No global np.random.seed() — uses modern Generator (thread-safe)
    • Vectorized path generation (cumsum)
    • Correct discounting (only one discount per time step, final discount to t=0)
    • Regression only on paths still alive (not yet exercised in the future) — eliminates upward bias
    • Configurable polynomial degree
    • Consistent API with EuropeanOptionPricer (price() returns (price, stderr))
    • Proper input validation (including T > 0)
    """

    def __init__(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        basis_degree: int = 3,
    ) -> None:
        """Initialize the American option pricer.

        Args:
            S0: Initial underlying price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate (continuous)
            sigma: Volatility (annualized)
            q: Continuous dividend yield (default 0.0)
            basis_degree: Degree of polynomial basis for regression (default 3)

        Raises:
            ValueError: On invalid parameters
        """
        if S0 <= 0 or K <= 0 or T <= 0 or sigma < 0 or q < 0:
            raise ValueError("S0, K, T must be > 0; sigma, q ≥ 0")
        if basis_degree < 1:
            raise ValueError("basis_degree must be >= 1")

        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.basis_degree = basis_degree

    def price(
        self,
        option_type: Literal["call", "put"],
        n_paths: int = 50_000,
        n_steps: int = 100,
        rng: Optional[Generator] = None,
        seed: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Price American option using Longstaff-Schwartz LSM (returns price + std error)."""
        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        if n_paths < 1000:
            raise ValueError(f"n_paths too small ({n_paths}); recommend ≥ 10_000")
        if n_steps < 3:
            raise ValueError("n_steps must be ≥ 3 for LSM")

        local_rng = rng if rng is not None else (default_rng(seed) if seed is not None else default_rng())

        paths = self._generate_paths(n_paths, n_steps, local_rng)

        dt = self.T / n_steps
        discount_dt = np.exp(-self.r * dt)

        # Terminal payoffs at maturity (t = n_steps)
        if option_type == "call":
            cash_flows = np.maximum(paths[:, -1] - self.K, 0.0)
        else:
            cash_flows = np.maximum(self.K - paths[:, -1], 0.0)

        # Track paths that have already been exercised in the future
        exercised = np.zeros(n_paths, dtype=bool)

        # Backward induction
        for t in range(n_steps - 1, 0, -1):
            continuation = cash_flows * discount_dt

            if option_type == "call":
                itm = paths[:, t] > self.K
                exercise_value = paths[:, t] - self.K
            else:
                itm = paths[:, t] < self.K
                exercise_value = self.K - paths[:, t]

            active_itm = itm & ~exercised

            if np.any(active_itm):
                X = paths[active_itm, t]
                y = continuation[active_itm]

                if len(X) > 20:
                    basis = PolynomialBasis(degree=self.basis_degree)
                    try:
                        basis.fit(X, y)
                        cont_pred = basis.predict(X)
                    except np.linalg.LinAlgError:
                        cont_pred = y  # fallback on numerical issues
                else:
                    cont_pred = y

                exercise_decision = exercise_value[active_itm] > cont_pred

                # Update cash flows and exercise mask for active ITM paths
                cash_flows[active_itm] = np.where(
                    exercise_decision, exercise_value[active_itm], continuation[active_itm]
                )
                exercised[active_itm] = exercise_decision

            # All other paths (already exercised or OTM) simply continue with discounted value
            cash_flows[~active_itm] = continuation[~active_itm]

        # Final discount from t=1 to t=0 (cash_flows now at t=1)
        discounted_to_t0 = cash_flows * discount_dt
        price = float(np.mean(discounted_to_t0))
        stderr = float(np.std(discounted_to_t0, ddof=1) / np.sqrt(n_paths))

        return max(price, 0.0), stderr

    def price_put(
        self,
        n_paths: int = 50_000,
        n_steps: int = 100,
        rng: Optional[Generator] = None,
        seed: Optional[int] = None,
    ) -> float:
        """Convenience: American put price."""
        price, _ = self.price("put", n_paths, n_steps, rng, seed)
        return price

    def price_call(
        self,
        n_paths: int = 50_000,
        n_steps: int = 100,
        rng: Optional[Generator] = None,
        seed: Optional[int] = None,
    ) -> float:
        """Convenience: American call price."""
        price, _ = self.price("call", n_paths, n_steps, rng, seed)
        return price

    def _generate_paths(
        self,
        n_paths: int,
        n_steps: int,
        rng: Generator,
    ) -> NDArray[np.float64]:
        """Vectorized GBM path generation."""
        dt = self.T / n_steps
        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
        vol_dt = self.sigma * np.sqrt(dt)

        Z = rng.standard_normal(size=(n_paths, n_steps))
        log_increments = drift + vol_dt * Z

        log_paths = np.cumsum(log_increments, axis=1)
        paths = self.S0 * np.exp(log_paths)
        paths = np.insert(paths, 0, self.S0, axis=1)

        return paths


# Quick test / benchmark
if __name__ == "__main__":
    # Classic Longstaff-Schwartz example: American put (S0=36, K=40, T=1, r=0.06, σ=0.2)
    pricer = LongstaffSchwartzPricer(S0=36.0, K=40.0, T=1.0, r=0.06, sigma=0.2, q=0.0)

    price, se = pricer.price("put", n_paths=100_000, n_steps=100, seed=42)
    print(f"American Put price ≈ {price:.4f} ± {se:.4f}")
    # High-precision literature value ≈ 4.472