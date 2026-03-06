# src/validation.py
"""Input validation for option pricing models.

Provides comprehensive validation for option parameters with clear,
actionable error messages. Includes the Feller condition check for
stochastic volatility models.
"""

from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Concatenate, ParamSpec, TypeVar

# Type variables for decorator
P = ParamSpec("P")
T = TypeVar("T")


class ValidationError(ValueError):
    """Raised when option parameters fail validation.
    
    Attributes:
        field: Which parameter failed validation
        value: The invalid value provided
        constraint: Description of the valid range
    """
    
    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        constraint: str | None = None,
    ) -> None:
        self.field = field
        self.value = value
        self.constraint = constraint
        
        parts = [message]
        if field:
            parts.append(f" (field: '{field}')")
        if value is not None:
            parts.append(f" (got: {value!r})")
        if constraint:
            parts.append(f" (expected: {constraint})")
            
        super().__init__("".join(parts))


@dataclass(frozen=True)
class OptionParameters:
    """Immutable container for validated option parameters."""
    
    S0: float  # Spot price
    K: float   # Strike price
    T: float   # Time to maturity
    r: float   # Risk-free rate
    sigma: float  # Volatility
    n_paths: int  # Number of MC paths
    q: float = 0.0  # Dividend yield (optional)
    
    def __post_init__(self) -> None:
        """Ensure all values are valid types."""
        object.__setattr__(self, 'S0', float(self.S0))
        object.__setattr__(self, 'K', float(self.K))
        object.__setattr__(self, 'T', float(self.T))
        object.__setattr__(self, 'r', float(self.r))
        object.__setattr__(self, 'sigma', float(self.sigma))
        object.__setattr__(self, 'n_paths', int(self.n_paths))
        object.__setattr__(self, 'q', float(self.q))


@dataclass(frozen=True)
class HestonParameters:
    """Parameters for Heston stochastic volatility model."""
    
    S0: float      # Spot price
    K: float       # Strike price
    T: float       # Time to maturity
    r: float       # Risk-free rate
    v0: float      # Initial variance
    theta: float   # Long-term variance
    kappa: float   # Mean reversion speed
    sigma: float   # Vol of vol
    rho: float     # Correlation
    n_paths: int   # Number of paths
    
    def check_feller_condition(self) -> None:
        """Validate Feller condition: 2*kappa*theta > sigma^2.
        
        The Feller condition ensures that the variance process stays
        strictly positive. If violated, the variance can hit zero,
        causing numerical instabilities in Monte Carlo simulations.
        
        Raises:
            ValidationError: If Feller condition is violated with
                detailed explanation of the implications.
        """
        lhs = 2 * self.kappa * self.theta
        rhs = self.sigma ** 2
        feller_ratio = lhs / rhs if rhs > 0 else float('inf')
        
        if lhs <= rhs:
            raise ValidationError(
                f"Feller condition violated: 2κθ ({lhs:.6f}) ≤ σ² ({rhs:.6f}). "
                f"Ratio is {feller_ratio:.4f}, must be > 1.0. "
                f"This means the variance process can hit zero, causing "
                f"numerical instabilities and poor convergence in MC simulations. "
                f"To fix: increase kappa (mean reversion) or theta (long-term var), "
                f"or decrease sigma (vol of vol).",
                field="feller_condition",
                value=f"2*{self.kappa}*{self.theta} = {lhs} vs {self.sigma}² = {rhs}",
                constraint="2κθ > σ² (ratio > 1.0)",
            )


def validate_option_params(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    q: float = 0.0,
    *,
    model_type: str = "black_scholes",
    allow_zero_strike: bool = False,
) -> OptionParameters:
    """Validate parameters for option pricing models.
    
    Performs comprehensive validation with specific error messages
    for each constraint violation.
    
    Args:
        S0: Initial spot price (must be > 0)
        K: Strike price (must be > 0, or >= 0 if allow_zero_strike)
        T: Time to maturity (must be >= 0)
        r: Risk-free rate (must be in [-0.5, 0.5])
        sigma: Volatility (must be >= 0)
        n_paths: Number of MC paths (must be >= 1000)
        q: Dividend yield (must be >= 0)
        model_type: Model identifier for error context
        allow_zero_strike: Whether to allow K=0 (for some exotic options)
        
    Returns:
        Validated OptionParameters dataclass
        
    Raises:
        ValidationError: With detailed message about which constraint failed
        TypeError: If parameters have wrong types
    """
    # Type checks
    if not isinstance(S0, (int, float)):
        raise TypeError(f"S0 must be numeric, got {type(S0).__name__}")
    if not isinstance(K, (int, float)):
        raise TypeError(f"K must be numeric, got {type(K).__name__}")
    if not isinstance(T, (int, float)):
        raise TypeError(f"T must be numeric, got {type(T).__name__}")
    if not isinstance(r, (int, float)):
        raise TypeError(f"r must be numeric, got {type(r).__name__}")
    if not isinstance(sigma, (int, float)):
        raise TypeError(f"sigma must be numeric, got {type(sigma).__name__}")
    if not isinstance(n_paths, int):
        raise TypeError(f"n_paths must be integer, got {type(n_paths).__name__}")
    if not isinstance(q, (int, float)):
        raise TypeError(f"q must be numeric, got {type(q).__name__}")
    
    # Convert to float for comparison
    S0_f = float(S0)
    K_f = float(K)
    T_f = float(T)
    r_f = float(r)
    sigma_f = float(sigma)
    q_f = float(q)
    
    # Spot price validation
    if S0_f <= 0:
        raise ValidationError(
            "Spot price must be strictly positive. "
            "A stock price of zero or negative implies bankruptcy or invalid instrument.",
            field="S0",
            value=S0,
            constraint="S0 > 0",
        )
        
    if not (0.01 <= S0_f <= 1_000_000):
        raise ValidationError(
            f"Spot price {S0_f} seems unreasonable. "
            "Typical range is $0.01 to $1,000,000. "
            "Check your units (cents vs dollars) or data source.",
            field="S0",
            value=S0,
            constraint="0.01 <= S0 <= 1,000,000 (warning range)",
        )
    
    # Strike validation
    if K_f < 0:
        raise ValidationError(
            "Strike price cannot be negative. "
            "A negative strike would imply the option holder pays to exercise.",
            field="K",
            value=K,
            constraint="K >= 0",
        )
        
    if K_f == 0 and not allow_zero_strike:
        raise ValidationError(
            "Strike price cannot be zero for standard options. "
            "A zero-strike option is essentially a forward contract on the spot. "
            "Set allow_zero_strike=True if this is intentional.",
            field="K",
            value=K,
            constraint="K > 0 (or K >= 0 with allow_zero_strike=True)",
        )
        
    if K_f > 10 * S0_f:
        raise ValidationError(
            f"Strike {K_f} is more than 10x spot {S0_f}. "
            "This is extremely deep OTM/ITM and may cause numerical issues. "
            "Consider if you meant to use a different strike or if units are correct.",
            field="K",
            value=K,
            constraint="K not extreme vs S0",
        )
    
    # Time validation
    if T_f < 0:
        raise ValidationError(
            "Time to maturity cannot be negative. "
            "For expired options, use T=0 to get intrinsic value.",
            field="T",
            value=T,
            constraint="T >= 0",
        )
        
    if T_f > 100:
        raise ValidationError(
            f"Maturity {T_f} years seems excessive. "
            "Typical equity options expire within 2-3 years. "
            "LEAPS max out around 3 years. "
            "Verify your time unit (years vs days).",
            field="T",
            value=T,
            constraint="T <= 100 (warning range)",
        )
    
    # Interest rate validation
    if not (-0.5 <= r_f <= 0.5):
        raise ValidationError(
            f"Interest rate {r_f} is outside reasonable bounds [-50%, +50%]. "
            f"Current global rates typically range from -1% to +15%. "
            f"Extreme rates may indicate data error or hyperinflation scenario.",
            field="r",
            value=r,
            constraint="-0.5 <= r <= 0.5 (-50% to +50%)",
        )
    
    # Volatility validation
    if sigma_f < 0:
        raise ValidationError(
            "Volatility cannot be negative. "
            "Volatility is the standard deviation of log-returns and must be non-negative. "
            "For deterministic assets (zero vol), use sigma=0.",
            field="sigma",
            value=sigma,
            constraint="sigma >= 0",
        )
        
    if sigma_f > 5.0:
        raise ValidationError(
            f"Volatility {sigma_f} ({sigma_f*100:.0f}%) is extremely high. "
            f"VIX all-time high was ~80%. "
            f"Volatilities above 500% suggest possible data error "
            f"(e.g., variance vs std dev confusion).",
            field="sigma",
            value=sigma,
            constraint="sigma <= 5.0 (500%) (warning range)",
        )
    
    # Dividend yield validation
    if q_f < 0:
        raise ValidationError(
            "Dividend yield cannot be negative. "
            "Negative yields imply the holder pays dividends to own the stock, "
            "which is not standard. For cost of carry models, adjust the rate instead.",
            field="q",
            value=q,
            constraint="q >= 0",
        )
        
    if q_f > r_f + 0.5:
        raise ValidationError(
            f"Dividend yield {q_f} exceeds risk-free rate {r_f} by >50%. "
            f"This implies strong negative carry. "
            f"Verify if this is a high-dividend stock or data error.",
            field="q",
            value=q,
            constraint=f"q <= r + 0.5 ({r_f + 0.5})",
        )
    
    # Monte Carlo paths validation
    if n_paths < 1000:
        raise ValidationError(
            f"Number of paths {n_paths} is too few for reliable Monte Carlo. "
            f"Standard error scales as 1/sqrt(N). With {n_paths} paths, "
            f"std error is ~{1/n_paths**0.5*100:.1f}% of option value. "
            f"Use at least 10,000 for production, 1,000 minimum for testing.",
            field="n_paths",
            value=n_paths,
            constraint="n_paths >= 1000",
        )
        
    if n_paths > 10_000_000:
        raise ValidationError(
            f"Number of paths {n_paths} is extremely large. "
            f"This may cause memory issues or excessive computation time. "
            f"Consider variance reduction techniques instead of brute force.",
            field="n_paths",
            value=n_paths,
            constraint="n_paths <= 10,000,000 (warning range)",
        )
    
    return OptionParameters(
        S0=S0_f,
        K=K_f,
        T=T_f,
        r=r_f,
        sigma=sigma_f,
        n_paths=n_paths,
        q=q_f,
    )


def validate_heston_params(
    S0: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    theta: float,
    kappa: float,
    sigma: float,
    rho: float,
    n_paths: int,
) -> HestonParameters:
    """Validate Heston model parameters including Feller condition.
    
    Args:
        S0: Spot price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        v0: Initial variance (not volatility!)
        theta: Long-term variance
        kappa: Mean reversion speed
        sigma: Volatility of variance (vol of vol)
        rho: Correlation between price and variance
        n_paths: Number of MC paths
        
    Returns:
        Validated HestonParameters
        
    Raises:
        ValidationError: If any parameter invalid or Feller condition violated
    """
    # First validate base option parameters
    validate_option_params(S0, K, T, r, v0**0.5, n_paths)  # v0**0.5 as vol proxy
    
    # Heston-specific validations
    if v0 <= 0:
        raise ValidationError(
            "Initial variance v0 must be strictly positive. "
            "Note: this is variance (sigma^2), not volatility. "
            f"If you meant vol={v0}, use v0={v0**2}.",
            field="v0",
            value=v0,
            constraint="v0 > 0",
        )
        
    if theta <= 0:
        raise ValidationError(
            "Long-term variance theta must be strictly positive. "
            "This is the mean-reversion level of the variance process.",
            field="theta",
            value=theta,
            constraint="theta > 0",
        )
        
    if kappa <= 0:
        raise ValidationError(
            "Mean reversion speed kappa must be strictly positive. "
            "Zero or negative kappa means variance doesn't revert to mean, "
            "causing non-stationary behavior.",
            field="kappa",
            value=kappa,
            constraint="kappa > 0",
        )
        
    if sigma <= 0:
        raise ValidationError(
            "Vol of vol (sigma) must be strictly positive. "
            "Zero would make the variance process deterministic.",
            field="sigma",
            value=sigma,
            constraint="sigma > 0",
        )
        
    if not (-1 <= rho <= 1):
        raise ValidationError(
            f"Correlation rho must be in [-1, 1], got {rho}. "
            f"This is the correlation between the Brownian motions "
            f"driving price and variance.",
            field="rho",
            value=rho,
            constraint="-1 <= rho <= 1",
        )
    
    params = HestonParameters(
        S0=float(S0),
        K=float(K),
        T=float(T),
        r=float(r),
        v0=float(v0),
        theta=float(theta),
        kappa=float(kappa),
        sigma=float(sigma),
        rho=float(rho),
        n_paths=int(n_paths),
    )
    
    # Check Feller condition
    params.check_feller_condition()
    
    return params


def validate_inputs(
    func: Callable[Concatenate[Any, P], T]
) -> Callable[Concatenate[Any, P], T]:
    """Decorator to automatically validate pricing function inputs.
    
    Assumes the decorated function has signature:
        method(self, S0, K, T, r, sigma, n_paths=..., ...)
    
    The decorator extracts these parameters from the method call and
    validates them before execution.
    
    Example:
        class MyPricer:
            @validate_inputs
            def price(self, S0, K, T, r, sigma, n_paths=10000):
                # S0, K, etc. are guaranteed valid here
                return calculate_price(...)
    """
    @wraps(func)
    def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> T:
        # Try to extract standard parameters from args/kwargs
        # This is a heuristic based on common naming conventions
        
        param_names = ['S0', 'K', 'T', 'r', 'sigma', 'n_paths', 'q']
        params: dict[str, Any] = {}
        
        # Map positional args
        for i, arg in enumerate(args):
            if i < len(param_names):
                params[param_names[i]] = arg
                
        # Override with keyword args
        params.update(kwargs)
        
        # Validate if we have the minimum required params
        required = ['S0', 'K', 'T', 'r', 'sigma']
        if all(p in params for p in required):
            try:
                # Check if this is Heston model by looking for v0
                if 'v0' in params or 'theta' in params:
                    # Heston validation
                    validate_heston_params(
                        params.get('S0', 100),
                        params.get('K', 100),
                        params.get('T', 1.0),
                        params.get('r', 0.05),
                        params.get('v0', 0.04),
                        params.get('theta', 0.04),
                        params.get('kappa', 2.0),
                        params.get('sigma', 0.3),
                        params.get('rho', -0.7),
                        params.get('n_paths', 10000),
                    )
                else:
                    # Standard Black-Scholes validation
                    validate_option_params(
                        params.get('S0', 100),
                        params.get('K', 100),
                        params.get('T', 1.0),
                        params.get('r', 0.05),
                        params.get('sigma', 0.2),
                        params.get('n_paths', 10000),
                        params.get('q', 0.0),
                    )
            except ValidationError:
                raise
            except Exception as e:
                # If validation itself fails, just proceed and let original error surface
                pass
                
        return func(self, *args, **kwargs)
    return wrapper


def validate_method_inputs(method: str = "price") -> Callable:
    """Factory for creating validation decorators with specific method names.
    
    More flexible than @validate_inputs - specify which method signature to expect.
    
    Args:
        method: One of "price", "heston", "black_scholes"
        
    Returns:
        Decorator function
    """
    def decorator(
        func: Callable[Concatenate[Any, P], T]
    ) -> Callable[Concatenate[Any, P], T]:
        @wraps(func)
        def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> T:
            if method == "heston":
                # Expect Heston parameters
                sig = func.__code__.co_varnames[:func.__code__.co_argcount]
                bound = dict(zip(sig[1:], args))  # Skip 'self'
                bound.update(kwargs)
                
                required = ['S0', 'K', 'T', 'r', 'v0', 'theta', 'kappa', 'sigma_v', 'rho']
                if all(r in bound for r in required):
                    validate_heston_params(
                        bound['S0'], bound['K'], bound['T'], bound['r'],
                        bound['v0'], bound['theta'], bound['kappa'],
                        bound['sigma_v'], bound['rho'],
                        bound.get('n_paths', 10000)
                    )
            else:
                # Standard validation via decorator
                pass  # Let @validate_inputs handle it
                
            return func(self, *args, **kwargs)
        return wrapper
    return decorator