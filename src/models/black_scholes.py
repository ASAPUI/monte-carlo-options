"""Black-Scholes closed-form analytical solutions."""

from typing import Literal, Tuple

import numpy as np
from scipy.stats import norm


class BlackScholesModel:
    """Black-Scholes analytical option pricing model.
    
    Provides closed-form solutions for European options and their Greeks.
    
    Attributes:
        S: Spot price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        sigma: Volatility
        q: Dividend yield
    """
    
    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> None:
        """Initialize Black-Scholes model.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield
        """
        self.S: float = S
        self.K: float = K
        self.T: float = T
        self.r: float = r
        self.sigma: float = sigma
        self.q: float = q
        
    def _d1(self) -> float:
        """Calculate d1 term in Black-Scholes formula."""
        return (np.log(self.S / self.K) + 
                (self.r - self.q + 0.5 * self.sigma ** 2) * self.T) / \
               (self.sigma * np.sqrt(self.T))
               
    def _d2(self) -> float:
        """Calculate d2 term in Black-Scholes formula."""
        return self._d1() - self.sigma * np.sqrt(self.T)
        
    def price(
        self,
        option_type: Literal["call", "put"] = "call",
    ) -> float:
        """Calculate option price.
        
        Args:
            option_type: "call" or "put"
            
        Returns:
            Option price
        """
        if self.T <= 0:
            if option_type == "call":
                return max(self.S - self.K, 0.0)
            return max(self.K - self.S, 0.0)
            
        d1 = self._d1()
        d2 = self._d2()
        
        if option_type == "call":
            price = (self.S * np.exp(-self.q * self.T) * norm.cdf(d1) -
                    self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
        else:
            price = (self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) -
                    self.S * np.exp(-self.q * self.T) * norm.cdf(-d1))
            
        return float(price)
        
    def delta(self, option_type: Literal["call", "put"] = "call") -> float:
        """Calculate Delta.
        
        Args:
            option_type: "call" or "put"
            
        Returns:
            Delta value
        """
        if self.T <= 0:
            if option_type == "call":
                return 1.0 if self.S > self.K else 0.0
            return -1.0 if self.S < self.K else 0.0
            
        d1 = self._d1()
        
        if option_type == "call":
            return float(np.exp(-self.q * self.T) * norm.cdf(d1))
        return float(np.exp(-self.q * self.T) * (norm.cdf(d1) - 1))
        
    def gamma(self) -> float:
        """Calculate Gamma (same for calls and puts).
        
        Returns:
            Gamma value
        """
        if self.T <= 0:
            return 0.0
            
        d1 = self._d1()
        return float(np.exp(-self.q * self.T) * norm.pdf(d1) / 
                    (self.S * self.sigma * np.sqrt(self.T)))
                    
    def vega(self) -> float:
        """Calculate Vega (same for calls and puts).
        
        Returns:
            Vega value (per 1% change in volatility)
        """
        if self.T <= 0:
            return 0.0
            
        d1 = self._d1()
        # Raw vega (for 1% change, multiply by 0.01)
        return float(self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * np.sqrt(self.T))
        
    def theta(
        self,
        option_type: Literal["call", "put"] = "call",
    ) -> float:
        """Calculate Theta (time decay).
        
        Args:
            option_type: "call" or "put"
            
        Returns:
            Theta value (daily, divide by 365 for annual)
        """
        if self.T <= 0:
            return 0.0
            
        d1 = self._d1()
        d2 = self._d2()
        
        term1 = (-self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * self.sigma / 
                (2 * np.sqrt(self.T)))
        term2 = self.q * self.S * np.exp(-self.q * self.T)
        term3 = self.r * self.K * np.exp(-self.r * self.T)
        
        if option_type == "call":
            theta = term1 - term2 * norm.cdf(d1) + term3 * norm.cdf(d2)
        else:
            theta = term1 + term2 * norm.cdf(-d1) - term3 * norm.cdf(-d2)
            
        return float(theta) / 365.0  # Convert to daily
        
    def rho(self, option_type: Literal["call", "put"] = "call") -> float:
        """Calculate Rho (sensitivity to interest rate).
        
        Args:
            option_type: "call" or "put"
            
        Returns:
            Rho value (per 1% change in rates)
        """
        if self.T <= 0:
            return 0.0
            
        d2 = self._d2()
        
        if option_type == "call":
            return float(self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) * 0.01)
        return float(-self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2) * 0.01)
        
    def implied_volatility(
        self,
        market_price: float,
        option_type: Literal["call", "put"] = "call",
        tol: float = 1e-5,
        max_iter: int = 100,
    ) -> float:
        """Calculate implied volatility using Newton-Raphson method.
        
        Args:
            market_price: Observed market price
            option_type: "call" or "put"
            tol: Convergence tolerance
            max_iter: Maximum iterations
            
        Returns:
            Implied volatility
            
        Raises:
            ValueError: If method fails to converge
        """
        # Initial guess
        sigma = 0.2
        
        for _ in range(max_iter):
            self.sigma = sigma
            price = self.price(option_type)
            vega = self.vega()
            
            if vega == 0:
                raise ValueError("Vega is zero, cannot calculate implied vol")
                
            diff = price - market_price
            
            if abs(diff) < tol:
                return float(sigma)
                
            sigma = sigma - diff / (vega * 100)  # Adjust for vega scaling
            
            if sigma <= 0:
                sigma = 0.001
                
        raise ValueError(f"Failed to converge after {max_iter} iterations")