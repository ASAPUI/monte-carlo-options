"""
Market Data Integration and Parameter Calibration
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from scipy.optimize import minimize

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


class MarketData:
    """Fetch and process real market data for option pricing."""
    
    @staticmethod
    def get_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance not installed. Run: pip install yfinance")
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        return df
    
    @staticmethod
    def calculate_historical_volatility(prices: pd.Series, window: int = 252, annualize: bool = True) -> float:
        log_returns = np.log(prices / prices.shift(1))
        vol = log_returns.std() * np.sqrt(252) if annualize else log_returns.std()
        return vol
    
    @staticmethod
    def ewma_volatility(returns: pd.Series, lambda_param: float = 0.94, annualize: bool = True) -> pd.Series:
        squared_returns = returns ** 2
        ewma_var = squared_returns.ewm(alpha=(1 - lambda_param), adjust=False).mean()
        ewma_vol = np.sqrt(ewma_var)
        if annualize:
            ewma_vol = ewma_vol * np.sqrt(252)
        return ewma_vol
    
    @staticmethod
    def get_risk_free_rate(maturity_years: float = 1.0) -> float:
        rates = {0.25: 0.045, 0.5: 0.045, 1.0: 0.046, 2.0: 0.048, 5.0: 0.050, 10.0: 0.052}
        maturities = sorted(rates.keys())
        if maturity_years <= maturities[0]:
            return rates[maturities[0]]
        elif maturity_years >= maturities[-1]:
            return rates[maturities[-1]]
        else:
            for i in range(len(maturities) - 1):
                if maturities[i] <= maturity_years <= maturities[i+1]:
                    t1, t2 = maturities[i], maturities[i+1]
                    r1, r2 = rates[t1], rates[t2]
                    return r1 + (r2 - r1) * (maturity_years - t1) / (t2 - t1)
        return 0.05
    
    @staticmethod
    def get_option_chain(ticker: str) -> pd.DataFrame:
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance not installed")
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return pd.DataFrame()
        opt = stock.option_chain(expirations[0])
        calls = opt.calls.copy()
        puts = opt.puts.copy()
        calls['option_type'] = 'call'
        puts['option_type'] = 'put'
        return pd.concat([calls, puts], ignore_index=True)


class VolatilityCalibrator:
    """Calibrate volatility models to market data."""
    
    def __init__(self):
        pass
    
    def calibrate_gbm_volatility(self, stock_prices: pd.Series, method: str = 'historical') -> Dict:
        if method == 'historical':
            vol = MarketData.calculate_historical_volatility(stock_prices)
            return {'sigma': vol, 'method': 'historical'}
        elif method == 'ewma':
            returns = np.log(stock_prices / stock_prices.shift(1)).dropna()
            ewma_vol = MarketData.ewma_volatility(returns)
            return {'sigma': ewma_vol.iloc[-1], 'sigma_series': ewma_vol, 'method': 'ewma'}
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def calibrate_heston(self, market_vols: np.ndarray, strikes: np.ndarray, spot: float, 
                         T: float, r: float) -> Dict:
        x0 = [0.04, 2.0, 0.04, 0.3, -0.7]
        bounds = [(0.001, 0.5), (0.1, 10.0), (0.001, 0.5), (0.01, 1.0), (-0.99, 0.99)]
        
        def objective(params):
            return np.sum((market_vols - 0.2) ** 2)
        
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        
        return {
            'v0': result.x[0], 'kappa': result.x[1], 'theta': result.x[2],
            'xi': result.x[3], 'rho': result.x[4], 'success': result.success
        }
    
    def build_volatility_surface(self, ticker: str, moneyness_range: np.ndarray = None,
                                  maturity_range: np.ndarray = None) -> pd.DataFrame:
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance not installed")
        stock = yf.Ticker(ticker)
        spot = stock.history(period="1d")['Close'].iloc[-1]
        surface_data = []
        
        for expiry in stock.options[:6]:
            opt_chain = stock.option_chain(expiry)
            calls = opt_chain.calls
            expiry_date = pd.to_datetime(expiry)
            T = (expiry_date - pd.Timestamp.now()).days / 365.25
            
            for _, row in calls.iterrows():
                K = row['strike']
                market_price = (row['bid'] + row['ask']) / 2
                if market_price > 0:
                    from .black_scholes import BlackScholes
                    try:
                        iv = BlackScholes.implied_volatility(market_price, spot, K, T,
                                                              MarketData.get_risk_free_rate(T), 'call')
                        moneyness = K / spot
                        surface_data.append({
                            'maturity': T, 'moneyness': moneyness, 'strike': K,
                            'implied_vol': iv, 'price': market_price
                        })
                    except:
                        continue
        return pd.DataFrame(surface_data)


class ParameterEstimator:
    """Estimate drift, volatility, and other parameters from historical data."""
    
    @staticmethod
    def estimate_drift(prices: pd.Series, risk_free_rate: Optional[float] = None) -> float:
        log_returns = np.log(prices / prices.shift(1)).dropna()
        mu = log_returns.mean() * 252
        return mu
    
    @staticmethod
    def estimate_jump_parameters(returns: pd.Series, threshold: float = 3.0) -> Dict:
        sigma = returns.std()
        jumps = returns[np.abs(returns) > threshold * sigma]
        if len(jumps) == 0:
            return {'lambda': 0.0, 'mu_j': 0.0, 'delta': 0.0}
        lambda_param = len(jumps) / len(returns) * 252
        mu_j = jumps.mean()
        delta = jumps.std()
        return {'lambda': lambda_param, 'mu_j': mu_j, 'delta': delta, 'n_jumps': len(jumps)}