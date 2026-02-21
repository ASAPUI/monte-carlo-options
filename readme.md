# Monte Carlo Options Pricing System

A comprehensive, production-ready options pricing engine using Monte Carlo simulation methods.

## 🚀 Features

### Core Pricing Models
- **European Options** (Call/Put) with analytical Black-Scholes validation
- **American Options** using Longstaff-Schwartz Least Squares Monte Carlo
- **Exotic Options**: Asian, Barrier, Lookback, Binary/Digital

### Advanced Capabilities
- **Greeks Calculation**: Delta, Gamma, Vega, Theta, Rho using pathwise and finite difference methods
- **Variance Reduction**: Antithetic Variates, Control Variates, Importance Sampling
- **Multi-Asset Pricing**: Basket options, Best-of/Worst-of with correlated Brownian motion
- **Stochastic Models**: Heston stochastic volatility, Merton jump-diffusion, CEV model

### Production Features
- High-performance Numba JIT compilation
- Parallel processing with multiprocessing
- Real market data integration (yfinance)
- Implied volatility calculation
- Portfolio risk management (VaR, scenario analysis)
- Interactive Streamlit web interface

## 📦 Installation

```bash
git clone &lt;repository-url&gt;
cd monte_carlo_options
pip install -r requirements.txt