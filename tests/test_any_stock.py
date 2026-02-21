import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import yfinance as yf
from simulator import GBMSimulator
from option_pricer import EuropeanPricer
import numpy as np

# CHANGE THIS LINE TO ANY STOCK YOU WANT
STOCK_SYMBOL = "TSLA"  # <-- CHANGE HERE!

print(f"Pricing options for: {STOCK_SYMBOL}")

ticker = yf.Ticker(STOCK_SYMBOL)
hist = ticker.history(period="6mo")

if hist.empty:
    print("❌ No data found")
else:
    price = hist['Close'].iloc[-1]
    returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
    vol = returns.std() * np.sqrt(252)
    
    print(f"Current Price: ${price:.2f}")
    print(f"Volatility: {vol:.2%}")
    
    # Price option
    sim = GBMSimulator(seed=42)
    pricer = EuropeanPricer(sim)
    
    call = pricer.price_call(price, price, 0.25, 0.05, vol, n_sims=100000)
    
    print(f"\nATM Call Option (3 months): ${call['price']:.4f}")