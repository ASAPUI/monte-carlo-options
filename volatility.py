import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib.pyplot as plt
from simulator import GBMSimulator
from option_pricer import EuropeanPricer

sim = GBMSimulator(seed=42)
pricer = EuropeanPricer(sim)

S0, T, r, sigma = 100, 1.0, 0.05, 0.2
strikes = np.arange(80, 121, 5)
prices = []

for K in strikes:
    result = pricer.price_call(S0, K, T, r, sigma, 50000)
    prices.append(result['price'])

plt.figure(figsize=(10, 6))
plt.plot(strikes, prices, 'o-', linewidth=2, markersize=8)
plt.axvline(S0, color='red', linestyle='--', label=f'Spot = ${S0}')
plt.xlabel('Strike Price')
plt.ylabel('Option Price')
plt.title('Option Price vs Strike (Volatility Smile)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('volatility_smile.png')
print("✅ Saved to volatility_smile.png")
plt.show()