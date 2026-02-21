import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from simulator import GBMSimulator
from option_pricer import EuropeanPricer, AmericanPricer

sim = GBMSimulator(seed=42)
eu_pricer = EuropeanPricer(sim)
am_pricer = AmericanPricer(sim)

S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

# Compare prices at different strikes
print("Strike | European Call | European Put | American Put")
print("-" * 55)
for strike in [80, 90, 100, 110, 120]:
    eu_call = eu_pricer.price_call(S0, strike, T, r, sigma, 50000)['price']
    eu_put = eu_pricer.price_put(S0, strike, T, r, sigma, 50000)['price']
    am_put = am_pricer.price_put(S0, strike, T, r, sigma, 50000, 50)['price']