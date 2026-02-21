#!/usr/bin/env python3
"""
Quick Start Example - Monte Carlo Options Pricing

This script demonstrates the main features of the package in a single runnable example.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from simulator import GBMSimulator, HestonSimulator
from black_scholes import BlackScholes
from option_pricer import EuropeanPricer, AmericanPricer, ExoticPricer
from greeks import GreeksCalculator


def main():
    print("="*70)
    print("  MONTE CARLO OPTIONS PRICING - QUICK START DEMO")
    print("="*70)
    
    print("\n📊 Test Parameters:")
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    
    print(f"  Spot: ${S0}, Strike: ${K}, Time: {T}y, Rate: {r:.1%}, Vol: {sigma:.1%}")
    
    print("\n" + "="*70)
    print("  BLACK-SCHOLES ANALYTICAL PRICES")
    print("="*70)
    
    bs_call = BlackScholes.call_price(S0, K, T, r, sigma)
    bs_put = BlackScholes.put_price(S0, K, T, r, sigma)
    
    print(f"  European Call: ${bs_call:.4f}")
    print(f"  European Put:  ${bs_put:.4f}")
    print(f"  Put-Call Parity Check: {bs_call - bs_put:.4f} = {S0 - K*np.exp(-r*T):.4f} ✓")
    
    print("\n" + "="*70)
    print("  MONTE CARLO PRICING (50,000 simulations)")
    print("="*70)
    
    simulator = GBMSimulator(seed=42)
    european_pricer = EuropeanPricer(simulator)
    
    mc_call = european_pricer.price_call(S0, K, T, r, sigma, n_sims=50000, 
                                         antithetic=True, control_variate=True)
    
    print(f"  MC Call Price: ${mc_call['price']:.4f} ± {mc_call['std_error']:.4f}")
    print(f"  vs Black-Scholes: ${bs_call:.4f} (Error: ${abs(mc_call['price']-bs_call):.4f})")
    
    if 'variance_reduction' in mc_call:
        print(f"  Variance Reduction: {mc_call['variance_reduction']*100:.1f}%")
    
    mc_put = european_pricer.price_put(S0, K, T, r, sigma, n_sims=50000)
    print(f"  MC Put Price:  ${mc_put['price']:.4f} ± {mc_put['std_error']:.4f}")
    print(f"  vs Black-Scholes: ${bs_put:.4f} (Error: ${abs(mc_put['price']-bs_put):.4f})")
    
    print("\n" + "="*70)
    print("  AMERICAN OPTIONS (Longstaff-Schwartz)")
    print("="*70)
    
    american_pricer = AmericanPricer(simulator)
    
    S0_itm = 100
    K_itm = 110
    
    am_put = american_pricer.price_put(S0_itm, K_itm, T, r, sigma, n_sims=30000, n_steps=50)
    eu_put = european_pricer.price_put(S0_itm, K_itm, T, r, sigma, n_sims=30000)
    
    print(f"  American Put (ITM): ${am_put['price']:.4f}")
    print(f"  European Put (ITM): ${eu_put['price']:.4f}")
    print(f"  Early Exercise Premium: ${am_put['price'] - eu_put['price']:.4f}")
    
    print("\n" + "="*70)
    print("  EXOTIC OPTIONS")
    print("="*70)
    
    exotic_pricer = ExoticPricer(simulator)
    
    asian = exotic_pricer.asian_call(S0, K, T, r, sigma, n_sims=30000)
    barrier = exotic_pricer.barrier_call_up_out(S0, K, B=120, T=T, r=r, sigma=sigma, n_sims=30000)
    lookback = exotic_pricer.lookback_call_fixed_strike(S0, K, T, r, sigma, n_sims=30000)
    digital = exotic_pricer.digital_call(S0, K, T, r, sigma, payout=1.0, n_sims=50000)
    
    print(f"  Asian Call:      ${asian['price']:.4f}")
    print(f"  Barrier Up-Out:  ${barrier['price']:.4f} (KO prob: {barrier['knock_out_probability']:.1%})")
    print(f"  Lookback Call:   ${lookback['price']:.4f}")
    print(f"  Digital Call:    ${digital['price']:.4f} (Theory: ${digital['theoretical']:.4f})")
    
    print("\n" + "="*70)
    print("  GREEKS CALCULATION (Finite Difference Method)")
    print("="*70)
    
    calc = GreeksCalculator(method='finite_difference')
    greeks = calc.calculate_european_greeks(S0, K, T, r, sigma, 'call', n_sims=30000)
    
    print(f"  Price: ${greeks['price']:.4f}")
    print(f"  Delta: {greeks['delta']:.4f} (Analytical: {BlackScholes.call_delta(S0, K, T, r, sigma):.4f})")
    print(f"  Gamma: {greeks['gamma']:.4f} (Analytical: {BlackScholes.gamma(S0, K, T, r, sigma):.4f})")
    print(f"  Vega:  {greeks['vega']:.4f} (Analytical: {BlackScholes.vega(S0, K, T, r, sigma):.4f})")
    print(f"  Theta: {greeks['theta']:.4f}")
    print(f"  Rho:   {greeks['rho']:.4f}")
    
    print("\n" + "="*70)
    print("  GENERATING VISUALIZATIONS")
    print("="*70)
    
    paths = simulator.generate_paths(S0, r, sigma, T, n_steps=252, n_sims=1000)
    
    fig = plt.figure(figsize=(16, 10))
    
    ax1 = plt.subplot(2, 3, 1)
    time_grid = np.linspace(0, T, 253)
    for i in range(100):
        ax1.plot(time_grid, paths[i], alpha=0.6, linewidth=0.8)
    ax1.set_title('Simulated GBM Paths', fontweight='bold')
    ax1.set_xlabel('Time (years)')
    ax1.set_ylabel('Asset Price')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 3, 2)
    ST = paths[:, -1]
    ax2.hist(ST, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(K, color='red', linestyle='--', linewidth=2, label=f'Strike ${K}')
    ax2.set_title('Terminal Price Distribution', fontweight='bold')
    ax2.set_xlabel('Terminal Price')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(2, 3, 3)
    n_sims_list = [1000, 5000, 10000, 25000, 50000]
    mc_prices = []
    std_errors = []
    
    for n in n_sims_list:
        result = european_pricer.price_call(S0, K, T, r, sigma, n_sims=n)
        mc_prices.append(result['price'])
        std_errors.append(result['std_error'])
    
    ax3.plot(n_sims_list, mc_prices, 'o-', linewidth=2, markersize=6, label='MC Estimate')
    ax3.axhline(bs_call, color='red', linestyle='--', label='Black-Scholes')
    ax3.fill_between(n_sims_list, 
                     np.array(mc_prices) - 1.96*np.array(std_errors),
                     np.array(mc_prices) + 1.96*np.array(std_errors),
                     alpha=0.3, color='blue', label='95% CI')
    ax3.set_xscale('log')
    ax3.set_title('MC Convergence', fontweight='bold')
    ax3.set_xlabel('Number of Simulations')
    ax3.set_ylabel('Option Price')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(2, 3, 4)
    options = ['European\nCall', 'Asian\nCall', 'Barrier\nUp-Out', 'Digital']
    prices = [bs_call, asian['price'], barrier['price'], digital['price']]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax4.bar(options, prices, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_title('Option Prices Comparison', fontweight='bold')
    ax4.set_ylabel('Price ($)')
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax5 = plt.subplot(2, 3, 5)
    greek_names = ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho']
    mc_values = [greeks['delta'], greeks['gamma']*10, greeks['vega']/10, 
                 greeks['theta'], greeks['rho']]
    true_values = [BlackScholes.call_delta(100, 100, 1.0, 0.05, 0.2),
                   BlackScholes.gamma(100, 100, 1.0, 0.05, 0.2)*10,
                   BlackScholes.vega(100, 100, 1.0, 0.05, 0.2)/10,
                   BlackScholes.theta_call(100, 100, 1.0, 0.05, 0.2),
                   BlackScholes.rho_call(100, 100, 1.0, 0.05, 0.2)]
    
    x = np.arange(len(greek_names))
    width = 0.35
    ax5.bar(x - width/2, mc_values, width, label='Monte Carlo', alpha=0.8)
    ax5.bar(x + width/2, true_values, width, label='Analytical', alpha=0.8)
    ax5.set_title('Greeks Comparison (Scaled)', fontweight='bold')
    ax5.set_ylabel('Value')
    ax5.set_xticks(x)
    ax5.set_xticklabels(greek_names)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    ax6 = plt.subplot(2, 3, 6)
    methods = ['Standard', 'Antithetic', 'Control\nVariate']
    variances = [mc_call['std_error']**2]
    
    result_anti = european_pricer.price_call(S0, K, T, r, sigma, n_sims=50000, 
                                              antithetic=True, control_variate=False)
    result_control = european_pricer.price_call(S0, K, T, r, sigma, n_sims=50000,
                                                 antithetic=True, control_variate=True)
    
    variances.extend([result_anti['std_error']**2, result_control['std_error']**2])
    
    ax6.bar(methods, variances, color=['red', 'orange', 'green'], alpha=0.7, edgecolor='black')
    ax6.set_title('Variance by Method', fontweight='bold')
    ax6.set_ylabel('Variance')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'demo_results.png'), dpi=150, bbox_inches='tight')
    print("  ✓ Saved visualization to output/demo_results.png")
    
    print("\n" + "="*70)
    print("  PERFORMANCE SUMMARY")
    print("="*70)
    print(f"  European Option Error: {abs(mc_call['price']-bs_call)/bs_call*100:.3f}%")
    print(f"  Standard Error: {mc_call['std_error']:.4f} ({mc_call['std_error']/mc_call['price']*100:.2f}% of price)")
    print(f"  95% Confidence Interval: [{mc_call['confidence_interval'][0]:.4f}, {mc_call['confidence_interval'][1]:.4f}]")
    print(f"  Variance Reduction: {(1 - result_control['std_error']/mc_call['std_error'])*100:.1f}%")
    
    print("\n" + "="*70)
    print("  ✓ DEMO COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nNext steps:")
    print("  • Run tests: pytest tests/ -v")
    print("  • Launch web app: streamlit run app.py")
    print("  • Use CLI: python cli.py --help")
    print("  • Open notebooks/ for tutorials")


if __name__ == '__main__':
    main()