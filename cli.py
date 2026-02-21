#!/usr/bin/env python3
"""
Command Line Interface for Monte Carlo Options Pricing

Usage:
    python cli.py european --type call --spot 100 --strike 100 --time 1.0 --rate 0.05 --vol 0.2
    python cli.py american --type put --spot 100 --strike 110 --time 1.0 --rate 0.05 --vol 0.2
    python cli.py greeks --spot 100 --strike 100 --time 1.0 --rate 0.05 --vol 0.2 --type call
"""

import argparse
import sys
import json
import time

import numpy as np

from src.simulator import GBMSimulator
from src.black_scholes import BlackScholes
from src.option_pricer import EuropeanPricer, AmericanPricer, ExoticPricer
from src.greeks import GreeksCalculator


def format_currency(value):
    return f"${value:.4f}"

def format_percent(value):
    return f"{value*100:.2f}%"

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_result(label, value, width=20):
    print(f"  {label:<{width}}: {value}")

def european_command(args):
    print_header("EUROPEAN OPTION PRICING")
    print("\n📊 Parameters:")
    print_result("Spot Price", format_currency(args.spot))
    print_result("Strike Price", format_currency(args.strike))
    print_result("Time to Maturity", f"{args.time} years")
    print_result("Risk-free Rate", format_percent(args.rate))
    print_result("Volatility", format_percent(args.vol))
    print_result("Simulations", f"{args.sims:,}")
    
    start_time = time.time()
    simulator = GBMSimulator(seed=args.seed)
    pricer = EuropeanPricer(simulator)
    
    if args.type.lower() == 'call':
        result = pricer.price_call(args.spot, args.strike, args.time, args.rate, args.vol,
                                   n_sims=args.sims, antithetic=not args.no_antithetic, 
                                   control_variate=not args.no_control)
        bs_price = BlackScholes.call_price(args.spot, args.strike, args.time, args.rate, args.vol)
    else:
        result = pricer.price_put(args.spot, args.strike, args.time, args.rate, args.vol,
                                  n_sims=args.sims, antithetic=not args.no_antithetic,
                                  control_variate=not args.no_control)
        bs_price = BlackScholes.put_price(args.spot, args.strike, args.time, args.rate, args.vol)
    
    elapsed = time.time() - start_time
    
    print("\n💰 Results:")
    print_result("Monte Carlo Price", format_currency(result['price']))
    print_result("Black-Scholes Price", format_currency(bs_price))
    print_result("Standard Error", f"±{result['std_error']:.4f}")
    print_result("95% Confidence Interval", 
                f"[{result['confidence_interval'][0]:.4f}, {result['confidence_interval'][1]:.4f}]")
    print_result("Error vs BS", format_currency(abs(result['price'] - bs_price)))
    print_result("Relative Error", format_percent(abs(result['price'] - bs_price) / bs_price))
    
    if 'variance_reduction' in result:
        print_result("Variance Reduction", format_percent(result['variance_reduction']))
    
    print(f"\n⏱️  Computation Time: {elapsed:.3f} seconds")
    
    if args.json:
        output = {
            'option_type': f'european_{args.type}',
            'parameters': vars(args),
            'results': {
                'mc_price': result['price'],
                'bs_price': bs_price,
                'std_error': result['std_error'],
                'confidence_interval': list(result['confidence_interval']),
                'error': abs(result['price'] - bs_price),
                'computation_time': elapsed
            }
        }
        print("\n📄 JSON Output:")
        print(json.dumps(output, indent=2))

def american_command(args):
    print_header("AMERICAN OPTION PRICING (Longstaff-Schwartz)")
    print("\n📊 Parameters:")
    print_result("Spot Price", format_currency(args.spot))
    print_result("Strike Price", format_currency(args.strike))
    print_result("Time to Maturity", f"{args.time} years")
    print_result("Risk-free Rate", format_percent(args.rate))
    print_result("Volatility", format_percent(args.vol))
    print_result("Simulations", f"{args.sims:,}")
    print_result("Time Steps", args.steps)
    
    start_time = time.time()
    simulator = GBMSimulator(seed=args.seed)
    pricer = AmericanPricer(simulator)
    
    if args.type.lower() == 'put':
        result = pricer.price_put(args.spot, args.strike, args.time, args.rate, args.vol,
                                  n_sims=args.sims, n_steps=args.steps)
        european_pricer = EuropeanPricer(simulator)
        european_price = european_pricer.price_put(args.spot, args.strike, args.time, 
                                                   args.rate, args.vol, args.sims)['price']
    else:
        result = pricer.price_call(args.spot, args.strike, args.time, args.rate, args.vol,
                                   n_sims=args.sims, n_steps=args.steps)
        european_pricer = EuropeanPricer(simulator)
        european_price = european_pricer.price_call(args.spot, args.strike, args.time,
                                                    args.rate, args.vol, args.sims)['price']
    
    elapsed = time.time() - start_time
    
    print("\n💰 Results:")
    print_result("American Price", format_currency(result['price']))
    print_result("European Price", format_currency(european_price))
    print_result("Early Exercise Premium", format_currency(result['price'] - european_price))
    print_result("Standard Error", f"±{result['std_error']:.4f}")
    print_result("95% Confidence Interval", 
                f"[{result['confidence_interval'][0]:.4f}, {result['confidence_interval'][1]:.4f}]")
    
    print(f"\n⏱️  Computation Time: {elapsed:.3f} seconds")
    
    if args.json:
        output = {
            'option_type': f'american_{args.type}',
            'parameters': vars(args),
            'results': {
                'american_price': result['price'],
                'european_price': european_price,
                'early_exercise_premium': result['price'] - european_price,
                'std_error': result['std_error'],
                'computation_time': elapsed
            }
        }
        print("\n📄 JSON Output:")
        print(json.dumps(output, indent=2))

def exotic_command(args):
    print_header(f"EXOTIC OPTION PRICING - {args.type.upper()}")
    print("\n📊 Parameters:")
    print_result("Spot Price", format_currency(args.spot))
    print_result("Strike Price", format_currency(args.strike))
    print_result("Time to Maturity", f"{args.time} years")
    print_result("Risk-free Rate", format_percent(args.rate))
    print_result("Volatility", format_percent(args.vol))
    print_result("Simulations", f"{args.sims:,}")
    
    if args.barrier:
        print_result("Barrier Level", format_currency(args.barrier))
    
    start_time = time.time()
    simulator = GBMSimulator(seed=args.seed)
    pricer = ExoticPricer(simulator)
    exotic_type = args.type.lower()
    
    if exotic_type == 'asian':
        result = pricer.asian_call(args.spot, args.strike, args.time, args.rate, args.vol, args.sims)
    elif exotic_type == 'asian_put':
        result = pricer.asian_put(args.spot, args.strike, args.time, args.rate, args.vol, args.sims)
    elif exotic_type == 'barrier':
        if not args.barrier:
            print("❌ Error: Barrier level required (--barrier)")
            sys.exit(1)
        result = pricer.barrier_call_up_out(args.spot, args.strike, args.barrier, args.time, 
                                            args.rate, args.vol, args.sims)
    elif exotic_type == 'lookback':
        result = pricer.lookback_call_fixed_strike(args.spot, args.strike, args.time, 
                                                   args.rate, args.vol, args.sims)
    elif exotic_type == 'digital':
        result = pricer.digital_call(args.spot, args.strike, args.time, args.rate, args.vol, 
                                     payout=args.payout, n_sims=args.sims)
    else:
        print(f"❌ Error: Unknown exotic type '{args.type}'")
        sys.exit(1)
    
    elapsed = time.time() - start_time
    
    print("\n💰 Results:")
    print_result("Option Price", format_currency(result['price']))
    print_result("Standard Error", f"±{result['std_error']:.4f}")
    
    if 'knock_out_probability' in result:
        print_result("Knock-out Probability", format_percent(result['knock_out_probability']))
    
    if 'theoretical' in result:
        print_result("Theoretical Price", format_currency(result['theoretical']))
    
    print(f"\n⏱️  Computation Time: {elapsed:.3f} seconds")

def greeks_command(args):
    print_header("GREEKS CALCULATION")
    print("\n📊 Parameters:")
    print_result("Spot Price", format_currency(args.spot))
    print_result("Strike Price", format_currency(args.strike))
    print_result("Time to Maturity", f"{args.time} years")
    print_result("Risk-free Rate", format_percent(args.rate))
    print_result("Volatility", format_percent(args.vol))
    print_result("Option Type", args.type)
    print_result("Method", args.method)
    
    start_time = time.time()
    calc = GreeksCalculator(method=args.method)
    greeks = calc.calculate_european_greeks(args.spot, args.strike, args.time, args.rate, args.vol,
                                            option_type=args.type, n_sims=args.sims)
    elapsed = time.time() - start_time
    
    print("\n📈 Greeks:")
    print_result("Price", format_currency(greeks['price']))
    print_result("Delta (Δ)", f"{greeks['delta']:.4f}")
    print_result("Gamma (Γ)", f"{greeks['gamma']:.4f}")
    print_result("Vega (V)", f"{greeks['vega']:.4f}")
    print_result("Theta (Θ)", f"{greeks['theta']:.4f}")
    print_result("Rho (ρ)", f"{greeks['rho']:.4f}")
    
    if args.type == 'call':
        true_delta = BlackScholes.call_delta(args.spot, args.strike, args.time, args.rate, args.vol)
        true_gamma = BlackScholes.gamma(args.spot, args.strike, args.time, args.rate, args.vol)
        true_vega = BlackScholes.vega(args.spot, args.strike, args.time, args.rate, args.vol)
    else:
        true_delta = BlackScholes.put_delta(args.spot, args.strike, args.time, args.rate, args.vol)
        true_gamma = BlackScholes.gamma(args.spot, args.strike, args.time, args.rate, args.vol)
        true_vega = BlackScholes.vega(args.spot, args.strike, args.time, args.rate, args.vol)
    
    print("\n✅ Analytical Comparison:")
    print_result("Delta Error", f"{abs(greeks['delta'] - true_delta):.4f}")
    print_result("Gamma Error", f"{abs(greeks['gamma'] - true_gamma):.4f}")
    print_result("Vega Error", f"{abs(greeks['vega'] - true_vega):.4f}")
    
    print(f"\n⏱️  Computation Time: {elapsed:.3f} seconds")

def main():
    parser = argparse.ArgumentParser(description='Monte Carlo Options Pricing CLI',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="""
Examples:
  python cli.py european --type call --spot 100 --strike 100 --time 1.0 --rate 0.05 --vol 0.2
  python cli.py american --type put --spot 100 --strike 110 --sims 100000 --steps 100
  python cli.py exotic --type asian --spot 100 --strike 100 --time 1.0 --rate 0.05 --vol 0.2
  python cli.py greeks --spot 100 --strike 100 --time 1.0 --rate 0.05 --vol 0.2 --type call
        """)
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument('--spot', type=float, required=True, help='Spot price')
    common_args.add_argument('--strike', type=float, required=True, help='Strike price')
    common_args.add_argument('--time', type=float, required=True, help='Time to maturity (years)')
    common_args.add_argument('--rate', type=float, required=True, help='Risk-free rate (decimal)')
    common_args.add_argument('--vol', type=float, required=True, help='Volatility (decimal)')
    common_args.add_argument('--sims', type=int, default=50000, help='Number of simulations')
    common_args.add_argument('--seed', type=int, default=42, help='Random seed')
    common_args.add_argument('--json', action='store_true', help='Output JSON format')
    
    euro_parser = subparsers.add_parser('european', parents=[common_args], 
                                        help='Price European options')
    euro_parser.add_argument('--type', choices=['call', 'put'], required=True, 
                            help='Option type')
    euro_parser.add_argument('--no-antithetic', action='store_true',
                            help='Disable antithetic variates')
    euro_parser.add_argument('--no-control', action='store_true',
                            help='Disable control variates')
    
    american_parser = subparsers.add_parser('american', parents=[common_args],
                                           help='Price American options')
    american_parser.add_argument('--type', choices=['call', 'put'], required=True,
                                help='Option type')
    american_parser.add_argument('--steps', type=int, default=50,
                                help='Number of time steps')
    
    exotic_parser = subparsers.add_parser('exotic', parents=[common_args],
                                         help='Price exotic options')
    exotic_parser.add_argument('--type', required=True,
                              choices=['asian', 'asian_put', 'barrier', 'lookback', 'digital'],
                              help='Exotic option type')
    exotic_parser.add_argument('--barrier', type=float, help='Barrier level')
    exotic_parser.add_argument('--payout', type=float, default=1.0, help='Digital payout')
    
    greeks_parser = subparsers.add_parser('greeks', parents=[common_args],
                                         help='Calculate option Greeks')
    greeks_parser.add_argument('--type', choices=['call', 'put'], required=True,
                              help='Option type')
    greeks_parser.add_argument('--method', choices=['finite_difference', 'pathwise'],
                              default='finite_difference', help='Calculation method')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'european':
            european_command(args)
        elif args.command == 'american':
            american_command(args)
        elif args.command == 'exotic':
            exotic_command(args)
        elif args.command == 'greeks':
            greeks_command(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()