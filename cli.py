#!/usr/bin/env python3
"""
Command Line Interface for Monte Carlo Options Pricing
(hnta txouf)

Usage:
    python cli.py european --type call --spot 100 --strike 100 --time 1.0 --rate 0.05 --vol 0.2
    python cli.py american --type put --spot 100 --strike 110 --time 1.0 --rate 0.05 --vol 0.2
    python cli.py greeks --spot 100 --strike 100 --time 1.0 --rate 0.05 --vol 0.2 --type call
"""

import argparse
import sys
import json
import time
import os

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.simulator import GBMSimulator
from src.models.black_scholes import BlackScholesModel
from src.pricing.european import EuropeanOptionPricer
from src.pricing.american import LongstaffSchwartzPricer
from src.pricing.greeks import GreeksCalculator


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
    print_result("Dividend Yield", format_percent(args.div_yield))
    print_result("Simulations", f"{args.sims:,}")

    start_time = time.time()

    pricer = EuropeanOptionPricer(
        S0=args.spot,
        K=args.strike,
        T=args.time,
        r=args.rate,
        sigma=args.vol,
        q=args.div_yield
    )

    price, stderr = pricer.price(
        option_type=args.type,
        n_paths=args.sims,
        n_steps=args.steps,
        antithetic=not args.no_antithetic,
        seed=args.seed
    )

    bs_model = BlackScholesModel(
        S=args.spot,
        K=args.strike,
        T=args.time,
        r=args.rate,
        sigma=args.vol,
        q=args.div_yield
    )
    bs_price = bs_model.price(args.type)

    ci_lower = price - 1.96 * stderr
    ci_upper = price + 1.96 * stderr
    elapsed = time.time() - start_time

    print("\n💰 Results:")
    print_result("Monte Carlo Price", format_currency(price))
    print_result("Black-Scholes Price", format_currency(bs_price))
    print_result("Standard Error", f"±{stderr:.4f}")
    print_result("95% Confidence Interval", f"[{ci_lower:.4f}, {ci_upper:.4f}]")
    print_result("Error vs BS", format_currency(abs(price - bs_price)))
    if bs_price != 0:
        print_result("Relative Error", format_percent(abs(price - bs_price) / bs_price))
    print(f"\n⏱️  Computation Time: {elapsed:.3f} seconds")

    if args.json:
        output = {
            'option_type': f'european_{args.type}',
            'parameters': vars(args),
            'results': {
                'mc_price': price,
                'bs_price': bs_price,
                'std_error': stderr,
                'confidence_interval': [ci_lower, ci_upper],
                'error': abs(price - bs_price),
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
    print_result("Dividend Yield", format_percent(args.div_yield))
    print_result("Simulations", f"{args.sims:,}")
    print_result("Time Steps", args.steps)
    print_result("Basis Degree", args.basis_degree)

    start_time = time.time()

    pricer = LongstaffSchwartzPricer(
        S0=args.spot,
        K=args.strike,
        T=args.time,
        r=args.rate,
        sigma=args.vol,
        q=args.div_yield,
        basis_degree=args.basis_degree
    )

    price, stderr = pricer.price(
        option_type=args.type,
        n_paths=args.sims,
        n_steps=args.steps,
        seed=args.seed
    )

    european_pricer = EuropeanOptionPricer(
        S0=args.spot,
        K=args.strike,
        T=args.time,
        r=args.rate,
        sigma=args.vol,
        q=args.div_yield
    )
    european_price, _ = european_pricer.price(
        option_type=args.type,
        n_paths=args.sims,
        n_steps=args.steps,
        seed=args.seed
    )

    elapsed = time.time() - start_time
    ci_lower = price - 1.96 * stderr
    ci_upper = price + 1.96 * stderr

    print("\n💰 Results:")
    print_result("American Price", format_currency(price))
    print_result("European Price", format_currency(european_price))
    print_result("Early Exercise Premium", format_currency(price - european_price))
    print_result("Standard Error", f"±{stderr:.4f}")
    print_result("95% Confidence Interval", f"[{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"\n⏱️  Computation Time: {elapsed:.3f} seconds")

    if args.json:
        output = {
            'option_type': f'american_{args.type}',
            'parameters': vars(args),
            'results': {
                'american_price': price,
                'european_price': european_price,
                'early_exercise_premium': price - european_price,
                'std_error': stderr,
                'confidence_interval': [ci_lower, ci_upper],
                'computation_time': elapsed
            }
        }
        print("\n📄 JSON Output:")
        print(json.dumps(output, indent=2))


def greeks_command(args):
    print_header("GREEKS CALCULATION")
    print("\n📊 Parameters:")
    print_result("Spot Price", format_currency(args.spot))
    print_result("Strike Price", format_currency(args.strike))
    print_result("Time to Maturity", f"{args.time} years")
    print_result("Risk-free Rate", format_percent(args.rate))
    print_result("Volatility", format_percent(args.vol))
    print_result("Dividend Yield", format_percent(args.div_yield))
    print_result("Option Type", args.type)
    print_result("Calculation Method", args.method)
    print_result("Simulations", f"{args.sims:,}")

    start_time = time.time()

    calc = GreeksCalculator(
        S0=args.spot,
        K=args.strike,
        T=args.time,
        r=args.rate,
        sigma=args.vol,
        q=args.div_yield
    )

    # 'finite_difference' -> 'central', 'pathwise' -> 'pathwise'
    method = 'pathwise' if args.method == 'pathwise' else 'central'
    if args.type == 'call':
        delta = calc.delta_call(method=method, n_paths=args.sims)
    else:
        delta = calc.delta_put(method=method, n_paths=args.sims)

    gamma = calc.gamma(n_paths=args.sims)
    vega = calc.vega(n_paths=args.sims)
    theta = calc.theta_call(n_paths=args.sims) if args.type == 'call' else calc.theta_put(n_paths=args.sims)

    bs_model = BlackScholesModel(args.spot, args.strike, args.time, args.rate, args.vol, args.div_yield)
    price = bs_model.price(args.type)
    bs_delta = bs_model.delta(args.type)
    bs_gamma = bs_model.gamma()
    bs_vega = bs_model.vega()

    elapsed = time.time() - start_time

    print("\n📈 Greeks:")
    print_result("Price (BS)", format_currency(price))
    print_result("Delta (Δ)", f"{delta:.4f}")
    print_result("Gamma (Γ)", f"{gamma:.4f}")
    print_result("Vega (V)", f"{vega:.4f}")
    print_result("Theta (Θ)", f"{theta:.4f}")

    print("\n✅ Analytical (Black-Scholes) Comparison:")
    print_result("BS Delta", f"{bs_delta:.4f}")
    print_result("Delta Error", f"{abs(delta - bs_delta):.4f}")
    print_result("BS Gamma", f"{bs_gamma:.4f}")
    print_result("Gamma Error", f"{abs(gamma - bs_gamma):.4f}")
    print_result("BS Vega", f"{bs_vega:.4f}")
    print_result("Vega Error", f"{abs(vega - bs_vega):.4f}")

    print(f"\n⏱️  Computation Time: {elapsed:.3f} seconds")

    if args.json:
        output = {
            'option_type': args.type,
            'method': args.method,
            'parameters': vars(args),
            'results': {
                'price': price,
                'delta': delta,
                'gamma': gamma,
                'vega': vega,
                'theta': theta,
                'analytical_delta': bs_delta,
                'analytical_gamma': bs_gamma,
                'analytical_vega': bs_vega,
                'computation_time': elapsed
            }
        }
        print("\n📄 JSON Output:")
        print(json.dumps(output, indent=2))


def simulate_command(args):
    print_header("GBM PATH SIMULATION")
    print("\n📊 Parameters:")
    print_result("Spot Price", format_currency(args.spot))
    print_result("Drift (r - q)", format_percent(args.rate - args.div_yield))
    print_result("Volatility", format_percent(args.vol))
    print_result("Time to Maturity", f"{args.time} years")
    print_result("Time Steps", args.steps)
    print_result("Simulations", f"{args.sims:,}")
    print_result("Use Numba", str(not args.no_numba))
    print_result("Antithetic", str(not args.no_antithetic))

    start_time = time.time()

    sim = GBMSimulator(seed=args.seed)

    paths = sim.generate_paths(
        S0=args.spot,
        r=args.rate - args.div_yield,
        sigma=args.vol,
        T=args.time,
        n_steps=args.steps,
        n_sims=args.sims,
        antithetic=not args.no_antithetic,
        use_numba=(not args.no_numba and args.seed is not None)
    )

    elapsed = time.time() - start_time

    terminal_prices = paths[:, -1]
    theoretical_mean = args.spot * np.exp((args.rate - args.div_yield) * args.time)

    print("\n📈 Simulation Results:")
    print_result("Computation Time", f"{elapsed:.3f} seconds")
    print_result("Mean Terminal Price", format_currency(float(np.mean(terminal_prices))))
    print_result("Std Dev Terminal", format_currency(float(np.std(terminal_prices))))
    print_result("Min Price", format_currency(float(np.min(terminal_prices))))
    print_result("Max Price", format_currency(float(np.max(terminal_prices))))
    print_result("Theoretical Mean", format_currency(theoretical_mean))
    print_result("Mean Error", format_currency(float(np.mean(terminal_prices)) - theoretical_mean))


def main():
    parser = argparse.ArgumentParser(
        description='Monte Carlo Options Pricing CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py european --type call --spot 100 --strike 100 --time 1.0 --rate 0.05 --vol 0.2
  python cli.py american --type put --spot 100 --strike 110 --sims 100000 --steps 100
  python cli.py greeks --spot 100 --strike 100 --time 1.0 --rate 0.05 --vol 0.2 --type call
  python cli.py simulate --spot 100 --time 1.0 --rate 0.05 --vol 0.2 --sims 10000 --steps 252
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument('--spot', type=float, required=True, help='Spot price')
    common_args.add_argument('--time', type=float, required=True, help='Time to maturity (years)')
    common_args.add_argument('--rate', type=float, required=True, help='Risk-free rate (decimal)')
    common_args.add_argument('--vol', type=float, required=True, help='Volatility (decimal)')
    common_args.add_argument('--div-yield', type=float, default=0.0, dest='div_yield',
                             help='Dividend yield (decimal)')
    common_args.add_argument('--seed', type=int, default=42, help='Random seed')

    euro_parser = subparsers.add_parser('european', parents=[common_args],
                                        help='Price European options')
    euro_parser.add_argument('--strike', type=float, required=True, help='Strike price')
    euro_parser.add_argument('--type', choices=['call', 'put'], required=True,
                             help='Option type')
    euro_parser.add_argument('--sims', type=int, default=50000, help='Number of simulations')
    euro_parser.add_argument('--steps', type=int, default=100, help='Number of time steps')
    euro_parser.add_argument('--no-antithetic', action='store_true',
                             help='Disable antithetic variates')
    euro_parser.add_argument('--json', action='store_true', help='Output JSON format')

    american_parser = subparsers.add_parser('american', parents=[common_args],
                                            help='Price American options')
    american_parser.add_argument('--strike', type=float, required=True, help='Strike price')
    american_parser.add_argument('--type', choices=['call', 'put'], required=True,
                                 help='Option type')
    american_parser.add_argument('--sims', type=int, default=50000, help='Number of simulations')
    american_parser.add_argument('--steps', type=int, default=100, help='Number of time steps')
    american_parser.add_argument('--basis-degree', type=int, default=3, dest='basis_degree',
                                 help='Polynomial basis degree for LSM')
    american_parser.add_argument('--json', action='store_true', help='Output JSON format')

    greeks_parser = subparsers.add_parser('greeks', parents=[common_args],
                                          help='Calculate option Greeks')
    greeks_parser.add_argument('--strike', type=float, required=True, help='Strike price')
    greeks_parser.add_argument('--type', choices=['call', 'put'], required=True,
                               help='Option type')
    greeks_parser.add_argument('--sims', type=int, default=20000, help='Number of simulations')
    greeks_parser.add_argument('--method', choices=['finite_difference', 'pathwise'],
                               default='finite_difference', help='Calculation method')
    greeks_parser.add_argument('--json', action='store_true', help='Output JSON format')

    sim_parser = subparsers.add_parser('simulate', parents=[common_args],
                                       help='Simulate GBM paths')
    sim_parser.add_argument('--sims', type=int, default=1000, help='Number of paths')
    sim_parser.add_argument('--steps', type=int, default=100, help='Number of time steps')
    sim_parser.add_argument('--no-numba', action='store_true', dest='no_numba',
                            help='Disable Numba acceleration')
    sim_parser.add_argument('--no-antithetic', action='store_true',
                            help='Disable antithetic variates')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == 'european':
            european_command(args)
        elif args.command == 'american':
            american_command(args)
        elif args.command == 'greeks':
            greeks_command(args)
        elif args.command == 'simulate':
            simulate_command(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()