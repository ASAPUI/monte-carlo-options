"""
Streamlit Web Interface for Monte Carlo Options Pricing
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Import the actual classes from the modules
from src.pricing.european import EuropeanOptionPricer
from src.pricing.american import LongstaffSchwartzPricer
from src.models.black_scholes import BlackScholesModel
from src.pricing.greeks import GreeksCalculator
from src.simulator import GBMSimulator

# Page configuration
st.set_page_config(
    page_title="Monte Carlo Options Pricing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def plot_paths(paths, T, title="Simulated GBM Paths"):
    """Helper function to plot price paths."""
    fig, ax = plt.subplots(figsize=(10, 5))
    time_grid = np.linspace(0, T, paths.shape[1])
    
    # Plot subset of paths
    n_plot = min(50, paths.shape[0])
    for i in range(n_plot):
        ax.plot(time_grid, paths[i], alpha=0.6, linewidth=0.8)
    
    ax.set_xlabel('Time (years)', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    return fig

def main():
    st.markdown('<div class="main-header">📊 Monte Carlo Options Pricing</div>', 
                unsafe_allow_html=True)
    
    st.sidebar.header("⚙️ Market Parameters")
    
    # Market parameters
    S0 = st.sidebar.number_input("Spot Price ($)", min_value=1.0, value=100.0, step=1.0)
    K = st.sidebar.number_input("Strike Price ($)", min_value=1.0, value=100.0, step=1.0)
    T = st.sidebar.slider("Time to Maturity (years)", 0.1, 5.0, 1.0, 0.1)
    r = st.sidebar.slider("Risk-free Rate (%)", 0.0, 20.0, 5.0, 0.5) / 100
    sigma = st.sidebar.slider("Volatility (%)", 1.0, 100.0, 20.0, 1.0) / 100
    q = st.sidebar.slider("Dividend Yield (%)", 0.0, 10.0, 0.0, 0.1) / 100
    
    st.sidebar.header("🔧 Simulation Settings")
    n_sims = st.sidebar.select_slider(
        "Number of Simulations", 
        options=[1000, 5000, 10000, 25000, 50000, 100000], 
        value=50000
    )
    n_steps = st.sidebar.slider("Time Steps", 10, 200, 100, 10)
    seed = st.sidebar.number_input("Random Seed", min_value=0, value=42, step=1)
    use_antithetic = st.sidebar.checkbox("Use Antithetic Variates", value=True)
    use_numba = st.sidebar.checkbox("Use Numba Acceleration", value=True)
    
    st.sidebar.header("📋 Option Type")
    option_type = st.sidebar.selectbox(
        "Select Option", 
        ["European Call", "European Put", "American Call", "American Put"]
    )
    
    calculate = st.sidebar.button("🚀 Calculate Price", type="primary", use_container_width=True)
    
    if calculate:
        with st.spinner("Running Monte Carlo simulation..."):
            try:
                # Initialize simulator for path generation
                sim = GBMSimulator(seed=int(seed))
                
                if "European" in option_type:
                    # European Option Pricing
                    pricer = EuropeanOptionPricer(S0, K, T, r, sigma, q)
                    opt_type = "call" if "Call" in option_type else "put"
                    
                    # Use pricer's built-in Monte Carlo
                    price, stderr = pricer.price(
                        option_type=opt_type,
                        n_paths=n_sims,
                        n_steps=n_steps,
                        antithetic=use_antithetic,
                        seed=int(seed)
                    )
                    
                    # Black-Scholes benchmark
                    bs_model = BlackScholesModel(S0, K, T, r, sigma, q)
                    bs_price = bs_model.price(opt_type)
                    
                    # Greeks calculation
                    greeks_calc = GreeksCalculator(S0, K, T, r, sigma, q)
                    delta = greeks_calc.delta_call() if opt_type == "call" else greeks_calc.delta_put()
                    gamma = greeks_calc.gamma()
                    vega = greeks_calc.vega()
                    theta = greeks_calc.theta_call() if opt_type == "call" else greeks_calc.theta_put()
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    col1.metric("💰 MC Price", f"${price:.4f}")
                    col2.metric("📊 Black-Scholes", f"${bs_price:.4f}")
                    col3.metric("📉 Std Error", f"±{stderr:.4f}")
                    
                    # Display Greeks
                    st.subheader("📈 Greeks")
                    gcol1, gcol2, gcol3, gcol4 = st.columns(4)
                    gcol1.metric("Delta", f"{delta:.4f}")
                    gcol2.metric("Gamma", f"{gamma:.4f}")
                    gcol3.metric("Vega", f"{vega:.4f}")
                    gcol4.metric("Theta (daily)", f"{theta:.4f}")
                    
                    # Generate paths using simulator for visualization
                    paths = sim.generate_paths(
                        S0=S0, 
                        r=r - q,  # Adjust for dividend yield in GBM
                        sigma=sigma, 
                        T=T, 
                        n_steps=n_steps, 
                        n_sims=min(1000, n_sims),  # Fewer paths for viz
                        antithetic=use_antithetic,
                        use_numba=use_numba
                    )
                    
                else:  # American Option
                    # American Option Pricing
                    pricer = LongstaffSchwartzPricer(S0, K, T, r, sigma, q)
                    opt_type = "call" if "Call" in option_type else "put"
                    
                    price, stderr = pricer.price(
                        option_type=opt_type,
                        n_paths=n_sims,
                        n_steps=n_steps,
                        seed=int(seed)
                    )
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    col1.metric("💰 American Option Price", f"${price:.4f}")
                    col2.metric("📉 Std Error", f"±{stderr:.4f}")
                    
                    # For American options, also show European price for comparison
                    bs_model = BlackScholesModel(S0, K, T, r, sigma, q)
                    european_price = bs_model.price(opt_type)
                    early_premium = price - european_price
                    col3.metric("⏰ Early Exercise Premium", f"${early_premium:.4f}", 
                               delta_color="normal" if early_premium > 0 else "off")
                    
                    # Generate paths using simulator for visualization
                    paths = sim.generate_paths(
                        S0=S0, 
                        r=r - q,  # Adjust for dividend yield in GBM
                        sigma=sigma, 
                        T=T, 
                        n_steps=n_steps, 
                        n_sims=min(1000, n_sims),
                        antithetic=use_antithetic,
                        use_numba=use_numba
                    )
                
                # Visualization
                st.subheader("📉 Simulated Price Paths")
                fig = plot_paths(paths, T, f"Geometric Brownian Motion Paths ({n_sims:,} simulations)")
                st.pyplot(fig)
                
                # Price distribution at maturity
                st.subheader("📊 Price Distribution at Maturity")
                ST = paths[:, -1]
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                ax2.hist(ST, bins=50, density=True, alpha=0.7, color='#1f77b4', edgecolor='black')
                ax2.axvline(K, color='red', linestyle='--', linewidth=2, label=f'Strike K=${K:.2f}')
                ax2.axvline(S0, color='green', linestyle='--', linewidth=2, label=f'Spot S0=${S0:.2f}')
                ax2.set_xlabel('Price at Maturity ($)', fontsize=12)
                ax2.set_ylabel('Probability Density', fontsize=12)
                ax2.set_title('Distribution of Terminal Stock Prices', fontsize=14, fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)
                
                # Additional: Show payoff diagram
                st.subheader("💰 Option Payoff Diagram")
                fig3, ax3 = plt.subplots(figsize=(10, 5))
                
                # Generate range of terminal prices for payoff
                S_range = np.linspace(max(0.5 * S0, 1), 1.5 * S0, 100)
                
                if "Call" in option_type:
                    payoff = np.maximum(S_range - K, 0)
                    payoff_label = "Call Payoff: max(S_T - K, 0)"
                else:
                    payoff = np.maximum(K - S_range, 0)
                    payoff_label = "Put Payoff: max(K - S_T, 0)"
                
                ax3.plot(S_range, payoff, 'b-', linewidth=2, label=payoff_label)
                ax3.axvline(K, color='red', linestyle='--', alpha=0.7, label=f'Strike K=${K}')
                ax3.axvline(S0, color='green', linestyle='--', alpha=0.7, label=f'Spot S0=${S0}')
                ax3.fill_between(S_range, 0, payoff, alpha=0.3)
                ax3.set_xlabel('Terminal Stock Price ($)', fontsize=12)
                ax3.set_ylabel('Payoff ($)', fontsize=12)
                ax3.set_title('Option Payoff at Maturity', fontsize=14, fontweight='bold')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                st.pyplot(fig3)
                
            except Exception as e:
                st.error(f"❌ Error during calculation: {str(e)}")
                st.exception(e)
    else:
        # Welcome message
        st.info("👈 **Configure your option parameters in the sidebar and click Calculate to begin.**")
        
        st.markdown("""
        ### 🎯 Features
        
        This application provides **Monte Carlo option pricing** with the following capabilities:
        
        **Option Types:**
        - ✅ European Call/Put (Black-Scholes benchmark + Greeks)
        - ✅ American Call/Put (Longstaff-Schwartz Least Squares Monte Carlo)
        
        **Models:**
        - Geometric Brownian Motion (GBM) for price simulation
        - Black-Scholes closed-form solution for European options
        - Longstaff-Schwartz algorithm for American early exercise
        
        **Greeks Calculation (European only):**
        - Delta, Gamma, Vega, Theta
        - Finite difference and pathwise methods
        
        **Simulation Features:**
        - **GBMSimulator** with Numba JIT acceleration (10-100x faster)
        - Antithetic variates for variance reduction
        - Adjustable simulation paths and time steps
        - Reproducible random seeds
        - Support for continuous dividend yields
        - Optimized terminal value generation
        
        **Visualizations:**
        - Simulated price paths
        - Terminal price distribution
        - Option payoff diagrams
        """)

if __name__ == "__main__":
    main()