"""
Streamlit Web Interface for Monte Carlo Options Pricing
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Fix: Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import directly from modules
import simulator
import black_scholes
import option_pricer
import greeks

# Get classes
GBMSimulator = simulator.GBMSimulator
BlackScholes = black_scholes.BlackScholes
EuropeanPricer = option_pricer.EuropeanPricer
AmericanPricer = option_pricer.AmericanPricer
ExoticPricer = option_pricer.ExoticPricer
GreeksCalculator = greeks.GreeksCalculator

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
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<div class="main-header">📊 Monte Carlo Options Pricing</div>', 
                unsafe_allow_html=True)
    
    st.sidebar.header("⚙️ Parameters")
    
    # Inputs
    S0 = st.sidebar.number_input("Spot Price ($)", min_value=1.0, value=100.0)
    K = st.sidebar.number_input("Strike Price ($)", min_value=1.0, value=100.0)
    T = st.sidebar.slider("Time to Maturity (years)", 0.1, 5.0, 1.0)
    r = st.sidebar.slider("Risk-free Rate (%)", 0.0, 20.0, 5.0) / 100
    sigma = st.sidebar.slider("Volatility (%)", 1.0, 100.0, 20.0) / 100
    n_sims = st.sidebar.select_slider("Simulations", [1000, 5000, 10000, 50000, 100000], 50000)
    
    option_type = st.sidebar.selectbox("Option Type", ["European Call", "European Put", "American Put"])
    
    calculate = st.sidebar.button("🚀 Calculate", type="primary")
    
    if calculate:
        with st.spinner("Calculating..."):
            sim = GBMSimulator(seed=42)
            
            if option_type == "European Call":
                pricer = EuropeanPricer(sim)
                result = pricer.price_call(S0, K, T, r, sigma, n_sims)
                bs = BlackScholes.call_price(S0, K, T, r, sigma)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("MC Price", f"${result['price']:.4f}")
                col2.metric("Black-Scholes", f"${bs:.4f}")
                col3.metric("Std Error", f"±{result['std_error']:.4f}")
                
            elif option_type == "European Put":
                pricer = EuropeanPricer(sim)
                result = pricer.price_put(S0, K, T, r, sigma, n_sims)
                bs = BlackScholes.put_price(S0, K, T, r, sigma)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("MC Price", f"${result['price']:.4f}")
                col2.metric("Black-Scholes", f"${bs:.4f}")
                col3.metric("Std Error", f"±{result['std_error']:.4f}")
                
            else:  # American Put
                pricer = AmericanPricer(sim)
                result = pricer.price_put(S0, K, T, r, sigma, n_sims, n_steps=50)
                
                col1, col2 = st.columns(2)
                col1.metric("American Put", f"${result['price']:.4f}")
                col2.metric("Std Error", f"±{result['std_error']:.4f}")
            
            # Simple visualization
            st.subheader("Price Paths")
            paths = sim.generate_paths(S0, r, sigma, T, n_steps=100, n_sims=100)
            
            fig, ax = plt.subplots()
            time_grid = np.linspace(0, T, 101)
            for i in range(50):
                ax.plot(time_grid, paths[i], alpha=0.6, linewidth=0.8)
            ax.set_xlabel('Time')
            ax.set_ylabel('Price')
            ax.set_title('Simulated GBM Paths')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    else:
        st.info("👈 Set parameters and click Calculate")

if __name__ == "__main__":
    main()