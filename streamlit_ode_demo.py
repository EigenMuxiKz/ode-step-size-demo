import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="ODE Step Size Analysis Demo",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2c3e50;
        padding: 1rem 0;
        border-bottom: 3px solid #3498db;
        margin-bottom: 2rem;
    }
    
    .equation-box {
        background: #2c3e50;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-family: 'Courier New', monospace;
        font-size: 1.1em;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .danger-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_analytical_solution(k=0.5, C0=4, t_max=10):
    """Generate analytical solution C(t) = C0 * exp(-k*t)"""
    t = np.linspace(0, t_max, 1000)
    C_analytical = C0 * np.exp(-k * t)
    return t, C_analytical

def generate_explicit_euler(h, k=0.5, C0=4, t_max=10):
    """Generate numerical solution using Explicit Euler method"""
    n_steps = int(t_max / h) + 1
    t = np.linspace(0, t_max, n_steps)
    C = np.zeros(n_steps)
    C[0] = C0
    
    for i in range(n_steps - 1):
        C[i + 1] = C[i] + h * (-k * C[i])
        # Safety check for extreme instability
        if abs(C[i + 1]) > 1000:
            C[i + 1:] = np.nan
            break
    
    return t, C

def generate_implicit_euler(h, k=0.5, C0=4, t_max=10):
    """Generate numerical solution using Implicit Euler method"""
    n_steps = int(t_max / h) + 1
    t = np.linspace(0, t_max, n_steps)
    C = np.zeros(n_steps)
    C[0] = C0
    
    for i in range(n_steps - 1):
        C[i + 1] = C[i] / (1 + h * k)
    
    return t, C

def main():
    # Title and introduction
    st.markdown('<div class="main-header"><h1>📊 ODE Numerical Methods Demo</h1><h3>Analysis 1: The Effect of Step Size (h)</h3></div>', unsafe_allow_html=True)
    
    # Equation display
    st.markdown("""
    <div class="equation-box">
        dC/dt = -kC, where k = 0.5, C(0) = 4<br>
        Analytical Solution: C(t) = 4 × exp(-0.5t)
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("🎛️ Control Parameters")
    
    # Step size slider
    h = st.sidebar.slider(
        "Step Size (h)", 
        min_value=0.1, 
        max_value=5.0, 
        value=0.5, 
        step=0.1,
        help="Adjust the step size to see its effect on accuracy and stability"
    )
    
    # Method selection
    method = st.sidebar.radio(
        "Select Numerical Method",
        ["Explicit Euler", "Implicit Euler"],
        help="Choose between explicit and implicit methods"
    )
    
    # Show both methods option
    show_both = st.sidebar.checkbox(
        "Compare Both Methods",
        value=False,
        help="Display both methods on the same plot"
    )
    
    # Parameters
    k = 0.5
    C0 = 4
    t_max = 10
    critical_h = 2 / k  # Critical step size for explicit method
    
    # Generate solutions
    t_analytical, C_analytical = generate_analytical_solution(k, C0, t_max)
    t_explicit, C_explicit = generate_explicit_euler(h, k, C0, t_max)
    t_implicit, C_implicit = generate_implicit_euler(h, k, C0, t_max)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot analytical solution
    ax.plot(t_analytical, C_analytical, 'k-', linewidth=3, label='Analytical Solution', alpha=0.8)
    
    # Plot numerical solutions
    if show_both:
        ax.plot(t_explicit, C_explicit, 'ro-', markersize=6, linewidth=2, 
                label=f'Explicit Euler (h={h})', alpha=0.7, markerfacecolor='red')
        ax.plot(t_implicit, C_implicit, 'gs-', markersize=6, linewidth=2, 
                label=f'Implicit Euler (h={h})', alpha=0.7, markerfacecolor='green')
    else:
        if method == "Explicit Euler":
            ax.plot(t_explicit, C_explicit, 'ro-', markersize=6, linewidth=2, 
                    label=f'Explicit Euler (h={h})', alpha=0.7, markerfacecolor='red')
        else:
            ax.plot(t_implicit, C_implicit, 'gs-', markersize=6, linewidth=2, 
                    label=f'Implicit Euler (h={h})', alpha=0.7, markerfacecolor='green')
    
    # Customize plot
    ax.set_xlabel('Time (t)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Concentration C(t)', fontsize=14, fontweight='bold')
    ax.set_title(f'Analytical vs Numerical Solution (h = {h})', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='upper right')
    ax.set_xlim(0, t_max)
    
    # Set y-axis limits to handle instability
    y_min = max(-10, min(0, np.nanmin(C_explicit) if not np.all(np.isnan(C_explicit)) else 0))
    y_max = min(10, max(C0, np.nanmax(C_explicit) if not np.all(np.isnan(C_explicit)) else C0))
    ax.set_ylim(y_min, y_max)
    
    # Display the plot
    st.pyplot(fig)
    
    # Analysis section
    st.markdown("## 📊 Real-time Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Accuracy Analysis")
        
        # Calculate errors
        C_analytical_final = C0 * np.exp(-k * t_max)
        C_explicit_final = C_explicit[-1] if not np.isnan(C_explicit[-1]) else float('inf')
        C_implicit_final = C_implicit[-1]
        
        error_explicit = abs(C_explicit_final - C_analytical_final)
        error_implicit = abs(C_implicit_final - C_analytical_final)
        
        # Display accuracy info
        if method == "Explicit Euler":
            if error_explicit < 0.01:
                st.markdown(f"""
                <div class="success-box">
                    <strong>✅ Excellent Accuracy!</strong><br>
                    Final Value: {C_explicit_final:.4f}<br>
                    Analytical: {C_analytical_final:.4f}<br>
                    Relative Error: {(error_explicit/C_analytical_final)*100:.3f}%
                </div>
                """, unsafe_allow_html=True)
            elif error_explicit < 0.1:
                st.markdown(f"""
                <div class="warning-box">
                    <strong>⚠️ Good Accuracy</strong><br>
                    Final Value: {C_explicit_final:.4f}<br>
                    Analytical: {C_analytical_final:.4f}<br>
                    Relative Error: {(error_explicit/C_analytical_final)*100:.2f}%
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="danger-box">
                    <strong>❌ Poor Accuracy!</strong><br>
                    Final Value: {C_explicit_final:.4f}<br>
                    Analytical: {C_analytical_final:.4f}<br>
                    Large numerical error detected!
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="success-box">
                <strong>✅ Implicit Euler Results</strong><br>
                Final Value: {C_implicit_final:.4f}<br>
                Analytical: {C_analytical_final:.4f}<br>
                Relative Error: {(error_implicit/C_analytical_final)*100:.3f}%
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### ⚖️ Stability Analysis")
        
        if method == "Explicit Euler":
            if h > critical_h:
                st.markdown(f"""
                <div class="danger-box">
                    <strong>🚨 UNSTABLE!</strong><br>
                    h = {h} > h_critical = {critical_h:.1f}<br>
                    Explicit Euler becomes unstable.<br>
                    Solution may oscillate or blow up!
                </div>
                """, unsafe_allow_html=True)
            elif h > critical_h * 0.8:
                st.markdown(f"""
                <div class="warning-box">
                    <strong>⚠️ Near Stability Limit</strong><br>
                    h = {h}, h_critical = {critical_h:.1f}<br>
                    Approaching instability region.<br>
                    Consider reducing step size.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-box">
                    <strong>✅ Stable Region</strong><br>
                    h = {h} < h_critical = {critical_h:.1f}<br>
                    Explicit Euler is stable.<br>
                    Solution behaves correctly.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="success-box">
                <strong>✅ Always Stable</strong><br>
                Implicit Euler is unconditionally stable.<br>
                No restrictions on step size h.<br>
                Perfect for stiff equations!
            </div>
            """, unsafe_allow_html=True)
    
    # Educational content
    st.markdown("---")
    st.markdown("## 🎓 Key Learning Points")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔍 What to Observe:
        
        **Small Step Sizes (h < 1.0):**
        - High accuracy, close to analytical solution
        - Stable behavior
        - More computational steps required
        
        **Medium Step Sizes (1.0 ≤ h ≤ 3.0):**
        - Reduced accuracy but manageable
        - Still stable for explicit method
        - Good balance of speed vs accuracy
        """)
    
    with col2:
        st.markdown("""
        ### ⚠️ Critical Observations:
        
        **Large Step Sizes (h > 4.0):**
        - Explicit method becomes unstable
        - Solution may become negative (unphysical!)
        - Implicit method remains stable
        
        **Stability Boundary:**
        - Critical h = 2/k = 4.0 for this problem
        - Beyond this, explicit Euler fails
        """)
    
    # Experimental suggestions
    st.markdown("---")
    st.markdown("## 🧪 Try These Experiments:")
    
    experiments = [
        "Set h = 0.1 and observe high accuracy",
        "Gradually increase h to see accuracy decrease",
        "Set h = 4.5 with Explicit Euler to see instability",
        "Compare the same large h with Implicit Euler",
        "Use 'Compare Both Methods' to see differences directly"
    ]
    
    for i, exp in enumerate(experiments, 1):
        st.markdown(f"**{i}.** {exp}")

if __name__ == "__main__":
    main()
