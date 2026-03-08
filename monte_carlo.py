import streamlit as st
import pandas as pd
import requests
import json 
from pathlib import Path
import os
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


# --- HELPER: YELLOW/GOLD THEME ---
def apply_monte_carlo_theme():
    st.markdown("""
        <style>
        /* Primary Button: Yellow/Gold */
        div.stButton > button:first-child { 
            background-color: #f1c40f; border-color: #f1c40f; color: black; font-weight: bold;
        }
        div.stButton > button:hover { 
            background-color: #d4ac0d; border-color: #d4ac0d; color: black; 
        }
        /* Style secondary buttons to match */
        div.stButton > button { border-color: #f1c40f; color: #f1c40f; }
        </style>
    """, unsafe_allow_html=True)

# --- STAGE 1: DATA SAMPLING ---
def show_sampling_stage():
    apply_monte_carlo_theme()
    st.markdown("<h1 style='text-align: center;'>Monte Carlo Simulation 🎲</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #f1c40f;'>Stochastic Data Generation</h2>", unsafe_allow_html=True)

    st.markdown("""
        <div style="display: flex; justify-content: center; margin-bottom: 25px;">
            <div style="background-color: rgba(241, 196, 15, 0.1); border-radius: 10px; padding: 20px; border: 1px solid #f1c40f; width: 80%; text-align: center;">
                <p style="font-size: 0.9rem; margin: 0; color: #f0f2f6; line-height: 1.5;">
                    🎯 The objective is to simulate thousands of stochastic paths to determine the Value at Risk (VaR)
                    and ensure the bank maintains sufficient liquidity for unexpected portfolio shocks.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    _, col_btn, _ = st.columns([1, 1, 1])
    with col_btn:
        if st.button("🎲 Generate a Sample", use_container_width=True):
            with st.spinner("Generating Stochastic Sample..."):
                try:
                    # UPDATED: Now hits the isolated MC sampler endpoint
                    res = requests.post(f"{BACKEND_URL}/run_mc_sampler").json()
                    if res.get("status") == "success":
                        st.session_state.mc_sampled = True
                        st.toast("Stochastic sample generated!")
                    else: st.error("Sampler failed.")
                except Exception as e: st.error(f"Connection Error: {e}")

    SAMPLE_PATH = Path("data/generated/markovian_sample.csv").resolve()
    
    if st.session_state.get("mc_sampled") and SAMPLE_PATH.exists():
        st.write("##")
        df_sample = pd.read_csv(SAMPLE_PATH)
        
        def style_mc_cols(df):
            styles = pd.DataFrame('', index=df.index, columns=df.columns)
            if 'loan_status' in df.columns:
                # 0.2 Alpha = 80% Transparent Yellow
                styles['loan_status'] = 'background-color: rgba(241, 196, 15, 0.2); color: #f1c40f; font-weight: bold;'
            return styles

        st.dataframe(df_sample.style.apply(style_mc_cols, axis=None), height=350, use_container_width=True)

        if st.button("Proceed to Matrix Setup ➡️", type="primary", use_container_width=True):
            st.session_state.current_page = "MatrixDisplay"
            st.rerun()

# --- STAGE 2: MATRIX DISPLAY ---
def show_matrix_stage():
    apply_monte_carlo_theme()
    st.markdown("<h1 style='text-align: center;'>Monte Carlo Simulation 🎲</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #f1c40f;'>Stochastic Transition Probabilities</h2>", unsafe_allow_html=True)

    try:
        # UPDATED: Hits the isolated MC matrix endpoint
        res = requests.get(f"{BACKEND_URL}/get_mc_matrix").json()
        if res.get("status") == "success":
            matrix = res["matrix"]
            labels = res["labels"]
            df_matrix = pd.DataFrame(matrix, index=labels, columns=labels)
            
            st.markdown("""
                <style>
                table { width: 100% !important; border-collapse: collapse; margin: auto; }
                th, td { border: 1px solid rgba(241, 196, 15, 0.3) !important; text-align: center !important; padding: 15px !important; }
                </style>
            """, unsafe_allow_html=True)

            _, col_mid, _ = st.columns([0.05, 0.9, 0.05])
            with col_mid:
                # Applied 80% transparent background with a subtle gradient
                st.table(
                    df_matrix.style.set_properties(**{
                        'background-color': 'rgba(241, 196, 15, 0.2)', 
                        'color': '#f1c40f', 'font-weight': 'bold'
                    }).background_gradient(cmap="YlOrBr", axis=None, low=0.9, high=0.7)
                      .format("{:.2%}")
                )

            st.markdown("""
                <div style="display: flex; justify-content: center; margin-top: 20px;">
                    <div style="background-color: rgba(241, 196, 15, 0.1); 
                                border-radius: 10px; padding: 15px; 
                                border: 1px solid #f1c40f; width: 80%; text-align: center;">
                        <p style="font-size: 0.9rem; margin: 0; color: #f0f2f6;">
                            💡 These stochastic transitions feed into the simulation engine to model tail-risk events.
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.write("##")
            if st.button("Proceed to Simulation ➡️", type="primary", use_container_width=True):
                st.session_state.current_page = "MCSimulation"
                st.rerun()
        else: st.error("Could not load matrix.")
    except Exception as e: st.error(f"Backend connection failed: {e}")





# --- STAGE 3: SIMULATION & MONITORING ---
def show_simulation_stage():
    apply_monte_carlo_theme()
    st.markdown("<h1 style='text-align: center;'>Monte Carlo Simulation 🎲</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #f1c40f;'>Stochastic Path Analysis</h2>", unsafe_allow_html=True)

    # Paths
    SAMPLE_PATH = Path("data/generated/markovian_sample.csv").resolve()
    # Path where the MC/Markov monitoring logic outputs results
    REPORT_PATH = Path("data/models/markov_chains/detailed_action_report.csv").resolve()

    # Initialize sub-states
    if "mc_sim_view" not in st.session_state:
        st.session_state.mc_sim_view = "original"
    if "mc_sim_executed" not in st.session_state:
        st.session_state.mc_sim_executed = False
    if "show_mc_final_summary" not in st.session_state:
        st.session_state.show_mc_final_summary = False

    st.write("##")
    
    # 1. Dynamic Table Display
    if st.session_state.mc_sim_view == "original" and SAMPLE_PATH.exists():
        st.markdown("### 📊 Baseline Portfolio Sample")
        df_orig = pd.read_csv(SAMPLE_PATH)
        
        def style_mc_original(df):
            styles = pd.DataFrame('', index=df.index, columns=df.columns)
            if 'loan_status' in df.columns:
                styles['loan_status'] = 'background-color: rgba(241, 196, 15, 0.2); color: #f1c40f;'
            return styles
        st.dataframe(df_orig.style.apply(style_mc_original, axis=None), use_container_width=True, height=400)

    elif st.session_state.mc_sim_view == "simulated" and REPORT_PATH.exists():
        st.markdown("### 🎲 Stochastic Risk Report")
        df_report = pd.read_csv(REPORT_PATH)

        def style_mc_action_report(row):
            styles = [''] * len(row)
            idx_tag = row.index.get_loc('status_tag') if 'status_tag' in row.index else -1
            idx_act = row.index.get_loc('action') if 'action' in row.index else -1
            
            if idx_tag != -1 and idx_act != -1:
                tag = str(row['status_tag']).lower()
                # Use semi-transparent status colors
                if 'stable' in tag: color = 'rgba(72, 201, 176, 0.2)' 
                elif 'warning' in tag: color = 'rgba(231, 76, 60, 0.2)' 
                else: color = 'rgba(241, 196, 15, 0.2)' 
                
                styles[idx_tag] = styles[idx_act] = f'background-color: {color}; color: #f1c40f; font-weight: bold;'
            return styles

        st.dataframe(df_report.style.apply(style_mc_action_report, axis=1), use_container_width=True, height=500)
    
    elif st.session_state.mc_sim_view == "simulated" and not REPORT_PATH.exists():
        st.error(f"Simulation report missing at {REPORT_PATH}")

    # 2. Action Buttons
    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 See Original Data", use_container_width=True):
            st.session_state.mc_sim_view = "original"
            st.rerun()
    with col2:
        if st.button("⚙️ Run Monte Carlo Simulation", type="primary", use_container_width=True):
            with st.spinner("Simulating 10,000 Stochastic Paths..."):
                try:
                    # UPDATED: Specifically hitting the MC RWA pipeline endpoint
                    res = requests.post(f"{BACKEND_URL}/run_mc_simulation").json()
                    if res.get("status") == "success":
                        st.session_state.mc_sim_view = "simulated"
                        st.session_state.mc_sim_executed = True
                        st.rerun()
                except Exception as e: st.error(f"Simulation Error: {e}")

    # 3. Final Proceed Button
    st.write("##")
    if st.button("Proceed to Risk Summary 🏁", use_container_width=True, 
                 disabled=not st.session_state.mc_sim_executed):
        st.session_state.show_mc_final_summary = True

    if st.session_state.show_mc_final_summary:
        show_mc_final_summary()

def show_mc_final_summary():
    st.write("---")
    st.markdown("<h2 style='text-align: center; color: #f1c40f;'>Stochastic RWA & Capital Adequacy Summary</h2>", unsafe_allow_html=True)
    
    METRICS_PATH = Path("json_files/monte_carlo/portfolio_rwa_comparison.json").resolve()

    _, col_mid, _ = st.columns([0.1, 0.8, 0.1])

    with col_mid:
        st.markdown("### 💰 Risk-Weighted Assets (RWA) Comparison")
        try:
            if not METRICS_PATH.exists():
                st.warning("RWA metrics file not found.")
                return

            with open(METRICS_PATH, 'r') as f:
                rwa_data = json.load(f)["rwa_report"]
            
            approach_metadata = {
                "standardized_approach": {"label": "Standardized Approach", "badge": "BASE", "color": "#f1c40f"},
                "irb_formula_approach": {"label": "IRB Formula Approach", "badge": "REGULATORY", "color": "#f39c12"},
                "monte_carlo_stochastic_approach": {"label": "Monte Carlo Stochastic", "badge": "STRESS TEST", "color": "#e67e22"}
            }

            for key, meta in approach_metadata.items():
                if key in rwa_data:
                    val = rwa_data[key]
                    st.markdown(f"""
                        <div style="border-left: 5px solid {meta['color']}; background-color: rgba(241, 196, 15, 0.1); padding: 15px; border-radius: 5px; margin-bottom: 15px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <h4 style="margin: 0; color: {meta['color']}; text-transform: uppercase; font-size: 0.85rem;">{meta['label']}</h4>
                                <span style="font-size: 0.65rem; background: {meta['color']}; color: black; padding: 2px 8px; border-radius: 4px; font-weight: bold;">{meta['badge']}</span>
                            </div>
                            <p style="margin: 10px 0 5px 0; font-size: 1.4rem; font-weight: bold; color: #f0f2f6;">${val:,.2f}</p>
                            <p style="margin: 0; font-size: 0.75rem; opacity: 0.7; color: #f1c40f;">Currency: {rwa_data.get('unit', 'USD')}</p>
                        </div>
                    """, unsafe_allow_html=True)

            st.caption(f"✨ Total Stochastic Scenarios Performed: {rwa_data.get('simulations_performed', 0):,}")

        except Exception as e:
            st.error(f"Error loading RWA metrics: {e}")

        st.write("##")
        st.markdown("""
            <div style="background-color: rgba(241, 196, 15, 0.15); border: 1px solid #f1c40f; border-radius: 10px; padding: 20px;">
                <p style="font-size: 0.85rem; margin: 0; color: #f0f2f6; line-height: 1.6; text-align: justify;">
                    💡 <b>Note:</b> This pipeline assesses liquidity survival for worst-case portfolio scenarios.
                    Usually, the RWA assigned by ALCO is monitored to determine if the bank needs to raise equity or adjust dividends.
                </p>
            </div>
        """, unsafe_allow_html=True)


        
# --- DISPATCHER ---
def run_monte_carlo_logic():
    apply_monte_carlo_theme()
    if st.session_state.current_page == "Home":
        show_sampling_stage()
    elif st.session_state.current_page == "MatrixDisplay":
        show_matrix_stage()
    elif st.session_state.current_page == "MCSimulation":
        show_simulation_stage()