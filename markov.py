import streamlit as st
import pandas as pd
import requests
from pathlib import Path
import json
import os
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
# --- HELPER: PURPLE BUTTON STYLING ---
def apply_purple_theme():
    st.markdown("""
        <style>
        div.stButton > button:first-child { background-color: #9b59b6; border-color: #9b59b6; color: white; }
        div.stButton > button:hover { background-color: #8e44ad; border-color: #8e44ad; color: white; }
        /* Style secondary buttons to match the purple vibe */
        div.stButton > button { border-color: #9b59b6; color: #9b59b6; }
        </style>
    """, unsafe_allow_html=True)

# --- STAGE 1: DATA SAMPLING ---
def show_sampling_stage():
    apply_purple_theme()
    st.markdown("<h1 style='text-align: center;'>Markov Transition Analysis ⛓️</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #9b59b6;'>Transition Matrix Simulations</h2>", unsafe_allow_html=True)

    st.markdown("""
        <div style="display: flex; justify-content: center; margin-bottom: 25px;">
            <div style="background-color: rgba(155, 89, 182, 0.2); border-radius: 10px; padding: 20px; border: 1px solid #9b59b6; width: 80%; text-align: center;">
                <p style="font-size: 0.9rem; margin: 0; color: #f0f2f6; line-height: 1.5;">
                    🎯 The goal is to keep the bank notified about the expected loss established in the portfolio 
                    and determine when they need to liquidate assets for the provisioning account.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    _, col_btn, _ = st.columns([1, 1, 1])
    with col_btn:
        if st.button("🎲 Generate a Sample", use_container_width=True):
            with st.spinner("Sampling Markovian Data..."):
                try:
                    # UPDATED: Specifically hitting the isolated Markov sampler
                    res = requests.post(f"{BACKEND_URL}/run_markov_sampler").json()
                    if res.get("status") == "success":
                        st.session_state.markov_sampled = True
                        st.toast("Stratified sample generated!")
                    else: st.error("Sampler failed.")
                except Exception as e: st.error(f"Connection Error: {e}")

    SAMPLE_PATH = Path("data/generated/markovian_sample.csv").resolve()
    if st.session_state.get("markov_sampled") and SAMPLE_PATH.exists():
        st.write("##")
        df_sample = pd.read_csv(SAMPLE_PATH)
        
        def style_cols(df):
            styles = pd.DataFrame('', index=df.index, columns=df.columns)
            status_cols = [c for c in ['loan_status', 'status_tag'] if c in df.columns]
            for col in status_cols:
                # Purple highlight with 20% alpha for 80% transparency
                styles[col] = 'background-color: rgba(155, 89, 182, 0.2); color: #e1bee7;'
            return styles

        st.dataframe(df_sample.style.apply(style_cols, axis=None), height=350, use_container_width=True)

        if st.button("Proceed to Matrix Setup ➡️", type="primary", use_container_width=True):
            st.session_state.current_page = "MatrixDisplay"
            st.rerun()

# --- STAGE 2: MATRIX DISPLAY ---
def show_matrix_stage():
    apply_purple_theme()
    st.markdown("<h1 style='text-align: center;'>Markov Transition Analysis ⛓️</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #9b59b6;'>Standard Migration Probabilities</h2>", unsafe_allow_html=True)

    try:
        # UPDATED: Hits the isolated Markov matrix endpoint
        res = requests.get(f"{BACKEND_URL}/get_markov_matrix").json()
        if res.get("status") == "success":
            matrix = res["matrix"]
            labels = res["labels"]
            df_matrix = pd.DataFrame(matrix, index=labels, columns=labels)
            
            st.markdown("""
                <style>
                table { width: 700px !important; height: 700px !important; table-layout: fixed !important; 
                        border-collapse: collapse; margin-left: auto; margin-right: auto; }
                th, td { width: 140px !important; height: 115px !important; 
                         border: 1px solid rgba(155, 89, 182, 0.5) !important;
                         text-align: center !important; padding: 0px !important; overflow: hidden; }
                .stTable { display: flex; justify-content: center; }
                </style>
            """, unsafe_allow_html=True)

            _, col_mid, _ = st.columns([0.1, 0.8, 0.1])
            with col_mid:
                st.table(
                    df_matrix.style.set_properties(**{
                        'background-color': 'rgba(155, 89, 182, 0.85)', 
                        'color': 'white', 'font-weight': 'bold'
                    }).background_gradient(cmap="Purples", axis=None, low=0.5, high=0.2)
                      .format("{:.2%}")
                )

            st.markdown("""
                <div style="display: flex; justify-content: center; margin-top: 20px;">
                    <div style="background-color: rgba(155, 89, 182, 0.2); 
                                border-radius: 10px; padding: 15px; 
                                border: 1px solid #9b59b6; width: 80%; text-align: center;">
                        <p style="font-size: 0.9rem; margin: 0; color: #f0f2f6;">
                            💡 This transition matrix represents historical rating migration probabilities.
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.write("##")
            if st.button("Proceed to Simulation ➡️", type="primary", use_container_width=True):
                st.session_state.current_page = "MarkovSimulation"
                st.rerun()
        else: st.error("Could not load transition matrix.")
    except Exception as e: st.error(f"Backend connection failed: {e}")




# --- STAGE 3: SIMULATION & MONITORING ---
def show_simulation_stage():
    apply_purple_theme()
    st.markdown("<h1 style='text-align: center;'>Markov Transition Analysis ⛓️</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #9b59b6;'>Portfolio Monitoring & Actions</h2>", unsafe_allow_html=True)

    SAMPLE_PATH = Path("data/generated/markovian_sample.csv").resolve()
    REPORT_PATH = Path("data/models/markov_chains/detailed_action_report.csv").resolve()

    # Initialize sub-states
    if "sim_view" not in st.session_state:
        st.session_state.sim_view = "original"
    if "sim_executed" not in st.session_state:
        st.session_state.sim_executed = False
    if "show_final_summary" not in st.session_state:
        st.session_state.show_final_summary = False

    st.write("##")
    
    # 1. Dynamic Table Display
    if st.session_state.sim_view == "original" and SAMPLE_PATH.exists():
        st.markdown("### Original Sampled Data")
        df_orig = pd.read_csv(SAMPLE_PATH)
        
        def style_original(df):
            styles = pd.DataFrame('', index=df.index, columns=df.columns)
            if 'loan_status' in df.columns:
                # 80% Opacity Purple
                styles['loan_status'] = 'background-color: rgba(155, 89, 182, 0.8); color: white;'
            return styles
        st.dataframe(df_orig.style.apply(style_original, axis=None), use_container_width=True, height=400)

    elif st.session_state.sim_view == "simulated" and REPORT_PATH.exists():
        st.markdown("### Detailed Action Report")
        df_report = pd.read_csv(REPORT_PATH)

        def style_action_report(row):
            styles = [''] * len(row)
            idx_tag = row.index.get_loc('status_tag') if 'status_tag' in row.index else -1
            idx_act = row.index.get_loc('action') if 'action' in row.index else -1
            if idx_tag != -1 and idx_act != -1:
                tag, act = str(row['status_tag']).lower(), str(row['action']).lower()
                # Use semi-transparent status colors for high contrast
                if 'stable' in tag and 'monitor' in act: color = 'rgba(72, 201, 176, 0.8)'
                elif 'late' in tag and 'direct outreach' in act: color = 'rgba(155, 89, 182, 0.8)'
                elif 'attention' in tag and 'escalate' in act: color = 'rgba(241, 196, 15, 0.8)'
                elif 'warning' in tag and 'legal' in act: color = 'rgba(231, 76, 60, 0.8)'
                else: color = ''
                if color: styles[idx_tag] = styles[idx_act] = f'background-color: {color}; color: white;'
            return styles

        st.dataframe(df_report.style.apply(style_action_report, axis=1), use_container_width=True, height=500)

    # 2. Action Buttons
    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 See Original Data", use_container_width=True):
            st.session_state.sim_view = "original"
            st.rerun()
    with col2:
        if st.button("⚙️ Apply Simulations", type="primary", use_container_width=True):
            try:
                # UPDATED: Specifically hitting the Markov monitoring pipeline
                res = requests.post(f"{BACKEND_URL}/run_markov_simulation").json()
                if res.get("status") == "success":
                    st.session_state.sim_view = "simulated"
                    st.session_state.sim_executed = True
                    st.rerun()
            except Exception as e: st.error(f"Error: {e}")

    # 3. Final Proceed Button
    st.write("##")
    if st.button("Proceed to Final Summary 🏁", use_container_width=True, disabled=not st.session_state.sim_executed):
        st.session_state.show_final_summary = True

    if st.session_state.show_final_summary:
        show_final_summary()


def show_final_summary():
    st.write("---")
    st.markdown("<h2 style='text-align: center; color: #9b59b6;'>Executive Provisioning & Outreach Summary</h2>", unsafe_allow_html=True)
    
    METRICS_PATH = Path("json_files/markov_chains/bank_provisioning_metrics.json").resolve()
    LATE_BORROWERS_PATH = Path("json_files/markov_chains/late_borrower_actions.json").resolve()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 💰 Capital Provisioning")
        try:
            with open(METRICS_PATH, 'r') as f:
                prov_data = json.load(f)["provisioning_report"]
            
            for level, metrics in prov_data.items():
                is_warning = "warning" in level
                border_color = "#e74c3c" if is_warning else "#f1c40f"
                bg_color = "rgba(231, 76, 60, 0.1)" if is_warning else "rgba(241, 196, 15, 0.1)"
                
                timeline = "Immediate (Next 30 Days)" if is_warning else "Short-term (Next 60 Days)"
                priority_badge = "CRITICAL" if is_warning else "ELEVATED"
                base_val = metrics.get("primary_requirement_base") or metrics.get("estimated_need_base")

                st.markdown(f"""
                    <div style="border-left: 5px solid {border_color}; background-color: {bg_color}; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
                        <div style="display: flex; justify-content: space-between;">
                            <h4 style="margin: 0; color: {border_color}; text-transform: capitalize;">{level.replace('_', ' ')}</h4>
                            <span style="font-size: 0.7rem; background: {border_color}; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold;">{priority_badge}</span>
                        </div>
                        <p style="margin: 10px 0 5px 0; font-size: 1.2rem; font-weight: bold; color: #f0f2f6;">${base_val:,.2f}</p>
                        <p style="margin: 0; font-size: 0.85rem; color: #9b59b6;"><b>Required by:</b> {timeline}</p>
                        <hr style="margin: 10px 0; border: 0; border-top: 1px solid rgba(255,255,255,0.1);">
                        <p style="margin: 0; font-size: 0.75rem; opacity: 0.8;">
                            Stress Range: ${metrics['optimistic_lower_bound']:,.2f} - ${metrics['stressed_upper_bound']:,.2f}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading metrics: {e}")

    with col_right:
        st.markdown("### 📢 Operational Outreach")
        try:
            with open(LATE_BORROWERS_PATH, 'r') as f:
                late_data = json.load(f)
            df_late = pd.DataFrame(late_data)
            df_late['id'] = df_late['id'].astype(int).astype(str)
            
            st.markdown('<div style="background-color: rgba(155, 89, 182, 0.1); border: 1px solid #9b59b6; border-radius: 10px; padding: 10px;">', unsafe_allow_html=True)
            st.dataframe(
                df_late.style.set_properties(**{
                    'background-color': 'rgba(155, 89, 182, 0.05)',
                    'color': '#e1bee7',
                    'border-color': 'rgba(155, 89, 182, 0.2)'
                }), 
                use_container_width=True, height=300
            )
            st.markdown("</div>", unsafe_allow_html=True)
            st.info(f"Priority outreach actions: {len(df_late)}")
        except Exception as e:
            st.error(f"Error loading outreach data: {e}")


# --- CLEANED DISPATCHER ---
def run_markov_logic():
    apply_purple_theme()
    if st.session_state.current_page == "Home":
        show_sampling_stage()
    elif st.session_state.current_page == "MatrixDisplay":
        show_matrix_stage()
    elif st.session_state.current_page == "MarkovSimulation":
        show_simulation_stage()