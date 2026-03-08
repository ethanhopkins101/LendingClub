import streamlit as st
import pandas as pd
import requests
from pathlib import Path
import os
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# --- 1. PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="Banking Risk App", page_icon="🏦")

# --- 2. GLASSY UI STYLING ---
st.markdown("""
    <style>
    .stButton>button { border-radius: 10px; height: 3em; }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
        min-height: 80px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. INITIALIZE STATE ---
# We use 'app_mode' to track which tool we are using (Portfolio, Markov, or Monte Carlo)
if "app_mode" not in st.session_state: st.session_state.app_mode = "Portfolio"
if "current_page" not in st.session_state: st.session_state.current_page = "Home"


# --- 4. TOP NAVIGATION BAR ---
nav_fill1, nav_col, nav_fill2 = st.columns([1, 2, 1])
with nav_col:
    c1, c2, c3 = st.columns(3)
    
    if c1.button("💼 Portfolio", use_container_width=True): 
        st.session_state.app_mode = "Portfolio"
        st.session_state.current_page = "Home" # Reset to Stage 1
        st.rerun()
        
    if c2.button("⛓️ Markov", use_container_width=True): 
        st.session_state.app_mode = "Markov"
        st.session_state.current_page = "Home" # Reset to Stage 1
        st.rerun()
        
    if c3.button("🎲 Monte Carlo", use_container_width=True): 
        st.session_state.app_mode = "Monte Carlo"
        st.session_state.current_page = "Home" # Reset to Stage 1
        st.rerun()

st.divider()

# --- 5. ROUTING ENGINE (The Canvas) ---
main_canvas = st.container()

with main_canvas:
    # --- A. PORTFOLIO WORKFLOW ---
    if st.session_state.app_mode == "Portfolio":
        # --- 5. HOME PAGE ---
        if st.session_state.current_page == "Home":
            # 1. Centered H1 with Bank Icon
            st.markdown("<h1 style='text-align: center;'>Risk Modeling Web App 🏦</h1>", unsafe_allow_html=True)
            
            # 2. Centered H2
            st.markdown("<h2 style='text-align: center;'>How will you provide your data?</h2>", unsafe_allow_html=True)

            # 3. Centered Generate Button
            # Using columns to create a "center" wrapper for the button
            _, col_btn, _ = st.columns([1, 2, 1])
            
            with col_btn:
                # Keeping the exact button style/shape requested
                if st.button("🎲 Generate Your Sample Data Now", use_container_width=True):
                    try:
                        res = requests.get(f"{BACKEND_URL}/generate_random_data").json()
                        if res.get("status") == "success":
                            st.session_state.df = pd.DataFrame(res["data"])
                            st.toast("Sample data generated successfully!")
                        else:
                            st.error("Backend error: " + res.get("detail", "Unknown error"))
                    except Exception as e:
                        st.error(f"Backend unreachable: {e}")

            # 4. Data Preview & Navigation
            if st.session_state.get("df") is not None:
                st.write("---")
                st.subheader("Data Preview")
                st.dataframe(st.session_state.df, height=280, use_container_width=True)
                
                if st.button("Proceed to Risk Engine", type="primary", use_container_width=True):
                    st.session_state.current_page = "Engine"
                    st.rerun()

        # --- 6. ENGINE PAGE ---
        elif st.session_state.current_page == "Engine":
            # 1. INITIALIZE ALL STATE
            if "engine_init" not in st.session_state:
                st.session_state.approval = "98.83%"
                st.session_state.fn = "19.60%"
                st.session_state.fp = "0.55%"
                st.session_state.view_mode = "original"
                st.session_state.model_applied = False  # Track if model has been run
                st.session_state.engine_init = True

            # 2. Load Strategy Data for Slider Constraints
            STRATEGY_PATH = Path("data/models/initial_review/strategy_analysis_report.csv")
            
            if "valid_thresholds" not in st.session_state:
                if STRATEGY_PATH.exists():
                    strategy_df = pd.read_csv(STRATEGY_PATH)
                    st.session_state.valid_thresholds = sorted(strategy_df['Threshold'].unique().tolist())
                    st.session_state.default_thresh = min(st.session_state.valid_thresholds, key=lambda x: abs(x - 0.5))
                else:
                    st.session_state.valid_thresholds = [0.5]
                    st.session_state.default_thresh = 0.5

            # 3. Metric Update Helper
            def update_metrics(t):
                try:
                    res = requests.get(f"{BACKEND_URL}/get_metrics?threshold={t}").json()
                    if res["status"] == "success":
                        st.session_state.approval = res.get("approval", "--")
                        st.session_state.fp = res.get("fp", "--")
                        st.session_state.fn = res.get("fn", "--")
                except Exception as e:
                    st.error(f"Metric Fetch Error: {e}")

            # 4. Row Highlighting Logic
            def highlight_rows(row):
                if st.session_state.get("view_mode") == "applied" and st.session_state.results_df is not None:
                    try:
                        pred = st.session_state.results_df.iloc[row.name]['initial_prediction']
                        color = 'background-color: rgba(144, 238, 144, 0.2)' if pred == 0 else 'background-color: rgba(255, 182, 193, 0.2)'
                        return [color] * len(row)
                    except: return [''] * len(row)
                return [''] * len(row)

            # 5. Main Data Display
            if st.session_state.df is not None:
                st.dataframe(
                    st.session_state.df.style.apply(highlight_rows, axis=1), 
                    height=400, 
                    use_container_width=True
                )

                # Stage 1 Action Buttons
                col_btn1, col_btn2 = st.columns(2)
                if col_btn1.button("View Original Data", use_container_width=True):
                    st.session_state.view_mode = "original"
                    st.rerun()

                if col_btn2.button("Apply Risk Model", use_container_width=True, type="primary"):
                    with st.spinner("Executing Risk Engine..."):
                        t_val = st.session_state.get('slider_thresh', st.session_state.default_thresh)
                        try:
                            requests.post(f"{BACKEND_URL}/run_engine?threshold={t_val}")
                            res_path = Path("data/models/initial_review/initial_review_results.csv").resolve()
                            if res_path.exists():
                                st.session_state.results_df = pd.read_csv(res_path)
                                st.session_state.view_mode = "applied"
                                st.session_state.model_applied = True  # Unlock the next button
                                update_metrics(t_val)
                                st.rerun()
                        except Exception as e: 
                            st.error(f"Engine Error: {e}")

                st.divider()
                st.markdown("<h2 style='text-align: center;'>Bank Risk Appetite</h2>", unsafe_allow_html=True)
                
                col_slide, col_m1, col_m2, col_m3 = st.columns([2, 1, 1, 1])
                
                with col_slide:
                    threshold = st.select_slider(
                        "Adjust Decision Probability Threshold", 
                        options=st.session_state.valid_thresholds, 
                        value=st.session_state.default_thresh, 
                        key="slider_thresh"
                    )
                    
                    if st.button("Refresh Metrics", use_container_width=True):
                        update_metrics(threshold)
                        st.rerun()
                
                # 6. Display Metrics
                with col_m1: 
                    st.markdown(f'<div class="metric-card"><p>Approval Rate</p><h2>{st.session_state.approval}</h2></div>', unsafe_allow_html=True)
                with col_m2: 
                    st.markdown(f'<div class="metric-card"><p>False Negatives</p><h2>{st.session_state.fn}</h2></div>', unsafe_allow_html=True)
                with col_m3: 
                    st.markdown(f'<div class="metric-card"><p>False Positives</p><h2>{st.session_state.fp}</h2></div>', unsafe_allow_html=True)

                # Final Navigation
                st.write("##") 
                
                # Logic: Disable button until model_applied is True
                proceed_disabled = not st.session_state.model_applied
                
                if st.button(
                    "Proceed to Strategy Reports", 
                    type="primary", 
                    use_container_width=True, 
                    disabled=proceed_disabled,
                    help="Please 'Apply Risk Model' first to unlock this report." if proceed_disabled else None
                ):
                    st.session_state.current_page = "Reports"
                    st.rerun()



        # --- 7. REPORTS PAGE (Approved Loans & Interest Prep) ---
        elif st.session_state.current_page == "Reports":
            # 1. Aesthetic Centered Header
            st.markdown("<h1 style='text-align: center;'>✅ Currently Approved Loans</h1>", unsafe_allow_html=True)
            
            # 2. Custom CSS for boxes
            st.markdown("""
                <style>
                    .custom-info-box {
                        background-color: rgba(30, 144, 255, 0.3);
                        border-radius: 10px;
                        padding: 15px;
                        border-left: 5px solid #1E90FF;
                        margin-bottom: 20px;
                    }
                    .yellow-warning-box {
                        background-color: rgba(255, 215, 0, 0.3); /* Yellow with 30% opacity */
                        border-radius: 8px;
                        padding: 10px;
                        border-left: 5px solid #FFD700;
                        margin-bottom: 15px;
                    }
                    .small-text {
                        font-size: 0.75rem !important;
                        color: #f0f2f6;
                        margin: 0;
                        text-align: center;
                    }
                </style>
                <div class="custom-info-box">
                    <p class="small-text">
                        📋 The bank now needs additional profile information to proceed with calculating the risk-adjusted interest rate.
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # Paths - Using absolute resolution
            FILTERED_CSV = Path("data/generated/sample_filtered.csv").resolve()
            GENERATED_CSV = Path("data/generated/risk_engine_sample_generated.csv").resolve()
            FINAL_CSV = Path("data/generated/final_pricing.csv").resolve()

            # 1. Initialization & State Locking
            if "reports_init" not in st.session_state:
                st.session_state.samples_generated = False 
                st.session_state.pricing_calculated = False 
                st.session_state.reports_init = True

            # Initial Filtering (Runs once)
            if "filtered_df" not in st.session_state:
                with st.spinner("Filtering approved loans..."):
                    try:
                        resp = requests.post(f"{BACKEND_URL}/filter_loans")
                        if resp.status_code == 200 and FILTERED_CSV.exists():
                            st.session_state.filtered_df = pd.read_csv(FILTERED_CSV)
                            st.session_state.current_view = "filtered"
                        else:
                            st.error(f"Backend failed to filter data or file missing at {FILTERED_CSV}")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")

            # 2. Table Selection Logic
            if st.session_state.get("current_view") == "final":
                current_table_df = st.session_state.get("final_pricing_df")
            elif st.session_state.get("current_view") == "sampled":
                current_table_df = st.session_state.get("risk_generated_df")
            else:
                current_table_df = st.session_state.get("filtered_df")

            # 3. Styling
            def style_final_table(df):
                styles = pd.DataFrame('', index=df.index, columns=df.columns)
                if 'risk_premium_rate' in df.columns:
                    styles['risk_premium_rate'] = 'background-color: rgba(255, 192, 203, 0.4); color: white' 
                if 'int_rate' in df.columns:
                    styles['int_rate'] = 'background-color: rgba(211, 211, 211, 0.4); color: white'
                if 'base_interest_rate_pct' in df.columns:
                    styles['base_interest_rate_pct'] = 'background-color: rgba(255, 255, 224, 0.4); color: white'
                return styles

            # Display Table
            with st.container(border=True):
                if current_table_df is not None:
                    if st.session_state.get("current_view") == "final":
                        st.dataframe(current_table_df.style.apply(style_final_table, axis=None), height=300, use_container_width=True)
                    else:
                        st.dataframe(current_table_df, height=300, use_container_width=True)
                else:
                    st.warning("No data available to display.")

            st.write("##") 
            
            # 4. Action Buttons (Step 1)
            col_act1, col_act2 = st.columns(2)
            with col_act1:
                st.button("⌨️ Fill Information Manually", use_container_width=True, disabled=True)

            with col_act2:
                if st.button("🎲 Randomly Generate Data", use_container_width=True, type="primary"):
                    with st.spinner("Generating risk samples..."):
                        try:
                            requests.post(f"{BACKEND_URL}/generate_risk_samples")
                            if GENERATED_CSV.exists():
                                st.session_state.risk_generated_df = pd.read_csv(GENERATED_CSV)
                                st.session_state.current_view = "sampled"
                                st.session_state.samples_generated = True 
                                st.rerun()
                        except Exception as e:
                            st.error(f"Generation Error: {e}")

            # 5. Calculate Section (Step 2)
            st.write("---")
            calc_disabled = not st.session_state.get("samples_generated", False)
            
            if st.button("💹 Calculate Interest Rate", 
                        use_container_width=True, 
                        disabled=calc_disabled,
                        help="Generate data first to unlock pricing calculation."):
                with st.spinner("Compiling Final Pricing..."):
                    try:
                        requests.post(f"{BACKEND_URL}/compile_final_pricing")
                        if FINAL_CSV.exists():
                            st.session_state.final_pricing_df = pd.read_csv(FINAL_CSV)
                            st.session_state.current_view = "final"
                            st.session_state.pricing_calculated = True 
                            st.rerun()
                    except Exception as e:
                        st.error(f"Pricing Error: {e}")

            # 6. Final Proceed (Step 3)
            st.write("---")
            
            # Yellow Compliance Box before the button
            st.markdown("""
                <div class="yellow-warning-box">
                    <p class="small-text">
                        ⚠️ Proceed only when the clients are aware and are okay with the displayed interest rates.
                    </p>
                </div>
            """, unsafe_allow_html=True)

            proceed_disabled = not st.session_state.get("pricing_calculated", False)
            
            _, col_proc, _ = st.columns([1, 2, 1])
            with col_proc:
                if st.button("Proceed", 
                            use_container_width=True, 
                            type="primary", 
                            key="final_proceed", 
                            disabled=proceed_disabled,
                            help="Calculate interest rates first to proceed."):
                    st.session_state.current_page = "Scorecards"
                    st.rerun()



        # --- 8. SCORECARDS PAGE (Probability of Default) ---
        elif st.session_state.current_page == "Scorecards":
            st.markdown("<h1 style='text-align: center;'>Designing Scorecards 📊</h1>", unsafe_allow_html=True)

            # Paths
            FINAL_PRICING_CSV = Path("data/generated/final_pricing.csv").resolve()
            RISK_REPORT_CSV = Path("data/models/probability_of_default/final_risk_report.csv").resolve()

            # 1. Initialize View and Lock State
            if "scorecard_init" not in st.session_state:
                st.session_state.scorecard_view = "original"
                st.session_state.scores_obtained = False # Lock for Proceed button
                st.session_state.scorecard_init = True

            # 2. Styling Functions
            def style_pricing_table(df):
                styles = pd.DataFrame('', index=df.index, columns=df.columns)
                if 'int_rate' in df.columns:
                    styles['int_rate'] = 'background-color: rgba(255, 192, 203, 0.4); color: white'
                return styles

            def style_risk_table(df):
                styles = pd.DataFrame('', index=df.index, columns=df.columns)
                if 'credit_score' in df.columns:
                    styles['credit_score'] = 'background-color: rgba(255, 255, 224, 0.4); color: white'
                if 'Expected_Loss' in df.columns:
                    styles['Expected_Loss'] = 'background-color: rgba(255, 192, 203, 0.4); color: white'
                return styles

            # 3. Data Selection Logic & Table Display
            with st.container(border=True):
                if st.session_state.scorecard_view == "original":
                    if FINAL_PRICING_CSV.exists():
                        df_to_show = pd.read_csv(FINAL_PRICING_CSV)
                        st.dataframe(df_to_show.style.apply(style_pricing_table, axis=None), height=350, use_container_width=True)
                    else:
                        st.error("Final pricing data not found.")
                elif st.session_state.scorecard_view == "scores":
                    if RISK_REPORT_CSV.exists():
                        df_to_show = pd.read_csv(RISK_REPORT_CSV)
                        st.dataframe(df_to_show.style.apply(style_risk_table, axis=None), height=350, use_container_width=True)
                    else:
                        st.warning("Risk report not found. Please click 'Obtain Scores' first.")

            st.write("##")

            # 4. Action Buttons (Step 1)
            col_score1, col_score2 = st.columns(2)
            
            with col_score1:
                if st.button("📄 Display Original Data", use_container_width=True):
                    st.session_state.scorecard_view = "original"
                    st.rerun()

            with col_score2:
                if st.button("🎯 Obtain Scores", use_container_width=True, type="primary"):
                    with st.spinner("Running PD Pipeline..."):
                        try:
                            resp = requests.post(f"{BACKEND_URL}/run_pd_pipeline")
                            if resp.status_code == 200:
                                st.session_state.scorecard_view = "scores"
                                st.session_state.scores_obtained = True # UNLOCK PROCEED
                                st.rerun()
                            else:
                                st.error("Failed to execute PD pipeline.")
                        except Exception as e:
                            st.error(f"Error: {e}")

            # 5. Final Proceed Section (Step 2)
            st.write("---")

            # Yellow Box with smaller font and reduced opacity
            st.markdown("""
                <style>
                    .yellow-warning-box {
                        background-color: rgba(255, 215, 0, 0.3); /* Yellow with ~30% opacity */
                        border-radius: 8px;
                        padding: 10px;
                        border-left: 5px solid #FFD700;
                        margin-bottom: 15px;
                    }
                    .yellow-text {
                        font-size: 0.75rem !important;
                        color: #f0f2f6;
                        margin: 0;
                        text-align: center;
                    }
                </style>
                <div class="yellow-warning-box">
                    <p class="yellow-text">
                        ⚠️ Proceed only when the Asset-Liability Committee gives you the green light.
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # Dependency Logic for Proceed
            proceed_locked = not st.session_state.get("scores_obtained", False)

            _, col_fin, _ = st.columns([1, 2, 1])
            with col_fin:
                if st.button(
                    "Proceed", 
                    key="scorecard_proceed", 
                    use_container_width=True, 
                    type="primary",
                    disabled=proceed_locked,
                    help="You must 'Obtain Scores' before optimizing the portfolio." if proceed_locked else None
                ):
                    st.balloons()
                    st.session_state.current_page = "Optimization" 
                    st.toast("Loading Portfolio Optimization Step...")
                    st.rerun()

        # --- 9. PORTFOLIO OPTIMIZATION PAGE ---
        elif st.session_state.current_page == "Optimization":
            st.markdown("<h1 style='text-align: center;'>Portfolio Optimization Step ⚖️</h1>", unsafe_allow_html=True)

            # Paths
            RISK_REPORT_CSV = Path("data/models/probability_of_default/final_risk_report.csv").resolve()
            METRICS_CSV = Path("data/generated/portfolio_metrics.csv").resolve()
            ALCO_CSV = Path("data/generated/alco_generated.csv").resolve()
            OPTIMAL_CSV = Path("data/generated/optimal.csv").resolve()
            OPT_METRICS_CSV = Path("data/generated/opt_metrics.csv").resolve()

            # 1. Initialization for Logic Locking
            if "opt_init" not in st.session_state:
                st.session_state.alco_set = False  # Lock for Optimization button
                st.session_state.opt_init = True

            # Helper function for currency formatting
            def format_metric_val(metric_name, val):
                currency_metrics = ["Total Exposure at Default (EAD)", "Total Expected Loss (EL)", "Total Expected Revenue", "Total Expected Profit"]
                if metric_name in currency_metrics:
                    return f"${float(val):,.22f}".split('.')[0] + f".{str(val).split('.')[-1][:2]}"
                return str(val)

            # 2. Main Data Table
            if RISK_REPORT_CSV.exists():
                df_risk = pd.read_csv(RISK_REPORT_CSV)
                def style_risk(df):
                    styles = pd.DataFrame('', index=df.index, columns=df.columns)
                    if 'credit_score' in df.columns:
                        styles['credit_score'] = 'background-color: rgba(255, 255, 224, 0.4); color: white'
                    if 'Expected_Loss' in df.columns:
                        styles['Expected_Loss'] = 'background-color: rgba(255, 192, 203, 0.4); color: white'
                    return styles
                
                with st.container(border=True):
                    st.dataframe(df_risk.style.apply(style_risk, axis=None), height=250, use_container_width=True)

            st.write("---")

            # 3. Automatic Metrics Calculation
            if "metrics_calculated" not in st.session_state:
                with st.spinner("Calculating Portfolio KPIs..."):
                    requests.post(f"{BACKEND_URL}/calculate_portfolio_metrics")
                    st.session_state.metrics_calculated = True

            # 4. Grid Layout: Metrics vs ALCO
            col_left, col_right = st.columns([1, 1], gap="large")

            with col_left:
                st.markdown("### 📈 Portfolio KPIs")
                if st.button("🔄 Refresh Metrics", use_container_width=True):
                    requests.post(f"{BACKEND_URL}/calculate_portfolio_metrics")
                    st.rerun()
                
                if METRICS_CSV.exists():
                    metrics_df = pd.read_csv(METRICS_CSV)
                    with st.container(border=True):
                        for _, row in metrics_df.iterrows():
                            formatted_val = format_metric_val(row['Metric'], row['Value'])
                            c1, c2 = st.columns([2, 1])
                            c1.markdown(f"<span style='font-size:0.85rem;'>**{row['Metric']}**</span>", unsafe_allow_html=True)
                            c2.markdown(f"<p style='text-align:right; font-family: sans-serif; color:#007AFF; font-weight:bold; font-size:0.85rem;'>{formatted_val}</p>", unsafe_allow_html=True)

            with col_right:
                st.markdown("### 🏛️ ALCO Limits")
                # Aesthetic info box for ALCO
                st.markdown("""
                    <style>
                        .alco-info { background-color: rgba(30, 144, 255, 0.2); border-radius: 8px; padding: 10px; border-left: 4px solid #1E90FF; margin-bottom: 10px; }
                        .alco-text { font-size: 0.7rem !important; color: #f0f2f6; margin: 0; }
                    </style>
                    <div class="alco-info"><p class="alco-text">📌 Define the constraints provided by the Asset-Liability Committee to guide the solver.</p></div>
                """, unsafe_allow_html=True)

                if "alco_vals" not in st.session_state:
                    st.session_state.alco_vals = [0.0, 0.0, 0.0]

                c1, c2, c3 = st.columns(3)
                v1 = c1.number_input("RWA Limit", value=float(st.session_state.alco_vals[0]))
                v2 = c2.number_input("Provision", value=float(st.session_state.alco_vals[1]))
                v3 = c3.number_input("Liquidity", value=float(st.session_state.alco_vals[2]))
                st.session_state.alco_vals = [v1, v2, v3]

                btn_man, btn_rand = st.columns(2)
                if btn_man.button("✍️ Input Manually", use_container_width=True):
                    st.session_state.alco_mode = "manual"
                    st.session_state.alco_set = True # Unlock button
                    st.toast("ALCO constraints updated manually.")

                if btn_rand.button("🎲 Generate Randomly", use_container_width=True):
                    with st.spinner("Generating ALCO via Script..."):
                        requests.post(f"{BACKEND_URL}/generate_alco")
                        if ALCO_CSV.exists():
                            alco_df = pd.read_csv(ALCO_CSV)
                            st.session_state.alco_vals = alco_df.iloc[0].tolist()
                            st.session_state.alco_mode = "random"
                            st.session_state.alco_set = True # Unlock button
                            st.rerun()

            # 5. Optimization Action
            st.write("##")
            
            # Yellow warning box before optimization
            st.markdown("""
                <style>
                    .opt-warning { background-color: rgba(255, 215, 0, 0.2); border-radius: 8px; padding: 10px; border-left: 4px solid #FFD700; margin-bottom: 15px; }
                </style>
                <div class="opt-warning"><p class="small-text" style="font-size:0.75rem; text-align:center; color:#f0f2f6;">⚠️ Ensure ALCO limits are finalized before executing the Mathematical Solver.</p></div>
            """, unsafe_allow_html=True)

            # Logic: Disable if ALCO hasn't been set via button
            opt_disabled = not st.session_state.alco_set

            if st.button("🚀 Optimize Portfolio", 
                        type="primary", 
                        use_container_width=True, 
                        disabled=opt_disabled,
                        help="Please set ALCO limits (Manual or Random) to unlock the solver."):
                with st.spinner("Executing Solver & Calculating Opt Metrics..."):
                    mode = st.session_state.get("alco_mode", "manual")
                    payload = {"mode": mode, "values": st.session_state.alco_vals, "path": str(ALCO_CSV)}
                    resp = requests.post(f"{BACKEND_URL}/optimize_portfolio", json=payload)
                    
                    if resp.status_code == 200:
                        requests.post(f"{BACKEND_URL}/calculate_opt_metrics")
                        st.session_state.show_optimal = True
                        st.rerun()
                    else:
                        st.error("Optimization failed. Check ALCO limits.")

            # 6. Display Comparison View
            if st.session_state.get("show_optimal") and OPTIMAL_CSV.exists():
                st.write("---")
                st.markdown("### 🏆 Optimal Portfolio Selection")
                st.balloons()
                
                df_opt = pd.read_csv(OPTIMAL_CSV)
                with st.container(border=True):
                    st.dataframe(df_opt.style.apply(style_risk, axis=None), height=300, use_container_width=True)
                
                st.write("##")
                st.markdown("### 📊 Performance Comparison: Raw vs Optimized")
                
                comp_left, comp_right = st.columns(2)
                
                with comp_left:
                    st.caption("Raw Portfolio (Baseline)")
                    if METRICS_CSV.exists():
                        m_df = pd.read_csv(METRICS_CSV)
                        with st.container(border=True):
                            for _, row in m_df.iterrows():
                                disp_val = format_metric_val(row['Metric'], row['Value'])
                                c1, c2 = st.columns([2, 1])
                                c1.markdown(f"<span style='font-size:0.8rem;'>{row['Metric']}</span>", unsafe_allow_html=True)
                                c2.markdown(f"<p style='text-align:right; font-size:0.8rem;'>{disp_val}</p>", unsafe_allow_html=True)

                with comp_right:
                    st.caption("Optimized Portfolio (Post-ALCO)")
                    if OPT_METRICS_CSV.exists():
                        o_df = pd.read_csv(OPT_METRICS_CSV)
                        with st.container(border=True):
                            for _, row in o_df.iterrows():
                                disp_val = format_metric_val(row['Metric'], row['Value'])
                                c1, c2 = st.columns([2, 1])
                                c1.markdown(f"<span style='font-size:0.8rem; font-weight:bold;'>{row['Metric']}</span>", unsafe_allow_html=True)
                                c2.markdown(f"<p style='text-align:right; color:#34C759; font-weight:bold; font-size:0.8rem;'>{disp_val}</p>", unsafe_allow_html=True)






# --- B. MARKOV WORKFLOW ---
    elif st.session_state.app_mode == "Markov":
        import markov  # Ensure markov.py is in the same directory
        
        with main_canvas:
            # We simply call the dispatcher. 
            # markov.py already knows which stage to show based on st.session_state.current_page
            markov.run_markov_logic()


# --- C. MONTE CARLO WORKFLOW ---
    elif st.session_state.app_mode == "Monte Carlo":
        import monte_carlo  # Ensure monte_carlo.py is in the same directory
        
        with main_canvas:
            # The yellow theme is applied inside the dispatcher
            monte_carlo.run_monte_carlo_logic()