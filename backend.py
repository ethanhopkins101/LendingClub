from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import subprocess
import sys
import os
from pathlib import Path
import json

# --- CORE IMPORTS FROM YOUR MODELS ---
from src.models.initial_review.execution import run_full_pipe
from src.models.probability_of_default.execution import run_full_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- PATH CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
STRATEGY_FILE = BASE_DIR / "data" / "models" / "initial_review" / "strategy_analysis_report.csv"
DATA_IN = BASE_DIR / "data" / "generated" / "sample_data.csv"

# --- STAGE 1 & 2: INITIAL REVIEW ---

@app.get("/generate_random_data")
def generate_and_get_data():
    """Stage 1: Execute generate_ir_sample.py and load resulting data"""
    
    # 1. Define Paths - Anchored to BASE_DIR for Docker compatibility
    script_path = (BASE_DIR / "src" / "data" / "generate_ir_sample.py").resolve()
    generated_csv_path = (BASE_DIR / "data" / "generated" / "sample_data.csv").resolve()

    try:
        # 2. Execute the generation script and wait
        # Added cwd=str(BASE_DIR) so the script knows where the 'data' folder is
        # Added PYTHONPATH so the script can find 'src' modules if needed
        env = os.environ.copy()
        env["PYTHONPATH"] = str(BASE_DIR)
        
        subprocess.run(
            [sys.executable, str(script_path)], 
            check=True, 
            cwd=str(BASE_DIR),
            env=env
        )

        # 3. Check if the script actually produced the file
        if not generated_csv_path.exists():
            raise FileNotFoundError(f"Script ran but output file missing at: {generated_csv_path}")

        # 4. Load the newly generated data
        df = pd.read_csv(generated_csv_path, low_memory=False).fillna("")
        
        return {
            "status": "success", 
            "message": "Data generated and loaded successfully",
            "data": df.head(30).to_dict(orient="records")
        }

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Generation script failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backend Error: {str(e)}")
    

@app.get("/get_metrics")
def get_metrics(threshold: float = Query(0.5)):
    """Stage 2: Fetch metrics for UI"""
    if not STRATEGY_FILE.exists():
        return {"status": "error", "message": f"Strategy file missing at: {STRATEGY_FILE}"}
    try:
        df = pd.read_csv(STRATEGY_FILE)
        idx = (df['Threshold'] - threshold).abs().idxmin()
        row = df.iloc[idx]
        return {
            "status": "success",
            "threshold_used": float(row['Threshold']),
            "approval": f"{float(row['Approval %']):.2f}%",
            "fp": f"{float(row['FP % of Total']):.2f}%",
            "fn": f"{float(row['FN % of Total']):.2f}%"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import os

# --- STAGE 2: INITIAL REVIEW ---

@app.post("/run_engine")
def run_engine(threshold: float = Query(0.5)):
    """Stage 2: Execute Initial Review Pipeline"""
    try:
        # DATA_IN is already anchored to BASE_DIR from the previous section
        run_full_pipe(str(DATA_IN.resolve()), threshold=threshold)
        return {"status": "success", "message": "Engine complete"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- STAGE 3: FILTERING & SAMPLING ---

@app.post("/filter_loans")
def filter_loans():
    """Stage 3: Run filter_price_engine.py"""
    script_path = (BASE_DIR / "src" / "data" / "filter_price_engine.py").resolve()
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(BASE_DIR)
        
        subprocess.run(
            [sys.executable, str(script_path)], 
            check=True,
            cwd=str(BASE_DIR),
            env=env
        )
        return {"status": "success", "message": "Loans filtered"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_risk_samples")
def generate_risk_samples():
    """Stage 3: Run risk_engine_sampler.py"""
    script_path = (BASE_DIR / "src" / "data" / "risk_engine_sampler.py").resolve()
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(BASE_DIR)

        subprocess.run(
            [sys.executable, str(script_path)], 
            check=True,
            cwd=str(BASE_DIR),
            env=env
        )
        return {"status": "success", "message": "Risk data generated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- STAGE 4: PRICING & PD PIPELINE ---

@app.post("/compile_final_pricing")
def compile_final_pricing():
    # Define absolute paths using the BASE_DIR
    execution_script = (BASE_DIR / "src" / "models" / "price_engine" / "execution.py").resolve()
    compiler_script = (BASE_DIR / "src" / "data" / "engine_compiler.py").resolve()
    
    # Path to the data generated in the previous step
    input_data = (BASE_DIR / "data" / "generated" / "risk_engine_sample_generated.csv").resolve()

    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(BASE_DIR)

        # 1. RUN THE PRICE ENGINE (execution.py)
        subprocess.run(
            [sys.executable, str(execution_script), "--data", str(input_data)], 
            check=True,
            cwd=str(BASE_DIR),
            env=env
        )
        print("✅ Step 1: Price Engine execution complete.")

        # 2. RUN THE COMPILER (engine_compiler.py)
        subprocess.run(
            [sys.executable, str(compiler_script)], 
            check=True,
            cwd=str(BASE_DIR),
            env=env
        )
        print("✅ Step 2: Compiler execution complete.")

        return {"status": "success", "message": "Full pipeline executed"}
        
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Script failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import os

# --- STAGE 4: Execute PD pipeline ---

@app.post("/run_pd_pipeline")
def run_pd_pipeline():
    """Stage 4: Execute PD pipeline with absolute pathing"""
    # FIX: Anchor path to BASE_DIR
    input_path = (BASE_DIR / "data" / "generated" / "final_pricing.csv").resolve()
    try:
        if not input_path.exists():
            raise FileNotFoundError(f"Missing: {input_path}")
        
        # run_full_pipeline is an imported function, ensure it receives the absolute string path
        run_full_pipeline(str(input_path))
        return {"status": "success", "message": "PD Pipeline complete"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- STAGE 5: PORTFOLIO OPTIMIZATION ---

@app.post("/calculate_portfolio_metrics")
def calculate_portfolio_metrics_endpoint():
    """Initial KPIs for the raw portfolio"""
    script_path = (BASE_DIR / "src" / "data" / "portfolio_metrics.py").resolve()
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(BASE_DIR)
        
        subprocess.run(
            [sys.executable, str(script_path)], 
            check=True,
            cwd=str(BASE_DIR),
            env=env
        )
        return {"status": "success", "message": "Metrics calculated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_alco")
def generate_alco_endpoint():
    """Triggers random ALCO limit generation"""
    script_path = (BASE_DIR / "src" / "data" / "generate_alco.py").resolve()
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(BASE_DIR)

        subprocess.run(
            [sys.executable, str(script_path)], 
            check=True,
            cwd=str(BASE_DIR),
            env=env
        )
        return {"status": "success", "message": "ALCO generated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize_portfolio")
def optimize_portfolio_endpoint(data: dict):
    """Runs the PuLP solver script"""
    script_path = (BASE_DIR / "src" / "data" / "portfolio_opt.py").resolve()
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(BASE_DIR)

        args = [sys.executable, str(script_path)]
        if data.get('mode') == 'manual':
            args.extend([str(v) for v in data['values']])
        else:
            # Ensure the path passed in the data dict is handled as absolute if relative
            provided_path = Path(data['path'])
            target_path = provided_path if provided_path.is_absolute() else (BASE_DIR / data['path']).resolve()
            args.append(str(target_path))
            
        subprocess.run(
            args, 
            check=True,
            cwd=str(BASE_DIR),
            env=env
        )
        return {"status": "success", "message": "Optimization complete"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/calculate_opt_metrics")
def calculate_opt_metrics_endpoint():
    """NEW: Aggregates KPIs for the post-optimization results"""
    script_path = (BASE_DIR / "src" / "data" / "opt_metrics.py").resolve()
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(BASE_DIR)

        subprocess.run(
            [sys.executable, str(script_path)], 
            check=True,
            cwd=str(BASE_DIR),
            env=env
        )
        return {"status": "success", "message": "Optimized metrics calculated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from src.models.markov_monitoring.execution import run_markov_monitoring_pipeline
from src.models.monte_carlo.execution import run_monte_carlo_rwa_pipeline

import os

# =================================================================
# SECTION 1: MARKOV CHAIN MODULE
# =================================================================

@app.post("/run_markov_sampler")
def run_markov_sampler():
    """Isolated: Triggers the sampler for the Markov workflow"""
    # FIX: Anchor to BASE_DIR
    script_path = (BASE_DIR / "src" / "data" / "markovian_sampler.py").resolve()
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(BASE_DIR)
        
        subprocess.run(
            [sys.executable, str(script_path)], 
            check=True,
            cwd=str(BASE_DIR),
            env=env
        )
        return {"status": "success", "message": "Markov sample generated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Markov Sampler Error: {str(e)}")

@app.get("/get_markov_matrix")
def get_markov_matrix():
    """Isolated: Retrieves matrix specifically for Markov UI"""
    # FIX: Anchor to BASE_DIR
    matrix_path = (BASE_DIR / "artifacts" / "markov_chains" / "transition_matrix.json").resolve()
    labels = ["Current", "Grace Period", "Late (16-30)", "Late (31-120)", "Default"]
    try:
        if not matrix_path.exists():
            return {"status": "error", "message": f"Matrix file not found at {matrix_path}"}
        with open(matrix_path, 'r') as f:
            data = json.load(f)
        return {"status": "success", "matrix": data["tpm"], "labels": labels}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run_markov_simulation")
def run_markov_simulation():
    """Isolated: Executes the Markov monitoring pipeline"""
    # FIX: Anchor to BASE_DIR and ensure it is an absolute string for the imported function
    data_path = (BASE_DIR / "data" / "generated" / "markovian_sample.csv").resolve()
    try:
        # Pass the absolute path string
        run_markov_monitoring_pipeline(str(data_path))
        return {"status": "success", "message": "Markov Simulation complete"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run_markov_migration")
def run_markov_migration(matrix_data: dict):
    """Isolated: Placeholder for Markov-specific rating migration"""
    return {"status": "success", "message": "Markov migration calculated"}


import os

# =================================================================
# SECTION 2: MONTE CARLO MODULE
# =================================================================

@app.post("/run_mc_sampler")
def run_mc_sampler():
    """Isolated: Triggers the sampler for the Monte Carlo workflow"""
    # FIX: Anchor to BASE_DIR
    script_path = (BASE_DIR / "src" / "data" / "markovian_sampler.py").resolve()
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(BASE_DIR)
        
        subprocess.run(
            [sys.executable, str(script_path)], 
            check=True,
            cwd=str(BASE_DIR),
            env=env
        )
        return {"status": "success", "message": "Monte Carlo sample generated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MC Sampler Error: {str(e)}")

@app.get("/get_mc_matrix")
def get_mc_matrix():
    """Isolated: Retrieves matrix specifically for Monte Carlo UI"""
    # FIX: Anchor to BASE_DIR
    matrix_path = (BASE_DIR / "artifacts" / "markov_chains" / "transition_matrix.json").resolve()
    labels = ["Current", "Grace Period", "Late (16-30)", "Late (31-120)", "Default"]
    try:
        if not matrix_path.exists():
            return {"status": "error", "message": f"Matrix file not found at {matrix_path}"}
        with open(matrix_path, 'r') as f:
            data = json.load(f)
        return {"status": "success", "matrix": data["tpm"], "labels": labels}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run_mc_simulation")
def run_mc_simulation_endpoint():
    """Isolated: Executes the MC RWA pipeline and returns JSON metrics"""
    # FIX: Anchor both paths to BASE_DIR
    DATA_PATH = (BASE_DIR / "data" / "generated" / "markovian_sample.csv").resolve()
    JSON_RESULT_PATH = (BASE_DIR / "json_files" / "monte_carlo" / "portfolio_rwa_comparison.json").resolve()
    
    try:
        # 1. Execute stochastic pipeline
        # Pass absolute path string to the imported function
        run_monte_carlo_rwa_pipeline(str(DATA_PATH))
        
        # 2. Extract results for the frontend summary
        if not JSON_RESULT_PATH.exists():
            raise FileNotFoundError(f"RWA report missing at {JSON_RESULT_PATH}")
            
        with open(JSON_RESULT_PATH, "r") as f:
            rwa_data = json.load(f)
            
        return {
            "status": "success",
            "message": "Monte Carlo RWA Simulation complete",
            "data": rwa_data["rwa_report"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MC Simulation failed: {str(e)}")