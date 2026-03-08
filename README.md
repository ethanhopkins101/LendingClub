# 🏦 CreditRisk AI — End-to-End Risk Management Engine

<p align="center">
    <picture>
        <source media="(prefers-color-scheme: light)" srcset="https://img.shields.io/badge/Status-Live-brightgreen?style=for-the-badge">
        <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit" alt="Streamlit">
    </picture>
    <picture>
        <img src="https://img.shields.io/badge/Math-Quantitative_Finance-blue?style=for-the-badge" alt="Quantitative Finance">
    </picture>
</p>

<p align="center">
  <strong>A multi-stage Credit Risk Pipeline: From automated underwriting to Basel-compliant stress testing.</strong>
</p>

---

## 📖 Project Overview
CreditRisk AI is an enterprise-grade risk modeling suite designed to handle the full lifecycle of a loan. It replaces traditional, siloed decision-making with an integrated pipeline that optimizes for both **profit maximization** and **regulatory compliance (Basel III/IV)**.

The system utilizes six specialized models to ensure that every stage of the lending process—from the moment a customer walks in to the annual regulatory reporting—is backed by statistical certainty.

---

## 🛠 The Quantitative Modeling Pipeline: A Multi-Stage Risk Framework

The CreditRisk AI engine utilizes a decoupled, six-stage architecture to ensure that every phase of the credit lifecycle is governed by statistically sound decisioning. Unlike "black-box" models, this pipeline is designed for **high interpretability** and **regulatory auditability**.

---

### 1. Stage I: Automated Underwriting & Initial Review (The Gatekeeper)
The first line of defense is a high-precision classification layer designed to filter out high-risk "fat-tail" losses before they reach the pricing engine. This stage acts as a binary/ternary decisioning tool (**Accept / Reject / Refer**).

* **Algorithmic Architecture:** Gradient Boosted Decision Trees (XGBoost) configured with **Monotonic Constraints**. This ensures that as risk indicators (like Debt-to-Income or Delinquency Count) increase, the Probability of Default (PD) is mathematically forced to follow a non-decreasing trend.
* **Feature Importance & SHAP Analysis:** We utilize SHAP (SHapley Additive exPlanations) to ensure "Right to Explanation" compliance, allowing the bank to provide specific reasons for adverse actions (denials).
* **Operational Focus:** Eliminating systemic risk and "hard-stop" policy violations (e.g., LTV > 100% or minimum FICO thresholds) to optimize the downstream computational load.



---

### 2. Stage II: Risk-Adjusted Pricing Engine (RAP)
Traditional lending uses flat-rate "buckets." This engine implements **Risk-Based Pricing (RBP)**, where every basis point of interest is justified by the borrower's specific risk profile.

* **The Quantitative Pricing Formula:**
    $$R = R_{f} + CR + OC + k \cdot EC$$
    * **$R_{f}$ (Risk-Free Rate):** The base cost of funds (e.g., SOFR or 10-Year Treasury).
    * **$CR$ (Credit Risk Premium):** The spread required to cover the Expected Loss (EL) of the specific risk decile.
    * **$OC$ (Operational Costs):** The fixed and variable costs associated with loan servicing and origination.
    * **$k \cdot EC$ (Economic Capital Charge):** The hurdle rate required to provide a return on the capital held against the loan's Unexpected Loss (UL).
* **Dynamic Decile Assignment:** Borrowers are ranked by their percentile risk. The engine ensures a consistent **Risk-Adjusted Return on Capital (RAROC)** across all segments, ensuring the bank is equally profitable on a "High-Risk/High-Yield" loan as it is on a "Low-Risk/Low-Yield" loan.



---

### 3. Stage III: Loss Given Default (LGD) & Recovery Analytics
Predicting the severity of a loss is often more volatile than predicting the event itself. LGD modeling accounts for the value of collateral and the costs of the collection process.

* **Stochastic Modeling Approach:** Since recovery rates are typically bimodal (recovering either the full amount or zero), we avoid standard OLS regression in favor of **Beta Regression** or **Tobit Models** to handle the [0, 1] bounded nature of recovery data.
* **Recovery Discounting:** The model calculates the **Economic LGD**, which discounts recovered cash flows back to the date of default using the loan's original effective interest rate.
* **Formula for Economic LGD:**
    $$LGD = 1 - \frac{\sum_{t=1}^{n} \frac{\text{Recoveries}_{t} - \text{Collection Costs}_{t}}{(1+r)^{t}}}{EAD}$$
* **Downturn LGD (DLGD):** Incorporates a "stress factor" to account for the correlation between high default rates and low collateral values (e.g., a housing market crash).



---

### 4. Stage IV: PD & EAD Engine (The Basel Pillars)
This is the core of the **Internal Ratings-Based (IRB)** approach, aligning the project with **Basel III/IV** and **IFRS 9 / CECL** accounting standards.

#### A. Probability of Default (PD) via Weight of Evidence (WoE)
* **Feature Transformation:** We utilize a sophisticated **Binning Process**. Raw variables are transformed into **Weight of Evidence (WoE)** values to ensure a linear relationship with the log-odds of default.
* **Information Value (IV):** Features are selected based on their IV to ensure only the most predictive variables enter the model, preventing over-fitting.
* **Calibration:** Raw model outputs are calibrated using **Platt Scaling** or **Isotonic Regression** to ensure that a predicted PD of 5.0% actually results in a 5.0% historical default rate.

#### B. Exposure at Default (EAD) & CCF
* **Credit Conversion Factor (CCF):** For revolving lines of credit, we model the probability that a borrower will "draw down" their remaining limit as they approach a default event.
* **Formula for EAD:**
    $$EAD = \text{Current Balance} + (\text{Undrawn Limit} \times CCF)$$

#### C. Expected Loss (EL) Integration
The final output of this stage is the **Expected Loss**, which directly dictates the bank's "Provisions for Credit Losses" on the balance sheet.
$$EL = PD \times LGD \times EAD$$

### 5. Credit Scoring
Translating probabilities into a standardized score (300-850 range). 
* **Scaling:** $Score = Offset + Factor \times \ln(\text{odds})$
* Ensures that every 20 points increase in score represents a doubling of the odds of repayment.

---

## ⚖️ Portfolio Optimization (PuLP)
Once individual risks are calculated, the system shifts to **Constrained Profit Maximization**. We treat the loan portfolio as an investment basket that must stay within **ALCO (Asset-Liability Committee)** limits.

**Objective Function:**
Maximize $\sum (\text{Interest Income}_i - EL_i)$

**Constraints:**
* **Total Capital:** $\sum EAD_i \leq \text{Available Liquidity}$
* **Risk Tolerance:** $\text{Portfolio PD}_{avg} \leq \text{Target PD}$
* **Regulatory Floor:** $\sum RWA_i \times 8\% \leq \text{Total Equity}$

---
## 📈 Portfolio Monitoring & Systemic Resilience

Static models fail when markets move. The CreditRisk AI suite employs dynamic monitoring to track **Credit Migration** and perform **Capital Adequacy Stress Testing** under non-linear economic conditions.

---

### 1. Markov Chain Migration Analysis (Credit Drift)
Rather than a "snapshot" of current risk, we utilize **Migration Analysis** to track the velocity of credit deterioration. By modeling the probability of a loan moving between "buckets" (e.g., AAA to BBB, or Current to Default), the bank can forecast future provisioning needs.

* **Transition Probability Matrix ($P$):** We construct a $n \times n$ matrix where each element $p_{ij}$ represents the probability of a borrower transitioning from state $i$ to state $j$ within a specific time horizon (typically 3–12 months).
    $$S_{t+1} = S_{t} \cdot P$$
* **Absorbing States:** The "Default" and "Recovered" states are treated as absorbing, allowing us to calculate the **Mean Time to Default (MTTD)** for the current portfolio.
* **Early Warning System (EWS):** By identifying a statistical increase in the "Current-to-30DPD" (30 Days Past Due) transition rate, the bank can trigger proactive restructuring or limit reductions before a loss event occurs.



---

### 2. Monte Carlo Stress Testing (The Survival Check)
To comply with **Basel III/IV** and **CCAR (Comprehensive Capital Analysis and Review)**, we perform 10,000+ stochastic simulations to assess how the portfolio reacts to "Black Swan" events.

* **Stochastic Variables:** We simulate shocks to the **Unemployment Rate**, **Interest Rates**, and **GDP Growth**, observing how these macro factors correlate with $PD$ and $LGD$ spikes.
* **Tail Risk Metrics:**
    * **Value at Risk (VaR):** The maximum loss expected over a given time horizon at a $99.9\%$ confidence level.
    * **Expected Shortfall (ES):** Also known as Conditional VaR (CVaR), this measures the average loss in the *tail*—specifically, the average loss if the VaR threshold is breached.
* **Capital Buffer Analysis:**
    The engine compares the simulated losses against the bank's **CET1 (Common Equity Tier 1)** capital. If the 1-in-100-year loss exceeds current equity, the system alerts the **ALCO** to increase capital reserves.



---

### 3. RWA & Regulatory Compliance (Basel Framework)
The ultimate output of the resilience module is the calculation of **Risk-Weighted Assets (RWA)**. This determines exactly how much capital the bank must legally hold to remain solvent.

* **The Regulatory Capital Formula (K):**
    Following the **A-IRB (Advanced Internal Ratings-Based)** approach:
    $$K = \left[ LGD \times N\left( \frac{G(PD)}{\sqrt{1-R}} + \sqrt{\frac{R}{1-R}} G(0.999) \right) - (PD \times LGD) \right] \times \text{Maturity Adjustment}$$
    * Where $G(PD)$ is the inverse of the normal distribution, and $R$ is the asset correlation.
* **Strategic Utility:** By optimizing the portfolio to reduce RWA, the bank can "free up" capital for more profitable lending activities, directly improving the **ROE (Return on Equity)**.
---

## 🚀 Technical Stack
* **Logic:** Python (Pandas, NumPy, Scikit-Learn, XGBoost)
* **Optimization:** PuLP (Linear Programming)
* **Simulation:** SciPy (Monte Carlo), Markov Models
* **Interface:** Streamlit (Custom CSS/UX)
* **Deployment:** Docker / Hugging Face Spaces

---

1.  **Basel III/IV Context:** Mentioning RWA (Risk-Weighted Assets) and the 8% Capital Adequacy Ratio makes the project sound much more realistic for banking.
2.  **LaTeX Math:** Added formal notation for Expected Loss ($EL = PD \times LGD \times EAD$) and Scoring logic.
3.  **ALCO Reference:** In banking, the Asset-Liability Committee (ALCO) sets the limits—referencing them shows deep industry knowledge.
4.  **Wait of Evidence (WoE):** Mentioning "Binning Process" specifically as WoE is the industry standard for PD modeling.
5.  **Visual Structure:** Used clean headers, emojis, and shields to mimic the aesthetic of the high-quality README you shared.


## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/LendingClub.git](https://github.com/your-username/LendingClub.git)
   cd LendingClub