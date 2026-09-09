# ⚓ VarunaPath AI — Maritime Logistics & Fleet Optimization Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)
[![CI Status](https://img.shields.io/badge/CI-Passing-brightgreen.svg)]()
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**VarunaPath AI** is an enterprise-grade maritime decision cockpit and voyage planning intelligence system designed for dry bulk supply chains. It optimizes procurement schedules, carrier fixtures, and port discharges across Indian Ocean shipping corridors, focusing on Indian East Coast terminals (Paradip, Dhamra, Visakhapatnam, Haldia, Ennore).

---

## 🌟 Key Features & Capabilities

### 1. 🧭 Two-Level Navigation Architecture
- **Level 1: Main Landing Hub**: Intuitive operational launcher categorized into *Planning & Decision*, *Maritime Intelligence*, and *Analysis & Management*.
- **Level 2: Deep Workspaces**: Dedicated execution screens equipped with standardized control bars, unified KPI cards, and instant return navigation (`← Back to Home`).

### 2. 📊 Command Centre
- **Executive KPI Row**: Instant visibility into Cargo Requirement (150,000 t), Shortfall (130,000 t), Recommended Cost (₹28.60 Cr), Risk Score (24/100), and ETA (19 days).
- **Decision Snapshot**: 3-card strategic overview connecting Forecasts, Fixtures, and Supply Readiness.
- **Visual Analytics**: Interactive 90-day inventory burn-down and replenishment curve, plus baseline vs. optimized cost comparison.
- **Explainable AI (XAI)**: Explicit *Why This Plan?* decision rationale card.

### 3. 📈 Forecasting Studio
- **Multi-Model Predictive Engine**: Ensemble Recommended, Linear Regression, Moving Average, and Seasonal Trend models.
- **Horizons**: Flexible projections across 30 Days, 60 Days, 90 Days, 120 Days, 6 Months, and 12 Months.
- **High-Precision Calibration**: Thermal Coal default achieves a **0.82% MAPE** with a **90% Confidence Score**.

### 4. 🚢 Shipment Planner (4-Step Voyage Architect)
- **Step 1 — Requirement**: Automated shortfall formula ($\text{Shortfall} = \text{Req} + \text{Safety} - \text{Inv}$).
- **Step 2 — Constraints**: User-defined risk thresholds, budgets, arrival windows, vessel types, and discharge ports.
- **Step 3 — Generated Plans**: Tri-plan comparison (*Lowest Cost*, *Lowest Risk*, and *Balanced Recommended*).
- **Strict Vessel Feasibility**: Enforces capacity constraints—rejects 2 × Supramax (116,000 t) for 130,000 t shortfall with explicit deficit warnings.
- **Step 4 — Commitment**: One-click fixture confirmation and plan export (CSV).

### 5. ⚡ Optimization Hub
- **Multi-Constraint Optimization**: Analyzes 33 feasible combinations across global bulk suppliers, vessel classes, and port drafts.
- **Metric Row**: Recommended Port (Paradip), Vessel Plan (2 × Panamax), Cost (₹28.60 Cr), Savings (₹2.59 Cr / 8.3%), and Feasible Plans (33 Solutions).

### 6. ⚓ Vessel & Port Intelligence
- **Vessel Silhouettes**: Dynamic inline SVG bulk carrier silhouettes (Handysize, Supramax, Panamax, Capesize) with DWT scale specifications and hatch layouts.
- **Shipping Corridor Simulation**: Interactive pseudo-3D animated sea-lane route from Richards Bay (ZA) to East Coast India.
- **Draft & Turnaround Telemetry**: Master port data covering draft clearances up to 180,000 DWT and waiting times.

### 7. 🧪 Scenario Lab & Risk Engine
- **Disruption Simulator**: Interactive what-if stress tests for fuel price surges, freight rate shifts, weather volatility, and port outages.
- **Risk Assessment**: Multi-pillar risk rating synthesizing supplier geopolitical risk, vessel reliability, and port congestion.

### 8. 📑 Certified Reports & Data Connectors
- **Certified Audit Documentation**: Executive summaries, cost breakdowns, and port comparative audits.
- **Synchronized State**: Seamlessly audits confirmed fixtures from Shipment Planner or defaults to prototype recommendations.

---

## 📐 Official Certified Baseline Benchmark

| Parameter | Baseline Value | Units / Notes |
| :--- | :--- | :--- |
| **Commodity** | Thermal Coal | Richards Bay, South Africa (ZA) |
| **Planning Horizon** | 90 Days | 3 Months |
| **Cargo Requirement** | 150,000 | Tonnes |
| **Current Stockpile** | 40,000 | Tonnes |
| **Safety Buffer** | 20,000 | Tonnes |
| **Procurement Shortfall** | **130,000** | Tonnes net procurement need |
| **Optimal Vessel Class** | **2 × Panamax** | 82,000 t capacity / vessel |
| **Combined Capacity** | **164,000** | Tonnes (79.3% fleet utilization) |
| **Discharge Port** | **Paradip Port** | 180,000 DWT draft clearance |
| **Turnaround Time** | **19 Days** | 16d steaming + 3d waiting (≤ 45d SLA) |
| **Baseline Cost** | **₹31.19 Cr** | Conventional single-fixture spot baseline |
| **Optimized Total Cost** | **₹28.60 Cr** | Least-cost multi-constraint solution |
| **Direct Financial Savings**| **₹2.59 Cr** | **8.3% net cost reduction** |
| **Composite Risk Score** | **24 / 100** | Low Risk category (Tier 1 rating) |
| **Model Confidence** | **90%** | MAPE: 0.82% |
| **Feasible Voyage Plans** | **33** | Constraint-verified combinations |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/utkarsh0204web/VarunaPath-AI.git
   cd VarunaPath-AI
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the application**:
   ```bash
   # Windows launcher
   START_VARUNAPATH.bat

   # Or direct CLI
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

---

## 🧪 Running Unit Tests

Execute the automated test suite to verify calculation integrity and feasibility logic:
```bash
python -m unittest discover tests/
```

---

## 📂 Repository Structure

```
VarunaPath-AI/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI automated testing
├── .streamlit/
│   └── config.toml              # Streamlit maritime theme configuration
├── data/
│   ├── sample_cargo_demand.csv  # Sample historical demand dataset
│   └── ports_master.csv         # Master port specifications & tariffs
├── docs/
│   └── ARCHITECTURE.md          # Technical specifications & algorithm details
├── tests/
│   └── test_calculations.py     # Unit test suite for formulas & constraints
├── app.py                       # Main VarunaPath AI Streamlit Web Application
├── requirements.txt             # Pinned project dependencies
├── START_VARUNAPATH.bat         # One-click Windows launch script
├── .gitignore                   # Git ignore rules for clean repository
├── LICENSE                      # MIT Open Source License
└── README.md                    # Project documentation & benchmark guide
```

---

## 🛡️ License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

**VarunaPath AI** &bull; Engineered by **Team Novara**
