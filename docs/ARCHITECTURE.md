# VarunaPath AI — System Architecture & Technical Specifications

VarunaPath AI is an enterprise-grade maritime bulk logistics decision cockpit and voyage planning engine. It addresses the end-to-end supply chain lifecycle for dry bulk commodities (Thermal Coal, Coking Coal, Iron Ore, Bauxite, Limestone, Grain, Cement, and Petroleum Coke) imported via Indian East Coast discharge terminals.

---

## 1. Two-Level Navigation Architecture

The system decouples high-level operational discovery from deep technical modules using a resilient two-level hierarchy:

```
[Level 1: Main Landing Hub]
  │
  ├── Category 1: Planning & Decision Workspaces
  │     ├── 📊 Command Centre (Executive KPIs & Decision Snapshot)
  │     ├── 📈 Forecasting Studio (Multi-Model AI Demand Prediction)
  │     └── 🚢 Shipment Planner (4-Step Voyage Architect)
  │
  ├── Category 2: Maritime Intelligence
  │     ├── ⚡ Optimization Hub (Multi-Constraint Fleet Allocation)
  │     ├── ⚓ Vessel Intelligence (Silhouettes, DWT Specs & Utilization)
  │     ├── 🏗️ Port Intelligence (Draft Clearance, Tariffs & Turnaround)
  │     └── ⚠️ Risk & Alerts (Geopolitical, Seaworthiness & Weather)
  │
  └── Category 3: Analysis & Management
        ├── 🧪 Scenario Lab (What-If Disruption Simulator)
        ├── 📑 Reports (Certified Audit Documentation & Exports)
        └── ⚙️ Data & Settings (Connectors & Model Parameters)
```

- **Seamless State Persistence**: Navigation is driven strictly through `st.session_state["active_page"]`, guaranteeing that scenario selections, inventory states, and confirmed plans persist without fragile URL hash parameters.
- **Immediate Re-routing**: Every Level 2 workspace includes a prominent `← Back to Home` control alongside sidebar synchronization.

---

## 2. Mathematical Formulation & Core Algorithms

### 2.1 Cargo Procurement Equation
Net bulk procurement is governed by safety stock thresholding:
$$\text{Cargo Shortfall} = \max(\text{Cargo Requirement} + \text{Safety Stock} - \text{Current Inventory}, 0)$$

*Baseline Default*:
$$150,000 + 20,000 - 40,000 = 130,000\text{ tonnes}$$

### 2.2 Fleet Capacity & Utilization
Carrier allocation evaluates deadweight tonnage (DWT) limits:
$$\text{Combined Capacity} = \text{Vessel Count} \times \text{Vessel Capacity}$$
$$\text{Capacity Utilization (\%)} = \left( \frac{\text{Cargo Shortfall}}{\text{Combined Capacity}} \right) \times 100$$

### 2.3 Strict Vessel Feasibility Rule
$$\text{Feasible} \iff \text{Combined Capacity} \ge \text{Cargo Shortfall}$$
- **2 × Supramax (116,000 t)**: strictly **Infeasible** (Deficit: 14,000 tonnes).
- **2 × Panamax (164,000 t)**: strictly **Feasible** (79.3% utilization).
- **1 × Capesize (180,000 t)**: strictly **Feasible** (72.2% utilization).

### 2.4 Multi-Factor Voyage Costing
Logistics expenditure includes cargo FOB, vessel chartering, port handling, anchorage waiting, capacity under-utilization penalty, and operational risk reserves:
$$C_{\text{total}} = C_{\text{cargo}} + C_{\text{charter}} + C_{\text{port}} + C_{\text{waiting}} + P_{\text{under\_util}} + R_{\text{reserve}}$$

- **Baseline Benchmark**: ₹31.19 Cr
- **Optimized Recommended Cost**: ₹28.60 Cr
- **Direct Financial Savings**: ₹2.59 Cr (8.3% cost reduction)

---

## 3. Technology Stack

- **Frontend / Application Engine**: Streamlit (Python)
- **Visualizations**: Plotly Express & Plotly Graph Objects, Native Inline SVG
- **Forecasting & ML**: Scikit-Learn (Linear Regression), Seasonal Trend Decomposition, Moving Averages
- **Data Manipulation**: Pandas, NumPy
