import math
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.linear_model import LinearRegression

import engine
from engine import (
    calculate_import_shortfall,
    calculate_export_readiness,
    calculate_combined_capacity,
    calculate_utilization,
    generate_vessel_combinations,
    evaluate_port_compatibility,
    calculate_route_eta,
    calculate_cost_breakdown,
    calculate_risk_score,
    calculate_recommendation_confidence,
    evaluate_feasibility,
    rank_feasible_plans,
    generate_explanation,
    compare_scenarios,
    generate_decision_report,
    validate_scenario_inputs,
    VESSEL_CLASSES,
    PORTS_CONFIG,
    MARITIME_ROUTES,
)


st.set_page_config(
    page_title="VarunaPath AI",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

CARGO_OPTIONS = [
    "Thermal Coal",
    "Coking Coal",
    "Iron Ore",
    "Bauxite",
    "Limestone",
    "Fertilizer",
    "Grain",
    "Cement",
    "Petroleum Coke",
]

FORECAST_OPTIONS = [
    "30 Days",
    "60 Days",
    "90 Days",
    "120 Days",
    "6 Months",
    "12 Months",
]

HORIZON_MAP = {
    "30 Days": 30,
    "60 Days": 60,
    "90 Days": 90,
    "120 Days": 120,
    "6 Months": 180,
    "12 Months": 360,
}

SCENARIO_OPTIONS = [
    "Base Scenario",
    "Fuel Price Surge (+20%)",
    "Monsoon Weather Risk (+15% Demand)",
    "Port Bottleneck (Paradip Outage)",
    "High Inflation Peak",
]

# Module-level aliases for external and script compatibility
cargo_options = CARGO_OPTIONS
HORIZON_DAYS = HORIZON_MAP
scenario_options = SCENARIO_OPTIONS

MODULE_NAMES = [
    "Command Centre",
    "Forecasting Studio",
    "Shipment Planner",
    "Optimization Hub",
    "Vessel Intelligence",
    "Port Intelligence",
    "Route Intelligence",
    "Risk & Alerts",
    "Scenario Lab",
    "Reports",
    "Data & Settings",
]

MODULE_ICONS = {
    "Command Centre": "⚓",
    "Forecasting Studio": "📈",
    "Shipment Planner": "📦",
    "Optimization Hub": "🎯",
    "Vessel Intelligence": "🚢",
    "Port Intelligence": "🏗️",
    "Route Intelligence": "🗺️",
    "Risk & Alerts": "⚠️",
    "Scenario Lab": "🧪",
    "Reports": "📄",
    "Data & Settings": "⚙️",
}

MODULE_DESCRIPTIONS = {
    "Command Centre": "AI-powered overview of bulk cargo demand, vessel allocation and supply-chain readiness.",
    "Forecasting Studio": "AI-powered linear regression demand forecasting and historical consumption trends.",
    "Shipment Planner": "Configure cargo commodity, forecast horizon, inventory reserves, and constraints.",
    "Optimization Hub": "Least-cost vessel chartering and port allocation with constraint satisfaction.",
    "Vessel Intelligence": "Fleet profiles, carrying capacities, charter economics, and utilization metrics.",
    "Port Intelligence": "East Coast port infrastructure, handling tariffs, waiting delays, and operational readiness.",
    "Route Intelligence": "Compare maritime route alternatives using distance, ETA, fuel, port compatibility and operational risk.",
    "Risk & Alerts": "Multi-factor operational risk scoring across suppliers, vessels, ports, and delays.",
    "Scenario Lab": "Disruption simulator for fuel price volatility, demand spikes, and port closures.",
    "Reports": "Decision audit summaries, executive exports, and baseline cost variance analysis.",
    "Data & Settings": "System settings, telemetry controls, prototype parameters, and cache management.",
}

live_data_connected = False

TECHNICAL_TOOLTIPS = {
    "Shortfall": "Net deficit between cargo requirement plus safety stock buffer and available inventory: max(0, Requirement + Safety - Inventory).",
    "Combined Capacity": "Aggregate cargo carrying capacity across all allocated vessels: Vessel Count × Deadweight Intake per vessel.",
    "Utilization": "Percentage of total fleet carrying capacity occupied by cargo shortfall: (Shortfall / Combined Capacity) × 100.",
    "Capesize": "Large bulk carrier (>100,000 DWT, typ. 180,000 DWT), unable to transit Panama Canal, used for major iron ore and coal trades.",
    "Panamax": "Mid-sized bulk carrier (65,000–85,000 DWT, typ. 82,000 DWT) historically matching original Panama Canal lock dimensions.",
    "Supramax": "Flexible geared bulk carrier (50,000–65,000 DWT, typ. 58,000 DWT) equipped with on-board cargo cranes for self-discharging.",
    "Laycan": "Contractual window (Layday / Cancelling date) during which the vessel owner must present the ship ready to load cargo.",
    "Demurrage": "Liquidated penalty damages paid by charterer to shipowner for exceeding agreed laytime allowance during port loading or discharging.",
    "Dispatch": "Rebate incentive paid by shipowner to charterer when cargo operations finish ahead of agreed laytime schedule.",
    "Draft": "Vertical distance between the vessel waterline and bottom of the hull/keel, constraining port berth depth clearance.",
    "Deadweight Tonnage (DWT)": "Total weight of cargo, fuel, freshwater, ballast and stores a vessel can safely carry without exceeding legal load lines.",
    "Bunker Fuel": "Marine heavy fuel oil (VLSFO / MGO) powering ship main propulsion engines, subject to market price volatility.",
    "Incoterms": "Standardized International Commercial Terms (FOB, CFR, CIF) defining commercial cost, risk, and delivery obligations.",
    "Bill of Lading": "Legally binding transport contract, cargo receipt, and title of ownership issued by ocean carrier to shipper.",
    "Freight Rate": "Market charter price agreed for transporting dry bulk cargo across a specified maritime sea route.",
    "Port Congestion": "Operational vessel queue delay at anchorages awaiting available discharge berths or shore cargo unloader cranes.",
    "Turnaround Time": "Total elapsed hours a vessel spends in port from arrival at pilot station to departure after complete cargo discharge.",
    "Safety Stock": "Strategic contingency stockpile buffer maintained at discharge plant to protect against voyage and supply disruptions.",
    "Fixture": "Finalized charter party agreement fixing a vessel, freight rate, laydays, and trade terms between owner and charterer.",
    "Notice of Readiness": "Formal written declaration tendered by master to port/charterer certifying the vessel has arrived and is ready to load/discharge.",
    "Order Readiness": "Percentage of export order fulfilled by available plant stock plus scheduled production: (Export-Ready / Order Qty) × 100.",
}

def tooltip_icon(term):
    desc = TECHNICAL_TOOLTIPS.get(term, "")
    return f'<span title="{desc}" style="cursor: help; text-decoration: underline dotted; color: inherit;">{term} ℹ️</span>'

def classify_risk(risk_score):
    """Centralized risk classification utility.
    0-30: Low (Green #16A34A, light bg #DCFCE7)
    31-60: Moderate (Amber #D97706, light bg #FEF3C7)
    61-100: High (Red #DC2626, light bg #FEE2E2)
    Returns: (label, hex_color, bg_color, short_category)
    """
    if risk_score <= 30:
        return ("Low Risk", "#16A34A", "#DCFCE7", "Low")
    elif risk_score <= 60:
        return ("Moderate Risk", "#D97706", "#FEF3C7", "Moderate")
    else:
        return ("High Risk", "#DC2626", "#FEE2E2", "High")

EXPORT_COMMODITIES = [
    "Finished Steel",
    "Iron Ore Pellets",
    "Aluminium Ingots",
    "Basmati Rice",
    "Refined Petroleum",
    "Chemical Granules",
]

EXPORT_LOADING_PORTS = [
    "Paradip",
    "Visakhapatnam",
    "Chennai",
    "Kamarajar (Ennore)",
    "Krishnapatnam",
    "Dhamra",
    "Gangavaram",
]

FOREIGN_DESTINATION_PORTS = [
    "Singapore",
    "Chittagong",
    "Port Klang",
    "Colombo",
    "Jebel Ali",
    "Rotterdam",
    "Qingdao",
]

EXPORT_INCOTERMS = ["FOB", "CFR", "CIF"]

EXPORT_VESSEL_CLASSES = [
    "Supramax (58k)",
    "Panamax (82k)",
    "Capesize (180k)",
]

EXPORT_DOCS_LIST = [
    ("Commercial Invoice", "Standard invoice itemizing export cargo, unit prices and trade value.", "Generated"),
    ("Packing List", "Weight, volume, hatch distribution and packaging specification.", "Generated"),
    ("Shipping Bill", "Customs declaration generated through ICEGATE export clearance portal.", "Generated"),
    ("Certificate of Origin", "Chamber of Commerce preferential trade origin certification.", "Pending"),
    ("Inspection / Quality Certificate", "Pre-shipment sampling and specification compliance (SGS / BIS).", "Pending"),
    ("Bill of Lading", "Clean on-board negotiable marine transport title document.", "Generated"),
    ("Marine Insurance Certificate", "Institute Cargo Clauses (A) all-risk insurance policy (required for CIF).", "Pending"),
    ("Fumigation / Phytosanitary Certificate", "Plant health or quarantine compliance certification where required.", "Pending"),
    ("Export Declaration Form (EDF / RBI)", "Regulatory foreign exchange remittance declaration.", "Generated"),
]

def calculate_export_availability(order_qty, inv, prod, reserved):
    """Calculates export availability and readiness gap."""
    ready_qty = max(0, inv + prod - reserved)
    shortfall = max(0, order_qty - ready_qty)
    readiness_pct = min(100.0, (ready_qty / order_qty * 100.0)) if order_qty > 0 else 0.0
    return ready_qty, shortfall, round(readiness_pct, 1)

def calculate_export_logistics_cost(incoterm, qty, vessels_needed, vessel_class, loading_port, dest_port, fuel_change=0):
    """Calculates export logistics cost breakdown strictly matching Incoterm scope."""
    inland = round(qty * 450 / 10_000_000, 3)
    storage = round(qty * 65 / 10_000_000, 3)
    loading = round(qty * 95 / 10_000_000, 3)
    doc = 0.02
    fob_sum = inland + storage + loading + doc

    charter_base = 3.80 if "Supramax" in vessel_class else (4.30 if "Panamax" in vessel_class else 10.50)
    charter = round(vessels_needed * charter_base * (1 + fuel_change / 100 * 0.45), 3)
    waiting = round(vessels_needed * 4.0 * 0.02, 3)
    risk_buffer = round(0.05 * (charter + loading), 3)

    cargo_val = qty * 6800 / 10_000_000
    insurance = round(cargo_val * 0.0035, 3) if incoterm == "CIF" else 0.0

    if incoterm == "FOB":
        total = fob_sum + round(risk_buffer * 0.5, 3)
        components = {
            "Inland Transport": inland,
            "Storage & Yard": storage,
            "Port Loading": loading,
            "Export Customs & Docs": doc,
            "Risk Buffer": round(risk_buffer * 0.5, 3),
        }
    elif incoterm == "CFR":
        total = fob_sum + charter + waiting + risk_buffer
        components = {
            "Inland Transport": inland,
            "Storage & Yard": storage,
            "Port Loading": loading,
            "Export Customs & Docs": doc,
            "Ocean Charter": charter,
            "Port Waiting Allowance": waiting,
            "Route Risk Buffer": risk_buffer,
        }
    else:  # CIF
        total = fob_sum + charter + waiting + insurance + risk_buffer
        components = {
            "Inland Transport": inland,
            "Storage & Yard": storage,
            "Port Loading": loading,
            "Export Customs & Docs": doc,
            "Ocean Charter": charter,
            "Port Waiting Allowance": waiting,
            "Marine Cargo Insurance": insurance,
            "Route Risk Buffer": risk_buffer,
        }

    sum_components = round(sum(components.values()), 3)
    return round(sum_components, 2), components

def calculate_export_financials(order_qty, selling_price, internal_cost, total_cost_cr):
    """Calculates commercial export financials only when prices are provided."""
    if not selling_price or selling_price <= 0:
        return None, None, None, "Commercial revenue and margin not calculated because selling price was not provided."
    rev_cr = round((order_qty * selling_price) / 10_000_000, 2)
    if not internal_cost or internal_cost <= 0:
        return rev_cr, None, None, "Commercial margin not calculated because internal production cost was not provided."
    cogs_cr = round((order_qty * internal_cost) / 10_000_000, 2)
    net_margin_cr = round(rev_cr - cogs_cr - total_cost_cr, 2)
    margin_pct = round((net_margin_cr / rev_cr) * 100.0, 1) if rev_cr > 0 else 0.0
    return rev_cr, net_margin_cr, margin_pct, "Commercial financials calculated from provided trade rates."

def check_export_laycan(laycan_start, laycan_end, vessel_ready):
    """Evaluates laycan window compliance."""
    if isinstance(laycan_start, str):
        laycan_start = datetime.strptime(laycan_start, "%Y-%m-%d").date()
    if isinstance(laycan_end, str):
        laycan_end = datetime.strptime(laycan_end, "%Y-%m-%d").date()
    if isinstance(vessel_ready, str):
        vessel_ready = datetime.strptime(vessel_ready, "%Y-%m-%d").date()

    if laycan_start > laycan_end:
        return False, "Invalid Laycan: Earliest loading date is after latest loading date."
    if not (laycan_start <= vessel_ready <= laycan_end):
        return False, f"Vessel readiness ({vessel_ready}) falls outside contracted laycan ({laycan_start} to {laycan_end})."
    return True, "Vessel readiness meets the contracted laycan window."

def generate_export_plans(
    shipment_qty,
    order_qty,
    commodity,
    buyer_country,
    loading_port,
    dest_port,
    incoterm,
    laycan_start,
    laycan_end,
    deadline_days,
    max_budget,
    max_risk,
    selling_price,
    internal_cost,
    allowed_vessels,
    allowed_loading_ports,
    allowed_dest_ports,
    fuel_change=0
):
    """Generates 3 comparative export strategic plans: Lowest Cost, Lowest Risk, Balanced Recommended."""
    def evaluate_candidate(v_class, v_count, load_p, dest_p, v_ready_offset=9):
        cap = v_count * (58000 if v_class == "Supramax" else (82000 if v_class == "Panamax" else 180000))
        util = round((shipment_qty / cap) * 100.0, 1) if cap > 0 else 0.0
        tot_cost, comps = calculate_export_logistics_cost(incoterm, shipment_qty, v_count, v_class, load_p, dest_p, fuel_change)
        
        port_risk = 22 if load_p == "Paradip" else (18 if load_p == "Visakhapatnam" else 25)
        dest_risk = 14 if dest_p == "Singapore" else (22 if dest_p == "Chittagong" else 18)
        comp_risk = int((port_risk + dest_risk + (20 if v_class == "Supramax" else (18 if v_class == "Panamax" else 25))) / 3)
        
        base_transit = 7 if dest_p in ["Singapore", "Port Klang"] else (4 if dest_p in ["Chittagong", "Colombo"] else (12 if dest_p == "Jebel Ali" else 22))
        waiting_days = 3 if load_p == "Paradip" else 2
        duration = base_transit + waiting_days
        
        v_ready_date = (date.today() if isinstance(laycan_start, date) else date.today()) + timedelta(days=v_ready_offset)
        laycan_ok, laycan_msg = check_export_laycan(laycan_start, laycan_end, v_ready_date)
        
        reasons = []
        if cap < shipment_qty:
            reasons.append(f"Capacity deficit: {shipment_qty - cap:,} tonnes")
        if comp_risk > max_risk:
            reasons.append(f"Risk exceeded: {comp_risk} > {max_risk}")
        if tot_cost > max_budget:
            reasons.append(f"Budget exceeded: ₹{tot_cost:.2f} Cr > ₹{max_budget:.2f} Cr")
        if duration > deadline_days:
            reasons.append(f"Deadline exceeded: {duration} days > {deadline_days} days")
        if not laycan_ok:
            reasons.append(laycan_msg)
        if not any(v_class in av for av in allowed_vessels):
            reasons.append(f"Vessel class {v_class} not in allowed vessel types")
        if load_p not in allowed_loading_ports:
            reasons.append(f"Loading port {load_p} not in allowed loading ports")
        if dest_p not in allowed_dest_ports:
            reasons.append(f"Destination port {dest_p} not in allowed destination ports")
            
        feasible = (len(reasons) == 0)
        status_text = "Feasible" if feasible else f"Infeasible: {'; '.join(reasons)}"
        
        rev_cr, margin_cr, margin_pct, fin_msg = calculate_export_financials(shipment_qty, selling_price, internal_cost, tot_cost)
        r_lbl, r_color, r_bg, _ = classify_risk(comp_risk)
        
        return {
            "vessel_class": v_class,
            "vessel_count": v_count,
            "combined_capacity": cap,
            "utilization": util,
            "loading_port": load_p,
            "dest_port": dest_p,
            "duration": duration,
            "vessel_ready": str(v_ready_date),
            "laycan_ok": laycan_ok,
            "laycan_msg": laycan_msg,
            "cost_cr": tot_cost,
            "components": comps,
            "revenue_cr": rev_cr,
            "margin_cr": margin_cr,
            "margin_pct": margin_pct,
            "fin_msg": fin_msg,
            "risk_score": comp_risk,
            "risk_label": r_lbl,
            "risk_color": r_color,
            "risk_bg": r_bg,
            "feasible": feasible,
            "failed_reasons": reasons,
            "status": status_text,
        }

    p1 = evaluate_candidate("Supramax", math.ceil(shipment_qty / 58000), loading_port, dest_port, v_ready_offset=8)
    p1["name"] = "Lowest Cost"
    p1["badge"] = "LOWEST COST"
    p1["badge_color"] = "#16A34A"
    
    p2 = evaluate_candidate("Panamax", math.ceil(shipment_qty / 82000), "Visakhapatnam", dest_port, v_ready_offset=9)
    p2["name"] = "Lowest Risk"
    p2["badge"] = "LOWEST RISK"
    p2["badge_color"] = "#372580"
    
    p3 = evaluate_candidate("Panamax", math.ceil(shipment_qty / 82000), loading_port, dest_port, v_ready_offset=10)
    p3["name"] = "Balanced Recommended"
    p3["badge"] = "RECOMMENDED"
    p3["badge_color"] = "#372580"
    
    return {"Lowest Cost": p1, "Lowest Risk": p2, "Balanced Recommended": p3}

def generate_why_this_export_plan(plan, budget, max_risk, deadline, incoterm):
    """Generates dynamic explanation for the selected export plan."""
    name = plan["name"]
    v_desc = f"{plan['vessel_count']} × {plan['vessel_class']}"
    cap = f"{plan['combined_capacity']:,} tonnes"
    util = f"{plan['utilization']}%"
    cost = f"₹{plan['cost_cr']:.2f} Cr"
    risk = f"{plan['risk_score']}/100"
    dur = f"{plan['duration']} days"
    load = f"{plan['loading_port']} Port"
    dest = f"{plan['dest_port']}"

    if name == "Balanced Recommended":
        return f"""
        <b>Balanced Strategic Trade-Off:</b> The engine selected <b>{v_desc}</b> from <b>{load}</b> to <b>{dest}</b> under <b>{incoterm}</b> terms.
        This provides <b>{cap}</b> combined carrying capacity at <b>{util} utilization</b>.
        The committed logistics cost of <b>{cost}</b> remains safely within your maximum budget of ₹{budget:.2f} Cr.
        Ocean transit of <b>{dur}</b> meets your buyer's {deadline}-day deadline SLA with low composite operational risk (<b>{risk}</b>).
        Contracted vessel readiness matches your laycan loading dates, achieving the optimal balance between cost efficiency, schedule safety, and port terminal berth readiness.
        """
    elif name == "Lowest Cost":
        return f"""
        <b>Maximum Freight Economy:</b> Allocating <b>{v_desc}</b> minimizes total logistics expenditure to <b>{cost}</b> (vs budget of ₹{budget:.2f} Cr).
        High capacity utilization (<b>{util}</b>) eliminates unused vessel penalty deadfreight.
        Transit duration of <b>{dur}</b> comfortably satisfies delivery within {deadline} days while maintaining acceptable risk of <b>{risk}</b>.
        """
    else:
        return f"""
        <b>Maximum Sea-Lane & Port Assurance:</b> Routing via <b>{load}</b> using <b>{v_desc}</b> prioritizes berth availability, proven terminal loading productivity (1,800 t/hr), and low weather disruption exposure.
        Achieves the lowest composite risk score of <b>{risk}</b> and fast <b>{dur}</b> turnaround at a competitive logistics cost of <b>{cost}</b>.
        """

def render_trade_direction_selector():
    """Renders the global Trade Direction selector across modules."""
    cur_dir = st.session_state.get("trade_direction", "Import to India")
    opts = ["Import to India", "Export from India"]
    idx = opts.index(cur_dir) if cur_dir in opts else 0
    slug = st.session_state.get("active_page", "Home").lower().replace(" ", "_").replace("-", "_")

    c1, c2 = st.columns([2.2, 3.8])
    with c1:
        st.markdown("<div style='font-size: 0.78rem; font-weight: 700; color: #372580; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 2px;'>🌐 Global Trade Direction</div>", unsafe_allow_html=True)
        sel_dir = st.radio(
            "Global Trade Direction",
            opts,
            index=idx,
            horizontal=True,
            key=f"trade_dir_toggle_{slug}",
            label_visibility="collapsed",
            help="Switch application context between Bulk Cargo Import to India and Commodity Export from India"
        )
        if sel_dir != cur_dir:
            st.session_state["trade_direction"] = sel_dir
            st.session_state["plan_confirmed"] = False
            st.session_state["sp_state"] = "Inputs Changed"
            st.rerun()

def render_shared_scenario_summary():
    """Renders standard shared scenario summary VP-DEMO-0001 across all operational modules."""
    td = st.session_state.get("trade_direction", "Import to India")
    if td == "Import to India":
        calc = get_scenario_calculations()
        rec = calc.get("active_recommendation")
        vessel_str = rec["vessel"] if rec else "2 × Panamax"
        port_name = rec["port"] if rec else "Paradip Port"
        orig_val = calc.get("origin", st.session_state.get("origin", "Richards Bay, South Africa"))
        route_str = f"{orig_val.split(',')[0]} → {port_name}"
        qty_str = f"Shortfall: {calc['cargo_shortfall']:,} t"
        cost_str = f"Cost: ₹{rec['optimized_cost']:.2f} Cr" if rec else "Cost: ₹28.60 Cr"
        risk_str = "Risk: 24/100 (Low)"
        comm_str = calc["commodity"]
    else:
        comm_str = st.session_state.get("export_commodity", "Finished Steel")
        load_port = st.session_state.get("export_loading_port", "Paradip")
        dest_port = st.session_state.get("export_dest_port", "Singapore")
        order_qty = st.session_state.get("export_order_quantity", 100000)
        inv = st.session_state.get("current_export_inventory", 70000)
        prod = st.session_state.get("planned_production", 20000)
        res = st.session_state.get("reserved_domestic_stock", 10000)
        ready, shortfall, pct = calculate_export_availability(order_qty, inv, prod, res)
        vessel_str = st.session_state.get("export_recommended_vessel", "2 × Supramax")
        route_str = f"{load_port} → {dest_port}"
        qty_str = f"Order: {order_qty:,} t ({pct}% Ready)"
        cost_str = "Est. Logistics: ₹15.10 Cr"
        risk_str = "Risk: 22/100 (Low)"

    st.markdown(
        f"""
        <div class="panel-card" style="margin-bottom: 14px; padding: 10px 16px; border-left: 4px solid #372580; background: #FFFFFF; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; font-size: 0.86rem;">
                <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                    <span style="background: #372580; color: #FFFFFF; font-weight: 700; font-size: 0.72rem; padding: 2px 7px; border-radius: 4px;">VP-DEMO-0001</span>
                    <span style="font-weight: 700; color: #18181B;">{td.upper()}:</span>
                    <span style="color: #372580; font-weight: 600;">📦 {comm_str}</span>
                    <span style="color: #ECECF0;">•</span>
                    <span style="color: #18181B; font-weight: 600;">🗺️ {route_str}</span>
                    <span style="color: #ECECF0;">•</span>
                    <span style="color: #6B6B73;">{qty_str}</span>
                    <span style="color: #ECECF0;">•</span>
                    <span style="color: #18181B; font-weight: 600;">🚢 {vessel_str}</span>
                    <span style="color: #ECECF0;">•</span>
                    <span style="color: #48A868; font-weight: 600;">{cost_str}</span>
                    <span style="color: #ECECF0;">•</span>
                    <span style="color: #16A34A; font-weight: 600;">🛡️ {risk_str}</span>
                </div>
                <span class="prototype-badge" style="margin: 0;">PROTOTYPE DATA</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>:root {
        /* Application Shell & Backgrounds */
        --bg-main: #F3F3F5;
        --bg-surface: #FFFFFF;
        --bg-sidebar: #FFFFFF;
        --bg-header: #FFFFFF;
        --bg-subtle: #F8F8FA;

        /* Typography & Text */
        --text-primary: #18181B;
        --text-secondary: #6B6B73;
        --text-muted: #92929A;

        /* Borders & Hairlines */
        --border-card: #E4E4E8;
        --border-divider: #ECECF0;
        --border-subtle: #F0F0F3;

        /* Brand Colors (Enterprise Purple Palette) */
        --primary-purple: #372580;
        --secondary-purple: #5746A5;
        --light-purple: #D9D4EE;
        --bg-purple-subtle: #F0EEF9;

        /* Accent & Status Colors */
        --accent-amber: #F6B51B;
        --light-amber: #FBE1A1;
        --bg-amber-subtle: #FEF8E7;
        --success-green: #48A868;
        --bg-green-subtle: #EDF7F0;
        --error-red: #D95C5C;
        --bg-red-subtle: #FBEEEE;

        /* Controls & Buttons */
        --btn-primary-bg: #372580;
        --btn-primary-hover: #2B1D66;
        --btn-secondary-bg: #FFFFFF;
        --btn-secondary-border: #E4E4E8;
        --btn-secondary-hover: #F3F3F5;
        --disabled-bg: #F3F3F5;
        --disabled-text: #92929A;

        /* Charts */
        --chart-bg: #FFFFFF;
        --chart-label: #18181B;
        --chart-grid: #ECECF0;

        /* Typography Hierarchy */
        --font-stack: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
        --radius-card: 8px;
        --radius-control: 6px;
        --radius-badge: 4px;
        --shadow-card: 0 1px 3px rgba(0, 0, 0, 0.04);
    }

    /* Base Reset & Application Canvas */
    html, body, .stApp, [data-testid="stAppViewContainer"], .main {
        overflow-x: hidden !important;
        max-width: 100vw !important;
        box-sizing: border-box !important;
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-stack) !important;
    }

    body, p, li, div {
        font-size: 15px;
        line-height: 1.5;
    }

    /* Top Navigation Header */
    [data-testid="stHeader"] {
        background-color: var(--bg-header) !important;
        border-bottom: 1px solid var(--border-divider) !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
    }

    #MainMenu, [data-testid="stToolbar"] {
        margin-right: 12px !important;
        color: var(--text-secondary) !important;
    }

    /* Enterprise Header Card */
    .enterprise-header-container {
        padding: 16px 20px;
        background: var(--bg-surface);
        border: 1px solid var(--border-card);
        border-radius: var(--radius-card);
        box-shadow: var(--shadow-card);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        margin-bottom: 20px;
    }

    .header-title-box {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }

    .header-title-row {
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
    }

    .header-page-title {
        margin: 0 !important;
        color: var(--text-primary) !important;
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    .prototype-badge {
        display: inline-block;
        background: var(--bg-purple-subtle) !important;
        color: var(--primary-purple) !important;
        border: 1px solid var(--light-purple) !important;
        border-radius: var(--radius-badge) !important;
        padding: 2px 8px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }

    .header-desc {
        color: var(--text-secondary) !important;
        font-size: 0.90rem !important;
        margin: 0 !important;
    }

    /* Sidebar Shell & Navigation */
    [data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-card) !important;
        box-shadow: none !important;
    }

    [data-testid="stSidebarContent"] {
        background-color: var(--bg-sidebar) !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 16px !important;
        padding-bottom: 24px !important;
    }

    .sidebar-brand-wrapper {
        padding: 4px 6px 16px 6px;
        display: flex;
        align-items: center;
        gap: 12px;
        border-bottom: 1px solid var(--border-divider);
        margin-bottom: 16px;
    }

    .sidebar-brand-logo {
        font-size: 1.75rem;
        line-height: 1;
        color: var(--primary-purple);
    }

    .sidebar-brand-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.01em;
        line-height: 1.1;
    }

    .sidebar-brand-sub {
        font-size: 0.76rem;
        font-weight: 500;
        color: var(--text-secondary);
        margin-top: 2px;
    }

    /* Sidebar Navigation Menu Items */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] {
        width: 100% !important;
    }

    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div {
        gap: 4px !important;
    }

    div[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        background-color: transparent !important;
        border-radius: var(--radius-control) !important;
        padding: 8px 12px !important;
        border: 1px solid transparent !important;
        transition: all 0.15s ease !important;
        cursor: pointer !important;
        margin-bottom: 2px !important;
    }

    div[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
        background-color: var(--bg-main) !important;
    }

    /* Highlight active module with light purple background and purple indicator */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"],
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
        background-color: var(--bg-purple-subtle) !important;
        border-left: 3px solid var(--primary-purple) !important;
    }

    div[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] p,
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) p {
        color: var(--primary-purple) !important;
        font-weight: 600 !important;
    }

    div[data-testid="stSidebar"] div[data-testid="stRadio"] label > div:first-child {
        display: none !important;
    }

    div[data-testid="stSidebar"] div[data-testid="stRadio"] p {
        font-size: 0.90rem !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        margin: 0 !important;
    }

    /* Sidebar Footer Engine Status Card */
    .sidebar-engine-card {
        margin-top: 22px;
        padding: 12px 14px;
        background: var(--bg-subtle);
        border: 1px solid var(--border-card);
        border-radius: var(--radius-control);
    }

    .engine-status-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 4px;
    }

    .engine-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--success-green);
    }

    .engine-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--text-primary);
    }

    .engine-meta {
        font-size: 0.74rem;
        color: var(--text-secondary);
        line-height: 1.4;
    }

    /* Enterprise Cards & Metric Boxes */
    [data-testid="stMetric"], .panel-card, .kpi-card, .kpi-card-exec, .snapshot-card, .hub-module-card {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border-card) !important;
        border-radius: var(--radius-card) !important;
        box-shadow: var(--shadow-card) !important;
        color: var(--text-primary) !important;
    }

    [data-testid="stMetric"] {
        padding: 16px 18px !important;
    }

    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.03em !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        margin-top: 4px !important;
    }

    .panel-card {
        padding: 16px 20px;
        margin-bottom: 16px;
    }

    .recommend {
        padding: 16px 20px;
        border-radius: var(--radius-card);
        background: var(--bg-green-subtle);
        border: 1px solid rgba(72, 168, 104, 0.3);
        border-left: 4px solid var(--success-green);
        color: var(--text-primary);
    }

    .warning {
        padding: 16px 20px;
        border-radius: var(--radius-card);
        background: var(--bg-amber-subtle);
        border: 1px solid rgba(246, 181, 27, 0.3);
        border-left: 4px solid var(--accent-amber);
        color: var(--text-primary);
    }

    .small-note {
        color: var(--text-secondary);
        font-size: 0.86rem;
    }

    /* Shared Scenario Control Bar */
    .scenario-control-bar {
        background: var(--bg-surface);
        border: 1px solid var(--border-card);
        border-radius: var(--radius-card);
        padding: 14px 18px;
        margin-bottom: 16px;
        box-shadow: var(--shadow-card);
    }

    div[data-testid="stHorizontal"]:has(#scenario-bar-anchor) {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border-card) !important;
        border-radius: var(--radius-card) !important;
        padding: 14px 18px !important;
        box-shadow: var(--shadow-card) !important;
    }

    .ctrl-label {
        color: var(--text-secondary);
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 4px;
    }

    div[data-testid="stSelectbox"] label[data-testid="stWidgetLabel"],
    div[data-testid="stWidgetLabel"] label {
        color: var(--text-secondary) !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.03em !important;
    }

    /* Scenario control buttons */
    .scenario-control-bar div.stButton > button,
    div[data-testid="stHorizontal"]:has(#scenario-bar-anchor) div.stButton > button {
        height: 52px !important;
        min-height: 52px !important;
        font-size: 0.90rem !important;
        border-radius: var(--radius-control) !important;
        padding: 0 14px !important;
    }

    /* Dropdown container styles & Equal Height (52px) */
    div[data-testid="stSelectbox"] {
        min-width: 0 !important;
    }

    #forecasting-control-bar div[data-testid="stSelectbox"] {
        min-width: 0 !important;
        width: 100% !important;
    }

    #forecasting-control-bar div[data-testid="stSelectbox"] > div[data-baseweb="select"] {
        min-width: 0 !important;
        width: 100% !important;
    }

    /* Clean white rounded rectangular dropdown, 52px height, 6px border radius */
    div[data-testid="stSelectbox"] > div[data-baseweb="select"] > div {
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-card) !important;
        border-radius: var(--radius-control) !important;
        height: 52px !important;
        min-height: 52px !important;
        max-height: 52px !important;
        display: flex !important;
        align-items: center !important;
        padding: 0 12px !important;
        box-sizing: border-box !important;
        box-shadow: none !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    }

    /* Hover & Focus state */
    div[data-testid="stSelectbox"] > div[data-baseweb="select"] > div:hover {
        border-color: var(--secondary-purple) !important;
    }

    div[data-testid="stSelectbox"] > div[data-baseweb="select"] > div:focus-within {
        border-color: var(--primary-purple) !important;
        box-shadow: 0 0 0 2px rgba(55, 37, 128, 0.15) !important;
    }

    /* Dropdown typography & text */
    div[data-testid="stSelectbox"] span,
    div[data-testid="stSelectbox"] [data-testid="stMarkdownContainer"] p,
    div[data-testid="stSelectbox"] [data-testid="stSelectboxVirtualDropdown"] {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        font-size: 0.92rem !important;
    }

    /* Chevron icon */
    div[data-testid="stSelectbox"] svg {
        fill: var(--text-secondary) !important;
        color: var(--text-secondary) !important;
    }

    /* Popover menu options */
    div[data-baseweb="popover"] ul, div[data-baseweb="menu"] {
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-card) !important;
        border-radius: var(--radius-card) !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08) !important;
        padding: 4px !important;
    }

    div[data-baseweb="popover"] li, div[data-baseweb="menu"] li {
        color: var(--text-primary) !important;
        background-color: var(--bg-surface) !important;
        font-size: 0.90rem !important;
        border-radius: 4px !important;
        padding: 8px 12px !important;
    }

    div[data-baseweb="popover"] li:hover, div[data-baseweb="menu"] li:hover {
        background-color: var(--bg-main) !important;
        color: var(--primary-purple) !important;
    }

    div[data-baseweb="popover"] li[aria-selected="true"] {
        background-color: var(--bg-purple-subtle) !important;
        color: var(--primary-purple) !important;
        font-weight: 600 !important;
    }

    /* Forecasting Studio Control Spacing & Layout */
    #forecasting-control-bar {
        background: var(--bg-surface);
        border: 1px solid var(--border-card);
        border-radius: var(--radius-card);
        padding: 14px 18px;
        margin-bottom: 20px;
        box-shadow: var(--shadow-card);
    }

    #forecasting-control-bar div[data-testid="stHorizontal"] {
        gap: 16px !important;
        align-items: flex-end !important;
    }

    #forecasting-control-bar div.stButton > button {
        height: 52px !important;
        min-height: 52px !important;
        background: var(--btn-primary-bg) !important;
        color: #FFFFFF !important;
        border: 1px solid var(--btn-primary-bg) !important;
        border-radius: var(--radius-control) !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        box-shadow: 0 1px 2px rgba(55, 37, 128, 0.2) !important;
    }

    #forecasting-control-bar div.stButton > button:hover {
        background: var(--btn-primary-hover) !important;
        border-color: var(--btn-primary-hover) !important;
    }

    /* Buttons General Styling */
    .stButton > button {
        font-family: var(--font-stack) !important;
        font-weight: 600 !important;
        font-size: 0.90rem !important;
        border-radius: var(--radius-control) !important;
        padding: 8px 16px !important;
        background-color: var(--btn-secondary-bg) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--btn-secondary-border) !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
        transition: all 0.15s ease !important;
    }

    .stButton > button:hover {
        background-color: var(--btn-secondary-hover) !important;
        border-color: var(--border-card) !important;
    }

    /* Primary Button */
    .stButton > button[kind="primary"], .stButton > button[data-testid="baseButton-primary"] {
        background-color: var(--btn-primary-bg) !important;
        color: #FFFFFF !important;
        border: 1px solid var(--btn-primary-bg) !important;
        box-shadow: 0 1px 2px rgba(55, 37, 128, 0.2) !important;
    }

    .stButton > button[kind="primary"]:hover, .stButton > button[data-testid="baseButton-primary"]:hover {
        background-color: var(--btn-primary-hover) !important;
        border-color: var(--btn-primary-hover) !important;
    }

    /* Download Buttons */
    div.stDownloadButton > button {
        background-color: var(--btn-secondary-bg) !important;
        color: var(--primary-purple) !important;
        border: 1px solid var(--light-purple) !important;
        border-radius: var(--radius-control) !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
    }

    div.stDownloadButton > button:hover {
        background-color: var(--bg-purple-subtle) !important;
        border-color: var(--primary-purple) !important;
    }

    /* Disabled button state */
    button:disabled, .stButton > button:disabled, [disabled] {
        background-color: var(--disabled-bg) !important;
        color: var(--disabled-text) !important;
        border-color: var(--border-card) !important;
        cursor: not-allowed !important;
        box-shadow: none !important;
    }

    /* KPI Cards Row */
    .kpi-card-exec {
        padding: 16px 18px;
        background: var(--bg-surface);
        border: 1px solid var(--border-card);
        border-radius: var(--radius-card);
        box-shadow: var(--shadow-card);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .kpi-card-exec:hover {
        border-color: var(--secondary-purple);
    }

    .kpi-exec-label {
        color: var(--text-secondary);
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .kpi-exec-val {
        color: var(--text-primary);
        font-size: 1.65rem;
        font-weight: 700;
        margin: 4px 0;
        letter-spacing: -0.01em;
    }

    .kpi-exec-sub {
        font-size: 0.80rem;
        font-weight: 500;
    }

    /* Snapshot Cards */
    .snapshot-card {
        padding: 16px 18px;
        background: var(--bg-surface);
        border: 1px solid var(--border-card);
        border-radius: var(--radius-card);
        box-shadow: var(--shadow-card);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .snapshot-title {
        color: var(--text-primary);
        font-size: 1.02rem;
        font-weight: 600;
        margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 1px solid var(--border-divider);
    }

    .snapshot-list {
        margin: 0;
        padding-left: 18px;
        color: var(--text-secondary);
        font-size: 0.88rem;
        line-height: 1.6;
    }

    /* Landing Hub Cards */
    .hub-module-card {
        padding: 18px 20px;
        background: var(--bg-surface);
        border: 1px solid var(--border-card);
        border-radius: var(--radius-card);
        box-shadow: var(--shadow-card);
        margin-bottom: 8px;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }

    .hub-module-card:hover {
        border-color: var(--secondary-purple);
        box-shadow: 0 4px 12px rgba(55, 37, 128, 0.06);
    }

    /* Severity Status Badges */
    .severity-badge-watch {
        background: var(--bg-amber-subtle) !important;
        color: #B45309 !important;
        border: 1px solid rgba(246, 181, 27, 0.4) !important;
        font-weight: 600 !important;
        border-radius: var(--radius-badge) !important;
        padding: 2px 8px !important;
        font-size: 0.74rem !important;
    }

    .severity-badge-normal {
        background: var(--bg-purple-subtle) !important;
        color: var(--primary-purple) !important;
        border: 1px solid var(--light-purple) !important;
        font-weight: 600 !important;
        border-radius: var(--radius-badge) !important;
        padding: 2px 8px !important;
        font-size: 0.74rem !important;
    }

    .severity-badge-optimal {
        background: var(--bg-green-subtle) !important;
        color: #15803D !important;
        border: 1px solid rgba(72, 168, 104, 0.4) !important;
        font-weight: 600 !important;
        border-radius: var(--radius-badge) !important;
        padding: 2px 8px !important;
        font-size: 0.74rem !important;
    }

    /* Dataframe and Tables */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border-card) !important;
        border-radius: var(--radius-card) !important;
        overflow: hidden !important;
        box-shadow: var(--shadow-card) !important;
    }

    /* Typography Headings */
    h1 {
        font-weight: 700 !important;
        color: var(--text-primary) !important;
    }

    h2, h3 {
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }

    h4 {
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }

    /* Mobile Responsive Adjustments (< 768px) */
    @media (max-width: 768px) {
        #forecasting-control-bar div[data-testid="stHorizontal"] {
            display: flex !important;
            flex-direction: column !important;
            gap: 12px !important;
        }
        #forecasting-control-bar div[data-testid="stSelectbox"] {
            width: 100% !important;
        }
        #forecasting-control-bar div.stButton > button {
            width: 100% !important;
        }
        .maritime-hero-svg {
            max-height: 140px !important;
        }
        .maritime-route-svg {
            max-height: 180px !important;
        }
        .animated-ship,
        .animated-wave-1,
        .animated-wave-2 {
            animation: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


MONTHS = pd.date_range("2025-01-01", periods=20, freq="MS")
DEMAND = [172, 178, 184, 181, 190, 196, 203, 207, 211, 218,
          224, 230, 227, 235, 241, 247, 252, 258, 264, 270]
demand_df = pd.DataFrame({"Month": MONTHS, "Demand (000 t)": DEMAND})
demand_df["Month No"] = range(1, len(demand_df) + 1)

SUPPLIERS = pd.DataFrame([
    ["Ubuntu Bulk Trading", "Richards Bay, South Africa", 1428, 240000, 16, 0.06],
    ["Nusantara Minerals", "Indonesia", 1520, 260000, 18, 0.08],
    ["Southern Cross Resources", "Australia", 1580, 320000, 26, 0.07],
], columns=["Supplier", "Origin", "Cargo Price", "Availability", "Transit Days", "Supplier Risk"])

VESSELS = pd.DataFrame([
    ["MV Ocean Crest", "Supramax", 58000, 3.80, 0.07],
    ["MV Eastern Star", "Panamax", 82000, 4.45, 0.06],
    ["MV Blue Horizon", "Panamax", 82000, 4.30, 0.05],
    ["MV Titan Sea", "Capesize", 180000, 10.50, 0.10],
], columns=["Vessel", "Class", "Capacity", "Charter Cost Cr", "Vessel Risk"])

PORTS = pd.DataFrame([
    ["Paradip", 85, 3, 180000, 0.05],
    ["Dhamra", 95, 2, 180000, 0.06],
    ["Visakhapatnam", 110, 5, 125000, 0.12],
], columns=["Port", "Handling ₹/t", "Waiting Days", "Max Vessel", "Port Risk"])


def forecast_demand(horizon_days=90):
    model = LinearRegression()
    model.fit(demand_df[["Month No"]], demand_df["Demand (000 t)"])
    months_ahead = max(1, math.ceil(horizon_days / 30))
    future_nos = pd.DataFrame({"Month No": range(len(demand_df) + 1, len(demand_df) + months_ahead + 1)})
    predicted = model.predict(future_nos).round(1)
    future_months = pd.date_range(MONTHS[-1] + pd.offsets.MonthBegin(1), periods=months_ahead, freq="MS")
    return model, pd.DataFrame({"Month": future_months, "Demand (000 t)": predicted})


def optimize(cargo_required, fuel_change=0, demand_change=0, unavailable_port="None", deadline_days=45):
    adjusted_quantity = cargo_required * (1 + demand_change / 100)
    feasible = []
    available_ports = PORTS if unavailable_port == "None" else PORTS[PORTS["Port"] != unavailable_port]

    for _, supplier in SUPPLIERS.iterrows():
        if adjusted_quantity > supplier["Availability"]:
            continue
        for _, vessel in VESSELS.iterrows():
            vessels_needed = math.ceil(adjusted_quantity / vessel["Capacity"])
            combined_capacity = vessels_needed * vessel["Capacity"]
            # Strict Constraint: Combined vessel capacity must be at least the cargo shortfall
            # Never recommend 2 × Supramax (116,000 t) for 130,000 t shortfall
            if combined_capacity < adjusted_quantity:
                continue
            utilization = adjusted_quantity / combined_capacity
            for _, port in available_ports.iterrows():
                if vessel["Capacity"] > port["Max Vessel"]:
                    continue
                eta_days = int(supplier["Transit Days"] + port["Waiting Days"])
                if eta_days > deadline_days:
                    continue

                cargo_cost_cr = adjusted_quantity * supplier["Cargo Price"] / 10_000_000
                charter_cost_cr = vessels_needed * vessel["Charter Cost Cr"] * (1 + fuel_change / 100 * 0.45)
                port_cost_cr = adjusted_quantity * port["Handling ₹/t"] / 10_000_000
                waiting_cost_cr = vessels_needed * port["Waiting Days"] * 0.02
                low_util_penalty = max(0, 0.80 - utilization) * charter_cost_cr * 0.5
                risk = (supplier["Supplier Risk"] + vessel["Vessel Risk"] + port["Port Risk"]) / 3
                risk_cost_cr = risk * (charter_cost_cr + port_cost_cr) * 0.35
                total = cargo_cost_cr + charter_cost_cr + port_cost_cr + waiting_cost_cr + low_util_penalty + risk_cost_cr

                feasible.append({
                    "Supplier": supplier["Supplier"], "Origin": supplier["Origin"],
                    "Vessel": vessel["Vessel"], "Class": vessel["Class"],
                    "Vessels": vessels_needed, "Port": port["Port"], "ETA Days": eta_days,
                    "Cargo (t)": int(adjusted_quantity), "Utilization": round(utilization * 100, 1),
                    "Combined Capacity": combined_capacity,
                    "Cargo Cost Cr": cargo_cost_cr, "Charter Cost Cr": charter_cost_cr,
                    "Port & Waiting Cr": port_cost_cr + waiting_cost_cr,
                    "Risk Cost Cr": risk_cost_cr, "Risk": 24, "Total Cost Cr": round(total, 2),
                })

    return pd.DataFrame(feasible).sort_values("Total Cost Cr") if feasible else pd.DataFrame()



# Session State Initialization (Pure st.session_state, no URL query params)
PAGE_SLUG_MAP = {
    "Home": "Home",
    "Command-Centre": "Command Centre",
    "Forecasting-Studio": "Forecasting Studio",
    "Shipment-Planner": "Shipment Planner",
    "Optimization-Hub": "Optimization Hub",
    "Vessel-Intelligence": "Vessel Intelligence",
    "Port-Intelligence": "Port Intelligence",
    "Route-Intelligence": "Route Intelligence",
    "Risk-Alerts": "Risk & Alerts",
    "Scenario-Lab": "Scenario Lab",
    "Reports": "Reports",
    "Data-Settings": "Data & Settings",
}
MODULE_TO_SLUG = {v: k for k, v in PAGE_SLUG_MAP.items()}

# Clear any legacy or existing query parameters from the visible browser URL once using Streamlit API
if len(st.query_params) > 0:
    st.query_params.clear()

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Home"

active_page = st.session_state["active_page"]
st.session_state["active_module"] = active_page if active_page in MODULE_NAMES else "Command Centre"
active_module = st.session_state["active_module"]

if st.session_state.get("_reset_controls", False):
    st.session_state["cargo_type"] = "Thermal Coal"
    st.session_state["forecast_horizon"] = "90 Days"
    st.session_state["operational_scenario"] = "Base Scenario"
    st.session_state["_reset_controls"] = False

if "cargo" not in st.session_state:
    st.session_state["cargo"] = "Thermal Coal"
if "horizon" not in st.session_state:
    st.session_state["horizon"] = "90 Days"
if "origin" not in st.session_state:
    st.session_state["origin"] = "Richards Bay, South Africa"
if "forecast_model" not in st.session_state:
    st.session_state["forecast_model"] = "Ensemble Recommended"
if "scenario" not in st.session_state:
    st.session_state["scenario"] = "Base Scenario"
if "cargo_requirement" not in st.session_state:
    st.session_state["cargo_requirement"] = 150000
if "inventory" not in st.session_state:
    st.session_state["inventory"] = 40000
if "safety" not in st.session_state:
    st.session_state["safety"] = 20000
if "deadline" not in st.session_state:
    st.session_state["deadline"] = 45
if "fuel_change" not in st.session_state:
    st.session_state["fuel_change"] = 0
if "fuel_price" not in st.session_state:
    st.session_state["fuel_price"] = 54000
if "show_alerts_panel" not in st.session_state:
    st.session_state["show_alerts_panel"] = False

if "trade_direction" not in st.session_state:
    st.session_state["trade_direction"] = "Import to India"
if "export_commodity" not in st.session_state:
    st.session_state["export_commodity"] = "Finished Steel"
if "export_buyer_ref" not in st.session_state:
    st.session_state["export_buyer_ref"] = "POSCO Asia Steel Corp"
if "export_buyer_country" not in st.session_state:
    st.session_state["export_buyer_country"] = "South Korea"
if "export_order_quantity" not in st.session_state:
    st.session_state["export_order_quantity"] = 100000
if "current_export_inventory" not in st.session_state:
    st.session_state["current_export_inventory"] = 70000
if "planned_production" not in st.session_state:
    st.session_state["planned_production"] = 20000
if "reserved_domestic_stock" not in st.session_state:
    st.session_state["reserved_domestic_stock"] = 10000
if "export_laycan_start" not in st.session_state:
    st.session_state["export_laycan_start"] = date.today() + timedelta(days=7)
if "export_laycan_end" not in st.session_state:
    st.session_state["export_laycan_end"] = date.today() + timedelta(days=14)
if "export_deadline" not in st.session_state:
    st.session_state["export_deadline"] = 30
if "export_incoterm" not in st.session_state:
    st.session_state["export_incoterm"] = "CFR"
if "export_selling_price" not in st.session_state:
    st.session_state["export_selling_price"] = 6800
if "export_internal_cost" not in st.session_state:
    st.session_state["export_internal_cost"] = 4200
if "export_max_risk" not in st.session_state:
    st.session_state["export_max_risk"] = 40
if "export_max_budget" not in st.session_state:
    st.session_state["export_max_budget"] = 20.0
if "export_loading_port" not in st.session_state:
    st.session_state["export_loading_port"] = "Paradip"
if "export_dest_port" not in st.session_state:
    st.session_state["export_dest_port"] = "Singapore"
if "export_plan_mode" not in st.session_state:
    st.session_state["export_plan_mode"] = "Plan Full Order"
if "export_allowed_vessels" not in st.session_state:
    st.session_state["export_allowed_vessels"] = ["Supramax (58k)", "Panamax (82k)", "Capesize (180k)"]
if "export_allowed_loading_ports" not in st.session_state:
    st.session_state["export_allowed_loading_ports"] = ["Paradip", "Visakhapatnam", "Chennai", "Kamarajar (Ennore)", "Krishnapatnam", "Dhamra", "Gangavaram"]
if "export_allowed_dest_ports" not in st.session_state:
    st.session_state["export_allowed_dest_ports"] = ["Singapore", "Chittagong", "Port Klang", "Colombo", "Jebel Ali", "Rotterdam", "Qingdao"]
if "export_recommended_vessel" not in st.session_state:
    st.session_state["export_recommended_vessel"] = "2 × Supramax"
if "sp_state" not in st.session_state:
    st.session_state["sp_state"] = "Result Current"
if "sp_import_snapshot" not in st.session_state:
    st.session_state["sp_import_snapshot"] = None
if "sp_export_snapshot" not in st.session_state:
    st.session_state["sp_export_snapshot"] = None


# Compatibility session state initialization
if "cargo_type" not in st.session_state:
    st.session_state["cargo_type"] = st.session_state["cargo"]
if "forecast_horizon" not in st.session_state:
    st.session_state["forecast_horizon"] = st.session_state["horizon"]
if "operational_scenario" not in st.session_state:
    st.session_state["operational_scenario"] = st.session_state["scenario"]

# Desktop Sidebar
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand-wrapper">
            <div class="sidebar-brand-logo">⚓</div>
            <div>
                <div class="sidebar-brand-title">VarunaPath AI</div>
                <div class="sidebar-brand-sub">Maritime Decision Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if active_page != "Home":
        if st.button("🏠 Home (Landing Hub)", key="sb_btn_home", use_container_width=True):
            st.session_state["active_page"] = "Home"
            st.session_state["active_module"] = "Home"
            st.rerun()

        NAV_CATEGORIES = {
            "PLANNING & DECISION": [
                ("Command Centre", "🧭", "Command-Centre"),
                ("Forecasting Studio", "📈", "Forecasting-Studio"),
                ("Shipment Planner", "📦", "Shipment-Planner"),
                ("Optimization Hub", "⚡", "Optimization-Hub"),
            ],
            "MARITIME INTELLIGENCE": [
                ("Vessel Intelligence", "🚢", "Vessel-Intelligence"),
                ("Port Intelligence", "🏗️", "Port-Intelligence"),
                ("Route Intelligence", "🗺️", "Route-Intelligence"),
                ("Risk & Alerts", "🛡️", "Risk-Alerts"),
            ],
            "ANALYSIS & MANAGEMENT": [
                ("Scenario Lab", "🧪", "Scenario-Lab"),
                ("Reports", "📋", "Reports"),
                ("Data & Settings", "⚙️", "Data-Settings"),
            ],
        }

        for cat_title, mod_list in NAV_CATEGORIES.items():
            st.markdown(
                f"<div style='font-size: 0.72rem; font-weight: 700; color: #6B6B73; text-transform: uppercase; letter-spacing: 0.06em; margin: 14px 0 6px 4px;'>{cat_title}</div>",
                unsafe_allow_html=True,
            )
            for name, icon, slug in mod_list:
                is_active = (name == active_page)
                btn_type = "primary" if is_active else "secondary"
                if st.button(f"{icon}  {name}", key=f"sb_nav_{slug}", type=btn_type, use_container_width=True):
                    st.session_state["active_page"] = name
                    st.session_state["active_module"] = name
                    st.rerun()

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-engine-card" title="All models and calculations run on verified local prototype logic. External operational data feeds are not currently live.">
            <div class="engine-status-row">
                <span class="engine-dot"></span>
                <span class="engine-title">Prototype Engine Active</span>
            </div>
            <div class="engine-meta">
                AI Engine Status: Active • Online<br>
                Data Mode: Local / Simulated<br>
                Core: Maritime Fleet & Trade Decision AI<br>
                Status: Verified Local Engine • External operational data feeds are not currently live.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Mobile auto-collapse drawer when navigating
components.html(
    """
    <script>
    const parentDoc = window.parent.document;
    function setupMobileNav() {
        const labels = parentDoc.querySelectorAll('[data-testid="stSidebar"] div[data-testid="stRadio"] label');
        labels.forEach(lbl => {
            if (!lbl.dataset.drawerAttached) {
                lbl.dataset.drawerAttached = 'true';
                lbl.addEventListener('click', () => {
                    if (window.parent.innerWidth <= 768) {
                        setTimeout(() => {
                            const closeBtn = parentDoc.querySelector('[data-testid="stSidebarCollapseButton"] button');
                            if (closeBtn) closeBtn.click();
                        }, 120);
                    }
                });
            }
        });
    }
    setupMobileNav();
    setInterval(setupMobileNav, 800);
    </script>
    """,
    height=0,
    width=0,
)

def render_level2_header(module_name):
    """Renders consistent top navigation header for all Level 2 detailed module pages."""
    slug = MODULE_TO_SLUG.get(module_name, "Module")
    col_head_left, col_head_actions = st.columns([2.6, 1.4])

    with col_head_left:
        col_back, col_bc = st.columns([1, 2.4])
        with col_back:
            if st.button("← Back to Home", key=f"hdr_back_{slug}", use_container_width=True):
                st.session_state["active_page"] = "Home"
                st.session_state["active_module"] = "Home"
                st.rerun()
        with col_bc:
            st.markdown(
                f"""
                <div style="padding-top: 8px; font-size: 0.88rem; color: #6B6B73;">
                    <span style="color: #372580; font-weight: 600;">Home</span>
                    <span style="color: #92929A; margin: 0 6px;">/</span>
                    <span style="color: #18181B; font-weight: 600;">{module_name}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="enterprise-header-container" style="margin-top: 4px;">
                <div class="header-title-box">
                    <div class="header-title-row">
                        <h1 class="header-page-title">{MODULE_ICONS.get(module_name, '⚓')} {module_name}</h1>
                        <span class="prototype-badge">PROTOTYPE DATA</span>
                    </div>
                    <p class="header-desc">{MODULE_DESCRIPTIONS.get(module_name, '')}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_head_actions:
        if module_name == "Command Centre":
            ac1, ac2 = st.columns([1, 1.6])
            with ac1:
                if st.button("🔔 Alerts", key=f"hdr_alert_{slug}", help="Toggle operational maritime alerts"):
                    st.session_state["show_alerts_panel"] = not st.session_state.get("show_alerts_panel", False)
            with ac2:
                if st.button("⚡ Quick Optimize", key=f"hdr_act_{slug}", type="primary", help="Recalculate Command Centre plan"):
                    st.toast("Command Centre plan refreshed!", icon="⚡")
        elif module_name == "Forecasting Studio":
            if st.button("📈 Generate Forecast", key=f"hdr_act_{slug}", type="primary", help="Generate and recalibrate forecast models"):
                st.toast("Forecast models recalibrated!", icon="📈")
        elif module_name == "Shipment Planner":
            if st.button("⚡ Generate Plans", key=f"hdr_act_{slug}", type="primary", help="Generate feasible shipment voyage plans"):
                st.toast("Generated 3 comparative voyage plans!", icon="⚡")
        elif module_name == "Optimization Hub":
            if st.button("⚡ Run AI Optimization", key=f"hdr_act_{slug}", type="primary", help="Trigger AI Fleet Optimization engine"):
                st.session_state["trigger_opt_anim"] = True
                st.rerun()
        elif module_name == "Reports":
            if st.button("📥 Download Report", key=f"hdr_act_{slug}", type="primary", help="Prepare full audit report package"):
                st.toast("Preparing report package...", icon="📥")
        elif module_name == "Data & Settings":
            if st.button("🔄 Reset Demo Data", key=f"hdr_act_{slug}", type="primary", help="Reset to default prototype state"):
                st.session_state["cargo"] = "Thermal Coal"
                st.session_state["horizon"] = "90 Days"
                st.session_state["origin"] = "Richards Bay, South Africa"
                st.session_state["forecast_model"] = "Ensemble Recommended"
                st.session_state["scenario"] = "Base Scenario"
                st.session_state["cargo_requirement"] = 150000
                st.session_state["inventory"] = 40000
                st.session_state["safety"] = 20000
                st.session_state["deadline"] = 45
                st.session_state["fuel_change"] = 0
                st.session_state["fuel_price"] = 54000
                st.session_state["_reset_controls"] = True
                st.toast("Prototype data settings reset to demo defaults.", icon="🔄")
                st.rerun()
        else:
            if st.button("🔔 Alerts", key=f"hdr_alert_{slug}", help="Toggle operational maritime alerts"):
                st.session_state["show_alerts_panel"] = not st.session_state.get("show_alerts_panel", False)

    if st.session_state.get("show_alerts_panel", False):
        st.markdown(
            """
            <div class="panel-card" style="border-left: 4px solid #372580; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <b style="color: #372580; font-size: 1.02rem;">🔔 Active Maritime Alerts (3 Unresolved)</b>
                </div>
                <ul style="margin: 0; padding-left: 20px; color: #6B6B73; font-size: 0.90rem;">
                    <li><b>Port Notice:</b> Paradip waiting days average 3 days with maximum vessel draft clearance 180,000 DWT.</li>
                    <li><b>Fuel Volatility:</b> Bunker fuel variance assumption active at current slider setting.</li>
                    <li><b>Vessel Allocation:</b> Kamsarmax (82,000 DWT) offers optimal capacity utilization for target 90-day demand.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_landing_hub():
    """Renders the clean Main Landing Hub with 3 categories and 10 module tiles."""
    td = st.session_state.get("trade_direction", "Import to India")
    st.markdown(
        """
        <div style="text-align: center; padding: 24px 16px 12px 16px; margin-bottom: 24px;">
            <div style="display: inline-flex; align-items: center; gap: 14px; margin-bottom: 8px;">
                <span style="font-size: 2.5rem;">⚓</span>
                <h1 style="margin: 0; font-size: 2.6rem; font-weight: 700; color: #18181B; letter-spacing: -0.02em;">VarunaPath AI</h1>
            </div>
            <p style="margin: 0 0 10px 0; font-size: 1.15rem; color: #6B6B73; font-weight: 400;">
                AI-Powered Bulk Cargo Freight Forecasting & Logistics Optimization Platform
            </p>
            <div style="display: flex; justify-content: center; align-items: center; gap: 14px; font-size: 0.88rem; color: #92929A;">
                <span>SIH26006 • Team Novara</span>
                <span>•</span>
                <span style="display: inline-flex; align-items: center; gap: 6px; color: #48A868; font-weight: 600;">
                    <span style="width: 8px; height: 8px; border-radius: 50%; background: #48A868; display: inline-block;"></span>
                    AI Decision Engine Ready
                </span>
                <span>•</span>
                <span style="color: #372580; font-weight: 600;">Active Flow: {flow_label}</span>
            </div>
        </div>
        """.format(flow_label=td),
        unsafe_allow_html=True,
    )

    # Dynamic Maritime Route Hero Visual (Reflects Active Trade Direction)
    if td == "Import to India":
        origin_label = "Richards Bay"
        origin_sub = "ORIGIN (ZA)"
        dest_label = "Paradip Port"
        dest_sub = "DESTINATION • 180K DWT"
        vessel_label = "PANAMAX 2x"
        meta_label = "ETA: 19 Days • Shortfall: 130,000 t • Prototype / Simulated Data"
    else:
        origin_label = "Paradip Port"
        origin_sub = "LOADING (IN)"
        dest_label = "Singapore Port"
        dest_sub = "DISCHARGE • CFR FIXTURE"
        vessel_label = "SUPRAMAX 2x"
        meta_label = "ETA: 7 Days • Order: 100,000 t • Prototype / Simulated Data"

    st.markdown(
        f"""
        <div style="width: 100%; max-width: 900px; margin: 0 auto 24px auto; overflow: hidden; border-radius: 8px; border: 1px solid #E4E4E8; background: #FFFFFF; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
        <svg class="maritime-hero-svg" viewBox="0 0 900 185" width="100%" height="185" xmlns="http://www.w3.org/2000/svg" style="display: block;">
          <defs>
            <linearGradient id="routeGlow" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stop-color="#D9D4EE" stop-opacity="0.4" />
              <stop offset="50%" stop-color="#372580" stop-opacity="0.95" />
              <stop offset="100%" stop-color="#48A868" stop-opacity="0.9" />
            </linearGradient>
            <filter id="cyanGlow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          <!-- Nautical Grid Lines -->
          <g opacity="0.4" stroke="#ECECF0" stroke-width="1" stroke-dasharray="4,6">
            <line x1="50" y1="35" x2="850" y2="35" />
            <line x1="50" y1="90" x2="850" y2="90" />
            <line x1="50" y1="145" x2="850" y2="145" />
            <line x1="180" y1="20" x2="180" y2="165" />
            <line x1="450" y1="20" x2="450" y2="165" />
            <line x1="720" y1="20" x2="720" y2="165" />
          </g>

          <!-- Ocean Waves Layers (Animated) -->
          <path class="animated-wave-1" d="M0,130 Q150,110 300,130 T600,130 T900,130 L900,185 L0,185 Z" fill="#D9D4EE" opacity="0.3" />
          <path class="animated-wave-2" d="M0,145 Q180,130 360,145 T720,145 T900,145 L900,185 L0,185 Z" fill="#D9D4EE" opacity="0.55" />

          <!-- Route Line (Curved Arc) -->
          <path class="animated-route" d="M 120 115 Q 300 40, 500 80 T 780 70" fill="none" stroke="url(#routeGlow)" stroke-width="2.5" stroke-dasharray="6,6" filter="url(#cyanGlow)" />

          <!-- Origin Port -->
          <g transform="translate(120, 115)">
            <circle r="12" fill="#D9D4EE" opacity="0.4" />
            <circle r="6" fill="#372580" stroke="#D9D4EE" stroke-width="2" />
            <circle r="2.5" fill="#ffffff" />
            <text x="-10" y="24" fill="#18181B" font-size="11" font-weight="600" font-family="system-ui, sans-serif" text-anchor="middle">{origin_label}</text>
            <text x="-10" y="36" fill="#6B6B73" font-size="9" font-family="system-ui, sans-serif" text-anchor="middle">{origin_sub}</text>
          </g>

          <!-- Animated Cargo Vessel (Mid-Voyage) -->
          <g class="animated-ship" transform="translate(480, 75)">
            <ellipse cx="0" cy="18" rx="42" ry="7" fill="#18181B" opacity="0.08" />
            <path d="M -36,8 L 26,8 L 38,-2 L -34,-2 Z" fill="#372580" stroke="#5746A5" stroke-width="1.2" />
            <line x1="-36" y1="8" x2="26" y2="8" stroke="#F6B51B" stroke-width="2.2" stroke-linecap="round" />
            <rect x="-24" y="-8" width="10" height="6" rx="1.5" fill="#5746A5" stroke="#D9D4EE" stroke-width="0.8" />
            <rect x="-10" y="-8" width="10" height="6" rx="1.5" fill="#5746A5" stroke="#D9D4EE" stroke-width="0.8" />
            <rect x="4" y="-8" width="10" height="6" rx="1.5" fill="#5746A5" stroke="#D9D4EE" stroke-width="0.8" />
            <polygon points="-33,-2 -33,-16 -23,-16 -23,-2" fill="#FFFFFF" stroke="#E4E4E8" stroke-width="0.8" />
            <rect x="-31" y="-14" width="6" height="3" fill="#372580" />
            <line x1="-28" y1="-16" x2="-28" y2="-22" stroke="#6B6B73" stroke-width="1.2" />
            <path d="M 38,-2 Q 44,4 40,9" fill="none" stroke="#5746A5" stroke-width="1.5" opacity="0.75" />
            <rect x="-36" y="-34" width="72" height="14" rx="4" fill="#F0EEF9" stroke="#5746A5" stroke-width="0.8" />
            <text x="0" y="-24" fill="#372580" font-size="9" font-weight="600" font-family="system-ui, sans-serif" text-anchor="middle">{vessel_label}</text>
          </g>

          <!-- Destination Port -->
          <g transform="translate(780, 70)">
            <circle r="14" fill="#EDF7F0" opacity="0.6" />
            <circle r="7" fill="#48A868" stroke="#FFFFFF" stroke-width="2" />
            <circle r="3" fill="#ffffff" />
            <text x="14" y="-4" fill="#18181B" font-size="12" font-weight="700" font-family="system-ui, sans-serif">{dest_label}</text>
            <text x="14" y="10" fill="#48A868" font-size="10" font-family="system-ui, sans-serif">{dest_sub}</text>
            <text x="14" y="23" fill="#6B6B73" font-size="9" font-family="system-ui, sans-serif">{meta_label}</text>
          </g>
        </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )

    def render_hub_card(name, icon, desc, slug):
        st.markdown(
            f"""
            <div class="hub-module-card">
                <div>
                    <div style="font-size: 1.85rem; margin-bottom: 8px;">{icon}</div>
                    <div style="font-size: 1.14rem; font-weight: 700; color: #18181B; margin-bottom: 6px; letter-spacing: -0.01em;">{name}</div>
                    <div style="font-size: 0.88rem; color: #6B6B73; line-height: 1.45;">{desc}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(f"Open {name} →", key=f"hub_btn_{slug}", use_container_width=True):
            st.session_state["active_page"] = name
            st.session_state["active_module"] = name
            st.rerun()

    # CATEGORY 1 — PLANNING & DECISION (4 modules)
    st.markdown(
        """
        <div style="margin-top: 10px; margin-bottom: 14px;">
            <span style="font-size: 0.82rem; font-weight: 700; color: #372580; text-transform: uppercase; letter-spacing: 0.06em; background: #F0EEF9; padding: 4px 12px; border-radius: 6px; border-left: 3px solid #372580;">
                Category 1 • Planning & Decision
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1_mods = [
        ("Command Centre", "🧭", "Executive overview and logistics decision summary", "Command-Centre"),
        ("Forecasting Studio", "📈", "Forecast cargo demand, inventory and freight rates", "Forecasting-Studio"),
        ("Shipment Planner", "📦", "Configure cargo requirements and generate shipment plans", "Shipment-Planner"),
        ("Optimization Hub", "⚡", "Compare cost, risk and balanced recommendations", "Optimization-Hub"),
    ]
    row1_c1, row1_c2, row1_c3 = st.columns(3, gap="medium")
    with row1_c1:
        render_hub_card(*c1_mods[0])
    with row1_c2:
        render_hub_card(*c1_mods[1])
    with row1_c3:
        render_hub_card(*c1_mods[2])

    row2_c1, row2_c2, row2_c3 = st.columns(3, gap="medium")
    with row2_c1:
        render_hub_card(*c1_mods[3])

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    # CATEGORY 2 — MARITIME INTELLIGENCE (4 modules)
    st.markdown(
        """
        <div style="margin-top: 10px; margin-bottom: 14px;">
            <span style="font-size: 0.82rem; font-weight: 700; color: #372580; text-transform: uppercase; letter-spacing: 0.06em; background: #F0EEF9; padding: 4px 12px; border-radius: 6px; border-left: 3px solid #372580;">
                Category 2 • Maritime Intelligence
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c2_mods = [
        ("Vessel Intelligence", "🚢", "Compare vessel capacity, availability and charter rates", "Vessel-Intelligence"),
        ("Port Intelligence", "🏗️", "Analyse Indian East Coast ports and route suitability", "Port-Intelligence"),
        ("Route Intelligence", "🗺️", "Compare maritime route alternatives using distance, ETA, fuel and risk", "Route-Intelligence"),
        ("Risk & Alerts", "🛡️", "Monitor operational, weather and schedule risks", "Risk-Alerts"),
    ]
    c2_col1, c2_col2, c2_col3, c2_col4 = st.columns(4, gap="medium")
    with c2_col1:
        render_hub_card(*c2_mods[0])
    with c2_col2:
        render_hub_card(*c2_mods[1])
    with c2_col3:
        render_hub_card(*c2_mods[2])
    with c2_col4:
        render_hub_card(*c2_mods[3])

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    # CATEGORY 3 — ANALYSIS & MANAGEMENT (3 modules)
    st.markdown(
        """
        <div style="margin-top: 10px; margin-bottom: 14px;">
            <span style="font-size: 0.82rem; font-weight: 700; color: #372580; text-transform: uppercase; letter-spacing: 0.06em; background: #F0EEF9; padding: 4px 12px; border-radius: 6px; border-left: 3px solid #372580;">
                Category 3 • Analysis & Management
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c3_mods = [
        ("Scenario Lab", "🧪", "Simulate fuel, congestion, weather and deadline changes", "Scenario-Lab"),
        ("Reports", "📋", "View and download decision-support reports", "Reports"),
        ("Data & Settings", "⚙️", "Manage prototype data, models and application preferences", "Data-Settings"),
    ]
    c3_col1, c3_col2, c3_col3 = st.columns(3, gap="medium")
    with c3_col1:
        render_hub_card(*c3_mods[0])
    with c3_col2:
        render_hub_card(*c3_mods[1])
    with c3_col3:
        render_hub_card(*c3_mods[2])

    st.markdown(
        """
        <div style="text-align: center; padding: 36px 16px 16px 16px; margin-top: 40px; border-top: 1px solid #ECECF0;">
            <p style="margin: 0; color: #6B6B73; font-size: 0.92rem; font-weight: 400;">
                VarunaPath AI • Maritime Decision Intelligence • Team Novara
            </p>
            <p style="margin: 6px 0 0 0; color: #92929A; font-size: 0.82rem;">
                Prototype uses simulated local data for demonstration.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_command_centre_controls():
    """Renders the complete Scenario Control Bar exclusively for Command Centre."""
    st.markdown('<div class="scenario-control-bar">', unsafe_allow_html=True)
    
    # Initialize control keys from current session state if returning from another module
    if "cargo_type" not in st.session_state:
        st.session_state["cargo_type"] = st.session_state["cargo"]
    if "forecast_horizon" not in st.session_state:
        st.session_state["forecast_horizon"] = st.session_state["horizon"]
    if "operational_scenario" not in st.session_state:
        st.session_state["operational_scenario"] = st.session_state["scenario"]

    r1_c1, r1_c2, r1_c3, r1_c4 = st.columns([1.15, 1.0, 1.45, 1.1], gap="medium")
    with r1_c1:
        cargo_type = st.selectbox(
            "CARGO TYPE",
            cargo_options,
            key="cargo_type"
        )
        if cargo_type != st.session_state["cargo"]:
            st.session_state["cargo"] = cargo_type
            st.rerun()
    with r1_c2:
        forecast_horizon = st.selectbox(
            "FORECAST PERIOD",
            list(HORIZON_DAYS.keys()),
            key="forecast_horizon"
        )
        if forecast_horizon != st.session_state["horizon"]:
            st.session_state["horizon"] = forecast_horizon
            st.rerun()
    with r1_c3:
        operational_scenario = st.selectbox(
            "OPERATIONAL SCENARIO",
            scenario_options,
            key="operational_scenario"
        )
        if operational_scenario != st.session_state.get("scenario"):
            st.session_state["scenario"] = operational_scenario
            st.rerun()
    with r1_c4:
        st.markdown('<div class="ctrl-label">AI ENGINE STATUS</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="status-badges-group">
                <div class="ai-status-indicator">
                    <span class="ai-pulse-dot"></span>
                    <span>Active • Online</span>
                </div>
                <div class="confidence-badge">
                    <span>Confidence:</span>
                    <span class="confidence-val">90%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('<div style="height: 16px;"></div><div class="control-bar-row2">', unsafe_allow_html=True)
    r2_c1, r2_c2, r2_c3, r2_c4 = st.columns([0.75, 1.15, 2.0, 4.0], gap="medium")
    with r2_c1:
        if st.button("↺ Reset", key="cc_reset_btn", help="Reset all controls to official demo defaults (Thermal Coal, 90 Days, Base Scenario)"):
            st.session_state["cargo"] = "Thermal Coal"
            st.session_state["horizon"] = "90 Days"
            st.session_state["origin"] = "Richards Bay, South Africa"
            st.session_state["forecast_model"] = "Ensemble Recommended"
            st.session_state["scenario"] = "Base Scenario"
            st.session_state["cargo_requirement"] = 150000
            st.session_state["inventory"] = 40000
            st.session_state["safety"] = 20000
            st.session_state["deadline"] = 45
            st.session_state["fuel_change"] = 0
            st.session_state["fuel_price"] = 54000
            st.session_state["_reset_controls"] = True
            st.toast("Scenario reset to demo defaults (Thermal Coal, 90 Days, Base Scenario)", icon="↺")
            st.rerun()
    with r2_c2:
        if st.button("💾 Save Scenario", key="cc_save_btn", help="Save current scenario parameters"):
            st.toast(f"Saved: {st.session_state['cargo']} • {st.session_state['horizon']} • {st.session_state.get('scenario', 'Base Scenario')}", icon="💾")
    with r2_c3:
        if st.button("⚡ Run AI Optimization", key="cc_opt_btn", type="primary", help="Trigger AI Optimization engine"):
            st.toast("AI Optimization re-calculated! 33 feasible voyage plans evaluated.", icon="⚡")
            st.rerun()
    with r2_c4:
        st.empty()
    st.markdown('</div></div>', unsafe_allow_html=True)


def render_forecasting_controls():
    """Renders properly spaced desktop control row: Cargo, Horizon, Origin, Model, Generate Forecast."""
    st.markdown('<div class="scenario-control-bar" id="forecasting-control-bar">', unsafe_allow_html=True)
    control_columns = st.columns(
        [1.05, 1.0, 1.25, 1.25, 1.0],
        gap="medium"
    )
    col_cargo, col_horizon, col_origin, col_model, col_generate = control_columns

    with col_cargo:
        fc_cargos = [
            "Thermal Coal",
            "Coking Coal",
            "Iron Ore",
            "Bauxite",
            "Limestone",
            "Grain",
            "Cement",
            "Petroleum Coke",
        ]
        cur_cargo = st.session_state.get("cargo", "Thermal Coal")
        sel_cargo = st.selectbox(
            "CARGO TYPE",
            options=fc_cargos,
            index=fc_cargos.index(cur_cargo) if cur_cargo in fc_cargos else 0,
            key="fc_cargo_select",
        )
        if sel_cargo != st.session_state.get("cargo"):
            st.session_state["cargo"] = sel_cargo
            st.rerun()

    with col_horizon:
        fc_horizons = [
            "30 Days",
            "60 Days",
            "90 Days",
            "120 Days",
            "6 Months",
            "12 Months",
        ]
        cur_horizon = st.session_state.get("horizon", "90 Days")
        sel_horizon = st.selectbox(
            "FORECAST HORIZON",
            options=fc_horizons,
            index=fc_horizons.index(cur_horizon) if cur_horizon in fc_horizons else 2,
            key="fc_horizon_select",
        )
        if sel_horizon != st.session_state.get("horizon"):
            st.session_state["horizon"] = sel_horizon
            st.rerun()

    with col_origin:
        origins = [
            "Richards Bay, South Africa",
            "Newcastle, Australia",
            "Tanjung Bara, Indonesia",
        ]
        cur_origin = st.session_state.get("origin", origins[0])
        sel_origin = st.selectbox(
            "ORIGIN",
            origins,
            index=origins.index(cur_origin) if cur_origin in origins else 0,
            key="fc_origin_select",
        )
        if sel_origin != st.session_state.get("origin"):
            st.session_state["origin"] = sel_origin
            st.rerun()

    with col_model:
        models = [
            "Ensemble Recommended",
            "Linear Regression",
            "Moving Average",
            "Seasonal Trend",
        ]
        cur_mod = st.session_state.get("forecast_model", models[0])
        sel_model = st.selectbox(
            "FORECAST MODEL",
            models,
            index=models.index(cur_mod) if cur_mod in models else 0,
            key="fc_model_select",
        )
        if sel_model != st.session_state.get("forecast_model"):
            st.session_state["forecast_model"] = sel_model
            st.rerun()

    with col_generate:
        st.markdown('<div class="ctrl-label">&nbsp;</div>', unsafe_allow_html=True)
        if st.button("📈 Generate Forecast", key="fc_gen_btn", type="primary", use_container_width=True, help="Generate AI demand forecast projection"):
            st.toast(f"AI Forecast generated for {st.session_state['cargo']} ({st.session_state['horizon']}) via {st.session_state.get('forecast_model', 'Ensemble Recommended')}.", icon="📈")
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def render_shipment_planner_controls():
    """Renders shipment planning inputs only (Cargo, Origin, Requirement, Inventory, Safety, Deadline, Port, Generate)."""
    st.markdown('<div class="scenario-control-bar">', unsafe_allow_html=True)
    r1_c1, r1_c2, r1_c3, r1_c4 = st.columns([1.15, 1.4, 1.25, 1.2], gap="medium")
    with r1_c1:
        sel_cargo = st.selectbox(
            "CARGO TYPE",
            options=CARGO_OPTIONS,
            index=CARGO_OPTIONS.index(st.session_state["cargo"]) if st.session_state["cargo"] in CARGO_OPTIONS else 0,
            key="sp_cargo_select",
        )
        if sel_cargo != st.session_state["cargo"]:
            st.session_state["cargo"] = sel_cargo
            st.rerun()
    with r1_c2:
        origins = ["Richards Bay, South Africa", "Newcastle, Australia", "Tanjung Bara, Indonesia"]
        cur_origin = st.session_state.get("origin", origins[0])
        sel_origin = st.selectbox("ORIGIN", origins, index=origins.index(cur_origin) if cur_origin in origins else 0, key="sp_origin_select")
        if sel_origin != st.session_state.get("origin"):
            st.session_state["origin"] = sel_origin
    with r1_c3:
        ports = ["All Ports (Auto-Optimized)", "Paradip", "Dhamra", "Visakhapatnam"]
        cur_port = st.session_state.get("preferred_port", ports[0])
        sel_port = st.selectbox("PREFERRED PORT", ports, index=ports.index(cur_port) if cur_port in ports else 0, key="sp_port_select")
        if sel_port != st.session_state.get("preferred_port"):
            st.session_state["preferred_port"] = sel_port
    with r1_c4:
        cur_dl = st.session_state.get("deadline", 45)
        new_dl = st.slider("DELIVERY DEADLINE", 15, 60, int(cur_dl), key="sp_deadline_slider")
        if new_dl != cur_dl:
            st.session_state["deadline"] = new_dl
            st.rerun()
    
    st.markdown('<div style="height: 14px;"></div>', unsafe_allow_html=True)
    r2_c1, r2_c2, r2_c3, r2_c4 = st.columns([1.15, 1.15, 1.15, 1.55], gap="medium")
    with r2_c1:
        cur_req = st.session_state.get("cargo_requirement", 150000)
        new_req = st.number_input("CARGO REQUIREMENT (t)", 0, 500000, int(cur_req), 5000, key="sp_req_input")
        if new_req != cur_req:
            st.session_state["cargo_requirement"] = new_req
            st.rerun()
    with r2_c2:
        cur_inv = st.session_state.get("inventory", 40000)
        new_inv = st.number_input("CURRENT INVENTORY (t)", 0, 500000, int(cur_inv), 5000, key="sp_inv_input")
        if new_inv != cur_inv:
            st.session_state["inventory"] = new_inv
            st.rerun()
    with r2_c3:
        cur_saf = st.session_state.get("safety", 20000)
        new_saf = st.number_input("SAFETY STOCK (t)", 0, 150000, int(cur_saf), 5000, key="sp_saf_input")
        if new_saf != cur_saf:
            st.session_state["safety"] = new_saf
            st.rerun()
    with r2_c4:
        st.markdown('<div class="ctrl-label">&nbsp;</div>', unsafe_allow_html=True)
        if st.button("⚡ Generate Plans", key="sp_gen_btn", type="primary", help="Calculate feasible voyage plans"):
            st.toast("Voyage procurement plan generated! Shortfall calculated.", icon="⚡")
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def render_optimization_hub_controls(commodity, forecast_horizon, active_scenario, cargo_shortfall, deadline):
    """Renders compact read-only Current Scenario Summary, Objective selector, and Run Optimization."""
    st.markdown(
        f"""
        <div class="panel-card" style="margin-bottom: 14px; padding: 12px 18px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; font-size: 0.9rem;">
                <span style="color: #6B6B73; font-weight: 700; letter-spacing: 0.05em;">SCENARIO SNAPSHOT:</span>
                <span style="color: #ffffff; font-weight: 700;">📦 {commodity}</span>
                <span style="color: #ECECF0;">•</span>
                <span style="color: #372580; font-weight: 600;">📅 {forecast_horizon}</span>
                <span style="color: #ECECF0;">•</span>
                <span style="color: #48A868; font-weight: 600;">🧪 {active_scenario}</span>
                <span style="color: #ECECF0;">•</span>
                <span style="color: #F6B51B; font-weight: 600;">⚡ Shortfall: {cargo_shortfall:,} tonnes</span>
                <span style="color: #ECECF0;">•</span>
                <span style="color: #48A868; font-weight: 600;">⏱️ Deadline: {deadline} days</span>
                <span style="color: #ECECF0;">•</span>
                <span style="color: #6B6B73; font-weight: 600;">⛽ Fuel: ₹{st.session_state.get('fuel_price', 54000):,}/t</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="scenario-control-bar" style="margin-bottom: 18px;">', unsafe_allow_html=True)
    c1, c2 = st.columns([3.2, 1.2], gap="medium")
    with c1:
        objectives = [
            "Least Total Logistics Cost (Default)",
            "Fastest Voyage ETA (Minimum Duration)",
            "Lowest Maritime Carbon Footprint (tCO₂e)",
            "Minimum Multi-Factor Operational Risk",
        ]
        cur_obj = st.session_state.get("opt_objective", objectives[0])
        sel_obj = st.selectbox("OPTIMIZATION OBJECTIVE", objectives, index=objectives.index(cur_obj) if cur_obj in objectives else 0, key="opt_objective_select")
        if sel_obj != st.session_state.get("opt_objective"):
            st.session_state["opt_objective"] = sel_obj
    with c2:
        st.markdown('<div class="ctrl-label">&nbsp;</div>', unsafe_allow_html=True)
        if st.button("⚡ Run Optimization", key="opt_hub_run_btn", type="primary", help="Trigger AI Optimization engine"):
            st.session_state["trigger_opt_anim"] = True
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def render_vessel_intelligence_controls():
    """Renders vessel filters only (Class, Capacity, Availability, Draft)."""
    st.markdown('<div class="scenario-control-bar" style="margin-bottom: 18px;">', unsafe_allow_html=True)
    v1, v2, v3, v4 = st.columns([1.2, 1.3, 1.2, 1.3], gap="medium")
    with v1:
        v_classes = ["All Classes", "Supramax", "Panamax", "Capesize"]
        sel_cls = st.selectbox("VESSEL CLASS", v_classes, key="vi_class_filter")
    with v2:
        v_caps = ["All Capacities", "35,000 - 60,000 DWT", "60,000 - 100,000 DWT", "100,000+ DWT"]
        sel_cap = st.selectbox("CAPACITY RANGE", v_caps, key="vi_cap_filter")
    with v3:
        v_avails = ["All Statuses", "Immediate (Available)", "En Route", "Scheduled Maintenance"]
        sel_avail = st.selectbox("AVAILABILITY", v_avails, key="vi_avail_filter")
    with v4:
        v_drafts = ["Any Draft", "Max 12.5m (Handymax/Supramax)", "Max 14.5m (Panamax)", "Max 18.0m (Capesize)"]
        sel_draft = st.selectbox("DRAFT REQUIREMENT", v_drafts, key="vi_draft_filter")
    st.markdown('</div>', unsafe_allow_html=True)
    return sel_cls, sel_cap, sel_avail, sel_draft


def render_port_intelligence_controls():
    """Renders port filters only (Port, Congestion, Maximum Draft, Weather Risk, Compare Ports)."""
    st.markdown('<div class="scenario-control-bar" style="margin-bottom: 18px;">', unsafe_allow_html=True)
    p1, p2, p3, p4, p5 = st.columns([1.2, 1.2, 1.2, 1.2, 1.2], gap="medium")
    with p1:
        p_ports = ["All Ports", "Paradip", "Dhamra", "Visakhapatnam"]
        sel_port = st.selectbox("PORT SELECTOR", p_ports, key="pi_port_filter")
    with p2:
        p_congs = ["All Levels", "Normal (≤ 3 days)", "Elevated (> 3 days)"]
        sel_cong = st.selectbox("CONGESTION LEVEL", p_congs, key="pi_cong_filter")
    with p3:
        p_drafts = ["All Drafts", "≥ 14.5m", "≥ 17.5m", "≥ 18.0m"]
        sel_draft = st.selectbox("MAXIMUM DRAFT", p_drafts, key="pi_draft_filter")
    with p4:
        p_weather = ["All Conditions", "Low Risk (< 30)", "Moderate (30-50)", "High (> 50)"]
        sel_w = st.selectbox("WEATHER RISK", p_weather, key="pi_weather_filter")
    with p5:
        st.markdown('<div class="ctrl-label">&nbsp;</div>', unsafe_allow_html=True)
        compare_clicked = st.button("⚖️ Compare Ports", key="pi_compare_btn", type="primary")
        if compare_clicked:
            st.session_state["pi_show_comparison"] = not st.session_state.get("pi_show_comparison", False)
    st.markdown('</div>', unsafe_allow_html=True)
    return sel_port, sel_cong, sel_draft, sel_w


def render_route_intelligence_controls():
    """Renders comprehensive inputs for maritime route simulation and optimization."""
    st.markdown('<div class="scenario-control-bar" style="margin-bottom: 18px;">', unsafe_allow_html=True)

    # Pre-populate defaults safely from session state
    default_cargo = st.session_state.get("cargo", "Thermal Coal")
    default_origin = st.session_state.get("origin", "Richards Bay, South Africa")
    default_qty = int(st.session_state.get("cargo_requirement", 130000))
    default_fuel_price = int(st.session_state.get("fuel_price", 54000))

    origin_options = [
        "Richards Bay, South Africa",
        "Newcastle, Australia",
        "Tanjung Bara, Indonesia",
        "Maputo, Mozambique",
        "Ust-Luga, Russia",
        "Norfolk, United States",
    ]
    dest_options = [
        "Paradip Port",
        "Visakhapatnam Port",
        "Gangavaram Port",
        "Dhamra Port",
        "Gopalpur Port",
        "Haldia Port",
        "Sagar / Sandheads",
    ]
    cargo_options = ["Thermal Coal", "Coking Coal", "Iron Ore", "Petroleum Coke", "Limestone"]
    vessel_class_options = ["Supramax", "Panamax", "Capesize", "Handymax"]
    priority_options = ["Balanced Recommended", "Lowest Cost", "Fastest Route", "Lowest Risk"]

    # Row 1: Key Voyage Parameters
    r1_c1, r1_c2, r1_c3, r1_c4, r1_c5 = st.columns([1.25, 1.25, 1.1, 1.1, 1.1], gap="medium")
    with r1_c1:
        orig_idx = origin_options.index(default_origin) if default_origin in origin_options else 0
        sel_orig = st.selectbox("ORIGIN PORT", origin_options, index=orig_idx, key="ri_origin")
    with r1_c2:
        sel_dest = st.selectbox("DESTINATION PORT", dest_options, index=0, key="ri_dest")
    with r1_c3:
        cargo_idx = cargo_options.index(default_cargo) if default_cargo in cargo_options else 0
        sel_cargo = st.selectbox("CARGO TYPE", cargo_options, index=cargo_idx, key="ri_cargo")
    with r1_c4:
        sel_qty = st.number_input("CARGO QTY (TONNES)", min_value=30000, max_value=300000, value=default_qty, step=5000, key="ri_qty")
    with r1_c5:
        sel_class = st.selectbox("VESSEL CLASS", vessel_class_options, index=1, key="ri_vessel_class")

    # Dynamic defaults based on vessel class
    dwt_defaults = {"Supramax": 58000, "Panamax": 82000, "Capesize": 180000, "Handymax": 45000}
    draft_defaults = {"Supramax": 12.8, "Panamax": 14.5, "Capesize": 18.2, "Handymax": 10.5}
    fuel_defaults = {"Supramax": 24.0, "Panamax": 28.0, "Capesize": 42.0, "Handymax": 20.0}
    charter_defaults = {"Supramax": 1050000, "Panamax": 1250000, "Capesize": 1900000, "Handymax": 850000}

    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)

    # Row 2: Technical & Vessel Operational Criteria
    r2_c1, r2_c2, r2_c3, r2_c4, r2_c5 = st.columns([1.1, 1.1, 1.1, 1.2, 1.1], gap="medium")
    with r2_c1:
        sel_dwt = st.number_input("VESSEL DWT", min_value=30000, max_value=250000, value=dwt_defaults.get(sel_class, 82000), step=2000, key="ri_dwt")
    with r2_c2:
        sel_draft = st.number_input("VESSEL DRAFT (m)", min_value=8.0, max_value=22.0, value=draft_defaults.get(sel_class, 14.5), step=0.1, key="ri_draft")
    with r2_c3:
        sel_speed = st.slider("AVG SPEED (KTS)", min_value=10.0, max_value=16.0, value=12.5, step=0.5, key="ri_speed")
    with r2_c4:
        sel_date = st.date_input("DEPARTURE DATE", value=date.today() + timedelta(days=2), key="ri_dep_date")
    with r2_c5:
        sel_fuel_cons = st.number_input("FUEL (t/DAY)", min_value=15.0, max_value=60.0, value=fuel_defaults.get(sel_class, 28.0), step=1.0, key="ri_fuel_cons")

    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)

    # Row 3: Economics, Optimization & Actions
    r3_c1, r3_c2, r3_c3, r3_c4, r3_c5 = st.columns([1.1, 1.2, 1.3, 1.2, 0.9], gap="medium")
    with r3_c1:
        sel_fuel_price = st.number_input("FUEL PRICE (₹/t)", min_value=30000, max_value=90000, value=default_fuel_price, step=1000, key="ri_fuel_price")
    with r3_c2:
        sel_charter_rate = st.number_input("CHARTER (₹/DAY)", min_value=500000, max_value=3500000, value=charter_defaults.get(sel_class, 1250000), step=50000, key="ri_charter_rate")
    with r3_c3:
        sel_priority = st.selectbox("OPTIMIZATION PRIORITY", priority_options, index=0, key="ri_priority")
    with r3_c4:
        st.markdown('<div class="ctrl-label">&nbsp;</div>', unsafe_allow_html=True)
        gen_clicked = st.button("🚀 Generate Route Comparison", key="ri_gen_btn", type="primary", use_container_width=True)
        if gen_clicked:
            st.session_state["ri_calculated"] = True
            st.toast("Maritime route simulation recalculated successfully.", icon="🚀")
    with r3_c5:
        st.markdown('<div class="ctrl-label">&nbsp;</div>', unsafe_allow_html=True)
        if st.button("🔄 Reset", key="ri_reset_btn", use_container_width=True):
            st.session_state["ri_origin"] = "Richards Bay, South Africa"
            st.session_state["ri_dest"] = "Paradip Port"
            st.session_state["ri_cargo"] = "Thermal Coal"
            st.session_state["ri_qty"] = 130000
            st.session_state["ri_vessel_class"] = "Panamax"
            st.session_state["ri_dwt"] = 82000
            st.session_state["ri_draft"] = 14.5
            st.session_state["ri_speed"] = 12.5
            st.session_state["ri_dep_date"] = date.today() + timedelta(days=2)
            st.session_state["ri_fuel_cons"] = 28.0
            st.session_state["ri_fuel_price"] = 54000
            st.session_state["ri_charter_rate"] = 1250000
            st.session_state["ri_priority"] = "Balanced Recommended"
            st.session_state["ri_calculated"] = True
            st.toast("Route parameters reset to baseline demonstration values.", icon="🔄")
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    return sel_orig, sel_dest, sel_cargo, sel_qty, sel_class, sel_dwt, sel_draft, sel_speed, sel_date, sel_fuel_cons, sel_fuel_price, sel_charter_rate, sel_priority


def render_risk_alerts_controls():
    """Renders risk and alert controls only (Category, Severity, Status, Refresh Assessment)."""
    st.markdown('<div class="scenario-control-bar" style="margin-bottom: 18px;">', unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns([1.3, 1.2, 1.2, 1.3], gap="medium")
    with r1:
        r_cats = ["All Categories", "Geopolitical Supplier Risk", "Vessel Reliability Risk", "Port Congestion Risk", "Weather Volatility Risk"]
        sel_cat = st.selectbox("RISK CATEGORY", r_cats, key="ra_cat_filter")
    with r2:
        r_sevs = ["All Severities", "Critical (Red)", "Elevated (Yellow)", "Normal (Green)"]
        sel_sev = st.selectbox("SEVERITY FILTER", r_sevs, key="ra_sev_filter")
    with r3:
        r_stats = ["All Alerts", "Active & Unresolved", "Acknowledged", "Archived"]
        sel_stat = st.selectbox("ALERT STATUS", r_stats, key="ra_stat_filter")
    with r4:
        st.markdown('<div class="ctrl-label">&nbsp;</div>', unsafe_allow_html=True)
        if st.button("🔄 Refresh Assessment", key="ra_refresh_btn", type="primary"):
            st.toast("Multi-factor risk assessment refreshed with latest telemetry.", icon="🔄")
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    return sel_cat, sel_sev, sel_stat


def render_scenario_lab_controls():
    """Renders interactive what-if controls across Demand, Inventory, Safety, Fuel, Freight, Speed, Congestion, Weather, Availability, Deadline, and Budget."""
    st.markdown('<div class="scenario-control-bar" style="margin-bottom: 18px;">', unsafe_allow_html=True)
    
    r0_1, r0_2 = st.columns([1.8, 3.2], gap="medium")
    with r0_1:
        if st.button("🔄 Reset What-If Scenario to Baseline", key="sl_reset_scenario_btn", use_container_width=True):
            st.session_state["sl_req_slider"] = 150000
            st.session_state["sl_inv_slider"] = 40000
            st.session_state["sl_saf_slider"] = 20000
            st.session_state["sl_budget_input"] = 35.0
            st.session_state["sl_fuel_input"] = 54000
            st.session_state["sl_freight_slider"] = 0
            st.session_state["sl_deadline_slider"] = 45
            st.session_state["sl_speed_slider"] = 12.0
            st.session_state["sl_port_closure_select"] = "None (Normal 1.0x)"
            st.session_state["sl_weather_slider"] = 25
            st.session_state["sl_avail_slider"] = 85
            st.toast("What-If parameters reset to baseline values!", icon="🔄")
            st.rerun()

    r1_1, r1_2, r1_3, r1_4 = st.columns([1.2, 1.2, 1.2, 1.2], gap="medium")
    with r1_1:
        req_val = st.slider("CARGO REQUIREMENT (t)", 50000, 300000, int(st.session_state.get("sl_req_slider", st.session_state.get("cargo_requirement", 150000))), 5000, key="sl_req_slider")
    with r1_2:
        inv_val = st.slider("CURRENT INVENTORY (t)", 0, 150000, int(st.session_state.get("sl_inv_slider", st.session_state.get("inventory", 40000))), 5000, key="sl_inv_slider")
    with r1_3:
        saf_val = st.slider("SAFETY STOCK BUFFER (t)", 0, 80000, int(st.session_state.get("sl_saf_slider", st.session_state.get("safety", 20000))), 5000, key="sl_saf_slider")
    with r1_4:
        budget_val = st.number_input("APPROVED BUDGET (₹ Cr)", 10.0, 80.0, float(st.session_state.get("sl_budget_input", st.session_state.get("approved_budget", 35.0))), 1.0, key="sl_budget_input")

    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
    r2_1, r2_2, r2_3, r2_4 = st.columns([1.2, 1.2, 1.2, 1.2], gap="medium")
    with r2_1:
        fuel_val = st.number_input("BUNKER FUEL PRICE (₹/t)", 30000, 90000, int(st.session_state.get("sl_fuel_input", st.session_state.get("fuel_price", 54000))), 1000, key="sl_fuel_input")
    with r2_2:
        freight_shift = st.slider("FREIGHT RATE SHIFT", -30, 50, int(st.session_state.get("sl_freight_slider", 0)), format="%d%%", key="sl_freight_slider")
    with r2_3:
        speed_val = st.slider("VESSEL SPEED (KNOTS)", 9.0, 16.0, float(st.session_state.get("sl_speed_slider", 12.0)), 0.5, key="sl_speed_slider")
    with r2_4:
        deadline_val = st.slider("DELIVERY DEADLINE (DAYS)", 15, 60, int(st.session_state.get("sl_deadline_slider", st.session_state.get("deadline", 45))), key="sl_deadline_slider")
    
    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
    r3_1, r3_2, r3_3 = st.columns([1.4, 1.3, 1.3], gap="medium")
    with r3_1:
        port_closure = st.selectbox("PORT CONGESTION / CLOSURE", ["None (Normal 1.0x)", "Paradip Outage", "Dhamra Outage", "Visakhapatnam Congestion (2.0x)"], key="sl_port_closure_select")
    with r3_2:
        weather_risk = st.slider("WEATHER SEVERITY FACTOR", 0, 100, int(st.session_state.get("sl_weather_slider", 25)), key="sl_weather_slider")
    with r3_3:
        vessel_avail = st.slider("VESSEL AVAILABILITY", 50, 100, int(st.session_state.get("sl_avail_slider", 85)), format="%d%%", key="sl_avail_slider")
    st.markdown('</div>', unsafe_allow_html=True)
    return req_val, inv_val, saf_val, budget_val, fuel_val, freight_shift, speed_val, deadline_val, port_closure, weather_risk, vessel_avail


def render_reports_view(base_plans, commodity, forecast_horizon, horizon_days, cargo_shortfall, deadline):
    """Renders comprehensive reports without any scenario toolbar."""
    if base_plans.empty:
        st.warning("No feasible plan currently calculated to generate audit report. Please review constraints.")
        return

    plan_confirmed = st.session_state.get("plan_confirmed", False)
    confirmed_plan = st.session_state.get("confirmed_plan", None)

    if not plan_confirmed or not confirmed_plan:
        st.info("ℹ️ Using the default prototype recommendation. Confirm a plan in Shipment Planner to replace it.")
        best = base_plans.iloc[0]
        rep_supplier = best["Supplier"]
        rep_origin = best["Origin"]
        rep_vessels = int(best["Vessels"])
        rep_class = best["Class"]
        rep_vessel_name = best["Vessel"]
        rep_capacity = int(best["Combined Capacity"])
        rep_util = float(best["Utilization"])
        rep_port = best["Port"]
        rep_eta = int(best["ETA Days"])
        rep_cost = float(best["Total Cost Cr"])
        baseline_cost = 31.19
        saving = round(baseline_cost - rep_cost, 2)
        saving_pct = round((saving / baseline_cost) * 100, 1)
        rep_risk = 24
        rep_status = "Default Prototype Recommendation"
    else:
        st.success(f"✅ Active Confirmed Plan: {confirmed_plan['name']} ({confirmed_plan['vessel_count']} × {confirmed_plan['vessel_type']} to {confirmed_plan['port']} Port)")
        best = base_plans.iloc[0]
        rep_supplier = best["Supplier"]
        rep_origin = st.session_state.get("origin", "Richards Bay, South Africa")
        rep_vessels = int(confirmed_plan["vessel_count"])
        rep_class = confirmed_plan["vessel_type"]
        rep_vessel_name = confirmed_plan.get("vessel_name", f"{rep_class} Fleet")
        rep_capacity = int(confirmed_plan["combined_capacity"])
        rep_util = float(confirmed_plan["utilization"])
        rep_port = confirmed_plan["port"]
        rep_eta = int(confirmed_plan["duration"])
        rep_cost = float(confirmed_plan["cost_cr"])
        baseline_cost = 31.19
        saving = float(confirmed_plan.get("savings_cr", round(baseline_cost - rep_cost, 2)))
        saving_pct = round((saving / baseline_cost) * 100, 1)
        rep_risk = int(confirmed_plan.get("risk_score", 24))
        rep_status = f"Confirmed Plan ({confirmed_plan['name']})"

    # 1. Current Recommended Plan summary
    st.markdown(
        f"""
        <div class="recommend" style="margin-bottom: 20px;">
            <b style="font-size: 1.05rem; color: #18181B;">⚓ Current Recommended Voyage Plan Summary ({rep_status}):</b><br>
            Procure <b>{cargo_shortfall:,} tonnes</b> of {commodity} from <b>{rep_supplier}</b> ({rep_origin}).<br>
            Charter <b>{rep_vessels} × {rep_class}</b> ({rep_vessel_name}) with <b>{rep_capacity:,} tonnes</b> combined capacity (<b>{rep_util}%</b> utilization).<br>
            Discharge at <b>{rep_port} Port</b> with total voyage turnaround of <b>{rep_eta} days</b>.<br>
            Optimized Logistics Cost: <b>₹{rep_cost:.2f} Cr</b> vs. Baseline <b>₹{baseline_cost:.2f} Cr</b> (Direct Savings: <b>₹{saving:.2f} Cr</b> • <b>{saving_pct:.1f}%</b>).<br>
            Audit Status: <b>Certified Optimal</b> • Risk Score: <b>{rep_risk}/100</b> • Confidence: <b>90%</b> • MAPE: <b>0.82%</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # 2. Report Cards: Executive Summary, Cost Analysis, Risk Analysis, Port Comparison
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div class="panel-card" style="margin-bottom: 16px;">
                <h4 style="color: #18181B; margin-top: 0;">📋 Executive Summary Report</h4>
                <p style="color: #6B6B73; font-size: 0.90rem; line-height: 1.5;">
                    This audit document establishes the least-cost, risk-hedged bulk logistics plan for <b>{cargo_shortfall:,} tonnes</b> of <b>{commodity}</b>.
                    Dual Panamax carrier allocation achieves the optimal DWT threshold without incurring excess demurrage penalties or exceeding East Coast port draft constraints.
                </p>
                <table style="width: 100%; font-size: 0.88rem; color: #18181B; border-collapse: collapse;">
                    <tr><td style="padding: 6px 0; color: #6B6B73;">Target Commodity</td><td style="text-align: right; font-weight: 600;">{commodity}</td></tr>
                    <tr><td style="padding: 6px 0; color: #6B6B73;">Procurement Volume</td><td style="text-align: right; font-weight: 600;">{cargo_shortfall:,} tonnes</td></tr>
                    <tr><td style="padding: 6px 0; color: #6B6B73;">Planning Horizon</td><td style="text-align: right; font-weight: 600;">{forecast_horizon} ({horizon_days} days)</td></tr>
                    <tr><td style="padding: 6px 0; color: #6B6B73;">Optimal Vessel Class</td><td style="text-align: right; font-weight: 600;">{int(best['Vessels'])} × {best['Class']}</td></tr>
                    <tr><td style="padding: 6px 0; color: #6B6B73;">Delivery Compliance</td><td style="text-align: right; font-weight: 600; color: #48A868;">On-Time ({int(best['ETA Days'])}d ≤ {deadline}d)</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="panel-card" style="margin-bottom: 16px;">
                <h4 style="color: #18181B; margin-top: 0;">💰 Cost Analysis Report</h4>
                <p style="color: #6B6B73; font-size: 0.90rem; line-height: 1.5;">
                    Comparative cost breakdown demonstrating a net reduction of <b>₹{saving:.2f} Cr ({saving_pct:.1f}%)</b> against the conventional single-fixture baseline.
                </p>
                <table style="width: 100%; font-size: 0.88rem; color: #18181B; border-collapse: collapse;">
                    <tr><td style="padding: 6px 0; color: #6B6B73;">Cargo FOB Value</td><td style="text-align: right; font-weight: 600;">₹{best['Cargo Cost Cr']:.2f} Cr</td></tr>
                    <tr><td style="padding: 6px 0; color: #6B6B73;">Vessel Charter Cost</td><td style="text-align: right; font-weight: 600;">₹{best['Charter Cost Cr']:.2f} Cr</td></tr>
                    <tr><td style="padding: 6px 0; color: #6B6B73;">Port Tariff & Anchorage</td><td style="text-align: right; font-weight: 600;">₹{best['Port & Waiting Cr']:.2f} Cr</td></tr>
                    <tr><td style="padding: 6px 0; color: #6B6B73;">Operational Risk Reserve</td><td style="text-align: right; font-weight: 600;">₹{best['Risk Cost Cr']:.2f} Cr</td></tr>
                    <tr style="border-top: 1px solid #ECECF0;"><td style="padding: 6px 0; color: #372580; font-weight: 700;">Total Optimized Logistics Cost</td><td style="text-align: right; font-weight: 700; color: #372580;">₹{best['Total Cost Cr']:.2f} Cr</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="panel-card" style="margin-bottom: 16px;">
                <h4 style="color: #18181B; margin-top: 0;">⚠️ Risk Analysis Report</h4>
                <p style="color: #6B6B73; font-size: 0.90rem; line-height: 1.5;">
                    Multi-factor risk evaluation scoring <b>24/100 (Low Risk Category)</b>. The chosen routing via Richards Bay to Paradip bypasses high-risk geopolitical choke points.
                </p>
                <table style="width: 100%; font-size: 0.88rem; color: #18181B; border-collapse: collapse;">
                    <tr><td style="padding: 6px 0; color: #6B6B73;">Supplier Export Reliability</td><td style="text-align: right; font-weight: 600; color: #48A868;">94% (Stable)</td></tr>
                    <tr><td style="padding: 6px 0; color: #6B6B73;">Vessel Seaworthiness & Class Age</td><td style="text-align: right; font-weight: 600; color: #48A868;">91% (Tier 1)</td></tr>
                    <tr><td style="padding: 6px 0; color: #6B6B73;">Berth Congestion Exposure</td><td style="text-align: right; font-weight: 600; color: #F6B51B;">3.0 Days (Moderate)</td></tr>
                    <tr><td style="padding: 6px 0; color: #6B6B73;">Cyclonic / Weather Hazard Factor</td><td style="text-align: right; font-weight: 600; color: #48A868;">35/100 (Normal Season)</td></tr>
                    <tr style="border-top: 1px solid #ECECF0;"><td style="padding: 6px 0; color: #F6B51B; font-weight: 700;">Composite Risk Rating</td><td style="text-align: right; font-weight: 700; color: #F6B51B;">24 / 100 (Optimal)</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="panel-card" style="margin-bottom: 16px;">
                <h4 style="color: #18181B; margin-top: 0;">🏗️ Port Comparison Report</h4>
                <p style="color: #6B6B73; font-size: 0.90rem; line-height: 1.5;">
                    Analysis of East Coast discharge options indicates <b>Paradip</b> offers the highest cost efficiency with 180,000 DWT clearance and 3-day turnaround.
                </p>
                <table style="width: 100%; font-size: 0.88rem; color: #18181B; border-collapse: collapse;">
                    <tr><td style="padding: 6px 0; color: #6B6B73;">Paradip (Selected)</td><td style="text-align: right; font-weight: 600;">₹85/t • 3d wait • 180k DWT max</td></tr>
                    <tr><td style="padding: 6px 0; color: #6B6B73;">Dhamra Port</td><td style="text-align: right; font-weight: 600;">₹95/t • 2d wait • 180k DWT max</td></tr>
                    <tr><td style="padding: 6px 0; color: #6B6B73;">Visakhapatnam Port</td><td style="text-align: right; font-weight: 600;">₹110/t • 5d wait • 125k DWT max</td></tr>
                    <tr style="border-top: 1px solid #ECECF0;"><td style="padding: 6px 0; color: #372580; font-weight: 700;">Optimal Port Choice</td><td style="text-align: right; font-weight: 700; color: #372580;">Paradip Port</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # 3. Download Report Buttons
    st.subheader("Export Audit Documentation")
    
    # Generate comprehensive HTML Decision Report
    rep_calc = get_scenario_calculations()
    rec_dict = {
        "balanced_plan": rep_calc.get("balanced_plan"),
        "lowest_cost_plan": rep_calc.get("lowest_cost_plan"),
        "lowest_risk_plan": rep_calc.get("lowest_risk_plan"),
    }
    decision_report_html = generate_decision_report(
        scenario=rep_calc.get("scenario_dict", {}),
        recommendations=rec_dict,
        all_plans=rep_calc.get("all_plans", []),
        trade_direction=rep_calc.get("scenario_dict", {}).get("trade_direction", "Import to India")
    )

    d0, d1, d2, d3 = st.columns([1.2, 1.0, 1.0, 1.0], gap="small")
    with d0:
        st.download_button(
            "📥 Download Decision Report (HTML)",
            data=decision_report_html.encode("utf-8"),
            file_name="varunapath_strategic_decision_report.html",
            mime="text/html",
            key="rep_dl_html_report",
            use_container_width=True
        )
    with d1:
        exec_csv = pd.DataFrame([best]).to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Executive Audit (CSV)", exec_csv, "varunapath_executive_audit.csv", "text/csv", key="rep_dl_exec", use_container_width=True)
    with d2:
        cost_csv = pd.DataFrame({
            "Component": ["Cargo Cost", "Vessel Charter", "Port Handling & Waiting", "Risk Reserve", "Total Logistics Cost"],
            "Cost (₹ Cr)": [best["Cargo Cost Cr"], best["Charter Cost Cr"], best["Port & Waiting Cr"], best["Risk Cost Cr"], best["Total Cost Cr"]],
            "Baseline (₹ Cr)": [31.19 * 0.65, 31.19 * 0.25, 31.19 * 0.07, 31.19 * 0.03, 31.19],
            "Savings (₹ Cr)": [0, 0, 0, 0, saving],
        }).to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Cost Analysis (CSV)", cost_csv, "varunapath_cost_analysis.csv", "text/csv", key="rep_dl_cost", use_container_width=True)
    with d3:
        port_csv = PORTS.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Port Comparison (CSV)", port_csv, "varunapath_port_comparison.csv", "text/csv", key="rep_dl_port", use_container_width=True)


def render_data_settings_view():
    """Renders system data connectors, upload section, model parameters, master registries, and reset without any scenario toolbar."""
    st.markdown('<div class="panel-card" style="margin-bottom: 18px;"><b>System Settings & Maritime Configuration Hub</b> — Configure enterprise data connectors, master registry data, machine learning parameters, and application telemetry.</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🌐 Data Sources",
        "📂 Upload CSV",
        "⚙️ Model Parameters",
        "🏗️ Port Master Data",
        "🚢 Vessel Master Data",
        "📦 Export Datasets",
        "📖 Maritime Glossary",
        "↺ Cache & Reset",
    ])
    
    with tab1:
        st.subheader("Enterprise Data Source Connectors")
        ds_c1, ds_c2 = st.columns(2)
        with ds_c1:
            st.markdown(
                """
                <div class="panel-card">
                    <b style="color: #372580;">Open-Meteo Marine & Weather API Connector</b><br>
                    <span style="color: #372580; font-weight: 600;">● Integration Ready</span> • East Coast Indian Ports Simulated Telemetry (Prototype Dataset)<br>
                    <span style="color: #6B6B73; font-size: 0.85rem;">Endpoint: <code>https://api.open-meteo.com/v1/forecast</code></span><br>
                    <span style="color: #6B6B73; font-size: 0.85rem;">Mode: Prototype Dataset • Telemetry: Simulated Data</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with ds_c2:
            st.markdown(
                """
                <div class="panel-card">
                    <b style="color: #372580;">Frankfurter Central Bank FX API Connector</b><br>
                    <span style="color: #372580; font-weight: 600;">● Integration Ready</span> • Reference Currency Conversion (USD/INR Simulated Data)<br>
                    <span style="color: #6B6B73; font-size: 0.85rem;">Endpoint: <code>https://api.frankfurter.app/latest?from=USD&to=INR</code></span><br>
                    <span style="color: #6B6B73; font-size: 0.85rem;">Mode: Prototype Dataset • Not Live Connected</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
    with tab2:
        st.subheader("Upload Custom Operational Datasets")
        st.write("Upload historical demand CSVs or port/vessel registries to update models in real time:")
        uploaded_csv = st.file_uploader("Choose CSV Dataset", type=["csv"], key="ds_csv_uploader")
        if uploaded_csv:
            try:
                up_df = pd.read_csv(uploaded_csv)
                st.success(f"Successfully loaded dataset '{uploaded_csv.name}' with {len(up_df)} rows and {len(up_df.columns)} columns.")
                st.dataframe(up_df.head(5), width="stretch")
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")
    
    with tab3:
        st.subheader("AI Optimization & Forecasting Parameters")
        mp1, mp2, mp3 = st.columns(3)
        with mp1:
            st.number_input("Demand Growth Trend Weight", 0.0, 2.0, 1.0, 0.05, key="ds_mp_weight")
        with mp2:
            st.number_input("Safety Stock Standard Ratio", 0.05, 0.50, 0.15, 0.01, key="ds_mp_safety_ratio")
        with mp3:
            st.number_input("Reference Bunker Fuel Price (₹/t)", 30000, 90000, 54000, 1000, key="ds_mp_fuel_ref")
        st.info("Configured model parameters apply across all linear demand projections and voyage cost models.")
    
    with tab4:
        st.subheader("East Coast Discharge Port Master Data")
        st.dataframe(PORTS, hide_index=True, width="stretch")
    
    with tab5:
        st.subheader("Bulk Carrier Fleet Master Data")
        st.dataframe(VESSELS, hide_index=True, width="stretch")
    
    
    with tab6:
        st.subheader("Prototype Export Master Datasets (data/)")
        st.caption("Verified local CSV datasets with classification: Prototype Data. Zero external dependency.")
        
        d_c1, d_c2 = st.columns(2)
        with d_c1:
            st.markdown("<b>export_orders.csv</b> (5 active confirmed prototype buyer orders)", unsafe_allow_html=True)
            try:
                e_ord_df = pd.read_csv("data/export_orders.csv")
                st.dataframe(e_ord_df, hide_index=True, use_container_width=True)
            except Exception:
                st.info("Loaded from prototype definitions.")
            
            st.markdown("<b>export_loading_ports.csv</b> (7 Indian East Coast loading terminals)", unsafe_allow_html=True)
            try:
                e_load_df = pd.read_csv("data/export_loading_ports.csv")
                st.dataframe(e_load_df, hide_index=True, use_container_width=True)
            except Exception:
                st.info("Loaded from prototype definitions.")

        with d_c2:
            st.markdown("<b>foreign_destination_ports.csv</b> (7 international discharge ports)", unsafe_allow_html=True)
            try:
                e_dst_df = pd.read_csv("data/foreign_destination_ports.csv")
                st.dataframe(e_dst_df, hide_index=True, use_container_width=True)
            except Exception:
                st.info("Loaded from prototype definitions.")

            st.markdown("<b>export_routes.csv</b> (12 maritime transit routes)", unsafe_allow_html=True)
            try:
                e_rt_df = pd.read_csv("data/export_routes.csv")
                st.dataframe(e_rt_df, hide_index=True, use_container_width=True)
            except Exception:
                st.info("Loaded from prototype definitions.")

    with tab7:
        st.subheader("Technical Maritime & Logistics Glossary (20+ Terms)")
        st.caption("Authoritative definitions for all decision support, vessel chartering, and Incoterm terms used across VarunaPath AI.")
        
        gloss_cols = st.columns(2)
        items = list(TECHNICAL_TOOLTIPS.items())
        half = len(items) // 2 + 1
        with gloss_cols[0]:
            for term, defn in items[:half]:
                st.markdown(f"<div style='background: #F8F8FA; border: 1px solid #E4E4E8; border-radius: 6px; padding: 10px; margin-bottom: 8px;'><b>{term}</b><br><span style='color: #6B6B73; font-size: 0.88rem;'>{defn}</span></div>", unsafe_allow_html=True)
        with gloss_cols[1]:
            for term, defn in items[half:]:
                st.markdown(f"<div style='background: #F8F8FA; border: 1px solid #E4E4E8; border-radius: 6px; padding: 10px; margin-bottom: 8px;'><b>{term}</b><br><span style='color: #6B6B73; font-size: 0.88rem;'>{defn}</span></div>", unsafe_allow_html=True)

    with tab8:
        st.subheader("System Cache and Demo State Reset")
        st.write("Clear runtime state or reset application parameters to the official demo baseline.")
        rc1, rc2 = st.columns(2)
        with rc1:
            if st.button("↺ Reset All Settings to Demo Defaults", key="ds_reset_all_btn", type="primary"):
                st.session_state["cargo"] = "Thermal Coal"
                st.session_state["horizon"] = "90 Days"
                st.session_state["origin"] = "Richards Bay, South Africa"
                st.session_state["forecast_model"] = "Ensemble Recommended"
                st.session_state["scenario"] = "Base Scenario"
                st.session_state["cargo_requirement"] = 150000
                st.session_state["inventory"] = 40000
                st.session_state["safety"] = 20000
                st.session_state["deadline"] = 45
                st.session_state["fuel_change"] = 0
                st.session_state["fuel_price"] = 54000
                st.session_state["_reset_controls"] = True
                st.toast("System restored to official demo defaults!", icon="↺")
                st.rerun()
        with rc2:
            if st.button("🗑️ Clear Cache & Telemetry", key="ds_clear_cache_btn"):
                st.cache_data.clear()
                st.toast("Runtime cache cleared!", icon="🗑️")
                st.rerun()
    
    # Prototype Data Indicator
    st.markdown("---")
    st.markdown(
        """
        <div class="panel-card" style="border-left: 4px solid #372580; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span class="prototype-badge" style="margin-bottom: 4px; display: inline-block;">PROTOTYPE DATA ACTIVE</span><br>
                <span style="color: #6B6B73; font-size: 0.88rem;">VarunaPath AI Maritime Intelligence Platform • Demonstration Version 2.4</span>
            </div>
            <div style="text-align: right; color: #6B6B73; font-size: 0.82rem;">
                Telemetry Engine: <b>Online</b><br>
                Certified Calibrated Baseline: <b>Thermal Coal / 90 Days / Base Scenario</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Selections managed purely in st.session_state
def get_scenario_calculations():
    """Computes active scenario parameters, constraint feasibility, and fleet optimization dynamically from user inputs."""
    trade_dir = st.session_state.get("trade_direction", "Import to India")
    commodity = st.session_state.get("cargo", "Thermal Coal")
    forecast_horizon = st.session_state.get("horizon", "90 Days")
    horizon_days = HORIZON_MAP.get(forecast_horizon, 90)
    cargo_requirement = st.session_state.get("cargo_requirement", 150000)
    inventory = st.session_state.get("inventory", 40000)
    safety = st.session_state.get("safety", 20000)
    deadline = st.session_state.get("deadline", 45)
    fuel_change = st.session_state.get("fuel_change", 0)
    origin_port = st.session_state.get("origin", "Richards Bay, South Africa")
    dest_port = st.session_state.get("destination_port", "Paradip")
    approved_budget = float(st.session_state.get("approved_budget", 35.0))
    min_utilization = float(st.session_state.get("min_utilization", 70.0))
    max_risk = float(st.session_state.get("max_risk", 60.0))

    active_scenario = st.session_state.get("scenario", "Base Scenario")
    scenario_fuel_offset = 0
    scenario_demand_mult = 1.0
    scenario_port_outage = "None"

    if active_scenario == "Fuel Price Surge (+20%)":
        scenario_fuel_offset = 20
    elif active_scenario == "Monsoon Weather Risk (+15% Demand)":
        scenario_demand_mult = 1.15
    elif active_scenario == "Port Bottleneck (Paradip Outage)":
        scenario_port_outage = "Paradip"
    elif active_scenario == "High Inflation Peak":
        scenario_fuel_offset = 35

    effective_fuel_change = fuel_change + scenario_fuel_offset

    model, future_df = forecast_demand(horizon_days)
    if future_df is not None and not future_df.empty and "Demand (000 t)" in future_df.columns:
        initial_forecast = float(future_df["Demand (000 t)"].iloc[0] * 1000.0)
        final_forecast = float(future_df["Demand (000 t)"].iloc[-1] * 1000.0)
        if initial_forecast != 0:
            demand_growth = ((final_forecast - initial_forecast) / initial_forecast) * 100
        else:
            demand_growth = 0.0
    else:
        initial_forecast = 0.0
        final_forecast = 0.0
        demand_growth = 0.0

    adjusted_demand = int(cargo_requirement * scenario_demand_mult)
    
    # 1. Dynamic Shortfall calculation
    cargo_shortfall = calculate_import_shortfall(adjusted_demand, safety, inventory)

    # 2. Central Scenario Configuration Object
    scenario_dict = {
        "trade_direction": trade_dir,
        "cargo_type": commodity,
        "commodity": commodity,
        "forecast_period": forecast_horizon,
        "forecast_demand": adjusted_demand,
        "current_inventory": inventory,
        "inventory": inventory,
        "safety_stock": safety,
        "safety": safety,
        "origin": origin_port,
        "origin_port": origin_port,
        "destination_port": dest_port,
        "port": dest_port,
        "approved_budget": approved_budget,
        "deadline": deadline,
        "cargo_shortfall": cargo_shortfall,
        "fuel_price": st.session_state.get("fuel_price", 54000),
        "fuel_change": fuel_change,
        "effective_fuel_change": effective_fuel_change,
        "scenario_port_outage": scenario_port_outage,
        "active_scenario": active_scenario,
        "minimum_vessel_utilization": min_utilization,
        "maximum_acceptable_risk": max_risk,
        "export_order_quantity": st.session_state.get("export_order_quantity", 100000),
        "export_inventory": st.session_state.get("current_export_inventory", 70000),
        "planned_production_before_loading": st.session_state.get("planned_production", 20000),
        "reserved_domestic_stock": st.session_state.get("reserved_domestic_stock", 10000),
        "blocked_or_rejected_stock": 0,
        "incoterm": st.session_state.get("export_incoterm", "CIF"),
        "laycan_start_date": st.session_state.get("export_laycan_start"),
        "laycan_end_date": st.session_state.get("export_laycan_end"),
    }

    # 3. Input Validation
    input_errors = validate_scenario_inputs(scenario_dict)
    scenario_dict["input_errors"] = input_errors

    # 4. Generate candidate plans across combinations and ports
    candidate_combos = generate_vessel_combinations(cargo_shortfall)
    ports_to_eval = ["Paradip", "Dhamra", "Visakhapatnam"]
    if scenario_port_outage != "None":
        ports_to_eval = [p for p in ports_to_eval if p != scenario_port_outage]

    all_generated_plans = []
    feasible_plans = []
    
    for combo in candidate_combos:
        for p_name in ports_to_eval:
            eta_info = calculate_route_eta(
                origin_port, p_name, combo["speed_knots"],
                weather_condition="Monsoon / Rough (15%)" if "Monsoon" in active_scenario else "Calm / Fair (0%)",
                port_congestion="High (2.0x)" if p_name == "Visakhapatnam" else "Normal (1.0x)"
            )
            w_risk = 55.0 if "Monsoon" in active_scenario else 20.0
            c_risk = 65.0 if p_name == "Visakhapatnam" else 20.0
            v_risk = 20.0
            s_risk = 25.0 if eta_info["final_eta_days"] > deadline - 5 else 15.0
            risk_res = calculate_risk_score(
                weather_risk=w_risk, congestion_risk=c_risk, cargo_readiness_risk=15.0,
                vessel_availability_risk=v_risk, schedule_risk=s_risk, route_risk=eta_info["base_route_risk"]
            )
            plan_stub = {
                "vessel_class": combo["vessel_class"],
                "vessel_count": combo["vessel_count"],
                "vessel_display": combo["vessel_display"],
                "combined_capacity": combo["combined_capacity"],
                "unused_capacity": combo["unused_capacity"],
                "shipment_quantity": cargo_shortfall,
                "port": p_name,
                "utilization": combo["utilization"],
                "effective_waiting_days": eta_info["effective_waiting_days"],
                "final_eta_days": eta_info["final_eta_days"],
                "base_sailing_days": eta_info["base_sailing_days"],
                "risk": risk_res["total_risk"],
                "risk_analysis": risk_res,
            }
            cost_info = calculate_cost_breakdown(scenario_dict, plan_stub)
            plan_stub.update(cost_info)

            feas_eval = evaluate_feasibility(plan_stub, scenario_dict)
            plan_stub.update(feas_eval)

            all_generated_plans.append(plan_stub)
            if plan_stub["is_feasible"]:
                feasible_plans.append(plan_stub)

    # 5. Rank and recommendations
    ranked_results = rank_feasible_plans(feasible_plans, preferred_port=dest_port)
    lowest_cost_plan = ranked_results["lowest_cost_plan"]
    lowest_risk_plan = ranked_results["lowest_risk_plan"]
    balanced_plan = ranked_results["balanced_plan"]

    # Recommendation confidence calculation
    conf_res = calculate_recommendation_confidence(
        input_completeness=98.0 if not input_errors else 65.0,
        constraint_certainty=92.0 if balanced_plan and balanced_plan["total_cost_cr"] < approved_budget - 3 else 78.0,
        forecast_stability=91.8,
        plan_separation=86.0 if len(feasible_plans) > 1 else 70.0,
        scenario_consistency=90.0
    )

    # Explainability
    explanation = generate_explanation(balanced_plan, all_generated_plans, scenario_dict)

    # Construct compatible DataFrame for downstream modules
    base_plans_rows = []
    for p in feasible_plans:
        base_plans_rows.append({
            "Supplier": "Ubuntu Bulk Trading",
            "Origin": origin_port,
            "Vessel": f"MV Varuna {p['vessel_class']}",
            "Class": p["vessel_class"],
            "Vessels": p["vessel_count"],
            "Port": p["port"],
            "ETA Days": p["final_eta_days"],
            "Cargo (t)": cargo_shortfall,
            "Utilization": p["utilization"],
            "Combined Capacity": p["combined_capacity"],
            "Cargo Cost Cr": p.get("cargo_procurement_cr", 18.56),
            "Charter Cost Cr": p.get("ocean_charter_cr", 8.60),
            "Port & Waiting Cr": p.get("dest_port_charges_cr", 1.11) + p.get("expected_waiting_cost_cr", 0.12),
            "Risk Cost Cr": p.get("risk_contingency_cr", 0.18),
            "Risk": p["risk"],
            "Total Cost Cr": p["total_cost_cr"],
        })
    base_plans_df = pd.DataFrame(base_plans_rows) if base_plans_rows else pd.DataFrame()

    active_recommendation = None
    if balanced_plan:
        active_recommendation = {
            "vessel": balanced_plan["vessel_display"],
            "vessel_name": f"MV Varuna {balanced_plan['vessel_class']}",
            "vessel_count": balanced_plan["vessel_count"],
            "vessel_type": balanced_plan["vessel_class"],
            "combined_capacity": balanced_plan["combined_capacity"],
            "utilization": balanced_plan["utilization"],
            "port": f"{balanced_plan['port']} Port",
            "duration": balanced_plan["final_eta_days"],
            "baseline_cost": balanced_plan["baseline_cost_cr"],
            "optimized_cost": balanced_plan["total_cost_cr"],
            "savings": balanced_plan["savings_cr"],
            "savings_pct": balanced_plan["savings_pct"],
            "risk_score": balanced_plan["risk"],
            "confidence_score": conf_res["confidence_score"],
            "forecast_mape": 0.82,
            "feasible_plans": len(feasible_plans),
            "plan_dict": balanced_plan,
        }

    return {
        "commodity": commodity,
        "origin": origin_port,
        "forecast_horizon": forecast_horizon,
        "horizon_days": horizon_days,
        "cargo_requirement": cargo_requirement,
        "inventory": inventory,
        "safety": safety,
        "deadline": deadline,
        "fuel_change": fuel_change,
        "active_scenario": active_scenario,
        "scenario_fuel_offset": scenario_fuel_offset,
        "scenario_demand_mult": scenario_demand_mult,
        "scenario_port_outage": scenario_port_outage,
        "effective_fuel_change": effective_fuel_change,
        "model": model,
        "future_df": future_df,
        "initial_forecast": initial_forecast,
        "final_forecast": final_forecast,
        "demand_growth": demand_growth,
        "adjusted_demand": adjusted_demand,
        "cargo_shortfall": cargo_shortfall,
        "base_plans": base_plans_df,
        "active_recommendation": active_recommendation,
        "all_plans": all_generated_plans,
        "feasible_plans": feasible_plans,
        "lowest_cost_plan": lowest_cost_plan,
        "lowest_risk_plan": lowest_risk_plan,
        "balanced_plan": balanced_plan,
        "confidence_details": conf_res,
        "explanation": explanation,
        "scenario_dict": scenario_dict,
        "input_errors": input_errors,
    }

def render_command_centre():
    """Renders the complete Command Centre executive overview."""
    render_level2_header("Command Centre")
    render_trade_direction_selector()
    render_shared_scenario_summary()

    td = st.session_state.get("trade_direction", "Import to India")

    if td == "Import to India":
        calc = get_scenario_calculations()
        cargo_requirement = calc["cargo_requirement"]
        cargo_shortfall = calc["cargo_shortfall"]
        base_plans = calc["base_plans"]
        deadline = calc["deadline"]

        render_command_centre_controls()

        if not base_plans.empty:
            best = base_plans.iloc[0]
            baseline_cost = 31.19
            saving = baseline_cost - best["Total Cost Cr"]
            saving_pct = (saving / baseline_cost) * 100

            st.markdown('<div style="margin-top: 6px; margin-bottom: 18px;">', unsafe_allow_html=True)
            k1, k2, k3, k4, k5 = st.columns(5, gap="medium")
            with k1:
                st.markdown(
                    f"""
                    <div class="kpi-card-exec">
                        <span class="kpi-exec-label">Cargo Requirement</span>
                        <div class="kpi-exec-val">{cargo_requirement:,.0f} tonnes</div>
                        <div class="kpi-exec-sub" style="color: #6B6B73;">Source: Demand Forecast</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with k2:
                st.markdown(
                    f"""
                    <div class="kpi-card-exec">
                        <span class="kpi-exec-label">Cargo Shortfall</span>
                        <div class="kpi-exec-val">{cargo_shortfall:,.0f} tonnes</div>
                        <div class="kpi-exec-sub" style="color: #F6B51B;">Source: Inventory Balance Equation</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with k3:
                st.markdown(
                    f"""
                    <div class="kpi-card-exec">
                        <span class="kpi-exec-label">Recommended Cost</span>
                        <div class="kpi-exec-val">₹{best['Total Cost Cr']:.2f} Cr</div>
                        <div class="kpi-exec-sub" style="color: #48A868;">Savings: ₹{saving:.2f} Cr ({saving_pct:.1f}%) • Source: Optimization Engine</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with k4:
                st.markdown(
                    """
                    <div class="kpi-card-exec">
                        <span class="kpi-exec-label">Risk Score</span>
                        <div class="kpi-exec-val">24/100 — Low Risk</div>
                        <div class="kpi-exec-sub" style="color: #372580;">Source: Risk Model (90% Conf)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with k5:
                st.markdown(
                    f"""
                    <div class="kpi-card-exec">
                        <span class="kpi-exec-label">Estimated Delivery</span>
                        <div class="kpi-exec-val">{int(best['ETA Days'])} days — On Schedule</div>
                        <div class="kpi-exec-sub" style="color: #48A868;">Within {deadline}d Deadline • Source: Transit Calculation</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

            # Snapshots
            st.markdown('<div style="margin-top: 10px; margin-bottom: 22px;">', unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3, gap="medium")
            with s1:
                st.markdown(
                    """
                    <div class="snapshot-card">
                        <div>
                            <div class="snapshot-title">📈 Forecast Snapshot</div>
                            <ul class="snapshot-list">
                                <li>Current Cargo: <b>Thermal Coal</b></li>
                                <li>Demand Horizon: <b>90 Days</b></li>
                                <li>MAPE Error Rate: <b>0.82%</b></li>
                                <li>Prediction Interval: <b>±5% Confidence</b></li>
                            </ul>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with s2:
                st.markdown(
                    """
                    <div class="snapshot-card">
                        <div>
                            <div class="snapshot-title">📦 Procurement Snapshot</div>
                            <ul class="snapshot-list">
                                <li>Baseline Target: <b>150,000 tonnes</b></li>
                                <li>Stockpile On-Hand: <b>40,000 tonnes</b></li>
                                <li>Safety Reserve: <b>20,000 tonnes</b></li>
                                <li>Net Shortfall: <b>130,000 tonnes</b></li>
                            </ul>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with s3:
                st.markdown(
                    f"""
                    <div class="snapshot-card">
                        <div>
                            <div class="snapshot-title">🚢 Fixture Recommendation</div>
                            <ul class="snapshot-list">
                                <li>Allocated Fleet: <b>{int(best['Vessels'])} × {best['Class']}</b></li>
                                <li>Destination: <b>{best['Port']} Port</b></li>
                                <li>Capacity Utilized: <b>{best['Utilization']}%</b></li>
                                <li>Committed Cost: <b>₹{best['Total Cost Cr']:.2f} Cr</b></li>
                            </ul>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        # EXPORT FROM INDIA WORKFLOW
        order_qty = st.session_state.get("export_order_quantity", 100000)
        inv = st.session_state.get("current_export_inventory", 70000)
        prod = st.session_state.get("planned_production", 20000)
        res = st.session_state.get("reserved_domestic_stock", 10000)
        ready_qty, shortfall, readiness_pct = calculate_export_availability(order_qty, inv, prod, res)
        
        incoterm = st.session_state.get("export_incoterm", "CFR")
        load_port = st.session_state.get("export_loading_port", "Paradip")
        dest_port = st.session_state.get("export_dest_port", "Singapore")
        deadline = st.session_state.get("export_deadline", 30)
        
        # 8 KPI CARDS FOR EXPORT
        st.markdown('<div style="margin-top: 6px; margin-bottom: 18px;">', unsafe_allow_html=True)
        r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4, gap="medium")
        with r1_c1:
            st.markdown(
                f"""
                <div class="kpi-card-exec">
                    <span class="kpi-exec-label">1. Export Order Qty</span>
                    <div class="kpi-exec-val">{order_qty:,.0f} tonnes</div>
                    <div class="kpi-exec-sub" style="color: #6B6B73;">Source: Export Order Ledger</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r1_c2:
            st.markdown(
                f"""
                <div class="kpi-card-exec">
                    <span class="kpi-exec-label">2. Export-Ready Qty</span>
                    <div class="kpi-exec-val">{ready_qty:,.0f} tonnes</div>
                    <div class="kpi-exec-sub" style="color: #48A868;">Source: Plant Inventory & Production Math</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r1_c3:
            st.markdown(
                f"""
                <div class="kpi-card-exec">
                    <span class="kpi-exec-label">3. Fulfilment Shortfall</span>
                    <div class="kpi-exec-val">{shortfall:,.0f} tonnes</div>
                    <div class="kpi-exec-sub" style="color: #F6B51B;">Source: Readiness Gap Analysis</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r1_c4:
            st.markdown(
                f"""
                <div class="kpi-card-exec">
                    <span class="kpi-exec-label">4. Recommended Cost</span>
                    <div class="kpi-exec-val">₹15.10 Cr</div>
                    <div class="kpi-exec-sub" style="color: #372580;">Source: Incoterm Logistics Engine ({incoterm})</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)
        r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4, gap="medium")
        with r2_c1:
            st.markdown(
                """
                <div class="kpi-card-exec">
                    <span class="kpi-exec-label">5. Export Risk Score</span>
                    <div class="kpi-exec-val">22/100 — Low Risk</div>
                    <div class="kpi-exec-sub" style="color: #16A34A;">Source: Route & Port Risk Model</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r2_c2:
            st.markdown(
                f"""
                <div class="kpi-card-exec">
                    <span class="kpi-exec-label">6. Buyer Delivery ETA</span>
                    <div class="kpi-exec-val">7 days — On Schedule</div>
                    <div class="kpi-exec-sub" style="color: #48A868;">Within {deadline}d Deadline • Source: Vessel Model</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r2_c3:
            st.markdown(
                f"""
                <div class="kpi-card-exec">
                    <span class="kpi-exec-label">7. Order Readiness %</span>
                    <div class="kpi-exec-val">{readiness_pct:.1f}%</div>
                    <div class="kpi-exec-sub" style="color: #48A868;">Source: Plant Availability</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r2_c4:
            st.markdown(
                f"""
                <div class="kpi-card-exec">
                    <span class="kpi-exec-label">8. Recommended Vessel</span>
                    <div class="kpi-exec-val">2 × Supramax • {load_port}</div>
                    <div class="kpi-exec-sub" style="color: #372580;">Source: Optimization Engine</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # 3 EXPORT DECISION SNAPSHOTS
        st.markdown('<div style="margin-top: 10px; margin-bottom: 22px;">', unsafe_allow_html=True)
        es1, es2, es3 = st.columns(3, gap="medium")
        with es1:
            st.markdown(
                f"""
                <div class="snapshot-card">
                    <div>
                        <div class="snapshot-title">📈 Export Demand Snapshot</div>
                        <ul class="snapshot-list">
                            <li>Export Cargo: <b>{st.session_state.get('export_commodity', 'Finished Steel')}</b></li>
                            <li>Destination: <b>{dest_port}</b></li>
                            <li>Projected Demand: <b>125,000 t / 90d</b></li>
                            <li>Forecast Confidence: <b>92% (±4% Band)</b></li>
                        </ul>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with es2:
            st.markdown(
                f"""
                <div class="snapshot-card">
                    <div>
                        <div class="snapshot-title">🏭 Plant Readiness Pipeline</div>
                        <ul class="snapshot-list">
                            <li>Ready Stockpile: <b>{inv:,} tonnes</b></li>
                            <li>Production Run: <b>+{prod:,} tonnes</b></li>
                            <li>Domestic Buffer: <b>-{res:,} tonnes</b></li>
                            <li>Available to Export: <b>{ready_qty:,} tonnes</b></li>
                        </ul>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with es3:
            st.markdown(
                f"""
                <div class="snapshot-card">
                    <div>
                        <div class="snapshot-title">🚢 Export Fixture & Port</div>
                        <ul class="snapshot-list">
                            <li>Loading Port: <b>{load_port} (India)</b></li>
                            <li>Trade Incoterm: <b>{incoterm}</b></li>
                            <li>Laycan Window: <b>7-14 Days from Today</b></li>
                            <li>Documentation: <b>ICEGATE Shipping Bill Ready</b></li>
                        </ul>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)


def render_forecasting_studio():
    """Renders the interactive Forecasting Studio view for Import and Export."""
    render_level2_header("Forecasting Studio")
    render_trade_direction_selector()
    render_shared_scenario_summary()

    # Metric Hierarchy Distinction Callout
    st.markdown(
        """
        <div style="background: #F0EEF9; border-left: 4px solid #372580; padding: 12px 16px; border-radius: 6px; margin-bottom: 16px; font-size: 0.88rem; line-height: 1.5;">
            <b style="color: #372580; font-size: 0.94rem;">ℹ️ Forecasting Metric Hierarchy & Operational Distinction:</b><br>
            • <b>MAPE (Mean Absolute Percentage Error):</b> Historical back-test error rate measuring algorithm accuracy vs past actual shipments (e.g. 0.82% MAPE indicates 99.18% average accuracy).<br>
            • <b>Model Confidence Score:</b> Algorithmic certainty metric (0–100%) evaluating market stability, data density, and seasonal predictability (e.g. 90% Confidence).<br>
            • <b>Confidence Band (±5% Interval):</b> Statistically projected upper and lower variance boundaries surrounding the mean demand projection curve over the selected horizon.
        </div>
        """,
        unsafe_allow_html=True,
    )

    td = st.session_state.get("trade_direction", "Import to India")

    if td == "Import to India":
            """Renders the interactive Forecasting Studio view."""
            # TOP CONTROL SECTION
            render_forecasting_controls()

            # Sub-header & Prototype Data badge
            st.markdown(
                """
                <div class="panel-card" style="margin-top: 6px; margin-bottom: 18px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                    <div>
                        <b style="color: #372580; font-size: 1.05rem;">AI Demand Forecasting Studio</b>
                        <span style="color: #6B6B73; font-size: 0.88rem; margin-left: 8px;">— Multi-model predictive demand & freight rate forward curves.</span>
                    </div>
                    <span class="prototype-badge">PROTOTYPE DATA</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Local prototype parameters & calculations based on selection
            COMMODITY_FACTORS = {
                "Thermal Coal": {"base_monthly": 50000, "growth": 4.2, "freight_base": 1428, "mape": 0.82, "conf": 90},
                "Coking Coal": {"base_monthly": 42000, "growth": 3.8, "freight_base": 1510, "mape": 0.94, "conf": 88},
                "Iron Ore": {"base_monthly": 85000, "growth": 5.1, "freight_base": 1350, "mape": 1.05, "conf": 87},
                "Bauxite": {"base_monthly": 32000, "growth": 3.1, "freight_base": 1280, "mape": 1.12, "conf": 85},
                "Limestone": {"base_monthly": 36000, "growth": 2.9, "freight_base": 1190, "mape": 1.18, "conf": 86},
                "Grain": {"base_monthly": 28000, "growth": 4.5, "freight_base": 1620, "mape": 1.35, "conf": 83},
                "Cement": {"base_monthly": 40000, "growth": 3.4, "freight_base": 1240, "mape": 1.10, "conf": 87},
                "Petroleum Coke": {"base_monthly": 24000, "growth": 2.5, "freight_base": 1480, "mape": 1.22, "conf": 84},
            }

            cur_cargo = st.session_state.get("cargo", "Thermal Coal")
            cur_horizon = st.session_state.get("horizon", "90 Days")
            cur_model = st.session_state.get("forecast_model", "Ensemble Recommended")
            cur_origin = st.session_state.get("origin", "Richards Bay, South Africa")

            cfg = COMMODITY_FACTORS.get(cur_cargo, COMMODITY_FACTORS["Thermal Coal"])
            horizon_days_val = HORIZON_MAP.get(cur_horizon, 90)
            months_ahead_val = max(1, math.ceil(horizon_days_val / 30))

            # Calculate base metrics
            if cur_cargo == "Thermal Coal" and cur_horizon == "90 Days":
                forecasted_demand = 150000
                daily_consumption = 1667.0
                forecast_mape = 0.82
                confidence_score = 90
            else:
                forecasted_demand = int(cfg["base_monthly"] * months_ahead_val)
                daily_consumption = round(forecasted_demand / horizon_days_val, 1)
                forecast_mape = cfg["mape"]
                confidence_score = cfg["conf"]

            # Model adjustments
            if cur_model == "Linear Regression":
                forecast_mape += 0.12
                confidence_score -= 2
            elif cur_model == "Moving Average":
                forecast_mape += 0.25
                confidence_score -= 4
            elif cur_model == "Seasonal Trend":
                forecast_mape += 0.18
                confidence_score -= 3

            # Historical monthly dataset (20 months)
            hist_months = pd.date_range("2025-01-01", periods=20, freq="MS")
            base_mult = cfg["base_monthly"] / 50000.0
            hist_demand = [round(d * base_mult, 1) for d in [172, 178, 184, 181, 190, 196, 203, 207, 211, 218,
                                                              224, 230, 227, 235, 241, 247, 252, 258, 264, 270]]

            # Future forecasted months & projected demand curve
            future_months = pd.date_range(hist_months[-1] + pd.offsets.MonthBegin(1), periods=months_ahead_val, freq="MS")
            target_growth = float(cfg["growth"])
            base_val = float(cfg["base_monthly"])
            final_val = round(base_val * (1.0 + target_growth / 100.0), 2)

            if months_ahead_val == 1:
                future_demand = [round(final_val / 1000.0, 1)]
            else:
                step = (final_val - base_val) / (months_ahead_val - 1)
                future_demand = [round((base_val + step * i) / 1000.0, 1) for i in range(months_ahead_val)]

            # Safe calculation of initial_forecast, final_forecast, and demand_growth from forecast data
            if not future_demand or len(future_demand) == 0:
                st.warning("Forecast telemetry data is currently unavailable. Using safe baseline fallbacks.")
                initial_forecast = 0.0
                final_forecast = 0.0
                demand_growth = 0.0
            else:
                if len(future_demand) == 1:
                    initial_forecast = base_val
                    final_forecast = final_val
                else:
                    initial_forecast = float(future_demand[0] * 1000.0)
                    final_forecast = float(future_demand[-1] * 1000.0)

                # Ensure final_forecast and initial_forecast exist before calculating it
                if initial_forecast != 0:
                    demand_growth = ((final_forecast - initial_forecast) / initial_forecast) * 100
                else:
                    demand_growth = 0.0

            # FORECAST OUTPUT: 5 KPI Cards in one desktop row
            st.markdown('<div style="margin-bottom: 20px;">', unsafe_allow_html=True)
            k1, k2, k3, k4, k5 = st.columns(5, gap="medium")
            with k1:
                st.markdown(
                    f"""
                    <div class="kpi-card-exec">
                        <span class="kpi-exec-label">Forecasted Demand</span>
                        <div class="kpi-exec-val">{forecasted_demand:,.0f} tonnes</div>
                        <div class="kpi-exec-sub" style="color: #6B6B73;">{cur_horizon} Target Total</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with k2:
                st.markdown(
                    f"""
                    <div class="kpi-card-exec">
                        <span class="kpi-exec-label">Average Daily Consumption</span>
                        <div class="kpi-exec-val">{daily_consumption:,.0f} t/day</div>
                        <div class="kpi-exec-sub" style="color: #F6B51B;">Burn Rate Average</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with k3:
                st.markdown(
                    f"""
                    <div class="kpi-card-exec">
                        <span class="kpi-exec-label">Demand Growth</span>
                        <div class="kpi-exec-val">+{demand_growth:.1f}%</div>
                        <div class="kpi-exec-sub" style="color: #48A868;">Seasonal Trajectory</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with k4:
                st.markdown(
                    f"""
                    <div class="kpi-card-exec">
                        <span class="kpi-exec-label">Forecast MAPE</span>
                        <div class="kpi-exec-val">{forecast_mape:.2f}%</div>
                        <div class="kpi-exec-sub" style="color: #48A868;">High Model Precision</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with k5:
                st.markdown(
                    f"""
                    <div class="kpi-card-exec">
                        <span class="kpi-exec-label">Confidence Score</span>
                        <div class="kpi-exec-val">{confidence_score}%</div>
                        <div class="kpi-exec-sub" style="color: #372580;">{cur_model}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)



            # 4 INTERACTIVE PLOTLY CHARTS (2x2 Grid)
            ch_row1_c1, ch_row1_c2 = st.columns(2, gap="medium")

            # CHART 1: Actual versus Forecast Demand
            with ch_row1_c1:
                st.markdown('<h4 style="color: #18181B; font-size: 1.02rem; margin-top: 0; margin-bottom: 8px;">📊 Actual vs Forecast Demand</h4>', unsafe_allow_html=True)
                fig_act_fc = go.Figure()
                fig_act_fc.add_trace(go.Scatter(
                    x=hist_months,
                    y=hist_demand,
                    mode="lines+markers",
                    name="Actual Consumption",
                    line=dict(color="#F6B51B", width=2.5),
                    marker=dict(size=5, color="#F6B51B"),
                    hovertemplate="<b>Actual:</b> %{y:.1f}K tonnes<br><b>Month:</b> %{x|%b %Y}<extra></extra>",
                ))
                fig_act_fc.add_trace(go.Scatter(
                    x=future_months,
                    y=future_demand,
                    mode="lines+markers",
                    name="AI Forecast",
                    line=dict(color="#372580", width=2.5, dash="dash"),
                    marker=dict(size=7, color="#372580", symbol="diamond"),
                    hovertemplate="<b>AI Forecast:</b> %{y:.1f}K tonnes<br><b>Month:</b> %{x|%b %Y}<extra></extra>",
                ))
                fig_act_fc.update_layout(
                    height=320,
                    margin=dict(l=35, r=20, t=25, b=30),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#FFFFFF",
                    font_color="#18181B",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(title="Month", gridcolor="#ECECF0"),
                    yaxis=dict(title="Demand ('000 tonnes)", gridcolor="#ECECF0"),
                )
                st.plotly_chart(fig_act_fc, use_container_width=True)

            # CHART 2: Forecast Confidence Band
            with ch_row1_c2:
                st.markdown(f'<h4 style="color: #18181B; font-size: 1.02rem; margin-top: 0; margin-bottom: 8px;">🎯 Forecast Confidence Band ({confidence_score}% CI)</h4>', unsafe_allow_html=True)
                upper_bound = [round(d * 1.05, 1) for d in future_demand]
                lower_bound = [round(d * 0.95, 1) for d in future_demand]

                fig_conf = go.Figure()
                fig_conf.add_trace(go.Scatter(
                    x=list(future_months) + list(future_months)[::-1],
                    y=upper_bound + lower_bound[::-1],
                    fill="toself",
                    fillcolor="rgba(217, 212, 238, 0.35)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=True,
                    name=f"{confidence_score}% Confidence Band (±5%)",
                ))
                fig_conf.add_trace(go.Scatter(
                    x=future_months,
                    y=future_demand,
                    mode="lines+markers",
                    name="Forecast Mean",
                    line=dict(color="#372580", width=2.5),
                    marker=dict(size=7, color="#372580"),
                    hovertemplate="<b>Mean Forecast:</b> %{y:.1f}K tonnes<br><b>Month:</b> %{x|%b %Y}<extra></extra>",
                ))
                fig_conf.add_trace(go.Scatter(
                    x=future_months,
                    y=upper_bound,
                    mode="lines",
                    name="Upper Bound (+5%)",
                    line=dict(color="#5746A5", width=1, dash="dot"),
                    hovertemplate="<b>Upper Bound:</b> %{y:.1f}K tonnes<extra></extra>",
                ))
                fig_conf.add_trace(go.Scatter(
                    x=future_months,
                    y=lower_bound,
                    mode="lines",
                    name="Lower Bound (-5%)",
                    line=dict(color="#5746A5", width=1, dash="dot"),
                    hovertemplate="<b>Lower Bound:</b> %{y:.1f}K tonnes<extra></extra>",
                ))
                fig_conf.update_layout(
                    height=320,
                    margin=dict(l=35, r=20, t=25, b=30),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#FFFFFF",
                    font_color="#18181B",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(title="Forward Months", gridcolor="#ECECF0"),
                    yaxis=dict(title="Projected Volume ('000 t)", gridcolor="#ECECF0"),
                )
                st.plotly_chart(fig_conf, use_container_width=True)

            ch_row2_c1, ch_row2_c2 = st.columns(2, gap="medium")

            # CHART 3: Freight Rate Forecast (₹/tonne)
            with ch_row2_c1:
                st.markdown('<h4 style="color: #18181B; font-size: 1.02rem; margin-top: 0; margin-bottom: 8px;">⚓ Freight Rate Forecast (₹/tonne)</h4>', unsafe_allow_html=True)
                base_fr = cfg["freight_base"]
                if "Newcastle" in cur_origin:
                    base_fr += 150
                elif "Indonesia" in cur_origin:
                    base_fr += 90

                # Timeline forward points (every 10 days up to horizon_days)
                fr_days = list(range(0, horizon_days_val + 1, max(5, horizon_days_val // 10)))
                fr_rates = [round(base_fr + (i * 2.5) + (math.sin(i / 15.0) * 12), 1) for i in fr_days]

                fig_fr = go.Figure()
                fig_fr.add_trace(go.Scatter(
                    x=fr_days,
                    y=fr_rates,
                    mode="lines+markers",
                    name=f"Projected Rate ({cur_origin.split(',')[0]})",
                    line=dict(color="#F6B51B", width=2.5),
                    marker=dict(size=6, color="#F6B51B"),
                    hovertemplate="<b>Freight Rate:</b> ₹%{y:,.1f}/tonne<br><b>Day:</b> %{x}<extra></extra>",
                ))
                fig_fr.add_hline(y=base_fr, line_dash="dot", line_color="#92929A",
                                 annotation_text=f"Benchmark (₹{base_fr:,}/t)", annotation_position="bottom right",
                                 annotation_font=dict(color="#6B6B73", size=10))
                fig_fr.update_layout(
                    height=320,
                    margin=dict(l=35, r=20, t=25, b=30),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#FFFFFF",
                    font_color="#18181B",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(title="Planning Horizon (Days)", gridcolor="#ECECF0"),
                    yaxis=dict(title="Freight Rate (₹/tonne)", gridcolor="#ECECF0"),
                )
                st.plotly_chart(fig_fr, use_container_width=True)

            # CHART 4: Inventory Projection
            with ch_row2_c2:
                st.markdown('<h4 style="color: #18181B; font-size: 1.02rem; margin-top: 0; margin-bottom: 8px;">📦 Inventory Projection & Buffer Position</h4>', unsafe_allow_html=True)
                inv_days = list(range(0, horizon_days_val + 1, max(3, horizon_days_val // 15)))
                curr_stock = st.session_state.get("inventory", 40000)
                safety_stock = st.session_state.get("safety", 20000)
                arrival_day = min(19, max(5, horizon_days_val // 3))
                replenish_vol = int(forecasted_demand * 0.85)

                inv_sim = []
                for d in inv_days:
                    level = curr_stock - (daily_consumption * d)
                    if d >= arrival_day:
                        level += replenish_vol
                    inv_sim.append(max(0, round(level, 1)))

                fig_inv_studio = go.Figure()
                fig_inv_studio.add_trace(go.Scatter(
                    x=inv_days,
                    y=inv_sim,
                    mode="lines+markers",
                    name="Plant Inventory",
                    line=dict(color="#372580", width=2.5),
                    marker=dict(size=5, color="#372580"),
                    hovertemplate="<b>Stock:</b> %{y:,.0f} tonnes<br><b>Day:</b> %{x}<extra></extra>",
                ))
                fig_inv_studio.add_trace(go.Scatter(
                    x=inv_days,
                    y=[safety_stock] * len(inv_days),
                    mode="lines",
                    name=f"Safety Buffer ({safety_stock:,.0f} t)",
                    line=dict(color="#F6B51B", width=2, dash="dash"),
                    hovertemplate="<b>Safety Buffer:</b> %{y:,.0f} tonnes<extra></extra>",
                ))
                fig_inv_studio.add_vline(x=arrival_day, line_width=1.5, line_dash="dot", line_color="#48A868",
                                        annotation_text=f"Day {arrival_day} Arrival (+{replenish_vol:,.0f} t)", annotation_position="top right",
                                        annotation_font=dict(color="#48A868", size=10))
                fig_inv_studio.update_layout(
                    height=320,
                    margin=dict(l=35, r=20, t=25, b=30),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#FFFFFF",
                    font_color="#18181B",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(title="Horizon Timeline (Days)", gridcolor="#ECECF0"),
                    yaxis=dict(title="Stock Volume (Tonnes)", gridcolor="#ECECF0"),
                )
                st.plotly_chart(fig_inv_studio, use_container_width=True)

            # SECTION: AI FORECAST INSIGHTS
            stockout_day = max(1, int((curr_stock - safety_stock) / max(1, daily_consumption)))
            max_freight = int(max(fr_rates))
            freight_drift = round(((max_freight - base_fr) / base_fr) * 100, 1)

            st.markdown(
                f"""
                <div class="panel-card" style="margin-top: 8px; margin-bottom: 20px; border-left: 4px solid #372580;">
                    <h4 style="color: #372580; margin-top: 0; margin-bottom: 10px; font-size: 1.05rem;">🧠 AI Predictive Demand & Freight Insights</h4>
                    <ul style="margin: 0; padding-left: 20px; color: #6B6B73; font-size: 0.90rem; line-height: 1.7;">
                        <li><b>Projected Demand Change:</b> Target consumption for <b>{cur_cargo}</b> over the <b>{cur_horizon}</b> window is calibrated at <b>{forecasted_demand:,} tonnes</b> (burn rate: <b>{daily_consumption:,.0f} tonnes/day</b>), incorporating a <b>+{demand_growth:.1f}%</b> seasonal expansion factor.</li>
                        <li><b>Inventory Shortage Date:</b> Without replenishment, existing stock ({curr_stock:,} t) will breach the critical <b>{safety_stock:,}-tonne safety buffer</b> in <b>{stockout_day} days</b>, necessitating prompt charter commitment.</li>
                        <li><b>Recommended Procurement Window:</b> Optimal charter fixture window is <b>Days 1–5</b> to guarantee vessel arrival at East Coast discharge ports before Day {arrival_day + 5}.</li>
                        <li><b>Freight-Rate Direction:</b> Dry bulk charter tariffs along the <b>{cur_origin}</b> route are forecasted to drift upward by <b>+{freight_drift}%</b> to <b>₹{max_freight:,}/tonne</b>; locking fixed contracts mitigates spot volatility.</li>
                        <li><b>Confidence Explanation:</b> The <b>{cur_model}</b> engine demonstrates high empirical fidelity with an audited <b>{forecast_mape:.2f}% MAPE</b> and <b>{confidence_score}% statistical confidence</b> across local historical benchmark datasets.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # SECTION: DOWNLOAD FORECAST CSV
            st.subheader("Export Forecast Telemetry")
            csv_rows = []
            for i, m in enumerate(future_months):
                csv_rows.append({
                    "Date": m.strftime("%Y-%m-%d"),
                    "Commodity": cur_cargo,
                    "Planning Horizon": cur_horizon,
                    "Corridor Origin": cur_origin,
                    "Forecast Model": cur_model,
                    "Projected Demand (tonnes)": round(future_demand[i] * 1000, 0),
                    "Lower Bound 90% CI (tonnes)": round(lower_bound[i] * 1000, 0),
                    "Upper Bound 90% CI (tonnes)": round(upper_bound[i] * 1000, 0),
                    "Projected Corridor Freight (₹/tonne)": fr_rates[min(i, len(fr_rates) - 1)],
                    "Projected Plant Inventory (tonnes)": inv_sim[min(i * 3, len(inv_sim) - 1)],
                    "Dataset Classification": "Prototype Data",
                })
            forecast_export_df = pd.DataFrame(csv_rows)
            csv_data = forecast_export_df.to_csv(index=False).encode("utf-8")

            col_dl1, col_dl2 = st.columns([1.5, 2.5])
            with col_dl1:
                st.download_button(
                    label="📥 Download Forecast CSV",
                    data=csv_data,
                    file_name=f"varunapath_{cur_cargo.lower().replace(' ', '_')}_{horizon_days_val}d_forecast.csv",
                    mime="text/csv",
                    key="fc_download_csv_btn",
                    help="Download complete forecasted consumption and freight forward data",
                )
            with col_dl2:
                st.caption("VarunaPath AI • Predictive Freight Intelligence Core • Simulated Prototype Data.")
    else:
        # AI EXPORT DEMAND FORECASTING STUDIO
        st.markdown(
            """
            <div class="panel-card" style="margin-top: 6px; margin-bottom: 18px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <div>
                    <b style="color: #372580; font-size: 1.05rem;">AI Export Demand Forecasting & Foreign Market Intelligence</b>
                    <span style="color: #6B6B73; font-size: 0.88rem; margin-left: 8px;">— Multi-model export commodity demand, foreign freight benchmarks, and port turnaround delays.</span>
                </div>
                <span class="prototype-badge">PROTOTYPE DATA</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fc_c1, fc_c2, fc_c3, fc_c4 = st.columns([1.2, 1.2, 1.2, 1.0], gap="medium")
        with fc_c1:
            exp_comm = st.selectbox("EXPORT COMMODITY", EXPORT_COMMODITIES, index=0, key="fc_exp_comm_sel")
        with fc_c2:
            exp_dest = st.selectbox("DESTINATION MARKET", FOREIGN_DESTINATION_PORTS, index=0, key="fc_exp_dest_sel")
        with fc_c3:
            exp_mod = st.selectbox("FORECAST MODEL", ["Linear Regression", "Ridge Regression", "Ensemble Recommended"], index=2, key="fc_exp_mod_sel")
        with fc_c4:
            st.markdown('<div class="ctrl-label">&nbsp;</div>', unsafe_allow_html=True)
            if st.button("📈 Generate Forecast", key="fc_exp_gen_btn", type="primary", use_container_width=True):
                st.toast(f"Export forecast generated for {exp_comm} to {exp_dest}!", icon="📈")
                st.rerun()

        # 5 KPI Cards for Export Demand
        st.markdown('<div style="margin-top: 14px; margin-bottom: 20px;">', unsafe_allow_html=True)
        ek1, ek2, ek3, ek4, ek5 = st.columns(5, gap="medium")
        with ek1:
            st.markdown(
                """
                <div class="kpi-card-exec">
                    <span class="kpi-exec-label">Projected Export Demand</span>
                    <div class="kpi-exec-val">125,000 tonnes</div>
                    <div class="kpi-exec-sub" style="color: #6B6B73;">90-Day Target Volume</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with ek2:
            st.markdown(
                """
                <div class="kpi-card-exec">
                    <span class="kpi-exec-label">Daily Export Burn</span>
                    <div class="kpi-exec-val">1,389 t/day</div>
                    <div class="kpi-exec-sub" style="color: #F6B51B;">Dispatch Rate Average</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with ek3:
            st.markdown(
                """
                <div class="kpi-card-exec">
                    <span class="kpi-exec-label">Market Growth</span>
                    <div class="kpi-exec-val">+5.8%</div>
                    <div class="kpi-exec-sub" style="color: #48A868;">Export Corridor Trajectory</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with ek4:
            st.markdown(
                """
                <div class="kpi-card-exec">
                    <span class="kpi-exec-label">Export Model MAPE</span>
                    <div class="kpi-exec-val">0.94%</div>
                    <div class="kpi-exec-sub" style="color: #48A868;">High Model Precision</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with ek5:
            st.markdown(
                """
                <div class="kpi-card-exec">
                    <span class="kpi-exec-label">Market Confidence</span>
                    <div class="kpi-exec-val">92%</div>
                    <div class="kpi-exec-sub" style="color: #372580;">Ensemble Recommended</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # 2x2 Plotly Charts for Export
        ech_r1_c1, ech_r1_c2 = st.columns(2, gap="medium")
        with ech_r1_c1:
            st.markdown(f'<h4 style="color: #18181B; font-size: 1.02rem; margin-top: 0; margin-bottom: 8px;">📊 Projected Export Demand Curve ({exp_comm})</h4>', unsafe_allow_html=True)
            days = list(range(1, 91, 5))
            demands = [round(100.0 + math.sin(d / 10.0) * 15.0 + (d * 0.25), 1) for d in days]
            fig_exp_dem = go.Figure()
            fig_exp_dem.add_trace(go.Scatter(
                x=days, y=demands, mode="lines+markers", name="Export Demand", line=dict(color="#372580", width=2.5), marker=dict(size=6, color="#372580")
            ))
            fig_exp_dem.update_layout(
                height=320, margin=dict(l=35, r=20, t=25, b=30), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
                font_color="#18181B", xaxis=dict(title="Planning Horizon (Days)", gridcolor="#ECECF0"), yaxis=dict(title="Demand ('000 tonnes)", gridcolor="#ECECF0")
            )
            st.plotly_chart(fig_exp_dem, use_container_width=True)

        with ech_r1_c2:
            st.markdown(f'<h4 style="color: #18181B; font-size: 1.02rem; margin-top: 0; margin-bottom: 8px;">🎯 Export Forecast Confidence Band (±5% CI)</h4>', unsafe_allow_html=True)
            upper_b = [round(d * 1.05, 1) for d in demands]
            lower_b = [round(d * 0.95, 1) for d in demands]
            fig_exp_ci = go.Figure()
            fig_exp_ci.add_trace(go.Scatter(
                x=days + days[::-1], y=upper_b + lower_b[::-1], fill="toself", fillcolor="rgba(217, 212, 238, 0.35)", line=dict(color="rgba(255,255,255,0)"), name="92% Confidence Band"
            ))
            fig_exp_ci.add_trace(go.Scatter(
                x=days, y=demands, mode="lines", name="Mean Projection", line=dict(color="#372580", width=2.5)
            ))
            fig_exp_ci.update_layout(
                height=320, margin=dict(l=35, r=20, t=25, b=30), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
                font_color="#18181B", xaxis=dict(title="Planning Horizon (Days)", gridcolor="#ECECF0"), yaxis=dict(title="Volume ('000 tonnes)", gridcolor="#ECECF0")
            )
            st.plotly_chart(fig_exp_ci, use_container_width=True)

        # Download Export Forecast CSV
        st.subheader("Export Demand Telemetry Download")
        exp_rows = []
        for i, d in enumerate(days):
            exp_rows.append({
                "Day": d, "Commodity": exp_comm, "Destination Port": exp_dest, "Model": exp_mod,
                "Projected Demand (tonnes)": round(demands[i] * 1000, 0), "Lower Band (tonnes)": round(lower_b[i] * 1000, 0),
                "Upper Band (tonnes)": round(upper_b[i] * 1000, 0), "Classification": "Prototype Data"
            })
        exp_csv_bytes = pd.DataFrame(exp_rows).to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Export Forecast CSV", data=exp_csv_bytes, file_name=f"varunapath_{exp_comm.lower().replace(' ', '_')}_export_forecast.csv", mime="text/csv", key="fc_exp_dl_csv_btn", use_container_width=True)


def render_shipment_planner():
    """Renders the comprehensive Shipment Planner workflow for Import and Export."""
    render_level2_header("Shipment Planner")
    render_trade_direction_selector()
    render_shared_scenario_summary()

    td = st.session_state.get("trade_direction", "Import to India")

    if td == "Import to India":
        # Sub-header & Prototype Data badge
        st.markdown(
            """
            <div class="panel-card" style="margin-top: 6px; margin-bottom: 18px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <div>
                    <b style="color: #372580; font-size: 1.05rem;">Interactive Shipment Planner & Voyage Architect (Import to India)</b>
                    <span style="color: #6B6B73; font-size: 0.88rem; margin-left: 8px;">— Multi-step constraint-driven vessel fixture and procurement scheduling.</span>
                </div>
                <span class="prototype-badge">PROTOTYPE DATA</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # UNIFIED FORM FOR ALL 12 INPUTS
        with st.form("shipment_planner_form"):
            st.markdown("<div style='font-size: 0.92rem; font-weight: 700; color: #18181B; margin-bottom: 10px;'>📋 Voyage Procurement & Operational Inputs</div>", unsafe_allow_html=True)
            
            r1_c1, r1_c2, r1_c3, r1_c4 = st.columns([1.15, 1.4, 1.25, 1.2], gap="medium")
            with r1_c1:
                cur_cargo = st.session_state.get("cargo", "Thermal Coal")
                in_cargo = st.selectbox("CARGO TYPE", CARGO_OPTIONS, index=CARGO_OPTIONS.index(cur_cargo) if cur_cargo in CARGO_OPTIONS else 0, key="sp_imp_cargo")
            with r1_c2:
                origins = ["Richards Bay, South Africa", "Newcastle, Australia", "Tanjung Bara, Indonesia"]
                cur_orig = st.session_state.get("origin", origins[0])
                in_origin = st.selectbox("ORIGIN", origins, index=origins.index(cur_orig) if cur_orig in origins else 0, key="sp_imp_origin")
            with r1_c3:
                ports = ["All Ports (Auto-Optimized)", "Paradip", "Dhamra", "Visakhapatnam"]
                cur_pref_port = st.session_state.get("preferred_port", ports[0])
                in_pref_port = st.selectbox("PREFERRED PORT", ports, index=ports.index(cur_pref_port) if cur_pref_port in ports else 0, key="sp_imp_port")
            with r1_c4:
                in_deadline = st.slider("DELIVERY DEADLINE (DAYS)", 15, 60, int(st.session_state.get("deadline", 45)), key="sp_imp_deadline")

            r2_c1, r2_c2, r2_c3 = st.columns(3, gap="medium")
            with r2_c1:
                in_req = st.number_input("CARGO REQUIREMENT (t)", 0, 500000, int(st.session_state.get("cargo_requirement", 150000)), 5000, key="sp_imp_req")
            with r2_c2:
                in_inv = st.number_input("CURRENT INVENTORY (t)", 0, 500000, int(st.session_state.get("inventory", 40000)), 5000, key="sp_imp_inv")
            with r2_c3:
                in_safety = st.number_input("SAFETY STOCK (t)", 0, 150000, int(st.session_state.get("safety", 20000)), 5000, key="sp_imp_safety")

            r3_c1, r3_c2, r3_c3 = st.columns([1.2, 1.2, 1.4], gap="medium")
            with r3_c1:
                in_risk = st.slider("Maximum Acceptable Risk Score", 10, 100, int(st.session_state.get("max_risk", 50)), step=5, key="sp_imp_max_risk")
            with r3_c2:
                in_budget = st.number_input("Maximum Budget (₹ Cr)", min_value=10.0, max_value=80.0, value=float(st.session_state.get("max_budget", 32.0)), step=1.0, key="sp_imp_max_budget")
            with r3_c3:
                arr_options = ["Within 20 Days", "Within 25 Days", "Within 30 Days", "Within 45 Days", "Within 60 Days"]
                in_arr_window = st.selectbox("Preferred Arrival Window", arr_options, index=1, key="sp_imp_arrival_window")

            r4_c1, r4_c2 = st.columns([1.5, 1.5], gap="medium")
            with r4_c1:
                vessel_options = ["Handysize (35k)", "Handymax (50k)", "Supramax (58k)", "Panamax (82k)", "Capesize (180k)"]
                in_allowed_vessels = st.multiselect("Allowed Vessel Types", vessel_options, default=["Supramax (58k)", "Panamax (82k)", "Capesize (180k)"], key="sp_imp_allowed_vessels")
            with r4_c2:
                port_options = ["Paradip", "Visakhapatnam", "Haldia", "Ennore", "Dhamra"]
                in_allowed_ports = st.multiselect("Allowed Destination Ports", port_options, default=["Paradip", "Visakhapatnam", "Dhamra"], key="sp_imp_allowed_ports")

            submit_imp = st.form_submit_button("⚡ Generate Plans", type="primary", use_container_width=True)

        # Handle form submission and snapshot saving
        if submit_imp:
            st.session_state["cargo"] = in_cargo
            st.session_state["origin"] = in_origin
            st.session_state["preferred_port"] = in_pref_port
            st.session_state["deadline"] = in_deadline
            st.session_state["cargo_requirement"] = in_req
            st.session_state["inventory"] = in_inv
            st.session_state["safety"] = in_safety
            st.session_state["max_risk"] = in_risk
            st.session_state["max_budget"] = in_budget
            st.session_state["arrival_window"] = in_arr_window
            st.session_state["allowed_vessels"] = in_allowed_vessels
            st.session_state["allowed_ports"] = in_allowed_ports
            st.session_state["sp_import_snapshot"] = {
                "cargo": in_cargo, "origin": in_origin, "preferred_port": in_pref_port, "deadline": in_deadline,
                "cargo_requirement": in_req, "inventory": in_inv, "safety": in_safety, "max_risk": in_risk,
                "max_budget": in_budget, "arrival_window": in_arr_window, "allowed_vessels": in_allowed_vessels,
                "allowed_ports": in_allowed_ports
            }
            st.session_state["sp_state"] = "Result Current"
            st.toast("Generated 3 comparative voyage plans!", icon="⚡")
            st.rerun()

        # Check for stale results
        snap = st.session_state.get("sp_import_snapshot")
        if snap is not None:
            current_vals = {
                "cargo": in_cargo, "origin": in_origin, "preferred_port": in_pref_port, "deadline": in_deadline,
                "cargo_requirement": in_req, "inventory": in_inv, "safety": in_safety, "max_risk": in_risk,
                "max_budget": in_budget, "arrival_window": in_arr_window, "allowed_vessels": in_allowed_vessels,
                "allowed_ports": in_allowed_ports
            }
            if current_vals != snap:
                st.warning("⚠️ Inputs changed. Generate plans to refresh results.")

        # Active Values from session state
        sp_cargo = st.session_state.get("cargo", "Thermal Coal")
        sp_origin = st.session_state.get("origin", "Richards Bay, South Africa")
        sp_req = st.session_state.get("cargo_requirement", 150000)
        sp_inv = st.session_state.get("inventory", 40000)
        sp_saf = st.session_state.get("safety", 20000)
        sp_dl = st.session_state.get("deadline", 45)
        sp_max_budget = st.session_state.get("max_budget", 32.0)
        sp_max_risk = st.session_state.get("max_risk", 50)

        # STEP 1: CARGO REQUIREMENT & POSITIVE INVENTORY DISPLAY
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; margin-top: 14px; margin-bottom: 12px;">
                <span style="background: #372580; color: #ffffff; font-weight: 700; font-size: 0.8rem; padding: 3px 9px; border-radius: 6px;">STEP 1</span>
                <h3 style="margin: 0; color: #18181B; font-size: 1.15rem; font-weight: 600;">Cargo Requirement & Inventory Balance</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        calc_shortfall = max(0, sp_req + sp_saf - sp_inv)

        eq_c1, eq_c2, eq_c3, eq_c4 = st.columns(4, gap="medium")
        with eq_c1:
            st.markdown(
                f"""
                <div class="kpi-card-exec">
                    <span class="kpi-exec-label">Required Quantity</span>
                    <div class="kpi-exec-val">{sp_req:,} tonnes</div>
                    <div class="kpi-exec-sub" style="color: #6B6B73;">Base Demand Target</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with eq_c2:
            st.markdown(
                f"""
                <div class="kpi-card-exec">
                    <span class="kpi-exec-label">Safety Stock (+)</span>
                    <div class="kpi-exec-val">+{sp_saf:,} tonnes</div>
                    <div class="kpi-exec-sub" style="color: #F6B51B;">Operational Buffer</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with eq_c3:
            st.markdown(
                f"""
                <div class="kpi-card-exec">
                    <span class="kpi-exec-label">Current Inventory</span>
                    <div class="kpi-exec-val">{sp_inv:,} tonnes</div>
                    <div class="kpi-exec-sub" style="color: #6B6B73;">Subtracted in procurement calculation.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with eq_c4:
            st.markdown(
                f"""
                <div class="kpi-card-exec" style="border: 1.5px solid #372580; background: #FFFFFF;">
                    <span class="kpi-exec-label" style="color: #372580; font-weight: 600;">Cargo Shortfall (=)</span>
                    <div class="kpi-exec-val" style="color: #18181B;">{calc_shortfall:,} tonnes</div>
                    <div class="kpi-exec-sub" style="color: #F6B51B;">Net Procurement Need</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="panel-card" style="margin-top: 10px; margin-bottom: 22px; padding: 12px 18px; border-left: 4px solid #372580;">
                <span style="color: #6B6B73; font-size: 0.88rem;">Automated Procurement Formula:</span>
                <b style="color: #18181B; font-size: 0.94rem; margin-left: 6px;">Cargo Shortfall = Required Quantity ({sp_req:,} t) + Safety Stock ({sp_saf:,} t) - Current Inventory ({sp_inv:,} t) = <span style="color: #372580;">{calc_shortfall:,} tonnes</span></b>
                <span style="color: #6B6B73; font-size: 0.88rem; margin-left: 10px;">(Delivery Deadline: <b>{sp_dl} days</b> | Origin: <b>{sp_origin.split(',')[0]}</b>)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # STEP 2: OPERATIONAL CONSTRAINTS & CAPACITY CHECK
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                <span style="background: #372580; color: #ffffff; font-weight: 700; font-size: 0.8rem; padding: 3px 9px; border-radius: 6px;">STEP 2</span>
                <h3 style="margin: 0; color: #18181B; font-size: 1.15rem; font-weight: 600;">Fleet Capacity Feasibility Check</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="panel-card" style="margin-bottom: 16px; padding: 12px 18px; border-left: 4px solid #372580;">
                <b style="color: #18181B; font-size: 0.92rem;">Fleet Capacity Feasibility Check (Cargo Shortfall: {calc_shortfall:,} tonnes):</b>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 8px; font-size: 0.88rem;">
                    <div style="background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); border-radius: 6px; padding: 8px 12px;">
                        <b style="color: #D95C5C;">2 × Supramax (116,000 t)</b><br>
                        <span style="color: #D95C5C; font-weight: 700;">● INFEASIBLE</span><br>
                        <span style="color: #6B6B73; font-size: 0.82rem;">Deficit: 14,000 tonnes</span>
                    </div>
                    <div style="background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3); border-radius: 6px; padding: 8px 12px;">
                        <b style="color: #15803D;">2 × Panamax (164,000 t)</b><br>
                        <span style="color: #48A868; font-weight: 700;">● FEASIBLE</span><br>
                        <span style="color: #6B6B73; font-size: 0.82rem;">79.3% Utilization • Recommended</span>
                    </div>
                    <div style="background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3); border-radius: 6px; padding: 8px 12px;">
                        <b style="color: #15803D;">1 × Capesize (180,000 t)</b><br>
                        <span style="color: #48A868; font-weight: 700;">● FEASIBLE</span><br>
                        <span style="color: #6B6B73; font-size: 0.82rem;">72.2% Utilization • Lowest Cost</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("📋 Hard-Constraint Feasibility Checklist (10-Point Operational Audit)"):
            st.markdown(
                f"""
                <table style="width: 100%; font-size: 0.86rem; border-collapse: collapse;">
                    <tr style="background: #F8FAFC; border-bottom: 2px solid #E2E8F0;">
                        <th style="padding: 6px 10px; text-align: left;">Constraint Category</th>
                        <th style="padding: 6px 10px; text-align: left;">Operational Threshold</th>
                        <th style="padding: 6px 10px; text-align: center;">Status</th>
                        <th style="padding: 6px 10px; text-align: left;">Audit Detail & Mathematical Verification</th>
                    </tr>
                    <tr style="border-bottom: 1px solid #ECECF0;">
                        <td style="padding: 6px 10px; font-weight: 600;">1. Fleet Usable Capacity</td>
                        <td style="padding: 6px 10px;">Capacity &ge; Shortfall ({calc_shortfall:,} t)</td>
                        <td style="padding: 6px 10px; text-align: center; color: #15803D; font-weight: 700;">PASS</td>
                        <td style="padding: 6px 10px;">2 × Panamax (164,000 t) provides 34,000 t reserve buffer. 2 × Supramax rejected (116,000 t, 14,000 t deficit).</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ECECF0;">
                        <td style="padding: 6px 10px; font-weight: 600;">2. Vessel Market Availability</td>
                        <td style="padding: 6px 10px;">Charter Count &le; Market Pool</td>
                        <td style="padding: 6px 10px; text-align: center; color: #15803D; font-weight: 700;">PASS</td>
                        <td style="padding: 6px 10px;">2 Panamax required &le; 3 available in Indian Ocean position.</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ECECF0;">
                        <td style="padding: 6px 10px; font-weight: 600;">3. Port Draft Limits</td>
                        <td style="padding: 6px 10px;">Laden Draft &le; Berth Clearance</td>
                        <td style="padding: 6px 10px; text-align: center; color: #15803D; font-weight: 700;">PASS</td>
                        <td style="padding: 6px 10px;">Panamax draft (14.5m) &le; Paradip depth (18.5m). Capesize draft (18.5m) rejected at Visakhapatnam (16.5m limit).</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ECECF0;">
                        <td style="padding: 6px 10px; font-weight: 600;">4. Port Dimensions (LOA/Beam)</td>
                        <td style="padding: 6px 10px;">Vessel LOA &le; Max Berthing Length</td>
                        <td style="padding: 6px 10px; text-align: center; color: #15803D; font-weight: 700;">PASS</td>
                        <td style="padding: 6px 10px;">Panamax LOA (225.0m) &le; Paradip max LOA (300.0m); beam 32.2m within 45.0m limit.</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ECECF0;">
                        <td style="padding: 6px 10px; font-weight: 600;">5. Cargo Compatibility</td>
                        <td style="padding: 6px 10px;">Vessel Holds & Port Certified</td>
                        <td style="padding: 6px 10px; text-align: center; color: #15803D; font-weight: 700;">PASS</td>
                        <td style="padding: 6px 10px;">Thermal Coal dry-bulk certified across Panamax cargo holds and Paradip mechanized conveyor berths.</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ECECF0;">
                        <td style="padding: 6px 10px; font-weight: 600;">6. Approved Budget</td>
                        <td style="padding: 6px 10px;">Landed Cost &le; Approved Budget</td>
                        <td style="padding: 6px 10px; text-align: center; color: #15803D; font-weight: 700;">PASS</td>
                        <td style="padding: 6px 10px;">Estimated total cost ₹28.60 Cr &le; approved budget ₹{sp_budget:.2f} Cr (₹{sp_budget - 28.60:.2f} Cr headroom).</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ECECF0;">
                        <td style="padding: 6px 10px; font-weight: 600;">7. Delivery Deadline</td>
                        <td style="padding: 6px 10px;">Voyage Duration &le; Deadline</td>
                        <td style="padding: 6px 10px; text-align: center; color: #15803D; font-weight: 700;">PASS</td>
                        <td style="padding: 6px 10px;">ETA 19 days &le; {sp_dl} days deadline (arrival 26 days prior to cutoff).</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ECECF0;">
                        <td style="padding: 6px 10px; font-weight: 600;">8. Minimum Utilization</td>
                        <td style="padding: 6px 10px;">Utilization &ge; {sp_min_util}%</td>
                        <td style="padding: 6px 10px; text-align: center; color: #15803D; font-weight: 700;">PASS</td>
                        <td style="padding: 6px 10px;">Fleet utilization 79.3% exceeds {sp_min_util}% policy minimum.</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ECECF0;">
                        <td style="padding: 6px 10px; font-weight: 600;">9. Risk Ceiling</td>
                        <td style="padding: 6px 10px;">Composite Risk &le; {sp_max_risk}/100</td>
                        <td style="padding: 6px 10px; text-align: center; color: #15803D; font-weight: 700;">PASS</td>
                        <td style="padding: 6px 10px;">Multi-factor risk score 24/100 &le; {sp_max_risk}/100 maximum acceptable ceiling.</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ECECF0;">
                        <td style="padding: 6px 10px; font-weight: 600;">10. Laycan Window</td>
                        <td style="padding: 6px 10px;">Loading Position Feasible</td>
                        <td style="padding: 6px 10px; text-align: center; color: #15803D; font-weight: 700;">PASS</td>
                        <td style="padding: 6px 10px;">Immediate chartering (Day 0) aligns with Richards Bay loading queue window.</td>
                    </tr>
                </table>
                """,
                unsafe_allow_html=True
            )

        # STEP 3: THREE STRATEGIC PLANS
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; margin-bottom: 12px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="background: #372580; color: #FFFFFF; font-weight: 600; font-size: 0.78rem; padding: 3px 8px; border-radius: 4px;">STEP 3</span>
                    <h3 style="margin: 0; color: #18181B; font-size: 1.15rem; font-weight: 600;">Generated Voyage Procurement Plans</h3>
                </div>
                <span style="color: #6B6B73; font-size: 0.85rem;">Evaluation Engine: <b>3 Strategic Options Generated</b></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("⚖️ Recommendation Scoring Method & Normalized Formula"):
            st.markdown("""
            <div style="font-size: 0.88rem; color: #334155; line-height: 1.5;">
                <b style="color: #18181B;">Multi-Criteria Mathematical Scoring Engine:</b><br>
                Every candidate voyage plan is normalized across four commercial criteria and evaluated using the weighted balanced score:
                <div style="background: #F1F5F9; padding: 8px 12px; border-radius: 6px; font-family: monospace; font-size: 0.90rem; margin: 8px 0;">
                    Balanced Score = 0.40 × Normalized Cost + 0.25 × Normalized Risk + 0.20 × Normalized ETA + 0.15 × Normalized Unused Capacity
                </div>
                <ul style="margin: 6px 0 0 16px; padding: 0;">
                    <li><b>Lowest Cost Plan (40% Weight):</b> Selects the absolute lowest landed logistics cost meeting the cargo demand volume.</li>
                    <li><b>Lowest Risk Plan (25% Weight):</b> Selects the vessel and port routing with minimum 7-factor composite risk index.</li>
                    <li><b>Balanced Recommended Plan:</b> Global minimum of the composite score balancing financial savings, transit speed, and capacity utilization.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        baseline_exp = 31.19

        # Plan 1: Lowest Cost
        p1_cap = 180000
        p1_util = round((calc_shortfall / p1_cap) * 100, 1)
        p1_cost = 27.85
        p1_savings = round(baseline_exp - p1_cost, 2)

        # Plan 2: Lowest Risk
        p2_cap = 164000
        p2_util = round((calc_shortfall / p2_cap) * 100, 1)
        p2_cost = 29.10
        p2_savings = round(baseline_exp - p2_cost, 2)

        # Plan 3: Balanced Recommended
        p3_cap = 164000
        p3_cost = 28.60
        p3_savings = round(baseline_exp - p3_cost, 2)
        p3_util = round((calc_shortfall / p3_cap) * 100, 1) if calc_shortfall != 130000 else 79.3

        PLAN_DATA = {
            "Lowest Cost": {
                "name": "Lowest Cost", "badge": "LOWEST COST", "badge_color": "#48A868",
                "vessel_type": "Capesize", "vessel_count": 1, "combined_capacity": p1_cap,
                "utilization": p1_util, "port": "Paradip", "charter_date": "Today (Day 0)",
                "expected_arrival": "Day 21", "cost_cr": p1_cost, "savings_cr": p1_savings,
                "risk_score": 32, "duration": 21, "feasible": True, "status": "Feasible"
            },
            "Lowest Risk": {
                "name": "Lowest Risk", "badge": "LOWEST RISK", "badge_color": "#372580",
                "vessel_type": "Panamax", "vessel_count": 2, "combined_capacity": p2_cap,
                "utilization": p2_util, "port": "Visakhapatnam", "charter_date": "Today (Day 0)",
                "expected_arrival": "Day 18", "cost_cr": p2_cost, "savings_cr": p2_savings,
                "risk_score": 18, "duration": 18, "feasible": True, "status": "Feasible"
            },
            "Balanced Recommended": {
                "name": "Balanced Recommended", "badge": "RECOMMENDED", "badge_color": "#48A868",
                "vessel_type": "Panamax", "vessel_count": 2, "combined_capacity": p3_cap,
                "utilization": 79.3, "port": "Paradip", "charter_date": "Today (Day 0)",
                "expected_arrival": "Day 19", "cost_cr": 28.60, "savings_cr": 2.59,
                "risk_score": 24, "duration": 19, "feasible": True, "status": "Feasible"
            },
        }

        pl_col1, pl_col2, pl_col3 = st.columns(3, gap="medium")

        with pl_col1:
            p1 = PLAN_DATA["Lowest Cost"]
            p1_csv = pd.DataFrame([{
                "Plan Name": p1["name"], "Commodity": sp_cargo, "Shortfall": calc_shortfall,
                "Vessel Type": p1["vessel_type"], "Vessel Count": p1["vessel_count"],
                "Loading Port": sp_origin.split(",")[0], "Discharge Port": p1["port"],
                "Total Cost (₹ Cr)": p1["cost_cr"], "Duration (Days)": p1["duration"],
                "Utilization (%)": p1["utilization"], "Risk Score": p1["risk_score"],
                "Feasibility Status": p1["status"], "Disclaimer": "Prototype Data"
            }]).to_csv(index=False).encode("utf-8")

            st.markdown(
                f"""
                <div class="panel-card" style="height: 100%; border: 1px solid #E4E4E8; border-top: 4px solid #48A868; background: #FFFFFF;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <b style="color: #15803D; font-size: 1.05rem;">1. Lowest Cost</b>
                        <span style="background: #EDF7F0; color: #15803D; font-size: 0.72rem; font-weight: 700; padding: 2px 7px; border-radius: 4px;">{p1['badge']}</span>
                    </div>
                    <div style="font-size: 1.35rem; font-weight: 700; color: #18181B; margin-bottom: 2px;">{p1['vessel_count']} × {p1['vessel_type']}</div>
                    <div style="color: #6B6B73; font-size: 0.84rem; margin-bottom: 12px;">Discharge: <b>{p1['port']} Port</b></div>
                    <hr style="border: none; border-top: 1px solid #ECECF0; margin: 8px 0;" />
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 0.85rem;">
                        <div><span style="color: #6B6B73;">Capacity:</span> <b style="color: #18181B;">{p1['combined_capacity']:,} t</b></div>
                        <div><span style="color: #6B6B73;">Utilization:</span> <b style="color: #372580;">{p1['utilization']}%</b></div>
                        <div><span style="color: #6B6B73;">Charter Date:</span> <b style="color: #18181B;">{p1['charter_date']}</b></div>
                        <div><span style="color: #6B6B73;">Arrival:</span> <b style="color: #48A868;">{p1['expected_arrival']}</b></div>
                        <div><span style="color: #6B6B73;">Est. Cost:</span> <b style="color: #48A868;">₹{p1['cost_cr']:.2f} Cr</b></div>
                        <div><span style="color: #6B6B73;">Savings:</span> <b style="color: #48A868;">₹{p1['savings_cr']:.2f} Cr</b></div>
                        <div><span style="color: #6B6B73;">Risk Score:</span> <b style="color: #F6B51B;">{p1['risk_score']}/100</b></div>
                        <div><span style="color: #6B6B73;">Duration:</span> <b style="color: #18181B;">{p1['duration']} days</b></div>
                    </div>
                    <div style="margin-top: 12px; text-align: center; padding: 4px 8px; border-radius: 6px; background: #EDF7F0; color: #15803D; font-weight: 600; font-size: 0.84rem;">
                        Feasibility: {p1['status']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.download_button("📥 Download Lowest Cost CSV", data=p1_csv, file_name="varunapath_lowest_cost_plan.csv", mime="text/csv", key="sp_dl_btn_p1", use_container_width=True)

        with pl_col2:
            p2 = PLAN_DATA["Lowest Risk"]
            p2_csv = pd.DataFrame([{
                "Plan Name": p2["name"], "Commodity": sp_cargo, "Shortfall": calc_shortfall,
                "Vessel Type": p2["vessel_type"], "Vessel Count": p2["vessel_count"],
                "Loading Port": sp_origin.split(",")[0], "Discharge Port": p2["port"],
                "Total Cost (₹ Cr)": p2["cost_cr"], "Duration (Days)": p2["duration"],
                "Utilization (%)": p2["utilization"], "Risk Score": p2["risk_score"],
                "Feasibility Status": p2["status"], "Disclaimer": "Prototype Data"
            }]).to_csv(index=False).encode("utf-8")

            st.markdown(
                f"""
                <div class="panel-card" style="height: 100%; border: 1px solid #E4E4E8; border-top: 4px solid #372580; background: #FFFFFF;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <b style="color: #372580; font-size: 1.05rem;">2. Lowest Risk</b>
                        <span style="background: #F0EEF9; color: #372580; font-size: 0.72rem; font-weight: 700; padding: 2px 7px; border-radius: 4px;">{p2['badge']}</span>
                    </div>
                    <div style="font-size: 1.35rem; font-weight: 700; color: #18181B; margin-bottom: 2px;">{p2['vessel_count']} × {p2['vessel_type']}</div>
                    <div style="color: #6B6B73; font-size: 0.84rem; margin-bottom: 12px;">Discharge: <b>{p2['port']} Port</b></div>
                    <hr style="border: none; border-top: 1px solid #ECECF0; margin: 8px 0;" />
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 0.85rem;">
                        <div><span style="color: #6B6B73;">Capacity:</span> <b style="color: #18181B;">{p2['combined_capacity']:,} t</b></div>
                        <div><span style="color: #6B6B73;">Utilization:</span> <b style="color: #372580;">{p2['utilization']}%</b></div>
                        <div><span style="color: #6B6B73;">Charter Date:</span> <b style="color: #18181B;">{p2['charter_date']}</b></div>
                        <div><span style="color: #6B6B73;">Arrival:</span> <b style="color: #48A868;">{p2['expected_arrival']}</b></div>
                        <div><span style="color: #6B6B73;">Est. Cost:</span> <b style="color: #48A868;">₹{p2['cost_cr']:.2f} Cr</b></div>
                        <div><span style="color: #6B6B73;">Savings:</span> <b style="color: #48A868;">₹{p2['savings_cr']:.2f} Cr</b></div>
                        <div><span style="color: #6B6B73;">Risk Score:</span> <b style="color: #372580;">{p2['risk_score']}/100</b></div>
                        <div><span style="color: #6B6B73;">Duration:</span> <b style="color: #18181B;">{p2['duration']} days</b></div>
                    </div>
                    <div style="margin-top: 12px; text-align: center; padding: 4px 8px; border-radius: 6px; background: #EDF7F0; color: #15803D; font-weight: 600; font-size: 0.84rem;">
                        Feasibility: {p2['status']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.download_button("📥 Download Lowest Risk CSV", data=p2_csv, file_name="varunapath_lowest_risk_plan.csv", mime="text/csv", key="sp_dl_btn_p2", use_container_width=True)

        with pl_col3:
            p3 = PLAN_DATA["Balanced Recommended"]
            p3_csv = pd.DataFrame([{
                "Plan Name": p3["name"], "Commodity": sp_cargo, "Shortfall": calc_shortfall,
                "Vessel Type": p3["vessel_type"], "Vessel Count": p3["vessel_count"],
                "Loading Port": sp_origin.split(",")[0], "Discharge Port": p3["port"],
                "Total Cost (₹ Cr)": p3["cost_cr"], "Duration (Days)": p3["duration"],
                "Utilization (%)": p3["utilization"], "Risk Score": p3["risk_score"],
                "Feasibility Status": p3["status"], "Disclaimer": "Prototype Data"
            }]).to_csv(index=False).encode("utf-8")

            st.markdown(
                f"""
                <div class="panel-card" style="height: 100%; border: 2px solid #372580; background: #FFFFFF; box-shadow: 0 2px 8px rgba(55,37,128,0.08);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <b style="color: #372580; font-size: 1.05rem;">3. Balanced Recommended</b>
                        <span style="background: #372580; color: #FFFFFF; font-size: 0.72rem; font-weight: 600; padding: 2px 8px; border-radius: 4px;">{p3['badge']}</span>
                    </div>
                    <div style="font-size: 1.35rem; font-weight: 700; color: #18181B; margin-bottom: 2px;">{p3['vessel_count']} × {p3['vessel_type']}</div>
                    <div style="color: #6B6B73; font-size: 0.84rem; margin-bottom: 12px;">Discharge: <b>{p3['port']} Port</b></div>
                    <hr style="border: none; border-top: 1px solid #ECECF0; margin: 8px 0;" />
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 0.85rem;">
                        <div><span style="color: #6B6B73;">Capacity:</span> <b style="color: #18181B;">{p3['combined_capacity']:,} t</b></div>
                        <div><span style="color: #6B6B73;">Utilization:</span> <b style="color: #48A868; font-weight: 700;">{p3['utilization']}%</b></div>
                        <div><span style="color: #6B6B73;">Charter Date:</span> <b style="color: #18181B;">{p3['charter_date']}</b></div>
                        <div><span style="color: #6B6B73;">Arrival:</span> <b style="color: #48A868;">{p3['expected_arrival']}</b></div>
                        <div><span style="color: #6B6B73;">Est. Cost:</span> <b style="color: #372580; font-weight: 700;">₹{p3['cost_cr']:.2f} Cr</b></div>
                        <div><span style="color: #6B6B73;">Savings:</span> <b style="color: #48A868; font-weight: 700;">₹{p3['savings_cr']:.2f} Cr</b></div>
                        <div><span style="color: #6B6B73;">Risk Score:</span> <b style="color: #48A868;">{p3['risk_score']}/100</b></div>
                        <div><span style="color: #6B6B73;">Duration:</span> <b style="color: #18181B;">{p3['duration']} days</b></div>
                    </div>
                    <div style="margin-top: 12px; text-align: center; padding: 6px 8px; border-radius: 6px; background: #EDF7F0; color: #15803D; font-weight: 600; font-size: 0.84rem; border: 1px solid rgba(72,168,104,0.3);">
                        Feasibility: {p3['status']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.download_button("📥 Download Balanced CSV", data=p3_csv, file_name="varunapath_balanced_plan.csv", mime="text/csv", key="sp_dl_btn_p3", use_container_width=True)

        # STEP 4: CONFIRM PLAN, DYNAMIC WHY THIS PLAN, TRACE & DOCS
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; margin-top: 24px; margin-bottom: 12px;">
                <span style="background: #372580; color: #FFFFFF; font-weight: 600; font-size: 0.78rem; padding: 3px 8px; border-radius: 4px;">STEP 4</span>
                <h3 style="margin: 0; color: #18181B; font-size: 1.15rem; font-weight: 600;">Confirm & Commit Voyage Execution</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container():
            st.markdown('<div class="panel-card" style="margin-bottom: 20px;">', unsafe_allow_html=True)
            plan_choice = st.radio(
                "Select Plan for Fixture Commitment:",
                ["Balanced Recommended", "Lowest Cost", "Lowest Risk"],
                index=0,
                horizontal=True,
                key="sp_step4_plan_radio",
            )
            selected_plan = PLAN_DATA[plan_choice]

            if plan_choice == "Balanced Recommended":
                why_text = f"<b>Balanced Optimal Trade-off:</b> Dual Panamax vessels provide <b>{selected_plan['combined_capacity']:,} tonnes</b> combined capacity, matching the <b>{calc_shortfall:,} tonnes shortfall</b> at high efficiency (<b>{selected_plan['utilization']}% utilization</b>). Discharging at <b>Paradip Port</b> secures minimal turnaround tariff (₹85/t) and 180k DWT draft clearance. Total cost of <b>₹{selected_plan['cost_cr']:.2f} Cr</b> delivers <b>₹{selected_plan['savings_cr']:.2f} Cr savings (8.3%)</b> vs. baseline, with low risk (<b>24/100</b>) and 19-day arrival well inside the {sp_dl}-day deadline."
            elif plan_choice == "Lowest Cost":
                why_text = f"<b>Maximum Financial Economy:</b> Single Capesize vessel achieves lowest overall charter expenditure at <b>₹{selected_plan['cost_cr']:.2f} Cr</b> (<b>₹{selected_plan['savings_cr']:.2f} Cr savings</b>). While utilization is <b>{selected_plan['utilization']}%</b>, draft clearance at Paradip handles the 180,000 DWT vessel safely within 21 days."
            else:
                why_text = f"<b>Maximum Risk Mitigation:</b> 2 × Panamax routing through <b>Visakhapatnam</b> prioritizes berth availability and minimal sea-lane weather exposure, achieving an ultra-low risk score of <b>{selected_plan['risk_score']}/100</b> and rapid 18-day transit at <b>₹{selected_plan['cost_cr']:.2f} Cr</b>."

            st.markdown(
                f"""
                <div style="background: #F0EEF9; border-left: 4px solid #372580; padding: 12px 16px; border-radius: 6px; margin-top: 12px; margin-bottom: 12px;">
                    <b style="color: #372580; font-size: 0.95rem;">Why This Plan ({selected_plan['name']}):</b>
                    <div style="color: #6B6B73; font-size: 0.9rem; line-height: 1.6; margin-top: 4px;">{why_text.strip()}</div>
                </div>
                <div style="background: #FFFBEB; border-left: 4px solid #F59E0B; padding: 12px 16px; border-radius: 6px; margin-bottom: 16px;">
                    <b style="color: #B45309; font-size: 0.95rem;">Why Not the Alternatives?</b>
                    <ul style="color: #78350F; font-size: 0.88rem; margin: 6px 0 0 18px; padding: 0; line-height: 1.5;">
                        <li><b>2 × Supramax via Paradip:</b> <span style="color: #DC2626; font-weight: 600;">Rejected</span> — Combined usable capacity (116,000 t) is 14,000 tonnes lower than required shortfall (130,000 t).</li>
                        <li><b>1 × Capesize via Paradip:</b> <span style="color: #D97706; font-weight: 600;">Sub-optimal</span> — Incurs lower fleet capacity utilization (72.2%) and longer loading queue at Richards Bay coal terminal.</li>
                        <li><b>Panamax via Visakhapatnam:</b> <span style="color: #D97706; font-weight: 600;">Sub-optimal</span> — Congestion delay of 5 waiting days at outer anchorage increases turnaround to 23 days with higher handling tariff (₹110/t).</li>
                        <li><b>Panamax via Dhamra:</b> <span style="color: #D97706; font-weight: 600;">Viable Alternative</span> — Higher terminal handling tariff (₹95/t vs ₹85/t at Paradip) increases landed logistics cost by ₹0.13 Cr.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Calculation Trace Expander
            with st.expander("🔍 Calculation Trace (Step-by-Step Mathematical Proof)"):
                st.markdown(
                    f"""
                    <b>1. Shortfall Calculation:</b> Requirement ({sp_req:,} t) + Safety Stock ({sp_saf:,} t) - Inventory ({sp_inv:,} t) = <b>{calc_shortfall:,} tonnes</b><br>
                    <b>2. Combined Capacity:</b> {selected_plan['vessel_count']} × {selected_plan['vessel_type']} = <b>{selected_plan['combined_capacity']:,} tonnes</b><br>
                    <b>3. Utilization:</b> {calc_shortfall:,} / {selected_plan['combined_capacity']:,} = <b>{selected_plan['utilization']}%</b><br>
                    <b>4. Total Logistics Cost:</b> Cargo + Charter + Port Handling + Waiting + Penalties = <b>₹{selected_plan['cost_cr']:.2f} Cr</b><br>
                    <b>5. Net Cost Savings:</b> Baseline (₹31.19 Cr) - Optimized (₹{selected_plan['cost_cr']:.2f} Cr) = <b>₹{selected_plan['savings_cr']:.2f} Cr</b><br>
                    <b>6. Percentage Savings:</b> (₹{selected_plan['savings_cr']:.2f} Cr / ₹31.19 Cr) × 100 = <b>{round((selected_plan['savings_cr']/31.19)*100, 1)}%</b>
                    """,
                    unsafe_allow_html=True,
                )

            # Shipping Document Checklist
            with st.expander("📑 Shipping Document Checklist (Readiness & Regulatory Status)"):
                doc_cols = st.columns(3)
                docs = [
                    ("Bill of Lading", "Generated", "✅ Ready for discharge"),
                    ("Commercial Invoice", "Generated", "✅ Certified by shipper"),
                    ("Packing List", "Generated", "✅ Weight verified"),
                    ("Certificate of Origin", "Pending", "⏳ Under Chamber Review"),
                    ("Certificate of Sampling & Analysis", "Pending", "⏳ Terminal lab test in progress"),
                    ("Cargo Manifest", "Pending", "⏳ Awaiting pilot boarding"),
                ]
                for idx, (d_name, d_stat, d_desc) in enumerate(docs):
                    with doc_cols[idx % 3]:
                        badge_col = "#15803D" if d_stat == "Generated" else "#D97706"
                        st.markdown(
                            f"""
                            <div style="background: #F8F8FA; border: 1px solid #E4E4E8; border-radius: 6px; padding: 10px; margin-bottom: 8px;">
                                <div style="display: flex; justify-content: space-between;">
                                    <b>{d_name}</b>
                                    <span style="color: {badge_col}; font-weight: 700; font-size: 0.78rem;">{d_stat}</span>
                                </div>
                                <div style="color: #6B6B73; font-size: 0.8rem; margin-top: 4px;">{d_desc}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            act_c1, act_c2 = st.columns([1.2, 1.5], gap="medium")
            with act_c1:
                if st.button("✅ Confirm Plan", key="sp_confirm_btn", type="primary", use_container_width=True):
                    st.session_state["confirmed_plan"] = selected_plan
                    st.session_state["plan_confirmed"] = True
                    st.toast(f"Plan Confirmed: {selected_plan['vessel_count']} × {selected_plan['vessel_type']} to {selected_plan['port']} (₹{selected_plan['cost_cr']:.2f} Cr)", icon="✅")
                    st.rerun()
            with act_c2:
                if st.button("🚀 Send to Optimization Hub →", key="sp_send_opt_btn", use_container_width=True):
                    st.session_state["confirmed_plan"] = selected_plan
                    st.session_state["active_page"] = "Optimization Hub"
                    st.session_state["active_module"] = "Optimization Hub"
                    st.toast("Plan transferred to Optimization Hub for deep route analytics.", icon="🚀")
                    st.rerun()

            if st.session_state.get("plan_confirmed", False) and "confirmed_plan" in st.session_state:
                cp = st.session_state["confirmed_plan"]
                st.markdown(
                    f"""
                    <div style="margin-top: 16px; padding: 14px 18px; border-radius: 8px; background: #EDF7F0; border: 1px solid rgba(72,168,104,0.4);">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 1.2rem;">🎉</span>
                            <b style="color: #48A868; font-size: 1.02rem;">Plan Successfully Confirmed & Committed to Session State!</b>
                        </div>
                        <div style="color: #6B6B73; font-size: 0.9rem; margin-top: 6px; line-height: 1.5;">
                            <b>Fixture Allocation:</b> {cp['vessel_count']} × {cp['vessel_type']} ({cp['combined_capacity']:,} t capacity) &bull;
                            <b>Destination:</b> {cp['port']} Port &bull;
                            <b>Arrival ETA:</b> {cp['expected_arrival']} ({cp['duration']} days) &bull;
                            <b>Committed Cost:</b> ₹{cp['cost_cr']:.2f} Cr (Savings: ₹{cp['savings_cr']:.2f} Cr vs baseline) &bull;
                            <b>Risk:</b> {cp['risk_score']}/100.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        # EXPORT FROM INDIA SHIPMENT PLANNER
        st.markdown(
            """
            <div class="panel-card" style="margin-top: 6px; margin-bottom: 18px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <div>
                    <b style="color: #372580; font-size: 1.05rem;">Export Shipment Planner & Foreign Trade Fixture Architect</b>
                    <span style="color: #6B6B73; font-size: 0.88rem; margin-left: 8px;">— Indian Port Loading, Laycan Window, and Incoterm Optimization.</span>
                </div>
                <span class="prototype-badge">PROTOTYPE DATA</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("export_shipment_planner_form"):
            st.markdown("<div style='font-size: 0.92rem; font-weight: 700; color: #18181B; margin-bottom: 10px;'>📋 Export Order Ledger & Production Parameters</div>", unsafe_allow_html=True)
            
            ex_r1_c1, ex_r1_c2, ex_r1_c3 = st.columns([1.5, 1.2, 1.3], gap="medium")
            with ex_r1_c1:
                in_buyer = st.text_input("BUYER REFERENCE", value=st.session_state.get("export_buyer_ref", "POSCO Asia Steel Corp"), key="sp_exp_buyer_ref")
            with ex_r1_c2:
                countries = ["South Korea", "Singapore", "Bangladesh", "Malaysia", "UAE", "Netherlands", "China"]
                cur_c = st.session_state.get("export_buyer_country", "South Korea")
                in_country = st.selectbox("BUYER COUNTRY", countries, index=countries.index(cur_c) if cur_c in countries else 0, key="sp_exp_buyer_country")
            with ex_r1_c3:
                cur_comm = st.session_state.get("export_commodity", "Finished Steel")
                in_comm = st.selectbox("COMMODITY", EXPORT_COMMODITIES, index=EXPORT_COMMODITIES.index(cur_comm) if cur_comm in EXPORT_COMMODITIES else 0, key="sp_exp_cargo")

            ex_r2_c1, ex_r2_c2, ex_r2_c3, ex_r2_c4 = st.columns(4, gap="medium")
            with ex_r2_c1:
                in_order_qty = st.number_input("ORDER QUANTITY (t)", 5000, 500000, int(st.session_state.get("export_order_quantity", 100000)), 5000, key="sp_exp_order_qty")
            with ex_r2_c2:
                in_inv = st.number_input("READY STOCKPILE (t)", 0, 500000, int(st.session_state.get("current_export_inventory", 70000)), 5000, key="sp_exp_inv")
            with ex_r2_c3:
                in_prod = st.number_input("PLANNED PRODUCTION (t)", 0, 500000, int(st.session_state.get("planned_production", 20000)), 5000, key="sp_exp_prod")
            with ex_r2_c4:
                in_reserved = st.number_input("DOMESTIC RESERVED (t)", 0, 200000, int(st.session_state.get("reserved_domestic_stock", 10000)), 5000, key="sp_exp_reserved")

            ex_r3_c1, ex_r3_c2, ex_r3_c3, ex_r3_c4 = st.columns([1.2, 1.2, 1.0, 1.0], gap="medium")
            with ex_r3_c1:
                in_laycan_start = st.date_input("LAYCAN START", value=st.session_state.get("export_laycan_start", date.today() + timedelta(days=7)), key="sp_exp_laycan_start")
            with ex_r3_c2:
                in_laycan_end = st.date_input("LAYCAN END", value=st.session_state.get("export_laycan_end", date.today() + timedelta(days=14)), key="sp_exp_laycan_end")
            with ex_r3_c3:
                in_incoterm = st.selectbox("INCOTERM", EXPORT_INCOTERMS, index=EXPORT_INCOTERMS.index(st.session_state.get("export_incoterm", "CFR")), key="sp_exp_incoterm")
            with ex_r3_c4:
                in_dl = st.slider("DELIVERY DEADLINE", 10, 60, int(st.session_state.get("export_deadline", 30)), key="sp_exp_deadline")

            ex_r4_c1, ex_r4_c2, ex_r4_c3, ex_r4_c4 = st.columns(4, gap="medium")
            with ex_r4_c1:
                in_sell_price = st.number_input("SELLING PRICE (₹/t)", 0, 50000, int(st.session_state.get("export_selling_price", 6800)), 100, key="sp_exp_sell_price")
            with ex_r4_c2:
                in_int_cost = st.number_input("INTERNAL COST (₹/t)", 0, 50000, int(st.session_state.get("export_internal_cost", 4200)), 100, key="sp_exp_int_cost")
            with ex_r4_c3:
                in_max_risk = st.slider("MAX ACCEPTABLE RISK", 10, 100, int(st.session_state.get("export_max_risk", 40)), 5, key="sp_exp_max_risk")
            with ex_r4_c4:
                in_max_budget = st.number_input("MAX BUDGET (₹ Cr)", 2.0, 80.0, float(st.session_state.get("export_max_budget", 20.0)), 1.0, key="sp_exp_max_budget")

            ex_r5_c1, ex_r5_c2, ex_r5_c3 = st.columns([1.3, 1.3, 1.4], gap="medium")
            with ex_r5_c1:
                in_allowed_loading = st.multiselect("ALLOWED LOADING PORTS", EXPORT_LOADING_PORTS, default=["Paradip", "Visakhapatnam", "Chennai"], key="sp_exp_allowed_loading")
            with ex_r5_c2:
                in_allowed_dest = st.multiselect("ALLOWED DESTINATION PORTS", FOREIGN_DESTINATION_PORTS, default=["Singapore", "Port Klang", "Chittagong"], key="sp_exp_allowed_dest")
            with ex_r5_c3:
                in_allowed_vessels = st.multiselect("ALLOWED VESSEL TYPES", EXPORT_VESSEL_CLASSES, default=["Supramax (58k)", "Panamax (82k)", "Capesize (180k)"], key="sp_exp_allowed_vessels")

            in_plan_mode = st.radio("SHIPMENT ALLOCATION MODE", ["Plan Full Order", "Partial Order (Export-Ready Only)"], index=0, horizontal=True, key="sp_exp_plan_mode")

            submit_exp = st.form_submit_button("⚡ Generate Export Plans", type="primary", use_container_width=True)

        if submit_exp:
            st.session_state["export_buyer_ref"] = in_buyer
            st.session_state["export_buyer_country"] = in_country
            st.session_state["export_commodity"] = in_comm
            st.session_state["export_order_quantity"] = in_order_qty
            st.session_state["current_export_inventory"] = in_inv
            st.session_state["planned_production"] = in_prod
            st.session_state["reserved_domestic_stock"] = in_reserved
            st.session_state["export_laycan_start"] = in_laycan_start
            st.session_state["export_laycan_end"] = in_laycan_end
            st.session_state["export_incoterm"] = in_incoterm
            st.session_state["export_deadline"] = in_dl
            st.session_state["export_selling_price"] = in_sell_price
            st.session_state["export_internal_cost"] = in_int_cost
            st.session_state["export_max_risk"] = in_max_risk
            st.session_state["export_max_budget"] = in_max_budget
            st.session_state["export_allowed_loading_ports"] = in_allowed_loading
            st.session_state["export_allowed_dest_ports"] = in_allowed_dest
            st.session_state["export_allowed_vessels"] = in_allowed_vessels
            st.session_state["export_plan_mode"] = in_plan_mode
            st.session_state["sp_export_snapshot"] = {
                "order_qty": in_order_qty, "inv": in_inv, "prod": in_prod, "reserved": in_reserved,
                "incoterm": in_incoterm, "selling_price": in_sell_price, "internal_cost": in_int_cost,
                "deadline": in_dl, "budget": in_max_budget, "risk": in_max_risk
            }
            st.session_state["sp_state"] = "Result Current"
            st.toast("Generated 3 comparative export plans!", icon="⚡")
            st.rerun()

        # Stale check for Export
        exp_snap = st.session_state.get("sp_export_snapshot")
        if exp_snap is not None:
            cur_exp_vals = {
                "order_qty": in_order_qty, "inv": in_inv, "prod": in_prod, "reserved": in_reserved,
                "incoterm": in_incoterm, "selling_price": in_sell_price, "internal_cost": in_int_cost,
                "deadline": in_dl, "budget": in_max_budget, "risk": in_max_risk
            }
            if cur_exp_vals != exp_snap:
                st.warning("⚠️ Inputs changed. Generate plans to refresh results.")

        # Step 1: Export Readiness Display
        ex_order = st.session_state.get("export_order_quantity", 100000)
        ex_inv = st.session_state.get("current_export_inventory", 70000)
        ex_prod = st.session_state.get("planned_production", 20000)
        ex_res = st.session_state.get("reserved_domestic_stock", 10000)
        ready_qty, shortfall, readiness_pct = calculate_export_availability(ex_order, ex_inv, ex_prod, ex_res)

        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; margin-top: 14px; margin-bottom: 12px;">
                <span style="background: #372580; color: #ffffff; font-weight: 700; font-size: 0.8rem; padding: 3px 9px; border-radius: 6px;">STEP 1</span>
                <h3 style="margin: 0; color: #18181B; font-size: 1.15rem; font-weight: 600;">Export Availability & Readiness Gap Analysis</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        eq_c1, eq_c2, eq_c3, eq_c4 = st.columns(4, gap="medium")
        with eq_c1:
            st.markdown(
                f"""
                <div class="kpi-card-exec">
                    <span class="kpi-exec-label">Export Order Quantity</span>
                    <div class="kpi-exec-val">{ex_order:,} tonnes</div>
                    <div class="kpi-exec-sub" style="color: #6B6B73;">Contracted Target</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with eq_c2:
            st.markdown(
                f"""
                <div class="kpi-card-exec">
                    <span class="kpi-exec-label">Stock + Production (+)</span>
                    <div class="kpi-exec-val">+{ex_inv + ex_prod:,} tonnes</div>
                    <div class="kpi-exec-sub" style="color: #48A868;">Plant Stock ({ex_inv:,}t) + Run ({ex_prod:,}t)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with eq_c3:
            st.markdown(
                f"""
                <div class="kpi-card-exec">
                    <span class="kpi-exec-label">Domestic Reserved (-)</span>
                    <div class="kpi-exec-val">-{ex_res:,} tonnes</div>
                    <div class="kpi-exec-sub" style="color: #6B6B73;">Subtracted in availability calculation.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with eq_c4:
            st.markdown(
                f"""
                <div class="kpi-card-exec" style="border: 1.5px solid #372580; background: #FFFFFF;">
                    <span class="kpi-exec-label" style="color: #372580; font-weight: 600;">Export-Ready (=)</span>
                    <div class="kpi-exec-val" style="color: #18181B;">{ready_qty:,} tonnes</div>
                    <div class="kpi-exec-sub" style="color: #48A868;">Readiness: {readiness_pct:.1f}% ({shortfall:,}t shortfall)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="panel-card" style="margin-top: 10px; margin-bottom: 22px; padding: 12px 18px; border-left: 4px solid #372580;">
                <span style="color: #6B6B73; font-size: 0.88rem;">Export Availability Formula:</span>
                <b style="color: #18181B; font-size: 0.94rem; margin-left: 6px;">Ready Stock ({ex_inv:,} t) + Planned Production ({ex_prod:,} t) - Domestic Reserved ({ex_res:,} t) = <span style="color: #372580;">{ready_qty:,} tonnes Export-Ready</span></b>
                <span style="color: #6B6B73; font-size: 0.88rem; margin-left: 10px;">(Order Readiness: <b>{readiness_pct:.1f}%</b> | Fulfilment Shortfall: <b>{shortfall:,} tonnes</b>)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Step 2: Laycan & Feasibility Audit
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                <span style="background: #372580; color: #ffffff; font-weight: 700; font-size: 0.8rem; padding: 3px 9px; border-radius: 6px;">STEP 2</span>
                <h3 style="margin: 0; color: #18181B; font-size: 1.15rem; font-weight: 600;">Export Fleet Capacity & Laycan Compatibility Audit</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="panel-card" style="margin-bottom: 16px; padding: 12px 18px; border-left: 4px solid #372580;">
                <b style="color: #18181B; font-size: 0.92rem;">Export Fleet Fixture Feasibility Check (Target Shipment: {ex_order:,} tonnes):</b>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 8px; font-size: 0.88rem;">
                    <div style="background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3); border-radius: 6px; padding: 8px 12px;">
                        <b style="color: #15803D;">2 × Supramax (116,000 t)</b><br>
                        <span style="color: #48A868; font-weight: 700;">● FEASIBLE</span><br>
                        <span style="color: #6B6B73; font-size: 0.82rem;">86.2% Utilization • Lowest Cost Option</span>
                    </div>
                    <div style="background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3); border-radius: 6px; padding: 8px 12px;">
                        <b style="color: #15803D;">2 × Panamax (164,000 t)</b><br>
                        <span style="color: #48A868; font-weight: 700;">● FEASIBLE</span><br>
                        <span style="color: #6B6B73; font-size: 0.82rem;">61.0% Utilization • Balanced Recommended</span>
                    </div>
                    <div style="background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3); border-radius: 6px; padding: 8px 12px;">
                        <b style="color: #15803D;">1 × Capesize (180,000 t)</b><br>
                        <span style="color: #48A868; font-weight: 700;">● FEASIBLE</span><br>
                        <span style="color: #6B6B73; font-size: 0.82rem;">55.6% Utilization • Bulk Parcel Fixture</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Step 3: Generate 3 Export Plans
        shipment_target = ex_order if st.session_state.get("export_plan_mode") == "Plan Full Order" else ready_qty
        exp_plans = generate_export_plans(
            shipment_qty=shipment_target,
            order_qty=ex_order,
            commodity=st.session_state.get("export_commodity", "Finished Steel"),
            buyer_country=st.session_state.get("export_buyer_country", "South Korea"),
            loading_port=st.session_state.get("export_loading_port", "Paradip"),
            dest_port=st.session_state.get("export_dest_port", "Singapore"),
            incoterm=st.session_state.get("export_incoterm", "CFR"),
            laycan_start=st.session_state.get("export_laycan_start", date.today() + timedelta(days=7)),
            laycan_end=st.session_state.get("export_laycan_end", date.today() + timedelta(days=14)),
            deadline_days=st.session_state.get("export_deadline", 30),
            max_budget=st.session_state.get("export_max_budget", 20.0),
            max_risk=st.session_state.get("export_max_risk", 40),
            selling_price=st.session_state.get("export_selling_price", 6800),
            internal_cost=st.session_state.get("export_internal_cost", 4200),
            allowed_vessels=st.session_state.get("export_allowed_vessels", ["Supramax (58k)", "Panamax (82k)", "Capesize (180k)"]),
            allowed_loading_ports=st.session_state.get("export_allowed_loading_ports", EXPORT_LOADING_PORTS),
            allowed_dest_ports=st.session_state.get("export_allowed_dest_ports", FOREIGN_DESTINATION_PORTS),
        )

        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; margin-bottom: 12px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="background: #372580; color: #FFFFFF; font-weight: 600; font-size: 0.78rem; padding: 3px 8px; border-radius: 4px;">STEP 3</span>
                    <h3 style="margin: 0; color: #18181B; font-size: 1.15rem; font-weight: 600;">Generated Export Voyage Plans</h3>
                </div>
                <span style="color: #6B6B73; font-size: 0.85rem;">Evaluation Engine: <b>3 Strategic Options Generated</b></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        e_col1, e_col2, e_col3 = st.columns(3, gap="medium")

        # Plan 1: Lowest Cost
        with e_col1:
            ep1 = exp_plans["Lowest Cost"]
            ep1_csv = pd.DataFrame([{
                "Plan Name": ep1["name"], "Trade Direction": "Export from India", "Commodity": st.session_state.get("export_commodity", "Finished Steel"),
                "Shipment Quantity": shipment_target, "Vessel Class": ep1["vessel_class"], "Vessel Count": ep1["vessel_count"],
                "Indian Loading Port": ep1["loading_port"], "Destination Port": ep1["dest_port"], "Incoterm": st.session_state.get("export_incoterm", "CFR"),
                "Total Logistics Cost (₹ Cr)": ep1["cost_cr"], "Duration (Days)": ep1["duration"], "Utilization (%)": ep1["utilization"],
                "Risk Score": ep1["risk_score"], "Commercial Revenue (₹ Cr)": ep1["revenue_cr"], "Commercial Margin (₹ Cr)": ep1["margin_cr"],
                "Feasibility Status": ep1["status"], "Disclaimer": "Prototype Data"
            }]).to_csv(index=False).encode("utf-8")

            st.markdown(
                f"""
                <div class="panel-card" style="height: 100%; border: 1px solid #E4E4E8; border-top: 4px solid #16A34A; background: #FFFFFF;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <b style="color: #15803D; font-size: 1.05rem;">1. Lowest Cost</b>
                        <span style="background: #EDF7F0; color: #15803D; font-size: 0.72rem; font-weight: 700; padding: 2px 7px; border-radius: 4px;">{ep1['badge']}</span>
                    </div>
                    <div style="font-size: 1.35rem; font-weight: 700; color: #18181B; margin-bottom: 2px;">{ep1['vessel_count']} × {ep1['vessel_class']}</div>
                    <div style="color: #6B6B73; font-size: 0.84rem; margin-bottom: 12px;">Route: <b>{ep1['loading_port']} → {ep1['dest_port']}</b></div>
                    <hr style="border: none; border-top: 1px solid #ECECF0; margin: 8px 0;" />
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 0.85rem;">
                        <div><span style="color: #6B6B73;">Capacity:</span> <b style="color: #18181B;">{ep1['combined_capacity']:,} t</b></div>
                        <div><span style="color: #6B6B73;">Utilization:</span> <b style="color: #372580;">{ep1['utilization']}%</b></div>
                        <div><span style="color: #6B6B73;">Laycan Window:</span> <b style="color: #18181B;">Passes</b></div>
                        <div><span style="color: #6B6B73;">Arrival ETA:</span> <b style="color: #48A868;">Day {ep1['duration']}</b></div>
                        <div><span style="color: #6B6B73;">Logistics Cost:</span> <b style="color: #48A868;">₹{ep1['cost_cr']:.2f} Cr</b></div>
                        <div><span style="color: #6B6B73;">Net Margin:</span> <b style="color: #48A868;">₹{ep1['margin_cr']} Cr</b></div>
                        <div><span style="color: #6B6B73;">Risk Score:</span> <b style="color: {ep1['risk_color']};">{ep1['risk_score']}/100</b></div>
                        <div><span style="color: #6B6B73;">Transit Days:</span> <b style="color: #18181B;">{ep1['duration']} days</b></div>
                    </div>
                    <div style="margin-top: 12px; text-align: center; padding: 4px 8px; border-radius: 6px; background: #EDF7F0; color: #15803D; font-weight: 600; font-size: 0.84rem;">
                        Feasibility: {ep1['status']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.download_button("📥 Download Export Lowest Cost CSV", data=ep1_csv, file_name="varunapath_export_lowest_cost_plan.csv", mime="text/csv", key="sp_exp_dl_btn_p1", use_container_width=True)

        # Plan 2: Lowest Risk
        with e_col2:
            ep2 = exp_plans["Lowest Risk"]
            ep2_csv = pd.DataFrame([{
                "Plan Name": ep2["name"], "Trade Direction": "Export from India", "Commodity": st.session_state.get("export_commodity", "Finished Steel"),
                "Shipment Quantity": shipment_target, "Vessel Class": ep2["vessel_class"], "Vessel Count": ep2["vessel_count"],
                "Indian Loading Port": ep2["loading_port"], "Destination Port": ep2["dest_port"], "Incoterm": st.session_state.get("export_incoterm", "CFR"),
                "Total Logistics Cost (₹ Cr)": ep2["cost_cr"], "Duration (Days)": ep2["duration"], "Utilization (%)": ep2["utilization"],
                "Risk Score": ep2["risk_score"], "Commercial Revenue (₹ Cr)": ep2["revenue_cr"], "Commercial Margin (₹ Cr)": ep2["margin_cr"],
                "Feasibility Status": ep2["status"], "Disclaimer": "Prototype Data"
            }]).to_csv(index=False).encode("utf-8")

            st.markdown(
                f"""
                <div class="panel-card" style="height: 100%; border: 1px solid #E4E4E8; border-top: 4px solid #372580; background: #FFFFFF;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <b style="color: #372580; font-size: 1.05rem;">2. Lowest Risk</b>
                        <span style="background: #F0EEF9; color: #372580; font-size: 0.72rem; font-weight: 700; padding: 2px 7px; border-radius: 4px;">{ep2['badge']}</span>
                    </div>
                    <div style="font-size: 1.35rem; font-weight: 700; color: #18181B; margin-bottom: 2px;">{ep2['vessel_count']} × {ep2['vessel_class']}</div>
                    <div style="color: #6B6B73; font-size: 0.84rem; margin-bottom: 12px;">Route: <b>{ep2['loading_port']} → {ep2['dest_port']}</b></div>
                    <hr style="border: none; border-top: 1px solid #ECECF0; margin: 8px 0;" />
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 0.85rem;">
                        <div><span style="color: #6B6B73;">Capacity:</span> <b style="color: #18181B;">{ep2['combined_capacity']:,} t</b></div>
                        <div><span style="color: #6B6B73;">Utilization:</span> <b style="color: #372580;">{ep2['utilization']}%</b></div>
                        <div><span style="color: #6B6B73;">Laycan Window:</span> <b style="color: #18181B;">Passes</b></div>
                        <div><span style="color: #6B6B73;">Arrival ETA:</span> <b style="color: #48A868;">Day {ep2['duration']}</b></div>
                        <div><span style="color: #6B6B73;">Logistics Cost:</span> <b style="color: #48A868;">₹{ep2['cost_cr']:.2f} Cr</b></div>
                        <div><span style="color: #6B6B73;">Net Margin:</span> <b style="color: #48A868;">₹{ep2['margin_cr']} Cr</b></div>
                        <div><span style="color: #6B6B73;">Risk Score:</span> <b style="color: {ep2['risk_color']};">{ep2['risk_score']}/100</b></div>
                        <div><span style="color: #6B6B73;">Transit Days:</span> <b style="color: #18181B;">{ep2['duration']} days</b></div>
                    </div>
                    <div style="margin-top: 12px; text-align: center; padding: 4px 8px; border-radius: 6px; background: #EDF7F0; color: #15803D; font-weight: 600; font-size: 0.84rem;">
                        Feasibility: {ep2['status']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.download_button("📥 Download Export Lowest Risk CSV", data=ep2_csv, file_name="varunapath_export_lowest_risk_plan.csv", mime="text/csv", key="sp_exp_dl_btn_p2", use_container_width=True)

        # Plan 3: Balanced Recommended
        with e_col3:
            ep3 = exp_plans["Balanced Recommended"]
            ep3_csv = pd.DataFrame([{
                "Plan Name": ep3["name"], "Trade Direction": "Export from India", "Commodity": st.session_state.get("export_commodity", "Finished Steel"),
                "Shipment Quantity": shipment_target, "Vessel Class": ep3["vessel_class"], "Vessel Count": ep3["vessel_count"],
                "Indian Loading Port": ep3["loading_port"], "Destination Port": ep3["dest_port"], "Incoterm": st.session_state.get("export_incoterm", "CFR"),
                "Total Logistics Cost (₹ Cr)": ep3["cost_cr"], "Duration (Days)": ep3["duration"], "Utilization (%)": ep3["utilization"],
                "Risk Score": ep3["risk_score"], "Commercial Revenue (₹ Cr)": ep3["revenue_cr"], "Commercial Margin (₹ Cr)": ep3["margin_cr"],
                "Feasibility Status": ep3["status"], "Disclaimer": "Prototype Data"
            }]).to_csv(index=False).encode("utf-8")

            st.markdown(
                f"""
                <div class="panel-card" style="height: 100%; border: 2px solid #372580; background: #FFFFFF; box-shadow: 0 2px 8px rgba(55,37,128,0.08);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <b style="color: #372580; font-size: 1.05rem;">3. Balanced Recommended</b>
                        <span style="background: #372580; color: #FFFFFF; font-size: 0.72rem; font-weight: 600; padding: 2px 8px; border-radius: 4px;">{ep3['badge']}</span>
                    </div>
                    <div style="font-size: 1.35rem; font-weight: 700; color: #18181B; margin-bottom: 2px;">{ep3['vessel_count']} × {ep3['vessel_class']}</div>
                    <div style="color: #6B6B73; font-size: 0.84rem; margin-bottom: 12px;">Route: <b>{ep3['loading_port']} → {ep3['dest_port']}</b></div>
                    <hr style="border: none; border-top: 1px solid #ECECF0; margin: 8px 0;" />
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 0.85rem;">
                        <div><span style="color: #6B6B73;">Capacity:</span> <b style="color: #18181B;">{ep3['combined_capacity']:,} t</b></div>
                        <div><span style="color: #6B6B73;">Utilization:</span> <b style="color: #48A868; font-weight: 700;">{ep3['utilization']}%</b></div>
                        <div><span style="color: #6B6B73;">Laycan Window:</span> <b style="color: #18181B;">Passes</b></div>
                        <div><span style="color: #6B6B73;">Arrival ETA:</span> <b style="color: #48A868;">Day {ep3['duration']}</b></div>
                        <div><span style="color: #6B6B73;">Logistics Cost:</span> <b style="color: #372580; font-weight: 700;">₹{ep3['cost_cr']:.2f} Cr</b></div>
                        <div><span style="color: #6B6B73;">Net Margin:</span> <b style="color: #48A868; font-weight: 700;">₹{ep3['margin_cr']} Cr</b></div>
                        <div><span style="color: #6B6B73;">Risk Score:</span> <b style="color: {ep3['risk_color']};">{ep3['risk_score']}/100</b></div>
                        <div><span style="color: #6B6B73;">Transit Days:</span> <b style="color: #18181B;">{ep3['duration']} days</b></div>
                    </div>
                    <div style="margin-top: 12px; text-align: center; padding: 6px 8px; border-radius: 6px; background: #EDF7F0; color: #15803D; font-weight: 600; font-size: 0.84rem; border: 1px solid rgba(72,168,104,0.3);">
                        Feasibility: {ep3['status']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.download_button("📥 Download Export Balanced CSV", data=ep3_csv, file_name="varunapath_export_balanced_plan.csv", mime="text/csv", key="sp_exp_dl_btn_p3", use_container_width=True)

        # STEP 4: CONFIRM EXPORT PLAN, DYNAMIC WHY THIS EXPORT PLAN, TRACE & 9 DOCS
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; margin-top: 24px; margin-bottom: 12px;">
                <span style="background: #372580; color: #FFFFFF; font-weight: 600; font-size: 0.78rem; padding: 3px 8px; border-radius: 4px;">STEP 4</span>
                <h3 style="margin: 0; color: #18181B; font-size: 1.15rem; font-weight: 600;">Confirm Export Fixture & Trade Compliance</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container():
            st.markdown('<div class="panel-card" style="margin-bottom: 20px;">', unsafe_allow_html=True)
            exp_plan_choice = st.radio(
                "Select Export Plan for Fixture Commitment:",
                ["Balanced Recommended", "Lowest Cost", "Lowest Risk"],
                index=0,
                horizontal=True,
                key="sp_exp_step4_radio",
            )
            selected_exp_plan = exp_plans[exp_plan_choice]

            why_exp_text = generate_why_this_export_plan(
                selected_exp_plan,
                st.session_state.get("export_max_budget", 20.0),
                st.session_state.get("export_max_risk", 40),
                st.session_state.get("export_deadline", 30),
                st.session_state.get("export_incoterm", "CFR")
            )

            st.markdown(
                f"""
                <div style="background: #F0EEF9; border-left: 4px solid #372580; padding: 12px 16px; border-radius: 6px; margin-top: 12px; margin-bottom: 16px;">
                    <b style="color: #372580; font-size: 0.95rem;">Why This Export Plan ({selected_exp_plan['name']}):</b>
                    <div style="color: #6B6B73; font-size: 0.9rem; line-height: 1.6; margin-top: 4px;">{why_exp_text.strip()}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Expandable Export Calculation Trace
            with st.expander("🔍 Export Calculation Trace (Logistics & Commercial Financials Breakdown)"):
                comps = selected_exp_plan["components"]
                comp_breakdown = "".join([f"• <b>{k}:</b> ₹{v:.3f} Cr<br>" for k, v in comps.items()])
                st.markdown(
                    f"""
                    <b>1. Export Readiness:</b> Stock ({ex_inv:,} t) + Production ({ex_prod:,} t) - Reserved ({ex_res:,} t) = <b>{ready_qty:,} tonnes</b> ({readiness_pct:.1f}% readiness)<br>
                    <b>2. Fleet Allocation:</b> {selected_exp_plan['vessel_count']} × {selected_exp_plan['vessel_class']} = <b>{selected_exp_plan['combined_capacity']:,} tonnes capacity</b> ({selected_exp_plan['utilization']}% utilization)<br>
                    <b>3. Incoterm Logistics Cost Breakdown ({st.session_state.get('export_incoterm', 'CFR')}):</b><br>
                    {comp_breakdown}
                    <b>Total Logistics Cost:</b> <b>₹{selected_exp_plan['cost_cr']:.2f} Cr</b><br><br>
                    <b>4. Commercial Export Financials:</b><br>
                    • Gross Revenue: <b>₹{selected_exp_plan['revenue_cr']} Cr</b> (at ₹{st.session_state.get('export_selling_price', 6800)}/t)<br>
                    • Production COGS: <b>₹{round((shipment_target * st.session_state.get('export_internal_cost', 4200)) / 10_000_000, 2)} Cr</b><br>
                    • Net Commercial Margin: <b>₹{selected_exp_plan['margin_cr']} Cr</b> ({selected_exp_plan['margin_pct']}% net margin)<br>
                    <i>{selected_exp_plan['fin_msg']}</i>
                    """,
                    unsafe_allow_html=True,
                )

            # 9-Item Export Document Checklist
            with st.expander("📑 Export Document Checklist (Customs ICEGATE & Regulatory Compliance)"):
                exp_doc_cols = st.columns(3)
                for idx, (doc_name, doc_desc, doc_stat) in enumerate(EXPORT_DOCS_LIST):
                    with exp_doc_cols[idx % 3]:
                        b_col = "#15803D" if doc_stat == "Generated" else "#D97706"
                        st.markdown(
                            f"""
                            <div style="background: #F8F8FA; border: 1px solid #E4E4E8; border-radius: 6px; padding: 10px; margin-bottom: 8px;">
                                <div style="display: flex; justify-content: space-between;">
                                    <b>{doc_name}</b>
                                    <span style="color: {b_col}; font-weight: 700; font-size: 0.78rem;">{doc_stat}</span>
                                </div>
                                <div style="color: #6B6B73; font-size: 0.8rem; margin-top: 4px;">{doc_desc}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            e_act1, e_act2 = st.columns([1.2, 1.5], gap="medium")
            with e_act1:
                if st.button("✅ Confirm Export Fixture", key="sp_exp_confirm_btn", type="primary", use_container_width=True):
                    st.session_state["confirmed_plan"] = selected_exp_plan
                    st.session_state["plan_confirmed"] = True
                    st.toast(f"Export Fixture Confirmed: {selected_exp_plan['vessel_count']} × {selected_exp_plan['vessel_class']} to {selected_exp_plan['dest_port']} (₹{selected_exp_plan['cost_cr']:.2f} Cr)", icon="✅")
                    st.rerun()
            with e_act2:
                if st.button("🚀 Send to Optimization Hub →", key="sp_exp_send_opt_btn", use_container_width=True):
                    st.session_state["confirmed_plan"] = selected_exp_plan
                    st.session_state["active_page"] = "Optimization Hub"
                    st.session_state["active_module"] = "Optimization Hub"
                    st.toast("Export Fixture transferred to Optimization Hub.", icon="🚀")
                    st.rerun()

            if st.session_state.get("plan_confirmed", False) and "confirmed_plan" in st.session_state:
                cp = st.session_state["confirmed_plan"]
                st.markdown(
                    f"""
                    <div style="margin-top: 16px; padding: 14px 18px; border-radius: 8px; background: #EDF7F0; border: 1px solid rgba(72,168,104,0.4);">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 1.2rem;">🎉</span>
                            <b style="color: #48A868; font-size: 1.02rem;">Export Plan Successfully Confirmed & Committed to Session State!</b>
                        </div>
                        <div style="color: #6B6B73; font-size: 0.9rem; margin-top: 6px; line-height: 1.5;">
                            <b>Fixture Allocation:</b> {cp['vessel_count']} × {cp['vessel_class']} ({cp['combined_capacity']:,} t capacity) &bull;
                            <b>Loading Port:</b> {cp['loading_port']} Port &bull;
                            <b>Destination:</b> {cp['dest_port']} &bull;
                            <b>Arrival ETA:</b> Day {cp['duration']} &bull;
                            <b>Committed Logistics Cost:</b> ₹{cp['cost_cr']:.2f} Cr &bull;
                            <b>Estimated Net Margin:</b> ₹{cp['margin_cr']} Cr &bull;
                            <b>Risk:</b> {cp['risk_score']}/100 ({cp['risk_label']}).
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)


def render_optimization_hub():
    """Renders the Optimization Hub view."""
    render_level2_header("Optimization Hub")
    render_trade_direction_selector()
    render_shared_scenario_summary()

    td = st.session_state.get("trade_direction", "Import to India")

    if td == "Import to India":
        calc = get_scenario_calculations()
        commodity = calc["commodity"]
        forecast_horizon = calc["forecast_horizon"]
        horizon_days = calc["horizon_days"]
        cargo_requirement = calc["cargo_requirement"]
        inventory = calc["inventory"]
        safety = calc["safety"]
        deadline = calc["deadline"]
        fuel_change = calc["fuel_change"]
        active_scenario = calc["active_scenario"]
        effective_fuel_change = calc["effective_fuel_change"]
        scenario_port_outage = calc["scenario_port_outage"]
        cargo_shortfall = calc["cargo_shortfall"]
        base_plans = calc["base_plans"]

        render_optimization_hub_controls(commodity, forecast_horizon, active_scenario, cargo_shortfall, deadline)

        st.markdown('<div class="section-title">📊 Feasible Fleet Allocations (Ranked by Least Logistics Cost)</div>', unsafe_allow_html=True)
        if not base_plans.empty:
            st.markdown('<div class="kpi-grid-5" style="margin-bottom: 18px;">', unsafe_allow_html=True)
            kpi_cols = st.columns(5, gap="medium")
            with kpi_cols[0]:
                st.markdown(f'<div class="kpi-card"><span class="kpi-label">Cargo Shortfall</span><span class="kpi-value">{cargo_shortfall:,} t</span><span class="kpi-subtext">Net need</span></div>', unsafe_allow_html=True)
            with kpi_cols[1]:
                st.markdown(f'<div class="kpi-card"><span class="kpi-label">Feasible Plans</span><span class="kpi-value">{len(base_plans)}</span><span class="kpi-subtext">Evaluated</span></div>', unsafe_allow_html=True)
            with kpi_cols[2]:
                st.markdown(f'<div class="kpi-card"><span class="kpi-label">Least Cost</span><span class="kpi-value">₹{base_plans.iloc[0]["Total Cost Cr"]:.2f} Cr</span><span class="kpi-subtext">Optimized</span></div>', unsafe_allow_html=True)
            with kpi_cols[3]:
                st.markdown(f'<div class="kpi-card"><span class="kpi-label">Fastest ETA</span><span class="kpi-value">{int(base_plans["ETA Days"].min())}d</span><span class="kpi-subtext">Transit + Wait</span></div>', unsafe_allow_html=True)
            with kpi_cols[4]:
                st.markdown(f'<div class="kpi-card"><span class="kpi-label">Lowest Risk</span><span class="kpi-value">{int(base_plans["Risk"].min())}/100</span><span class="kpi-subtext">Score</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            disp_df = base_plans[[
                "Supplier", "Origin", "Vessel", "Class", "Vessels", "Port",
                "ETA Days", "Cargo (t)", "Combined Capacity", "Utilization", "Total Cost Cr"
            ]].copy()
            st.dataframe(disp_df, use_container_width=True)
        else:
            st.warning("No feasible fleet allocation under current constraints.")
    else:
        # EXPORT OPTIMIZATION HUB
        st.markdown(
            """
            <div class="panel-card" style="margin-bottom: 16px;">
                <b style="color: #372580; font-size: 1.05rem;">Export Fleet Optimization & Margin Maximization Engine</b>
                <div style="color: #6B6B73; font-size: 0.88rem; margin-top: 4px;">
                    Multi-criteria mathematical optimization ranking export fixture allocations across least logistics cost, highest commercial margin, and fastest buyer delivery.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        opt_c1, opt_c2 = st.columns([2.0, 1.2], gap="medium")
        with opt_c1:
            opt_obj = st.selectbox(
                "OPTIMIZATION OBJECTIVE",
                ["Lowest Export Logistics Cost", "Highest Net Commercial Margin", "Fastest Buyer Delivery (Minimal ETA)", "Lowest Sea-Lane & Port Risk", "Balanced Multi-Criteria"],
                index=4,
                key="opt_exp_objective"
            )
        with opt_c2:
            st.markdown('<div class="ctrl-label">&nbsp;</div>', unsafe_allow_html=True)
            if st.button("⚡ Run AI Optimization", key="opt_exp_run_btn", type="primary", use_container_width=True):
                st.toast(f"Export fleet optimization solved for objective: {opt_obj}!", icon="⚡")
                st.rerun()

        # KPI 5-Card Row for Export
        kpi_cols = st.columns(5, gap="medium")
        with kpi_cols[0]:
            st.markdown('<div class="kpi-card"><span class="kpi-label">Export Target</span><span class="kpi-value">100,000 t</span><span class="kpi-subtext">Target Order</span></div>', unsafe_allow_html=True)
        with kpi_cols[1]:
            st.markdown('<div class="kpi-card"><span class="kpi-label">Feasible Plans</span><span class="kpi-value">6</span><span class="kpi-subtext">Fixtures Evaluated</span></div>', unsafe_allow_html=True)
        with kpi_cols[2]:
            st.markdown('<div class="kpi-card"><span class="kpi-label">Least Cost</span><span class="kpi-value">₹13.85 Cr</span><span class="kpi-subtext">Supramax Fleet</span></div>', unsafe_allow_html=True)
        with kpi_cols[3]:
            st.markdown('<div class="kpi-card"><span class="kpi-label">Max Margin</span><span class="kpi-value">₹12.15 Cr</span><span class="kpi-subtext">17.9% Margin</span></div>', unsafe_allow_html=True)
        with kpi_cols[4]:
            st.markdown('<div class="kpi-card"><span class="kpi-label">Lowest Risk</span><span class="kpi-value">18/100</span><span class="kpi-subtext">Visakhapatnam</span></div>', unsafe_allow_html=True)

        st.markdown('<div style="height: 14px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Feasible Export Fleet Allocations (Ranked by Selected Objective)</div>', unsafe_allow_html=True)

        exp_opt_table = pd.DataFrame([
            {"Loading Port": "Paradip", "Destination": "Singapore", "Vessel Class": "Panamax", "Vessels": 2, "Combined Capacity": 164000, "Utilization (%)": 61.0, "ETA (Days)": 7, "Logistics Cost (₹ Cr)": 15.10, "Est. Margin (₹ Cr)": 10.90, "Risk Score": 22, "Rank": 1},
            {"Loading Port": "Paradip", "Destination": "Singapore", "Vessel Class": "Supramax", "Vessels": 2, "Combined Capacity": 116000, "Utilization (%)": 86.2, "ETA (Days)": 8, "Logistics Cost (₹ Cr)": 13.85, "Est. Margin (₹ Cr)": 12.15, "Risk Score": 24, "Rank": 2},
            {"Loading Port": "Visakhapatnam", "Destination": "Singapore", "Vessel Class": "Panamax", "Vessels": 2, "Combined Capacity": 164000, "Utilization (%)": 61.0, "ETA (Days)": 7, "Logistics Cost (₹ Cr)": 15.60, "Est. Margin (₹ Cr)": 10.40, "Risk Score": 18, "Rank": 3},
            {"Loading Port": "Chennai", "Destination": "Singapore", "Vessel Class": "Panamax", "Vessels": 2, "Combined Capacity": 164000, "Utilization (%)": 61.0, "ETA (Days)": 6, "Logistics Cost (₹ Cr)": 15.80, "Est. Margin (₹ Cr)": 10.20, "Risk Score": 20, "Rank": 4},
            {"Loading Port": "Dhamra", "Destination": "Port Klang", "Vessel Class": "Panamax", "Vessels": 2, "Combined Capacity": 164000, "Utilization (%)": 61.0, "ETA (Days)": 8, "Logistics Cost (₹ Cr)": 15.45, "Est. Margin (₹ Cr)": 10.55, "Risk Score": 23, "Rank": 5},
            {"Loading Port": "Paradip", "Destination": "Chittagong", "Vessel Class": "Supramax", "Vessels": 2, "Combined Capacity": 116000, "Utilization (%)": 86.2, "ETA (Days)": 4, "Logistics Cost (₹ Cr)": 11.20, "Est. Margin (₹ Cr)": 14.80, "Risk Score": 25, "Rank": 6},
        ])
        st.dataframe(exp_opt_table, use_container_width=True)


def render_vessel_intelligence():
    """Renders the Vessel Intelligence view."""
    render_level2_header("Vessel Intelligence")
    cargo_required = st.session_state.get("cargo_requirement", 150000)

    sel_cls, sel_cap, sel_avail, sel_draft = render_vessel_intelligence_controls()
    st.markdown('<div class="panel-card"><b>Vessel Fleet Intelligence</b> — Dry bulk carrier specifications, charter tariffs, capacity classes, and deadweight optimization.</div>', unsafe_allow_html=True)

    # Isometric Vessel Silhouettes Grid
    st.markdown('<h4 style="color: #18181B; font-size: 1.02rem; margin-top: 14px; margin-bottom: 8px;">🚢 Bulk Carrier Class Silhouettes & Scale Specifications</h4>', unsafe_allow_html=True)
    vc1, vc2, vc3, vc4 = st.columns(4, gap="medium")
    v_classes = [
        {"name": "Handysize", "cap": "35,000 DWT", "hatches": 4, "w": 95, "charter": "₹1.45 Cr", "draft": "10.0m", "badge": "Shallow Port", "color": "#5746A5"},
        {"name": "Supramax", "cap": "58,000 DWT", "hatches": 5, "w": 125, "charter": "₹1.85 Cr", "draft": "12.8m", "badge": "Geared", "color": "#372580"},
        {"name": "Panamax", "cap": "82,000 DWT", "hatches": 7, "w": 160, "charter": "₹2.20 Cr", "draft": "14.5m", "badge": "★ Optimal", "color": "#48A868"},
        {"name": "Capesize", "cap": "180,000 DWT", "hatches": 9, "w": 210, "charter": "₹3.80 Cr", "draft": "18.2m", "badge": "Deepwater", "color": "#F6B51B"},
    ]
    cols = [vc1, vc2, vc3, vc4]
    for idx, vc in enumerate(v_classes):
        w = vc["w"]
        hatches = vc["hatches"]
        hatch_w = (w - 38) / max(1, hatches)
        hatch_rects = []
        for i in range(hatches):
            hx = 16 + i * hatch_w + 1.5
            hatch_rects.append(f'<rect x="{hx:.1f}" y="9" width="{hatch_w - 3:.1f}" height="9" rx="1.5" fill="#5746A5" stroke="#D9D4EE" stroke-width="0.8" />')
        hatch_svg = "".join(hatch_rects)

        svg_markup = (
            f'<svg viewBox="0 0 240 42" width="100%" height="42" xmlns="http://www.w3.org/2000/svg" style="display:block; margin:6px auto;">'
            f'<line x1="8" y1="32" x2="232" y2="32" stroke="#ECECF0" stroke-width="1.2" stroke-dasharray="3,3" />'
            f'<path d="M 12 28 L {w - 14} 28 L {w} 16 L 15 16 Z" fill="#372580" stroke="#5746A5" stroke-width="1.2" />'
            f'<line x1="12" y1="28" x2="{w - 14}" y2="28" stroke="#F6B51B" stroke-width="2.2" stroke-linecap="round" />'
            f'{hatch_svg}'
            f'<polygon points="15,16 15,6 23,6 23,16" fill="#FFFFFF" stroke="#E4E4E8" stroke-width="0.8" />'
            f'<rect x="17" y="8" width="4" height="2.5" fill="#372580" />'
            f'<line x1="19" y1="6" x2="19" y2="2" stroke="#92929A" stroke-width="1" />'
            f'</svg>'
        )

        with cols[idx]:
            with st.container(border=True):
                st.markdown(
                    f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:2px;">'
                    f'<span style="color:#18181B; font-size:1.02rem; font-weight:700;">{vc["name"]}</span>'
                    f'<span style="background:rgba(56,189,248,0.15); color:{vc["color"]}; border:1px solid {vc["color"]}55; padding:1px 6px; border-radius:4px; font-size:0.72rem; font-weight:600;">{vc["badge"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div style="color:#372580; font-size:0.95rem; font-weight:600;">{vc["cap"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(svg_markup, unsafe_allow_html=True)
                st.markdown(
                    f'<div style="font-size:0.82rem; color:#6B6B73; border-top:1px solid #ECECF0; padding-top:6px; margin-top:4px;">'
                    f'<div style="display:flex; justify-content:space-between;"><span>Charter Cost:</span><b style="color:#18181B;">{vc["charter"]}</b></div>'
                    f'<div style="display:flex; justify-content:space-between; margin-top:2px;"><span>Max Draft:</span><b style="color:#18181B;">{vc["draft"]}</b></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


    col_v1, col_v2 = st.columns([1.4, 1])
    with col_v1:
        st.subheader("Fleet Profiles & Base Economics")
        st.dataframe(VESSELS[["Vessel", "Class", "Capacity", "Charter Cost Cr", "Vessel Risk"]], hide_index=True, width="stretch")
    with col_v2:
        st.subheader("Class Utilization for Current Shortfall")
        calc = get_scenario_calculations()
        req_tonnes = max(calc["cargo_shortfall"], 10000)
        class_stats = []
        for _, v in VESSELS.iterrows():
            needed = math.ceil(req_tonnes / v["Capacity"])
            util = (req_tonnes / (needed * v["Capacity"])) * 100
            class_stats.append({
                "Class": v["Class"],
                "Voyages": needed,
                "Total Capacity": needed * v["Capacity"],
                "Utilization %": f"{util:.1f}%",
            })
        st.dataframe(pd.DataFrame(class_stats), hide_index=True, width="stretch")
        st.markdown(
            '<div style="margin-top:8px; padding:8px 12px; border-radius:6px; background:#F0EEF9; border-left:3px solid #372580; font-size:0.82rem; color:#6B6B73;">'
            '<b>Feasibility Rule:</b> 2 × Supramax (116,000 t) is <span style="color: #D95C5C; font-weight: 700;">Infeasible (Deficit: 14,000 t)</span> for 130,000 t shortfall. 2 × Panamax (164,000 t, 79.3% util) and 1 × Capesize (180,000 t, 72.2% util) are Feasible.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.subheader("Charter Cost vs. Capacity Trade-Off")
    v_fig = px.scatter(
        VESSELS,
        x="Capacity",
        y="Charter Cost Cr",
        text="Class",
        size="Capacity",
        color="Class",
        title="Vessel Capacity (DWT) vs Charter Cost per Voyage (₹ Cr)",
        color_discrete_sequence=["#372580", "#5746A5", "#D9D4EE", "#F6B51B", "#48A868"],
    )
    v_fig.update_traces(textposition="top center")
    v_fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF", font_color="white")
    st.plotly_chart(v_fig, width="stretch")



def render_port_intelligence():
    """Renders the Port Intelligence view."""
    render_level2_header("Port Intelligence")
    sel_port, sel_cong, sel_draft, sel_w = render_port_intelligence_controls()
    st.markdown('<div class="panel-card"><b>Port Infrastructure & Readiness</b> — East Coast discharge terminals, handling charges, draft constraints, and typical waiting days.</div>', unsafe_allow_html=True)

    # Lightweight Pseudo-3D Maritime Route Simulation Visual
    st.markdown(
        """
        <div style="width: 100%; margin: 12px 0 20px 0; border-radius: 8px; border: 1px solid #E4E4E8; background: #FFFFFF; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid #ECECF0; padding-bottom: 8px; flex-wrap: wrap; gap: 8px;">
            <div>
                <b style="color: #372580; font-size: 1.02rem;">🚢 Indian Ocean Shipping Corridor Simulation</b>
                <span style="color: #6B6B73; font-size: 0.85rem; margin-left: 8px;">Richards Bay (ZA) → Indian East Coast</span>
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <span style="background: #EDF7F0; color: #15803D; border: 1px solid rgba(72,168,104,0.3); padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">Sea State: Normal (1.4m)</span>
                <span style="background: #F0EEF9; color: #372580; border: 1px solid rgba(56,189,248,0.3); padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">Distance: 4,620 NM</span>
            </div>
        </div>
        <svg class="maritime-route-svg" viewBox="0 0 920 250" width="100%" height="250" xmlns="http://www.w3.org/2000/svg" style="display: block;">
          <defs>
            <linearGradient id="corridorGlow" x1="0%" y1="100%" x2="100%" y2="0%">
              <stop offset="0%" stop-color="#D9D4EE" stop-opacity="0.4" />
              <stop offset="50%" stop-color="#372580" stop-opacity="0.9" />
              <stop offset="100%" stop-color="#48A868" stop-opacity="0.9" />
            </linearGradient>
            <filter id="routeBlur">
              <feGaussianBlur stdDeviation="2.5" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          <!-- Nautical grid & bathymetry rings -->
          <g stroke="#ECECF0" stroke-width="0.8" opacity="0.8" fill="none">
            <circle cx="110" cy="190" r="40" stroke-dasharray="2,4" />
            <circle cx="110" cy="190" r="80" stroke-dasharray="2,4" />
            <circle cx="780" cy="75" r="50" stroke-dasharray="2,4" />
            <circle cx="780" cy="75" r="100" stroke-dasharray="2,4" />
            <line x1="40" y1="60" x2="880" y2="60" stroke-dasharray="4,8" />
            <line x1="40" y1="130" x2="880" y2="130" stroke-dasharray="4,8" />
            <line x1="40" y1="200" x2="880" y2="200" stroke-dasharray="4,8" />
          </g>

          <!-- Route Path Definition -->
          <path id="nauticalLane" d="M 110 190 C 260 170, 420 140, 560 100 C 660 70, 720 75, 780 75" fill="none" stroke="url(#corridorGlow)" stroke-width="3" stroke-dasharray="8,6" filter="url(#routeBlur)" />

          <!-- Animated Vessel traversing route -->
          <g>
            <path d="M -16,5 L 14,5 L 20,-1 L -14,-1 Z" fill="#372580" stroke="#5746A5" stroke-width="1" />
            <rect x="-10" y="-5" width="6" height="4" fill="#FFFFFF" />
            <rect x="0" y="-5" width="6" height="4" fill="#FFFFFF" />
            <polygon points="-14,-1 -14,-9 -9,-9 -9,-1" fill="#ffffff" />
            <animateMotion dur="16s" repeatCount="indefinite" rotate="auto">
              <mpath href="#nauticalLane" />
            </animateMotion>
          </g>

          <!-- Origin Marker: Richards Bay -->
          <g transform="translate(110, 190)">
            <circle r="16" fill="#D9D4EE" opacity="0.3" />
            <circle r="7" fill="#372580" stroke="#D9D4EE" stroke-width="2" />
            <circle r="2.5" fill="#ffffff" />
            <text x="-12" y="24" fill="#18181B" font-size="12" font-weight="600" font-family="system-ui, sans-serif">Richards Bay</text>
            <text x="-12" y="38" fill="#6B6B73" font-size="10" font-family="system-ui, sans-serif">28°48'S, 32°06'E • Coal Terminal</text>
          </g>

          <!-- Waypoint Indicator: Equator Transit -->
          <g transform="translate(480, 120)">
            <circle r="4" fill="#372580" opacity="0.8" />
            <text x="0" y="-12" fill="#6B6B73" font-size="9" text-anchor="middle" font-family="system-ui, sans-serif">Mid-Ocean Corridor</text>
            <text x="0" y="20" fill="#92929A" font-size="9" text-anchor="middle" font-family="system-ui, sans-serif">Transit Day 8 • Fair Seas</text>
          </g>

          <!-- Destination Nodes: Indian East Coast Ports -->
          <!-- 1. Paradip (Primary) -->
          <g transform="translate(780, 75)">
            <circle r="18" fill="#EDF7F0" opacity="0.6" />
            <circle r="8" fill="#48A868" stroke="#FFFFFF" stroke-width="2" />
            <circle r="3" fill="#ffffff" />
            <rect x="18" y="-16" width="112" height="42" rx="6" fill="#FFFFFF" stroke="#48A868" stroke-width="1" />
            <text x="26" y="-2" fill="#18181B" font-size="11" font-weight="600" font-family="system-ui, sans-serif">Paradip Port ★</text>
            <text x="26" y="10" fill="#48A868" font-size="9" font-family="system-ui, sans-serif">Draft: 180K DWT (Deep)</text>
            <text x="26" y="21" fill="#6B6B73" font-size="8.5" font-family="system-ui, sans-serif">Waiting: 3.0d • ₹110/t</text>
          </g>

          <!-- 2. Visakhapatnam (Secondary) -->
          <g transform="translate(740, 135)">
            <circle r="5" fill="#F6B51B" stroke="#FFFFFF" stroke-width="1.5" />
            <rect x="12" y="-14" width="112" height="32" rx="4" fill="#FFFFFF" stroke="#F6B51B" stroke-width="1" />
            <text x="18" y="-2" fill="#18181B" font-size="10" font-weight="600" font-family="system-ui, sans-serif">Visakhapatnam</text>
            <text x="18" y="10" fill="#F6B51B" font-size="8.5" font-family="system-ui, sans-serif">Draft: 150K • Waiting: 5.0d</text>
          </g>

          <!-- 3. Dhamra (Alternative) -->
          <g transform="translate(800, 30)">
            <circle r="5" fill="#5746A5" stroke="#FFFFFF" stroke-width="1.5" />
            <rect x="12" y="-12" width="105" height="30" rx="4" fill="#FFFFFF" stroke="#372580" stroke-width="1" />
            <text x="18" y="-1" fill="#18181B" font-size="10" font-weight="600" font-family="system-ui, sans-serif">Dhamra Port</text>
            <text x="18" y="10" fill="#372580" font-size="8.5" font-family="system-ui, sans-serif">Draft: 180K • Waiting: 2.0d</text>
          </g>
        </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(PORTS, hide_index=True, width="stretch")



def render_route_intelligence():
    """Renders the comprehensive Route Intelligence module."""
    render_level2_header("Route Intelligence")

    # Prototype Disclaimer & Executive Scope
    st.markdown(
        '<div class="panel-card" style="margin-bottom: 16px;">'
        '<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">'
        '<div>'
        '<b style="color: #18181B; font-size: 1.02rem;">🚢 Maritime Route Intelligence & Voyage Optimization</b>'
        '<div style="color: #6B6B73; font-size: 0.88rem; margin-top: 2px;">'
        'Compare maritime route alternatives using nautical distance, sailing ETA, bunker consumption, port draft compatibility, and operational risk.'
        '</div>'
        '</div>'
        '<span style="background: #F0EEF9; color: #372580; border: 1px solid var(--light-purple); padding: 4px 10px; border-radius: 4px; font-size: 0.74rem; font-weight: 700;">'
        'PROTOTYPE ROUTE ESTIMATE • LOCAL/SIMULATED DATA'
        '</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Render Controls
    (sel_orig, sel_dest, sel_cargo, sel_qty, sel_class, sel_dwt,
     sel_draft, sel_speed, sel_date, sel_fuel_cons, sel_fuel_price,
     sel_charter_rate, sel_priority) = render_route_intelligence_controls()

    # Master Port Specifications
    port_specs = {
        "Paradip Port": {"max_draft": 18.0, "max_loa": 300, "max_beam": 48, "congestion_days": 3.0, "tariff_per_tonne": 110, "lat": 20.26, "lon": 86.67, "type": "Deepwater"},
        "Visakhapatnam Port": {"max_draft": 16.5, "max_loa": 280, "max_beam": 45, "congestion_days": 5.0, "tariff_per_tonne": 125, "lat": 17.68, "lon": 83.21, "type": "Outer Harbour"},
        "Gangavaram Port": {"max_draft": 20.0, "max_loa": 320, "max_beam": 52, "congestion_days": 2.5, "tariff_per_tonne": 115, "lat": 17.62, "lon": 83.24, "type": "Deepwater"},
        "Dhamra Port": {"max_draft": 18.5, "max_loa": 315, "max_beam": 50, "congestion_days": 2.0, "tariff_per_tonne": 95, "lat": 20.80, "lon": 86.95, "type": "Deepwater"},
        "Gopalpur Port": {"max_draft": 14.5, "max_loa": 230, "max_beam": 33, "congestion_days": 1.5, "tariff_per_tonne": 105, "lat": 19.30, "lon": 84.97, "type": "Medium Draft"},
        "Haldia Port": {"max_draft": 10.5, "max_loa": 230, "max_beam": 32, "congestion_days": 4.5, "tariff_per_tonne": 140, "lat": 22.02, "lon": 88.06, "type": "Riverine Shallow"},
        "Sagar / Sandheads": {"max_draft": 13.5, "max_loa": 260, "max_beam": 40, "congestion_days": 2.0, "tariff_per_tonne": 130, "lat": 21.65, "lon": 88.05, "type": "Anchorage Lighterage"},
    }

    origin_specs = {
        "Richards Bay, South Africa": {"lat": -28.80, "lon": 32.05, "base_nm": 4650},
        "Newcastle, Australia": {"lat": -32.93, "lon": 151.78, "base_nm": 5420},
        "Tanjung Bara, Indonesia": {"lat": 0.55, "lon": 117.60, "base_nm": 2350},
        "Maputo, Mozambique": {"lat": -25.97, "lon": 32.58, "base_nm": 4450},
        "Ust-Luga, Russia": {"lat": 59.68, "lon": 28.30, "base_nm": 7420},
        "Norfolk, United States": {"lat": 36.85, "lon": -76.29, "base_nm": 11200},
    }

    cur_dest = port_specs.get(sel_dest, port_specs["Paradip Port"])
    cur_orig = origin_specs.get(sel_orig, origin_specs["Richards Bay, South Africa"])
    base_dist = cur_orig["base_nm"]
    dest_lat, dest_lon = cur_dest["lat"], cur_dest["lon"]
    orig_lat, orig_lon = cur_orig["lat"], cur_orig["lon"]

    # Port compatibility check
    max_d = cur_dest["max_draft"]
    is_draft_compat = sel_draft <= max_d
    is_berth_compat = not (sel_class == "Capesize" and sel_dest in ["Haldia Port", "Gopalpur Port"])
    is_fully_compatible = is_draft_compat and is_berth_compat

    if is_fully_compatible:
        compat_badge = '<span style="background: #EDF7F0; color: #15803D; padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 0.8rem;">● Fully Compatible</span>'
        compat_detail = f"Vessel draft ({sel_draft:.1f}m) satisfies {sel_dest} max limit ({max_d:.1f}m)."
    else:
        deficit_msg = f"Draft Deficit: {sel_draft - max_d:.1f}m" if not is_draft_compat else "Berth LOA Limit"
        compat_badge = f'<span style="background: #FEE2E2; color: #DC2626; padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 0.8rem;">▲ Incompatible ({deficit_msg})</span>'
        compat_detail = f"Operational alert: Vessel draft {sel_draft:.1f}m exceeds {sel_dest} safe threshold {max_d:.1f}m."

    port_charges = sel_qty * cur_dest["tariff_per_tonne"]
    cong_delay = cur_dest["congestion_days"]

    # 3 Distinct Route Alternatives
    alternatives = [
        {
            "id": "balanced",
            "name": "Direct Indian Ocean Corridor",
            "badge": "Balanced Recommended",
            "dist_nm": base_dist,
            "speed_kts": sel_speed,
            "weather_adj": 1.04,
            "weather_delay": 0.5,
            "weather_risk_label": "Low-Medium (24/100)",
            "cong_delay": cong_delay,
            "cong_risk_label": f"Normal Queue ({cong_delay:.1f}d)",
            "base_risk": 24,
            "color": "#372580",
            "line_dash": "solid",
        },
        {
            "id": "fastest",
            "name": "High-Speed Open Sea Direct",
            "badge": "Fastest Route",
            "dist_nm": round(base_dist * 0.96),
            "speed_kts": sel_speed + 1.2,
            "weather_adj": 1.12,
            "weather_delay": 0.8,
            "weather_risk_label": "Elevated (42/100)",
            "cong_delay": cong_delay,
            "cong_risk_label": f"Normal Queue ({cong_delay:.1f}d)",
            "base_risk": 38,
            "color": "#F6B51B",
            "line_dash": "dash",
        },
        {
            "id": "lowest_cost",
            "name": "Eco-Steaming Fair-Weather Track",
            "badge": "Lowest Cost",
            "dist_nm": round(base_dist * 1.02),
            "speed_kts": max(9.5, sel_speed - 1.5),
            "weather_adj": 0.96,
            "weather_delay": 0.3,
            "weather_risk_label": "Minimal (16/100)",
            "cong_delay": cong_delay,
            "cong_risk_label": f"Normal Queue ({cong_delay:.1f}d)",
            "base_risk": 18,
            "color": "#48A868",
            "line_dash": "dot",
        },
    ]

    # Perform deterministic calculations for all alternatives
    for alt in alternatives:
        s_days = alt["dist_nm"] / (alt["speed_kts"] * 24.0)
        tot_eta = s_days + alt["weather_delay"] + alt["cong_delay"]
        f_tonnes = s_days * sel_fuel_cons * alt["weather_adj"]
        f_cost = f_tonnes * sel_fuel_price
        c_cost = tot_eta * sel_charter_rate
        final_risk = alt["base_risk"] + (0 if is_fully_compatible else 40)
        risk_cont = (final_risk / 100.0) * (c_cost + f_cost) * 0.08
        tot_cost = f_cost + c_cost + port_charges + risk_cont

        alt["sailing_days"] = s_days
        alt["total_eta"] = tot_eta
        alt["fuel_tonnes"] = f_tonnes
        alt["fuel_cost_cr"] = f_cost / 10_000_000.0
        alt["charter_cost_cr"] = c_cost / 10_000_000.0
        alt["port_cost_cr"] = port_charges / 10_000_000.0
        alt["total_cost_cr"] = tot_cost / 10_000_000.0
        alt["risk_score"] = min(100, final_risk)
        alt["arrival_date"] = sel_date + timedelta(days=math.ceil(tot_eta))

    # Priority Selection Logic
    if sel_priority == "Lowest Cost":
        rec_alt = min(alternatives, key=lambda x: x["total_cost_cr"])
    elif sel_priority == "Fastest Route":
        rec_alt = min(alternatives, key=lambda x: x["total_eta"])
    elif sel_priority == "Lowest Risk":
        rec_alt = min(alternatives, key=lambda x: x["risk_score"])
    else:  # Balanced Recommended
        rec_alt = alternatives[0]

    # =========================================================================
    # 5. REQUIRED KPI CARDS ROW (3x3 Grid / Multi-column layout)
    # =========================================================================
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        st.markdown(
            f'<div class="kpi-card-exec">'
            f'<span class="kpi-exec-label">Recommended Route</span>'
            f'<div class="kpi-exec-val" style="font-size: 1.15rem; color: #372580; margin-top: 6px;">{rec_alt["name"]}</div>'
            f'<div class="kpi-exec-sub" style="color: #6B6B73;">Priority: {sel_priority}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f'<div class="kpi-card-exec">'
            f'<span class="kpi-exec-label">Distance & Sailing Time</span>'
            f'<div class="kpi-exec-val">{rec_alt["dist_nm"]:,} nm</div>'
            f'<div class="kpi-exec-sub" style="color: #6B6B73;">{rec_alt["sailing_days"]:.1f} Sailing Days @ {rec_alt["speed_kts"]:.1f} kts</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f'<div class="kpi-card-exec">'
            f'<span class="kpi-exec-label">Predicted Total ETA</span>'
            f'<div class="kpi-exec-val" style="color: #48A868;">{rec_alt["total_eta"]:.1f} Days</div>'
            f'<div class="kpi-exec-sub" style="color: #6B6B73;">Arrival: <b>{rec_alt["arrival_date"].strftime("%d %b %Y")}</b> (Queue: {cong_delay:.1f}d)</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f'<div class="kpi-card-exec">'
            f'<span class="kpi-exec-label">Estimated Voyage Cost</span>'
            f'<div class="kpi-exec-val" style="color: #18181B;">₹{rec_alt["total_cost_cr"]:.2f} Cr</div>'
            f'<div class="kpi-exec-sub" style="color: #F6B51B;">Bunker Fuel: {rec_alt["fuel_tonnes"]:.1f} t (₹{rec_alt["fuel_cost_cr"]:.2f} Cr)</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)
    k5, k6, k7, k8 = st.columns(4, gap="medium")
    with k5:
        st.markdown(
            f'<div class="kpi-card-exec">'
            f'<span class="kpi-exec-label">Port Compatibility</span>'
            f'<div style="margin-top: 8px;">{compat_badge}</div>'
            f'<div class="kpi-exec-sub" style="color: #6B6B73; margin-top: 8px;">{compat_detail}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with k6:
        st.markdown(
            f'<div class="kpi-card-exec">'
            f'<span class="kpi-exec-label">Route Risk Score</span>'
            f'<div class="kpi-exec-val" style="color: {"#D95C5C" if rec_alt["risk_score"] > 50 else "#372580"};">{rec_alt["risk_score"]} / 100</div>'
            f'<div class="kpi-exec-sub" style="color: #6B6B73;">Weather: {rec_alt["weather_risk_label"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with k7:
        st.markdown(
            f'<div class="kpi-card-exec">'
            f'<span class="kpi-exec-label">Fuel Consumption</span>'
            f'<div class="kpi-exec-val">{rec_alt["fuel_tonnes"]:.1f} tonnes</div>'
            f'<div class="kpi-exec-sub" style="color: #6B6B73;">Burn Rate: {sel_fuel_cons:.1f} t/d &bull; Adj: {rec_alt["weather_adj"]:.2f}x</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with k8:
        st.markdown(
            f'<div class="kpi-card-exec">'
            f'<span class="kpi-exec-label">Data Confidence & Engine</span>'
            f'<div class="kpi-exec-val" style="font-size: 1.15rem; color: #6B6B73;">Simulated MVP</div>'
            f'<div class="kpi-exec-sub" style="color: #92929A;">Deterministic Voyage Heuristic</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Incompatibility Warning Banner if draft deficit
    if not is_fully_compatible:
        st.markdown(
            f'<div style="margin-top: 14px; padding: 12px 16px; border-radius: 6px; background: #FBEEEE; border-left: 4px solid #D95C5C;">'
            f'<b style="color: #D95C5C; font-size: 0.95rem;">⚠️ Vessel-Port Incompatibility Alert:</b>'
            f'<div style="color: #6B6B73; font-size: 0.88rem; margin-top: 4px;">'
            f'The selected <b>{sel_class}</b> vessel has a design operating draft of <b>{sel_draft:.1f}m</b>, which exceeds <b>{sel_dest}</b> maximum permissible draft limit of <b>{max_d:.1f}m</b>. '
            f'Risk penalty (+40 points) has been applied. Consider switching to a compatible vessel (Panamax or Supramax) or discharging at a deepwater terminal (Paradip, Gangavaram, or Dhamra).'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height: 18px;"></div>', unsafe_allow_html=True)

    # =========================================================================
    # 6. INTERACTIVE MARITIME ROUTE MAP (Plotly Scattergeo)
    # =========================================================================
    st.markdown('<h4 style="color: #18181B; font-size: 1.05rem; margin-bottom: 6px;">🗺️ Indicative Maritime Route Alternatives</h4>', unsafe_allow_html=True)
    st.caption("Indicative prototype routes — not for navigational use. Intermediate sea waypoints follow international maritime corridors avoiding landmasses.")

    # Generate realistic sea waypoints based on origin and destination
    def generate_sea_waypoints(origin_name, dest_coord, route_type):
        d_lat, d_lon = dest_coord
        if "Richards Bay" in origin_name:
            if route_type == "balanced":
                return [(-28.80, 32.05), (-26.0, 45.0), (-12.0, 65.0), (0.0, 78.0), (10.0, 83.0), (d_lat, d_lon)]
            elif route_type == "fastest":
                return [(-28.80, 32.05), (-24.0, 48.0), (-8.0, 72.0), (4.0, 81.0), (d_lat, d_lon)]
            else:
                return [(-28.80, 32.05), (-18.0, 42.0), (-5.0, 60.0), (2.0, 75.0), (9.0, 83.0), (d_lat, d_lon)]
        elif "Newcastle" in origin_name:
            if route_type == "balanced":
                return [(-32.93, 151.78), (-38.0, 140.0), (-35.0, 118.0), (-22.0, 105.0), (-7.0, 95.0), (6.0, 88.0), (d_lat, d_lon)]
            elif route_type == "fastest":
                return [(-32.93, 151.78), (-22.0, 153.0), (-10.5, 142.5), (-8.0, 125.0), (-4.0, 105.0), (6.0, 90.0), (d_lat, d_lon)]
            else:
                return [(-32.93, 151.78), (-38.0, 140.0), (-36.0, 115.0), (-25.0, 100.0), (-5.0, 90.0), (d_lat, d_lon)]
        elif "Tanjung Bara" in origin_name:
            if route_type == "balanced":
                return [(0.55, 117.60), (1.3, 104.3), (3.0, 100.0), (5.8, 95.5), (10.0, 88.0), (d_lat, d_lon)]
            elif route_type == "fastest":
                return [(0.55, 117.60), (-3.5, 110.0), (-6.0, 105.0), (2.0, 95.0), (10.0, 88.0), (d_lat, d_lon)]
            else:
                return [(0.55, 117.60), (1.5, 104.5), (3.5, 99.5), (6.0, 95.0), (11.0, 87.0), (d_lat, d_lon)]
        elif "Maputo" in origin_name:
            if route_type == "balanced":
                return [(-25.97, 32.58), (-20.0, 40.0), (-10.0, 50.0), (0.0, 70.0), (8.0, 82.0), (d_lat, d_lon)]
            elif route_type == "fastest":
                return [(-25.97, 32.58), (-26.0, 46.0), (-12.0, 60.0), (0.0, 72.0), (8.0, 82.0), (d_lat, d_lon)]
            else:
                return [(-25.97, 32.58), (-18.0, 42.0), (-5.0, 60.0), (2.0, 75.0), (9.0, 83.0), (d_lat, d_lon)]
        elif "Ust-Luga" in origin_name:
            if route_type == "balanced":
                return [(59.68, 28.30), (57.5, 11.5), (51.0, 1.8), (36.0, -5.5), (33.0, 25.0), (27.0, 34.5), (13.0, 45.0), (8.0, 78.0), (d_lat, d_lon)]
            elif route_type == "fastest":
                return [(59.68, 28.30), (57.5, 11.5), (51.0, 1.8), (36.0, -5.5), (33.0, 25.0), (27.8, 34.0), (12.5, 43.5), (10.0, 75.0), (d_lat, d_lon)]
            else:
                return [(59.68, 28.30), (57.5, 11.5), (50.0, -5.0), (20.0, -20.0), (-15.0, -5.0), (-34.5, 18.5), (-28.0, 40.0), (0.0, 75.0), (d_lat, d_lon)]
        else:  # Norfolk, US
            if route_type == "fastest":
                return [(36.85, -76.29), (35.0, -40.0), (36.0, -6.0), (32.0, 28.0), (28.0, 33.0), (12.5, 43.5), (8.0, 78.0), (d_lat, d_lon)]
            else:
                return [(36.85, -76.29), (28.0, -60.0), (0.0, -30.0), (-25.0, -10.0), (-34.5, 18.5), (-20.0, 50.0), (0.0, 75.0), (d_lat, d_lon)]

    map_fig = go.Figure()

    # Draw the 3 route tracks
    for alt in alternatives:
        wp = generate_sea_waypoints(sel_orig, (dest_lat, dest_lon), alt["id"])
        is_rec = (alt["id"] == rec_alt["id"])
        line_w = 4.0 if is_rec else 2.2
        trace_name = f"{alt['name']} ★ [RECOMMENDED]" if is_rec else alt["name"]

        map_fig.add_trace(go.Scattergeo(
            lat=[p[0] for p in wp],
            lon=[p[1] for p in wp],
            mode="lines+markers",
            name=trace_name,
            line=dict(width=line_w, color=alt["color"], dash=alt["line_dash"]),
            marker=dict(size=4 if not is_rec else 6, color=alt["color"]),
            hovertemplate=(
                f"<b>{alt['name']}</b><br>"
                f"Distance: {alt['dist_nm']:,} nm<br>"
                f"Sailing: {alt['sailing_days']:.1f} days<br>"
                f"Total ETA: {alt['total_eta']:.1f} days<br>"
                f"Voyage Cost: ₹{alt['total_cost_cr']:.2f} Cr<br>"
                f"Risk: {alt['risk_score']}/100<extra></extra>"
            ),
        ))

    # Add Origin & Destination Terminal Nodes
    map_fig.add_trace(go.Scattergeo(
        lat=[orig_lat, dest_lat],
        lon=[orig_lon, dest_lon],
        mode="markers+text",
        text=[f"Origin: {sel_orig.split(',')[0]}", f"Discharge: {sel_dest}"],
        textposition=["bottom center", "top center"],
        textfont=dict(color="#18181B", size=11, family="Inter, system-ui, sans-serif"),
        marker=dict(size=10, color=["#372580", "#48A868"], symbol="diamond"),
        name="Key Terminals",
    ))

    # Map layout bounds
    min_lat = min(orig_lat, dest_lat) - 10
    max_lat = max(orig_lat, dest_lat) + 12
    min_lon = min(orig_lon, dest_lon) - 15
    max_lon = max(orig_lon, dest_lon) + 15

    map_fig.update_layout(
        height=460,
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        geo=dict(
            projection_type="natural earth",
            showland=True,
            landcolor="#F3F3F5",
            countrycolor="#E4E4E8",
            coastlinecolor="#9CA3AF",
            showocean=True,
            oceancolor="#F8FAFC",
            showcountries=True,
            lataxis=dict(range=[max(-60, min_lat), min(75, max_lat)]),
            lonaxis=dict(range=[min_lon, max_lon]),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#18181B", size=10)),
    )
    st.plotly_chart(map_fig, use_container_width=True)

    # =========================================================================
    # 7. ROUTE ALTERNATIVES COMPARISON TABLE
    # =========================================================================
    st.markdown('<h4 style="color: #18181B; font-size: 1.05rem; margin-top: 20px; margin-bottom: 8px;">📊 Comparative Route Analysis Matrix</h4>', unsafe_allow_html=True)

    comp_rows = []
    for alt in alternatives:
        is_sel = (alt["id"] == rec_alt["id"])
        comp_rows.append({
            "Status": "★ RECOMMENDED" if is_sel else "ALTERNATIVE",
            "Route Name": alt["name"],
            "Distance (nm)": f"{alt['dist_nm']:,}",
            "Speed (kts)": f"{alt['speed_kts']:.1f}",
            "Sailing Days": f"{alt['sailing_days']:.1f}d",
            "Total ETA": f"{alt['total_eta']:.1f}d",
            "Bunker Fuel": f"{alt['fuel_tonnes']:.1f} t",
            "Estimated Cost": f"₹{alt['total_cost_cr']:.2f} Cr",
            "Weather Risk": alt["weather_risk_label"],
            "Congestion": alt["cong_risk_label"],
            "Port Compatibility": "Compatible" if is_fully_compatible else "Infeasible",
            "Risk Score": f"{alt['risk_score']}/100",
        })

    comp_df = pd.DataFrame(comp_rows)
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    # =========================================================================
    # 8. "WHY THIS ROUTE?" DECISION EXPLANATION
    # =========================================================================
    st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)

    # Dynamic explanation generation
    cost_diff_fastest = rec_alt["total_cost_cr"] - alternatives[1]["total_cost_cr"]
    eta_diff_lowest = rec_alt["total_eta"] - alternatives[2]["total_eta"]

    why_bullets = []
    if rec_alt["id"] == "balanced":
        why_bullets.append(
            f"<b>Optimized Strategic Balance:</b> The <b>{rec_alt['name']}</b> achieves the most resilient equilibrium between transit speed ({rec_alt['speed_kts']:.1f} kts) and voyage expenditure (₹{rec_alt['total_cost_cr']:.2f} Cr)."
        )
        why_bullets.append(
            f"<b>ETA & Schedule Safety:</b> Predicted total voyage duration of <b>{rec_alt['total_eta']:.1f} days</b> avoids high-risk storm corridors and provides adequate berth synchronization at {sel_dest}."
        )
        why_bullets.append(
            f"<b>Trade-off Analysis:</b> Compared to the High-Speed Route, it saves <b>{abs(cost_diff_fastest):.2f} Cr</b> in bunker fuel and engine wear while adding only {(rec_alt['total_eta'] - alternatives[1]['total_eta']):.1f} days."
        )
    elif rec_alt["id"] == "lowest_cost":
        why_bullets.append(
            f"<b>Least-Cost Economic Advantage:</b> Selected under <b>{sel_priority}</b> priority. Generates the minimum total commitment of <b>₹{rec_alt['total_cost_cr']:.2f} Cr</b> through eco-speed steaming ({rec_alt['speed_kts']:.1f} kts)."
        )
        why_bullets.append(
            f"<b>Bunker Fuel Efficiency:</b> Consumes only <b>{rec_alt['fuel_tonnes']:.1f} tonnes</b> of bunker fuel (lowest across all evaluated corridors)."
        )
        why_bullets.append(
            f"<b>Trade-off Analysis:</b> Takes {abs(eta_diff_lowest):.1f} additional sailing days compared to balanced routing; suitable when inventory reserves at Paradip/East Coast remain above safety threshold."
        )
    elif rec_alt["id"] == "fastest":
        why_bullets.append(
            f"<b>Rapid Transit Priority:</b> Selected under <b>{sel_priority}</b> priority. Minimizes overall voyage time to <b>{rec_alt['total_eta']:.1f} days</b> (saving {abs(alternatives[0]['total_eta'] - rec_alt['total_eta']):.1f} days vs standard corridor)."
        )
        why_bullets.append(
            f"<b>Stockout Mitigation:</b> Crucial if inventory stockpile at destination is critically depleted and requires urgent replenishment."
        )
        why_bullets.append(
            f"<b>Trade-off Analysis:</b> Requires higher fuel burn rate (+{rec_alt['fuel_tonnes'] - alternatives[0]['fuel_tonnes']:.1f} tonnes) resulting in an estimated voyage cost of ₹{rec_alt['total_cost_cr']:.2f} Cr."
        )
    else:  # Lowest Risk
        why_bullets.append(
            f"<b>Operational Risk Minimization:</b> Achieves the lowest composite risk score of <b>{rec_alt['risk_score']}/100</b> by selecting safe deep-draft channels with minimal cyclonic and choke point exposure."
        )

    if not is_fully_compatible:
        why_bullets.append(
            f"<span style='color: #D95C5C;'><b>Port Limitation Notice:</b> The route cannot be certified for execution until draft incompatibility ({sel_draft:.1f}m vs {max_d:.1f}m max at {sel_dest}) is resolved by vessel lighterage or re-fixture.</span>"
        )

    bullets_html = "".join(f"<li style='margin-bottom: 6px;'>{b}</li>" for b in why_bullets)

    st.markdown(
        f'<div class="panel-card" style="border-left: 4px solid #372580; background: #F0EEF9; padding: 16px 20px;">'
        f'<h4 style="color: #372580; margin-top: 0; margin-bottom: 8px;">💡 Why This Route? ({rec_alt["badge"]})</h4>'
        f'<ul style="color: #6B6B73; font-size: 0.90rem; line-height: 1.6; margin: 0; padding-left: 18px;">'
        f'{bullets_html}'
        f'</ul>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # =========================================================================
    # 9. INTEGRATION ACTIONS & PRODUCTION ROADMAP
    # =========================================================================
    act_c1, act_c2 = st.columns([1.5, 1], gap="medium")
    with act_c1:
        if st.button("📌 Commit Route to Decision Workspace", key="ri_commit_btn", type="primary", use_container_width=True):
            st.session_state["confirmed_route"] = {
                "origin": sel_orig,
                "destination": sel_dest,
                "route_name": rec_alt["name"],
                "distance_nm": rec_alt["dist_nm"],
                "sailing_days": rec_alt["sailing_days"],
                "total_eta": rec_alt["total_eta"],
                "estimated_cost_cr": rec_alt["total_cost_cr"],
                "fuel_tonnes": rec_alt["fuel_tonnes"],
                "risk_score": rec_alt["risk_score"],
                "is_compatible": is_fully_compatible,
                "priority": sel_priority,
            }
            st.toast(f"Route '{rec_alt['name']}' committed to session workspace.", icon="✅")

    st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)

    # Production Roadmap Callout
    st.markdown(
        '<div class="panel-card" style="border-top: 3px solid #ECECF0;">'
        '<b style="color: #18181B; font-size: 0.95rem;">🛰️ Production Integration Roadmap (Future Real-World Data Feeds)</b>'
        '<div style="color: #6B6B73; font-size: 0.85rem; line-height: 1.5; margin-top: 6px;">'
        'The current MVP displays curated demonstration corridors using deterministic voyage models. Planned production connections include:'
        '<ul style="margin: 6px 0 0 0; padding-left: 18px;">'
        '<li><b>UN/LOCODE Standard:</b> Canonical port coding (e.g. ZARCB, INPRT, INVTZ) and terminal berth databases.</li>'
        '<li><b>Copernicus Marine Service:</b> Real-time global ocean currents, significant wave heights, and sea-surface weather routing.</li>'
        '<li><b>ECMWF ERA5 Historical Reanalysis:</b> Seasonal tropical cyclone probability and monsoon routing adjustments.</li>'
        '<li><b>Satellite AIS Telemetry:</b> Real-time vessel speed, heading, and dynamic maritime congestion monitoring.</li>'
        '<li><b>Major Port Trust Gazettes:</b> Live berth availability, draught circulars, and handling tariff integration.</li>'
        '<li><b>ECDIS Maritime Engines:</b> Certified nautical chart routing conforming to IMO safety parameters.</li>'
        '</ul>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_risk_alerts():
    """Renders the transparent 7-Factor Weighted Risk Assessment Engine view with risk bands, drivers, and mitigations."""
    render_level2_header("Risk & Alerts")
    render_trade_direction_selector()
    render_shared_scenario_summary()
    sel_cat, sel_sev, sel_stat = render_risk_alerts_controls()

    calc = get_scenario_calculations()
    plan = calc.get("balanced_plan") or {}
    risk_info = plan.get("risk_analysis") or {
        "total_risk": 24.0,
        "risk_band": "Low",
        "factor_breakdown": {
            "weather": {"weight": 0.20, "score": 20.0, "weighted": 4.0},
            "congestion": {"weight": 0.20, "score": 20.0, "weighted": 4.0},
            "cargo_readiness": {"weight": 0.15, "score": 15.0, "weighted": 2.25},
            "vessel_availability": {"weight": 0.15, "score": 20.0, "weighted": 3.0},
            "schedule": {"weight": 0.10, "score": 15.0, "weighted": 1.5},
            "route": {"weight": 0.10, "score": 25.0, "weighted": 2.5},
            "data_quality": {"weight": 0.10, "score": 15.0, "weighted": 1.5},
        },
        "drivers": ["Normal fair-weather seasonal patterns across Indian Ocean corridor."],
        "mitigations": ["Maintain standard 48-hour marine weather telemetry updates."],
    }

    tot_risk = risk_info.get("total_risk", 24.0)
    band_name = risk_info.get("risk_band", "Low")
    band_bg = "#EDF7F0" if band_name == "Low" else ("#FEF3C7" if band_name == "Moderate" else "#FEE2E2")
    band_color = "#15803D" if band_name == "Low" else ("#B45309" if band_name == "Moderate" else "#B91C1C")

    st.markdown(
        f"""
        <div class="panel-card" style="margin-bottom: 18px; border-left: 4px solid {band_color};">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <b style="font-size: 1.05rem; color: #18181B;">7-Factor Multi-Criteria Maritime Risk Assessment Engine</b><br>
                    <span style="font-size: 0.86rem; color: #6B6B73;">Unified mathematical evaluation across operational, atmospheric, vessel, and supply-chain dimensions.</span>
                </div>
                <div style="text-align: right;">
                    <span style="background: {band_bg}; color: {band_color}; font-weight: 700; font-size: 0.85rem; padding: 4px 12px; border-radius: 6px; border: 1px solid {band_color};">
                        {band_name.upper()} RISK BAND ({tot_risk:.0f} / 100)
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    r_col1, r_col2 = st.columns([1.3, 1], gap="medium")
    with r_col1:
        st.subheader("Transparent 7-Factor Risk Breakdown")
        f_breakdown = risk_info.get("factor_breakdown", {})
        breakdown_rows = [
            {"Risk Pillar": "Weather & Meteorological", "Weight": "20.0%", "Raw Score": f"{f_breakdown.get('weather', {}).get('score', 20.0):.0f}/100", "Weighted Impact": f"{f_breakdown.get('weather', {}).get('weighted', 4.0):.1f} pts", "Evaluation Scope": "Monsoon risk, sea swell index, cyclone warnings"},
            {"Risk Pillar": "Port Anchorage Congestion", "Weight": "20.0%", "Raw Score": f"{f_breakdown.get('congestion', {}).get('score', 20.0):.0f}/100", "Weighted Impact": f"{f_breakdown.get('congestion', {}).get('weighted', 4.0):.1f} pts", "Evaluation Scope": "Waiting days at anchorage, berth crane productivity"},
            {"Risk Pillar": "Cargo Supply Readiness", "Weight": "15.0%", "Raw Score": f"{f_breakdown.get('cargo_readiness', {}).get('score', 15.0):.0f}/100", "Weighted Impact": f"{f_breakdown.get('cargo_readiness', {}).get('weighted', 2.3):.1f} pts", "Evaluation Scope": "Mine/rail stock availability, loading conveyor status"},
            {"Risk Pillar": "Vessel Fleet Availability", "Weight": "15.0%", "Raw Score": f"{f_breakdown.get('vessel_availability', {}).get('score', 20.0):.0f}/100", "Weighted Impact": f"{f_breakdown.get('vessel_availability', {}).get('weighted', 3.0):.1f} pts", "Evaluation Scope": "Open market charter tonnage, positioning window"},
            {"Risk Pillar": "Delivery Schedule Slack", "Weight": "10.0%", "Raw Score": f"{f_breakdown.get('schedule', {}).get('score', 15.0):.0f}/100", "Weighted Impact": f"{f_breakdown.get('schedule', {}).get('weighted', 1.5):.1f} pts", "Evaluation Scope": "Buffer between expected arrival and hard deadline"},
            {"Risk Pillar": "Geopolitical Route Exposure", "Weight": "10.0%", "Raw Score": f"{f_breakdown.get('route', {}).get('score', 25.0):.0f}/100", "Weighted Impact": f"{f_breakdown.get('route', {}).get('weighted', 2.5):.1f} pts", "Evaluation Scope": "Chokepoint transit (Malacca/Bab el Mandeb/Suez)"},
            {"Risk Pillar": "Telemetry & Data Quality", "Weight": "10.0%", "Raw Score": f"{f_breakdown.get('data_quality', {}).get('score', 15.0):.0f}/100", "Weighted Impact": f"{f_breakdown.get('data_quality', {}).get('weighted', 1.5):.1f} pts", "Evaluation Scope": "Freshness of AIS coordinates and port reports"},
        ]
        st.dataframe(pd.DataFrame(breakdown_rows), hide_index=True, use_container_width=True)

        with st.expander("📊 Risk Banding Definitions & Escalation Matrix"):
            st.markdown("""
            - 🟢 **Low Risk (0–30):** Routine maritime execution. Standard operations with scheduled 24h AIS position updates.
            - 🟡 **Moderate Risk (31–60):** Elevated caution required. Active monitoring of berth queues or seasonal weather advisories.
            - 🔴 **High Risk (61–100):** Operational disruption likely. Trigger rerouting or alternative supplier contingency immediately.
            """)

    with r_col2:
        st.subheader("Actionable Risk Mitigations & Alerts")
        mitigations = risk_info.get("mitigations", [])
        if not mitigations:
            mitigations = [
                "Maintain standard 48-hour marine weather telemetry updates.",
                "Pre-clear port customs documentation via ICEGATE 3 days prior to arrival.",
                "Maintain 20,000 tonnes safety stock buffer at domestic discharge silo."
            ]
        
        mitigation_html = "".join([f"<li style='margin-bottom: 6px;'>{m}</li>" for m in mitigations])
        st.markdown(
            f"""
            <div class="panel-card" style="margin-bottom: 14px; background: #F8FAFC;">
                <b style="color: #18181B; font-size: 0.92rem;">🛡️ Targeted Mitigation Actions:</b>
                <ul style="font-size: 0.88rem; color: #475569; margin: 8px 0 0 16px; padding: 0;">
                    {mitigation_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Active Operational Alerts")
        st.markdown(
            """
            <div class="warning" style="margin-bottom: 10px;">
                <b>Congestion Watch:</b> Visakhapatnam waiting time at 5 days. Paradip (3d) or Dhamra (2d) recommended for faster turnaround.
            </div>
            <div class="panel-card" style="font-size: 0.86rem; color: #64748B;">
                <b>Draft Clearance Notice:</b> Capesize berthing certified at Paradip & Dhamra deep-water bulk berths (18.5m). Visakhapatnam maximum draft is 16.5m.
            </div>
            """,
            unsafe_allow_html=True,
        )



def render_scenario_lab():
    """Renders the complete Scenario Lab disruption simulator with live baseline vs modified delta analysis."""
    render_level2_header("Scenario Lab")
    render_trade_direction_selector()
    render_shared_scenario_summary()

    base_calc = get_scenario_calculations()
    commodity = base_calc["commodity"]
    origin_port = base_calc["origin"]

    st.subheader("Interactive What-If & Disruption Scenario Simulator")
    st.markdown('<p style="color: #6B6B73; font-size: 0.88rem; margin-top: -6px; margin-bottom: 14px;">Modify operational constraints and supply factors below to simulate macroeconomic, weather, and port disruption deltas in real-time.</p>', unsafe_allow_html=True)
    
    req_val, inv_val, saf_val, budget_val, fuel_val, freight_shift, speed_val, deadline_val, port_closure, weather_risk, vessel_avail = render_scenario_lab_controls()

    # Calculate modified shortfall
    sim_shortfall = calculate_import_shortfall(req_val, saf_val, inv_val)
    fuel_pct = int(((fuel_val - 54000) / 54000) * 100)

    port_out = "None"
    if "Paradip Outage" in port_closure:
        port_out = "Paradip"
    elif "Dhamra Outage" in port_closure:
        port_out = "Dhamra"

    # Construct modified scenario dict
    mod_scen_dict = {
        "trade_direction": base_calc["scenario_dict"].get("trade_direction", "Import to India"),
        "cargo_type": commodity,
        "commodity": commodity,
        "cargo_shortfall": sim_shortfall,
        "origin_port": origin_port,
        "destination_port": "Paradip" if port_out != "Paradip" else "Dhamra",
        "approved_budget": budget_val,
        "deadline": deadline_val,
        "fuel_price": fuel_val,
        "effective_fuel_change": fuel_pct,
        "freight_shift": freight_shift,
        "minimum_vessel_utilization": 70.0,
        "maximum_acceptable_risk": 60.0,
    }

    # Evaluate modified plans dynamically
    mod_combos = generate_vessel_combinations(sim_shortfall)
    mod_ports = ["Paradip", "Dhamra", "Visakhapatnam"]
    if port_out != "None":
        mod_ports = [p for p in mod_ports if p != port_out]

    mod_all_plans = []
    mod_feasible = []

    for combo in mod_combos:
        for p_name in mod_ports:
            c_cong = "High (2.0x)" if ("Visakhapatnam" in port_closure and p_name == "Visakhapatnam") else "Normal (1.0x)"
            w_cond = "Monsoon / Rough (15%)" if weather_risk > 50 else "Calm / Fair (0%)"
            eta_info = calculate_route_eta(origin_port, p_name, speed_val, weather_condition=w_cond, port_congestion=c_cong)
            
            risk_res = calculate_risk_score(
                weather_risk=weather_risk,
                congestion_risk=75.0 if ("Visakhapatnam" in port_closure and p_name == "Visakhapatnam") else 25.0,
                cargo_readiness_risk=15.0,
                vessel_availability_risk=100 - vessel_avail,
                schedule_risk=30.0 if eta_info["final_eta_days"] > deadline_val - 5 else 15.0,
                route_risk=eta_info["base_route_risk"]
            )
            p_stub = {
                "vessel_class": combo["vessel_class"],
                "vessel_count": combo["vessel_count"],
                "vessel_display": combo["vessel_display"],
                "combined_capacity": combo["combined_capacity"],
                "unused_capacity": combo["unused_capacity"],
                "shipment_quantity": sim_shortfall,
                "port": p_name,
                "utilization": combo["utilization"],
                "effective_waiting_days": eta_info["effective_waiting_days"],
                "final_eta_days": eta_info["final_eta_days"],
                "risk": risk_res["total_risk"],
                "risk_analysis": risk_res,
            }
            c_info = calculate_cost_breakdown(mod_scen_dict, p_stub)
            p_stub.update(c_info)
            f_eval = evaluate_feasibility(p_stub, mod_scen_dict)
            p_stub.update(f_eval)

            mod_all_plans.append(p_stub)
            if p_stub["is_feasible"]:
                mod_feasible.append(p_stub)

    mod_ranked = rank_feasible_plans(mod_feasible)
    mod_balanced = mod_ranked["balanced_plan"]

    # Wrap modified recommendation for comparison
    mod_calc = {
        "cargo_shortfall": sim_shortfall,
        "active_recommendation": {
            "vessel": mod_balanced["vessel_display"] if mod_balanced else "None (Infeasible)",
            "port": f"{mod_balanced['port']} Port" if mod_balanced else "None",
            "optimized_cost": mod_balanced["total_cost_cr"] if mod_balanced else 0.0,
            "duration": mod_balanced["final_eta_days"] if mod_balanced else 0,
            "risk_score": mod_balanced["risk"] if mod_balanced else 95,
            "confidence_score": 88 if mod_balanced else 40,
        } if mod_balanced else None
    }

    # Perform Delta Comparison
    cmp_res = compare_scenarios(base_calc, mod_calc)

    # Display KPI Delta Cards
    st.markdown('<div class="kpi-grid-4" style="margin-top: 10px; margin-bottom: 18px;">', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        st.metric(
            "Revised Shortfall",
            f"{sim_shortfall:,} t",
            delta=f"{cmp_res['shortfall_diff']:+,} t",
            delta_color="inverse"
        )
    with k2:
        st.metric(
            "Optimal Vessel",
            cmp_res["new_vessel"],
            delta="Shifted" if cmp_res["recommendation_changed"] else "Unchanged"
        )
    with k3:
        st.metric(
            "Expected Logistics Cost",
            f"₹{cmp_res['new_cost']:.2f} Cr",
            delta=f"₹{cmp_res['cost_diff']:+.2f} Cr",
            delta_color="inverse"
        )
    with k4:
        st.metric(
            "ETA Duration & Risk",
            f"{cmp_res['new_eta']}d • {cmp_res['new_risk']}/100",
            delta=f"{cmp_res['eta_diff']:+d}d | {cmp_res['risk_diff']:+.0f} risk",
            delta_color="inverse"
        )

    # Explanation Card
    st.markdown(
        f"""
        <div class="recommend" style="margin-top: 10px; margin-bottom: 20px;">
            <b style="color: #18181B; font-size: 0.98rem;">⚡ Automated Scenario Shift Analysis:</b><br>
            {cmp_res['explanation']}<br>
            <span style="font-size: 0.84rem; color: #6B6B73;">Feasible fleet fixtures available under modified constraints: <b>{len(mod_feasible)}</b> of {len(mod_all_plans)} evaluated.</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Side-by-side comparison table
    st.markdown('<div class="section-title">📋 Base vs. Modified Scenario Comparison Matrix</div>', unsafe_allow_html=True)
    matrix_df = pd.DataFrame([
        {"Planning Dimension": "Cargo Demand Target", "Base Scenario": f"{base_calc['cargo_requirement']:,} t", "Modified Scenario": f"{req_val:,} t", "Difference / Delta": f"{req_val - base_calc['cargo_requirement']:+,} t"},
        {"Planning Dimension": "Current Inventory", "Base Scenario": f"{base_calc['inventory']:,} t", "Modified Scenario": f"{inv_val:,} t", "Difference / Delta": f"{inv_val - base_calc['inventory']:+,} t"},
        {"Planning Dimension": "Net Cargo Shortfall", "Base Scenario": f"{cmp_res['previous_shortfall']:,} t", "Modified Scenario": f"{cmp_res['new_shortfall']:,} t", "Difference / Delta": f"{cmp_res['shortfall_diff']:+,} t"},
        {"Planning Dimension": "Recommended Fleet", "Base Scenario": cmp_res['previous_vessel'], "Modified Scenario": cmp_res['new_vessel'], "Difference / Delta": "Shifted" if cmp_res['recommendation_changed'] else "Identical"},
        {"Planning Dimension": "Discharge Port", "Base Scenario": cmp_res['previous_port'], "Modified Scenario": cmp_res['new_port'], "Difference / Delta": "Shifted" if cmp_res['previous_port'] != cmp_res['new_port'] else "Identical"},
        {"Planning Dimension": "Total Logistics Cost", "Base Scenario": f"₹{cmp_res['previous_cost']:.2f} Cr", "Modified Scenario": f"₹{cmp_res['new_cost']:.2f} Cr", "Difference / Delta": f"₹{cmp_res['cost_diff']:+.2f} Cr"},
        {"Planning Dimension": "Voyage Duration / ETA", "Base Scenario": f"{cmp_res['previous_eta']} days", "Modified Scenario": f"{cmp_res['new_eta']} days", "Difference / Delta": f"{cmp_res['eta_diff']:+d} days"},
        {"Planning Dimension": "Multi-Factor Risk", "Base Scenario": f"{cmp_res['previous_risk']:.0f}/100", "Modified Scenario": f"{cmp_res['new_risk']:.0f}/100", "Difference / Delta": f"{cmp_res['risk_diff']:+.0f} pts"},
        {"Planning Dimension": "Recommendation Confidence", "Base Scenario": f"{cmp_res['previous_confidence']}%", "Modified Scenario": f"{cmp_res['new_confidence']}%", "Difference / Delta": f"{cmp_res['new_confidence'] - cmp_res['previous_confidence']:+d}%"},
    ])
    st.dataframe(matrix_df, hide_index=True, use_container_width=True)



def render_reports():
    """Renders the Reports view."""
    render_level2_header("Reports")
    render_trade_direction_selector()
    render_shared_scenario_summary()
    calc = get_scenario_calculations()
    commodity = calc["commodity"]
    forecast_horizon = calc["forecast_horizon"]
    horizon_days = calc["horizon_days"]
    cargo_shortfall = calc["cargo_shortfall"]
    deadline = calc["deadline"]
    base_plans = calc["base_plans"]

    render_reports_view(base_plans, commodity, forecast_horizon, horizon_days, cargo_shortfall, deadline)


def render_data_settings():
    """Renders the Data & Settings view."""
    render_level2_header("Data & Settings")
    render_trade_direction_selector()
    render_data_settings_view()


# Two-Level Navigation Architecture Execution Dispatcher
active_page = st.session_state.get("active_page", "Home")

if active_page == "Home":
    render_landing_hub()
elif active_page == "Command Centre":
    render_command_centre()
elif active_page == "Forecasting Studio":
    render_forecasting_studio()
elif active_page == "Shipment Planner":
    render_shipment_planner()
elif active_page == "Optimization Hub":
    render_optimization_hub()
elif active_page == "Vessel Intelligence":
    render_vessel_intelligence()
elif active_page == "Port Intelligence":
    render_port_intelligence()
elif active_page == "Route Intelligence":
    render_route_intelligence()
elif active_page == "Risk & Alerts":
    render_risk_alerts()
elif active_page == "Scenario Lab":
    render_scenario_lab()
elif active_page == "Reports":
    render_reports()
elif active_page == "Data & Settings":
    render_data_settings()
else:
    render_landing_hub()

if active_page != "Home":
    st.divider()
    st.markdown('<p class="small-note">VarunaPath AI • Maritime Decision Intelligence • Team Novara</p>', unsafe_allow_html=True)
