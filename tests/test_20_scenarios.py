"""
VarunaPath AI - Comprehensive 20-Scenario Verification Test Suite
Tests all 20 non-negotiable operational and edge-case requirements:
1. Default import scenario
2. Inventory increased from 40,000 to 80,000 tonnes
3. Demand set to zero
4. Inventory greater than demand
5. No vessel available
6. Two Supramax capacity insufficient (deficit 14,000 t)
7. Two Panamax capacity feasible (164,000 t)
8. Port-draft constraint failure
9. Budget too low
10. Delivery deadline impossible
11. Fuel price increased by 10%
12. High port congestion
13. High weather risk
14. Origin and destination same port
15. Export order quantity set to zero
16. Export inventory insufficient
17. Export inventory greater than order quantity
18. Laycan end date before start date
19. No feasible plans available
20. Only one feasible plan available
"""

import sys
import os
import unittest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
    validate_scenario_inputs,
)

class TestTwentyScenarios(unittest.TestCase):
    def test_case_01_default_import_scenario(self):
        # 1. Default import scenario: 150k demand, 20k safety, 40k inv -> 130k shortfall
        sf = calculate_import_shortfall(150000, 20000, 40000)
        self.assertEqual(sf, 130000)
        plan = {
            "vessel_class": "Panamax", "vessel_count": 2, "combined_capacity": 164000,
            "shipment_quantity": 130000, "port": "Paradip", "utilization": 79.3,
            "risk": 24.0, "final_eta_days": 19, "total_cost_cr": 28.60,
        }
        scen = {"trade_direction": "Import to India", "cargo_shortfall": 130000, "destination_port": "Paradip", "approved_budget": 35.0, "deadline": 45}
        feas = evaluate_feasibility(plan, scen)
        self.assertTrue(feas["is_feasible"])
        costs = calculate_cost_breakdown(scen, plan)
        self.assertEqual(costs["baseline_cost_cr"], 31.19)
        self.assertEqual(costs["total_cost_cr"], 28.60)
        self.assertEqual(costs["savings_cr"], 2.59)

    def test_case_02_inventory_increased_to_80k(self):
        # 2. Inventory increased from 40k to 80k -> Shortfall becomes 150k + 20k - 80k = 90k tonnes
        sf = calculate_import_shortfall(150000, 20000, 80000)
        self.assertEqual(sf, 90000)
        # For 90k shortfall, 2 x Supramax (116k) now becomes FEASIBLE!
        comb = generate_vessel_combinations(90000)
        supra_2 = [c for c in comb if c["vessel_class"] == "Supramax" and c["vessel_count"] == 2][0]
        self.assertTrue(supra_2["combined_capacity"] >= 90000)

    def test_case_03_demand_set_to_zero(self):
        # 3. Demand set to zero -> Shortfall = max(0, 0 + 20k - 40k) = 0
        sf = calculate_import_shortfall(0, 20000, 40000)
        self.assertEqual(sf, 0)

    def test_case_04_inventory_greater_than_demand(self):
        # 4. Inventory greater than demand -> e.g. demand 100k, safety 10k, inv 150k -> shortfall 0
        sf = calculate_import_shortfall(100000, 10000, 150000)
        self.assertEqual(sf, 0)

    def test_case_05_no_vessel_available(self):
        # 5. No vessel available -> market availability exceeded (e.g. asking for 5 Capesize when only 2 available)
        plan = {
            "vessel_class": "Capesize", "vessel_count": 5, "combined_capacity": 900000,
            "shipment_quantity": 130000, "port": "Paradip", "total_cost_cr": 52.0, "final_eta_days": 19,
        }
        scen = {"trade_direction": "Import to India", "cargo_shortfall": 130000, "destination_port": "Paradip", "approved_budget": 60.0, "deadline": 45}
        feas = evaluate_feasibility(plan, scen)
        self.assertFalse(feas["is_feasible"])
        self.assertIn("Vessel Availability", feas["failed_constraints"])

    def test_case_06_two_supramax_capacity_insufficient(self):
        # 6. Two Supramax (116k t) insufficient for 130k t (Deficit: 14,000 tonnes)
        plan = {
            "vessel_class": "Supramax", "vessel_count": 2, "combined_capacity": 116000,
            "shipment_quantity": 130000, "port": "Paradip", "total_cost_cr": 26.50, "final_eta_days": 18,
        }
        scen = {"trade_direction": "Import to India", "cargo_shortfall": 130000, "destination_port": "Paradip", "approved_budget": 35.0, "deadline": 45}
        feas = evaluate_feasibility(plan, scen)
        self.assertFalse(feas["is_feasible"])
        self.assertIn("Capacity", feas["failed_constraints"])
        self.assertIn("14,000 tonnes lower", feas["checklist"]["Capacity"]["detail"])

    def test_case_07_two_panamax_capacity_feasible(self):
        # 7. Two Panamax (164k t) strictly feasible for 130k t shortfall
        plan = {
            "vessel_class": "Panamax", "vessel_count": 2, "combined_capacity": 164000,
            "shipment_quantity": 130000, "port": "Paradip", "total_cost_cr": 28.60, "final_eta_days": 19,
            "utilization": 79.3, "risk": 24.0,
        }
        scen = {"trade_direction": "Import to India", "cargo_shortfall": 130000, "destination_port": "Paradip", "approved_budget": 35.0, "deadline": 45}
        feas = evaluate_feasibility(plan, scen)
        self.assertTrue(feas["is_feasible"])

    def test_case_08_port_draft_constraint_failure(self):
        # 8. Port-draft constraint failure: Capesize draft 18.5m at Visakhapatnam (max 16.5m)
        is_comp, reasons = evaluate_port_compatibility("Visakhapatnam", "Capesize", "Thermal Coal", vessel_draft=18.5)
        self.assertFalse(is_comp)
        self.assertTrue(any("exceeds" in r or "draft" in r for r in reasons))

    def test_case_09_budget_too_low(self):
        # 9. Budget too low: e.g. budget = 20.0 Cr while plan cost = 28.60 Cr
        plan = {
            "vessel_class": "Panamax", "vessel_count": 2, "combined_capacity": 164000,
            "shipment_quantity": 130000, "port": "Paradip", "total_cost_cr": 28.60, "final_eta_days": 19,
            "utilization": 79.3, "risk": 24.0,
        }
        scen = {"trade_direction": "Import to India", "cargo_shortfall": 130000, "destination_port": "Paradip", "approved_budget": 20.0, "deadline": 45}
        feas = evaluate_feasibility(plan, scen)
        self.assertFalse(feas["is_feasible"])
        self.assertIn("Budget", feas["failed_constraints"])

    def test_case_10_delivery_deadline_impossible(self):
        # 10. Delivery deadline impossible: deadline = 10 days while route ETA = 19 days
        plan = {
            "vessel_class": "Panamax", "vessel_count": 2, "combined_capacity": 164000,
            "shipment_quantity": 130000, "port": "Paradip", "total_cost_cr": 28.60, "final_eta_days": 19,
            "utilization": 79.3, "risk": 24.0,
        }
        scen = {"trade_direction": "Import to India", "cargo_shortfall": 130000, "destination_port": "Paradip", "approved_budget": 35.0, "deadline": 10}
        feas = evaluate_feasibility(plan, scen)
        self.assertFalse(feas["is_feasible"])
        self.assertIn("Deadline", feas["failed_constraints"])

    def test_case_11_fuel_price_increased_by_10_percent(self):
        # 11. Fuel price increased by 10% -> Ocean freight adjustment increases
        scen_base = {"trade_direction": "Import to India", "cargo_shortfall": 130000, "destination_port": "Paradip", "effective_fuel_change": 0.0}
        scen_fuel10 = {"trade_direction": "Import to India", "cargo_shortfall": 130000, "destination_port": "Paradip", "effective_fuel_change": 10.0}
        plan = {"vessel_class": "Panamax", "vessel_count": 2, "combined_capacity": 164000, "shipment_quantity": 130000, "port": "Paradip", "utilization": 79.3, "risk": 24.0}
        cost_base = calculate_cost_breakdown(scen_base, plan)
        cost_fuel10 = calculate_cost_breakdown(scen_fuel10, plan)
        self.assertGreater(cost_fuel10["ocean_charter_cr"], cost_base["ocean_charter_cr"])

    def test_case_12_high_port_congestion(self):
        # 12. High port congestion: 2.0x waiting multiplier elevates final ETA and congestion risk
        eta_res = calculate_route_eta("Richards Bay, South Africa", "Paradip", 12.0, port_congestion="High (2.0x)")
        self.assertGreater(eta_res["congestion_buffer_days"], 0.0)
        self.assertGreater(eta_res["final_eta_days"], eta_res["base_sailing_days"] + eta_res["base_waiting_days"])

    def test_case_13_high_weather_risk(self):
        # 13. High weather risk -> activates weather buffer and increases risk score
        eta_res = calculate_route_eta("Richards Bay, South Africa", "Paradip", 12.0, weather_condition="Monsoon / Rough (15%)")
        self.assertGreater(eta_res["weather_buffer_days"], 0.0)
        risk_res = calculate_risk_score(80, 20, 20, 20, 20, 20, 15)
        self.assertEqual(risk_res["highest_risk_driver"], "Weather Severity")

    def test_case_14_origin_and_destination_same_port(self):
        # 14. Origin and destination selected as same port -> triggers validation error without crash
        scen = {"origin_port": "Paradip", "destination_port": "Paradip", "approved_budget": 35.0, "deadline": 45}
        errors = validate_scenario_inputs(scen)
        self.assertTrue(any("cannot be the same" in e for e in errors))

    def test_case_15_export_order_quantity_zero(self):
        # 15. Export order quantity set to zero -> validation error and 0 readiness %
        scen = {"trade_direction": "Export from India", "export_order_quantity": 0, "approved_budget": 35.0, "deadline": 45}
        errors = validate_scenario_inputs(scen)
        self.assertTrue(any("greater than zero" in e for e in errors))
        exp_res = calculate_export_readiness(0, 50000, 20000, 10000, 0)
        self.assertEqual(exp_res["order_readiness_percentage"], 0.0)

    def test_case_16_export_inventory_insufficient(self):
        # 16. Export inventory insufficient: 100k order, 30k inv, 10k prod, 10k res -> 30k ready, 70k shortfall
        exp_res = calculate_export_readiness(100000, 30000, 10000, 10000, 0)
        self.assertEqual(exp_res["export_ready_quantity"], 30000)
        self.assertEqual(exp_res["fulfilment_shortfall"], 70000)
        self.assertFalse(exp_res["is_fully_ready"])

    def test_case_17_export_inventory_greater_than_order(self):
        # 17. Export inventory greater than order quantity: 50k order, 80k inv, 10k prod, 10k res -> 80k ready, 0 shortfall, 100%
        exp_res = calculate_export_readiness(50000, 80000, 10000, 10000, 0)
        self.assertEqual(exp_res["fulfilment_shortfall"], 0)
        self.assertEqual(exp_res["order_readiness_percentage"], 100.0)
        self.assertTrue(exp_res["is_fully_ready"])

    def test_case_18_laycan_end_date_before_start_date(self):
        # 18. Laycan end date before start date -> triggers validation error without crash
        scen = {"trade_direction": "Export from India", "export_order_quantity": 50000, "laycan_start_date": "2026-10-15", "laycan_end_date": "2026-10-10", "approved_budget": 35.0, "deadline": 45}
        errors = validate_scenario_inputs(scen)
        self.assertTrue(any("Laycan end date cannot be earlier" in e for e in errors))

    def test_case_19_no_feasible_plans_available(self):
        # 19. No feasible plans available -> rank_feasible_plans handles empty list gracefully
        res = rank_feasible_plans([])
        self.assertIsNone(res["lowest_cost_plan"])
        self.assertIsNone(res["lowest_risk_plan"])
        self.assertIsNone(res["balanced_plan"])
        exp = generate_explanation(None, [], {})
        self.assertTrue(any("No feasible plan met" in r for r in exp["why_this_plan"]))

    def test_case_20_only_one_feasible_plan_available(self):
        # 20. Only one feasible plan available -> rank_feasible_plans assigns all 3 recommendations to it without error
        plan = {
            "vessel_class": "Panamax", "vessel_count": 2, "vessel_display": "2 × Panamax",
            "combined_capacity": 164000, "unused_capacity": 34000, "shipment_quantity": 130000,
            "port": "Paradip", "total_cost_cr": 28.60, "final_eta_days": 19, "utilization": 79.3,
            "risk": 24.0, "is_feasible": True,
        }
        res = rank_feasible_plans([plan])
        self.assertIsNotNone(res["balanced_plan"])
        self.assertEqual(res["balanced_plan"]["vessel_display"], "2 × Panamax")
        self.assertEqual(res["lowest_cost_plan"]["vessel_display"], "2 × Panamax")
        self.assertEqual(res["lowest_risk_plan"]["vessel_display"], "2 × Panamax")


if __name__ == "__main__":
    unittest.main()
