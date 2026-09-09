import math
import unittest
import pandas as pd

# Core calculation formulas as defined in VarunaPath AI
def calc_shortfall(req, safety, inv):
    return max(0, req + safety - inv)

def calc_combined_capacity(vessel_count, vessel_capacity):
    return vessel_count * vessel_capacity

def calc_utilization(shortfall, combined_capacity):
    if combined_capacity <= 0:
        return 0.0
    return (shortfall / combined_capacity) * 100.0

def calc_savings(baseline_cost, optimized_cost):
    return round(baseline_cost - optimized_cost, 2)

def calc_savings_percentage(savings, baseline_cost):
    if baseline_cost <= 0:
        return 0.0
    return round((savings / baseline_cost) * 100.0, 1)

def is_vessel_feasible(combined_capacity, shortfall):
    return combined_capacity >= shortfall


class TestVarunaPathCalculations(unittest.TestCase):
    def test_default_baseline_shortfall(self):
        # Default scenario: Req 150,000 t, Safety 20,000 t, Inv 40,000 t
        shortfall = calc_shortfall(150000, 20000, 40000)
        self.assertEqual(shortfall, 130000)

    def test_panamax_fleet_capacity_and_utilization(self):
        # 2 x Panamax (82,000 t each) for 130,000 t shortfall
        capacity = calc_combined_capacity(2, 82000)
        self.assertEqual(capacity, 164000)
        util = calc_utilization(130000, capacity)
        self.assertAlmostEqual(util, 79.268, places=2)
        self.assertEqual(round(util, 1), 79.3)

    def test_strict_vessel_feasibility(self):
        shortfall = 130000
        # 2 x Supramax (58,000 t each) = 116,000 t -> INFEASIBLE (Deficit: 14,000 t)
        supramax_cap = calc_combined_capacity(2, 58000)
        self.assertEqual(supramax_cap, 116000)
        self.assertFalse(is_vessel_feasible(supramax_cap, shortfall))
        self.assertEqual(shortfall - supramax_cap, 14000)

        # 2 x Panamax = 164,000 t -> FEASIBLE
        panamax_cap = calc_combined_capacity(2, 82000)
        self.assertTrue(is_vessel_feasible(panamax_cap, shortfall))

        # 1 x Capesize = 180,000 t -> FEASIBLE
        capesize_cap = calc_combined_capacity(1, 180000)
        self.assertTrue(is_vessel_feasible(capesize_cap, shortfall))

    def test_baseline_and_savings(self):
        baseline = 31.19
        optimized = 28.60
        savings = calc_savings(baseline, optimized)
        savings_pct = calc_savings_percentage(savings, baseline)
        self.assertEqual(savings, 2.59)
        self.assertEqual(savings_pct, 8.3)

    def test_cost_equation_breakdown(self):
        # 130,000 t from Richards Bay (Anglo/Ubuntu 1428/1450) to Paradip
        qty = 130000
        cargo_cost = qty * 1428 / 10_000_000 # 18.564 Cr
        charter_cost = 2 * 4.30 # 8.60 Cr
        port_cost = qty * 85 / 10_000_000 # 1.105 Cr
        waiting_cost = 2 * 3 * 0.02 # 0.12 Cr
        utilization = 130000 / 164000
        low_util_penalty = max(0, 0.80 - utilization) * charter_cost * 0.5
        risk = (0.06 + 0.05 + 0.05) / 3
        risk_cost = risk * (charter_cost + port_cost) * 0.35
        total = cargo_cost + charter_cost + port_cost + waiting_cost + low_util_penalty + risk_cost
        self.assertAlmostEqual(total, 28.60, places=1)


if __name__ == "__main__":
    unittest.main()
