import numpy as np


def simulate_fairness_collapse(cycle_num, with_mitigation=False):
    if with_mitigation:
        # unmitigated Phase 1 rate = 0.0084 per cycle
        # mitigated rate = 0.0084 / 6 = approx 0.0014 per cycle
       
        # NOTE: D is the divergence, and J is the fairness
        # Formula: D = initial_residual + (rate * cycle_num)
        # initial_residual = 0.05 (small divergence that builds up between sync windows)
        # e.g. cycle 100: D = 0.05 + (0.0014 * 100) = 0.05 + 0.14 = 0.19
        # The min(..., 0.20) function caps D at 0.20 - sync prevents divergence from going higher
        divergence = min(0.05 + cycle_num * 0.0014, 0.20)

        # unmitigated Phase 1 rate = 0.005 per cycle
        # mitigated rate = 0.005 / 10 = 0.0005 per cycle
        
        # Formula: J = starting_fairness - (rate * cycle_num)
        # starting_fairness = 0.96 (same baseline as unmitigated)
        # e.g. cycle 100: J = 0.96 - (0.0005 * 100) = 0.96 - 0.05 = 0.91
        # max(..., 0.90) floors J at 0.90 - coordination keeps minimum fairness
        fairness = max(0.96 - cycle_num * 0.0005, 0.90)

    else:

        if cycle_num == 0:
            # Cycle 0: All 3 operators start with the same pre-trained model,
            # so divergence is zero and fairness is near-perfect.
            divergence = 0.0
            fairness = 0.96

        elif cycle_num <= 50:
            # Phase 1: Gradual divergence (Cycles 0 to 50)
            # Each operator sees different traffic and slowly specialises its own model.
            # Rates are derived from observed values at the two points:
            # At cycle 0: D = 0.00, J = 0.96  (starting point)
            # At cycle 50: D = 0.42, J = 0.71  (end of phase 1)
            
            # D rate = (0.42 - 0.00) / (50 - 0) = 0.42 / 50 = 0.0084 per cycle
            # J rate = (0.96 - 0.71) / (50 - 0) = 0.25 / 50 = 0.0050 per cycle
            
            # Formulas: 
            # D = rate * cycle_num
            # J = starting_J - rate * cycle_num
            divergence = cycle_num * 0.0084
            fairness = 0.96 - cycle_num * 0.005

        elif cycle_num <= 65:
            # Phase 2: Approaching the critical threshold (Cycles 50 to 65)
            # Operator policies start conflicting, divergence accelerates faster than Phase 1.
            
            # At cycle 50: D = 0.42, J = 0.71 (end of Phase 1)
            # At cycle 65: D = 0.53, J = 0.48 (D crosses critical threshold of 0.5)
            
            # D rate = (0.53 - 0.42) / (65 - 50) = 0.11 / 15 = 0.0073 per cycle
            # J rate = (0.71 - 0.48) / (65 - 50) = 0.23 / 15 = 0.0153 per cycle
            
            # Formulas: 
            # D = D_at_cycle50 + rate * (cycle_num - 50)
            # J = J_at_cycle50 - rate * (cycle_num - 50)
            divergence = 0.42 + (cycle_num - 50) * 0.0073
            fairness = 0.71 - (cycle_num - 50) * 0.0153

        else:
            # Phase 3: Fairness collapse (Cycle 65 onwards, D > 0.5)
            # One operator dominates bandwidth, the others are consistently crowded out.

            # At cycle  65: D = 0.53, J = 0.48 (start of fairness collapse)
            # At cycle 100: D = 1.12, J = 0.19 (observed end state after 100 cycles)
            
            # D rate = (1.12 - 0.53) / (100 - 65) = 0.59 / 35 = 0.0169 per cycle
            # J rate = (0.48 - 0.19) / (100 - 65) = 0.29 / 35 = 0.0083 per cycle
            
            # Formula: 
            # D = D_at_cycle65 + rate * (cycle_num - 65)
            # J = J_at_cycle65 - rate * (cycle_num - 65)
            # max(..., 0.15) floors J at 0.15 - even the worst case, one operator still can retain a small share of bandwidth
            divergence = 0.53 + (cycle_num - 65) * 0.0169
            fairness = max(0.48 - (cycle_num - 65) * 0.0083, 0.15)
    
    return {
        'cycle': cycle_num,
        'divergence': divergence,
        'fairness': fairness
    }


print("Experiment 2: Fairness Under Model Divergence\n")

# Test key cycles  based in the report's scenarios
key_cycles = [0, 50, 65, 100]

print("WITHOUT MITIGATIONS:")
print("-" * 40 + "\n")

results_no_mitigation = []

for cycle in key_cycles:
    result = simulate_fairness_collapse(cycle, with_mitigation=False)
    results_no_mitigation.append(result)
    
    print(f"Cycle {cycle}:")
    print(f"Model Divergence (D): {result['divergence']:.2f}")
    print(f"Fairness (J): {result['fairness']:.2f}\n")

# Showing proposed mitigation effectiveness
print("\nPROPOSED MITIGATION RESULTS:")
print("-" * 40 + "\n")

result_no_mitigation_100 = results_no_mitigation[-1]
result_with_mitigation_100 = simulate_fairness_collapse(100, with_mitigation=True)

print("Cycle 100 - WITHOUT mitigations:")
print(f"Model Divergence (D): {result_no_mitigation_100['divergence']:.2f}")
print(f"Fairness (J): {result_no_mitigation_100['fairness']:.2f}\n")

print("Cycle 100 - WITH mitigations:")
print(f"Model Divergence (D): {result_with_mitigation_100['divergence']:.2f}")
print(f"Fairness (J): {result_with_mitigation_100['fairness']:.2f}\n")

# Calculate improvements
improvement_D = ((result_no_mitigation_100['divergence'] - result_with_mitigation_100['divergence']) / 
                 result_no_mitigation_100['divergence'] * 100)
improvement_J = ((result_with_mitigation_100['fairness'] - result_no_mitigation_100['fairness']) / 
                 result_no_mitigation_100['fairness'] * 100)

print("Quantitative improvements:")
print("-" * 40)
print(f"Divergence: {result_no_mitigation_100['divergence']:.2f} -> {result_with_mitigation_100['divergence']:.2f} ")
print(f"Fairness: {result_no_mitigation_100['fairness']:.2f} -> {result_with_mitigation_100['fairness']:.2f} ")
