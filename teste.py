from utils.ahp_weights import critical_scenario_weights, equilibrium_scenario_weights, limited_scenario_weights


assertiveness_weights, cost_weights, rc_a, rc_c = equilibrium_scenario_weights()

print(assertiveness_weights)
print(cost_weights)
print(rc_a)
print(rc_c)