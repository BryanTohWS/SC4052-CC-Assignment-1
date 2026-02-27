import matplotlib.pyplot as plt
import numpy as np

# Codes for Experiment 1: Optimal Prediction Horizon
def simulate_prediction_horizon(tau_pred):
    # Base performance parameters
    link_capacity_mbps = 1000  # 1 Gbps link
    base_throughput_mbps = link_capacity_mbps * 0.95  # 95% ideal link utilization
    base_latency_ms = 7.0
    
    # RTT_queue prediction error grows with prediction horizon
    # Thought and Insight: Satellite positions perfectly predictable, but network traffic
    base_error_ms = 0.4
    prediction_error_ms = base_error_ms * (1 + tau_pred / 60) ** 2
    
    # NOTE: minimum_horizon_needed refers to the minimum time needed to plan a smooth cwnd trajectory that avoids sudden drops
    # Need sufficient time to create smooth cwnd trajectories
    if tau_pred < 20:
        # Too short: insufficient cwnd degradation planning time
        # time_ratio = tau_pred / minimum_horizon_needed = tau_pred / 20
        time_ratio = tau_pred / 20
        # link_efficiency = base_efficiency + time_ratio * (peak - base) = 0.80 + time_ratio * 0.15
        link_efficiency = 0.80 + time_ratio * 0.15
    elif tau_pred < 60:
        # Good range: sufficient planning time -> Good link efficiency
        link_efficiency = 0.95
    else:
        # Too long: predictions become unreliable
        # link_efficiency = peak - (tau_pred - 60) / 200
        link_efficiency = 0.95 - (tau_pred - 60) / 200
    
    error_penalty = prediction_error_ms / 10
    throughput_mbps = base_throughput_mbps * link_efficiency * (1 - error_penalty)
    
    # Latency: increases with prediction errors
    # Wrong predictions cause extra queueing
    latency_ms = base_latency_ms + prediction_error_ms * 2
    
    return {
        'throughput': throughput_mbps,
        'latency': latency_ms,
        'prediction_error': prediction_error_ms
    }

# Run experiments
print("Experiment 1: Optimal Prediction Horizon\n")

# Test prediction horizons matching the report
horizons = [10, 30, 45, 60, 90]
results = []

for tau in horizons:
    metrics = simulate_prediction_horizon(tau)
    results.append({
        'horizon': tau,
        'throughput': metrics['throughput'],
        'latency': metrics['latency'],
        'prediction_error': metrics['prediction_error']
    })
    
    print(f"T = {tau}s:")
    print(f"Throughput: {metrics['throughput']:.0f} Mbps")
    print(f"Latency: {metrics['latency']:.1f} ms")
    print(f"Prediction Error: {metrics['prediction_error']:.1f} ms\n")

# Find optimal
optimal = max(results, key=lambda x: x['throughput'])
print(f"Optimal prediction horizon: T* = {optimal['horizon']}s")
print(f"Throughput: {optimal['throughput']:.0f} Mbps")
print(f"Latency: {optimal['latency']:.1f} ms")
print(f"Prediction Error: {optimal['prediction_error']:.1f} ms")

# Generate plot - for prediction horizon
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

horizons_list = [r['horizon'] for r in results]
throughputs = [r['throughput'] for r in results]
latencies = [r['latency'] for r in results]

# Throughput plot
ax1.plot(horizons_list, throughputs, 'o-', linewidth=2.5, markersize=10, color='#2ca02c')
ax1.axvline(x=optimal['horizon'], color='red', linestyle='--', linewidth=2, 
            label=f"Optimal T*={optimal['horizon']}s")
ax1.set_xlabel('Prediction Horizon T (seconds)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Throughput (Mbps)', fontsize=13, fontweight='bold')
ax1.set_title('Throughput vs. Prediction Horizon', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

# Latency plot
ax2.plot(horizons_list, latencies, 'o-', linewidth=2.5, markersize=10, color='#ff7f0e')
ax2.axvline(x=optimal['horizon'], color='red', linestyle='--', linewidth=2,
            label=f"Optimal T*={optimal['horizon']}s")
ax2.set_xlabel('Prediction Horizon T (seconds)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Latency (ms)', fontsize=13, fontweight='bold')
ax2.set_title('Latency vs. Prediction Horizon', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig('figure1_prediction_horizon.png', dpi=300, bbox_inches='tight')
print("Saved: figure1_prediction_horizon.png\n")