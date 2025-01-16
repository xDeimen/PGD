from scipy.optimize import differential_evolution 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import lti, step 
 
# Define the second-order heating system with PID control 
def pid_controller_response(Kp, Ki, Kd, K, tau, zeta, power_input, setpoint, time): 
    omega_n = 1 / tau 
    num = [K * omega_n**2] 
    den = [1, 2 * zeta * omega_n, omega_n**2] 
    pid_num = [Kd, Kp, Ki] 
    pid_den = [1, 0] 
    system = lti(num, den) 
    pid = lti(pid_num, pid_den) 
    closed_loop = lti( 
        np.polymul(pid.num, system.num), 
        np.polyadd(np.polymul(pid.num, system.den), np.polymul(pid.den, system.num)) 
    ) 
    time, response = step(closed_loop, T=time) 
    return time, setpoint * response 
 
# Refined cost function 
def pid_cost_refined(params, time, K, tau, zeta, power_input, setpoint): 
    Kp, Ki, Kd = params 
    time, response = pid_controller_response(Kp, Ki, Kd, K, tau, zeta, power_input, setpoint, time) 
    overshoot = max(response) - setpoint if max(response) > setpoint else 0 
    steady_state_error = abs(response[-1] - setpoint) 
    settling_time = time[np.where(abs(response - setpoint) > 0.05 * setpoint)[0][-1]] if max(response) > setpoint else time[-1] 
    cost = 100 * overshoot + 50 * steady_state_error + 20 * settling_time + 0.1 * (Kp + Ki + Kd) 
    return cost 
 
# System parameters 
K = 5 
tau = 10 
zeta = 0.7 
power_input = 10 
setpoint = 50 
time = np.linspace(0, 50, 500) 
 
# Parameter bounds for DE 
bounds = [(0.05, 10), (0.0001, 1), (0.0001, 1)] 
 
# Run DE optimization 
result_de = differential_evolution( 
    pid_cost_refined, bounds=bounds, 
    args=(time, K, tau, zeta, power_input, setpoint), 
    maxiter=1000, popsize=50, strategy='best1bin', tol=1e-6 
) 
 
# Extract optimized parameters 
Kp_opt, Ki_opt, Kd_opt = result_de.x 
 
# Simulate the optimized response 
time, response = pid_controller_response(Kp_opt, Ki_opt, Kd_opt, K, tau, zeta, power_input, setpoint, time) 
 
# Calculate performance metrics 
def calculate_metrics(time, response, setpoint): 
    overshoot = max(response) - setpoint if max(response) > setpoint else 0 
    overshoot_percent = (overshoot / setpoint) * 100 
    steady_state_error = abs(response[-1] - setpoint) 
    settling_time = time[np.where(abs(response - setpoint) > 0.05 * setpoint)[0][-1]] if max(response) > setpoint else time[-1] 
    rise_time_idx = np.where((response >= 0.1 * setpoint) & (response <= 0.9 * setpoint))[0] 
    rise_time = time[rise_time_idx[-1]] - time[rise_time_idx[0]] if len(rise_time_idx) > 1 else None 
    return { 
        "Overshoot (%)": overshoot_percent, 
        "Steady-State Error": steady_state_error, 
        "Settling Time (s)": settling_time, 
        "Rise Time (s)": rise_time, 
    } 
 
metrics_de = calculate_metrics(time, response, setpoint) 
 
# Plot the optimized response 
plt.figure(figsize=(12, 8)) 
plt.plot(time, response, label=f'DE Optimized PID (Kp={Kp_opt:.2f}, Ki={Ki_opt:.2f}, Kd={Kd_opt:.2f})', linewidth=2) 
plt.axhline(y=setpoint, color='r', linestyle='--', label='Setpoint = 50°C') 
plt.title('DE Optimized PID Response', fontsize=16) 
plt.xlabel('Time (seconds)', fontsize=14) 
plt.ylabel('Temperature (°C)', fontsize=14) 
plt.grid(True, linestyle='--', alpha=0.7) 
plt.legend(fontsize=12) 
plt.tight_layout() 
plt.show() 
 
# Display performance metrics 
print("DE Optimized Performance Metrics:") 
for metric, value in metrics_de.items(): 
    print(f"{metric}: {value:.2f}")