from pyswarm import pso 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import lti, step 
 
# Define the second-order heating system with PID control 
def pid_controller_response(Kp, Ki, Kd, K, tau, zeta, power_input, setpoint, time): 
    """ 
    Simulate the PID-controlled system's response. 
    """ 
    omega_n = 1 / tau  # Natural frequency 
    num = [K * omega_n**2]  # Numerator of the second-order system 
    den = [1, 2 * zeta * omega_n, omega_n**2]  # Denominator of the second-order system 
    pid_num = [Kd, Kp, Ki]  # PID numerator 
    pid_den = [1, 0]  # PID denominator 
     
    # Normalize the coefficients for numerical stability 
    num, den = np.array(num) / np.max(num), np.array(den) / np.max(den) 
    pid_num, pid_den = np.array(pid_num) / np.max(pid_num), np.array(pid_den) / np.max(pid_den) 
     
    # Define transfer functions 
    system = lti(num, den)  # Second-order system 
    pid = lti(pid_num, pid_den)  # PID controller 
    closed_loop = lti( 
        np.polymul(pid.num, system.num), 
        np.polyadd(np.polymul(pid.num, system.den), np.polymul(pid.den, system.num)) 
    ) 
     
    time, response = step(closed_loop, T=time) 
    return time, setpoint * response 
 
# Define the cost function for PSO 
def pid_cost_pso(params, time, K, tau, zeta, power_input, setpoint): 
    """ 
    Cost function to minimize during PSO. 
    """ 
    Kp, Ki, Kd = params 
    time, response = pid_controller_response(Kp, Ki, Kd, K, tau, zeta, power_input, setpoint, time) 
    overshoot = max(response) - setpoint if max(response) > setpoint else 0 
    steady_state_error = abs(response[-1] - setpoint) 
    settling_time = time[np.where(abs(response - setpoint) > 0.05 * setpoint)[0][-1]] if max(response) > setpoint else time[-1] 
     
    # Weighted cost function 
    cost = 20 * overshoot + 10 * steady_state_error + 5 * settling_time 
    return cost 
 
# System parameters 
K = 5  # System gain (°C/W) 
20 
 
tau = 10  # Time constant (seconds) 
zeta = 0.7  # Damping ratio 
power_input = 10  # Power input (W) 
setpoint = 50  # Desired temperature (°C) 
time = np.linspace(0, 50, 500)  # Time vector (seconds) 
 
# PSO parameter bounds 
lb = [0.1, 0.001, 0.001]  # Lower bounds for [Kp, Ki, Kd] 
ub = [5, 0.2, 0.5]        # Upper bounds for [Kp, Ki, Kd] 
 
# Run PSO to optimize PID parameters 
best_params, best_cost = pso(pid_cost_pso, lb, ub, args=(time, K, tau, zeta, power_input, setpoint), swarmsize=30, maxiter=100) 
# Extract optimized parameters 
Kp_opt, Ki_opt, Kd_opt = best_params 
# Simulate the optimized PID response 
time, response = pid_controller_response(Kp_opt, Ki_opt, Kd_opt, K, tau, zeta, power_input, setpoint, time) 
 
# Function to calculate performance metrics 
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
 
# Calculate performance metrics 
metrics_pso = calculate_metrics(time, response, setpoint) 
 
# Plot the optimized response 
plt.figure(figsize=(12, 8)) 
plt.plot(time, response, label=f'PSO Optimized PID (Kp={Kp_opt:.2f}, Ki={Ki_opt:.2f}, Kd={Kd_opt:.2f})', linewidth=2) 
plt.axhline(y=setpoint, color='r', linestyle='--', label='Setpoint = 50°C') 
plt.title('PSO Optimized PID Response', fontsize=16) 
plt.xlabel('Time (seconds)', fontsize=14) 
plt.ylabel('Temperature (°C)', fontsize=14) 
plt.grid(True, linestyle='--', alpha=0.7) 
plt.legend(fontsize=12) 
plt.tight_layout() 
plt.show() 
 
# Display performance metrics 
print("PSO Optimized Performance Metrics:") 
for metric, value in metrics_pso.items(): 
    print(f"{metric}: {value:.2f}")