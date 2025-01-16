from scipy.optimize import minimize 
from scipy.signal import lti, step 
import numpy as np 
import matplotlib.pyplot as plt 
 
def pid_controller_response(Kp, Ki, Kd, K, tau, zeta, power_input, setpoint, time): 
    omega_n = 1 / tau  # Natural frequency 
    num = [K * omega_n**2]  # Numerator of the second-order system 
    den = [1, 2 * zeta * omega_n, omega_n**2]  # Denominator of the second-order system 
    pid_num = [Kd, Kp, Ki]  # PID controller numerator 
    pid_den = [1, 0]  # PID controller denominator 
 
    # Normalize coefficients for numerical stability 
    num, den = np.array(num) / np.max(num), np.array(den) / np.max(den) 
    pid_num, pid_den = np.array(pid_num) / np.max(pid_num), np.array(pid_den) / np.max(pid_den) 
    # Define the transfer functions 
    system = lti(num, den) 
    pid = lti(pid_num, pid_den) 
    closed_loop = lti( 
        np.polymul(pid.num, system.num), 
        np.polyadd(np.polymul(pid.num, system.den), np.polymul(pid.den, system.num)) 
    ) 
 
    time, response = step(closed_loop, T=time) 
    return time, setpoint * response 
 
# System parameters 
K = 5 
tau = 10 
zeta = 0.7 
power_input = 10 
setpoint = 50 
time = np.linspace(0, 50, 500) 
 
# Cost function with regularization 
def pid_cost(params, K, tau, zeta, power_input, setpoint, time): 
    Kp, Ki, Kd = params 
    time, response = pid_controller_response(Kp, Ki, Kd, K, tau, zeta, power_input, setpoint, time) 
    overshoot = max(response) - setpoint if max(response) > setpoint else 0 
    steady_state_error = abs(response[-1] - setpoint) 
    settling_time = time[np.where(abs(response - setpoint) > 0.05 * setpoint)[0][-1]] if overshoot else time[-1] 
    cost = 10 * overshoot + settling_time + 100 * steady_state_error + 0.1 * (Kp**2 + Ki**2 + Kd**2) 
    return cost 
 
# Initial guesses for Kp, Ki, and Kd 
initial_params = [1, 0.1, 0.01] 
bounds = [(0.01, 10), (0.001, 1), (0.001, 1)] 
 
# Run optimization with Nelder-Mead for stability 
result = minimize(pid_cost, initial_params, args=(K, tau, zeta, power_input, setpoint, time), 
                  bounds=bounds, method='Nelder-Mead') 
 
# Optimized PID parameters 
Kp_opt, Ki_opt, Kd_opt = result.x 
 
# Simulate the optimized response 
time, response = pid_controller_response(Kp_opt, Ki_opt, Kd_opt, K, tau, zeta, power_input, setpoint, time) 
 
# Plot the optimized response 
plt.figure(figsize=(12, 8)) 
plt.plot(time, response, label=f'Optimized PID (Kp={Kp_opt:.2f}, Ki={Ki_opt:.2f}, Kd={Kd_opt:.2f})', linewidth=2) 
plt.axhline(y=setpoint, color='r', linestyle='--', label='Setpoint = 50°C') 
plt.title('Optimized PID Response', fontsize=16) 
plt.xlabel('Time (seconds)', fontsize=14) 
plt.ylabel('Temperature (°C)', fontsize=14) 
plt.grid(True, linestyle='--', alpha=0.7) 
plt.legend(fontsize=12) 
plt.tight_layout() 
plt.show()

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
 
# Simulate the optimized PID response 
time, response = pid_controller_response(Kp_opt, Ki_opt, Kd_opt, K, tau, zeta, power_input, setpoint, time) 
 
# Calculate performance metrics 
metrics = calculate_metrics(time, response, setpoint) 
 
# Display the metrics 
for metric, value in metrics.items(): 
    print(f"{metric}: {value:.2f}")
