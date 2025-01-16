import numpy as np 
import matplotlib.pyplot as plt 
 
# Adaptive PI-D Controller with Saturation and Anti-Windup 
def adaptive_pid_controller(Kp_init, Ki_init, Kd_init, K, tau, zeta, setpoint, time): 
    omega_n = 1 / tau 
    num = [K * omega_n**2] 
    den = [1, 2 * zeta * omega_n, omega_n**2] 
     
    # Initialize variables 
    integral = 0 
    prev_error = 0 
    response = np.zeros_like(time) 
    dt = time[1] - time[0] 
     
    # Adaptive gains 
    Kp, Ki, Kd = Kp_init, Ki_init, Kd_init 
    Kp_min, Kp_max = 0.5, Kp_init 
    Ki_min, Ki_max = 0.01, Ki_init 
    Kd_min, Kd_max = 0.01, Kd_init 
     
    for i in range(1, len(time)): 
        error = setpoint - response[i - 1] 
         
        # Adaptive gain adjustment 
        Kp = max(Kp_min, Kp_max * (1 - np.abs(error) / setpoint))  # Reduce Kp near setpoint 
        Ki = min(Ki_max, Ki_min + 0.01 * (np.abs(error) / setpoint))  # Increase Ki if error persists 
        Kd = Kd_max * (1 - np.abs(error) / setpoint)  # Reduce Kd near setpoint 
         
        # Anti-windup for integral term 
        if response[i - 1] < 1.5 * setpoint: 
            integral += error * dt 
        else: 
            integral = integral  # Freeze integral term when output is saturated 
         
        derivative = (error - prev_error) / dt 
        control_signal = Kp * error + Ki * integral - Kd * derivative 
        control_signal = np.clip(control_signal, 0, 1.5 * setpoint)  # Apply output saturation 
         
        # Update system response using simplified dynamics 
        response[i] = response[i - 1] + control_signal * dt / tau 
        prev_error = error 
     
    return time, response 
 
# System Parameters 
K, tau, zeta = 5, 10, 0.7 
setpoint = 50 
time_vector = np.linspace(0, 50, 500) 
 
# Initial Controller Gains 
Kp_init, Ki_init, Kd_init = 6.0, 1.0, 1.0 
 
# Run the Adaptive PI-D Controller 
time, response = adaptive_pid_controller(Kp_init, Ki_init, Kd_init, K, tau, zeta, setpoint, time_vector) 
 
# Calculate Performance Metrics 
def calculate_metrics(time, response, setpoint): 
    overshoot = max(response) - setpoint if max(response) > setpoint else 0 
    sse = abs(response[-1] - setpoint) 
    settling_time = time[np.where(abs(response - setpoint) > 0.05 * setpoint)[0][-1]] if max(response) > setpoint else time[-1] 
    return {"Overshoot (%)": (overshoot / setpoint) * 100, "Steady-State Error": sse, "Settling Time (s)": settling_time} 
 
metrics = calculate_metrics(time, response, setpoint) 
print("Adaptive PI-D Controller Metrics:") 
for metric, value in metrics.items(): 
    print(f"{metric}: {value:.2f}") 
 
# Plot the Response 
plt.figure(figsize=(12, 8)) 
plt.plot(time, response, label='Adaptive PI-D Controller Response', linewidth=2) 
plt.axhline(y=setpoint, color='r', linestyle='--', label='Setpoint') 
plt.title('Adaptive PI-D Controller Response with Saturation and Anti-Windup') 
plt.xlabel('Time (seconds)') 
plt.ylabel('Temperature (°C)') 
plt.legend() 
plt.grid() 
plt.show()