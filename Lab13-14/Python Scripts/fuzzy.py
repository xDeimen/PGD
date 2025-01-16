import numpy as np 
import skfuzzy as fuzz 
from skfuzzy import control as ctrl 
import matplotlib.pyplot as plt 
 
# Define Fuzzy Variables 
error = ctrl.Antecedent(np.linspace(-10, 10, 100), 'error')  # Error range: -10 to 10 
delta_error = ctrl.Antecedent(np.linspace(-5, 5, 100), 'delta_error')  # Change in error 
kp = ctrl.Consequent(np.linspace(0, 10, 100), 'kp')  # Proportional gain 
ki = ctrl.Consequent(np.linspace(0, 1, 100), 'ki')   # Integral gain 
kd = ctrl.Consequent(np.linspace(0, 1, 100), 'kd')   # Derivative gain 
 
# Membership Functions for Error 
error['negative'] = fuzz.trapmf(error.universe, [-10, -10, -5, 0]) 
error['zero'] = fuzz.trimf(error.universe, [-5, 0, 5]) 
error['positive'] = fuzz.trapmf(error.universe, [0, 5, 10, 10]) 
 
# Membership Functions for Change in Error 
delta_error['negative'] = fuzz.trapmf(delta_error.universe, [-5, -5, -2.5, 0]) 
delta_error['zero'] = fuzz.trimf(delta_error.universe, [-2.5, 0, 2.5]) 
delta_error['positive'] = fuzz.trapmf(delta_error.universe, [0, 2.5, 5, 5]) 
 
# Membership Functions for PID Gains 
kp['low'] = fuzz.trimf(kp.universe, [0, 0, 5]) 
kp['medium'] = fuzz.trimf(kp.universe, [0, 5, 10]) 
kp['high'] = fuzz.trimf(kp.universe, [5, 10, 10]) 
 
ki['low'] = fuzz.trimf(ki.universe, [0, 0, 0.5]) 
ki['medium'] = fuzz.trimf(ki.universe, [0, 0.5, 1]) 
ki['high'] = fuzz.trimf(ki.universe, [0.5, 1, 1]) 
 
kd['low'] = fuzz.trimf(kd.universe, [0, 0, 0.5]) 
kd['medium'] = fuzz.trimf(kd.universe, [0, 0.5, 1]) 
kd['high'] = fuzz.trimf(kd.universe, [0.5, 1, 1]) 
 
# Define Fuzzy Rules 
rule1 = ctrl.Rule(error['negative'] & delta_error['negative'], (kp['high'], ki['low'], kd['medium'])) 
rule2 = ctrl.Rule(error['zero'] & delta_error['zero'], (kp['medium'], ki['medium'], kd['low'])) 
rule3 = ctrl.Rule(error['positive'] & delta_error['positive'], (kp['low'], ki['low'], kd['medium'])) 
rule4 = ctrl.Rule(error['negative'] & delta_error['positive'], (kp['medium'], ki['low'], kd['low'])) 
rule5 = ctrl.Rule(error['positive'] & delta_error['negative'], (kp['medium'], ki['low'], kd['low'])) 
 
# Create Control System 
pid_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5]) 
pid_simulator = ctrl.ControlSystemSimulation(pid_control) 
 
# Simulate PID Response 
def simulate_pid_response(setpoint, time_vector): 
    response = np.zeros_like(time_vector) 
    error = setpoint - response[0] 
    prev_error = 0 
    integral = 0 
 
    for i in range(1, len(time_vector)): 
        delta_error = error - prev_error 
         
        # Clipping to ensure inputs are valid 
        error_clipped = np.clip(error, -10, 10) 
        delta_error_clipped = np.clip(delta_error, -5, 5) 
 
        # Simulate Fuzzy Logic Controller 
        pid_simulator.input['error'] = error_clipped 
 
        pid_simulator.input['delta_error'] = delta_error_clipped 
        pid_simulator.compute() 
 
        # Retrieve PID Gains 
        Kp = pid_simulator.output['kp'] 
        Ki = pid_simulator.output['ki'] 
        Kd = pid_simulator.output['kd'] 
 
        # PID control signal 
        integral += error 
        control_signal = Kp * error + Ki * integral + Kd * delta_error 
        response[i] = response[i - 1] + control_signal * (time_vector[1] - time_vector[0]) 
 
        # Update error 
        prev_error = error 
        error = setpoint - response[i] 
 
    return response 
 
# Time and Setpoint 
time = np.linspace(0, 50, 500) 
setpoint = 50 
 
# Simulate Response 
response = simulate_pid_response(setpoint, time) 
 
# Calculate Performance Metrics 
overshoot = (max(response) - setpoint) / setpoint * 100 
steady_state_error = abs(response[-1] - setpoint) 
rise_time = time[np.where(response >= 0.9 * setpoint)[0][0]] 
settling_time = time[np.where(abs(response - setpoint) > 0.05 * setpoint)[0][-1]] 
 
# Print Metrics 
print("Performance Metrics:") 
print(f"Overshoot (%): {overshoot:.2f}") 
print(f"Steady-State Error: {steady_state_error:.2f}") 
print(f"Rise Time (s): {rise_time:.2f}") 
print(f"Settling Time (s): {settling_time:.2f}") 
 
# Plot Response with Metrics 
plt.figure(figsize=(10, 6)) 
plt.plot(time, response, label="Fuzzy PID Response", linewidth=2) 
plt.axhline(y=setpoint, color='r', linestyle='--', label="Setpoint") 
plt.axhline(y=max(response), color='g', linestyle=':', label="Overshoot") 
plt.axvline(x=rise_time, color='orange', linestyle='--', label="Rise Time") 
plt.axvline(x=settling_time, color='purple', linestyle='--', label="Settling Time") 
plt.title("Fuzzy Logic PID Response with Metrics") 
plt.xlabel("Time (s)") 
plt.ylabel("Output") 
plt.legend() 
plt.grid() 
plt.show()