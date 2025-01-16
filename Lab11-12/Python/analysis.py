import numpy as np 
 
import matplotlib.pyplot as plt 
import control as ctrl 
 
# System Parameters 
K_t = 0.1   # Torque constant 
K_b = 0.1   # Back EMF constant 
R_a = 1.0   # Armature resistance 
L_a = 0.01  # Armature inductance 
J_m = 0.01  # Motor inertia 
b_m = 0.001 # Motor damping 
N = 10      # Gear ratio 
J_a = 0.05  # Arm inertia 
b_a = 0.01  # Arm damping 
K_p = 10.0  # Proportional gain 
K_i = 1.0   # Integral gain 
K_d = 0.5   # Derivative gain 
A = 0.1     # Disturbance amplitude 
omega = 2.0 # Disturbance frequency 
 
# Effective inertia 
J_eff = J_m + J_a * N**2 
 
# Motor + Gearbox Transfer Function 
numerator_motor = [K_t / N] 
denominator_motor = [L_a * J_eff, R_a * J_eff + L_a * b_m, R_a * b_m + K_t * K_b] 
G_motor = ctrl.TransferFunction(numerator_motor, denominator_motor) 
 
# PID Controller Transfer Function (made proper) 
epsilon = 0.01  # Small time constant 
numerator_pid = [K_d, K_p, K_i] 
denominator_pid = [epsilon, 1, 0] 
G_pid = ctrl.TransferFunction(numerator_pid, denominator_pid) 
 
G_open = G_pid * G_motor 
G_closed = ctrl.feedback(G_open, 1) 
 
# Simulation Setup 
time = np.linspace(0, 5, 500)  # 0 to 5 seconds, 500 points 
 
# 1. Step Response 
time_step, response_step = ctrl.step_response(G_closed, time) 
 
# 2. Ramp Input Response 
ramp_input = time  # Linearly increasing input 
time_ramp, response_ramp = ctrl.forced_response(G_closed, time, ramp_input) 
 
# 3. Control Signal (Motor Input) 
step_input = np.ones_like(time)  # Step input of magnitude 1 
_, control_signal = ctrl.forced_response(G_pid, time, step_input) 
 
# 4. Disturbance Torque Response 
disturbance_torque = A * np.sin(omega * time) 
G_disturbance = G_motor / (1 + G_open) 
time_dist, response_disturbance = ctrl.forced_response(G_disturbance, time, disturbance_torque) 
 
# 5. Velocity Response (Derivative of Position) 
velocity_response = np.gradient(response_step, time) 
 
# Plotting Results 
plt.figure(figsize=(14, 10)) 
 
# Plot Step Response 
plt.subplot(3, 1, 1) 
plt.plot(time_step, response_step, label='Position Response', color='b') 
plt.title('System Position and Velocity Responses') 
plt.xlabel('Time (s)') 
plt.ylabel('Position') 
plt.legend() 
plt.grid() 
 
# Plot Velocity Response 
plt.subplot(3, 1, 2) 
plt.plot(time_step, velocity_response, label='Velocity Response', color='g') 
plt.xlabel('Time (s)') 
plt.ylabel('Velocity') 
plt.legend() 
plt.grid() 
 
# Plot Control Signal 
plt.subplot(3, 1, 3) 
plt.plot(time, control_signal, label='Control Signal (Motor Input)', color='r') 
plt.title('Control Signal') 
plt.xlabel('Time (s)') 
plt.ylabel('Motor Input') 
plt.legend() 
plt.grid() 
 
plt.tight_layout() 
plt.show() 
 
# Plot Disturbance Torque Effect 
plt.figure(figsize=(10, 5)) 
plt.plot(time, disturbance_torque, label='Disturbance Torque (Input)', linestyle='dashed', color='r') 
plt.plot(time_dist, response_disturbance, label='Disturbance Response (Position)', color='b') 
plt.title('Effect of Disturbance Torque') 
plt.xlabel('Time (s)') 
plt.ylabel('Response') 
plt.legend() 
plt.grid() 
plt.show()