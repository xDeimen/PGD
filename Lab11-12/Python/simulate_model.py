import numpy as np 
import matplotlib.pyplot as plt 
import control as ctrl 
 
# Define system parameters 
# Motor parameters 
K_t = 0.1   # Torque constant 
K_b = 0.1   # Back EMF constant 
R_a = 1.0   # Armature resistance 
L_a = 0.01  # Armature inductance 
J_m = 0.01  # Motor inertia 
b_m = 0.001 # Motor damping 
 
# Gearbox parameters 
N = 10      # Gear ratio 
J_a = 0.05  # Arm inertia (robot arm) 
b_a = 0.01  # Arm damping 
 
# PID Controller parameters 
K_p = 10.0  # Proportional gain 
K_i = 1.0   # Integral gain 
K_d = 0.5   # Derivative gain 
 
# Disturbance torque parameters 
A = 0.1     # Amplitude of sinusoidal disturbance 
omega = 2.0 # Frequency of disturbance 
 
# Effective inertia 
J_eff = J_m + J_a * N**2 
 
# Transfer function for the motor + gearbox 
numerator_motor = [K_t / N] 
denominator_motor = [ 
    L_a * J_eff, 
    R_a * J_eff + L_a * b_m, 
    R_a * b_m + K_t * K_b 
] 
G_motor = ctrl.TransferFunction(numerator_motor, denominator_motor) 
 
# Transfer function for the PID controller 
numerator_pid = [K_d, K_p, K_i] 
 
denominator_pid = [1, 0] 
G_pid = ctrl.TransferFunction(numerator_pid, denominator_pid) 
 
# Open-loop transfer function 
G_open = G_pid * G_motor 
 
# Closed-loop transfer function 
G_closed = ctrl.feedback(G_open, 1) 
 
# Simulation setup 
time = np.linspace(0, 5, 500)  # 0 to 5 seconds, 500 points 
 
# Step Input 
time_step, response_step = ctrl.step_response(G_closed, time) 
 
# Ramp Input 
ramp_input = time  # Linearly increasing input 
time_ramp, response_ramp = ctrl.forced_response(G_closed, time, ramp_input) 
 
# Sinusoidal Disturbance Torque 
disturbance_torque = A * np.sin(omega * time) 
G_disturbance = G_motor / (1 + G_open) 
time_dist, response_disturbance = ctrl.forced_response(G_disturbance, time, disturbance_torque) 
 
# PID Parameter Changes (Example: Increase K_p) 
K_p_new = 20.0  # New proportional gain 
G_pid_new = ctrl.TransferFunction([K_d, K_p_new, K_i], [1, 0]) 
G_open_new = G_pid_new * G_motor 
G_closed_new = ctrl.feedback(G_open_new, 1) 
time_pid, response_pid = ctrl.step_response(G_closed_new, time) 
 
# Plotting the results 
plt.figure(figsize=(12, 10)) 
 
# Step Response 
plt.subplot(2, 2, 1) 
plt.plot(time_step, response_step, label='Step Response') 
plt.title('Step Response') 
plt.xlabel('Time (s)') 
plt.ylabel('Position') 
plt.legend() 
plt.grid() 
 
# Ramp Response 
plt.subplot(2, 2, 2) 
plt.plot(time, ramp_input, label='Ramp Input', linestyle='dashed', color='r') 
plt.plot(time_ramp, response_ramp, label='Ramp Response') 
plt.title('Ramp Input and Response') 
plt.xlabel('Time (s)') 
plt.ylabel('Response') 
plt.legend() 
plt.grid() 
 
# Disturbance Response 
plt.subplot(2, 2, 3) 
plt.plot(time, disturbance_torque, label='Disturbance Torque', linestyle='dashed', color='r') 
plt.plot(time_dist, response_disturbance, label='Disturbance Response') 
plt.title('Response to Disturbance Torque') 
plt.xlabel('Time (s)') 
plt.ylabel('Response') 
plt.legend() 
plt.grid() 
 
 
# Step Response with Modified PID 
plt.subplot(2, 2, 4) 
plt.plot(time_step, response_step, label='Original PID', linestyle='dashed') 
plt.plot(time_pid, response_pid, label='Modified PID (Kp=20)') 
plt.title('Impact of PID Parameter Changes') 
plt.xlabel('Time (s)') 
plt.ylabel('Response') 
plt.legend() 
plt.grid() 
 
plt.tight_layout() 
plt.show()