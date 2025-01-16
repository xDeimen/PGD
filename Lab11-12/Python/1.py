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
 
# Sensor parameters 
K_s = 1.0   # Sensor gain 
 
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
 
# Transfer function for the sensor 
G_sensor = ctrl.TransferFunction([K_s], [1]) 
 
# Open-loop transfer function 
G_open = G_pid * G_motor * G_sensor 
 
# Closed-loop transfer function 
G_closed = ctrl.feedback(G_open, 1) 
 
# Disturbance transfer function 
G_disturbance = G_motor / (1 + G_open) 
 
# Simulation: Step response for closed-loop system 
time = np.linspace(0, 2, 500)  # 0 to 2 seconds, 500 points 
time_out, response_closed = ctrl.step_response(G_closed, time) 
 
# Simulation: Response to sinusoidal disturbance 
disturbance_input = A * np.sin(omega * time) 
time_out, response_disturbance = ctrl.forced_response(G_disturbance, time, disturbance_input) 
 
# Plot the results 
plt.figure(figsize=(10, 6)) 
 
# Plot closed-loop step response 
plt.subplot(2, 1, 1) 
plt.plot(time_out, response_closed, label='Step Response (Closed Loop)') 
plt.title('Robot Axis System Responses') 
plt.xlabel('Time (s)') 
plt.ylabel('Position') 
plt.legend() 
plt.grid() 
 
# Plot disturbance response 
plt.subplot(2, 1, 2) 
plt.plot(time, disturbance_input, label='Disturbance Input (Torque)', linestyle='dashed') 
plt.plot(time_out, response_disturbance, label='Disturbance Response') 
plt.xlabel('Time (s)') 
plt.ylabel('Response') 
plt.legend() 
plt.grid() 
 
plt.tight_layout() 
plt.show()