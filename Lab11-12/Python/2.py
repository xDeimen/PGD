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
 
# Simulation: Step response for closed-loop system 
time = np.linspace(0, 2, 500)  # 0 to 2 seconds, 500 points 
time_out, response_closed = ctrl.step_response(G_closed, time) 
 
# Plot the step input and response 
plt.figure(figsize=(8, 5))  
plt.plot(time_out, response_closed, label='Step Response (Closed Loop)', color='b') 
plt.axhline(y=1, color='r', linestyle='--', label='Step Input') 
plt.title('Step Input and Step Response') 
plt.xlabel('Time (s)') 
plt.ylabel('Response (Position)') 
plt.legend() 
plt.grid() 
plt.show() 
 
time_out, response_impulse = ctrl.impulse_response(G_closed, time) 
 
plt.figure(figsize=(8, 5)) 
plt.plot(time_out, response_impulse, label='Impulse Response', color='g') 
plt.title('Impulse Input and System Response') 
plt.xlabel('Time (s)') 
plt.ylabel('Response') 
plt.legend() 
plt.grid() 
plt.show() 
 
# Ramp input signal 
ramp_input = time  # Linearly increasing input with time 
time_out, response_ramp = ctrl.forced_response(G_closed, time, ramp_input) 
 
# Plot ramp input and response 
plt.figure(figsize=(8, 5)) 
plt.plot(time, ramp_input, label='Ramp Input', linestyle='dashed', color='r') 
plt.plot(time_out, response_ramp, label='Ramp Response', color='b') 
plt.title('Ramp Input and System Response') 
plt.xlabel('Time (s)') 
plt.ylabel('Response') 
plt.legend() 
plt.grid() 
plt.show() 
 
# Sinusoidal input signal 
freq = 1.0  # Frequency of the sine wave (Hz) 
sine_input = np.sin(2 * np.pi * freq * time)  # Sinusoidal input 
time_out, response_sine = ctrl.forced_response(G_closed, time, sine_input) 
 
# Plot sinusoidal input and response 
plt.figure(figsize=(8, 5)) 
plt.plot(time, sine_input, label='Sinusoidal Input', linestyle='dashed', color='r') 
plt.plot(time_out, response_sine, label='Sinusoidal Response', color='b') 
plt.title('Sinusoidal Input and System Response') 
plt.xlabel('Time (s)') 
plt.ylabel('Response') 
plt.legend() 
plt.grid() 
plt.show() 
 
# Exponential input signal 
exponential_input = np.exp(-time)  # Exponentially decaying input 
time_out, response_exponential = ctrl.forced_response(G_closed, time, exponential_input) 
 
# Plot exponential input and response 
plt.figure(figsize=(8, 5)) 
plt.plot(time, exponential_input, label='Exponential Input', linestyle='dashed', color='r') 
plt.plot(time_out, response_exponential, label='Exponential Response', color='b') 
plt.title('Exponential Input and System Response') 
plt.xlabel('Time (s)') 
plt.ylabel('Response') 
 
plt.legend() 
plt.grid() 
plt.show() 
 
from scipy.signal import square 
 
# Square wave input signal 
freq = 1.0  # Frequency of the square wave (Hz) 
square_input = square(2 * np.pi * freq * time)  # Square wave signal 
time_out, response_square = ctrl.forced_response(G_closed, time, square_input) 
 
# Plot square wave input and response 
plt.figure(figsize=(8, 5)) 
plt.plot(time, square_input, label='Square Wave Input', linestyle='dashed', color='r') 
plt.plot(time_out, response_square, label='Square Wave Response', color='b') 
plt.title('Square Wave Input and System Response') 
plt.xlabel('Time (s)') 
plt.ylabel('Response') 
plt.legend() 
plt.grid() 
plt.show()