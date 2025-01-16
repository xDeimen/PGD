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
N_values = [5, 20, 50]  # Different Gear Ratios to Test
J_a = 0.05  # Arm inertia
b_a = 0.01  # Arm damping

# Tuned PID gains for better performance
K_p = 20.0
K_i = 5.0
K_d = 2.0

# Random impulse disturbance parameters
impulse_amplitude = 0.2
impulse_intervals = np.linspace(0.5, 4.5, 8)  # Times of impulses

# Time vector
time = np.linspace(0, 5, 500)  # 0 to 5 seconds, 500 points

for N in N_values:
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

    # Open and Closed Loop Systems
    G_open = G_pid * G_motor
    G_closed = ctrl.feedback(G_open, 1)

    # Step Response
    time_step, response_step = ctrl.step_response(G_closed, time)

    # Random Impulse Disturbance
    disturbance_torque = np.zeros_like(time)
    for t_impulse in impulse_intervals:
        idx = (np.abs(time - t_impulse)).argmin()
        disturbance_torque[idx] = impulse_amplitude

    G_disturbance = G_motor / (1 + G_open)
    time_dist, response_disturbance = ctrl.forced_response(G_disturbance, time, disturbance_torque)

    # Velocity Response (Derivative of Position)
    velocity_response = np.gradient(response_step, time)

    # Plotting Results for Current Gear Ratio
    plt.figure(figsize=(14, 10))

    # Plot Step Response
    plt.subplot(3, 1, 1)
    plt.plot(time_step, response_step, label=f'N={N}', color='b')
    plt.title(f'System Position and Velocity Responses (Gear Ratio N={N})')
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

    # Plot Disturbance Effect
    plt.subplot(3, 1, 3)
    plt.plot(time, disturbance_torque, label='Disturbance Torque (Input)', linestyle='dashed', color='r')
    plt.plot(time_dist, response_disturbance, label='Disturbance Response (Position)', color='b')
    plt.title('Effect of Random Impulse Disturbance')
    plt.xlabel('Time (s)')
    plt.ylabel('Response')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# Observations and Analysis
# Tuned PID gains (K_p=20.0, K_i=5.0, K_d=2.0) to reduce overshoot and improve settling time.
