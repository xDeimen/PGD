import numpy as np 
import matplotlib.pyplot as plt 
 
# Define the second-order system model 
def second_order_model(time, K, tau, zeta, power_input): 
    """ 
    Simulate the temperature response of a second-order heating system. 
 
    Parameters: 
    time (array): Time vector in seconds. 
    K (float): System gain (°C/W). 
    tau (float): Time constant (seconds). 
    zeta (float): Damping ratio. 
    power_input (float): Power input in watts.Returns: 
    array: Temperature response over time. 
    """ 
    omega_n = 1 / tau  # Natural frequency (rad/s) 
    omega_d = omega_n * np.sqrt(1 - zeta**2) if zeta < 1 else 0  # Damped frequency 
    response = np.zeros_like(time) 
     
    for i, t in enumerate(time): 
        if zeta < 1:  # Underdamped case 
            response[i] = K * power_input * (1 - np.exp(-zeta * omega_n * t) * ( 
                np.cos(omega_d * t) + (zeta / np.sqrt(1 - zeta**2)) * np.sin(omega_d * t) 
            )) 
        elif zeta == 1:  # Critically damped case 
            response[i] = K * power_input * (1 - (1 + omega_n * t) * np.exp(-omega_n * t)) 
        else:  # Overdamped case 
            r1 = -omega_n * (zeta - np.sqrt(zeta**2 - 1)) 
            r2 = -omega_n * (zeta + np.sqrt(zeta**2 - 1)) 
            response[i] = K * power_input * (1 - (np.exp(r1 * t) - np.exp(r2 * t)) / (r1 - r2)) 
    return response 
 
# Simulation parameters 
time = np.linspace(0, 50, 500)  # Time vector (0 to 50 seconds) 
K = 5  # System gain (°C/W) 
tau = 10  # Time constant (seconds) 
zeta_values = [0.5, 1.0, 1.5]  # Damping ratios for underdamped, critically damped, and overdamped cases 
power_input = 10  # Power input (W) 
 
# Simulate responses for different damping ratios 
responses = [second_order_model(time, K, tau, zeta, power_input) for zeta in zeta_values] 
 
# Plot the results 
plt.figure(figsize=(10, 6)) 
for zeta, response in zip(zeta_values, responses): 
    label = f"ζ = {zeta} (Underdamped)" if zeta < 1 else ( 
        f"ζ = {zeta} (Critically Damped)" if zeta == 1 else f"ζ = {zeta} (Overdamped)" 
    ) 
    plt.plot(time, response, label=label) 
 
# Add plot details 
plt.title('Second-Order Model of Heating System', fontsize=16) 
plt.xlabel('Time (seconds)', fontsize=14) 
plt.ylabel('Temperature (°C)', fontsize=14) 
plt.axhline(y=K * power_input, color='r', linestyle='--', label='Steady-State Value') 
plt.grid(True, linestyle='--', alpha=0.7) 
plt.legend(fontsize=12) 
plt.tight_layout() 
plt.show()