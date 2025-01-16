import numpy as np 
import matplotlib.pyplot as plt 
 
# Redefining the second-order model function 
def second_order_model(time, K, tau, zeta, power_input): 
    """ 
    Simulate the temperature response of a second-order heating system. 
 
    Parameters: 
    time (array): Time vector in seconds. 
    K (float): System gain (°C/W). 
    tau (float): Time constant (seconds). 
    zeta (float): Damping ratio. 
    power_input (float): Power input in watts. 
 
    Returns: 
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
 
# Parameters for the second-order model simulation with varying settings 
time = np.linspace(0, 50, 500)  # Time vector (0 to 50 seconds) 
K_values = [3, 5, 7]  # Different system gains (°C/W) 
tau_values = [5, 10, 15]  # Different time constants (seconds) 
zeta = 0.7  # Fixed damping ratio for this simulation 
power_input = 10  # Power input (W) 
 
# Function to simulate responses for different K and tau 
def simulate_responses(K_values, tau_values, zeta, power_input, time): 
    responses = [] 
    labels = [] 
    for K in K_values: 
        for tau in tau_values: 
            response = second_order_model(time, K, tau, zeta, power_input) 
            responses.append(response) 
            labels.append(f"K={K}, τ={tau}") 
    return responses, labels 
 
# Run the simulation 
responses, labels = simulate_responses(K_values, tau_values, zeta, power_input, time) 
 
# Plotting the results 
plt.figure(figsize=(12, 8)) 
for response, label in zip(responses, labels): 
    plt.plot(time, response, label=label)

# Add plot details 
plt.title('Second-Order Model: Effects of K and τ', fontsize=16) 
plt.xlabel('Time (seconds)', fontsize=14) 
plt.ylabel('Temperature (°C)', fontsize=14) 
plt.axhline(y=max(K_values) * power_input, color='r', linestyle='--', label='Maximum Steady-State Value') 
plt.grid(True, linestyle='--', alpha=0.7) 
plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left') 
plt.tight_layout() 
plt.show()