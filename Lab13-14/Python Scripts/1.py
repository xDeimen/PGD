import numpy as np 
import matplotlib.pyplot as plt 
 
def heating_system_model(time, K, tau, power_input): 
    """ 
    Simulate the temperature response of the heating system. 
 
    Parameters: 
    time (array): Time vector in seconds. 
    K (float): System gain (°C/W).  tau (float): Time constant in seconds. 
    power_input (float): Power input in watts. 
 
    Returns: 
    array: Temperature response over time. 
    """ 
    return K * power_input * (1 - np.exp(-time / tau))

time = np.linspace(0, 50, 500)
K = 5 
tau = 10 
power_inputs = [5, 10, 15]
 

responses = [heating_system_model(time, K, tau, P) for P in power_inputs] 
 
plt.figure(figsize=(10, 6)) 
for power, response in zip(power_inputs, responses): 
    plt.plot(time, response, label=f'Power Input = {power}W') 
 
# Adding plot details 
plt.title('Digital Twin of Heating System', fontsize=16) 
plt.xlabel('Time (seconds)', fontsize=14) 
plt.ylabel('Temperature (°C)', fontsize=14) 
plt.axhline(y=K * max(power_inputs), color='r', linestyle='--', label='Steady-State Limit') 
plt.legend(fontsize=12) 
plt.grid(True, linestyle='--', alpha=0.7) 
plt.tight_layout() 
plt.show() 

