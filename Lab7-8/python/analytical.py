import numpy as np 
import matplotlib.pyplot as plt 
# Time parameters 
time = np.linspace(0, 10, 1000)  # Time array from 0 to 10 seconds with 1000 points 
 
# Input voltage (V_GS) as a square wave: High (5V) for ON, Low (1V) for OFF 
V_GS = 5 * (time % 2 < 1) + 1 * (time % 2 >= 1)  # Alternates between 5V and 1V every second 
#V_GS = 3 * (time % 2 < 1) + 1 * (time % 2 >= 1)  # Alternates between 3V and 1V 
 
# Transistor properties 
V_th = 2.0  # Threshold voltage in volts for the MOSFET 
#V_th = 3.0  # Increase the threshold voltage to 3V 
#k = 0.5     # Constant for drain current calculation in the saturation region 
k = 0.3  # Adjust the constant for a different drain current response 
 
# Initialize an array to store the output current (I_D) 
I_D = np.zeros_like(V_GS) 
# Calculate I_D based on the transfer function 
for i in range(len(V_GS)): 
    if V_GS[i] < V_th: 
        I_D[i] = 0  # Transistor is OFF, no current flows 
    else: 
        I_D[i] = (k / 2) * (V_GS[i] - V_th)**2  # Transistor is ON, calculate current 
 
# Load resistance 
R_L = 1.0  # Ohms 
# Output voltage across the load resistor (V_out) 
V_out = I_D * R_L 
# Create a figure with multiple subplots 
plt.figure(figsize=(10, 8)) 
 
# Plot the input voltage (V_GS) 
plt.subplot(3, 1, 1) 
plt.plot(time, V_GS, label='Input Voltage (V_GS)', color='blue') 
plt.title('Input Voltage (V_GS)') 
plt.xlabel('Time (s)') 
plt.ylabel('V_GS (V)') 
plt.grid(True) 
plt.legend() 
 
# Plot the drain current (I_D) 
plt.subplot(3, 1, 2) 
plt.plot(time, I_D, label='Drain Current (I_D)', color='red') 
plt.title('Drain Current (I_D)') 
plt.xlabel('Time (s)') 
plt.ylabel('I_D (A)') 
plt.grid(True) 
plt.legend() 
 
# Plot the output voltage (V_out) 
plt.subplot(3, 1, 3) 
plt.plot(time, V_out, label='Output Voltage (V_out)', color='green') 
plt.title('Output Voltage (V_out)') 
plt.xlabel('Time (s)') 
plt.ylabel('V_out (V)') 
plt.grid(True) 
plt.legend() 
 
# Display the plots 
plt.tight_layout() 
plt.show() 