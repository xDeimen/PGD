import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import lti, step 
 
# Define the second-order heating system with PID control 
def pid_controller_response(Kp, Ki, Kd, K, tau, zeta, power_input, setpoint, time): 
    """ 
    Simulate the system response with PID control for a second-order heating system. 
     
    Parameters: 
    Kp (float): Proportional gain. 
    Ki (float): Integral gain. 
    Kd (float): Derivative gain. 
    K (float): System gain (°C/W). 
    tau (float): Time constant (seconds). 
    zeta (float): Damping ratio. 
    power_input (float): Power input in watts. 
    setpoint (float): Desired temperature (°C). 
    time (array): Time vector (seconds). 
 
    Returns: 
    time (array), response (array): Time and system response (temperature in °C). 
    """ 
    # System parameters 
    omega_n = 1 / tau  # Natural frequency 
    num = [K * omega_n**2]  # Numerator of the second-order system 
    den = [1, 2 * zeta * omega_n, omega_n**2]  # Denominator of the second-order system 
 
    # Define the PID controller transfer function 
    pid_num = [Kd, Kp, Ki]  # PID numerator (Kd * s^2 + Kp * s + Ki) 
    pid_den = [1, 0]  # PID denominator (accounts for integral action)

    # Combine the system and PID transfer function in a closed-loop feedback 
    system = lti(num, den)  # Second-order system 
    pid = lti(pid_num, pid_den)  # PID controller 
    closed_loop = lti( 
        np.polymul(pid.num, system.num), 
        np.polyadd(np.polymul(pid.num, system.den), np.polymul(pid.den, system.num)) 
    ) 
 
    # Step response of the closed-loop system to the setpoint 
    time, response = step(closed_loop, T=time) 
    return time, setpoint * response 
 
# Simulation parameters 
time = np.linspace(0, 50, 500)  # Time vector (seconds) 
K, tau, zeta = 5, 10, 0.7  # System gain, time constant, damping ratio 
setpoint = 50  # Desired temperature (°C) 
power_input = 10  # Power input (W) 
 
# Initial PID parameters 

def ziegler(Ku, Tu):
    Kp = 0.6*Ku
    Ki = (1.2*Ku)/Tu
    Kd = 0.075*Ku*Tu

    return Kp, Ki, Kd

Kp, Ki, Kd = 0.8, 0.01, 0.001
 
# Simulate the response with initial PID values 
time, response = pid_controller_response(Kp, Ki, Kd, K, tau, zeta, power_input, setpoint, time) 
 
# Plotting the initial response 
plt.figure(figsize=(10, 6)) 
plt.plot(time, response, label=f'Initial PID (Kp={Kp}, Ki={Ki}, Kd={Kd})', linewidth=2) 
plt.axhline(y=setpoint, color='r', linestyle='--', label='Setpoint = 50°C') 
plt.title('PID Tuning by Trial-and-Error: Initial Response', fontsize=16) 
plt.xlabel('Time (seconds)', fontsize=14) 
plt.ylabel('Temperature (°C)', fontsize=14) 
plt.grid(True, linestyle='--', alpha=0.7) 
plt.legend(fontsize=12) 
plt.tight_layout() 
plt.show() 
 
# Students can iteratively adjust Kp, Ki, and Kd to minimize overshoot, settling time, and steady-state error.