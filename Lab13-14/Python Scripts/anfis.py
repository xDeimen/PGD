import numpy as np 
import tensorflow as tf 
from tf.keras import Sequential 
from tf.keras.layers import Dense 
import matplotlib.pyplot as plt 
import skfuzzy as fuzz 
 
# Step 1: Define Membership Functions 
def gaussian_mf(x, mean, sigma): 
    return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2)) 
 
# Generate fuzzy membership functions for error and delta error 
x = np.linspace(-10, 10, 100) 
error_mf = { 
    "negative": gaussian_mf(x, -5, 2), 
    "zero": gaussian_mf(x, 0, 2), 
    "positive": gaussian_mf(x, 5, 2) 
} 
 
delta_error_mf = { 
    "negative": gaussian_mf(x, -5, 2), 
    "zero": gaussian_mf(x, 0, 2), 
    "positive": gaussian_mf(x, 5, 2) 
} 
 
# Step 2: Create Training Data 
def generate_training_data(): 
    np.random.seed(42) 
    training_data = np.random.uniform(-10, 10, (1000, 2))  # Inputs: error, delta_error 
    labels = np.array([ 
        [2 + 0.1 * e, 0.01 * abs(e), 0.2 - 0.01 * de]  # Heuristic rules for Kp, Ki, Kd 
        for e, de in training_data 
    ]) 
    return training_data, labels 
 
# Generate training data 
training_data, labels = generate_training_data() 
 
# Step 3: Build the Neural Network for ANFIS 
model = Sequential([ 
    Dense(16, input_dim=2, activation="relu"),  # Input: error, delta_error 
    Dense(16, activation="relu"), 
    Dense(3, activation="linear")  # Output: Kp, Ki, Kd 
]) 
 
model.compile(optimizer="adam", loss="mse") 
 
# Step 4: Train the ANFIS Model 
print("Training the ANFIS model...") 
model.fit(training_data, labels, epochs=50, batch_size=32) 
 
# Step 5: Simulate System Response Using ANFIS 
def simulate_anfis_pid(setpoint, time_vector, model): 
    response = np.zeros_like(time_vector) 
    error = setpoint - response[0] 
    prev_error = 0 
    integral = 0 
 
    for i in range(1, len(time_vector)): 
        delta_error = error - prev_error 
 
        # Predict PID gains using the trained model 
        Kp, Ki, Kd = model.predict(np.array([[error, delta_error]]))[0] 
 
        # Compute PID control signal 
        integral = np.clip(integral + error, -50, 50)  # Anti-windup 
        control_signal = Kp * error + Ki * integral + Kd * delta_error 
        control_signal = np.clip(control_signal, -100, 100)  # Saturation 
 
        # Update system response 
        response[i] = response[i - 1] + control_signal * (time_vector[1] - time_vector[0]) 
 
        # Update error 
        prev_error = error 
        error = setpoint - response[i] 
 
    return response 
 
# Step 6: Evaluate and Plot the Results 
time = np.linspace(0, 50, 500) 
setpoint = 50 
response = simulate_anfis_pid(setpoint, time, model) 
 
# Performance Metrics 
overshoot = (max(response) - setpoint) / setpoint * 100 
steady_state_error = abs(response[-1] - setpoint) 
rise_time = time[np.where(response >= 0.9 * setpoint)[0][0]] 
settling_time = time[np.where(abs(response - setpoint) > 0.05 * setpoint)[0][-1]] 
 
print("\nPerformance Metrics:") 
print(f"Overshoot (%): {overshoot:.2f}") 
print(f"Steady-State Error: {steady_state_error:.2f}") 
print(f"Rise Time (s): {rise_time:.2f}") 
print(f"Settling Time (s): {settling_time:.2f}") 
 
# Plot Response 
plt.figure(figsize=(10, 6)) 
plt.plot(time, response, label="ANFIS PID Response", linewidth=2) 
plt.axhline(y=setpoint, color='r', linestyle='--', label="Setpoint") 
plt.title("ANFIS-Based PID Controller Response") 
plt.xlabel("Time (s)") 
plt.ylabel("Output") 
plt.legend() 
plt.grid() 
plt.show()