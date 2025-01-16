import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit 
from sklearn.metrics import mean_squared_error 
from sklearn.neural_network import MLPRegressor 
from sklearn.model_selection import train_test_split 
 
# Simulate realistic battery degradation data 
C_0 = 3.0  # Initial battery capacity 
cycles = 1000 
 
# Generate a dataset with realistic capacity degradation 
data = { 
    "cycle": range(1, cycles + 1), 
    "temperature": [25 + (0.2 * (i % 50)) for i in range(cycles)],  # Temperature variation 
    "voltage": [4.2 - (0.01 * (i % 30)) for i in range(cycles)],  # Voltage drop pattern 
    "current": [2.0 + (0.03 * (i % 10)) for i in range(cycles)],  # Current variation 
    "discharge_time": [60 - (0.1 * (i % 100)) for i in range(cycles)],  # Variable discharge time 
    "state_of_charge": [80 - (0.05 * i) for i in range(cycles)],  # Gradual decline in state of charge 
} 
 
# Simulate realistic capacity degradation 
data["capacity"] = [ 
    max(0, C_0 - (0.001 * i) - (0.00005 * i**1.5)) for i in range(cycles) 
] 
 
# Add random noise to make it realistic 
np.random.seed(42) 
data["capacity"] += np.random.normal(0, 0.01, size=cycles) 
 
# Convert to a DataFrame 
df = pd.DataFrame(data) 
 
# Define the base model (quadratic non-linear model) 
def base_model(cycle, k1, k2): 
    return C_0 - (k1 * cycle) - (k2 * cycle**2) 
 
# Optimize the base model parameters 
popt, _ = curve_fit(base_model, df["cycle"], df["capacity"]) 
k1_opt, k2_opt = popt 
print(f"Optimized Base Model Coefficients: k1 = {k1_opt}, k2 = {k2_opt}") 
 
# Compute the base model predictions 
df["base_capacity"] = base_model(df["cycle"], k1_opt, k2_opt) 
 
# Calculate residuals (difference between measured and base model) 
df["residual"] = df["capacity"] - df["base_capacity"] 
 
# Train a neural network to predict residuals 
features = ["cycle", "temperature", "discharge_time", "voltage", "state_of_charge"] 
X = df[features] 
y = df["residual"] 
 
# Train-test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
 
# Train a neural network 
nn_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42) 
nn_model.fit(X_train, y_train) 
 
# Predict residuals using the neural network 
df["predicted_residual"] = nn_model.predict(X) 
 
# Smooth predicted residuals to reduce oscillations 
df["predicted_residual_smoothed"] = df["predicted_residual"].rolling(window=5, min_periods=1).mean() 
 
# Compute the final hybrid model predictions (smoothed residuals) 
df["hybrid_capacity"] = df["base_capacity"] + df["predicted_residual"] 
df["hybrid_capacity_smoothed"] = df["base_capacity"] + df["predicted_residual_smoothed"] 
 
# Evaluate the hybrid model 
mse_hybrid = mean_squared_error(df["capacity"], df["hybrid_capacity_smoothed"]) 
print(f"Smoothed Hybrid Model MSE: {mse_hybrid}") 
 
# Plot the results 
plt.figure(figsize=(12, 6)) 
 
# Measured capacity 
plt.plot( 
    df["cycle"], df["capacity"], label="Measured Capacity", linestyle="--", color="blue" 
) 
 
# Base model 
plt.plot( 
 
    df["cycle"], 
    df["base_capacity"], 
    label="Base Model (Non-linear)", 
    linestyle=":", 
    color="orange", 
) 
 
# Hybrid model 
plt.plot( 
    df["cycle"], 
    df["hybrid_capacity_smoothed"], 
    label="Smoothed Hybrid Model", 
    linestyle="-", 
    color="green", 
) 
 
# Add labels, title, and legend 
plt.xlabel("Cycle Number") 
plt.ylabel("Capacity (Ah)") 
plt.title("Comparison of Measured, Base, and Smoothed Hybrid Models") 
plt.legend() 
plt.grid(True) 
plt.show() 
 
# Display correlation matrix 
print("Correlation Matrix:") 
print(df[["temperature", "discharge_time", "capacity", "base_capacity", "hybrid_capacity_smoothed"]].corr()) 