import pandas as pd 
import matplotlib.pyplot as plt 
 
# Load the dataset 
df = pd.read_csv(r"C:\Masters\Y1\Sem1\PGD\Lab9-10\data\li_ion_battery_last_data.csv")  # Replace with actual path or file name 
print(df.head())

plt.figure(figsize=(10, 6)) 
plt.plot(df['cycle'], df['capacity'], label='Measured Capacity') 
plt.xlabel('Cycle Number') 
plt.ylabel('Capacity (Ah)') 
plt.title('Battery Capacity vs. Cycle Number') 
plt.legend() 
plt.show()
 
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

import matplotlib.pyplot as plt 
 
plt.figure(figsize=(10, 6)) 
plt.plot(df['cycle'], df['capacity'], label='Battery Capacity (Ah)', color='blue') 
plt.xlabel('Cycle Number') 
plt.ylabel('Capacity (Ah)') 
plt.title('Battery Capacity Over Charge-Discharge Cycles') 
plt.grid(True) 
plt.legend() 
plt.show() 


plt.figure(figsize=(10, 6)) 
plt.scatter(df['temperature'], df['capacity'], alpha=0.5, c='red') 
plt.xlabel('Temperature (°C)') 
plt.ylabel('Capacity (Ah)') 
plt.title('Impact of Temperature on Battery Capacity') 
plt.grid(True) 
plt.show() 
 
correlation_matrix = df.corr() 
print(correlation_matrix) 
 
import seaborn as sns 
 
plt.figure(figsize=(8, 6)) 
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0) 
plt.title('Correlation Matrix of Battery Parameters') 
plt.show()

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 
 
# Define input features (X) and target variable (y) 
X = df[['cycle', 'temperature', 'discharge_time']]  # Input features 
y = df['capacity']  # Target variable: battery capacity 
 
# Split the data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
 
# Create and train the Linear Regression model 
model = LinearRegression() 
model.fit(X_train, y_train) 
 
# Make predictions 
y_pred = model.predict(X_test) 
 
# Evaluate the model 
mse = mean_squared_error(y_test, y_pred) 
r2 = r2_score(y_test, y_pred) 
 
print(f"Mean Squared Error: {mse}") 
print(f"R^2 Score: {r2}")

from sklearn.tree import DecisionTreeRegressor 
 
# Create and train the Decision Tree model 
dt_model = DecisionTreeRegressor(max_depth=5)  # Limit the depth to avoid overfitting 
dt_model.fit(X_train, y_train) 
 
# Make predictions 
y_pred_dt = dt_model.predict(X_test) 
 
# Evaluate the model 
mse_dt = mean_squared_error(y_test, y_pred_dt) 
r2_dt = r2_score(y_test, y_pred_dt) 
 
print(f"Decision Tree Mean Squared Error: {mse_dt}") 
print(f"Decision Tree R^2 Score: {r2_dt}")


from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 
 
# Define input features (X) and target variable (y) 
X = df[['cycle', 'temperature', 'discharge_time']]  # Input features 
y = df['capacity']  # Target variable: battery capacity 
 
# Split the data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
 
# Create and train the Linear Regression model 
model = LinearRegression() 
model.fit(X_train, y_train) 
 
# Make predictions 
y_pred = model.predict(X_test) 
 
# Evaluate the model 
mse = mean_squared_error(y_test, y_pred) 
r2 = r2_score(y_test, y_pred) 
 
print(f"Mean Squared Error: {mse}") 
print(f"R^2 Score: {r2}")

from sklearn.tree import DecisionTreeRegressor 
 
# Create and train the Decision Tree model 
dt_model = DecisionTreeRegressor(max_depth=5)  # Limit the depth to avoid overfitting 
dt_model.fit(X_train, y_train) 
 
# Make predictions 
y_pred_dt = dt_model.predict(X_test) 
 
# Evaluate the model 
mse_dt = mean_squared_error(y_test, y_pred_dt) 
r2_dt = r2_score(y_test, y_pred_dt) 
 
print(f"Decision Tree Mean Squared Error: {mse_dt}") 
print(f"Decision Tree R^2 Score: {r2_dt}")