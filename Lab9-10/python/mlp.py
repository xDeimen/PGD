import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split 
from sklearn.neural_network import MLPRegressor 
from sklearn.preprocessing import MinMaxScaler 
from scipy.optimize import curve_fit 
 
# Load the dataset 
file_path = r'C:\Masters\Y1\Sem1\PGD\Lab9-10\data\li_ion_battery_last_data.csv' 
df = pd.read_csv(file_path) 
 
# Define the base model function 
def base_model(cycle, k1, k2): 
    """ 
    Quadratic base model for capacity degradation. 
    """ 
    C_0 = df['capacity'].iloc[0]  # Initial capacity 
    return C_0 - k1 * cycle - k2 * (cycle**2) 
 
# Fit the base model to find k1 and k2 
def fit_base_model(df): 
    """ 
    Fit the base model to the experimental data. 
    """ 
    popt, _ = curve_fit(base_model, df['cycle'], df['capacity']) 
 
    k1, k2 = popt 
    print(f"Fitted Parameters: k1 = {k1:.6f}, k2 = {k2:.6f}") 
    return k1, k2 
 
# Apply the base model 
def apply_base_model(df, k1, k2): 
    """ 
    Apply the base model to estimate capacity. 
    """ 
    df['base_capacity'] = base_model(df['cycle'], k1, k2) 
    df['residual'] = df['capacity'] - df['base_capacity'] 
    return df 
 
# Train Neural Network on residuals 
def train_nn_for_residuals(df, variables): 
    """ 
    Train a neural network to predict the residuals based on input variables. 
    """ 
    # Normalize features and target 
    scaler_X = MinMaxScaler() 
    scaler_y = MinMaxScaler() 
 
    X = df[variables] 
    y = df['residual'].values.reshape(-1, 1) 
 
    X_scaled = scaler_X.fit_transform(X) 
    y_scaled = scaler_y.fit_transform(y) 
 
    # Train-test split 
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42) 
 
    # Train the neural network 
    model = MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=1000, random_state=42) 
    model.fit(X_train, y_train.ravel()) 
 
    # Predict and reverse scaling 
    y_pred_scaled = model.predict(X_test) 
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)) 
    y_test = scaler_y.inverse_transform(y_test) 
 
    # Evaluate the model 
    mse = mean_squared_error(y_test, y_pred) 
    print(f"Neural Network (Residuals) Mean Squared Error (MSE): {mse:.4f}") 
 
    return model, scaler_X, scaler_y 
 
# Combine base model and neural network for hybrid predictions 
def apply_hybrid_model(df, k1, k2, nn_model, scaler_X, scaler_y, variables): 
    """ 
    Combine the base model and neural network predictions to form the hybrid model. 
    """ 
    # Apply the base model 
    df['base_capacity'] = base_model(df['cycle'], k1, k2) 
 
    # Predict residuals using the neural network 
    X_scaled = scaler_X.transform(df[variables]) 
    residual_pred_scaled = nn_model.predict(X_scaled) 
    residual_pred = scaler_y.inverse_transform(residual_pred_scaled.reshape(-1, 1)) 
 
    # Calculate hybrid capacity 
    df['predicted_residual'] = residual_pred[:, 0]  # Convert to 1D array 
    df['hybrid_capacity'] = df['base_capacity'] + df['predicted_residual'] 
 
    return df 
 
# Plot comparison of models 
def plot_comparison(df): 
    """ 
    Plot the comparison of experimental, base, and hybrid capacities. 
    """ 
    plt.figure(figsize=(12, 6)) 
    plt.plot(df['cycle'], df['capacity'], label='Experimental Capacity', linestyle='--', color='blue') 
    plt.plot(df['cycle'], df['base_capacity'], label='Base Model Capacity', linestyle='-', color='orange') 
    plt.plot(df['cycle'], df['hybrid_capacity'], label='Hybrid Model Capacity', linestyle='-', color='green') 
    plt.title("Comparison of Experimental, Base Model, and Hybrid Model Capacities") 
    plt.xlabel("Cycle") 
    plt.ylabel("Capacity (Ah)") 
    plt.legend() 
    plt.grid(True) 
    plt.show() 
 
# Main execution 
print("\n--- Base Model Fitting ---") 
k1, k2 = fit_base_model(df) 
 
# Apply the base model to calculate 'base_capacity' and 'residual' 
print("\n--- Applying Base Model ---") 
df = apply_base_model(df, k1, k2) 
 
# Train the neural network using the residuals 
print("\n--- Training Neural Network for Residuals ---") 
variables = ['cycle', 'temperature', 'discharge_time']  # Variables for neural network 
nn_model, scaler_X, scaler_y = train_nn_for_residuals(df, variables) 
 
# Apply the hybrid model 
print("\n--- Applying Hybrid Model ---") 
df = apply_hybrid_model(df, k1, k2, nn_model, scaler_X, scaler_y, variables) 
 
# Plot the results 
print("\n--- Plotting Results ---") 
plot_comparison(df)


from sklearn.metrics import r2_score 
 
# Calculate metrics for each model 
base_mse = mean_squared_error(df['capacity'], df['base_capacity']) 
base_r2 = r2_score(df['capacity'], df['base_capacity']) 
hybrid_mse = mean_squared_error(df['capacity'], df['hybrid_capacity']) 
hybrid_r2 = r2_score(df['capacity'], df['hybrid_capacity']) 
 
print(f"Base Model - MSE: {base_mse:.4f}, R²: {base_r2:.4f}") 
print(f"Hybrid Model - MSE: {hybrid_mse:.4f}, R²: {hybrid_r2:.4f}")


plt.figure(figsize=(10, 6)) 
plt.scatter(df['cycle'], df['capacity'] - df['base_capacity'], label='Base Model Residuals', alpha=0.6, color='orange') 
plt.scatter(df['cycle'], df['capacity'] - df['hybrid_capacity'], label='Hybrid Model Residuals', alpha=0.6, color='green') 
plt.axhline(0, color='black', linestyle='--') 
plt.title("Residuals of Base Model and Hybrid Model") 
plt.xlabel("Cycle") 
plt.ylabel("Residual (Ah)") 
plt.legend() 
plt.grid(True) 
plt.show() 

plt.figure(figsize=(10, 6)) 
plt.hist(df['capacity'] - df['base_capacity'], bins=30, alpha=0.6, label='Base Model Residuals', color='orange') 
plt.hist(df['capacity'] - df['hybrid_capacity'], bins=30, alpha=0.6, label='Hybrid Model Residuals', color='green') 
plt.title("Histogram of Residuals") 
plt.xlabel("Residual (Ah)") 
plt.ylabel("Frequency") 
plt.legend() 
plt.grid(True) 
plt.show()


 
importances = np.abs(nn_model.coefs_[0]).sum(axis=1)  # Sum of weights in the input layer 
feature_importance = dict(zip(variables, importances)) 
 
print("Feature Importance:") 
for feature, importance in feature_importance.items(): 
    print(f"{feature}: {importance:.4f}") 
 
plt.figure(figsize=(10, 6)) 
plt.bar(feature_importance.keys(), feature_importance.values(), color='blue', alpha=0.7) 
plt.title("Feature Importance") 
plt.xlabel("Features") 
plt.ylabel("Importance (Sum of Weights)") 
plt.grid(True) 
plt.show()


sensitivity_data = df.copy() 
sensitivity_data['temperature'] = np.linspace(df['temperature'].min(), df['temperature'].max(), len(df)) 
X_scaled = scaler_X.transform(sensitivity_data[variables]) 
residual_pred_scaled = nn_model.predict(X_scaled) 
residual_pred = scaler_y.inverse_transform(residual_pred_scaled.reshape(-1, 1)) 
sensitivity_data['sensitivity_capacity'] = base_model(sensitivity_data['cycle'], k1, k2) + residual_pred[:, 0] 
 
plt.figure(figsize=(10, 6)) 
plt.plot(sensitivity_data['temperature'], sensitivity_data['sensitivity_capacity'], label='Predicted Capacity', color='blue') 
plt.title("Sensitivity Analysis: Effect of Temperature on Capacity") 
plt.xlabel("Temperature (°C)") 
plt.ylabel("Predicted Capacity (Ah)") 
 
plt.grid(True) 
plt.show()