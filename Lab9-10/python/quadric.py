from sklearn.model_selection import train_test_split 
 
# Split the data into training and testing sets (80% training, 20% testing) 

 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split 
from sklearn.neural_network import MLPRegressor 
from sklearn.preprocessing import MinMaxScaler 
from scipy.optimize import curve_fit 
 
# Load the dataset 
file_path = r'C:\Masters\Y1\Sem1\PGD\Lab9-10\data\li_ion_battery_last_data.csv'  # Update this path if necessary 
df = pd.read_csv(file_path)

X = df[['cycle', 'temperature', 'discharge_time']]  # Input features 
y = df['capacity']  # Target variable: battery capacity 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
 
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
    return df 
 
# Train Neural Network with feature scaling 
def train_neural_network(df, variables, target): 
    """ 
    Train a neural network model to predict capacity based on input variables. 
    """ 
    # Normalize features and target 
    scaler_X = MinMaxScaler() 
    scaler_y = MinMaxScaler() 
 
    X = df[variables] 
    y = df[target].values.reshape(-1, 1) 
 
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
    print(f"Neural Network Mean Squared Error (MSE): {mse:.4f}") 
 
    # Plot predictions vs actual 
    plt.figure(figsize=(10, 6)) 
    plt.scatter(range(len(y_test)), y_test, label='Actual Capacity', alpha=0.6, marker='o', color='green') 
    plt.scatter(range(len(y_pred)), y_pred, label='Predicted Capacity', alpha=0.6, marker='x', color='red') 
    plt.title("Neural Network: Actual vs Predicted Capacity") 
    plt.xlabel("Sample Index") 
    plt.ylabel("Capacity (Ah)") 
    plt.legend() 
    plt.grid(True) 
    plt.show() 
 
    return model 
 
# Compare models and experimental data 
def compare_models(df, k1, k2, variables): 
    """ 
    Compare the base model and neural network predictions against experimental data. 
    """ 
    # Apply the base model 
 
    df = apply_base_model(df, k1, k2) 
 
    # Neural network approach 
    print("\n--- Neural Network Results ---") 
    nn_model = train_neural_network(df, variables, 'capacity') 
 
    # Plotting comparison of models 
    plt.figure(figsize=(10, 6)) 
    plt.plot(df['cycle'], df['capacity'], label='Experimental Capacity', linestyle='--', color='blue') 
    plt.plot(df['cycle'], df['base_capacity'], label='Estimated Capacity (Base Model)', linestyle='-', color='orange') 
    plt.title("Comparison of Measured and Base Model Capacities") 
    plt.xlabel("Cycle") 
    plt.ylabel("Capacity (Ah)") 
    plt.legend() 
    plt.grid(True) 
    plt.show() 
 
    return nn_model 
 
# Fit the base model to get parameters 
print("\n--- Base Model Fitting ---") 
k1, k2 = fit_base_model(df) 
 
# Compare models 
variables = ['cycle', 'temperature', 'discharge_time']  # Variables for neural network 
compare_models(df, k1, k2, variables)