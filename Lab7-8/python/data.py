import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Input 
import matplotlib.pyplot as plt 
 
# Load the data 
data = pd.read_csv(r'C:\Masters\Y1\Sem1\PGD\Lab7-8\data\transistor_data_generated.csv') 
 
# Define features (input variables) and target (output variable) 
X = data[['V_GS', 'V_th']].values 
y = data['I_D'].values 
 
# Split the data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
 
# Build the neural network model with an Input layer 
model = Sequential([ 
    Input(shape=(2,)),              # Define the input shape with 2 features (V_GS, V_th) 
    Dense(64, activation='relu'),   # First hidden layer with 64 neurons 
    Dense(64, activation='relu'),   # Second hidden layer with 64 neurons 
    Dense(1, activation='linear')   # Output layer for regression 
]) 
 
# Compile the model 
model.compile(optimizer='adam', loss='mean_squared_error') 
 
# Train the model 
 
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2) 
 
# Evaluate the model on the test set 
loss = model.evaluate(X_test, y_test) 
print(f"Test Loss: {loss}") 

model.save('my_model.h5')
 
# Predict the drain current using the test set 
y_pred = model.predict(X_test) 
 
# Plot true vs. predicted values 
plt.figure(figsize=(8, 6)) 
plt.scatter(y_test, y_pred, alpha=0.6) 
plt.xlabel('True I_D (A)') 
plt.ylabel('Predicted I_D (A)') 
plt.title('True vs. Predicted Drain Current') 
plt.grid(True) 
plt.show() 
 
# Function to simulate the transistor behavior using the trained model 
def simulate_transistor_behavior(model, V_th=2.0): 
    # Generate a range of V_GS values from 0V to 5V 
    V_GS_values = np.linspace(0, 5, 100) 
    V_th_values = np.full_like(V_GS_values, V_th)  # Keep V_th constant 
     
    # Create the input data for the model (V_GS, V_th) 
    X_simulation = np.column_stack((V_GS_values, V_th_values)) 
     
    # Predict I_D using the trained model 
    I_D_pred = model.predict(X_simulation) 
     
    # Plot the predicted I_D against V_GS 
    plt.figure(figsize=(8, 6)) 
    plt.plot(V_GS_values, I_D_pred, label=f'Predicted I_D (V_th={V_th}V)', color='blue') 
    plt.axvline(x=V_th, color='red', linestyle='--', label=f'Threshold Voltage (V_th={V_th}V)') 
    plt.xlabel('V_GS (V)') 
    plt.ylabel('Predicted I_D (A)') 
    plt.title('Simulated Transistor Behavior') 
    plt.grid(True) 
    plt.legend() 
    plt.show() 
 
# Simulate the transistor behavior using the trained model 
simulate_transistor_behavior(model, V_th=2.0) 
# Plot the training loss and validation loss 
plt.figure(figsize=(8, 6)) 
plt.plot(history.history['loss'], label='Training Loss', color='blue') 
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange') 
plt.xlabel('Epochs') 
plt.ylabel('Mean Squared Error Loss') 
plt.title('Training and Validation Loss Over Epochs') 
plt.grid(True) 
plt.legend() 
plt.show() 
 
# Calculate residuals 
residuals = y_test - y_pred.flatten() 
# Plot the histogram of residuals 
plt.figure(figsize=(8, 6)) 
plt.hist(residuals, bins=30, color='purple', edgecolor='black', alpha=0.7) 
plt.xlabel('Residuals') 
plt.ylabel('Frequency') 
plt.title('Distribution of Residuals') 
plt.grid(True) 
plt.show() 
 
 
# Plot residuals vs. predicted values 
plt.figure(figsize=(8, 6)) 
plt.scatter(y_pred, residuals, alpha=0.6, color='blue') 
plt.axhline(y=0, color='red', linestyle='--', linewidth=1) 
plt.xlabel('Predicted I_D (A)') 
plt.ylabel('Residuals') 
plt.title('Residuals vs. Predicted Values') 
plt.grid(True) 
plt.show() 
# Assuming y_test and y_pred are already defined 
# Calculate residuals 
residuals = y_test - y_pred.flatten()