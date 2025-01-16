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


# Initial capacity (C_0) and degradation constant (k) 
C_0 = df['capacity'].iloc[0]  # Assume initial capacity is the first recorded capacity 
k = 0.13  # Empirical constant 
 
# Calculate the degradation rate for each cycle 
df['estimated_capacity'] = C_0 - k * df['temperature'] * df['discharge_time'] 
 
# Plot the estimated capacity alongside the measured capacity 
plt.figure(figsize=(10, 6)) 
plt.plot(df['cycle'], df['capacity'], label='Measured Capacity', linestyle='--') 
plt.plot(df['cycle'], df['estimated_capacity'], label='Estimated Capacity', linestyle=':') 
plt.xlabel('Cycle Number') 
plt.ylabel('Capacity (Ah)') 
plt.title('Comparison of Measured and Estimated Capacity') 
plt.legend() 
plt.show()

from sklearn.metrics import mean_squared_error 
 
mse = mean_squared_error(df['capacity'], df['estimated_capacity']) 
print(f"Mean Squared Error: {mse}")