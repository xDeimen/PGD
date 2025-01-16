from paho.mqtt import client as mqtt_client
import time 
import random  # Simulating data; replace with actual sensor readings. 
 
# Set up the MQTT client 
client = mqtt_client.Client("Transistor_Sensor") 
client.connect("19990da2823846f68376aadb01a9948a.s1.eu.hivemq.cloud", 8883)  # Replace with the address of your MQTT broker 
 
# Simulate reading sensor data and sending it via MQTT 
while True: 
    V_GS = random.uniform(0, 5)  # Replace with actual sensor reading for V_GS 
    I_D = random.uniform(0, 3)   # Replace with actual sensor reading for I_D 
     
    client.publish("transistor/V_GS", V_GS) 
    client.publish("transistor/I_D", I_D) 
    time.sleep(1)  # Send data every second