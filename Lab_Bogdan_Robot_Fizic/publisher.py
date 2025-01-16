import tkinter as tk
from PIL import Image, ImageTk
import paho.mqtt.client as mqtt
import random
# MQTT broker settings
broker = 'broker.emqx.io'
port = 1883
topic = "button_topic/mqtt"
# generate client ID with pub prefix randomly
client_id = f'python-mqtt-{random.randint(0, 1000)}'
# username = 'emqx'
# password = 'public'

def button_pressed(number):
    mqtt_client.publish(topic, number)
    print(f"Button {number} pressed and sent to MQTT broker")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        connection_label.config(text="Connected", fg="green")
    else:
        connection_label.config(text="Not Connected", fg="red")

# Create the main window
window = tk.Tk()
window.title("Button Interface")

# Create a title label
title_label = tk.Label(window, text="Button Interface", font=("Arial", 16, "bold"))
title_label.pack(pady=20)


# List of button labels and image file paths
button_data = [
{"label": "Button 1", "image_path":
        r"Figure1.jpg"},
{"label": "Button 2", "image_path":
r"Figure2.jpg"},
{"label": "Button 3", "image_path":
r"Figure3.jpg"}
]
# Create a frame to hold the buttons
button_frame = tk.Frame(window)
button_frame.pack()
# Configure grid weight for resizing
button_frame.grid_columnconfigure(0, weight=1)
button_frame.grid_columnconfigure(1, weight=1)
button_frame.grid_columnconfigure(2, weight=1)
# Create the MQTT client
mqtt_client = mqtt.Client()
# Set username and password for MQTT broker
#mqtt_client.username_pw_set(username, password=password)
# Assign the on_connect callback function
mqtt_client.on_connect = on_connect
# Connect to the MQTT broker
#mqtt_client.connect(broker, port=port)
mqtt_client.connect(broker, port)



# Create the buttons dynamically
for index, data in enumerate(button_data):
    label = data["label"]
    image_path = data["image_path"]
    # Load the image and resize it
    image = Image.open(image_path)
    image = image.resize((350, 350)) # Adjust the dimensions as needed
    # Create the photo image object
    photo = ImageTk.PhotoImage(image)
    # Create the button with image and label
    button = tk.Button(button_frame, text=label, image=photo, compound=tk.TOP,
    command=lambda num=index+1: button_pressed(num))
    button.image = photo # Store a reference to the image to prevent garbage collection
    # Place the button in the frame
    button.grid(row=0, column=index, padx=10, pady=10, sticky="nsew")
    
# Create a label to indicate the connection status
connection_label = tk.Label(window, text="Not Connected", fg="red")
connection_label.pack()
# Start the MQTT loop
mqtt_client.loop_start()
# Start the GUI event loop
window.mainloop()
# Disconnect from the MQTT broker
mqtt_client.disconnect()