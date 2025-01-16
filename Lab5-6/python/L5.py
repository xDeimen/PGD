import paho.mqtt.client as mqtt 
import json 

MQTT_BROKER = "your_mqtt_broker_address" 
MQTT_PORT = 1883 
MQTT_TOPIC = "abb_robot/data" 
def on_message(client, userdata, msg): 
    try: 
        data = json.loads(msg.payload.decode()) 
        print(f"Received data: {data}") 
     
    except json.JSONDecodeError: 
        print("Failed to decode JSON") 

client = mqtt.Client() 
client.on_message = on_message 
client.connect(MQTT_BROKER, MQTT_PORT, 60) 
client.subscribe(MQTT_TOPIC) 
client.loop_forever()