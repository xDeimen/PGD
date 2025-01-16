import pandas as pd
import time
import win32com.client
import paho.mqtt.client as mqtt

# Configuration
CSV_FILE = r'C:\Masters\Y1\Sem1\PGD\Lab1-2\data\data.csv'
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_PUBLISH_TOPIC = "factory/sensor1"
MQTT_SUBSCRIBE_TOPIC = "factory/control"

class RobotController:
    def __init__(self):
        try:
            self.robot = win32com.client.Dispatch("RobotStudio.Robot")
            print("Connected to RobotStudio successfully.")
        except Exception as e:
            print(f"Failed to connect to RobotStudio: {e}")
            exit()

    def set_io(self, signal_name, value):
        try:
            self.robot.SetIO(signal_name, value)
            print(f"Set IO: {signal_name} to {value}")
        except Exception as e:
            print(f"Failed to set IO {signal_name}: {e}")

class MqttController:
    def __init__(self, broker, port, subscribe_topic, robot_controller):
        self.client = mqtt.Client()
        self.robot_controller = robot_controller
        self.subscribe_topic = subscribe_topic

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        try:
            self.client.connect(broker, port, 60)
            print(f"Connected to MQTT Broker at {broker}:{port}")
        except Exception as e:
            print(f"Failed to connect to MQTT Broker: {e}")
            exit()

    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected to MQTT Broker with result code {rc}")
        client.subscribe(self.subscribe_topic)

    def on_message(self, client, userdata, msg):
        print(f"Received message from {msg.topic}: {msg.payload.decode()}")
        if msg.topic == self.subscribe_topic:
            control_signal = msg.payload.decode()
            self.handle_control_signal(control_signal)

    def handle_control_signal(self, control_signal):
        if control_signal == "stop":
            self.robot_controller.set_io("MachineStatusSignal", 0)
            print("Robot stopped by MQTT control signal.")
        elif control_signal == "start":
            self.robot_controller.set_io("MachineStatusSignal", 1)
            print("Robot started by MQTT control signal.")

    def start(self):
        self.client.loop_start()

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()
        print("Disconnected from MQTT Broker.")

class CsvProcessor:
    def __init__(self, csv_file, robot_controller, mqtt_client, publish_topic):
        self.csv_file = csv_file
        self.robot_controller = robot_controller
        self.mqtt_client = mqtt_client
        self.publish_topic = publish_topic

    def process_and_publish(self):
        try:
            data = pd.read_csv(self.csv_file)
        except Exception as e:
            print(f"Failed to read CSV file: {e}")
            return

        for _, row in data.iterrows():
            machine_status = 1 if row['MachineStatus'] == "On" else 0
            production_count = row['ProductionCount']

            self.robot_controller.set_io("MachineStatusSignal", machine_status)
            self.robot_controller.set_io("ProductionCountSignal", production_count)

            mqtt_message = f"{row['MachineStatus']},{production_count}"
            self.mqtt_client.publish(self.publish_topic, mqtt_message)
            print(f"Published to MQTT: {mqtt_message}")
            time.sleep(1)

if __name__ == "__main__":
    try:
        robot_controller = RobotController()
        mqtt_controller = MqttController(MQTT_BROKER, MQTT_PORT, MQTT_SUBSCRIBE_TOPIC, robot_controller)
        csv_processor = CsvProcessor(CSV_FILE, robot_controller, mqtt_controller.client, MQTT_PUBLISH_TOPIC)

        mqtt_controller.start()
        csv_processor.process_and_publish()

    except KeyboardInterrupt:
        print("Shutting down gracefully...")

    finally:
        mqtt_controller.stop()