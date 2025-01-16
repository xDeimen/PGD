import csv
import socket
import time

SERVER_IP = '127.0.0.1'
SERVER_PORT = 30002

CSV_FILE_PATH = r'C:\Masters\Y1\Sem1\PGD\Lab3-4\data\simulated_data_robotic_cell.csv'

# Function to send data to the server
def send_data_to_server(data):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((SERVER_IP, SERVER_PORT))
            s.sendall(data.encode('utf-8'))
            response = s.recv(1024)
            print(f"Server response: {response.decode('utf-8')}")
    except Exception as e:
        print(f"Error: {e}")

def read_and_send_csv():
    with open(CSV_FILE_PATH, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data = f"{row['Timestamp']},{row['Machine ID']},{row['Cycle Time (s)']},{row['Production Count']},{row['Status']},{row['Energy Consumption (kWh)']}"
            send_data_to_server(data)
            time.sleep(1)

if __name__ == '__main__':
    read_and_send_csv()