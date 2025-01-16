import socket
import sys
import random
import paho.mqtt.client as mqtt


broker = 'broker.emqx.io'
port = 1883
topic = "button_topic/mqtt"
client_id = f'python-mqtt-{random.randint(0, 100)}'


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print("Failed to connect, return code %d\n", rc)


def on_message(client, userdata, msg):
    print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
    trimitere_primire_mesaje(msg.payload.decode())


def connect_mqtt() -> mqtt.Client:
    client = mqtt.Client(client_id)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker, port)
    client.subscribe(topic)
    return client


def trimitere_primire_mesaje(topiccc):
    adresa_serverului = ("192.168.1.200", 1025)
    print(sys.stderr, "Conectare la adresa %s si portul %s" % adresa_serverului)
    priza_ethernet = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    priza_ethernet.connect(adresa_serverului)
    print("Ca sa actionezi robotul ai de ales intre doua aplicatii")
    print("Tasteaza aplicatia_1 sau aplicatia_2")
    print("Orice alt mesaj va duce robotul in pozitia de parcare")
    mesajul_introdus = topiccc.encode()
    if mesajul_introdus == b"1":
        print("Ai ales prima aplicatie1")
        a = b"aplicatia_1"
    elif mesajul_introdus == b"2":
        print("Ai ales a doua aplicatie2")
        a = b"aplicatia_2"
    else:
        print("Ai ales sa parchezi robotul")
        a = b"aplica"
    print(sys.stderr, "Se va trimite mesajul %s" % a)
    priza_ethernet.sendall(a)
    mesaj_primit = repr(priza_ethernet.recv(1025))
    print("Mesajul primit este: %s" % mesaj_primit)
    priza_ethernet.close()
if __name__ == "__main__":
    client = connect_mqtt()
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        client.disconnect()
        client.loop_stop()