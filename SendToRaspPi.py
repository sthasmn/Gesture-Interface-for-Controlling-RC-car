import socket
from time import sleep

HEADER = 64
PORT = 5560 #ROBOT HAND
#PORT = 5055 #CAR
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
SERVER = 'raspberrypi.local'
ADDR = (SERVER, PORT)
print(socket.gethostbyname(socket.gethostname()))
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)
print("Connecting to raspberry pi")
def send(msg):
    message = msg.encode(FORMAT)
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' ' * (HEADER - len(send_length))
    client.send(send_length)
    client.send(message)
    sleep(0.05)
    #print(client.recv(2048).decode(FORMAT))

