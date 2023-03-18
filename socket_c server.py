import socket
import threading
import MotorAndHandle as mh
import servo_all

HEADER = 64
PORT = 5560
SERVER = '0.0.0.0'#socket.gethostbyname(socket.gethostname())
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
#print(SERVER)
ADDR = (SERVER, PORT)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)


def handle_client(conn, addr):
    print(f'new connection: {addr} connected.')

    connected = True
    while connected:
        msg_length = conn.recv(HEADER).decode(FORMAT)
        msg_length = int(msg_length)
        msg = conn.recv(msg_length).decode(FORMAT)
        #print(f"{addr}' {msg}")
        x = msg.split(', ')
        thumb = float(x[0])
        index = float(x[1])
        middle = float(x[2])
        ring = float(x[3])
        pinky = float(x[4])
        print(thumb, index, middle, ring, pinky)
        servo_all.hand(thumb, index, middle, ring, pinky)

        if msg == DISCONNECT_MESSAGE:
            conncted = False


def start():
    server.listen()
    print(f"LISTNING on {SERVER}")
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        print("ACTIVE CONNECTION {threading.activeCount() - 1}")


print("server is starting ........")
start()


