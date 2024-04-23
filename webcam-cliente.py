import socket
import cv2
import numpy as np
import struct

# Configuración del cliente
HOST = '10.214.6.218'  # Cambia la IP según la configuración del servidor
PORT = 44444

def recibir_frames_del_servidor(conn):
    cv2.namedWindow('Frame recibido del servidor', cv2.WINDOW_NORMAL)

    with conn:
        while True:
            # Recibir el tamaño del frame
            data_size = conn.recv(4)
            if not data_size:
                break
            size = struct.unpack('>L', data_size)[0]

            # Recibir los datos del frame
            data = b''
            while len(data) < size:
                packet = conn.recv(size - len(data))
                if not packet:
                    break
                data += packet

            # Decodificar los datos del frame con OpenCV
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Mostrar el frame recibido
            cv2.imshow('Frame recibido del servidor', frame)
            if cv2.waitKey(1) == 27:
                break

    cv2.destroyAllWindows()

def iniciar_cliente():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as conn:
        conn.connect((HOST, PORT))
        print('Conectado al servidor.')

        # Llamar a la función para recibir y mostrar frames del servidor
        recibir_frames_del_servidor(conn)

# Llamar a la función para iniciar el cliente
iniciar_cliente()
