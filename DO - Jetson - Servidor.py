import cv2
import socket
import struct
import numpy as np
import threading
import pyrealsense2 as rs
import RPi.GPIO as GPIO

# Definicion de pines de salida
output_pin_iz = 23
output_pin_de = 24
output_pin_fr = 10

contFreno = 0
contDerecha = 0
contIzquierda = 0

# Parametros de detección
anguloDetect = 10.0
contadorDetect = 2

# Definir la función px2grados aquí
def px2grados(px):
    px_cal = 640
    dx_cal = 1.015
    dz_cal = 0.958

    pz_cal = (px_cal/dx_cal)*dz_cal

    # Se genera un conjunto de variables para facilitar los calculos
    px_centro = px_cal/2
    dx_centro = dx_cal/2

    # Se realiza el calculo del angulo
    delta = px-px_centro
    rad = np.arctan(delta/pz_cal)
    grados = rad*180/np.pi

    return grados

def procesar_frame(frame_rgb, depth_frame):
    # Realizar la detección de objetos en el frame RGB
    classIds, confs, bbox = net.detect(frame_rgb, confThreshold=0.5)

    global contFreno, contDerecha, contIzquierda

    if len(classIds) == 0:
        contFreno = 0
        contDerecha = 0
        contIzquierda = 0

    for element, confidence, box in zip(classIds, confs, bbox):
        if element < 10:
	    # Obtener coordenadas del centro del objeto detectado
            x, y, w_, h_ = box
            cx = x + (w_ // 2)

            # Calcular el ángulo (simulado) utilizando la función px2grados
            alpha = px2grados(cx)  # Simulación del ángulo en grados

	        # Obtener la distancia al objeto desde el frame de profundidad
    	    
            depth = depth_frame.get_distance(cx, y + (h_ // 2))

            # Dibujar rectángulo y texto en el frame RGB
            cv2.rectangle(frame_rgb, box, color=(0, 255, 0), thickness=2)
            cv2.putText(frame_rgb, f'{classNames[int(element-1)]}', (box[0]+10, box[1]+30),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),2)
            cv2.putText(frame_rgb, f'{round(depth,2)} m', (box[0]+10, box[1]+50),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),2)
            cv2.putText(frame_rgb, f'{round(alpha,2)} g', (box[0]+10, box[1]+70),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),2)
           
            # Inicializar los pines en LOW
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(output_pin_iz, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(output_pin_de, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(output_pin_fr, GPIO.OUT, initial=GPIO.LOW)

            if alpha<-anguloDetect:
                contDerecha = contDerecha+1
                contIzquierda = 0
                contFreno = 0
                print("Contador derecha: ", contDerecha)
            elif alpha>anguloDetect:
                contIzquierda= contIzquierda+1
                contDerecha = 0
                contDerecha = 0
                print("Contador izquierda: ", contIzquierda)
            elif(alpha>-anguloDetect and alpha<anguloDetect):
                contFreno= contFreno+1
                contDerecha = 0
                contIzquierda = 0
                print("Contador freno: ", contFreno)
            else:
                contFreno = 0
                contDerecha = 0
                contIzquierda = 0
                print("Vacio")


            if alpha>anguloDetect and contIzquierda>=contadorDetect:
                #print('DERECHA') # printing the value
                GPIO.output(output_pin_de, GPIO.HIGH)
                GPIO.output(output_pin_iz, GPIO.LOW)
                GPIO.output(output_pin_fr, GPIO.LOW)
                cv2.putText(frame_rgb, 'IZQUIERDA', (220,330),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
            elif alpha<-anguloDetect and contDerecha>=contadorDetect:
                #print('IZQUIERDA') # printing the value
                GPIO.output(output_pin_iz, GPIO.HIGH)
                GPIO.output(output_pin_de, GPIO.LOW)
                GPIO.output(output_pin_fr, GPIO.LOW)
                cv2.putText(frame_rgb, 'DERECHA', (220,330),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
            else:
                if(contFreno>=contadorDetect):
                    GPIO.output(output_pin_fr, GPIO.HIGH)
                    GPIO.output(output_pin_iz, GPIO.LOW)
                    GPIO.output(output_pin_de, GPIO.LOW)
                    cv2.putText(frame_rgb, 'FRENO', (220,330),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
                
    return frame_rgb



def enviar_frames_al_cliente(conn):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Habilitar el stream de profundidad

    pipeline.start(config)
    
    with conn:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()  # Obtener el frame de profundidad

            if not color_frame or not depth_frame:
                continue

            # Convertir los frames a matrices numpy
            color_data = np.asanyarray(color_frame.get_data())

            # Procesar el frame RGB y el frame de profundidad
            frame_procesado = procesar_frame(color_data, depth_frame)

            # Convertir el frame procesado a JPEG
            _, buffer = cv2.imencode('.jpg', frame_procesado)
            data = buffer.tobytes()

            # Enviar el tamaño del frame
            data_size = len(data)
            conn.sendall(struct.pack(">L", data_size))

            # Enviar los datos del frame
            conn.sendall(data)

    pipeline.stop()

def iniciar_servidor():
    HOST = '10.214.6.218'
    PORT = 44444

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print('Esperando conexiones...')
        
        while True:
            conn, addr = s.accept()
            print('Conectado por', addr)

            # Iniciar un subproceso para enviar frames al cliente
            t = threading.Thread(target=enviar_frames_al_cliente, args=(conn,))
            t.start()

# Ruta del modelo y archivos de configuración
wd = '/home/nano/Descargas/RealsensePi-20240214T190836Z-001/RealsensePi/240_Detector_Redes/240_Detector_Redes'

# Cargar las categorías de la red
classNames = []
classFile = f'{wd}/ssd_mobilenet_v3_large_coco_2020_01_14/categorias.txt'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Cargar y configurar la red
configPath = f'{wd}/ssd_mobilenet_v3_large_coco_2020_01_14/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = f'{wd}/ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.0)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# soporte cuda
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
     print("Usando CUDA...")
     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
else:
    print("CUDA no está disponible; usando CPU.")

# Iniciar el servidor
iniciar_servidor()
