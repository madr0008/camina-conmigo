# Este codigo es una adaptacion del siguiente repositorio: https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/tree/master

import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import pyttsx3


# Fuente - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Inicializar camara
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Leer primer frame
        (self.grabbed, self.frame) = self.stream.read()

	# Variable de control por si la camara se para
        self.stopped = False

    def start(self):
	# Comenzar el hilo que lee los frames de la camara
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Leer frames en un bucle infinito
        while True:
            # Si la camara se para, se para el hilo
            if self.stopped:
                # Liberar los recursos de la camara
                self.stream.release()
                return

            # Si la camara continua, seguir leyendo frames
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Devolver el frame mas reciente
        return self.frame

    def stop(self):
	# Indicar que se debe parar la grabacion
        self.stopped = True


MODEL_NAME = 'coco'
MODEL_NAME_2 = 'semaforos'
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
min_conf_threshold = 0.5
imW, imH = 1280, 720

# Importar bibliotecas de TensorFlow
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter
  

# Obtener ruta del directorio actual
CWD_PATH = os.getcwd()

# Obtener rutas a los modelos. el uno es el de deteccion, el dos el de clasificacion
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)
PATH_TO_CKPT_2 = os.path.join(CWD_PATH,MODEL_NAME_2,GRAPH_NAME)
PATH_TO_LABELS_2 = os.path.join(CWD_PATH,MODEL_NAME_2,LABELMAP_NAME)

# Cargar el labelmap
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
    
with open(PATH_TO_LABELS_2, 'r') as f:
    labels2 = [line.strip() for line in f.readlines()]

# Borramos la etiqueta ??? del modelo de deteccion
if labels[0] == '???':
    del(labels[0])

# Cargamos los modelos
interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter2 = Interpreter(model_path=PATH_TO_CKPT_2)

interpreter.allocate_tensors()
interpreter2.allocate_tensors()

# Obtenemos los detalles de los modelos
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

input_details2 = interpreter2.get_input_details()
output_details2 = interpreter2.get_output_details()
height2 = input_details2[0]['shape'][1]
width2 = input_details2[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)
floating_model2 = (input_details2[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Comprobacion de la version de TensorFlow del modelo. Segun esta, las etiquetas ocuparan una posicion un otra del vector de salida
outname = output_details[0]['name']
outname2 = output_details2[0]['name']

if ('StatefulPartitionedCall' in outname): # Si es TensorFlow 2
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # Si es TensorFlow 1
    boxes_idx, classes_idx, scores_idx = 0, 1, 2


# Inicializar calculo de los frames
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Inicializar video
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

#Bucle principal
while True:

    # Comenzar temporizador (para calcular la tasa de frames)
    t1 = cv2.getTickCount()

    # Obtener el ultimo frame disponible
    frame1 = videostream.read()

    # Redimensionar el frame
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalizar los valores de los pixeles
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Deteccion de objetos
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Obtener resultados
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Coordenadas de cajas
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Indices de las clases
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Valores de confianza
    
    maximo = ""
    valorMaximo = 0
    elegido = { "xmin": 0 , "ymin": 0, "xmax": 0 , "ymax": 0 , "score2": 0.0   }
    
    # Iterar sobre las detecciones buscando semaforos, si se supera el umbral de confianza
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            object_name = labels[int(classes[i])] # Obtener el nombre de la etiqueta
            
            #Solo nos fijamos en los semaforos
            if (object_name == 'traffic light'):

                # Obtener coordenadas de la caja
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))

                #Quedarnos con el recorte del semaforo
                img_traffic_light = frame_rgb[ymin:ymax, xmin:xmax]
                img_inception = cv2.resize(img_traffic_light, (width2, height2))
                input_data2 = np.expand_dims(img_inception, axis=0)
                                
                # Normalizar valores de los pixeles
                if floating_model2:
                    input_data2 = (np.float32(input_data2) - input_mean) / input_std

                # Clasificar el semaforo
                interpreter2.set_tensor(input_details2[0]['index'],input_data2)
                interpreter2.invoke()

                # Obtener resultados
                prediction2 = interpreter2.get_tensor(output_details2[0]['index'])[0] # Indice de la clase predicha
                
                classes2 = np.argmax(prediction2)
                score2 = np.max(prediction2)
                object_name2 = labels2[int(classes2)] # Nombre de la clase
                total = xmax - xmin + ymax - ymin
                
                #Actualizamos los valores de la variable elegido
                if(total > valorMaximo and object_name != "nada"):
                    valorMaximo = total
                    maximo = object_name2
                    elegido["ymin"] = ymin
                    elegido["xmin"] = xmin
                    elegido["xmax"] = xmax
                    elegido["ymax"] = ymax
                    elegido["score2"] = score2
                

    #Generamos el mensaje de alerta
    if (maximo != "nada" and maximo != ""):
        engine = pyttsx3.init()
        engine.setProperty('rate', 50)
        engine.setProperty('voice', 'spanish')
        if(maximo == "Rojo"):
            engine.say("Cuidado el semaforo esta en " + maximo + " no puedes pasar")
            color = (0,0,255)
        else:
            engine.say("El semaforo esta en " + maximo + " puedes pasar")
            color = (0,255,0)
        engine.runAndWait()
        
        #Dibujamos el rectangulo en cuestion
        cv2.rectangle(frame, (elegido["xmin"],elegido["ymin"]), (elegido["xmax"],elegido["ymax"]), color, 2)
        label = '%s %f' % (maximo, elegido["score2"])
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_ymin = max(elegido["ymin"], labelSize[1] + 10)
        cv2.rectangle(frame, (elegido["xmin"], label_ymin-labelSize[1]-10), (elegido["xmin"]+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
        cv2.putText(frame, label, (elegido["xmin"], label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        
                     
    # Dibujamos la tasa de frames abajo
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # Mostramos el frame por pantalla (para la prueba)
    cv2.imshow('Object detector', frame)

    # Calculamos la tasa de frames
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Si pulsamos q, salir del programa
    if cv2.waitKey(1) == ord('q'):
        break

# Liberar recursos
cv2.destroyAllWindows()
videostream.stop()
