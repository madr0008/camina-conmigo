# Camina conmigo

Repositorio del proyecto final de prácticas de la asignatura Inteligencia Ambiental "Camina Conmigo": un sistema de detección de semáforos para personas invidentes, que les permite cruzar la calle de manera segura, notificando el estado de los semáforos de forma auditiva.

## Estructura de ficheros

- `coco/`: Ficheros del modelo de detección de objetos
- `semaforos/`: Ficheros del modelo de clasificación de semáforos
- `tflite1-env/`: Entorno virtual
- `get_requirements.sh`: Ejecutable para instalar dependencias
- `main.py`: Fichero a ejecutar

## Hardware

Este proyecto está diseñado para funcionar en una Raspberry Pi.

Las imágenes son suministradas por una Raspberry Pi Camera V2, y el sonido se reproduce en un altavoz de la propia Raspberry.

No obstante, el código es flexible, por lo que puede modificarse para adaptarse a otros componentes.

## Requisitos

Python 3.5 o posterior con las bibliotecas necesarias instaladas. Para instalarlas, basta con ejecutar el siguiente comando:

```bash
bash get_requirements.sh
```

## Modelos

Se han usado dos modelos en este proyecto: **Coco** y **Semaforos**.

### Coco

Se trata del modelo [mobilenet](https://keras.io/api/applications/mobilenet/), entrenado con el conjunto de datos [COCO](https://cocodataset.org).

Este modelo tiene el propósito de detectar los objetos presentes en la imagen, haciendo posible filtrar para quedarnos únicamente con los semáforos que aparecen en la misma.

### Semaforos

Se trata de una red neuronal con arquitectura [InceptionV3](https://cloud.google.com/tpu/docs/inception-v3-advanced?hl=es-419), entrenada con el conjunto de datos [imagenet](https://www.image-net.org/), y al que se ha hecho fine tuning con un [dataset de semáforos](https://drive.google.com/drive/folders/1UFcr-b4Ci5BsA72TZWJ77n-J3aneli6l).

## Uso

Para usar el conjunto de datos, una vez instaladas las dependencias necesarias, tan solo tenemos que ejecutar el siguiente comando:
```bash
python3 main.py
```
A partir de ese momento, la Raspberry comenzará a grabar y detectar y clasificar semáforos. Se podrá detener la ejecución pulsando la tecla q.

## Referencias

Este proyecto ha sido desarrollado usando como base el [repositorio](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/tree/master), cortesía de [EdjeeElectronics](https://github.com/EdjeElectronics).

De igual manera, el dataset con el que se ha entrenado el modelo de clasificación de semáforos ha sido obtenido del siguiente trabajo:

Tan, H., Chen, C., Luo, X., Zhang, J., Seibold, C., Yang, K., & Stiefelhagen, R. (2021, December). Flying guide dog: Walkable path discovery for the visually impaired utilizing drones and transformer-based semantic segmentation. In 2021 IEEE International Conference on Robotics and Biomimetics (ROBIO) (pp. 1123-1128). IEEE.
