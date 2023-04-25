# Camina conmigo

Repositorio del proyecto final de prácticas de la asignatura Inteligencia Ambiental "Camina Conmigo": un sistema de detección de semáforos para personas invidentes, que les permite cruzar la calle de manera segura, notificando el estado de los semáforos de forma auditiva.

## Estructura de ficheros

- `config/`: Config
  - `experiment/`: Config yaml files for different experiments
  - `default.py`: Default config
- `drone/`: Drone initialization and control
- `models/`: Deep Learning models
  - `segmentation/`: Segmentation models
  - `traffic_light_classification/`: Traffic light classification models
- `utils/`: Helper functions and scripts



## Hardware

Este proyecto está diseñado para funcionar en una Raspberry Pi.

Las imágenes son suministradas por una Raspberry Pi Camera V2, y el sonido se reproduce en un altavoz de la propia Raspberry.

No obstante, el código es flexible, por lo que puede modificarse para adaptarse a otros componentes.

## Requisitos

Python 3.7 o posterior con las bibliotecas recopiladas en [requirements.txt](./requirements.txt) instaladas, incluyendo `torch>=1.7`. 

Para instalar las dependencias, basta con ejecutar el siguiente comando:

```bash
pip install -r requirements.txt
```

### SegFormer

1. Install `mmcv-full`

   To use [SegFormer](https://github.com/NVlabs/SegFormer), you need to install `mmcv-full==1.2.7`. For example, to install `mmcv-full==1.2.7` with `CUDA 11` and `PyTorch 1.7.0`, use the following command:

   ```bash
   pip install mmcv-full==1.2.7 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
   ```

   To install `mmcv-full` with different version of PyTorch and CUDA, please see: [MMCV Installation](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).

2. Use submodule `SegFormer` 

   - Initialize the submodule(s):

     ```bash
     git submodule init
     ```

   - Run the update to pull down the files:

     ```bash
     git submodule update
     ```

3. Install the dependencies of `SegFormer`:

   ```bash
   pip install -e models/segmentation/SegFormer/ --user
   ```

4. Copy config file to `SegFormer/`

   ```bash
   cp models/segmentation/segformer.b0.768x768.mapillary.160k.py models/segmentation/SegFormer/local_configs/segformer/B0
   ```



## Modelos

Two types of models are used: **street view semantic segmentation** and **traffic lights classification**.

### Street view semantic segmentation

We adopt SegFormer-B0 (trained on [Mapillary Vistas](https://www.mapillary.com/dataset/vistas) for 160K iterations) for street-view semantic segmentation based on each frame captured by the drone.

### Traffic lights classification

We create a custom traffic lights dataset named **Pedestrian and Vehicle Traffic Lights (PVTL) Dataset** using traffic lights images cropped from  [Cityscapes](https://www.cityscapes-dataset.com/), [Mapillary Vistas](https://www.mapillary.com/dataset/vistas), and [PedestrianLights](https://www.uni-muenster.de/PRIA/en/forschung/index.shtml). The PVTL dataset can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1UFcr-b4Ci5BsA72TZWJ77n-J3aneli6l?usp=sharing).

It containes 5 classes: Others, Pedestrian-red, Pedestrian-green, Vehicle-red, and Vehicle-green. Each class contains about 300 images. Train-validation split is 3:1.

<img src="assets/traffic_light_eg.png" alt="Traffic light dataset examples" style="zoom: 67%;" />

We train 2 models on this dataset:

- **ResNet-18**: We fine-tune ResNet-18 from `torchvision.models`. After 25 epochs training, the accuracy achieves around 90%.
- **Simple CNN model**: We build our custom simple CNN model (5 CONV + 3 FC). After 25 epochs training, the accuracy achieves around 83%.

### Pesos entrenados

1. Create `weights` folder and its subfolder `segmentation` and `traffic_light_classification`

   ```bash
   mkdir -p weights/segmentation weights/traffic_light_classification
   ```

2. Download trained weights from [Google Drive](https://drive.google.com/drive/folders/1efvfGxh2f1nCppO9YaPn6SyUQjG--QkC?usp=sharing) and put them into corresponding folders

    

## Uso



## Referencias

Este proyecto ha sido desarrollado usando como base el repositorio [flying-guide-dog](https://github.com/EckoTan0804/flying-guide-dog), que se corresponde con el paper:

Tan, H., Chen, C., Luo, X., Zhang, J., Seibold, C., Yang, K., & Stiefelhagen, R. (2021, December). Flying guide dog: Walkable path discovery for the visually impaired utilizing drones and transformer-based semantic segmentation. In 2021 IEEE International Conference on Robotics and Biomimetics (ROBIO) (pp. 1123-1128). IEEE.