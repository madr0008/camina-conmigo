import argparse

import cv2
import numpy as np
from PIL import Image

import torch

import pyttsx3

from models import segmentation, traffic_light_classification
from config import cfg, update_config
from utils import colorize
from utils.utils import (
    create_logger,
    extract_walkable_area,
    display,
    display_info,
    extract_traffic_light,
    crop_traffic_lights,
    sound_alarm,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Sistema para ayudar a personas invidentes")
    parser.add_argument("--cfg", help="Experiment config file", required=True, type=str)
    parser.add_argument("--ready", help="Ready for flight", action="store_true")
    parser.add_argument(
        "opts", help="Modify config options using command line", default=None, nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def main():

    # Update config
    args = parse_args()
    update_config(cfg, args)
    print(cfg)

    # global logger
    logger = create_logger(cfg)

    #Sintesis de voz
    voice_engine = pyttsx3.init()

    #Camara de la Raspberry
    video = cv2.VideoCapture(0)

    # Segmentation model
    if cfg.SEGMENTATION.EXECUTE:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seg_model = segmentation.build_seg_model(cfg.SEGMENTATION.MODEL)()
        logger.info(f"Seg model: {cfg.SEGMENTATION.MODEL}, Device: {device}")

        classification_model = traffic_light_classification.build_classification_model(
            cfg.TRAFFIC_LIGHT_CLASSIFICATION.MODEL
        )()

    if cfg.READY:
        logger.info("Sistema preparado")

        num_frame = 0
        traffic_light_info = ""
        traffic_light_preds = []

        while True:

            ret_val, _frame = video.read()
            _frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB)
            _frame = cv2.resize(_frame, tuple(cfg.IMAGE.SIZE))
            img_display = _frame

            if cfg.SEGMENTATION.EXECUTE:

                img = Image.fromarray(_frame)
                seg_prediction = segmentation.predict_one(seg_model, img, device, cfg)

                if not cfg.SEGMENTATION.RETURN_PROB:

                    # Get (colorized) label image prediction
                    model_name = cfg.SEGMENTATION.MODEL
                    convert = "segformer" not in model_name
                    colorized = colorize.colorize(
                        seg_prediction, palette=cfg.SEGMENTATION.PALETTE, convert=convert
                    )  # for display
                    colorized_seg_pred = np.asarray(
                        colorized.copy()
                    )  # for traffic light recognition

                    walkable_area_mask = extract_walkable_area(colorized_seg_pred, cfg)

                    # Handle traffic lights
                    traffic_light_mask = extract_traffic_light(colorized_seg_pred, cfg)
                    anno_frame, cropped_traffic_lights = crop_traffic_lights(
                        anno_frame, traffic_light_mask, cfg, crop=True
                    )
                    if cropped_traffic_lights:
                        predictions = traffic_light_classification.predict(
                            classification_model, cropped_traffic_lights, device, cfg
                        )
                        final_pred = traffic_light_classification.get_final_prediction(predictions)

                        traffic_light_preds.append(final_pred)

                        # Choose traffic light which occurs the most in {cfg.TRAFFIC_LIGHT_CLASSIFICATION.NUM_AVERAGE} as final overall prediction
                        if len(traffic_light_preds) >= cfg.TRAFFIC_LIGHT_CLASSIFICATION.NUM_AVERAGE:
                            final_pred_overall = traffic_light_classification.get_average_prediction(
                                traffic_light_preds
                            )
                            logger.info(f"Traffic light: {final_pred_overall}")

                            # Adjust forward_backward_velocity based on traffic light prediction
                            forward_backward_velocity = cfg.DRONE.SPEED
                            if final_pred_overall == "pedestrian-red":
                                traffic_light_info = "Red"
                                sound_alarm(voice_engine, "Semáforo en rojo. Para")
                            elif final_pred_overall == "pedestrian-green":
                                traffic_light_info = "Green"
                                sound_alarm(voice_engine, "Semáforo en verde")
                            else:
                                traffic_light_info = ""

                    # Display
                    if cfg.SEGMENTATION.BLEND:
                        blended = colorize.blend(img, colorized, cfg.SEGMENTATION.ALPHA)

                    img_display = display(_frame, colorized, blended, anno_frame, cfg.DRONE.DISPLAY_IMAGE)
                    img_display = np.asarray(img_display)

                    #Esto es un poco a modo de log
                    if cfg.DRONE.DISPLAY_INFO:
                        display_info(img_display, traffic_light_info, cfg)

                    img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)

            #Mostrar video procesado en una ventana
            cv2.imshow("Video", img_display)

            # Control con teclas. Puede ser util, por eso lo dejo
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): #Tecla q para salir
                break

            num_frame += 1

        #Paramos de grabar
        video.release()

        #Dejamos de mostrar la ventana
        cv2.destroyAllWindows()

        logger.info("=> End")


if __name__ == "__main__":
    main()
