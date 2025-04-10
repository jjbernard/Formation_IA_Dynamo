import cv2
import numpy as np
import torch

from settings import MAX_RETRIES
from tools import init_camera

if __name__ == '__main__':
    retry, capture, width, height = init_camera()
    LOG = False

    # Chargement de Yolov5 small
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Configuration du modèle
    model.conf = 0.5  # Seuil de confiance (équivalent à notre 0.5 précédent avec Yolov4 tiny)
    model.iou = 0.4  # Seuil IoU pour NMS (équivalent à notre 0.4 précédent avec Yolov4 tiny)

    while True:
        ret, frame = capture.read()

        if not ret:
            retry += 1
            if retry > MAX_RETRIES:
                print("Erreur : impossible de capturer la vidéo!")
            else:
                continue

        # On réinitialise retry...
        retry = 0

        # Utilisation du modèle pour la détection
        # Le modèle YOLOv5 s'occupe de tout le prétraitement (redimensionnement, normalisation)
        # contrairement à ce qui est fait avec Yolov4
        results = model(frame)

        # Si on veut logger dans la console ce que l'on voit...
        if LOG:
            detections = results.pandas().xyxy[0] # On transforme en dataframe pandas pour plus de facilité
            for _, detection in detections.iterrows(): # On parcourt les lignes du tableau (dataframe)
                print(f"Vu: {detection['name']}")

        # Dessiner les détections sur l'image avec la méthode intégrée de YOLOv5
        frame = results.render()[0]  # Rend l'image avec les boîtes

        # Visualisation de la vidéo
        cv2.imshow('Détection temps réel', frame)

        # touche q pour quitter
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
