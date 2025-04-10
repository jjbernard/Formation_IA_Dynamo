import cv2
import numpy as np

from settings import MAX_RETRIES
from tools import init_camera

if __name__ == '__main__':
    retry, capture, width, height = init_camera()

    # Chargement du modèle Yolo v4 Tiny. On utilise le nom "net" car il s'agit
    # d'un réseau (network) de neurones
    net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
    layer_names = net.getLayerNames()
    print(f"Les couches du réseau Yolov4 tiny: {layer_names}")
    # On recherche les "couches" qui sont en "sortie" du modèle
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    print(f"Les couches non connectées du réseau qui nous donnent les résultats: {output_layers}")

    # On charge les noms (classes) issus du dataset COCO (qui a servi à l'entrainement de Yolo)
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

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

            height, width, channels = frame.shape

            # On prépare l'image à analyser avant de la passer au modèle
            # 1/255 est un facteur pour exprimer les canaux de l'image
            # entre 0 et 1 (ceux-ci peuvent être entre 0 et 255).
            # Yolo est en RGB mais OpenCV fonctionne en BGR, donc swapRB est à True

            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
            # On donne l'image en entrée du modèle
            net.setInput(blob)

            # On passe l'image dans le modèle jusqu'à nos "couches" de sortie
            outs = net.forward(output_layers)

            # On initialise nos éventuelles trouvailles (ce qui a été détecté par le modèle)
            boxes = []
            confidences = []
            class_ids = []

            # Analyse des résultats
            for out in outs:
                for detection in out:
                    # On récupère les résultats sur les classes de sortie (80 classes)
                    scores = detection[5:]
                    # On identifie la classe aillant la probabilité la plus élevée
                    class_id = np.argmax(scores)
                    # Pour cette classe, on récupère la probabilité
                    confidence = scores[class_id]
                    if confidence > 0.5:  # Seuil de confiance pour la détection
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        # On ajoute ce que l'on a trouvé à nos listes de "trouvailles"
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
                        # On peut afficher dans notre console ce que l'on a vu
                        print(f"Vu: {classes[class_id]}")

            # on utilise cette dernière étape pour supprimer les boites
            # qui pourraient se chevaucher pour un même objet
            # NMS : Non Maximal Suppression
            # Le 0.5 que l'on utilise ici est le même que celui de la ligne if confidence > 0.5:
            # nms_threshold = 0.4 est par contre un paramètre spécifique de l'algorithme
            # IoU: Intersection over Union
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            # On dessine les boîtes sur l'écran
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Visualisation de la vidéo
            cv2.imshow('Détection temps réel', frame)

            # touch q pour quitter
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()
