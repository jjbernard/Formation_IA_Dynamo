import cv2
import numpy as np

from settings import MAX_RETRIES
from tools import init_camera

if __name__ == '__main__':
    retry, capture, width, height = init_camera()

    # Capture des images issues de la caméra
    while True:
        ret, frame = capture.read()

        if not ret:
            retry += 1
            if retry > MAX_RETRIES:
                print("Erreur : Erreur : impossible de capturer la vidéo!")
                break
            else:
                continue

        # On réinitialise retry...
        retry = 0

        # On convertit en HSV (Hue, Saturation, Value)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # On recherche des éléments dans une certaine gamme de bleus
        lower_blue = np.array([100, 50, 0])
        upper_blue = np.array([140, 255, 255])

        # On définit un masque par rapport à ces éléments
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # On recherche les contours avec ce masque
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # On dessine les boites (bounding boxes) autour de ce que l'on a trouvé
        for contour in contours:
            # On récupère les coordonnées
            x, y, w, h = cv2.boundingRect(contour)

            # On dessine un rectangle sur l'image
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Objets détectés', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()



