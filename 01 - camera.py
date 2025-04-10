import cv2
from settings import MAX_RETRIES, NB, OTHER_COLOR_SPACE

if __name__ == '__main__':
    retry = 0
    capture = cv2.VideoCapture(0)

    # On définit la largeur et la hauteur
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Capture des images issues de la caméra
    while True:
        ret, frame = capture.read()
        # ret contient un booléen comme quoi l'image est bien capturée
        # frame contient cette image.

        if not ret:
            retry += 1
            if retry > MAX_RETRIES:
                print("Erreur : impossible de capturer la vidéo!")
            else:
                continue

        # On réinitialise retry...
        retry = 0

        # Supposons que l'on veuille transformer le flux video en N&B
        if NB:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            image = frame

        # On peut aussi transformer le flux dans un autre espace colorimétrique
        # Peut servir en fonction des algorithmes que l'on utilise
        if OTHER_COLOR_SPACE:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            image = frame

        # Visualisation de la vidéo
        cv2.imshow('frame', image)

        # touch q pour quitter
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()