import cv2

def init_camera():
    retry = 0
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        print("Erreur : impossible d'utiliser la caméra")
        exit(1)

    # On définit la largeur et la hauteur
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return retry, capture, width, height