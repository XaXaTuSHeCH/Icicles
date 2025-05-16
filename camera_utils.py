import cv2

def get_available_cameras():
    """
    Возвращает список доступных индексов камер.
    """
    index = 0
    cameras = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        cameras.append(index)
        cap.release()
        index += 1
    return cameras
