import cv2
import numpy as np
from picamera2 import Picamera2

def detect_ball_using_hough(frame):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Aplicar un desenfoque para reducir el ruido antes de la detección de círculos
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Detectar círculos usando la transformada de Hough
    circles = cv2.HoughCircles(blurred, 
                                cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                param1=50, param2=30, minRadius=10, maxRadius=50)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")  # Convertir a enteros
        
        # Dibujar los círculos detectados
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)  # Dibujar el círculo
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)  # Dibujar el centro

    return frame




    

def stream_video():


    picam = Picamera2()
    picam.preview_configuration.main.size = (1280, 720)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    while True:
        frame = picam.capture_array()

        # Llamar a la función para detectar la pelota usando Hough
        frame_with_ball = detect_ball_using_hough(frame)

        # Mostrar el video con la detección
        cv2.imshow("picam", frame_with_ball)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    stream_video()
