import cv2
import numpy as np
import math
from picamera2 import Picamera2


def segment_goal(frame):
    """
    Segmenta las regiones de la portería usando HSV con valores específicos.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 221])
    upper = np.array([179, 124, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Operaciones morfológicas para limpiar el ruido
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def detect_goal_lines(mask, frame):
    """
    Detecta todas las líneas verticales y las seis líneas más largas del resto usando la Transformada de Hough.
    """
    lines = cv2.HoughLinesP(
        mask, rho=1, theta=np.pi / 180, threshold=70, minLineLength=50, maxLineGap=10
    )

    vertical_lines = []
    other_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if 80 < abs(angle) < 100:  # Líneas verticales (ángulo cercano a 90°)
                vertical_lines.append((x1, y1, x2, y2, length))
            else:
                other_lines.append((x1, y1, x2, y2, length))

    # Dibujar todas las líneas verticales
    vertical_lines = sorted(vertical_lines, key=lambda line: line[4], reverse=True)[:20]
    for x1, y1, x2, y2, _ in vertical_lines:
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Verde: Verticales

    # Seleccionar las 6 líneas más largas del resto
    other_lines = sorted(other_lines, key=lambda line: line[4], reverse=True)[:10]

    for x1, y1, x2, y2, _ in other_lines:
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Azul: Las seis más largas

    return frame


def process_camera():
    """
    Captura video en tiempo real desde la cámara de Raspberry Pi y muestra el resultado final.
    """
    # Configurar la cámara de Raspberry Pi
    picam = Picamera2()
    picam.preview_configuration.main.size = (640, 480)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    while True:
        # Capturar frame de la cámara
        frame = picam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convertir a formato OpenCV

        # Fase 1: Segmentación de la portería
        goal_mask = segment_goal(frame)
        cv2.imshow("Goal Mask", goal_mask)  # Mostrar la máscara

        # Fase 2: Detectar líneas y determinar la portería
        frame_with_goal = detect_goal_lines(goal_mask, frame)

        # Mostrar el resultado final con la detección de la portería
        cv2.imshow("Goal Detection", frame_with_goal)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    picam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_camera()




