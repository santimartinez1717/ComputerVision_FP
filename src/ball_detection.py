import cv2
import numpy as np
from picamera2 import Picamera2
import math

# Parámetros de la cámara
FPS = 30
real_ball_radius = 0.0427  # Radio real de la pelota de fútbol en metros

# Función para segmentar la pelota por color
def segment_ball_by_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_color = np.array([0, 0, 103])  # Ajustar según el color de la pelota
    upper_color = np.array([105, 80, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    segmented = cv2.bitwise_and(frame, frame, mask=mask)
    return segmented, mask
    

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

# Función para detectar la pelota usando la transformada de Hough
def detect_ball_using_hough(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=50, param2=60, minRadius=40, maxRadius=120)

    best_circle = None
    best_circularity = 0  # Iniciar la circularidad máxima
    
    if circles is not None:
        circles = np.round(circles[0]).astype("int")

        for circle in circles:
            x, y, r = circle
            # Calcular el área y el perímetro del círculo
            area = np.pi * r ** 2
            perimeter = 2 * np.pi * r
            
            # Evitar la división por cero si el perímetro es muy pequeño
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
            else:
                circularity = 0

            # Si encontramos un círculo con mejor circularidad, lo elegimos
            if circularity > best_circularity:
                best_circularity = circularity
                best_circle = circle
    
    return best_circle

# Función principal para detectar la pelota y guardar el video
def detect_ball_in_video():
    picam2 = Picamera2()  # Crear instancia de la cámara
    picam2.preview_configuration.main.size = (320, 320)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()


    # Inicialización de variables para el seguimiento de Lucas-Kanade
    old_gray = None
    old_points = None
    frame_count = 0
    ball_center = None
    ball_radius = None
    last_center = None  # Para calcular la distancia recorrida
    speed_real = 0  # Para la velocidad real
    
    # Definir el formato para guardar el video (puedes elegir el formato .mp4 o .avi)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # Usamos MP4 como ejemplo
    out = cv2.VideoWriter('detected_ball_video.mp4', fourcc, FPS, (320, 320))  # Archivo de salida
    
    cv2.namedWindow("Detección de la Pelota", cv2.WINDOW_NORMAL)

    while True:
        # Capturar un frame de la cámara
        frame = picam2.capture_array()  # Obtén el frame como un arreglo numpy

        frame_count += 1

        # Si es el primer fotograma o cada 100 fotogramas, recalcular el círculo usando Hough
        if frame_count % 20 == 0 or old_gray is None:
            # Segmentación de la pelota por color
            segmented, mask = segment_ball_by_color(frame)
            # Detección de la pelota utilizando HoughCircles
            best_circle = detect_ball_using_hough(segmented)

            if best_circle is not None:
                ball_center = (best_circle[0], best_circle[1])
                ball_radius = best_circle[2]

            # Convertir la imagen a escala de grises
            old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            old_points = np.array([[ball_center]], dtype=np.float32) if ball_center is not None else None
            last_center = ball_center  # Inicializamos la última posición

        # Si hay puntos de interés para seguir (tracking)
        if old_points is not None:
            # Calcular el flujo óptico de Lucas-Kanade
            new_points, status, err = cv2.calcOpticalFlowPyrLK(old_gray, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), old_points, None)

            # Solo seguir puntos si el seguimiento fue exitoso
            if status[0][0] == 1:
                new_ball_center = tuple(new_points[0].ravel())
                # Dibujar el círculo de seguimiento
                cv2.circle(frame, (int(new_ball_center[0]), int(new_ball_center[1])), 10, (0, 255, 0), -1)

                # Calcular la distancia recorrida en píxeles
                if last_center is not None:
                    dist_px = np.linalg.norm(np.array(new_ball_center) - np.array(last_center))
                    # Escala de conversión: calculamos la relación entre el radio real y el radio en píxeles
                    if ball_radius is not None and ball_radius > 0:
                        scale = real_ball_radius / ball_radius  # Escala en metros por píxel
                        speed_real = (dist_px * scale) * FPS  # Velocidad real en metros por segundo
                    last_center = new_ball_center  # Actualizamos la última posición

            # Actualizar los puntos para el siguiente fotograma
            old_points = new_points
            old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Dibujar el marcador de la pelota
        if ball_center is not None:
            cv2.putText(frame, f"Ultima Posición detectada con Hough Lines: {ball_center}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Velocidad: {speed_real:.2f} m/s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        goal_mask = segment_goal(frame)
        frame_with_goal = detect_goal_lines(goal_mask, frame)

        # Mostrar el fotograma con la detección de la pelota
        cv2.imshow("Detección de la Pelota", frame)
        out.write(frame)


        # Esperar por la tecla 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    picam2.stop()  # Detener la cámara
    out.release()
    cv2.destroyAllWindows()

# Ejecutar la función para detectar la pelota y guardar el video
detect_ball_in_video()
