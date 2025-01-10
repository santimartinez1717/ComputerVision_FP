import cv2
import numpy as np

# Cargar los parámetros de calibración previamente guardados
calibration_data = np.load("calibration_data_left_camera.npz")
intrinsics = calibration_data["intrinsics"]
dist_coeffs = calibration_data["dist_coeffs"]

# Abrir la cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

# Obtener las dimensiones de la cámara
ret, frame = cap.read()
if not ret:
    print("Error al capturar el primer frame")
    cap.release()
    exit()

h, w = frame.shape[:2]
# Obtener una nueva matriz de cámara sin distorsión
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(intrinsics, dist_coeffs, (w, h), 1, (w, h))

# Define los rangos de color para detectar una pelota (en HSV)
lower_color = np.array([0, 0, 150])  # Blanco (ajusta según el color de la pelota)
upper_color = np.array([180, 60, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el frame")
        break

    # Corregir la distorsión del frame
    undistorted_frame = cv2.undistort(frame, intrinsics, dist_coeffs, None, new_camera_matrix)

    # Convertir el frame corregido a espacio de color HSV
    hsv = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2HSV)

    # Crear una máscara para detectar colores dentro del rango
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Procesar la máscara para reducir ruido
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Encuentra los contornos de los objetos detectados
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Filtra objetos pequeños basándote en el área
        if cv2.contourArea(contour) > 500:
            # Obtiene un círculo que encierra el contorno
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            if radius > 10:  # Filtra pequeños radios
                # Dibuja el círculo en el frame original
                cv2.circle(undistorted_frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                cv2.putText(undistorted_frame, "Pelota detectada", (int(x) - 20, int(y) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Muestra el frame con la detección
    cv2.imshow('Detección de pelota (corregido)', undistorted_frame)

    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra las ventanas
cap.release()
cv2.destroyAllWindows()
