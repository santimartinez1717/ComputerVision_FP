import cv2
import numpy as np
from picamera2 import Picamera2


def nothing(x):
    pass


def get_hsv_color_ranges_realtime():
    """
    Ajuste interactivo de los valores HSV usando Picamera2 en Raspberry Pi.
    """
    # Inicializar la cámara
    picam = Picamera2()
    picam.preview_configuration.main.size = (320, 320)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    # Crear ventana para ajustar HSV
    cv2.namedWindow('HSV Adjustment')

    # Crear barras deslizantes para HSV
    cv2.createTrackbar('HMin', 'HSV Adjustment', 0, 179, nothing)
    cv2.createTrackbar('SMin', 'HSV Adjustment', 0, 255, nothing)
    cv2.createTrackbar('VMin', 'HSV Adjustment', 0, 255, nothing)
    cv2.createTrackbar('HMax', 'HSV Adjustment', 0, 179, nothing)
    cv2.createTrackbar('SMax', 'HSV Adjustment', 0, 255, nothing)
    cv2.createTrackbar('VMax', 'HSV Adjustment', 0, 255, nothing)

    # Establecer valores predeterminados
    cv2.setTrackbarPos('HMax', 'HSV Adjustment', 179)
    cv2.setTrackbarPos('SMax', 'HSV Adjustment', 255)
    cv2.setTrackbarPos('VMax', 'HSV Adjustment', 255)

    # Variables para cambios de HSV
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    while True:
        # Capturar un fotograma de la cámara
        frame = picam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Obtener valores de las barras deslizantes
        hMin = cv2.getTrackbarPos('HMin', 'HSV Adjustment')
        sMin = cv2.getTrackbarPos('SMin', 'HSV Adjustment')
        vMin = cv2.getTrackbarPos('VMin', 'HSV Adjustment')

        hMax = cv2.getTrackbarPos('HMax', 'HSV Adjustment')
        sMax = cv2.getTrackbarPos('SMax', 'HSV Adjustment')
        vMax = cv2.getTrackbarPos('VMax', 'HSV Adjustment')

        # Crear los rangos HSV
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Convertir a espacio HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Aplicar la máscara HSV
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(frame, frame, mask=mask)

        # Mostrar valores HSV si cambian
        if (phMin != hMin) or (psMin != sMin) or (pvMin != vMin) or (phMax != hMax) or (psMax != sMax) or (pvMax != vMax):
            print(f"(hMin = {hMin}, sMin = {sMin}, vMin = {vMin}), (hMax = {hMax}, sMax = {sMax}, vMax = {vMax})")
            phMin, psMin, pvMin, phMax, psMax, pvMax = hMin, sMin, vMin, hMax, sMax, vMax

        # Mostrar la máscara y la imagen segmentada
        cv2.imshow('Original', frame)
        cv2.imshow('Masked', output)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Detener la cámara y cerrar ventanas
    picam.stop()
    cv2.destroyAllWindows()


# Llamar a la función principal
if __name__ == "__main__":
    get_hsv_color_ranges_realtime()




