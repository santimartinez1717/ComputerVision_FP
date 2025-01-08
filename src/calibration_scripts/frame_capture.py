import cv2
from picamera2 import Picamera2
import os

def stream_video():
    # Crear instancia de la cámara
    picam = Picamera2()
    picam.preview_configuration.main.size = (320, 320)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    # Crear carpeta si no existe
    output_dir = "../../assets/calibration_imgs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_count = 0  # Contador para nombrar los archivos de imagen

    while True:
        # Capturar el frame actual
        frame = picam.capture_array()
        cv2.imshow("picam", frame)

        # Esperar por teclas
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):  # Salir si se presiona 'q'
            break
        elif key == ord('a'):  # Guardar frame si se presiona Espacio
            file_path = os.path.join(output_dir, f"calibration_frame_{frame_count}.png")
            cv2.imwrite(file_path, frame)  # Guardar la imagen como archivo PNG
            print(f"Frame guardado: {file_path}")
            frame_count += 1

    # Liberar recursos
    cv2.destroyAllWindows()

if __name__ == "__main__":
    stream_video()
