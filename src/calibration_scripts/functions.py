import cv2
import numpy as np
import imageio
import glob


import copy
from typing import List, Tuple
from functions import *




def load_images(filenames: List) -> List:
    """Carga las imágenes desde las rutas proporcionadas."""
    return [imageio.imread(filename) for filename in filenames]



# Mostrar imágenes y guardar si es necesario
def show_image(title, img):
    """Muestra una imagen en una ventana."""
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def write_image(path, img):
    """Guarda una imagen en el disco."""
    cv2.imwrite(path, img)



# Obtener los puntos del tablero de ajedrez
def get_chessboard_points(chessboard_shape: Tuple[int, int], dx: float, dy: float) -> np.ndarray:
    """
    Genera un conjunto de puntos 3D que representan las esquinas de un patrón de ajedrez en el espacio 3D.

    Parámetros:
    - chessboard_shape (Tuple[int, int]): La cantidad de esquinas interiores del patrón de ajedrez en filas y columnas (ej. (8, 6)).
    - dx (float): La distancia entre las esquinas en la dirección x.
    - dy (float): La distancia entre las esquinas en la dirección y.

    Retorna:
    - np.ndarray: Un array de NumPy con las coordenadas 3D de los puntos del patrón de ajedrez, con elementos de tipo np.float32.
    """

    # Crear una matriz de ceros para almacenar los puntos 3D del tablero de ajedrez
    objp = np.zeros((chessboard_shape[0] * chessboard_shape[1], 3), np.float32)
    
    # Llenar las dos primeras columnas con las coordenadas x, y generadas con mgrid
    # np.mgrid crea una cuadrícula de puntos, que luego se reorganiza y asigna a las columnas de x e y
    objp[:, :2] = np.mgrid[0:chessboard_shape[0], 0:chessboard_shape[1]].T.reshape(-1, 2)
    
    # Escalar las coordenadas x e y de acuerdo a las distancias dadas (dx y dy)
    objp[:, 0] *= dx
    objp[:, 1] *= dy
    
    # Retornar el array de puntos 3D del tablero de ajedrez
    return objp