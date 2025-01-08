import cv2
import numpy as np
import imageio
import glob

import copy
from typing import List
from functions import *

left_camera = 'left'
right_camera = 'right'


#imgs_path = glob.glob(f'..\..\assets\calibration_imgs\*.png')



imgs_path = [ f'/home/pi/Documents/FinalProyect_CV_SB/ComputerVision_FP/assets/calibration_imgs/calibration_frame_{n}.png' for n in range(23)]
imgs = load_images(imgs_path)

# Find corners with cv2.findChessboardCorners()
chessboard_shape = (8, 6)  # Tamaño interno del tablero de ajedrez (esquinas internas)
corners = [cv2.findChessboardCorners(img, chessboard_shape, None) for img in imgs]


corners_copy = copy.deepcopy(corners)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)


imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]

# Refinar detección de esquinas
corners_refined = [cv2.cornerSubPix(gray, cor[1], (11, 11), (-1, -1), criteria) if cor[0] else [] for gray, cor in zip(imgs_gray, corners_copy)]

# Dibujar esquinas en las imágenes
imgs_w_corners = copy.deepcopy(imgs)
for i, cor in enumerate(corners_refined):
    if len(cor) > 0:
        cv2.drawChessboardCorners(imgs_w_corners[i], chessboard_shape, cor, True)

# Mostrar las imágenes con esquinas dibujadas
for i, img in enumerate(imgs_w_corners):
    #show_image(f"Image {i + 1}", img)
    write_image(f'/home/pi/Documents/FinalProyect_CV_SB/ComputerVision_FP/assets/calibration_imgs/chess_corners/chess_corners_{i + 1}.jpg', img)


# Puntos del tablero
chessboard_points = get_chessboard_points(chessboard_shape, 30, 30)


# Filtrar datos y obtener solo las detecciones adecuadas
valid_corners = [cor[1] for cor in corners if cor[0]]
valid_corners = np.asarray(valid_corners, dtype=np.float32)

# Preparar puntos de objeto y de imagen para la calibración
objpoints = [chessboard_points for _ in range(len(valid_corners))]
imgpoints = valid_corners




# Calibrar la cámara
rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgs_gray[0].shape[::-1], None, None)

# Obtener extrínsecos
#extrinsics = [np.hstack((cv2.Rodrigues(rvec)[0], tvec)) for rvec, tvec in zip(rvecs, tvecs)]

# Imprimir resultados
print("Intrinsics:\n", intrinsics)
print("Distortion coefficients:\n", dist_coeffs)
print("Root mean squared reprojection error:\n", rms)

# Guardar resultados de la calibración
#np.savez("calibration_data_left_camera.npz", intrinsics=intrinsics, dist_coeffs=dist_coeffs, rvecs=rvecs, tvecs=tvecs)

